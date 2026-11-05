"""Explicit bench actuation without cameras; never imported by the offline CLI.

Uses the same ramp encoding, CommandLink and thermal accounting as recording.
See controller/control/theory.md: Frequency-only bench check.
"""
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import threading
import time

import shutil

from LLM.record.io import write_json
from LLM.record.ramps import load_ramp
from controller.control import ramp
from controller.control.hover_controller_runner import CommandLink


def drive_ramp(ramp_path, serial_port, *, output_root, confirm_start=False):
    """Reset flight, require READY, execute once, then stop. No camera/ROI needed.

    Starts real drive after READY, or after an explicit START input when confirm_start=True.
    This lets the operator keep coil power off during reset. Cooldown is operator-managed.
    Interrupt and errors attempt stop; USB loss/process death cannot guarantee it.
    LED still follows flight firmware START, but no optical timing is measured.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Run on main thread so notebook Interrupt can stop the test')
    if not serial_port:
        raise ValueError('Select the actual ESP serial port')
    segments = load_ramp(ramp_path)
    duration = ramp.duration_s(segments)
    # Operator explicitly removed modeled-temperature waiting for LLM on 2026-10-28.
    # Ramp validation, READY, timed stop and SerialComm drive accounting remain.
    folder = Path(output_root).resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ramp_path, folder/'ramp.json')
    summary = dict(mode='drive_only', video_recorded=False, optical_sync_validated=False,
                   serial_port=serial_port, duration_s=duration, status='opening',
                   start_acknowledged=False, stop_sent=False, stop_acknowledged=False)
    write_json(folder/'result.json', summary)
    print(f'Ramp {Path(ramp_path).stem}: {ramp.label(segments)}, {duration:g} s of drive', flush=True)
    print(f'Logs: {folder}\nNo cameras. '
          + ('Waits for typed START after READY. ' if confirm_start else 'START follows READY. ')
          + 'Interrupt requests STOP.', flush=True)
    owner = None
    started = False

    def receive(line, sent=False):
        if sent:
            return
        if line.startswith('SYNC_START '):
            summary['start_acknowledged'] = True
        if summary['stop_sent'] and line.startswith('state=4 '):
            summary['stop_acknowledged'] = True
        if line.startswith(('SYNC_', 'seq go:', 'state=', 't=', '[block]', '!')):
            print(line, flush=True)
        if line.startswith(('[block]', '!')):
            raise RuntimeError(f'Firmware stopped/rejected the test: {line}')
        if started and line == 'SYNC_READY v=1':
            raise RuntimeError('ESP rebooted during test')

    try:
        # READY is required independently of video pre-roll. Do not fabricate
        # capture ticks or weaken sync_start gates used by the recording panel.
        owner = CommandLink(serial_port, False, str(folder/'host_commands.log'),
                            takeoff_cmd=ramp.seq_lines(segments),
                            raw_log_path=folder/'telemetry.jsonl', require_ready=True)
        owner.on_line = receive
        if confirm_start:
            summary['status'] = 'waiting_for_start'
            write_json(folder/'result.json', summary)
            answer = input('READY. Coil power may now be enabled. Type START to drive once '
                           '(anything else cancels): ')
            if answer.strip() != 'START':
                summary['status'] = 'cancelled'
                return folder
            owner.drain()
        print('READY received. Sending the ramp now.', flush=True)
        started = True
        t0 = time.monotonic()
        owner.send_takeoff()
        summary['status'] = 'running'
        while time.monotonic() - t0 < duration:
            owner.drain()
            if not summary['start_acknowledged'] and time.monotonic() - t0 >= 2:
                raise RuntimeError('No SYNC_START within 2 s; stopping')
            time.sleep(.02)
        if not summary['start_acknowledged']:
            raise RuntimeError('Pattern ended without SYNC_START; stopping')
        summary['status'] = 'completed'
    except BaseException as exc:
        summary['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        summary['error'] = str(exc) or type(exc).__name__
        raise
    finally:
        try:
            if owner is not None:
                try:
                    summary['stop_sent'] = True
                    summary['stop_acknowledged'] = owner.stop() or summary['stop_acknowledged']
                except BaseException as exc:
                    summary['stop_sent'] = False
                    summary['stop_error'] = str(exc) or type(exc).__name__
                    print(f'STOP could not be confirmed: {summary["stop_error"]}', flush=True)
                finally:
                    owner.close()  # existing SerialComm thermal stamp
        finally:
            write_json(folder/'result.json', summary)
            print(f'Test {summary["status"]}; STOP sent={summary["stop_sent"]}, '
                  f'firmware OFF reply={summary["stop_acknowledged"]}. Logs: {folder}', flush=True)
    return folder
