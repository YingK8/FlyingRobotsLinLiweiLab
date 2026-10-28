"""Explicit bench actuation without cameras; never imported by the offline CLI.

Uses the same ramp encoding, CommandLink and thermal accounting as recording.
See controller/control/theory.md: Frequency-only bench check.
"""
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import threading
import time

from LLM.io import read_json, write_json
from LLM.pattern import validate_pattern
from controller.control import ramp
from controller.control.hover_controller_runner import CommandLink


def run_frequency_pattern(pattern_path, serial_port, *, output_root):
    """Reset flight, require READY, execute once, then stop. No camera/ROI needed.

    Calling this function starts real drive after READY; cooldown is operator-managed.
    Interrupt and errors attempt stop; USB loss/process death cannot guarantee it.
    LED still follows flight firmware START, but no optical timing is measured.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Run on main thread so notebook Interrupt can stop the test')
    if not serial_port:
        raise ValueError('Select the actual ESP serial port')
    planned = validate_pattern(read_json(pattern_path))
    if (planned.get('channel_settings_status') != 'firmware_defaults_unrecorded'
            or planned.get('channel_updates')):
        raise ValueError('Only frequency patterns using firmware channel defaults are supported')
    segments = tuple(map(tuple, planned['frequency_segments']))
    duration = ramp.duration_s(segments)
    print(f'Selected pattern: {planned.get("name", Path(pattern_path).stem)}; '
          f'actual drive duration: {duration:g} s', flush=True)
    # Operator explicitly removed modeled-temperature waiting for LLM on 2026-10-28.
    # Pattern validation, READY, timed stop and SerialComm drive accounting remain.
    folder = Path(output_root).resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'pattern.json', planned)
    summary = dict(mode='drive_only', video_recorded=False, optical_sync_validated=False,
                   serial_port=serial_port, duration_s=duration, status='opening',
                   start_acknowledged=False, stop_sent=False, stop_acknowledged=False)
    write_json(folder/'result.json', summary)
    print(f'Frequency-only test: {planned.get("name", Path(pattern_path).stem)}, {duration:g} s', flush=True)
    print(f'Logs: {folder}\nNo cameras. START follows READY; Interrupt requests STOP.', flush=True)
    owner = None
    started = False

    def receive(line, sent=False):
        if sent:
            return
        if line.startswith('SYNC_START v=1 '):
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
        print('READY received. Sending selected pattern now.', flush=True)
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
                    try:
                        owner.send('stop')
                    except BaseException:
                        summary['stop_sent'] = False
                        raise
                    deadline = time.monotonic() + 1
                    while time.monotonic() < deadline and not summary['stop_acknowledged']:
                        owner.drain()
                        time.sleep(.02)
                except BaseException as exc:
                    summary['stop_error'] = str(exc) or type(exc).__name__
                    print(f'STOP could not be confirmed: {summary["stop_error"]}', flush=True)
                finally:
                    owner.close()  # existing SerialComm thermal stamp
        finally:
            write_json(folder/'result.json', summary)
            print(f'Test {summary["status"]}; STOP sent={summary["stop_sent"]}, '
                  f'firmware OFF reply={summary["stop_acknowledged"]}. Logs: {folder}', flush=True)
    return folder
