"""Local browser controls; all camera/serial operations run on the main thread.

UI callbacks only set events. This open-loop capture adapter reuses CommandLink,
ramp.seq_lines and SerialComm; cooldown is operator-managed. No custom seq encoding.
"""
from contextlib import ExitStack
from pathlib import Path
import copy
import threading
import time
import cv2
import numpy as np
from LLM.io import read_json, write_json, snapshot
from LLM.setup_geometry import camera_key
from controller.camera import sources
from controller.camera.record import FlightWriter


def _check_loopback():
    # Viser retries bind errors internally; a denied bind otherwise hangs startup.
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))


def run_panel(session, *, output_root, max_seconds=15, run=None, pattern_path=None,
              serial_port=None, roi_a=None, roi_b=None, port=8088):
    """Preview -> Record -> optional flight START -> Stop/finalize, one take per call.

    Without `run`: CAMERA ONLY, never imports a serial controller. With `run`:
    caller must have prepared a measured experiment or explicit sync check; opens/reset flight at launch,
    then waits for a separate START click after pre-roll and dark-LED validation.
    Recording Stop, duration end, Interrupt and errors also send firmware stop.
    This is not an autonomous hover controller and has no live pose feedback.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Run panel on main thread for reliable Interrupt/serial cleanup')
    if not np.isfinite(max_seconds) or max_seconds <= 0: raise ValueError('Finite recording duration required')
    session = copy.deepcopy(session); camera = session['camera']; driven = run is not None
    planned = None
    if driven:
        from LLM.pattern import validate_pattern
        from controller.control import ramp
        from controller.control.hover_controller_runner import CommandLink
        run = Path(run)
        manifest = read_json(run/'manifest.json')
        if manifest.get('camera_session') != session: raise ValueError('Prepared run/session mismatch')
        if not serial_port: raise ValueError('Choose the actual ESP serial port explicitly')
        planned = validate_pattern(read_json(pattern_path))
        if planned.get('channel_settings_status') != 'firmware_defaults_unrecorded' or planned.get('channel_updates'):
            raise ValueError('This runner executes frequency-only patterns; channel schedules are not implemented')
        if (run/'raw/telemetry.jsonl').exists() or any((run/'raw/captures').glob('*/meta.json')):
            raise ValueError('One START attempt per run; prepare a new run')
        duration = ramp.duration_s(planned['frequency_segments'])
        if max_seconds < duration + 3: raise ValueError('Recording cap must allow ramp plus pre/post roll (at least 3 s)')
        for roi in (roi_a, roi_b):
            if (roi is None or len(roi) != 4 or any(type(v) is not int for v in roi)
                    or min(roi[:2]) < 0 or min(roi[2:]) <= 0
                    or roi[0]+roi[2] > camera['width'] or roi[1]+roi[3] > camera['height']):
                raise ValueError('Specify in-bounds LED [x,y,width,height] for BOTH cameras')
        snapshot(pattern_path, run/'pattern.json')
        # No modeled-temperature wait, per operator request on 2026-10-28.
    _check_loopback()
    import viser
    events = {k: threading.Event() for k in ('record', 'start', 'stop', 'close', 'snapshot')}
    writer = link = None
    take = None; record_t = drive_t = stop_t = None; error = None; last_ui = 0
    result = {'take': None, 'reference_pair': None, 'drive_start_sent': False}
    with ExitStack() as cleanup:
        server = viser.ViserServer(host='127.0.0.1', port=port, label='LLM experiment')
        cleanup.callback(server.stop)
        print(f'Open http://127.0.0.1:{server.get_port()} — camera A is the first selected index.')
        status = server.gui.add_markdown('Opening cameras…')
        views = [server.gui.add_image(np.zeros((200, 320, 3), np.uint8), label=f'Camera {tag}') for tag in 'AB']
        confirmation = ('A/B labels correct; LED and static plumb reference visible; inputs-only sync check'
                        if driven and manifest.get('experiment_mode') == 'sync_check' else
                        'A/B labels correct; camera, fixture and focus unchanged since verification'
                        if driven else 'A/B labels correct; camera settings checked for this capture')
        physical = server.gui.add_checkbox(confirmation, False)
        buttons = {}
        for key, label in [('snapshot','Save static reference pair'), ('record','Record'),
                           ('start','START pattern + LED'), ('stop','Stop / finish'), ('close','Close preview')]:
            buttons[key] = server.gui.add_button(label, disabled=key=='start')
            buttons[key].on_click(lambda _, e=events[key]: e.set())
        options = {k: camera[k] for k in ('width','height','fps','rotate180','backend')}
        src = sources.open_stereo([f'camera:{i}' for i in camera['indices']], grayscale=True, max_skew_s=None, **options)
        cleanup.callback(src.close)
        def finalize():
            if writer is not None:
                if error: writer.meta['capture_error'] = error
                writer.meta['capture_counters'] = [{'grabbed': c.n_grabbed, 'overwritten_before_consumption': c.n_dropped}
                                                   for c in src.sources]
                result['take'] = str(writer.close())
        cleanup.callback(finalize)
        if driven:
            link = CommandLink(serial_port, False, str(run/'raw/host_commands.log'),
                               takeoff_cmd=ramp.seq_lines(planned['frequency_segments']),
                               raw_log_path=run/'raw/telemetry.jsonl', sync_start=True)
            cleanup.callback(link.close)
            # Registered last: stop drive BEFORE encoder finalisation/camera release.
            cleanup.callback(lambda: link.send('stop'))
        meta = {'session_id': session['session_id'], 'camera_config': camera_key(camera),
                'camera_indices': camera['indices'], 'rotate180': camera['rotate180'],
                'requested': {k: camera[k] for k in ('width','height','fps')},
                'cameras': [c.actual for c in src.sources], 'note': 'LLM browser panel',
                'mode_kind': 'flight_frequency_open_loop' if driven else 'camera_only'}
        try:
            while True:
                # Stop wins over START/record if both arrive before a loop iteration.
                if events['stop'].is_set() or events['close'].is_set(): break
                item = src.read()
                if item is None: raise OSError('Camera stream ended')
                if events['stop'].is_set() or events['close'].is_set(): break
                t, frames = item; frames = list(frames)
                if any(f.shape[:2] != (camera['height'],camera['width']) for f in frames):
                    raise ValueError('Camera delivered a different image mode')
                now = time.monotonic()
                if events['snapshot'].is_set():
                    events['snapshot'].clear()
                    if writer is None:
                        from uuid import uuid4
                        folder = Path(session['directory'])/('reference_'+uuid4().hex[:8]); folder.mkdir()
                        for tag, frame in zip('AB', frames):
                            if not cv2.imwrite(str(folder/f'{tag}.png'), frame): raise OSError('Reference image write failed')
                        write_json(folder/'meta.json', {'session_id': session['session_id'], 'camera': camera,
                                                       'times': list(src.last_stamps)})
                        result['reference_pair'] = str(folder)
                        print('Reference saved:', folder)
                if events['record'].is_set():
                    events['record'].clear()
                    if writer is None and physical.value:
                        meta['operator_scene_unchanged'] = True
                        writer = FlightWriter(output_root, fps=camera['fps'], meta=meta, codec=camera['codec'])
                        take = writer.dir; record_t = now
                if writer is not None:
                    writer.add(t, frames, src.last_stamps, src.last_skew)
                    if writer.errors or writer.dropped: raise OSError('Encoder errors/queue drops; stopping attempt')
                    if link is not None: link.note_capture(t)
                    if now-record_t >= max_seconds: break
                dark = False
                if link is not None:
                    link.drain()
                    dark = all(float(np.mean(f[y:y+h,x:x+w])) < 80 for f,(x,y,w,h) in zip(frames,(roi_a,roi_b)))
                    if events['start'].is_set():
                        events['start'].clear()
                        if (writer is not None and link.sync_can_start() and dark and physical.value
                                and max_seconds-(now-record_t) >= duration+2):
                            link.send_takeoff(); drive_t = time.monotonic(); result['drive_start_sent'] = True
                            writer.meta['host_start_send_complete_s'] = drive_t
                    if drive_t is not None and stop_t is None and now-drive_t >= duration:
                        link.send('stop'); stop_t = time.monotonic()
                        writer.meta['host_scheduled_stop_send_complete_s'] = stop_t
                    if stop_t is not None and now-stop_t >= 2: break
                if now-last_ui >= .2:
                    last_ui = now
                    for handle, frame in zip(views, frames):
                        thumb = cv2.resize(frame, (640, round(640*camera['height']/camera['width'])))
                        handle.image = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB) if thumb.ndim==2 else cv2.cvtColor(thumb,cv2.COLOR_BGR2RGB)
                    buttons['record'].disabled = writer is not None or not physical.value
                    buttons['start'].disabled = not (link is not None and writer is not None and link.sync_can_start() and dark and physical.value
                        and max_seconds-(now-record_t) >= duration+2)
                    buttons['snapshot'].disabled = writer is not None
                    state = 'PREVIEW' if writer is None else f'RECORDING {now-record_t:.1f}s — {writer.n} pairs'
                    status.content = f'**{state}**\n\n' + (f'LED dark in both ROIs: {dark}; START sent: {result["drive_start_sent"]}. Do not press ESP reset.' if driven else 'Camera only. No serial connection or drive.')
        except KeyboardInterrupt:
            print('Interrupted: stopping drive and finalising recording.')
        except BaseException as exc:
            error = str(exc)
            raise
    return result
