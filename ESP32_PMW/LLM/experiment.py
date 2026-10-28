"""Notebook workflow: immutable session inputs, current-take audit and postprocess.

Hardware is opened only by an explicit call to run_panel. No code runs on import.
"""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import copy
import numpy as np
from LLM.io import init_run, read_json, write_json, sha256, snapshot
from LLM.capture import audit_flight, import_flight
from LLM.setup_geometry import camera_key, require_verified


def new_session(root, camera):
    camera = camera_key(copy.deepcopy(camera))
    if len(camera['indices']) != 2 or len(set(camera['indices'])) != 2:
        raise ValueError('Choose distinct camera A and B indices')
    if min(camera['width'], camera['height'], camera['fps']) <= 0:
        raise ValueError('Positive camera size/FPS required')
    name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6]
    folder = Path(root).resolve() / name
    folder.mkdir(parents=True, exist_ok=False)
    session = {'session_id': name, 'directory': str(folder), 'camera': camera,
               'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(folder / 'session.json', session)
    return session


def audit_take(take, session):
    """Recompute on this take; never accept a notebook-global old REPORT."""
    take = Path(take)
    meta = read_json(take / 'meta.json')
    if (meta.get('session_id') != session['session_id']
            or meta.get('camera_config') != session['camera']):
        raise ValueError('Take belongs to a different session/configuration')
    c = session['camera']
    if meta.get('mode') != [c['width'], c['height']] or meta.get('rotate180') != c['rotate180']:
        raise ValueError('Recorded dimensions/rotation differ from session')
    report = audit_flight(take, target_fps=c['fps'])
    report['camera'] = copy.deepcopy(meta['camera_config'])
    report['session_id'] = meta['session_id']
    report['input_sha256'] = {str((take/p).resolve()): sha256(take/p)
                             for p in ['meta.json', 'frames.csv', 'A/A.mp4', 'B/B.mp4']}
    write_json(take/'capture_audit.json', report)
    return report


def check_trial(report, session, warning_review=''):
    if (report.get('integrity') != 'PASS' or report.get('session_id') != session['session_id']
            or report.get('camera') != session['camera']):
        raise ValueError('Complete a trial recording and audit for this camera session')
    for p, digest in report['input_sha256'].items():
        if sha256(p) != digest: raise ValueError('Trial files changed; repeat audit')
    if report['warnings'] and not warning_review.strip():
        raise ValueError('Review trial warnings and enter a concrete explanation first')


def prepare_sync_run(root, session, trial, reference_pair, *, note, warning_review):
    """Prepare an inputs-only bench capture; no geometry acceptance is implied."""
    check_trial(trial, session, warning_review)
    if not note.strip(): raise ValueError('Describe this synchronization check')
    if reference_pair is None: raise ValueError('Save a static reference pair in Block 4 first')
    reference = Path(reference_pair)
    meta = read_json(reference/'meta.json')
    if meta.get('session_id') != session['session_id'] or meta.get('camera') != session['camera']:
        raise ValueError('Reference pair belongs to a different session/configuration')
    import cv2
    for tag in 'AB':
        frame = cv2.imread(str(reference/f'{tag}.png'))
        if frame is None or frame.shape[:2] != (session['camera']['height'], session['camera']['width']):
            raise ValueError('Reference image missing or wrong dimensions')
    run = init_run(Path(root) / ('sync_check_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:6]))
    for name in ('A.png', 'B.png', 'meta.json'):
        snapshot(reference/name, run/'raw/static_reference'/name)
    write_json(run/'raw/trial_audit.json', trial)
    manifest = read_json(run/'manifest.json')
    manifest.update(camera_session=copy.deepcopy(session), operator_warning_review=warning_review,
                    experiment_mode='sync_check', geometry_status='not_assessed',
                    plumb_status='reference_images_only_not_reconstructed')
    manifest['initial_conditions']['notes'] = note
    write_json(run/'manifest.json', manifest)
    return run


def prepare_run(root, session, calibration, verification, world, trial, *, note, warning_review):
    check_trial(trial, session, warning_review)
    require_verified(verification, calibration, session['camera'], session['session_id'])
    evidence = world.get('evidence', {})
    if (evidence.get('session_id') != session['session_id'] or evidence.get('camera') != session['camera']
            or evidence.get('rig_sha256') != sha256(calibration)):
        raise ValueError('Localize fixture + gravity for this session before a measured experiment')
    for p, digest in evidence['input_sha256'].items():
        if sha256(p) != digest: raise ValueError('Fixture reference changed')
    if not note.strip(): raise ValueError('Describe robot, bearing, coil setup and experimental condition')
    run = init_run(Path(root) / (datetime.now().strftime('experiment_%Y%m%d_%H%M%S_') + uuid4().hex[:6]))
    snapshot(calibration, run/'calibration/stereo_rig.json')
    write_json(run/'calibration/world_frame.json', world)
    write_json(run/'calibration/verification.json', verification)
    write_json(run/'raw/trial_audit.json', trial)
    # Keep portable source evidence, not just paths into another machine's session.
    for category, sources in [('fixture', evidence['input_sha256']), ('verification', verification['input_sha256'])]:
        index = []
        for i, (source, digest) in enumerate(sources.items()):
            target = run/'calibration/evidence'/category/f'{i:03d}_{Path(source).name}'
            snapshot(source, target)
            index.append({'original': source, 'path': str(target.relative_to(run)), 'sha256': digest})
        write_json(run/f'calibration/{category}_evidence.json', index)
    manifest = read_json(run/'manifest.json')
    manifest.update(camera_session=copy.deepcopy(session), operator_warning_review=warning_review)
    manifest['initial_conditions']['notes'] = note
    write_json(run/'manifest.json', manifest)
    return run


def finish_take(take, session, run=None):
    report = audit_take(take, session)
    if run is not None:
        import_flight(take, run)
        write_json(Path(run)/'raw/capture_audit.json', report)
    return report


def postprocess(run, roi_a, roi_b, *, annotations=None, experimental_auto=False):
    """Explicit prerequisites, no synthetic stand-ins for missing measurements."""
    from LLM.start import synchronize_start
    from LLM.__main__ import _pose
    from LLM.pipeline import process_run
    run = Path(run)
    sync_only = read_json(run/'manifest.json').get('experiment_mode') == 'sync_check'
    if sync_only and (annotations is not None or experimental_auto):
        raise ValueError('sync_check has no accepted geometry; use the measured workflow for pose')
    if not (run/'raw/camera/meta.json').exists(): raise ValueError('Finish/import this take first')
    if not (run/'raw/telemetry.jsonl').exists(): raise ValueError('Missing flight START/current log')
    for name in ('overview.png', 'input_curves.png', 'input_curves.meta.json'):
        (run/'export'/name).unlink(missing_ok=True)
    sync = synchronize_start(run, roi_a, roi_b)
    input_curves(run, sync)
    if sync_only:
        return {'status': 'sync_check_inputs_only', 'sync': str(run/'processed/sync.json'),
                'input_curves': str(run/'export/input_curves.png'),
                'geometry_status': 'not_assessed', 'hardware_sync_validated': False,
                'drive_apply_interval_t_s': sync['device_start']['drive_apply_interval_t_s']}
    if annotations is None and not experimental_auto:
        return {'status': 'needs_pose_annotations', 'sync': str(run/'processed/sync.json'),
                'input_curves': str(run/'export/input_curves.png'),
                'next': 'Provide physical-point CSV, or explicitly select experimental ellipse-centre auto mode.'}
    _pose(run, annotations, run/'calibration/stereo_rig.json', automatic=experimental_auto)
    return process_run(run)


def input_curves(run, sync):
    """Frequency plan and observed low-rate current, even before pose annotations."""
    from LLM.telemetry import parse_log
    from LLM.pattern import sample_pattern
    from LLM.sync import map_times
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run)
    current = parse_log(run/'raw/telemetry.jsonl')
    current['t_s'] = map_times(current.t_device_s.to_numpy(float), sync['clocks']['esp']) if len(current) else []
    current.to_csv(run/'processed/current.csv', index=False)
    write_json(run/'processed/current.meta.json', current.attrs)
    plan = sample_pattern(read_json(run/'pattern.json'))
    plan.to_csv(run/'processed/command_planned.csv', index=False)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    axes[0].plot(plan.t_s, plan.frequency_hz, label='Planned frequency')
    if len(current):
        axes[0].plot(current.t_s, current.frequency_reported_hz, '.', label='Device-reported samples')
        for ch in 'ABCD': axes[1].plot(current.t_s,current[f'current_{ch}_A'],'.-',label=ch)
    else:
        axes[1].text(.03,.5,'No parseable CS samples',transform=axes[1].transAxes)
    axes[0].set_ylabel('Frequency (Hz)'); axes[1].set_ylabel('Filtered CS (A)')
    import pandas as pd
    for ch in 'AB':
        led = pd.read_csv(run/f'processed/led_{ch}.csv')
        axes[2].plot(map_times(led.source_time_s, sync['clocks'][f'camera_{ch}']),
                     led.brightness, label=f'Camera {ch} LED')
    axes[2].axhline(sync['led_config']['threshold'], color='gray', linestyle='--', label='Detection threshold')
    axes[2].set_ylabel('LED brightness (0–255)')
    axes[2].set_xlabel('LED-aligned time (s); alignment does not prove simultaneous current onset')
    for ax in axes:
        ax.grid(alpha=.2); ax.axvline(0,color='gray',linestyle=':')
        if ax.get_legend_handles_labels()[0]: ax.legend()
    fig.suptitle('Inputs only — pose/takeoff not assessed; current phase unavailable')
    fig.savefig(run/'export/input_curves.png',dpi=140);plt.close(fig)
    write_json(run/'export/input_curves.meta.json', {
        'status':'inputs_only_not_full_pipeline', 'current_samples':len(current),
        'hardware_sync_validated':False, 'input_sha256':{p:sha256(run/p) for p in
          ['pattern.json','processed/sync.json','raw/telemetry.jsonl']}})
