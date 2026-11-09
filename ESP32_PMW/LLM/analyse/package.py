"""One shareable folder per run: `export/package/`.

    setup.md / setup.json   camera intrinsics, A<->B extrinsics, the setup (world) frame and
                            each camera's pose in it, LED ROIs, clock sync -- with caveats
    A.mp4, B.mp4            the recorded videos, every captured frame (copies of raw/)
    frames.csv              per frame: host capture times and the same on the LED-synced
                            experiment clock (t = 0 at the START LED)
    run_parameters.json     written before START: randomised duty (seed, distribution,
                            draws), ramp, camera, calibration, time anchor, git, versions
    telemetry.jsonl, manifest.json, config.json, capture_meta.json   raw record copies
    ramp.json               the frequency ramp flown (the run's snapshot), plus the `seq=`
    host_commands.log         lines the host actually sent; command_planned.csv is the
                              planned f(t), current.csv the device-reported frequency
    *.csv                   the processed tables already in processed/

Every number is copied or derived from the run's own files, the rig and the world frame;
the source path and sha256 of each is in setup.json. Nothing is re-fitted here.

    .venv/bin/python -m LLM.analyse.package <run> <stereo_rig.json> [world_frame.json]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from LLM.record.io import read_json, sha256, write_json


def _f(x, n=3):
    return np.array2string(np.asarray(x, float), precision=n, suppress_small=True, separator=", ")


def describe(rig_path, world=None):
    """Intrinsics, extrinsics and (if fitted) the world frame, as plain JSON numbers."""
    from controller.calib.rig import StereoRig
    raw, rig = read_json(rig_path), StereoRig.load(rig_path)
    cams = {}
    for cam in rig.cameras:
        K = np.asarray(cam.K)
        cams[cam.name] = {"K": K.tolist(), "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                          "dist_k1k2p1p2k3": np.ravel(cam.dist).tolist(),
                          "rms_px": raw.get("rms_px_intrinsics", {}).get(cam.name),
                          "T_camA_from_cam": np.asarray(cam.T_world_cam).tolist()}
    T_ab = np.asarray(rig.b.T_world_cam)               # camera B -> camera A (rig frame = A)
    axes = [np.asarray(c.T_world_cam)[:3, 2] for c in rig.cameras]
    out = {"image_size": raw["image_size"], "rotate180": raw.get("rotate180"),
           "camera_indices": raw.get("camera_indices"), "elp_ids": raw.get("elp_ids"),
           "convention": "OpenCV camera axes (+x right, +y down, +z forward); p_A = T_camA_from_cam @ p_cam; mm",
           "cameras": cams,
           "extrinsics": {"T_camA_from_camB": T_ab.tolist(), "baseline_mm": float(np.linalg.norm(T_ab[:3, 3])),
                          "optical_axis_separation_deg": float(np.degrees(np.arccos(np.clip(axes[0] @ axes[1], -1, 1)))),
                          "stereo_rms_px": raw.get("rms_px"), "n_pairs": raw.get("n_pairs"),
                          "pair_spread": raw.get("pair_spread")},
           "board": {k: raw.get(k) for k in ("board", "squares", "square_mm", "marker_mm", "square_mm_measured")}}
    if world is not None:
        R_cw, o = np.asarray(world["R_camera_from_world"]), np.asarray(world["origin_camera_mm"])
        poses = {}
        for cam in rig.cameras:                        # p_world = R_cw^T (T_A,cam p_cam - o)
            T = np.asarray(cam.T_world_cam)
            R_wc = R_cw.T @ T[:3, :3]
            poses[cam.name] = {"position_world_mm": (R_cw.T @ (T[:3, 3] - o)).tolist(),
                               "R_world_from_cam": R_wc.tolist(),
                               "elevation_deg": float(np.degrees(np.arcsin(np.clip(-R_wc[2, 2], -1, 1)))),
                               "note": "elevation: optical axis below the world xy plane, positive looking down"}
        ev = world.get("evidence", {})
        out["world"] = {"definition": world.get("definition"), "units": world.get("units"),
                        "R_camA_from_world": R_cw.tolist(), "origin_in_camA_mm": o.tolist(),
                        "up_in_camA": world.get("up_camera"), "origin": ev.get("origin"),
                        "method": ev.get("method"), "fixture_fit_rms_px": (ev.get("fixture_fit") or {}).get("rms_px"),
                        "T_camA_from_fixture": ev.get("T_camera_A_from_fixture"),
                        "holder_tilt_from_rope_deg": ev.get("holder_tilt_from_rope_deg"),
                        "physical_accuracy_validated": ev.get("physical_accuracy_validated"),
                        "camera_poses": poses}
    return out


def frames_table(run, sync):
    """Every capture row; host times plus the synced experiment clock where it exists."""
    from LLM.analyse.led import map_times
    f = pd.read_csv(Path(run) / "raw/camera/frames.csv")
    f["mp4_frame"] = np.where(f.written == 1, np.cumsum(f.written) - 1, -1)   # -1: not in the mp4
    for tag, col in (("A", "t_a"), ("B", "t_b")):
        clock = sync.get("clocks", {}).get(f"camera_{tag}")
        if clock:
            f[f"t_exp_{tag.lower()}_s"] = map_times(f[col], clock)
    return f.rename(columns={"t_a": "t_host_a_s", "t_b": "t_host_b_s", "t_capture": "t_host_pair_s"})


def _modes():
    from controller.control import ramp
    return {ramp.POLY: "POLY", ramp.EASE: "EASE", ramp.EXP: "EXP"}


MODES = _modes()


def ramp_profile(run):
    """The flown ramp: segments, label, duration, and the seq= lines in the host log."""
    from controller.control.ramp import duration_s, label
    from LLM.record.ramps import load_ramp, run_ramp_path
    path = run_ramp_path(run)
    segs = load_ramp(path)
    log = Path(run) / "raw/host_commands.log"
    sent = [l.split("-> ", 1)[1].strip() for l in log.read_text().splitlines()
            if "-> seq=" in l] if log.exists() else []
    return {"file": path.name, "label": label(segs), "duration_s": duration_s(segs),
            "note": read_json(path).get("note"), "seq_sent": sent,
            "segments": [dict(zip(("from_hz", "to_hz", "seconds", "mode", "k"), sg)) for sg in segs]}


POWERED_MIN_A = 0.1     # A; a driven coil here peaks at 1.7-2.6 A, an unpowered one at <= 0.02
DRIVEN_MIN_HZ = 5.0     # only samples where the device reports it is ramping count


def coil_power(run):
    """Per coil: peak |current| while the device reports f > DRIVEN_MIN_HZ, and whether it
    was powered by that measure. None when the run never drove. A coil with applied duty
    but no current had no supply (operator, 2026-11-06: A and C were not powered)."""
    from LLM.record.telemetry import parse_log
    t = Path(run) / "raw/telemetry.jsonl"
    if not t.exists():
        return None
    d = parse_log(t)
    on = d[d.frequency_reported_hz > DRIVEN_MIN_HZ]
    if not len(on):
        return None
    return {c: {"current_max_A": float(on[f"current_{c}_A"].abs().max()),
                "powered": bool(on[f"current_{c}_A"].abs().max() >= POWERED_MIN_A)} for c in "ABCD"}


OPERATOR_POWER_NOTE = ("operator, 2026-11-06: 'the earlier recordings had no current on some coils "
                       "because they were not powered on'")


def annotate_coil_power(run, source=OPERATOR_POWER_NOTE):
    """`<run>/annotations/coil_power.json`: which coils were powered (measured), and the
    EFFECTIVE duty -- planned on a powered coil, 0 on an unpowered one. A post-hoc
    correction beside the raw record; raw/ and run_parameters.json are never edited."""
    power = coil_power(run)
    if power is None:
        return None
    ds = duty_summary(run) or {}
    planned = {c: (ds.get("percent") or {}).get(c, 100.0) for c in "ABCD"}
    out = {"schema_version": 1, "written_utc": pd.Timestamp.now(tz="UTC").isoformat(), "source": source,
           "rule": f"powered = peak |I| >= {POWERED_MIN_A} A while the device reports f > {DRIVEN_MIN_HZ} Hz",
           "coils": power, "duty_planned_pct": planned,
           "duty_effective_pct": {c: planned[c] if power[c]["powered"] else 0.0 for c in "ABCD"}}
    path = Path(run) / "annotations/coil_power.json"
    path.parent.mkdir(exist_ok=True)
    write_json(path, out)
    return out


def duty_summary(run):
    """Planned/host-sent duty (run_parameters.json) next to the device-applied duty
    (telemetry, while the device reports a nonzero frequency). None before 2026-11-06."""
    path = Path(run) / "raw/run_parameters.json"
    if not path.exists():
        return None
    duty = read_json(path)["duty"]
    applied = {}
    cur = Path(run) / "processed/current.csv"
    if cur.exists():
        c = pd.read_csv(cur)
        if "duty_applied_A_pct" in c and "frequency_reported_hz" in c:
            on = c[c.frequency_reported_hz > 0]
            applied = {ch: float(on[f"duty_applied_{ch}_pct"].median()) for ch in "ABCD"} if len(on) else {}
    return {**duty, "applied_median_pct": applied or None}


def _md(run, d, sync, leds, sources, verified, rp, ds, params):
    L = [f"# Setup and calibration: `{Path(run).name}`", ""]
    man = read_json(Path(run) / "manifest.json")
    L += [f"Note: {man.get('initial_conditions', {}).get('notes') or '(none)'}", ""]
    L += ["## Caveats (read first)", ""]
    if not verified:
        L.append("- **Calibration UNVERIFIED.** It has not passed a held-out verification; "
                 "geometry from it is not accepted for measurement.")
    b = d["board"]
    if not b.get("square_mm_measured"):
        L.append(f"- Board square size **{b.get('square_mm')} mm is nominal, not caliper-measured**. A scale "
                 "error is invisible to every reprojection check and scales all distances.")
    L.append(f"- Stereo fit from **{d['extrinsics']['n_pairs']} pairs**.")
    if "world" in d:
        w = d["world"]
        L.append(f"- World origin: **{w['origin']}**; physical accuracy validated: {w['physical_accuracy_validated']}.")
    L.append("- Times are host capture-receipt times (not exposure); the LED alignment does not "
             "prove when coil current started.")
    L += ["", "## Camera intrinsics", "",
          f"Image {d['image_size'][0]}x{d['image_size'][1]} px; rotate180 {d['rotate180']}; "
          f"device indices {d['camera_indices']}; distortion model OpenCV (k1, k2, p1, p2, k3).", "",
          "| camera | fx | fy | cx | cy | k1, k2, p1, p2, k3 | rms px |", "|---|---|---|---|---|---|---|"]
    for name, c in d["cameras"].items():
        L.append(f"| {name} | {c['fx']:.2f} | {c['fy']:.2f} | {c['cx']:.2f} | {c['cy']:.2f} | "
                 f"{_f(c['dist_k1k2p1p2k3'], 4)} | {c['rms_px'] or float('nan'):.3f} |")
    e = d["extrinsics"]
    L += ["", "## Extrinsics (camera B relative to camera A)", "",
          f"Baseline **{e['baseline_mm']:.1f} mm**, optical axes **{e['optical_axis_separation_deg']:.1f} deg** apart, "
          f"stereo rms {e['stereo_rms_px']:.3f} px.", "", "`T_camA_from_camB` (mm):", "", "```",
          _f(e["T_camA_from_camB"], 4), "```"]
    if "world" in d:
        w = d["world"]
        L += ["", "## Setup (world) frame", "", f"{w['definition']}. Units {w['units']}. Fitted by `{w['method']}`, "
              f"fixture rms {w['fixture_fit_rms_px'] or float('nan'):.2f} px, holder tilt from rope "
              f"{w['holder_tilt_from_rope_deg'] or float('nan'):.2f} deg.", "",
              "`R_camA_from_world`:", "", "```", _f(w["R_camA_from_world"], 5), "```", "",
              f"World origin in camera A: {_f(w['origin_in_camA_mm'], 2)} mm.  "
              f"Map: p_camA = R_camA_from_world @ p_world + origin_in_camA.", "",
              "| camera | position in world (mm) | looks down by (deg) |", "|---|---|---|"]
        for name, p in w["camera_poses"].items():
            L.append(f"| {name} | {_f(p['position_world_mm'], 1)} | {p['elevation_deg']:.1f} |")
    L += ["", "## Frequency ramp", "", f"`{rp['label']}`, {rp['duration_s']:.1f} s of drive "
          f"(`{rp['file']}`). t = 0 is the START LED; the ramp starts there.", ""]
    if rp["note"]:
        L += [rp["note"], ""]
    L += ["| segment | from Hz | to Hz | seconds | mode | k |", "|---|---|---|---|---|---|"]
    L += [f"| {i} | {g['from_hz']:g} | {g['to_hz']:g} | {g['seconds']:g} | {MODES.get(g['mode'], g['mode'])} | {g['k']:g} |"
          for i, g in enumerate(rp["segments"], 1)]
    L += ["", "Sent to the ESP32: " + (", ".join(f"`{c}`" for c in rp["seq_sent"]) or "(no seq= in the host log)"),
          "Planned f(t): `command_planned.csv`; device-reported frequency: `current.csv` "
          "(`frequency_reported_hz` vs `t_s`)."]
    L += ["", "## Randomised coil duty", ""]
    if ds is None:
        L.append("No `run_parameters.json`: recorded before per-run duty randomisation (2026-11-06); "
                 "duty was unrandomised 100 % on every channel.")
    else:
        L.append(f"Mode **{ds['mode']}**. {ds['meaning']}.")
        if isinstance(ds.get("distribution"), dict):
            dist = ds["distribution"]
            L.append(f"Distribution: {dist['form']}; scale {dist['scale_pct']} %, floor {dist['floor_pct']} %; "
                     f"seed `{ds['seed']}` ({ds['generator']}). Channels drawn independently, per run.")
        L.append(f"Host sent: `{ds['command']}`." if ds["command"] else "No duty command sent.")
    power = read_json(Path(run) / "annotations/coil_power.json") if (Path(run) / "annotations/coil_power.json").exists() else None
    if ds is not None or power:
        L += ["", "| channel | planned % | device-applied % (median while driven) | powered (peak A) | effective % |",
              "|---|---|---|---|---|"]
        for ch in "ABCD":
            plan = ((ds or {}).get("percent") or {}).get(ch, 100.0)
            app = ((ds or {}).get("applied_median_pct") or {}).get(ch)
            pw = (power or {}).get("coils", {}).get(ch)
            eff = (power or {}).get("duty_effective_pct", {}).get(ch)
            app_txt = "n/a" if app is None else f"{app:.1f}"
            pw_txt = "n/a" if pw is None else ("yes" if pw["powered"] else "**NO**") + f" ({pw['current_max_A']:.2f})"
            eff_txt = "n/a" if eff is None else f"{eff:g}"
            L.append(f"| {ch} | {plan:g} | {app_txt} | {pw_txt} | {eff_txt} |")
        if power and not all(v["powered"] for v in power["coils"].values()):
            L.append(f"\n**Unpowered coils** ({power['source']}): their effective duty is 0, whatever was planned.")
        L.append("\nApplied = the ramp's carrier duty times planned/100, as the device reports it; "
                 "the current that flowed is `current.csv` `current_*_A`. Duty is not current.")
    if params and params.get("design"):
        dz = params["design"]
        L += ["", "## Run design", "", f"Seed `{dz['seed']}` ({dz['generator']}); duty random scale "
              f"{dz['duty_scale_pct']} points. Step modes advance only after a kept run.", "",
              "| parameter | mode | rule | value this run | min | max | step |", "|---|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['mode']} | {v['rule']} | {v['value']:g} | {v['lo']:g} | {v['hi']:g} | {v['step']:g} |"
              for k, v in dz["parameters"].items()]
    if params:
        sw = params.get("software", {})
        L += ["", "## Run record", "", f"- Parameters written (UTC): {params.get('created_utc')}",
              f"- Host clock anchor: unix {params['time_anchor']['unix_s']:.6f} s = monotonic "
              f"{params['time_anchor']['monotonic_s']:.6f} s ({params['time_anchor']['meaning']})",
              f"- git {(sw.get('git') or {}).get('commit')} ({(sw.get('git') or {}).get('dirty_files')} uncommitted files); "
              f"python {sw.get('python')}, numpy {sw.get('numpy')}; {sw.get('platform')} on {sw.get('host')}",
              f"- Camera: {params.get('camera')}"]
    L += ["", "## LED and clock sync", ""]
    if leds:
        L.append(f"LED ROIs [x, y, w, h]: A {leds[0]}, B {leds[1]}.")
    L.append(f"Method **{sync.get('method', 'led')}**: {sync.get('association', '')}.")
    for name, c in sync.get("clocks", {}).items():
        L.append(f"- `{name}`: t_exp = {c['scale']:.9f} * t + ({c['offset_s']:.6f}) s")
    L += ["", "## Files", "", "- `A.mp4`, `B.mp4`: every recorded frame, as captured "
          f"({read_json(Path(run) / 'raw/camera/meta.json').get('fps_measured')} fps measured; the mp4 "
          "header rate is for playback only, use `frames.csv` for time).",
          "- `frames.csv`: one row per capture; `mp4_frame` is the frame number in both mp4s "
          "(-1 = dropped from the video); `t_exp_*_s` is the LED-synced experiment time.",
          "- `ramp.json`, `host_commands.log`: the ramp flown and every command the host sent.",
          "- `run_parameters.json`: written before START -- duty draw, seed, distribution, ramp, camera, "
          "calibration, time anchor, software. `telemetry.jsonl`: every serial line, host-stamped. "
          "`manifest.json`, `config.json`, `capture_meta.json`: run and capture records.",
          "- other `*.csv`: copies of `processed/` tables; `setup.json`: every number above, with sources.", "",
          "## Sources", ""]
    L += [f"- `{k}`: `{v['path']}` (sha256 {v['sha256'][:12]})" for k, v in sources.items()]
    return "\n".join(L) + "\n"


def package(run, rig_path, world=None, leds=None, verified=False):
    """Build `<run>/export/package/`; returns its path. `world` is the world-frame dict."""
    run = Path(run)
    sync_path = run / "processed/sync.json"
    sync = read_json(sync_path) if sync_path.exists() else {}
    out = run / "export/package"
    out.mkdir(parents=True, exist_ok=True)
    d = describe(rig_path, world)
    sources = {"stereo_rig": {"path": str(Path(rig_path).resolve()), "sha256": sha256(rig_path)}}
    if sync:
        sources["sync"] = {"path": str(sync_path), "sha256": sha256(sync_path)}
    for tag in "AB":
        src = run / f"raw/camera/{tag}/{tag}.mp4"
        shutil.copy2(src, out / f"{tag}.mp4")
        sources[f"video_{tag}"] = {"path": str(src), "sha256": sha256(src)}
    frames_table(run, sync).to_csv(out / "frames.csv", index=False)
    for csv in sorted((run / "processed").glob("*.csv")):
        shutil.copy2(csv, out / csv.name)
    for src, name in (("raw/run_parameters.json", "run_parameters.json"), ("raw/telemetry.jsonl", "telemetry.jsonl"),
                      ("manifest.json", "manifest.json"), ("config.json", "config.json"),
                      ("raw/camera/meta.json", "capture_meta.json")):
        if (run / src).exists():
            shutil.copy2(run / src, out / name)
            sources[name] = {"path": str(run / src), "sha256": sha256(run / src)}
    params = read_json(run / "raw/run_parameters.json") if (run / "raw/run_parameters.json").exists() else None
    ds = duty_summary(run)
    rp = ramp_profile(run)
    shutil.copy2(run / rp["file"], out / "ramp.json")
    sources["ramp"] = {"path": str(run / rp["file"]), "sha256": sha256(run / rp["file"])}
    if (run / "raw/host_commands.log").exists():
        shutil.copy2(run / "raw/host_commands.log", out / "host_commands.log")
    write_json(out / "setup.json", {"schema_version": 1, "run": run.name, "calibration_verified": verified,
                                    "led_rois_xywh": leds, "sync_method": sync.get("method", "led" if sync else None),
                                    "clocks": sync.get("clocks"), "ramp": rp, "duty": ds,
                                    "run_record": params, **d, "sources": sources})
    (out / "setup.md").write_text(_md(run, d, sync, leds, sources, verified, rp, ds, params), encoding="utf-8")
    return out


def index_runs(root):
    """`<root>/runs_index.csv`: one row per run folder -- time, note, ramp, duty draw
    (planned) and applied medians, sync, capture health -- for analysis across runs.
    Rebuilt from the folders each time, so it can never disagree with them."""
    rows = []
    for run in sorted(Path(root).glob("*_20*")):
        if not (run / "manifest.json").exists():
            continue
        man, row = read_json(run / "manifest.json"), {"run": run.name}
        row["note"] = (man.get("initial_conditions") or {}).get("notes")
        params = read_json(run / "raw/run_parameters.json") if (run / "raw/run_parameters.json").exists() else {}
        row["created_utc"] = params.get("created_utc")
        ds = duty_summary(run) or {}
        row["duty_mode"] = ds.get("mode", "none (before randomisation)")
        row["duty_seed"] = ds.get("seed")
        for ch in "ABCD":
            row[f"duty_planned_{ch}_pct"] = (ds.get("percent") or {}).get(ch, 100.0)
            row[f"duty_applied_{ch}_pct"] = (ds.get("applied_median_pct") or {}).get(ch)
        cp = run / "annotations/coil_power.json"
        if cp.exists():
            power = read_json(cp)
            for ch in "ABCD":
                row[f"coil_powered_{ch}"] = power["coils"][ch]["powered"]
                row[f"current_max_{ch}_A"] = power["coils"][ch]["current_max_A"]
                row[f"duty_effective_{ch}_pct"] = power["duty_effective_pct"][ch]
        for k, v in ((params.get("design") or {}).get("parameters") or {}).items():
            row[f"{k}"], row[f"{k}_mode"] = v["value"], v["mode"]
        try:
            row["ramp"] = ramp_profile(run)["label"]
        except (OSError, ValueError, KeyError):
            row["ramp"] = None
        sync = read_json(run / "processed/sync.json") if (run / "processed/sync.json").exists() else {}
        row["sync_method"] = sync.get("method", "led" if sync else None)
        meta = run / "raw/camera/meta.json"
        if meta.exists():
            m = read_json(meta)
            row.update(n_frames=m.get("n_frames"), dropped=m.get("dropped"), fps_measured=m.get("fps_measured"),
                       capture_error=m.get("capture_error"))
        row["package"] = (run / "export/package/setup.md").exists()
        rows.append(row)
    out = Path(root) / "runs_index.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _self_check():
    """Round trip on a synthetic rig: B 100 mm along A's x, axes 90 deg apart; a world
    frame looking straight down camera A's axis. Asserts the derived numbers."""
    import json
    import tempfile
    K = [[1000.0, 0, 640], [0, 1000.0, 400], [0, 0, 1]]
    Rb = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)          # B's z = A's -x... 90 deg
    Tb = np.eye(4); Tb[:3, :3], Tb[:3, 3] = Rb, [100.0, 0, 0]
    rig = {"cameras": [{"name": "A", "K": K, "dist": [0, 0, 0, 0, 0], "T_world_cam": np.eye(4).tolist()},
                       {"name": "B", "K": K, "dist": [0, 0, 0, 0, 0], "T_world_cam": Tb.tolist()}],
           "image_size": [1280, 800], "rotate180": [False, False], "rms_px": .3, "n_pairs": 6,
           "rms_px_intrinsics": {"A": .3, "B": .3}, "square_mm": 20.32, "square_mm_measured": False}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stereo_rig.json"
        path.write_text(json.dumps(rig))
        # world z = camera A's -z (A looks straight down), origin 500 mm in front of A
        R_cw = np.diag([1.0, -1.0, -1.0])
        d = describe(path, {"R_camera_from_world": R_cw.tolist(), "origin_camera_mm": [0, 0, 500.0]})
    e = d["extrinsics"]
    assert abs(e["baseline_mm"] - 100) < 1e-9 and abs(e["optical_axis_separation_deg"] - 90) < 1e-9
    pa = d["world"]["camera_poses"]["A"]
    assert np.allclose(pa["position_world_mm"], [0, 0, 500]), pa["position_world_mm"]
    assert abs(pa["elevation_deg"] - 90) < 1e-9, pa["elevation_deg"]    # straight down
    assert abs(d["world"]["camera_poses"]["B"]["elevation_deg"]) < 1e-9  # B looks horizontally
    f = pd.DataFrame({"index": [0, 1, 2], "t_capture": [10., 11, 12], "skew_s": 0, "t_a": [10., 11, 12],
                      "t_b": [10., 11, 12], "written": [1, 0, 1]})
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "raw/camera").mkdir(parents=True)
        f.to_csv(Path(tmp) / "raw/camera/frames.csv", index=False)
        t = frames_table(tmp, {"clocks": {"camera_A": {"scale": 1.0, "offset_s": -10.0}}})
    assert t.mp4_frame.tolist() == [0, -1, 1] and t.t_exp_a_s.tolist() == [0, 1, 2]
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "raw").mkdir()
        (Path(tmp) / "ramp.json").write_text(json.dumps({"segments": [
            {"from_hz": 1.0, "to_hz": 230.0, "seconds": 23.6, "shape": "poly", "k": 3.0},
            {"from_hz": 230.0, "to_hz": 230.0, "seconds": 10.0, "shape": "linear"}]}))
        (Path(tmp) / "raw/host_commands.log").write_text(
            "[1.0] -> seq=clear\n[1.0] -> seq=ramp:1:230:23600:0:3\n[1.1] -> stop\n")
        rp = ramp_profile(tmp)
    assert abs(rp["duration_s"] - 33.6) < 1e-9 and len(rp["segments"]) == 2
    assert rp["seq_sent"] == ["seq=clear", "seq=ramp:1:230:23600:0:3"], rp["seq_sent"]
    with tempfile.TemporaryDirectory() as tmp:          # B, D driven; A, C on duty but no supply
        (Path(tmp) / "raw").mkdir()
        lines = [json.dumps({"host_receive_s": 100.0 + i, "line": f"t={1000 * (i + 1)} freq={f} | I[A]: A=0.01 B={i_bd} "
                             f"C=0.00 D={i_bd} | duty[%]: A=100.0 B=100.0 C=100.0 D=100.0 | spread=0 bal=0 trip=0"})
                 for i, (f, i_bd) in enumerate([(0.0, 0.0), (20.0, 1.5), (150.0, 2.3)])]
        (Path(tmp) / "raw/telemetry.jsonl").write_text("\n".join(lines) + "\n")
        a = annotate_coil_power(tmp)
    assert [c for c in "ABCD" if a["coils"][c]["powered"]] == ["B", "D"], a["coils"]
    assert a["duty_effective_pct"] == {"A": 0.0, "B": 100.0, "C": 0.0, "D": 100.0}
    print("package: intrinsics/extrinsics/world poses, frame table, ramp profile, coil power ok")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _self_check()
    else:
        world = read_json(sys.argv[3]) if len(sys.argv) > 3 else None
        print(package(sys.argv[1], sys.argv[2], world))
