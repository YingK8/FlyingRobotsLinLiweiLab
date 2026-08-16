"""Build the self-contained diagnostic page.

    uv run python controller/pose/validation/visualise.py

Reads the sensitivity and sweep CSVs, renders the overlay galleries through
`gallery.py`, and writes one HTML file with everything inlined -- data as JSON,
images as base64 data URIs, charts as SVG drawn by a small script in the page.
Nothing is fetched at view time, which is both a hosting constraint (the artifact
CSP blocks every external host) and the right property for a lab record: the file
still works in a year with no network.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

RESULTS = HERE.parents[2] / "results" / "pose_validation"
DEFAULT_OUT = RESULTS / "pose_diagnostics.html"

FAIL_REL_DEPTH = 0.05


def _b64(png):
    return "data:image/jpeg;base64," + base64.b64encode(png).decode("ascii")


def _tile_json(t):
    return {
        "img": _b64(t.png), "title": t.title, "note": t.note,
        "tilt": round(t.tilt_deg, 1), "sigma": round(t.sigma, 1),
        "ambient": round(t.ambient, 2), "bg": round(t.bg_level, 2),
        "pos": None if math.isnan(t.pos_err_mm) else round(t.pos_err_mm, 2),
        "dz": None if math.isnan(t.dz_mm) else round(t.dz_mm, 2),
        "normal": None if math.isnan(t.normal_err_deg) else round(t.normal_err_deg, 2),
        "amb_deg": None if math.isnan(t.ambiguity_deg) else round(t.ambiguity_deg, 1),
        "rms": None if math.isnan(t.fit_rms_px) else round(t.fit_rms_px, 2),
        "detected": bool(t.detected),
    }


def sensitivity_series(path):
    """Per-level median / RMSE / p95 / failure rate for each swept axis."""
    d = pd.read_csv(path, comment="#")
    out = {}
    for axis in ("noise", "background"):
        sub = d[d.axis == axis]
        if sub.empty:
            continue
        rows = []
        for level, g in sub.groupby("level"):
            det = g[g.detected == 1]
            e = det.pos_err_mm.dropna()
            rows.append({
                "x": float(level),
                "median": float(e.median()) if len(e) else None,
                "rmse": float(math.sqrt((e ** 2).mean())) if len(e) else None,
                "p95": float(e.quantile(0.95)) if len(e) else None,
                "fail": float(g.failed.mean() * 100.0),
                "iou": float(det.seg_iou.median()) if len(det) else None,
                "normal": float(det.normal_err_best_deg.median()) if len(det) else None,
                "n": int(len(g)),
            })
        out[axis] = sorted(rows, key=lambda r: r["x"])
    return out


def tilt_series(dataset_path, radius, cal):
    """Residual vs tilt, from the held-out split of the tuning dataset."""
    import tune

    d, _, te = tune.load(dataset_path)
    r = tune.solve_all(d, te, radius, cal)
    pos = np.linalg.norm(r["dxyz"], axis=1)
    lat = np.hypot(r["dxyz"][:, 0], r["dxyz"][:, 1])
    dz = np.abs(r["dxyz"][:, 2])
    tilt = r["tilt"]

    rows = []
    edges = np.arange(0, 75, 7.5)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (tilt >= lo) & (tilt < hi)
        if m.sum() < 5:
            continue
        rows.append({
            "x": float((lo + hi) / 2),
            "median": float(np.median(pos[m])),
            "p95": float(np.percentile(pos[m], 95)),
            "lateral": float(np.median(lat[m])),
            "depth": float(np.median(dz[m])),
            "normal": float(np.median(r["normal_deg"][m])),
            "chosen": float(np.median(np.abs(r["tilt_err_deg"][m]))),
            "n": int(m.sum()),
        })

    # Scatter of every sample, for the residual-vs-orientation panel. Capped so
    # the inlined JSON stays a sane size; a random subset rather than the first
    # N, so it is not biased by however the dataset happened to be ordered.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pos))[:600]
    scatter = [
        {"tilt": round(float(tilt[i]), 2), "pos": round(float(pos[i]), 3),
         "dz": round(float(r["dxyz"][i, 2]), 3), "lat": round(float(lat[i]), 3),
         "z": round(float(r["z"][i]), 1),
         "normal": round(float(r["normal_deg"][i]), 2)}
        for i in idx
    ]

    # Depth vs lateral, by range band -- the depth amplification law.
    #
    # `predicted` is g(tilt) * z / (2R), not the bare z/(2R) this used to plot.
    # The bare form assumes the tilt is known; here it is estimated from the same
    # ellipse, which costs a further sqrt(3) (see `bounds.depth_lateral_ratio`
    # and lecture notes 13.4). Plotting the tilt-known curve against
    # tilt-unknown data understated the prediction by 73% and made the estimator
    # look worse than the geometry says it should be.
    import bounds

    bands = []
    for lo, hi in ((140, 190), (190, 240), (240, 290), (290, 360)):
        m = (r["z"] >= lo) & (r["z"] < hi)
        if m.sum() < 5:
            continue
        bands.append({
            "x": float((lo + hi) / 2),
            "lateral": float(np.median(lat[m])),
            "depth": float(np.median(dz[m])),
            "rel_depth": float(np.median(dz[m] / r["z"][m]) * 100),
            "predicted": float(bounds.depth_lateral_ratio(
                float(np.median(r["z"][m])), radius,
                float(np.median(tilt[m])))),
            "predicted_naive": float(np.median(r["z"][m]) / (2 * radius)),
            "measured": float(np.median(dz[m]) / max(np.median(lat[m]), 1e-9)),
            "n": int(m.sum()),
        })
    return {"bins": rows, "scatter": scatter, "bands": bands}


def headline(dataset_path, radius, cal):
    import tune

    d, _, te = tune.load(dataset_path)
    r = tune.solve_all(d, te, radius, cal)
    band = (r["tilt"] >= 10) & (r["tilt"] <= 45)
    pos = np.linalg.norm(r["dxyz"], axis=1)
    lat = np.hypot(r["dxyz"][:, 0], r["dxyz"][:, 1])
    return {
        "n_test": int(len(pos)),
        "pos": round(float(np.median(pos[band])), 3),
        "pos_pct": round(float(np.median(pos[band] / r["z"][band]) * 100), 3),
        "lateral": round(float(np.median(lat[band])), 3),
        "depth": round(float(np.median(np.abs(r["dxyz"][band, 2]))), 3),
        "depth_pct": round(float(np.median(np.abs(r["dxyz"][band, 2]) / r["z"][band]) * 100), 3),
        "normal": round(float(np.median(r["normal_deg"][band])), 3),
        "tilt": round(float(np.median(np.abs(r["tilt_err_deg"][band]))), 3),
        "ambiguity": round(float(np.median(r["margin_deg"])), 1),
        "radius": round(radius, 4),
    }


def sensor_modes(tag="final"):
    """The per-sensor-mode evaluation from `resolution_sweep.py`, or ``None``.

    ``resolution_final`` is by definition the sweep at the shipped configuration.
    This used to fall back to an older tag when that file was missing, so the
    page would build during a re-run; the fallback is gone on purpose. A page
    that quietly shows a superseded configuration is worse than a page with one
    section missing, and every earlier tag here was measured before the
    default-argument bug in journal Iteration 12 was found.
    """
    import json as _json

    f = RESULTS / f"resolution_{tag}.json"
    if not f.exists():
        return None
    d = _json.loads(f.read_text())
    tier = d["tiers"].get("core") or next(iter(d["tiers"].values()))
    return {
        "tag": tag, "poses": d.get("poses"), "gated": d.get("gated"),
        "rows": [{
            "mode": f"{m['width']}x{m['height']}",
            "fps": m["fps"],
            "n": m["n"],
            "detected": round(m["detect_rate"] * 100, 1),
            "angle_worst": None if m["n"] == 0 else round(m["ang_max"], 3),
            "pos_worst": None if m["n"] == 0 else round(m["pos_max"], 3),
            "in_spec": round(m["in_spec"] * 100, 1),
        } for m in tier],
    }


def information_floor():
    """The Cramer-Rao results, for the page's "what is possible" section.

    Reads `results/pose_validation/limits.json` (written by `validation/limits.py`)
    and adds the closed-form numbers straight from `bounds.py`, so the page never
    hard-codes a derived constant that might drift from the module.
    """
    import json as _json

    import bounds

    f = RESULTS / "limits.json"
    if not f.exists():
        return None
    d = _json.loads(f.read_text())
    s = d["summary"]

    def med(k):
        return s[k]["median"] if k in s else None

    levels = []
    for key, label, kind in (
        ("photon", "photon limit", "bound"),
        ("quant", "pixel-quantised", "bound"),
        ("hull", "+ convex hull", "bound"),
        ("noiseq", "noise-equivalent", "prediction"),
        ("obs", "measured", "actual"),
    ):
        levels.append({
            "label": label, "kind": kind,
            "pos": med(f"{key}_pos_mm"), "depth": med(f"{key}_depth_mm"),
            "lateral": med(f"{key}_lateral_mm"), "angle": med(f"{key}_angle_deg"),
        })

    return {
        "n": s["n"],
        "levels": levels,
        "boundary": {
            "bias": med("boundary_bias_px"),
            "residual_bias": med("residual_bias_px"),
            "scatter": med("boundary_scatter_px"),
            "robust": med("boundary_scatter_robust_px"),
            "contamination": med("contamination_ratio"),
            "corr_len": med("corr_len_px"),
            "n_points": med("n_points"),
            "n_effective": med("n_effective"),
            "photon_sigma": s["photon_sigma_px"],
            "quant_sigma": s["quantisation_sigma_px"],
        },
        "branch": {
            "wrong_fraction": s.get("branch_wrong_fraction"),
            "margin": s.get("branch_wrong_median_margin_deg"),
            "shipped_angle": med("shipped_angle_deg"),
            "oracle_angle": med("obs_angle_deg"),
        },
        "law": {
            "g0": math.sqrt(3.0),
            "g": [{"tilt": t,
                   "value": bounds.depth_lateral_ratio(1.0, 0.5, t)}
                  for t in (0, 10, 30, 60, 80)],
            "at_250": bounds.depth_lateral_ratio(250.0, 10.204, 30.0),
            "naive_at_250": 250.0 / (2 * 10.204),
        },
    }


def build_payload(dataset_path, sensitivity_path, gain=50.0):
    from calibration import TiltCalibration
    from estimator import RADIUS_MM

    import gallery

    cal = TiltCalibration.load()
    print("rendering galleries...")
    g = gallery.build(gain=gain)
    print("rendering before/after weighting pairs...")
    weighting = gallery.build_weighting(gain=gain)

    print("reading sensitivity...")
    sens = sensitivity_series(sensitivity_path)

    print("computing residual curves...")
    tilt = tilt_series(dataset_path, RADIUS_MM, cal)
    head = headline(dataset_path, RADIUS_MM, cal)

    return {
        "headline": head,
        "gain": gain,
        "modes": sensor_modes(),
        "floor": information_floor(),
        "galleries": {
            "weighting": [
                {"title": w["title"], "note": w["note"],
                 "pair": [{"label": k, **_tile_json(v)} for k, v in w["pair"].items()]}
                for w in weighting
            ],
            "layers": [{"label": k, **_tile_json(v)} for k, v in g["layers"].items()],
            "failures": [_tile_json(t) for t in g["failures"]],
            "conditions": [_tile_json(t) for t in g["conditions"]],
        },
        "sensitivity": sens,
        "tilt": tilt,
        "calibration": {"a": cal.a, "b": cal.b, "radius": RADIUS_MM},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=str(RESULTS / "dataset.npz"))
    ap.add_argument("--sensitivity", default=str(RESULTS / "sensitivity.csv"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--gain", type=float, default=50.0)
    ap.add_argument("--payload-only", action="store_true",
                    help="write the JSON payload without the page (for iterating on the HTML)")
    args = ap.parse_args(argv)

    payload = build_payload(args.dataset, args.sensitivity, gain=args.gain)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The payload is an intermediate the page inlines, not something to read, so
    # it lives with the other machine-generated working files.
    scratch = out.parent / "_ai_scratch"
    scratch.mkdir(exist_ok=True)
    pj = scratch / (out.stem + ".json")
    pj.write_text(json.dumps(payload))
    print(f"payload -> {pj}  ({pj.stat().st_size/1e6:.1f} MB)")

    if args.payload_only:
        return 0

    from page import render_page

    out.write_text(render_page(payload))
    print(f"page -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
