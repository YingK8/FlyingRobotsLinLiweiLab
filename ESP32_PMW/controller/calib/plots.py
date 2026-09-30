#!/usr/bin/env python3
"""What the solve looked like: coverage, residuals, and the lens model applied."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from controller.calib.calibrate import MAX_INCIDENCE_DEG, OUT_DIR, PAIR_DIR, load_views
from controller.calib.results import MAX_RMS_PX

if not sys.stdout.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def coverage_figure(spec, pair_dir, out_dir=None):
    """Where the saved corners actually landed. Holes are unconstrained distortion."""

    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    views_a, views_b, size = load_views(spec, pair_dir)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, tag, views, colour in zip(ax, "AB", (views_a, views_b), ("#2a78d6", "#1baf7a")):
        if views:
            pts = np.concatenate([v["corners"].reshape(-1, 2) for v in views])
            a.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.4, color=colour)
            a.set_xlim(0, size[0])
            a.set_ylim(size[1], 0)
        a.set_aspect("equal")
        a.set_title(f"camera {tag}: corner coverage")
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "capture_coverage.png"
    fig.savefig(path, dpi=140)
    print(f"coverage -> {path}")
    return fig


def figures(pairs, image_size, resid, out_dir=OUT_DIR):
    """Five-panel diagnostic of the solve, written next to the rig."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    x, per_pair = np.arange(len(pairs)), resid["per_pair"]

    for i, (tag, colour) in enumerate(zip("AB", ("#2a78d6", "#1baf7a"))):
        res, rad, low = resid[tag]["res"], resid[tag]["rad"], tag.lower()
        ax[0, 0].bar(x + 0.2 * (2 * i - 1), per_pair[:, i], 0.4, label=tag, color=colour)
        ax[0, 1].scatter(res[:, 0], res[:, 1], s=4, alpha=0.35, color=colour, label=tag)

        order = np.argsort(rad)
        mag = np.linalg.norm(res, axis=1)[order]
        k = max(1, len(mag) // 30)
        ax[0, 2].plot(rad[order][::k], np.convolve(mag, np.ones(k) / k, "same")[::k],
                      color=colour, label=tag)

        pts = np.concatenate([p[f"img_{low}"].reshape(-1, 2) for p in pairs])
        ax[1, 0].scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.4, color=colour, label=tag)
        ax[1, 1].hist([p[f"incidence_{low}"] for p in pairs], bins=12, alpha=0.6,
                      label=tag, color=colour)

    ax[0, 0].axhline(MAX_RMS_PX, color="#e34948", ls="--", label=f"{MAX_RMS_PX} px limit")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels([p["index"] for p in pairs], rotation=90, fontsize=7)
    ax[0, 0].set(ylabel="joint reprojection RMS (px)",
                 title=f"per pair (overall {resid['rms_px']:.3f} px)")

    ax[0, 1].axhline(0, lw=0.5, color="k")
    ax[0, 1].axvline(0, lw=0.5, color="k")
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set(xlabel="dx (px)", ylabel="dy (px)",
                 title="residuals - want a round, centred blob")
    ax[0, 2].set(xlabel="radius from principal point (px)", ylabel="|residual| (px)",
                 title="radial trend - a rising line is underfit distortion")

    ax[1, 0].set_xlim(0, image_size[0])
    ax[1, 0].set_ylim(image_size[1], 0)
    ax[1, 0].set_aspect("equal")
    ax[1, 0].set_title("corner coverage - holes are unconstrained distortion")

    ax[1, 1].axvline(MAX_INCIDENCE_DEG, color="#e34948", ls="--", label="reject limit")
    ax[1, 1].set(xlabel="board incidence (deg, 0 = face-on)",
                 title="incidence - can one board serve both cameras?")
    fig.delaxes(ax[1, 2])
    for a in ax.ravel()[:-1]:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"stereo calibration - {len(pairs)} pairs at "
                 f"{image_size[0]}x{image_size[1]}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / "stereo_calibration.png"
    fig.savefig(path, dpi=140)
    print(f"figures -> {path}")
    return fig

def undistort_figure(cal, pair_dir=PAIR_DIR, out_dir=OUT_DIR, index=0):
    """One saved pair, before and after the lens model. Straight edges should straighten."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for row, tag in enumerate("AB"):
        shots = sorted((Path(pair_dir) / tag).glob("pair_*.png"))
        if not shots:
            continue
        shot = shots[min(index, len(shots) - 1)]
        raw = cv2.imread(str(shot), cv2.IMREAD_GRAYSCALE)
        k = tag.lower()
        fixed = cv2.undistort(raw, cal[f"K_{k}"], cal[f"dist_{k}"])
        for a, img, what in zip(ax[row], (raw, fixed), ("distorted", "undistorted")):
            a.imshow(img, cmap="gray")
            a.set_title(f"camera {tag}: {what} ({shot.name})", fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()
    path = out_dir / "undistort_preview.png"
    fig.savefig(path, dpi=140)
    print(f"undistort preview -> {path}")
    return fig
