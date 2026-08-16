#!/usr/bin/env python3
"""Piecewise reference-trajectory profiles for the hover controller --
the position-domain analogue of the frequency segment tables in
model/frequency_modulation_piecewise_*_gui.m (Hold / Polynomial / Exponential)
and the firmware PhaseSequencer's TaskMode LINEAR/EASE/EXPONENTIAL.

Each axis (lateral, vertical) is a chained list of segments; each segment
blends start->end position over its duration with blend s in [0,1]:

  hold         : blend = 0
  linear       : blend = s                        (polynomial order 1)
  polynomial   : blend = s^n                      (n=2 quadratic ease, ...)
  exponential  : blend = (1-e^{-k s})/(1-e^{-k})

Velocity/acceleration come from closed-form derivatives of the blend, so
the controller's reference feedforward is exact (no numeric differencing).

JSON schema (mirrors the GUI segment-table columns):
  {"lateral":  [{"type":"hold","start":0,"end":0,"duration":2,"shape":0}, ...],
   "vertical": [...]}
Auto-chaining: a segment's start is snapped to the previous segment's end
(same as the GUI's auto-chain checkbox).

Demo plot: uv run python ai/reference_profiles.py
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

SEGMENT_TYPES = ("hold", "linear", "polynomial", "exponential")


@dataclass
class Segment:
    type: str          # hold | linear | polynomial | exponential
    start: float       # position at segment start [m]
    end: float         # position at segment end [m]
    duration: float    # [s]
    shape: float = 0.0  # polynomial order n (>=1) or exponential k

    def __post_init__(self):
        if self.type not in SEGMENT_TYPES:
            raise ValueError(f"unknown segment type {self.type!r} (use {SEGMENT_TYPES})")
        if self.duration <= 0:
            raise ValueError("segment duration must be > 0")
        if self.type == "polynomial":
            n = round(self.shape)
            if n < 1 or abs(self.shape - n) > 1e-9:
                raise ValueError("polynomial order must be a positive integer")
            self.shape = float(n)

    def eval(self, tau: float) -> tuple[float, float, float]:
        """(pos, vel, acc) at local time tau in [0, duration]."""
        T, dp = self.duration, self.end - self.start
        s = min(max(tau / T, 0.0), 1.0)
        if self.type == "hold" or dp == 0.0:
            return self.start if self.type == "hold" else self.end, 0.0, 0.0
        if self.type == "linear":
            b, bd, bdd = s, 1.0 / T, 0.0
        elif self.type == "polynomial":
            n = self.shape
            b = s**n
            bd = n * s ** (n - 1) / T
            bdd = n * (n - 1) * s ** (n - 2) / T**2 if n >= 2 else 0.0
        else:  # exponential
            k = self.shape
            if abs(k) < 1e-9:  # degenerates to linear (same convention as the GUI)
                b, bd, bdd = s, 1.0 / T, 0.0
            else:
                den = 1.0 - math.exp(-k)
                e = math.exp(-k * s)
                b = (1.0 - e) / den
                bd = k * e / (den * T)
                bdd = -(k**2) * e / (den * T**2)
        return self.start + dp * b, dp * bd, dp * bdd


class AxisProfile:
    """Chained segments for one axis. Past the end: hold the final position."""

    def __init__(self, segments: list[Segment], auto_chain: bool = True):
        if auto_chain:
            for prev, seg in zip(segments, segments[1:]):
                seg.start = prev.end
                if seg.type == "hold":
                    seg.end = seg.start
        self.segments = segments
        self.edges = [0.0]
        for seg in segments:
            self.edges.append(self.edges[-1] + seg.duration)

    @property
    def total_time(self) -> float:
        return self.edges[-1]

    def eval(self, t: float) -> tuple[float, float, float]:
        if not self.segments:
            return 0.0, 0.0, 0.0
        if t >= self.total_time:
            return self.segments[-1].end, 0.0, 0.0
        # linear scan is fine: profiles are a handful of segments
        for i, seg in enumerate(self.segments):
            if t < self.edges[i + 1]:
                return seg.eval(t - self.edges[i])
        return self.segments[-1].end, 0.0, 0.0  # unreachable


class Profile:
    """Two-axis (lateral, vertical) reference trajectory."""

    def __init__(self, lateral: AxisProfile, vertical: AxisProfile):
        self.lateral = lateral
        self.vertical = vertical

    @classmethod
    def hold(cls, x: float = 0.0, z: float = 0.0, duration: float = 1e9) -> "Profile":
        return cls(AxisProfile([Segment("hold", x, x, duration)]),
                   AxisProfile([Segment("hold", z, z, duration)]))

    def eval(self, t: float):
        """Returns (pos, vel, acc), each a 2-list [lateral, vertical]."""
        xl = self.lateral.eval(t)
        xv = self.vertical.eval(t)
        return [xl[0], xv[0]], [xl[1], xv[1]], [xl[2], xv[2]]

    @classmethod
    def from_json(cls, path: str) -> "Profile":
        with open(path) as f:
            d = json.load(f)
        return cls(AxisProfile([Segment(**s) for s in d["lateral"]]),
                   AxisProfile([Segment(**s) for s in d["vertical"]]))

    def to_json(self, path: str) -> None:
        d = {"lateral": [asdict(s) for s in self.lateral.segments],
             "vertical": [asdict(s) for s in self.vertical.segments]}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


def demo_profile() -> Profile:
    """hold -> quadratic-ease climb 10mm -> hold, with a linear 15mm lateral
    translate partway through -- the sim's profile-tracking scenario."""
    vertical = AxisProfile([
        Segment("hold", 0.0, 0.0, 2.0),
        Segment("polynomial", 0.0, 0.010, 3.0, shape=2),
        Segment("hold", 0.010, 0.010, 7.0),
    ])
    lateral = AxisProfile([
        Segment("hold", 0.0, 0.0, 6.0),
        Segment("linear", 0.0, 0.015, 3.0),
        Segment("hold", 0.015, 0.015, 3.0),
    ])
    return Profile(lateral, vertical)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    prof = demo_profile()
    t = np.linspace(0, 12, 4801)
    pos = np.array([prof.eval(ti)[0] for ti in t])
    vel = np.array([prof.eval(ti)[1] for ti in t])
    acc = np.array([prof.eval(ti)[2] for ti in t])

    # analytic vel/acc must match numeric differentiation of pos/vel
    for name, sig, dsig in (("vel", pos, vel), ("acc", vel, acc)):
        num = np.gradient(sig, t, axis=0)
        # compare away from segment edges (numeric diff smears the kinks)
        interior = np.ones(len(t), bool)
        for ax in (prof.lateral, prof.vertical):
            for e in ax.edges:
                interior &= np.abs(t - e) > 0.02
        err = np.max(np.abs(num[interior] - dsig[interior]))
        print(f"max |numeric - analytic| {name}: {err:.3e}")
        assert err < 1e-2, f"{name} derivative mismatch"
    print("derivative self-check PASS")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    for i, (ylab, sig) in enumerate((("pos [mm]", pos * 1e3),
                                     ("vel [mm/s]", vel * 1e3),
                                     ("acc [mm/s²]", acc * 1e3))):
        axes[i].plot(t, sig[:, 0], label="lateral")
        axes[i].plot(t, sig[:, 1], label="vertical")
        axes[i].set_ylabel(ylab)
        axes[i].grid(True)
    axes[0].legend()
    axes[0].set_title("reference_profiles demo: ease climb + linear translate")
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    out = "reference_profiles_demo.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
