#!/usr/bin/env python3
"""Offline validation of the fixed-iteration projected-gradient duty
allocator (mirrored in firmware as src/DutyAllocator.h/.cpp) BEFORE it ever
runs on hardware -- see the state-space plan's Step 2.

Solves: min ||Bd@u - x_target||^2 + r_duty*||u-u_ff||^2
        s.t. DUTY_MIN <= u_i <= DUTY_MAX

via u^(0) = clamp(duty_prev), then 15 fixed iterations of
u = clamp(u - alpha*grad, DUTY_MIN, DUTY_MAX), alpha = 1/L,
L = 2*||Bd||_F^2 + 2*r_duty (computed once, not per-iteration).

Validates against the known CCW channel-C-saturation scenario (channel C
needs ~90-100% duty to hit target, confirmed on hardware repeatedly this
session) and cross-checks against scipy.optimize.lsq_linear (a real bound-
constrained least-squares solver) as ground truth -- offline only, never
used on-device.

Usage:
  uv run python tools/validate_duty_allocator.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import lsq_linear

HERE = os.path.dirname(os.path.abspath(__file__))
DUTY_MIN, DUTY_MAX = 5.0, 100.0
N_ITERS = 15


def allocate(Bd, x_target, u_ff, duty_prev, r_duty):
    n = Bd.shape[0]
    L = 2 * np.linalg.norm(Bd, "fro") ** 2 + 2 * r_duty
    alpha = 1.0 / L
    u = np.clip(duty_prev, DUTY_MIN, DUTY_MAX).copy()
    for _ in range(N_ITERS):
        resid = Bd @ u - x_target
        grad = 2 * (Bd.T @ resid) + 2 * r_duty * (u - u_ff)
        u = np.clip(u - alpha * grad, DUTY_MIN, DUTY_MAX)
    return u


def naive_clamp(u_desired):
    return np.clip(u_desired, DUTY_MIN, DUTY_MAX)


def cost(Bd, u, x_target, u_ff, r_duty):
    resid = Bd @ u - x_target
    return float(resid @ resid + r_duty * np.sum((u - u_ff) ** 2))


def main() -> None:
    model_path = os.path.join(HERE, "state_space_model_ccw.json")
    with open(model_path) as f:
        model = json.load(f)
    Bd = np.array(model["Bd"])
    n = Bd.shape[0]
    channels = model["channels"]
    print(f"channels: {channels}")
    print(f"Bd cond: {np.linalg.cond(Bd):.3f}")

    # Known CCW scenario: r_target=2.0A shared target, channel C (index 2)
    # historically needs ~90-100% duty -- construct u_desired (the
    # unconstrained feedforward+feedback "ideal" duty) that pushes C past
    # DUTY_MAX to exercise the constrained case.
    r_target = 2.0
    x_target = np.full(n, r_target)
    FF = np.linalg.inv(Bd)
    u_ff = FF @ x_target
    print(f"u_ff (unconstrained feedforward) = {u_ff.round(2)}")

    # Simulate a feedback correction that pushes channel C's desired duty
    # further past 100% (as observed on hardware: unclamped duty_unclamped
    # for C exceeded DUTY_MAX before this allocator existed).
    u_desired = u_ff.copy()
    c_idx = channels.index("C")
    u_desired[c_idx] += 30.0  # exaggerate to guarantee the saturated case
    print(f"u_desired (unconstrained FF+feedback) = {u_desired.round(2)}")

    x_target_eff = Bd @ u_desired
    print(f"x_target_eff = Bd @ u_desired = {x_target_eff.round(3)}")

    r_duty = 0.001
    duty_prev = np.full(n, 50.0)  # arbitrary warm-start

    u_allocated = allocate(Bd, x_target_eff, u_ff, duty_prev, r_duty)
    u_naive = naive_clamp(u_desired)

    print(f"\nallocator result:   u = {u_allocated.round(3)}")
    print(f"naive clamp result: u = {u_naive.round(3)}")

    assert np.all(u_allocated >= DUTY_MIN - 1e-6) and np.all(u_allocated <= DUTY_MAX + 1e-6), \
        "allocator violated duty bounds"
    print("\n[PASS] allocator respects DUTY_MIN/DUTY_MAX")

    cost_allocated = cost(Bd, u_allocated, x_target_eff, u_ff, r_duty)
    cost_naive = cost(Bd, u_naive, x_target_eff, u_ff, r_duty)
    print(f"\ncost(allocator) = {cost_allocated:.5f}")
    print(f"cost(naive clamp) = {cost_naive:.5f}")
    assert cost_allocated <= cost_naive + 1e-6, "allocator did not beat naive clamp"
    print("[PASS] allocator cost <= naive per-channel clamp cost")

    # Ground truth via scipy's bound-constrained least squares (offline only).
    # Reformulate as min ||[Bd; sqrt(r_duty)*I] @ u - [x_target_eff; sqrt(r_duty)*u_ff]||^2
    A = np.vstack([Bd, np.sqrt(r_duty) * np.eye(n)])
    b = np.concatenate([x_target_eff, np.sqrt(r_duty) * u_ff])
    res = lsq_linear(A, b, bounds=(DUTY_MIN, DUTY_MAX))
    u_truth = res.x
    cost_truth = cost(Bd, u_truth, x_target_eff, u_ff, r_duty)
    print(f"\nscipy lsq_linear ground truth: u = {u_truth.round(3)}")
    print(f"cost(ground truth) = {cost_truth:.5f}")

    rel_gap = (cost_allocated - cost_truth) / max(cost_truth, 1e-9)
    print(f"relative cost gap (allocator vs ground truth): {rel_gap*100:.3f}%")
    assert rel_gap < 0.05, f"15-iteration allocator is {rel_gap*100:.2f}% worse than optimal -- too loose"
    print(f"[PASS] {N_ITERS}-iteration fixed allocator within 5% of true constrained optimum")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
