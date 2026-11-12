from __future__ import annotations

import math
import sys

import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov

from controller.control import axis_hover as hover
from controller.control.z_track import TorqueLimits

ASSUMED_DELAY_STEPS = 3
TRUE_DELAY_STEPS = (2, 3, 4)
LATERAL_BOX = {"gain": (0.7, 1.4), "phase_rad": (-math.radians(15), math.radians(15)),
               "thrust_ratio": (0.77, 1.26), "axis_leak_per_mm": (0.0, 0.0)}
LEAK_BOX = {"gain": (1.0, 1.0), "phase_rad": (0.0, 0.0), "thrust_ratio": (1.0, 1.0),
            "axis_leak_per_mm": (-0.004, 0.004)}
EVIDENCE_BOX = {**LATERAL_BOX, "axis_leak_per_mm": LEAK_BOX["axis_leak_per_mm"]}
EVIDENCE_SAMPLES = 3000
VERTICAL_BOX = {"thrust_ratio": (0.5, 2.0)}
DEEPEST_SPLIT = 24


def augmented(plant):
    dynamics, input_column = plant
    size = len(dynamics)
    block = np.zeros((size + 1, size + 1), complex)
    block[:size, :size], block[:size, size] = dynamics, input_column
    return block


def plant_step(plant, step_s):
    return expm(augmented(plant) * step_s)[:len(plant[0])]


def closed_loop(loop, true_plant, true_delay_steps, assumed_delay_steps):
    transition, input_effect = loop.transition, loop.input_effect
    rows, gain, feedback = loop.measured_rows, loop.observer_gain, loop.feedback
    model_size, plant_size = len(transition), len(true_plant[0])
    d, h = true_delay_steps, assumed_delay_steps
    widths = {"plant": plant_size * (d + 1), "slot": model_size, "inputs": h}
    starts, total = {}, 0
    for name, width in widths.items():
        starts[name], total = total, total + width

    def pick(name, offset=0, width=None):
        width = model_size if width is None else width
        selector = np.zeros((width, total), complex)
        selector[:, starts[name] + offset: starts[name] + offset + width] = np.eye(width)
        return selector

    slot = pick("slot")
    seen_plant = rows[:, :plant_size] @ pick("plant", plant_size * d, plant_size)
    corrected = slot + gain @ (seen_plant - rows @ slot)
    past_inputs = [pick("inputs", i, 1) for i in range(h)]
    estimate = np.linalg.matrix_power(transition, h) @ corrected + sum(
        (np.linalg.matrix_power(transition, h - 1 - i) @ input_effect[:, None] @ past_inputs[i]
         for i in range(h)), np.zeros((model_size, total), complex))
    command = -(feedback @ estimate).reshape(1, total)
    following = np.zeros((total, total), complex)
    for lag in range(1, d + 1):
        following[plant_size * lag: plant_size * (lag + 1)] = pick("plant", plant_size * (lag - 1), plant_size)
    following[starts["slot"]:starts["slot"] + model_size] = (
        transition @ corrected + input_effect[:, None] @ (past_inputs[0] if h else command))
    for i in range(h):
        following[starts["inputs"] + i] = past_inputs[i + 1] if i + 1 < h else command
    plant_rows = np.zeros((total, plant_size), complex)
    plant_rows[:plant_size] = np.eye(plant_size)
    state_and_command = np.vstack([pick("plant", 0, plant_size), command])
    following += plant_rows @ plant_step(true_plant, loop.step_s) @ state_and_command
    return following, plant_rows, state_and_command


class LateralUncertainty:
    names = ("gain", "phase_rad", "thrust_ratio", "axis_leak_per_mm")
    plant_size = 3

    def __init__(self, believed_precession):
        self.believed = believed_precession
        rate = abs(believed_precession)
        self.scaling = np.array([rate ** 2 / hover.GRAVITY_MM_PER_S2, rate / hover.GRAVITY_MM_PER_S2, 1.0, 1.0])

    def plant(self, gain, phase_rad, thrust_ratio, axis_leak_per_mm):
        dynamics, input_column = hover.lateral_plant(self.believed * gain * np.exp(1j * phase_rad),
                                                     thrust_ratio, axis_leak_per_mm)
        return dynamics[:3, :3], input_column[:3]

    def directions(self, gain, phase_rad, thrust_ratio, axis_leak_per_mm):
        precession = self.believed * gain * np.exp(1j * phase_rad)
        result = []
        for change in (precession / gain, 1j * precession):
            direction = np.zeros((4, 4), complex)
            direction[2, 0], direction[2, 2], direction[2, 3] = change * axis_leak_per_mm, -change, change
            result.append(direction)
        thrust, leak = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
        thrust[1, 2], leak[2, 0] = hover.GRAVITY_MM_PER_S2, precession
        return result + [thrust, leak]

    def curvature(self, center, half):
        gain, _, _, leak = center
        gain_half, phase_half, _, leak_half = half
        size = abs(self.believed)
        widest_gain = gain + gain_half
        precession_curvature = size * (gain_half * phase_half + widest_gain * phase_half ** 2 / 2)
        leak_curvature = ((abs(leak) + leak_half) * precession_curvature
                          + leak_half * size * (gain_half + widest_gain * phase_half))
        bound = np.zeros((4, 4))
        bound[2, 0], bound[2, 2], bound[2, 3] = leak_curvature, precession_curvature, precession_curvature
        return bound


class VerticalUncertainty:
    names = ("thrust_ratio",)
    plant_size = 2
    scaling = np.array([1.0, 1 / 3.0, 1 / 9.0])

    def plant(self, thrust_ratio):
        dynamics, input_column = hover.vertical_plant(thrust_ratio)
        return dynamics[:2, :2], input_column[:2]

    def directions(self, thrust_ratio):
        direction = np.zeros((3, 3), complex)
        direction[1, 2] = 1
        return [direction]

    def curvature(self, center, half):
        return np.zeros((3, 3))


def exponential_derivative(generator, direction):
    size = len(generator)
    block = np.zeros((2 * size, 2 * size), complex)
    block[:size, :size] = block[size:, size:] = generator
    block[:size, size:] = direction
    return expm(block)[:size, size:]


def peak_gain(following, plant_rows, state_and_command, tolerance=1e-3):
    identity = np.eye(len(following))
    shifted_inverse = np.linalg.inv(following + identity)
    dynamics = shifted_inverse @ (following - identity)
    inputs = math.sqrt(2) * shifted_inverse @ plant_rows
    outputs = math.sqrt(2) * state_and_command @ shifted_inverse
    direct = -state_and_command @ shifted_inverse @ plant_rows
    def crosses(level):
        weight = level ** 2 * np.eye(direct.shape[1]) - direct.conj().T @ direct
        solved = np.linalg.solve(weight, np.eye(len(weight)))
        top_left = dynamics + inputs @ solved @ direct.conj().T @ outputs
        hamiltonian = np.block([
            [top_left, inputs @ solved @ inputs.conj().T],
            [-outputs.conj().T @ (np.eye(len(direct)) + direct @ solved @ direct.conj().T) @ outputs,
             -top_left.conj().T]])
        eigenvalues = np.linalg.eigvals(hamiltonian)
        return bool(np.any(np.abs(eigenvalues.real) < 1e-9 * (1 + np.abs(eigenvalues))))
    low = np.linalg.norm(direct, 2)
    high = max(2 * low, 1.0)
    while crosses(high):
        low, high = high, 2 * high
    while high - low > tolerance * high:
        middle = (low + high) / 2
        low, high = (middle, high) if crosses(middle) else (low, middle)
    return high


def certify_cell(loop, uncertainty, center, half, delay):
    ratio = uncertainty.scaling[:, None] / uncertainty.scaling[None, :]
    plant = uncertainty.plant(*center)
    following, plant_rows, state_and_command = closed_loop(loop, plant, delay, ASSUMED_DELAY_STEPS)
    if np.max(np.abs(np.linalg.eigvals(following))) >= 1:
        return -math.inf
    inputs = plant_rows / uncertainty.scaling[None, :-1]
    outputs = uncertainty.scaling[:, None] * state_and_command
    size = uncertainty.plant_size
    generator = augmented(plant) * ratio * loop.step_s
    linear = 0.0
    first_order = 0.0
    for direction, width in zip(uncertainty.directions(*center), half):
        scaled = direction * ratio * loop.step_s
        derivative = exponential_derivative(generator, scaled)[:size]
        linear += width * peak_gain(following, inputs @ derivative, outputs)
        first_order += width * np.linalg.norm(scaled, 2)
    second_order = float(np.linalg.norm(uncertainty.curvature(center, half) * ratio)) * loop.step_s
    spread = first_order + second_order
    generator_size = np.linalg.norm(generator, 2)
    remainder = (second_order * math.exp(generator_size)
                 + spread ** 2 / 2 * math.exp(generator_size + spread))
    return 1 - linear - remainder * peak_gain(following, inputs, outputs)


def certify(loop, uncertainty, box, delay):
    low = np.array([box[name][0] for name in uncertainty.names])
    high = np.array([box[name][1] for name in uncertainty.names])
    span = np.where(high > low, high - low, np.inf)
    pending, cells, least_slack = [(low, high, 0)], 0, math.inf
    while pending:
        cell_low, cell_high, depth = pending.pop()
        center, half = (cell_low + cell_high) / 2, (cell_high - cell_low) / 2
        slack = certify_cell(loop, uncertainty, center, half, delay)
        if slack > 0:
            cells, least_slack = cells + 1, min(least_slack, slack)
            continue
        if slack == -math.inf or depth >= DEEPEST_SPLIT:
            return {"certified": False, "failed_at": np.round(center, 4).tolist(), "depth": depth}
        widest = int(np.argmax(half / span))
        lower_high, upper_low = cell_high.copy(), cell_low.copy()
        lower_high[widest] = upper_low[widest] = center[widest]
        pending += [(cell_low, lower_high, depth + 1), (upper_low, cell_high, depth + 1)]
    return {"certified": True, "cells": cells, "least_slack": round(least_slack, 4)}


def peak_command(loop, plant, offset, steps=6000):
    following, _, state_and_command = closed_loop(loop, plant, ASSUMED_DELAY_STEPS, ASSUMED_DELAY_STEPS)
    plant_size, model_size = len(plant[0]), len(loop.transition)
    state = np.zeros(len(following), complex)
    state[:plant_size * (ASSUMED_DELAY_STEPS + 1):plant_size] = offset
    state[plant_size * (ASSUMED_DELAY_STEPS + 1)] = offset
    commands = []
    for _ in range(steps):
        commands.append(state_and_command[-1] @ state)
        state = following @ state
    commands = np.array(commands)
    return float(np.max(np.abs(commands))), float(np.max(np.abs(np.diff(commands)))), float(np.linalg.norm(state))


def largest_unsaturated_offsets(controller):
    believed = controller.response.precession_per_s
    tilt_per_mm, _, lateral_tail = peak_command(controller.lateral, LateralUncertainty(believed).plant(1, 0, 1, 0), 1.0)
    acceleration_per_mm, change_per_mm, vertical_tail = peak_command(
        controller.vertical, VerticalUncertainty().plant(1.0), 1.0)
    hover_hz, (lowest_hz, highest_hz) = controller.hover_frequency_hz, controller.frequency_band_hz
    band_acceleration = hover.GRAVITY_MM_PER_S2 * min(1 - (lowest_hz / hover_hz) ** 2,
                                                       (highest_hz / hover_hz) ** 2 - 1)
    slew_acceleration = 2 * hover.GRAVITY_MM_PER_S2 * controller.frequency_step_hz / hover_hz
    return {"lateral_mm": controller.field_tilt_limit_rad / tilt_per_mm,
            "height_band_mm": band_acceleration / acceleration_per_mm,
            "height_slew_mm": slew_acceleration / change_per_mm,
            "frequency_hz_per_mm": hover_hz * acceleration_per_mm / (2 * hover.GRAVITY_MM_PER_S2),
            "decayed": max(lateral_tail, vertical_tail) < 1e-6}


def phase_lock_margin(controller, torque_scale=1.0):
    limits = TorqueLimits(f_hover=controller.hover_frequency_hz)
    inertial = 2 * math.pi * limits.i_robot * controller.frequency_step_hz / controller.step_s
    worst = math.inf
    for frequency_hz in np.linspace(*controller.frequency_band_hz, 61):
        peak_torque = limits.tau_max(frequency_hz) * torque_scale
        drag = limits.k_drag * frequency_hz ** 2
        load = (drag + inertial) / peak_torque
        if load >= 1:
            return -math.inf
        before = math.asin((drag - inertial) / peak_torque)
        potential = lambda angle: -math.cos(angle) - load * angle
        worst = min(worst, potential(math.pi - math.asin(load)) - potential(before))
    return worst


def weakest_torque_scale(controller):
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = (low + high) / 2
        low, high = (low, middle) if phase_lock_margin(controller, middle) > 0 else (middle, high)
    return high


def sampled_evidence(controller, seed=0):
    uncertainty = LateralUncertainty(controller.response.precession_per_s)
    low = np.array([EVIDENCE_BOX[name][0] for name in uncertainty.names])
    high = np.array([EVIDENCE_BOX[name][1] for name in uncertainty.names])
    corners = np.array(np.meshgrid(*zip(low, high))).reshape(len(low), -1).T
    points = np.vstack([corners, np.random.default_rng(seed).uniform(low, high, (EVIDENCE_SAMPLES, len(low)))])
    return max(float(np.max(np.abs(np.linalg.eigvals(
        closed_loop(controller.lateral, uncertainty.plant(*point), delay, ASSUMED_DELAY_STEPS)[0]))))
        for point in points for delay in TRUE_DELAY_STEPS)


def main(tex_path=None):
    controller = hover.AxisHoverController()
    believed = controller.response.precession_per_s
    results = {}
    for delay in TRUE_DELAY_STEPS:
        results[delay] = {
            "lateral": certify(controller.lateral, LateralUncertainty(believed), LATERAL_BOX, delay),
            "leak": certify(controller.lateral, LateralUncertainty(believed), LEAK_BOX, delay),
            "vertical": certify(controller.vertical, VerticalUncertainty(), VERTICAL_BOX, delay)}
        print(f"delay {delay} steps = {1000 * delay * controller.step_s:.1f} ms: {results[delay]}", flush=True)
    worst_radius = sampled_evidence(controller)
    region = largest_unsaturated_offsets(controller)
    margin, weakest = phase_lock_margin(controller), weakest_torque_scale(controller)
    print(f"sampled 4-D box incl. leak, {EVIDENCE_SAMPLES} points + corners x 3 delays: worst spectral radius {worst_radius:.6f}")
    print(f"largest initial offsets that never saturate, nominal plant: {region}")
    print(f"phase-lock energy margin {margin:.3f} (> 0 keeps lock), still > 0 down to {weakest:.3f} x modelled torque")
    if tex_path:
        with open(tex_path, "w") as file:
            for delay, parts in results.items():
                for name, result in parts.items():
                    file.write(f"\\newcommand{{\\cert{name.title()}{'ABCDE'[delay]}}}"
                               f"{{{result['cells'] if result['certified'] else 'FAIL'}}}\n")
            for name, key in (("LateralMm", "lateral_mm"), ("HeightBandMm", "height_band_mm"),
                              ("HeightSlewMm", "height_slew_mm"), ("HzPerMm", "frequency_hz_per_mm")):
                file.write(f"\\newcommand{{\\region{name}}}{{{region[key]:.2f}}}\n")
            file.write(f"\\newcommand{{\\evidenceRadius}}{{{worst_radius:.5f}}}\n")
            file.write(f"\\newcommand{{\\lockMargin}}{{{margin:.3f}}}\n")
            file.write(f"\\newcommand{{\\lockTorqueScale}}{{{weakest:.3f}}}\n")
    return results, worst_radius, region, margin


if __name__ == "__main__":
    results, worst_radius, region, margin = main(sys.argv[1] if len(sys.argv) > 1 else None)
    assert all(result["certified"] for parts in results.values() for result in parts.values())
    assert worst_radius < 1 and margin > 0 and region["decayed"]
    print("axis_hover_proof: every cell certified")
