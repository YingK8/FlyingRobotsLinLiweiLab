from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.signal import savgol_filter

GRAVITY_MM_PER_S2 = 9810.0
MODELLED_PRECESSION_PER_S = 3.52 * np.exp(-1j * math.radians(77.2))
LATERAL_POLES_PER_S = (-12.0, -4.0 + 4.0j, -4.0 - 4.0j)
VERTICAL_POLES_PER_S = (-2.22 + 2.22j, -2.22 - 2.22j)
POSITION_NOISE_MM, TILT_NOISE_RAD, TILT_TRUST_RAD = 0.034, 0.0039, 0.1
ACCELERATION_NOISE_MM_PER_S2, TILT_RATE_NOISE_RAD_PER_S = 30.0, 0.3
TILT_BIAS_DRIFT_RAD_PER_S, THRUST_BIAS_DRIFT_MM_PER_S3 = 0.01, 300.0
SINGULAR_SHELL_MM = (19.3, 22.3)


@dataclass
class AxisResponse:
    precession_per_s: complex = MODELLED_PRECESSION_PER_S
    field_gain: complex = 1.0
    field_offset_rad: complex = 0.0
    axis_leak_per_mm: complex = 0.0

    def save(self, path):
        with open(path, "w") as file:
            json.dump({name: [complex(value).real, complex(value).imag]
                       for name, value in asdict(self).items()}, file, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as file:
            return cls(**{name: complex(*pair) for name, pair in json.load(file).items()})


def discretised(dynamics, input_column, step_s):
    size = len(dynamics)
    block = np.zeros((size + 1, size + 1), complex)
    block[:size, :size], block[:size, size] = dynamics, input_column
    transition = expm(block * step_s)
    return transition[:size, :size], transition[:size, size]


def lateral_plant(precession, thrust_ratio=1.0, axis_leak_per_mm=0.0):
    dynamics = np.array([[0, 1, 0, 0],
                         [0, 0, GRAVITY_MM_PER_S2 * thrust_ratio, 0],
                         [precession * axis_leak_per_mm, 0, -precession, precession],
                         [0, 0, 0, 0]], complex)
    return dynamics, np.array([0, 0, precession, 0], complex)


def vertical_plant(thrust_ratio=1.0):
    dynamics = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], complex)
    return dynamics, np.array([0, thrust_ratio, 0], complex)


def lateral_feedback(precession, axis_leak_per_mm=0.0, poles=LATERAL_POLES_PER_S):
    attitude_rate, damping, stiffness = np.real(np.poly(poles))[1:]
    per_tilt = precession * GRAVITY_MM_PER_S2
    return np.array([stiffness / per_tilt + axis_leak_per_mm, damping / per_tilt,
                     attitude_rate / precession - 1, 1])


def vertical_feedback(poles=VERTICAL_POLES_PER_S):
    damping, stiffness = np.real(np.poly(poles))[1:]
    return np.array([stiffness, damping, 1], complex)


class DelayedLoop:
    def __init__(self, plant, measured_rows, feedback, step_s, process_noise, measurement_noise,
                 history_steps=12):
        self.transition, self.input_effect = discretised(*plant, step_s)
        self.measured_rows = np.asarray(measured_rows, complex)
        self.feedback, self.step_s = feedback, step_s
        predicted_covariance = solve_discrete_are(
            self.transition.conj().T, self.measured_rows.conj().T,
            np.diag(np.square(process_noise) * step_s), np.diag(np.square(measurement_noise)))
        innovation = self.measured_rows @ predicted_covariance @ self.measured_rows.conj().T
        self.observer_gain = (predicted_covariance @ self.measured_rows.conj().T
                              @ np.linalg.inv(innovation + np.diag(np.square(measurement_noise))))
        self.history = deque(maxlen=history_steps)
        self.estimate = None

    def start(self, measurement):
        self.estimate = np.linalg.pinv(self.measured_rows) @ measurement
        self.history = deque(maxlen=self.history.maxlen)

    def correct(self, measurement, age_steps):
        replayed = list(self.history)[len(self.history) - min(age_steps, len(self.history)):]
        then = replayed[0][0] if replayed else self.estimate
        corrected = then + self.observer_gain @ (measurement - self.measured_rows @ then)
        for entry in replayed:
            entry[0] = corrected
            corrected = self.transition @ corrected + self.input_effect * entry[1]
        self.estimate = corrected

    def advance(self, limit):
        applied = limit(-self.feedback @ self.estimate)
        self.history.append([self.estimate.copy(), applied])
        self.estimate = self.transition @ self.estimate + self.input_effect * applied
        return applied


@dataclass
class HoverCommand:
    vec_line: str
    frequency_hz: float
    field_tilt_sent: complex


class AxisHoverController:
    def __init__(self, response=AxisResponse(), hover_point_mm=(0.0, 0.0, 15.0),
                 hover_frequency_hz=101.0, frequency_band_hz=(85.0, 115.0),
                 frequency_slew_hz_per_s=5.0, field_tilt_limit_rad=0.35, field_mT=2.0,
                 coil_offset_mm=(0.0, 0.0, 0.0), coil_yaw_rad=0.0, step_s=1 / 120):
        self.response, self.step_s = response, step_s
        self.hover_point_mm = np.asarray(hover_point_mm, float)
        coil_height_mm = self.hover_point_mm[2] + coil_offset_mm[2]
        if SINGULAR_SHELL_MM[0] < coil_height_mm < SINGULAR_SHELL_MM[1]:
            raise ValueError(f"hover height {coil_height_mm:.1f} mm above the coils is on the singular "
                             f"shell {SINGULAR_SHELL_MM} mm, where no tilt is reachable")
        self.hover_frequency_hz, self.frequency_band_hz = hover_frequency_hz, frequency_band_hz
        self.frequency_step_hz = frequency_slew_hz_per_s * step_s
        self.field_tilt_limit_rad, self.field_mT = field_tilt_limit_rad, field_mT
        self.coil_offset_mm, self.coil_rotation = np.asarray(coil_offset_mm), np.exp(1j * coil_yaw_rad)
        self.lateral = DelayedLoop(
            lateral_plant(response.precession_per_s, 1.0, response.axis_leak_per_mm),
            [[1, 0, 0, 0], [0, 0, 1, 0]],
            lateral_feedback(response.precession_per_s, response.axis_leak_per_mm), step_s,
            [1e-3, ACCELERATION_NOISE_MM_PER_S2, TILT_RATE_NOISE_RAD_PER_S, TILT_BIAS_DRIFT_RAD_PER_S],
            [POSITION_NOISE_MM, TILT_TRUST_RAD])
        self.vertical = DelayedLoop(
            vertical_plant(), [[1, 0, 0]], vertical_feedback(), step_s,
            [1e-3, ACCELERATION_NOISE_MM_PER_S2, THRUST_BIAS_DRIFT_MM_PER_S3], [POSITION_NOISE_MM])
        self.frequency_hz, self.latest, self.started = None, None, False
        self.measurement_log, self.command_log = [], []

    def observe(self, position_mm, axis_unit, shutter_time_s):
        axis = np.asarray(axis_unit, float) * (1 if axis_unit[2] >= 0 else -1)
        offset = np.asarray(position_mm, float) - self.hover_point_mm
        tilt = complex(axis[0], axis[1]) / axis[2]
        self.latest = (np.array([complex(offset[0], offset[1]), tilt]),
                       np.array([complex(offset[2])]), shutter_time_s)
        self.measurement_log.append((shutter_time_s, tilt, complex(offset[0], offset[1])))

    def start(self, frequency_hz):
        self.frequency_hz = frequency_hz

    def limit_field_tilt(self, wanted):
        size = abs(wanted)
        return wanted if size <= self.field_tilt_limit_rad else wanted * self.field_tilt_limit_rad / size

    def frequency_limited_acceleration(self, wanted):
        lowest, highest = self.frequency_band_hz
        target = self.hover_frequency_hz * math.sqrt(max(1 + wanted.real / GRAVITY_MM_PER_S2, 0.0))
        target = min(max(target, lowest), highest)
        self.frequency_hz = min(max(target, self.frequency_hz - self.frequency_step_hz),
                                self.frequency_hz + self.frequency_step_hz)
        achieved = GRAVITY_MM_PER_S2 * ((self.frequency_hz / self.hover_frequency_hz) ** 2 - 1)
        return wanted if abs(achieved - wanted.real) < 1e-9 else complex(achieved)

    def command(self, now_s, field_tilt_override=None):
        if self.latest is not None:
            lateral_measured, vertical_measured, shutter_time_s = self.latest
            age_steps = max(round((now_s - shutter_time_s) / self.step_s), 0)
            if not self.started:
                self.lateral.start(lateral_measured)
                self.vertical.start(vertical_measured)
                self.started = True
            else:
                self.lateral.correct(lateral_measured, age_steps)
                self.vertical.correct(vertical_measured, age_steps)
            self.latest = None
        if not self.started:
            return None
        lateral_limit = self.limit_field_tilt if field_tilt_override is None else (
            lambda wanted: complex(field_tilt_override) * self.response.field_gain
            + self.response.field_offset_rad)
        predicted_offset = self.lateral.estimate[0]
        predicted_height = self.vertical.estimate[0].real
        field_tilt = self.lateral.advance(lateral_limit)
        self.vertical.advance(self.frequency_limited_acceleration)
        tilt_sent = (field_tilt - self.response.field_offset_rad) / self.response.field_gain
        self.command_log.append((now_s, tilt_sent))
        point = (self.coil_rotation * (predicted_offset + complex(*self.hover_point_mm[:2])),
                 predicted_height + self.hover_point_mm[2])
        return HoverCommand(self.vec_line(point, tilt_sent), self.frequency_hz, tilt_sent)

    def vec_line(self, point, tilt_sent):
        horizontal, height = point
        axis = np.array([tilt_sent.real, tilt_sent.imag, 1.0]) / math.hypot(abs(tilt_sent), 1.0)
        quaternion = np.array([1 + axis[2], -axis[1], axis[0], 0.0])
        quaternion /= np.linalg.norm(quaternion)
        x, y, z = (self.coil_offset_mm + [horizontal.real, horizontal.imag, height])
        return (f"vec={x:.2f}:{y:.2f}:{z:.2f}:" + ":".join(f"{q:.5f}" for q in quaternion)
                + f":{self.field_mT:.2f}")

    def save_log(self, path):
        measured, commanded = np.array(self.measurement_log), np.array(self.command_log)
        np.savez(path, measurement_times_s=measured[:, 0].real, measured_tilts=measured[:, 1],
                 measured_offsets_mm=measured[:, 2], command_times_s=commanded[:, 0].real,
                 field_commands=commanded[:, 1])


def tilt_step_schedule(tilt_rad=0.05, azimuths=8, hold_s=1.5):
    def field_tilt_at(elapsed_s):
        index = int(elapsed_s // hold_s)
        stepping = index % 2 == 1 and index < 2 * azimuths
        return tilt_rad * np.exp(2j * math.pi * (index // 2) / azimuths) if stepping else 0.0
    return field_tilt_at


def fit_axis_response(measurement_times_s, measured_tilts, measured_offsets_mm, command_times_s,
                      field_commands, window=15):
    step_s = float(np.median(np.diff(measurement_times_s)))
    grid = np.arange(measurement_times_s[0], measurement_times_s[-1], step_s)

    def resampled(signal, **derivative):
        return sum(savgol_filter(np.interp(grid, measurement_times_s, part), window, 2, delta=step_s,
                                 **derivative) * unit
                   for part, unit in ((np.real(signal), 1), (np.imag(signal), 1j)))

    held = np.searchsorted(command_times_s, grid, side="right") - 1
    usable = held >= 0
    regressors = np.column_stack([np.asarray(field_commands)[held[usable]],
                                  resampled(measured_tilts)[usable],
                                  resampled(measured_offsets_mm)[usable], np.ones(usable.sum())])
    command_gain, tilt_gain, offset_gain, constant = np.linalg.lstsq(
        regressors, resampled(measured_tilts, deriv=1)[usable], rcond=None)[0]
    precession = -tilt_gain
    return AxisResponse(precession, command_gain / precession, constant / precession,
                        offset_gain / precession)


def simulate(controller, true_response, true_hover_frequency_hz, latency_s, seconds,
             start_mm=(3.0, -2.0, 13.0), start_frequency_hz=None, seed=0):
    random = np.random.default_rng(seed)
    step_s = controller.step_s
    lateral_truth = discretised(*lateral_plant(true_response.precession_per_s, 1.0,
                                               true_response.axis_leak_per_mm), step_s)
    position = np.array(start_mm, float)
    velocity, tilt = np.zeros(3), 0.02 + 0.01j
    controller.start(start_frequency_hz or controller.hover_frequency_hz)
    ground_mm = start_mm[2]
    delay_steps = round(latency_s / step_s)
    past = deque(maxlen=delay_steps + 1)
    frequencies, positions = [], []
    for step in range(int(seconds / step_s)):
        now = step * step_s
        past.append((position.copy(), tilt, now))
        seen_position, seen_tilt, seen_at = past[0]
        noisy = seen_position + random.normal(0, POSITION_NOISE_MM, 3)
        axis = np.array([seen_tilt.real, seen_tilt.imag, 1.0]) + random.normal(0, TILT_NOISE_RAD, 3) * [1, 1, 0]
        controller.observe(noisy, axis / np.linalg.norm(axis), seen_at)
        command = controller.command(now)
        true_field = command.field_tilt_sent * true_response.field_gain + true_response.field_offset_rad
        lateral_state = np.array([complex(*position[:2]), complex(*velocity[:2]), tilt, 0])
        lateral_state = lateral_truth[0] @ lateral_state + lateral_truth[1] * true_field
        thrust = (command.frequency_hz / true_hover_frequency_hz) ** 2
        vertical_acceleration = GRAVITY_MM_PER_S2 * (thrust / math.hypot(1, abs(tilt)) - 1)
        position[2] += step_s * velocity[2] + 0.5 * step_s ** 2 * vertical_acceleration
        velocity[2] += step_s * vertical_acceleration
        if position[2] <= ground_mm:
            position[2], velocity[2] = ground_mm, max(velocity[2], 0.0)
        position[:2] = lateral_state[0].real, lateral_state[0].imag
        velocity[:2] = lateral_state[1].real, lateral_state[1].imag
        tilt = lateral_state[2]
        frequencies.append(command.frequency_hz)
        positions.append(position - controller.hover_point_mm)
    return np.array(frequencies), np.array(positions)


def self_check():
    random = np.random.default_rng(1)
    for trial in range(6):
        residual = random.uniform(0.75, 1.35) * np.exp(1j * math.radians(random.uniform(-18, 18)))
        truth = AxisResponse(MODELLED_PRECESSION_PER_S * residual, np.exp(1j * random.uniform(-0.3, 0.3)),
                             0.01 * random.normal(), random.uniform(-0.005, 0.005))
        believed = AxisResponse(MODELLED_PRECESSION_PER_S, truth.field_gain, truth.field_offset_rad)
        controller = AxisHoverController(believed)
        latency_s = random.uniform(0.016, 0.033)
        hover_frequency_hz = random.uniform(92, 110)
        frequencies, positions = simulate(controller, truth, hover_frequency_hz, latency_s, 30.0,
                                          start_frequency_hz=0.97 * hover_frequency_hz, seed=trial)
        offsets = np.linalg.norm(positions, axis=1)
        assert np.sqrt(np.mean(offsets[-1200:] ** 2)) < 1.0, (trial, offsets[-1200:].max())
        assert frequencies.min() >= 85 and frequencies.max() <= 115
        assert np.abs(np.diff(frequencies)).max() <= controller.frequency_step_hz + 1e-9
    truth = AxisResponse(3.1 * np.exp(-1.2j), 0.9 * np.exp(0.4j), 0.02 - 0.01j, 0.004)
    times = np.arange(0, 20, 1 / 120)
    schedule = tilt_step_schedule()
    commands = np.array([schedule(t) for t in times])
    offsets = 2 * np.sin(0.7 * times) + 1j * np.cos(0.4 * times)
    tilt, tilts = 0j, []
    for command, offset in zip(commands, offsets):
        tilts.append(tilt + random.normal(0, TILT_NOISE_RAD / 4) * (1 + 1j))
        target = command * truth.field_gain + truth.field_offset_rad + truth.axis_leak_per_mm * offset
        tilt += (1 / 120) * truth.precession_per_s * (target - tilt)
    fitted = fit_axis_response(times, np.array(tilts), offsets, times, commands)
    assert abs(fitted.precession_per_s / truth.precession_per_s - 1) < 0.1, fitted
    assert abs(fitted.field_gain / truth.field_gain - 1) < 0.1, fitted
    assert abs(fitted.field_offset_rad - truth.field_offset_rad) < 0.005, fitted
    assert abs(fitted.axis_leak_per_mm - truth.axis_leak_per_mm) < 0.001, fitted
    try:
        AxisHoverController(hover_point_mm=(0, 0, 21.0))
    except ValueError:
        pass
    else:
        raise AssertionError("a hover point on the singular shell was accepted")
    print("axis_hover self-check ok: 6 random plants hover, band and slew held, fit recovers response")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "fit":
        log = np.load(sys.argv[2])
        response = fit_axis_response(log["measurement_times_s"], log["measured_tilts"],
                                     log["measured_offsets_mm"], log["command_times_s"],
                                     log["field_commands"])
        response.save(sys.argv[3])
        print(response)
    else:
        self_check()
