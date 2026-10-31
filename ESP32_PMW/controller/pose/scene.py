#!/usr/bin/env python3
# Splits a frame into rotor ellipse, plumb line and sync LED; the plumb pair gives gravity.

from __future__ import annotations

import cv2
import numpy as np

from controller.pose import segment as segmod
from controller.pose.disc_pose import mast_direction

LED_ID = 0b1011
LED_WORD = 0b0110011        # hamming_encode(LED_ID); src/main_flight.cpp sends this literal
CODE_BIT_S = 0.080          # one slot; far from the 30 ms START pulses decode_markers looks for

_PARITY = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])


def binary_mask(gray):
    return cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def find_vertical_line(mask, max_tilt_deg=10.0, min_length_px=40):
    segments = cv2.HoughLinesP(cv2.Canny(mask * 255, 50, 150), 1, np.pi / 360, 30,
                               minLineLength=min_length_px, maxLineGap=10)
    if segments is None:
        return None
    segments = segments.reshape(-1, 4).astype(float)
    dx, dy = segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]
    tilt = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    vertical = segments[tilt <= max_tilt_deg]
    if not len(vertical):
        return None
    longest = vertical[np.argmax(np.hypot(*(vertical[:, 2:] - vertical[:, :2]).T))]
    p0, direction = longest[:2], (longest[2:] - longest[:2]) / np.linalg.norm(longest[2:] - longest[:2])
    normal = np.array([-direction[1], direction[0]])
    ends = vertical.reshape(-1, 2)
    on_line = np.abs((ends - p0) @ normal) <= 3.0
    collinear = ends[np.repeat(on_line.reshape(-1, 2).all(1), 2)]
    along = (collinear - p0) @ direction
    return p0 + along.min() * direction, p0 + along.max() * direction


def line_band(shape, line, pad_px):
    band = np.zeros(shape, np.uint8)
    p0, p1 = (tuple(int(round(v)) for v in p) for p in line)
    return cv2.line(band, p0, p1, 1, 2 * pad_px + 1)


def circle_mask(shape, circle):
    mask = np.zeros(shape, np.uint8)
    if circle is not None:
        cx, cy, r = circle
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(np.ceil(r)), 1, -1)
    return mask


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return np.zeros_like(mask)
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def segment_scene(gray, led_circle=None, pad_px=4):
    mask = binary_mask(gray)
    line = find_vertical_line(mask)
    band = line_band(mask.shape, line, pad_px) if line is not None else np.zeros_like(mask)
    led_mask = circle_mask(mask.shape, led_circle)
    line_mask = mask & band & (1 - led_mask)
    ellipse_mask = largest_component(mask & (1 - band) & (1 - led_mask))
    return ellipse_mask, line_mask, led_mask


def fit_ellipse(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    return segmod.fit_ellipse(outline) if len(outline) >= 6 else None


def fit_line(mask, cam=None):
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    pixels = np.c_[xs, ys].astype(np.float64)
    if cam is not None and cam.dist is not None:
        pixels = cv2.undistortPoints(pixels.reshape(-1, 1, 2), cam.K, cam.dist, P=cam.K).reshape(-1, 2)
    dx, dy, x0, y0 = cv2.fitLine(pixels.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    centroid, direction = np.array([x0, y0], float), np.array([dx, dy], float)
    offsets = pixels - centroid
    across, along = offsets @ [-direction[1], direction[0]], offsets @ direction
    length = max(np.ptp(along), 1.0)
    sigma_rad = float(np.sqrt(np.mean(across ** 2)) * np.sqrt(12.0 / len(xs)) / length)
    return centroid, direction, max(sigma_rad, 1e-6)


def gravity(pairs, cams, pad_px=4):
    directions, weights = [], []
    image_down = cams[0].R @ [0.0, 1.0, 0.0]
    for frames in pairs:
        fits = [fit_line(segment_scene(gray, pad_px=pad_px)[1], cam) for gray, cam in zip(frames, cams)]
        if None in fits:
            continue
        g = mast_direction([fit[:2] for fit in fits], cams, up=image_down)
        if g is not None:
            directions.append(g)
            weights.append(1.0 / sum(fit[2] ** 2 for fit in fits))
    if not directions:
        return None
    directions, weights = np.array(directions), np.array(weights)
    g = weights @ directions
    g /= np.linalg.norm(g)
    spread_deg = float(np.degrees(np.sqrt(weights @ np.arccos(np.clip(directions @ g, -1, 1)) ** 2 / weights.sum())))
    return g, spread_deg, len(directions)


def hamming_encode(nibble):
    d1, d2, d3, d4 = ((nibble >> shift) & 1 for shift in (3, 2, 1, 0))
    bits = [d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d1, d2 ^ d3 ^ d4, d2, d3, d4]
    return int("".join(map(str, bits)), 2)


def hamming_decode(bits):
    bits = np.array(bits, int)
    error_at = int(_PARITY @ bits % 2 @ [1, 2, 4])
    if error_at:
        bits[error_at - 1] ^= 1
    return int("".join(map(str, bits[[2, 4, 5, 6]])), 2), (error_at - 1 if error_at else None)


def blink_mask(frames, thresh=40):
    changes = [cv2.absdiff(a, b) > thresh for a, b in zip(frames[:-1], frames[1:])]
    return np.sum(changes, axis=0).astype(np.int32), changes


def read_code(trace, times):
    trace, times = np.asarray(trace, float), np.asarray(times, float)
    level = trace > (trace.min() + trace.max()) / 2
    rises = np.flatnonzero(~level[:-1] & level[1:]) + 1
    if not len(rises):
        return None
    t_start = (times[rises[0] - 1] + times[rises[0]]) / 2
    slot_centres = t_start + CODE_BIT_S * (np.arange(1, 8) + 0.5)
    if slot_centres[-1] > times[-1]:
        return None
    return [int(level[np.argmin(np.abs(times - t))]) for t in slot_centres]


def locate_led(frames, times, pad_px=6, thresh=40):
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    counts, changes = blink_mask(frames, thresh)
    region = largest_component((counts > 0).astype(np.uint8))
    if not region.any():
        return None
    trace = [float(f[region > 0].mean()) for f in frames]
    bits = read_code(trace, times)
    if bits is None or hamming_decode(bits)[0] != LED_ID:
        return None
    lit = (np.any(changes, axis=0) & (region > 0)).astype(np.uint8)
    (cx, cy), r = cv2.minEnclosingCircle(cv2.findNonZero(lit))
    return cx, cy, r + pad_px


def _self_check():
    from controller.pose.disc_axis import _synthetic_rig

    for nibble in range(16):
        word = [int(b) for b in f"{hamming_encode(nibble):07b}"]
        assert hamming_decode(word) == (nibble, None)
        for i in range(7):
            flipped = list(word)
            flipped[i] ^= 1
            assert hamming_decode(flipped) == (nibble, i)
    assert hamming_encode(LED_ID) == LED_WORD

    scene = np.full((400, 640), 20, np.uint8)
    cv2.line(scene, (100, 20), (130, 380), 200, 3)
    cv2.ellipse(scene, ((380, 220), (160, 80), 30), 200, -1)
    cv2.circle(scene, (560, 60), 10, 255, -1)
    ellipse_mask, line_mask, led_mask = segment_scene(scene, led_circle=(560, 60, 14))
    assert not (ellipse_mask & line_mask).any() and not (ellipse_mask & led_mask).any()
    _, direction, _ = fit_line(line_mask)
    assert abs(np.degrees(np.arctan2(abs(direction[0]), abs(direction[1]))) - np.degrees(np.arctan2(30, 360))) < 0.5
    (centre, _, _), _ = fit_ellipse(ellipse_mask)
    assert np.hypot(centre[0] - 380, centre[1] - 220) < 1.0

    rig = _synthetic_rig()
    cams = rig.cameras
    true_g = np.array([0.05, 1.0, 0.03]) / np.linalg.norm([0.05, 1.0, 0.03])
    string = np.array([-10.0, 0.0, 5.0]) + np.outer(np.linspace(-15, 15, 2), true_g)
    views = []
    for cam in cams:
        ends, _ = cv2.projectPoints((string - cam.position) @ cam.R, np.zeros(3), np.zeros(3), cam.K, None)
        img = np.full((400, 640), 20, np.uint8)
        cv2.line(img, *(tuple(int(round(v)) for v in p) for p in ends.reshape(-1, 2)), 200, 3)
        views.append(img)
    g, spread, n = gravity([views] * 3, cams)
    assert n == 3 and spread < 1e-3 and g @ (cams[0].R @ [0, 1, 0]) > 0
    assert np.degrees(np.arccos(abs(g @ true_g))) < 0.5

    fps = 120.0
    times = np.arange(int(1.2 * fps)) / fps
    bits = [1] + [int(b) for b in f"{LED_WORD:07b}"] + [0]
    t0 = 0.2
    frames = []
    for t in times:
        slot = int((t - t0) // CODE_BIT_S)
        on = 0 <= slot < len(bits) and bits[slot]
        frame = np.full((400, 640), 20, np.uint8)
        if on:
            cv2.circle(frame, (500, 300), 8, 250, -1)
        frames.append(frame)
    cx, cy, r = locate_led(frames, times)
    assert np.hypot(cx - 500, cy - 300) < 1.5 and 8 < r < 20
    print("scene: ok")


if __name__ == "__main__":
    _self_check()
