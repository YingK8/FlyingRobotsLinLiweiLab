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


def read_code(trace, times, min_contrast=0.5):
    """The 7 word slots after the first rise that frames a clean word, else None.

    A rise frames a word only if its start slot is lit, the slot after the word is dark,
    and every lit slot centre clears every dark one by `min_contrast` of the trace's
    range. Thresholding any varying trace at its midpoint and taking the first rise
    read a word out of every flickering region: on the 2026-11-06 bench, 7 Test LED
    presses put the "LED" in 7 different places, B's on bright background (mean 196
    with the LED off), which then failed the run's dark-before-START check."""
    trace, times = np.asarray(trace, float), np.asarray(times, float)
    lo, hi = trace.min(), trace.max()
    if hi <= lo:
        return None
    mid = (lo + hi) / 2
    level = trace > mid
    for rise in np.flatnonzero(~level[:-1] & level[1:]) + 1:
        t_start = (times[rise - 1] + times[rise]) / 2
        centres = t_start + CODE_BIT_S * (np.arange(9) + 0.5)     # start, 7 bits, dark
        if centres[-1] > times[-1]:
            return None
        v = trace[[np.argmin(np.abs(times - t)) for t in centres]]
        lit = v > mid
        if not lit[0] or lit[-1] or v[lit].min() - v[~lit].max() < min_contrast * (hi - lo):
            continue
        return [int(b) for b in lit[1:8]]
    return None


def locate_led(frames, times, pad_px=6, thresh=40, max_regions=200):
    """The LED as ``(cx, cy, r)``: the first changing region, largest first, whose
    brightness decodes the word to `LED_ID`; None if none does.

    Largest-only failed on the 2026-11-05 bench: camera A's biggest changing region
    during the word was something else, and the LED was never tried. The decode is
    the identity test -- framed, high-contrast and uncorrected (`read_code`) -- so
    trying the next regions does not mistake a distractor for it.
    """
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    counts, changes = blink_mask(frames, thresh)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((counts > 0).astype(np.uint8))
    stack, changed = np.stack(frames), np.any(changes, axis=0)
    # 8 regions was too few: with auto-exposure the large changing regions are scene
    # patches, and a small, clear LED was never tried (2026-11-06, "very clear" and not
    # found). The framed exact word makes 200 safe: noise frames a word 0.25% of the
    # time and is then the exact word 1 in 128, so ~0.4% per LED-free scene.
    for label in (1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]))[:max_regions]:
        x, y, w, h = stats[label, :4]
        region = labels[y:y + h, x:x + w] == label
        bits = read_code(stack[:, y:y + h, x:x + w][:, region].mean(axis=1), times)
        # Exact word only: correcting one bit accepts 8 of 128 words, and with 8
        # regions tried a flickering scene passed ~40% of the time.
        if bits is None or hamming_decode(bits) != (LED_ID, None):
            continue
        lit = (changed[y:y + h, x:x + w] & region).astype(np.uint8)
        (cx, cy), r = cv2.minEnclosingCircle(cv2.findNonZero(lit))
        return cx + x, cy + y, r + pad_px
    return None


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
        if np.random.default_rng(len(frames)).random() < .5:   # a bigger flicker that is not ours
            cv2.rectangle(frame, (60, 60), (160, 160), 230, -1)
        frames.append(frame)
    cx, cy, r = locate_led(frames, times)
    assert np.hypot(cx - 500, cy - 300) < 1.5 and 8 < r < 20
    # A small LED behind 12 larger flickering patches (auto-exposure-like) is still found.
    crowd = [f.copy() for f in frames]
    for j, f in enumerate(crowd):
        for q in range(12):
            if np.random.default_rng(1000 * j + q).random() < .5:
                cv2.rectangle(f, (20 + 50 * q, 200), (55 + 50 * q, 240), 230, -1)
    cx, cy, r = locate_led(crowd, times)
    assert np.hypot(cx - 500, cy - 300) < 1.5, (cx, cy)
    # No LED, only flicker: nothing. Then a drifting, noisy trace is not a word.
    dark = [f.copy() for f in frames]
    for f in dark:
        cv2.circle(f, (500, 300), 12, 20, -1)
    assert locate_led(dark, times) is None
    rng = np.random.default_rng(1)
    misses = sum(read_code(rng.normal(size=len(times)), times) is not None for _ in range(500))
    assert misses < 5, f"noise read as a framed word {misses}/500 times"
    print("scene: ok")


if __name__ == "__main__":
    _self_check()
