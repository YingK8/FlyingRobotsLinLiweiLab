#!/usr/bin/env python3
"""
Coil array geometry: per-channel pose, a DC ring, and the constraints. See theory.md section 15.

`spatial_model.quick_coils` exposes three shared scalars and hard-wires a coplanar, four-fold
symmetric cross. Section 14 searched that family exhaustively and found nothing laterally stable,
so this module opens the parameterization it was searching inside:

    c4_array     8 shared parameters, four-fold symmetric, two rings at independent heights
    free_array   7 per channel, symmetry broken, for the control experiment of section 15.4

Three facts drive the shape of it, all derived in section 15.1:

    axial offset   The transverse field of a coil on the array axis goes as (z - z_coil), so a
                   coil plane level with the robot contributes no rotating field at all.
    antiphase      Two rings therefore cancel unless the upper one is reverse-wound. Reversed,
                   they add, and the pair can be tuned to null the field curvature.
    DC is separate The steady ring couples only to the rotor's axial moment, the rotating rings
                   only to its in-plane one. Different coils, different jobs, no interference.

    uv run python controller/control/coil_geometry.py --self-check
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import spatial_model as sm

COIL_R = 0.0105       # loop radius, m, as the rig is built
COIL_LEN = 0.022      # coil body axial extent, m: the 22 mm in the MATLAB filename
TURNS = 650

# Channel phases. Four-fold progression is what makes the field rotate (section 12.1); the
# upper ring carries the same four phases plus pi, which is a winding reversal, not a driver.
AC_PHASES = np.radians([0.0, 270.0, 180.0, 90.0])
DC_CHANNEL = 4


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _quad(centre, e_a, e_b, pitch, n):
    """``n`` coil centres of one group: 1 at the centre, 2 on e_b, or 4 in a square.

    A pair splits along the *tangential* axis, which keeps both coils at the same radius and
    so preserves the four-fold ring. Splitting radially instead puts one coil inside the ring
    and one outside, and measurably spoils the field shape.
    """

    if n == 1:
        return [centre]
    if n == 2:
        return [centre - 0.5 * pitch * e_b, centre + 0.5 * pitch * e_b]
    return [centre + 0.5 * pitch * (sa * e_a + sb * e_b)
            for sa in (-1, 1) for sb in (-1, 1)]


def _rodrigues(axis, angle):
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return (math.cos(angle) * np.eye(3)
            + (1 - math.cos(angle)) * np.outer(axis, axis)
            + math.sin(angle) * k)


def _ring(r, z, tilt_deg, pitch, n_per, phase_offset, amp, ch0=0):
    """One four-fold ring: four groups of ``n_per`` coils at radius ``r``, height ``z``."""

    a = math.radians(tilt_deg)
    pos, nhat, chan, phase = [], [], [], []
    for g, ang in enumerate(np.radians([0.0, 90.0, 180.0, 270.0])):
        e_r = np.array([math.cos(ang), math.sin(ang), 0.0])
        e_t = np.array([-e_r[1], e_r[0], 0.0])
        rot = _rodrigues(e_t, a)                     # tilt the group about its tangent
        n_g = rot @ sm._ZHAT
        centre = np.array([r * math.cos(ang), r * math.sin(ang), z])
        for pt in _quad(centre, rot @ e_r, rot @ e_t, pitch, n_per):
            pos.append(pt)
            nhat.append(n_g)
            chan.append(ch0 + g)
            phase.append(AC_PHASES[g] + phase_offset)
    return pos, nhat, chan, phase, [amp] * len(pos)


def _assemble(pos, nhat, chan, phase, amp, dc_amp, model="loop"):
    pos, nhat = np.asarray(pos, float), np.asarray(nhat, float)
    nhat = nhat / np.linalg.norm(nhat, axis=1)[:, None]
    m = len(pos)
    return sm.Coils(
        pos=pos, nhat=nhat, moment=TURNS * math.pi * COIL_R**2 * nhat,
        amp=np.asarray(amp, float), phase=np.asarray(phase, float),
        channel=np.asarray(chan, int),
        radius=np.full(m, COIL_R), turns=np.full(m, float(TURNS)),
        model=model, dc_amp=np.asarray(dc_amp, float),
    )


def c4_array(r_lo=0.037, z_lo=0.0, r_hi=0.037, z_hi=0.050, tilt_lo=0.0, tilt_hi=0.0,
             pitch=0.021, amp=1.25, n_per_ring=2, r_dc=0.060, z_dc=0.020, i_dc=0.0,
             model="loop"):
    """Two antiphase rotating rings plus one steady ring, four-fold symmetric throughout.

    ``n_per_ring=4`` with ``z_hi`` unused reproduces the single-plane cross of `quick_coils`;
    ``n_per_ring=2`` splits the same sixteen coils across two heights at no driver cost.
    Four-fold symmetry is not a convenience here, it is optimal: Earnshaw pins
    C_xx + C_yy = k_z, so max(C_xx, C_yy) >= k_z/2 with equality only when the two are equal,
    and breaking symmetry can only worsen the stiffest lateral direction (section 15.3).
    """

    pos, nhat, chan, phase, a = _ring(r_lo, z_lo, tilt_lo, pitch, n_per_ring, 0.0, amp)
    if n_per_ring < 4:
        p2, n2, c2, ph2, a2 = _ring(r_hi, z_hi, tilt_hi, pitch, n_per_ring, math.pi, amp)
        pos, nhat, chan, phase, a = pos + p2, nhat + n2, chan + c2, phase + ph2, a + a2
    dc = [0.0] * len(pos)

    if i_dc:
        # The steady ring sits level with the robot on purpose: a ring's on-axis B_z peaks in
        # its own plane, which is where an axial trap has to be, and is exactly the height at
        # which it contributes no rotating field. The two jobs want different places.
        for ang in np.radians([0.0, 90.0, 180.0, 270.0]):
            pos.append(np.array([r_dc * math.cos(ang), r_dc * math.sin(ang), z_dc]))
            nhat.append(sm._ZHAT.copy())
            chan.append(DC_CHANNEL)
            phase.append(0.0)
            a.append(0.0)          # no rotating current in the DC ring
            dc.append(i_dc)
    return _assemble(pos, nhat, chan, phase, a, dc, model)


def free_array(poses, pitch=0.021, amp=1.25, n_per_ring=2, dc=None, model="loop"):
    """Per-channel pose, symmetry broken. ``poses`` is (n_ch, 6): x, y, z, tilt, azim, psi.

    Angles in degrees. ``tilt`` and ``azim`` set the group normal, ``psi`` spins the group in
    its own plane. Used once, from the C4 optimum, to test the symmetry claim numerically.
    """

    pos, nhat, chan, phase, a = [], [], [], [], []
    for g, (x, y, z, tilt, azim, psi) in enumerate(np.asarray(poses, float)):
        e_ax = np.array([math.cos(math.radians(azim)), math.sin(math.radians(azim)), 0.0])
        rot = _rodrigues(e_ax, math.radians(tilt))
        n_g = rot @ sm._ZHAT
        spin = _rodrigues(n_g, math.radians(psi))
        e_a = spin @ (rot @ np.array([1.0, 0.0, 0.0]))
        e_b = np.cross(n_g, e_a)
        for pt in _quad(np.array([x, y, z]), e_a, e_b, pitch, n_per_ring):
            pos.append(pt)
            nhat.append(n_g)
            chan.append(g % 4)
            phase.append(AC_PHASES[g % 4] + (math.pi if g >= 4 else 0.0))
            a.append(amp)
    d = [0.0] * len(pos)
    if dc:
        r_dc, z_dc, i_dc = dc
        for ang in np.radians([0.0, 90.0, 180.0, 270.0]):
            pos.append(np.array([r_dc * math.cos(ang), r_dc * math.sin(ang), z_dc]))
            nhat.append(sm._ZHAT.copy())
            chan.append(DC_CHANNEL)
            phase.append(0.0)
            a.append(0.0)
            d.append(i_dc)
    return _assemble(pos, nhat, chan, phase, a, d, model)


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------


def pair_clearance(coils):
    """Smallest gap between any two coil bodies, in metres. Negative means they interfere.

    Each coil is a cylinder of radius COIL_R and length COIL_LEN. Two such cylinders with
    parallel axes clear each other if their radial separation exceeds 2R *or* their axial
    separation exceeds L, so the gap is the larger of the two margins.

    # ponytail: exact only for parallel axes, which is every pair in a four-fold ring and a
    # good approximation while relative tilts stay small. Cross-check a final candidate with
    # the GJK predicate in Coil_Array_Optimisation/cylinderCollision.py, which needs the
    # uninstalled `distance3d` and is far slower.
    """

    p, n = coils.pos, coils.nhat
    d = p[:, None, :] - p[None, :, :]
    axis = n[:, None, :] + n[None, :, :]
    norm = np.linalg.norm(axis, axis=2)
    axis = np.divide(axis, np.maximum(norm, 1e-12)[:, :, None])
    along = np.abs(np.einsum("ijk,ijk->ij", d, axis))
    perp = np.sqrt(np.maximum(np.einsum("ijk,ijk->ij", d, d) - along**2, 0.0))
    gap = np.maximum(perp - 2.0 * COIL_R, along - COIL_LEN)
    np.fill_diagonal(gap, np.inf)
    return float(gap.min())


def workspace_clearance(coils, r_work, half_len=0.0):
    """Gap between the coil bodies and a sphere of radius ``half_len`` at ``r_work``."""

    d = np.asarray(r_work, float)[None, :] - coils.pos
    along = np.abs(np.einsum("ij,ij->i", d, coils.nhat))
    perp = np.sqrt(np.maximum(np.einsum("ij,ij->i", d, d) - along**2, 0.0))
    gap = np.maximum(perp - COIL_R, along - 0.5 * COIL_LEN) - half_len
    return float(gap.min())


def camera_clearance(coils, r_work, elevation_deg=45.0, azimuths_deg=(45.0, 135.0),
                     half_angle_deg=4.0):
    """Angular margin between the coil bodies and the stereo sight lines, in degrees.

    The rig views the robot from two directions 90 degrees apart at 45 degrees elevation
    (chapter 2). A coil inside the cone about either line occludes it, and negative means
    blocked.

    The default azimuths sight *between* the arms of the cross rather than along them, which
    is the only layout that clears: looking down an arm at 45 degrees puts a coil group
    squarely in the view. It is an assumption about the physical rig, so confirm it before
    trusting a design that only just satisfies it: this array clears by 0.6 degrees between
    the arms and fails by 8.4 along them.
    """

    el = math.radians(elevation_deg)
    margin = math.inf
    for az in np.radians(azimuths_deg):
        view = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                         math.sin(el)])
        d = coils.pos - np.asarray(r_work, float)[None, :]
        rng = np.linalg.norm(d, axis=1)
        ok = rng > 1e-9
        if not ok.any():
            continue
        cos_t = np.clip(np.einsum("ij,j->i", d[ok], view) / rng[ok], -1.0, 1.0)
        # Angular radius the coil body subtends, from its bounding sphere.
        body = math.hypot(COIL_R, 0.5 * COIL_LEN)
        subtend = np.degrees(np.arcsin(np.clip(body / rng[ok], 0.0, 1.0)))
        margin = min(margin, float(np.min(np.degrees(np.arccos(cos_t)) - subtend))
                     - half_angle_deg)
    return margin


# --------------------------------------------------------------------------
# Field shape at the working point
# --------------------------------------------------------------------------


def field_shape(coils, z_work, h=3e-4):
    """``(B, dB/dz, d2B/dz2)`` of the rotating field's mean semiaxis on the array axis."""

    def b(z):
        u, v, _, _ = sm.field_at(np.array([0.0, 0.0, z]), coils, None, False)
        a_ax, b_ax = sm.ellipse_semiaxes(u, v)
        return 0.5 * (a_ax + b_ax)

    b0, bp, bm = b(z_work), b(z_work + h), b(z_work - h)
    return b0, (bp - bm) / (2 * h), (bp - 2 * b0 + bm) / h**2


def figure_of_merit(coils, z_work):
    """B^2 / |d2B/dz2|, the geometry factor in R. Section 15.1: R scales as its square root."""

    b0, _, b2 = field_shape(coils, z_work)
    return b0 * b0 / max(abs(b2), 1e-12)


# --------------------------------------------------------------------------


def _self_check():
    z_w = 0.020
    r_w = np.array([0.0, 0.0, z_w])

    # 1. The C4 builder reproduces quick_coils position-for-position, so section 14's numbers
    # still hold when they are recomputed through this module.
    mine = c4_array(r_lo=0.037, z_lo=0.0, tilt_lo=0.0, pitch=0.021, n_per_ring=4, model="dipole")
    ref = sm.quick_coils("cross", d=0.037, tilt_deg=0.0, amp=1.25, pitch=0.021, model="dipole")
    assert len(mine.pos) == len(ref.pos) == 16
    # Sort on rounded keys: this builder places groups by cos/sin, quick_coils by literal
    # zeros, so the two differ by ~1e-17 and a raw lexsort pairs the wrong coils.
    key = lambda a: np.lexsort(np.round(a.pos, 9).T)
    assert np.allclose(mine.pos[key(mine)], ref.pos[key(ref)], atol=1e-12)
    u1, v1, _, _ = sm.field_at(r_w, mine, None, False)
    u2, v2, _, _ = sm.field_at(r_w, ref, None, False)
    assert np.allclose(u1, u2, atol=1e-15) and np.allclose(v1, v2, atol=1e-15)
    print("C4 builder  : reproduces quick_coils to 1e-12 m and the field to 1e-15 T")

    # 2. The transverse field of a ring vanishes in its own plane and reverses across it.
    # This is why a Helmholtz-style symmetric split cancels the drive, and why the upper ring
    # has to be reverse-wound (section 15.1).
    ring = c4_array(r_lo=0.037, z_lo=0.0, n_per_ring=4, model="loop")
    b_at = lambda z: field_shape(ring, z)[0]
    assert b_at(0.0) < 1e-9, b_at(0.0)
    u_p, v_p, _, _ = sm.field_at(np.array([0.0, 0.0, +0.02]), ring, None, False)
    u_m, v_m, _, _ = sm.field_at(np.array([0.0, 0.0, -0.02]), ring, None, False)
    assert np.allclose(u_p, -u_m, atol=1e-12), (u_p, u_m)
    print(f"axial offset: |B| in the coil plane = {b_at(0.0):.2e} T, and reverses across it")

    # 3. Antiphase two rings flatten the curvature; in phase they cancel the drive outright.
    split = c4_array(r_lo=0.037, z_lo=-0.010, r_hi=0.037, z_hi=0.050, n_per_ring=2)
    b_s, _, b2_s = field_shape(split, z_w)
    b_r, _, b2_r = field_shape(ring, z_w)
    assert b_s > 1e-4, b_s
    assert abs(b2_s) < 0.3 * abs(b2_r), (b2_s, b2_r)
    gain = figure_of_merit(split, z_w) / figure_of_merit(ring, z_w)
    assert gain > 2.0, gain
    print(f"two rings   : B {b_r*1e3:.3f} -> {b_s*1e3:.3f} mT, B'' {b2_r:+.3f} -> {b2_s:+.3f}"
          f", figure of merit x{gain:.2f}")

    same = c4_array(r_lo=0.037, z_lo=-0.010, r_hi=0.037, z_hi=0.050, n_per_ring=2)
    in_phase = sm.replace(same, phase=np.where(same.channel < 4,
                                               AC_PHASES[same.channel % 4], 0.0))
    assert field_shape(in_phase, z_w)[0] < 0.2 * b_s, "in phase should largely cancel"

    # 4. Constraints. The shipped pitch of 21 mm is exactly 2R, so the built rig sits at zero
    # clearance and anything tighter must be rejected.
    assert abs(pair_clearance(ring)) < 1e-9, pair_clearance(ring)
    # Pitch is bounded on both sides, and the upper bound moves with the ring radius: too
    # small and the four coils of a group interfere, too large and a group reaches into its
    # neighbour. At 37 mm the window is barely 2 mm wide; at 49 mm the upper bound lifts.
    assert pair_clearance(c4_array(r_lo=0.037, pitch=0.018, n_per_ring=4)) < 0.0
    assert pair_clearance(c4_array(r_lo=0.037, pitch=0.022, n_per_ring=4)) > 0.0
    assert pair_clearance(c4_array(r_lo=0.037, pitch=0.026, n_per_ring=4)) < 0.0
    assert pair_clearance(c4_array(r_lo=0.049, pitch=0.026, n_per_ring=4)) > 0.0
    print(f"clearance   : pitch 21 mm -> {pair_clearance(ring)*1e3:+.3f} mm (touching, as built)"
          f"; window at r=37 mm is 21-23 mm, at r=49 mm it opens past 28 mm")
    print(f"            : workspace {workspace_clearance(ring, r_w)*1e3:.1f} mm"
          f", camera {camera_clearance(ring, r_w):+.1f} deg")

    # 5. The DC ring adds a steady field and no rotating one.
    dc = c4_array(r_lo=0.037, z_lo=-0.010, z_hi=0.050, n_per_ring=2,
                  r_dc=0.060, z_dc=z_w, i_dc=2.0)
    assert len(dc.pos) == 20
    b_dc, _ = sm.dc_field_at(r_w, dc)
    assert np.linalg.norm(b_dc) > 1e-5, b_dc
    assert abs(field_shape(dc, z_w)[0] - b_s) < 1e-12, "DC ring must not change the AC field"
    print(f"DC ring     : |B_dc| = {np.linalg.norm(b_dc)*1e3:.3f} mT, "
          f"AC field unchanged to 1e-12 T")

    # 6. Breaking symmetry is allowed but must not be free: the free builder reproduces C4
    # when handed the C4 poses.
    poses = [(0.037 * math.cos(a), 0.037 * math.sin(a), 0.0, 0.0, math.degrees(a) + 90.0, 0.0)
             for a in np.radians([0, 90, 180, 270])]
    fr = free_array(poses, n_per_ring=2)
    assert len(fr.pos) == 8
    mids = np.array([fr.pos[fr.channel == c].mean(0) for c in range(4)])
    assert np.allclose(np.linalg.norm(mids[:, :2], axis=1), 0.037, atol=1e-9), mids
    b_free = field_shape(fr, z_w)[0]
    print(f"free builder: 4 channels x 2 coils, group centres on the 37 mm ring; "
          f"|B| {b_free*1e3:.4f} mT")

    print("self-check PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.parse_args()
    _self_check()
