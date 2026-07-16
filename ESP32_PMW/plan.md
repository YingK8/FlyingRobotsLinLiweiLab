# Deep-research session: coil coupling vs. tilt — root-cause + sensorless stabilization

## Context

The 4-coil rotating-field platform spins a passive magnetic disk suspended above the
coil array. Whenever a per-channel current-balancing compensation is enabled (feedforward
trims or the closed-loop `CoilBalancer`/`main_current_pid` PI schemes), the disk still
tilts uncontrollably during both CW and CCW spin-up and the run fails — even though CW
rotation should, if anything, be adding lift/tension on the suspension rather than
tipping it. The working hypothesis has been "magnetic coupling between coils," but the
user is explicitly skeptical: measured mutual coupling (k≈0.24 between A–C/B–D) seems too
small relative to per-coil inductance to explain a tilt this severe, an earlier/different
rig configuration ran mostly fine without this compensation, and moving both driver
boards onto one shared power supply (to rule out cross-supply coupling) did not fix it
either. The user wants a skeptical, quantitative teardown of every candidate mechanism
(not just coupling), a properly written-up hardware/experiment record for future
reference, and a design for stabilizing the spin to be flat using **only quantities
derivable from the existing per-coil current-sense (CS) signal** — no vision/outer control
loop.

Critically, prior on-disk experiments (`data/README.md`, session
`2026-07-04f_direction-disk-diagnosis/`) already found: (1) the per-coil ramp peak
staggering *does* rearrange with rotation direction (consistent with reactive coupling,
which is odd in Δφ), but (2) **the disk's tilt itself stays fixed in the lab frame
regardless of CW/CCW** — which a direction-flipping magnetic-coupling term cannot produce
— and (3) tilt is identical with the disk removed entirely (disk is "electrically
invisible" in CS), pointing at a **static geometric field asymmetry** (coil
mounting height/tilt across the B/C vs A/D pair) as the leading suspect, a thread that was
opened but never closed (caliper check "pending"). This existing evidence already cuts
against pure magnetic-coupling-as-tilt-cause and should anchor the research rather than
be re-derived from scratch.

## Why a remote/cloud session

Per the user's explicit choice, this will run as a **RemoteTrigger cloud agent session**
(not a local interactive turn) — a self-contained, long-running research task that
produces its own commits, so it doesn't block or share context with this terminal
session. Findings and docs get pushed to the repo's `origin`
(`https://github.com/YingK8/FlyingRobotsLinLiweiLab.git`) so they're reviewable online.

## What I already gathered (to hand to the remote agent so it doesn't re-derive it)

- **Coupling measurement on record:** `data/README.md` session `2026-07-04a`: solo vs.
  pairwise vs. ALL-driven sweeps (`tools/coupling_matrix.py`, `tools/fit_mutual_inductance.py`)
  found strong pairs A–C and B–D, **k≈0.24**, confirmed magnetic (not ground) since
  single-supply vs. two-supply gave matched currents within 1% (`2026-07-04c`).
- **Coil inductance:** no firmware-side figure of 1.4 mH exists in the repo; the closest
  documented values are in `calculations/resonance_calculation*.py` (coil/load L ≈
  6.64–6.7 mH, tank inductors 80 mH). `tools/fit_rlc_model.py` can produce a fitted
  per-channel R/L/C from real sweep data but its output (`tools/rlc_fit.json`) isn't
  currently on disk — the remote agent should regenerate it if scope captures allow,
  since the actual per-coil L is load-bearing for every coupling-magnitude calculation.
- **CS chain caveat:** VNH5019 `CS` pin → 1 kΩ load resistor → series resistor (R4
  unconfirmed value) → ADC. This is a scaled proxy of the *bridge's own conducting-leg*
  current, not true coil current, and reads ~0 on an *undriven* neighbor coil — so
  coupling from an undriven coil's perspective is invisible to CS and had to be measured
  with a PicoScope probe on that coil's own terminals instead (`fit_mutual_inductance.py`).
- **Known hardware confounders that can masquerade as "coupling":** GND1/PGND1 split
  ground (~400 mV offset under load, [[pwm-amp-pcb-inverter-issue]]) — **user confirms
  the star-ground fix (single-point GND1↔PGND1 jumper) is already applied on the
  hardware**, so this should be treated as resolved and only spot-checked (e.g., confirm
  CS traces don't show switching-correlated bounce), not re-opened as a live suspect;
  per-channel CS gain mismatch (`SENS[4] = {15.26, 15.28, 15.57, 15.34}` A/V); a prior
  finding that "CS→ADC taps were disconnected" during one session (`2026-07-04b`,
  `comp_ab_162530`), which — if it silently recurred — would explain compensation
  "working" on nothing.
- **Direction convention drift:** `main_current_pid.cpp`/`JsonPhaseSequencer` use
  `PHASES_CW={270,90,180,0}` / `PHASES_CCW={90,270,180,0}` (A,B,C,D); this differs from
  the README's older stated `CCW: A→C→B→D` — worth reconciling so "CW vs CCW" claims in
  the new report are unambiguous.
- **Recent local run artifacts** (repo root, today's date): `current_pid_iter1..16*.log`
  and matching `*_pid_telemetry.png`, `tilt_cw_run.csv` / `tilt_cw_run_rms.png` — these are the most
  recent compensation attempts referenced by the user ("current compensation ... still
  caused the drone to tilt uncontrollably") and should be the first data mined, before
  any new capture is planned.
- **Physical setup gap:** no coil spacing/cluster-diameter/mounting-geometry numbers
  exist anywhere in the repo (`hw_references/`, `docs/`, `devices/` are schematic/BOM
  only) — this must be captured as real-world measurements (or explicitly flagged as
  missing) in the new write-up, since geometric asymmetry is the leading alternative
  hypothesis.

## Remote session scope of work

Launch via `RemoteTrigger` (`action: "create"` then `action: "run"`, targeting this repo)
with a prompt instructing the agent to, end-to-end:

1. **Mine existing data first.** Read the current-branch `current_pid_iter*.log/png`,
   `tilt_cw_run.csv`, and all `data/2026-07-04*/` sessions before capturing anything new.
2. **Quantitatively evaluate each tilt hypothesis, with real numbers, in this order:**
   - *Magnetic mutual coupling* — using the actual fitted per-coil L (regenerate via
     `tools/fit_rlc_model.py` / `fit_mutual_inductance.py` if a scope/data source is
     available; otherwise state the 6.64–6.7 mH figure as the best on-record estimate),
     k≈0.24, and drive current levels, compute expected induced EMF/current
     redistribution magnitude and resulting differential force/torque on the disk.
     Compare that predicted torque against the torque needed to visibly tilt the disk
     (estimate disk mass/moment of inertia from the "aerial-robotics platform" magnetic
     disk description, or flag as another measurement gap). State explicitly whether
     k≈0.24 is physically capable of the observed tilt magnitude, or is too small by how
     many orders of magnitude.
   - *Power-supply coupling* — already empirically ruled out in `2026-07-04c` (matched
     within 1%) and by the user's own single-supply retry; confirm/quantify with the
     existing capture rather than re-testing blind, and explain via a shared-impedance
     model why it was expected but didn't reproduce.
   - *CS measurement artifacts* — ground-bounce-correlated CS error, per-channel gain
     mismatch, disconnected-tap recurrence, ADC mux settling; show whether the "current
     imbalance" the compensation loop is reacting to is even real coil-current imbalance
     or partly/wholly a measurement artifact of the compensation's own actuation (i.e.,
     does trimming duty visibly perturb CS readings on channels the trim doesn't touch?).
   - *Static geometric field asymmetry* — the leading candidate per `2026-07-04f`/`g`:
     design and (if hardware access allows within the remote environment — otherwise
     write this up as a precise field procedure for the user to execute) a caliper/
     height measurement of each coil's mounting relative to the disk plane, and compute
     expected field-per-amp asymmetry vs. observed tilt direction (B/C vs A/D).
     This is the hypothesis most consistent with "fixed-in-lab-frame tilt, direction-
     independent, present with or without compensation."
   - *Compensation-loop-induced instability* — separately from *why there's an
     imbalance*, analyze whether `CoilBalancer`'s pair-based PI (can only trim down,
     `deadband`, `trimMax=40%`) or `main_current_pid`'s global-argmin PID (`KP=2.2,
     KI=0.10, KD=0.15`) can themselves go unstable or fight a slowly-varying geometric
     bias (e.g., integrator windup pulling a structurally-weak channel's neighbors down
     past a stable operating point) — i.e., is the "compensation causes tilt" report a
     control-loop artifact layered on top of the true bias, not the coupling itself?
   - *CW vs CCW torque asymmetry* — explicitly reconcile why tilt appears in **both**
     directions despite CW nominally adding tension. Consider: gyroscopic precession of
     a spinning disk under an off-axis restoring torque (precesses independent of spin
     sign for many geometries), reaction torque direction vs. suspension string
     torsional stiffneess, and any field-orientation sign convention bug (cross-check the
     `PHASES_CW`/`PHASES_CCW` drift noted above — verify the firmware's CW is actually
     CW in the lab, not accidentally CCW on one code path).
3. **Document hardware + experimental setup properly**, consolidating (not duplicating)
   what's scattered across `docs/PCB_Design_Documentation.md`, `hw_references/`,
   `data/README.md`, and memory notes: driver board topology (2× VNH5019A-E boards, 2
   coils each), CS signal chain and its known caveats, GND1/PGND1 fix status, per-channel
   calibration constants and their calibration method, phase/carrier control
   architecture (`PhaseController`, `CoilBalancer` vs `main_current_pid`), the two-supply
   vs one-supply topology history, and — as an explicit "missing data" callout — coil
   array physical geometry (spacing, mounting height, disk suspension mechanism), since
   none of it is currently written down anywhere in the repo.
4. **Design a sensorless (CS-only) flattening/stabilization scheme**, without an outer
   vision loop, that:
   - States plainly what physical quantity would need to be inferred from CS to close a
     stabilizing loop (e.g., a per-coil current asymmetry correlated with disk
     tilt/position, if one exists and is distinguishable from the geometric-bias floor
     found above) — and is honest if CS *cannot* observe tilt at all (recall: CS was
     already shown "electrically invisible" to disk presence in `2026-07-04f`, which is
     a serious constraint on any CS-only feedback design and must be addressed head-on,
     not glossed over).
   - If CS genuinely can't sense tilt, the report should say so and pivot the
     recommendation toward *removing* the static bias (geometric fix / one-time
     calibrated feedforward trim table) rather than proposing a closed loop on a signal
     that structurally can't see the disturbance — plus what minimal *additional* sensing
     (if any) would be the smallest step up from CS-only.
   - Any proposed control law must reuse the existing actuation primitives
     (`PhaseController::setCarrierDutyCycle`, `setPhase`) and existing per-channel CS
     path (`current_sense.h`/`CoilBalancer`), not invent a new hardware signal.
5. **Deliverables**, committed to a **new, clean/empty branch** (`git checkout --orphan`)
   named `research/coupling-tilt-investigation`, pushed to `origin`:
   - `docs/coupling_tilt_investigation.md` — the full write-up: hardware/setup
     documentation, per-hypothesis math and verdict, the CW/CCW tilt analysis, and the
     sensorless stabilization proposal (or its honest rejection + fallback).
   - Any regenerated analysis artifacts (plots/fits) it produces along the way, under
     `results/coupling_tilt_investigation/`.
   - A short top-level summary (in the same doc) giving a ranked list of hypotheses by
     estimated contribution, explicitly stating whether magnetic coupling survives as a
     plausible primary cause or should be demoted in favor of geometric asymmetry.

## Execution steps (this session)

1. `RemoteTrigger action:"create"` a one-off routine against this repo with the scope-of-work
   above as its prompt (condensed appropriately for the trigger body), instructing it to
   work on an orphan branch and push to `origin`.
2. `RemoteTrigger action:"run"` to kick it off immediately.
3. Report back to the user: the trigger id, the parsed run time, and the claude.ai URL
   where results will appear, plus the branch name to watch for on GitHub.

## Verification

This is a research/analysis deliverable, not a runtime code change — verification is:
read the produced `docs/coupling_tilt_investigation.md` for internal consistency (does
the math for each hypothesis actually support its stated verdict; are cited file
paths/data sessions real and correctly summarized), confirm the branch was pushed to
`origin` and is visible on GitHub, and confirm no changes landed on `main`.


