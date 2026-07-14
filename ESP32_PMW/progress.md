# Progress: State-Space Current-Balance Control for a 4-Coil ESP32 Rig

This document records the full technical history of the current-balance
control work on the ESP32_PMW rig: the physical system, the control laws
implemented (with full derivations), every bug found and fixed, the
hardware experiments that ruled out competing hypotheses, and the currently
open root cause with its corrected derivation. It is written to be
self-contained enough to reconstruct every design decision from first
principles.

---

## 1. System description

### 1.1 Hardware

Four electromagnetic coils, labeled A, B, C, D, arranged in a 2x2 physical
grid:

```
C  B
A  D
```

so A-D and C-B are the "far diagonal" pairs (physically farthest apart) and
A-C, A-B, D-C, D-B are nearest-neighbor pairs. Each coil is driven by a
VNH5019 H-bridge motor driver IC. The VNH5019's `CS` (current-sense) pin
outputs a voltage proportional to the driver's **supply current**, sensed
via `CurrentSense.h` with gain `SENS_i` (A per V, per-channel calibrated,
e.g. `{15.26, 15.28, 15.57, 15.34}`) and a single-pole EMA filter
(`TAU_FILTER_MS = 50`).

Each coil is driven with a PWM carrier (`PWM_FREQ`, fixed, high frequency)
whose **duty cycle** `u_i in [0,100]%` sets the effective drive voltage, and
whose **phase offset** (`PHASES_CW`/`PHASES_CCW`, one per channel) together
with a **commutation/rotation frequency** `f` (swept 1 -> 210 Hz over the
run) creates a rotating multi-phase drive pattern, analogous to a BLDC
stator field. `f` is the "freq" reported in every telemetry line and is
*not* the PWM carrier frequency -- it is the frequency at which the duty
pattern itself is modulated/sequenced across the 4 channels.

### 1.2 Software/firmware layers

- `PWMController` / `PWMSequencer` (libs): low-level phase/carrier PWM
  generation and a frequency-ramp task sequencer.
- `CurrentSense`: per-channel ADC read + EMA filter + zero-calibration.
- `SerialComm`: line-framed serial command protocol.
- Application firmware (`src/main_*.cpp`, PlatformIO envs): different
  control policies built against the same hardware libraries; this document
  covers `main_current_pid.cpp` (baseline) and `main_state_space.cpp`
  (state-space/LQID), plus the various offline Python tools in `tools/`
  that fit models and generate compile-time gain headers.

---

## 2. Baseline: global min/max PI(+D) controller (`main_current_pid.cpp`)

**Policy** (no board-grouping, literal global min/max): every tick, the
lowest-current channel `idx_min` is driven toward 100% duty (maximizing
achievable current, since raising the group's floor is the only way to
raise everyone else's achievable target); every other channel `i` runs an
anti-windup PID against the shared target `i_min = i_meas[idx_min]`:

```
err_i(k)  = i_min(k) - i_meas_i(k)
d_i(k)    = (err_i(k) - err_i(k-1)) / (dt_k / T_nom)      # rate-normalized derivative
integrator_i(k+1) = integrator_i(k) + K_I * err_i(k) * (dt_k/T_nom)      [conditional, see below]
candidate  = integrator_i(k) + K_P*err_i(k) + K_D*d_i(k)
u_i(k)     = clamp(candidate, DUTY_MIN, DUTY_MAX)
```

Converged gains: `K_P=2.2, K_I=0.10, K_D=0.15` (duty-%-per-Amp and
duty-%-per-Amp-per-tick units). `dt_k/T_nom` rate-normalizes `K_I`/`K_D` so
their real-world meaning is independent of the actual loop period (`loop()`
is unthrottled, so `dt_k` varies tick to tick).

**Anti-windup** (asymmetric, a lesson later re-derived independently for
the state-space controller -- see Section 4.2): freeze the integrator only
when the *unclamped* candidate exceeds a bound **and** the error is pushing
further into that same saturation direction; never freeze when the error
would pull the integrator back the other way.

This is the empirical performance baseline used throughout this document:
peak spread 0.447 A, HOLD spread 0.394 A, over the full 1->210 Hz sweep
(`state_space_cw_2A_PI_baseline.log`).

---

## 3. Why move beyond PI: motivation for a model-based approach

The PI controller has no model of the plant -- it discovers the correct
duty purely by integrating error, tick by tick, from scratch every run. Two
concrete weaknesses motivated a state-space redesign:

1. **No feedforward** -> slow convergence: the integral term has to
   *rediscover* the roughly-correct duty-to-current relationship (which is
   knowable in advance from a plant model) every single run, which is why
   RAMP_UP took many seconds to converge under pure PI.
2. **No coupling model** -> reactive-only saturation handling: cross-channel
   coupling (stepping one channel measurably moves the others -- see
   Section 3.2) means correcting one channel's error can worsen another's;
   a model-based controller can account for this instead of discovering it
   via oscillation.

This motivated: (a) an explicit plant model (Section 3.1-3.2), (b) an
exact model-inversion feedforward (Section 3.3), (c) LQI/LQID optimal
feedback (Section 3.4) instead of hand-tuned PI, and (d) a constrained
duty allocator (Section 3.6) that redistributes saturation optimally using
the same coupling model instead of per-channel clamping.

### 3.1 Continuous-time plant model

Each coil is modeled as an `R`-`L` circuit driven by an effective voltage
`V_i(u_i)` set by duty `u_i`:

```
L_i * di_i/dt + (Rx)_i = V_i(u_i)
```

In vector form for the 4-channel system, `x = [i_A, i_B, i_C, i_D]^T`
(state = coil currents, the *controlled* quantity), `u = [u_A,...,u_D]^T`
(duty %, the *actuated* quantity):

```
L dx/dt + R x = V(u),      V(u) = diag(Kv) u
```

`L` is diagonal (self-inductance per channel; measured mutual-inductance
terms default to 0 -- justified below). `R` is a **full, non-diagonal**
matrix: an empirical step-response test found real steady-state
cross-channel coupling (stepping channel D moved A/B/C by 24-33% of D's own
current shift) that a mutual-inductance-only model cannot produce (mutual
inductance is a `di/dt`-only effect; the observed coupling persists at
steady state, `di/dt=0`). Rather than add explicit mutual-inductance
states, the measured coupling is folded directly into `R`: at steady state
`R x = V(u)` => `x = R^-1 V(u)`, so a *non-diagonal* `R^-1`, fit directly
from measured duty->current step responses (`tools/fit_coupling_matrix.py`),
exactly reproduces the observed cross-channel steady-state shifts. This is
a lumped modeling choice, not a claim about the underlying physical
mechanism (shared supply loading vs. genuine magnetic coupling too fast to
resolve at the achievable control rate are both consistent with it).

`Kv_i` converts duty percentage to the PWM drive's effective (fundamental)
voltage: `Kv_i = (4/pi) * V_supply / 100`, the standard Fourier
fundamental-harmonic amplitude of a switched square wave of peak
`V_supply`, linearized per duty percentage point.

Rearranging into standard state-space form:

```
dx/dt = -L^-1 R x + L^-1 diag(Kv) u  =  A x + B u
A = -L^-1 R
B =  L^-1 diag(Kv)
```

The output equation (measured quantity, the CS pin voltage, distinct from
the controlled state `x`):

```
y = C x + D u,   C = diag(1/SENS_i),   D = 0
```

(`D=0`: duty has no instantaneous path to the sensed voltage that bypasses
the coil current state.)

### 3.2 Discretization (zero-order hold)

The controller runs at a fixed sample period `Ts` (50 ms). Holding `u`
constant over each sample interval (zero-order hold), the exact discrete
update is derived from the continuous solution
`x(t) = e^{At}x(0) + int_0^t e^{A(t-tau)} B u(tau) dtau`. With `u`
piecewise-constant over `[kTs, (k+1)Ts)`:

```
x(k+1) = e^{A Ts} x(k) + ( int_0^{Ts} e^{A tau} dtau ) B u(k)
```

Since `d/dtau [e^{A tau}] = A e^{A tau}`, `int_0^{Ts} e^{A tau} dtau =
A^-1 (e^{A Ts} - I)` (valid for invertible `A`). So:

```
x(k+1) = Ad x(k) + Bd u(k)
Ad = expm(A Ts)
Bd = A^-1 (Ad - I) B
```

implemented in `tools/build_state_space_model.py::discretize()` via
`scipy.linalg.expm`.

### 3.3 The fast-electrical-settling approximation, and its consequence

The coils' electrical time constant is `tau_elec ~ L/R`. From the fitted
values (Section 5.4), `tau_elec ~ 1 mH / 0.25 Ohm ~ 4 ms`, i.e. `A`'s
eigenvalues are large and negative (roughly `-R/L ~ -250 rad/s` per
diagonal-dominant channel). Over `Ts = 50 ms`, `A*Ts ~ -12.5`, so
`Ad = e^{A Ts} ~ 0` (to within `e^-12 ~ 4e-6`) -- confirmed numerically
during model construction ("`Ad ~ 0`" is explicitly noted in
`tools/design_lqr_gains.py`'s docstring). This means the coil's electrical
dynamics are entirely settled *within* one control tick, and the discrete
model collapses to a **static (memoryless) gain**:

```
Bd = A^-1 (Ad - I) B  ~=  -A^-1 B   (since Ad ~ 0)
   = -(-L^-1 R)^-1 (L^-1 diag(Kv))
   =  (L^-1 R)^-1 L^-1 diag(Kv)
   =  R^-1 L L^-1 diag(Kv)
   =  R^-1 diag(Kv)
```

**`Bd` is, to a very good approximation, just `R^-1 diag(Kv)`** -- the pure
DC/steady-state Ohm's-law gain (`x = R^-1 V(u)`), independent of `L`
entirely. This is the origin of the root-cause bug in Section 5: `Bd`
carries **no frequency information whatsoever** -- it is built from a
step-response (steady-state, `omega=0`) characterization, not from any
particular drive frequency.

### 3.4 Coupling-aware feedforward

Given a static gain `x_ss = Bd u` (steady-state current for a held duty),
the feedforward that should drive the plant directly to a target current
`x_target` is the model inverse:

```
u_ff = Bd^-1 x_target =: FF * x_target
```

`FF = Bd^-1`, computed and validated offline (`tools/build_feedforward.py`):
`cond(Bd)` checked against a threshold (`<1e4`), and `FF @ Bd ~= I` verified
to machine precision (`~2e-16`). In firmware this is an explicit matrix-
vector product against the (currently uniform) target vector:

```
u_ff[i] = sum_j FF[i][j] * r_target        (x_target[j] == r_target, all j)
```

### 3.5 LQI/LQID: optimal feedback around the feedforward

Rather than hand-tuned PID gains, the feedback correction uses LQR gains
computed from an **integral-augmented** (and derivative-augmented) version
of the discrete plant, so that steady-state tracking error is driven to
zero (an LQR on `x` alone would not guarantee zero steady-state error under
model mismatch/disturbances) while gains remain provably optimal for a
quadratic cost.

Define the tracking error `e(k) = i_meas(k) - r_target(k)` and its discrete
integral state `z(k+1) = z(k) + Ts * e(k)`. With optional filtered
derivative state `d(k)`, the augmented state is `xi = [e; d; z]` (in the
actual implementation, feedback is computed directly in error-coordinates
rather than raw `x`-coordinates, since only relative tracking behavior --
not absolute state -- needs to be regulated; `Ad~=0` from Section 3.3
means state-feedback `K_x` acting on raw `x` would carry almost no useful
information anyway, consistent with the design note in
`design_lqr_gains.py` that `K_x` is expected to end up near-zero).

Standard discrete LQR cost:

```
J = sum_{k=0}^inf  xi(k)^T Q xi(k) + u(k)^T R_cost u(k)
```

with block-diagonal `Q = diag(Q_track, Q_derivative, Q_integral)` (tunable
per-block weights `--q-track --q-derivative --q-integral`) and
`R_cost = r_duty * I` (actuation-effort penalty). The optimal gain solves
the discrete algebraic Riccati equation (DARE):

```
P = Abar^T P Abar - Abar^T P Bbar (R_cost + Bbar^T P Bbar)^-1 Bbar^T P Abar + Q
K = (R_cost + Bbar^T P Bbar)^-1 Bbar^T P Abar
```

for the augmented `(Abar, Bbar)` pair built from `Bd`, solved once offline
via `scipy.linalg.solve_discrete_are` (`tools/design_lqr_gains.py`; a
single dense solve on a small, e.g. 12x12, augmented system -- fast enough
to call repeatedly inside an outer hyperparameter search, see Section 7).
`K` is partitioned into blocks `K_x, K_d, K_z` and baked into
`src/lqr_gains_{cw,ccw}.h`. Feasibility is checked via the closed-loop
spectral radius `max|eig(Abar - Bbar K)| < 1`; an infeasible (non-
stabilizing) parameter choice raises `SystemExit`, used as a hard
"infeasible" signal by the automated tuner (Section 7).

At runtime (`CurrentBalanceController::computeTick()`):

```
err[i]       = i_meas[i] - r_target
correction[i] = -(K_x @ err)[i] - (K_d @ derr_filt)[i] - (K_z @ z_integral)[i]
u_desired[i]  = u_ff[i] + correction[i]
```

### 3.6 Discrete derivative filter

Raw one-sample error differences amplify ADC/measurement noise directly
into duty chatter. A single-pole (EMA) low-pass is applied to the raw
derivative estimate:

```
derr_raw(k)  = (err(k) - err(k-1)) / Ts
derr_filt(k) = derr_filt(k-1) + alpha * (derr_raw(k) - derr_filt(k-1))
```

`alpha` is chosen as the standard discrete-EMA coefficient for a target
time constant `tau`:

```
alpha = Ts / (tau + Ts)
```

This is the small-`Ts/tau` (`Ts << tau`) approximation to a first-order
continuous low-pass `dy/dt = (x-y)/tau`; the *exact* zero-order-hold
discretization would give `alpha_exact = 1 - e^{-Ts/tau}`, which agrees
with `Ts/(tau+Ts)` to first order when `Ts << tau` (here `Ts=50ms`,
`tau=150ms`, ratio `1/3` -- a modest, not asymptotic, approximation, judged
acceptable since exact pole placement isn't safety-critical for a smoothing
filter). `tau=150ms >= 3*Ts` was chosen so the filter meaningfully smooths
without lagging the RAMP_UP transient it's meant to help damp.

### 3.7 Constrained duty allocator (bound-constrained QP)

Feedback correction can push `u_desired` outside `[DUTY_MIN, DUTY_MAX]`.
Rather than clamp each channel independently (which cannot use a
non-saturated channel to compensate for a saturated one, since the plant is
coupled via `Bd`'s off-diagonal terms), duty allocation is posed as a small
bound-constrained least-squares problem:

```
minimize_u   || Bd u - x_target ||^2  +  r_duty || u - u_ff ||^2
subject to   DUTY_MIN <= u_i <= DUTY_MAX
```

**Target construction.** Feedforward and feedback are folded into one
target *before* allocation: `x_target_eff = Bd @ u_desired` -- "the current
response the unconstrained controller wanted." The allocator then finds the
closest *feasible* duty that reproduces that response, letting
non-saturated channels pick up a saturated channel's slack optimally.

**Convexity and solution method.** Expanding the cost:

```
J(u) = (Bd u - x_target)^T (Bd u - x_target) + r_duty (u - u_ff)^T (u - u_ff)
grad J(u) = 2 Bd^T (Bd u - x_target) + 2 r_duty (u - u_ff)
Hessian H = 2 Bd^T Bd + 2 r_duty I
```

`H` is constant and positive definite for `r_duty > 0` (strongly convex,
strong-convexity modulus `mu = 2 r_duty`, from the regularizer's
contribution alone). The bound-constrained problem is solved by **projected
gradient descent**:

```
u^(0) = clamp(duty_prev, DUTY_MIN, DUTY_MAX)         # warm start
u^(k+1) = clamp( u^(k) - alpha * grad J(u^(k)),  DUTY_MIN, DUTY_MAX )
```

Convergence (geometric, rate `<= 1 - mu/L` per iteration for `mu`-strongly-
convex, `L`-smooth `J`) requires `alpha <= 1/L`, `L` = largest eigenvalue of
`H`. Computing the exact spectral norm on-device is unnecessary; the cheap
Frobenius-norm bound `||Bd^T Bd||_2 <= ||Bd||_F^2` gives a safe (slightly
conservative) Lipschitz constant:

```
L = 2 ||Bd||_F^2 + 2 r_duty        (computed once per direction, at startup)
alpha = 1 / L
```

15 fixed iterations (no early-exit convergence check) are used for
deterministic worst-case tick timing rather than data-dependent iteration
count -- offline-validated (`tools/validate_duty_allocator.py`) against
`scipy.optimize.lsq_linear` ground truth: **0.010% relative cost gap**,
well inside the 5% validation bound.

### 3.8 Directional anti-windup

Freezing the integral state unconditionally whenever a channel saturates
locks in whatever correction caused the saturation and can never release it
even after the situation reverses. The (twice-independently-discovered,
Section 4.1/4.2) correct rule is **directional**:

```
this_saturated_high      = (duty_out[i] >= DUTY_MAX)
would_deepen_saturation  = this_saturated_high AND (err[i] < 0)
z_integral[i] += Ts * err[i]   IFF   NOT would_deepen_saturation
                                AND  NOT any_overcurrent
                                AND  freq_hz >= FREQ_MODEL_VALID_HZ   (Section 4.5)
```

Never freeze on hitting `DUTY_MIN` (an earlier, separate hardware deadlock:
low-clamping just means duty undershot the floor, and the integral must
keep accumulating to escape it -- asymmetric with the `DUTY_MAX` case
because saturating low never "deepens" in the same runaway sense observed
at the high rail).

### 3.9 Dual-core architecture

Core 0 (dedicated FreeRTOS task): the entire low-level, timing-critical
domain -- ADC sampling, PWM/carrier stepping, the bounded
ARMING->RAMP_UP->HOLD->ENDING->STOPPED state machine, and an independent
hard overcurrent trip. It applies whatever duty is in shared "command
memory," indifferent to freshness (no staleness watchdog by design). Core 1
(Arduino's default loop): serial comm, the control law
(`CurrentBalanceController`), and the reference governor. This makes the
bounded/safe-run guarantee independent of core 1's health -- verified via a
deliberate core-1-stall injection test (core 0 completed its full run and
safely latched coils off with zero core-1 input, confirmed via the
board's LED). Cross-core data uses two independent `portMUX_TYPE`
spinlock-protected regions (log memory core0->core1, command memory
core1->core0), copy-in/copy-out only, no floating-point work inside a
critical section.

---

## 4. Bugs found and fixed (chronological)

### 4.1 Global anti-windup (pre-existing, motivated the redesign)

Freezing `z_integral` **globally** whenever *any* channel saturated starved
every other channel's still-needed correction the instant one channel hit
its ceiling -- confirmed as a real regression (HOLD-phase spread jumped
from ~0.08A to ~0.45A because one saturated channel froze every channel's
integrator). Fixed by making anti-windup **per-channel** (Section 3.8).

### 4.2 Directional anti-windup (found during allocator bring-up)

Even per-channel, an *unconditional* freeze at `DUTY_MAX` (regardless of
error sign) still locks in excess correction and can never unwind it.
Traced on hardware: CCW duty stuck at `[77,73,100,100]`, current climbed to
~4.5A against a 1.8A target, entirely unable to correct because the
integral was frozen solid despite the error having flipped to a large
overshoot. Fixed with the `would_deepen_saturation` condition (Section 3.8)
-- validated on hardware (`r` reached full target, no more runaway).

### 4.3 `ALLOCATOR_R_DUTY` scale mismatch (magnitude analysis)

`ALLOCATOR_R_DUTY` was initially set to `0.1`, inherited from
`design_lqr_gains.py`'s own `--r-duty` CLI default -- but that default
belongs to a **different cost formulation** (the offline LQR's augmented-
state cost, with `Q` terms scaled `O(1-50)`), not the allocator's QP
(Section 3.7), whose two terms have very different natural units:

```
tracking term:    ||Bd u - x_target||^2     ~  O(1-4)      [Amps^2]
regularizer term: r_duty * ||u - u_ff||^2   ~  r_duty * (Delta_u)^2   [percentage-points^2]
```

At `r_duty=0.1`, a mere `Delta_u = 10` percentage-point deviation already
contributes `0.1 * 10^2 = 10` -- larger than the *entire* tracking term.
The regularizer dominated the gradient everywhere, so duty barely moved
away from `u_ff` regardless of a large, persistent tracking error (observed
on hardware: A/B/C severely undershooting while duty stayed pinned near
`u_ff`). Fixed by re-deriving the constant for the QP's actual units:
`r_duty = 0.001` makes a `Delta_u ~ 30` deviation contribute `0.001*900 =
0.9` -- present as a tie-breaker/regularizer, not a term that fights the
primary objective. Re-validated offline (0.010% gap vs. ground truth,
Section 3.7) and on hardware (spread dropped from ~0.77-1.5A to as tight as
0.10-0.19A across multiple HOLD samples).

### 4.4 Soft/hard overcurrent threshold collision

Both the soft backoff (`CurrentBalanceController`, core 1, `Ts=50ms`
cadence) and the hard trip (`PwmActuator`, core 0, `~1ms` ADC cadence) used
the same threshold `I_MAX_A=12.0`. Since the hard trip samples ~50x faster,
it always wins the race for any transient that crosses the shared
threshold, making the soft path dead code (found by the `code-simplifier`
review agent, not by hardware testing). Fixed by introducing a distinct,
lower `I_SOFT_LIMIT_A = 10.0` (2A margin) for the soft path.

### 4.5 RAMP_UP overshoot: frequency-gated integral (partial fix, superseded root-cause work in Section 5)

Observed: channel D pinned at `DUTY_MAX` continuously for 13+ seconds
through RAMP_UP while its own current crossed from legitimate undershoot
into *sustained overshoot* (`D=2.03A` vs. `r=1.755A`, then continuing to
overshoot by `0.28-0.76A` for 8+ more seconds) without duty ever backing
off. Root cause at the time: `Bd`/`FF` (and hence the entire control law)
implicitly assumes a single fixed operating point, but the coils' true
duty->current gain is frequency-dependent (later precisely identified,
Section 5.3-5.4) -- integrating tracking error while the model doesn't yet
apply just accumulates stale correction. Partial fix: gate integral
accumulation on drive frequency (added to the directional anti-windup
condition, Section 3.8): `freq_hz >= FREQ_MODEL_VALID_HZ` (`130 Hz`,
matching the RLC fit's resonance band lower edge, see Section 5.4).
Confirmed on hardware: the specific stuck-at-100%-through-sustained-
overshoot failure mode is gone -- duty now actively unwinds within ~1-2s of
crossing into the valid band. This did **not**, however, fully resolve the
RAMP_UP spread spikes (peak spread only dropped from 2.875A -> 2.413A CW,
2.656A -> 1.868A CCW) -- because it addresses the *integral's* behavior,
not the *feedforward's* own wrong magnitude, which turned out to be the
actual dominant cause (Section 5).

---

## 5. Root-cause investigation: persistent RAMP_UP spread

After Section 4.5's fix, true whole-run peak spread remained large (2.41A
CW / 1.87A CCW vs. a target of <0.4A). Two competing hypotheses were tested
and ruled out before the actual root cause was isolated.

### 5.1 Hardware ruled out

The original PI controller (Section 2, no plant model at all) was flashed
and run through the identical 1->210Hz sweep, same rig, same day:

| controller | true whole-run peak spread | HOLD spread |
|---|---|---|
| Original PI | **0.447 A** | 0.394 A |
| State-space (post 4.5 fix) | 2.413 A (CW) / 1.868 A (CCW) | 0.505 / 0.680 A |

Since the PI controller crosses the *same* resonance band on the *same*
hardware with no spike, the coils/drive electronics are not the cause --
this rules out a hardware or fundamental-resonance-artifact explanation and
points at the state-space *control law* specifically.

### 5.2 Core-count/timing jitter ruled out

Hypothesis: the dual-core split (Section 3.9) introduces scheduling jitter
(e.g. `vTaskDelay(1)` between core-0 iterations, spinlock contention) that
degrades tracking independent of the control law itself. Tested directly:
a temporary single-core diagnostic build (`main_state_space_singlecore.cpp`,
`[env:state_space_singlecore]`) ran the **byte-identical**
`CurrentBalanceController`/`PwmActuator`/allocator/freq-gate code on one
core, unthrottled, matching the original PI controller's cadence:

| build | true whole-run peak spread (CW) |
|---|---|
| Dual-core | 2.413 A |
| Single-core (identical control law) | 2.209 A |

Within noise of each other -- **core count is not implicated**. (An
*earlier*, less careful comparison across differently-buggy historical logs
had suggested a 3-4x jitter effect; that comparison was confounded by
control-law bugs from Section 4.2/4.3 still being present in the dual-core
logs used, not a genuine core-count effect -- corrected by this controlled,
same-code test.)

### 5.3 Actual root cause: the feedforward model has no frequency term

From Section 3.3: `Bd ~= R^-1 diag(Kv)`, derived entirely from a
**step-response** (DC, `omega=0`) characterization. `FF = Bd^-1` is
therefore a single, frequency-independent matrix, evaluated every tick
regardless of the current drive/rotation frequency `f in [1,210] Hz`. But
`R` (pure DC resistance) is only the coil's *true* impedance magnitude
`|Z(f)|` in the limit `f -> 0` in a purely resistive model -- and Section
5.4 shows the coils are **not** purely resistive: `|Z(f)|` varies by more
than 500x across the swept range for some channels. The feedforward
computed from the static model is therefore wrong by a large,
frequency-dependent factor almost everywhere in the sweep, and only
approximately correct near wherever `|Z(f)| ~= R` (which turns out to be
*exactly* the resonance frequency `f0`, Section 5.4) -- i.e. correct
essentially nowhere except a narrow band, wrong (and by how much varies
continuously) everywhere else. This matches every symptom observed:

- One channel (D on CW, C on CCW) sits pinned near 100% duty from the very
  start of RAMP_UP, well before any feedback/integral term has had time to
  wind up -- this is feedforward alone commanding too little duty relative
  to what's actually achievable at low frequency (confirmed deterministic
  by Section 5.2's identical-code single-vs-dual-core result -- not a
  timing artifact).
- Spread spikes cluster specifically in the resonance-crossing region
  (roughly 90-190Hz in every RAMP_UP log this session) where the true gain
  is changing fastest while the model's assumed gain is constant.
- Spread settles back to a well-behaved 0.1-0.5A range once frequency
  reaches HOLD (190-210Hz), where the static model happens to be closest to
  correct.

### 5.4 The coil impedance model and its resonance (RLC fit)

`tools/fit_rlc_model.py` fits, per channel, a series R-L-C circuit directly
against **measured current** at each frequency of an open-loop, fixed-duty
frequency sweep (`main_experiment.cpp`'s solo-sweep mode):

```
Z(omega) = R + j( omega L - 1/(omega C) )         (series R-L-C impedance)
|Z(omega)| = sqrt( R^2 + (omega L - 1/(omega C))^2 )
i_true(omega) = v_fund / |Z(omega)|                (magnitude response)
```

(fit against measured current directly, with a correction for
`CurrentSense.h`'s own 50ms EMA filter lag, to avoid the poor SNR of
differentiating/inverting to reconstruct impedance directly from noisy
measurements). Fitted values (per channel, `tools/rlc_fit.json`):

| ch | R (Ohm) | L (mH) | C (mF) | f0 (fit, Hz) |
|---|---|---|---|---|
| A | 0.256 | 0.946 | 1.128 | 154.1 |
| B | 0.224 | 0.789 | 1.328 | 155.5 |
| C | 0.254 | 0.923 | 1.165 | 153.4 |
| D | 0.279 | 1.085 | 0.996 | 153.1 |

**Resonance derivation.** The reactance term vanishes (`|Z|` is minimized,
purely resistive) when `omega0 L = 1/(omega0 C)`, i.e.:

```
omega0 = 1/sqrt(LC)      =>      f0 = 1/(2*pi*sqrt(L*C))
```

Cross-checked numerically against the fitted `L, C` (this document):

```
f0_A = 1/(2*pi*sqrt(0.946e-3 * 1.128e-3)) = 154.07 Hz   (fit: 154.1)
f0_B = 1/(2*pi*sqrt(0.789e-3 * 1.328e-3)) = 155.48 Hz   (fit: 155.5)
f0_C = 1/(2*pi*sqrt(0.923e-3 * 1.165e-3)) = 153.48 Hz   (fit: 153.4)
f0_D = 1/(2*pi*sqrt(1.085e-3 * 0.996e-3)) = 153.10 Hz   (fit: 153.1)
```

exact agreement, confirming internal consistency of the fit. At resonance,
`|Z(f0)| = R` (the series RLC's impedance minimum); away from resonance in
*either* direction `|Z(f)| > R` strictly. Numerically (channel A, `R=0.256
Ohm`):

| f (Hz) | \|Z(f)\| (Ohm) | \|Z(f)\|/R |
|---|---|---|
| 1 | 141.09 | 551.1 |
| 60 | 2.011 | 7.86 |
| 105 | 0.764 | 2.98 |
| 150 | 0.261 | 1.02 |
| 154.1 (f0) | 0.256 | **1.00** |
| 190 | 0.464 | 1.81 |
| 210 | 0.631 | 2.46 |

(all four channels show the same qualitative shape; A is representative).
This confirms the earlier symptom analysis directionally: near `f=1Hz`, the
true impedance is ~550x the DC resistance the static model assumes --
current achievable for a given duty near the start of RAMP_UP is
*drastically* lower than the model's steady-state assumption, exactly
matching the observed "pinned near 100%, still undershooting" behavior at
low frequency, and the crossover to overshoot as frequency approaches `f0`
(where achievable current for the *same* duty rises sharply toward the
resonance peak, i.e. toward the model's implicit assumption).

### 5.5 Corrected feedforward derivation (careful sign analysis)

The static model implicitly assumes `x = (Kv/R) u`, i.e. a gain of exactly
`Kv/R` **at every frequency** -- which Section 5.4 shows is only true at
`f0`. The *true* frequency-dependent gain (same derivation as Section 3.3,
but keeping `L`/`C`'s reactance rather than dropping to the pure-DC limit)
is `x = (Kv/|Z(f)|) u`. Equating the true achievable current to the target
and solving for the duty actually required:

```
static (wrong) feedforward:    u_ff_static  = R * i_target / Kv
true required duty at f:       u_true(f)    = |Z(f)| * i_target / Kv
                                              = u_ff_static * ( |Z(f)| / R )
```

Since `|Z(f)|/R >= 1` for all `f` (equality only at `f0`), **the correction
factor must be >= 1**, i.e. duty must be *increased* relative to the static
estimate away from resonance, never decreased. (An earlier draft of the
fix plan defined a `scale = R/|Z(f)| <= 1` and proposed *multiplying*
`u_ff` by it -- this is backwards: it would *shrink* feedforward exactly
where the true gain is lower and more duty is needed, making the
low-frequency undershoot worse, not better. The corrected formula --
multiply by `|Z(f)|/R`, not `R/|Z(f)|` -- was caught during this
derivation and is what any implementation should use.) At `f=1Hz`,
channel A's correction factor is `~551x` -- physically real (matches the
impedance ratio) but far too large to apply as a naive multiplier without
an explicit upper clamp (duty saturates at `DUTY_MAX=100%` almost
immediately regardless), so any implementation needs a deliberately chosen
cap on the correction factor's magnitude at the low-frequency end, tuned
against real closed-loop data rather than applied blindly.

---

## 6. Outcome: executed through Phase 6, then reverted to the PI controller

The plan in Section 7 below was executed through Phase 6. Summary of what
actually happened, and why the project concluded by reverting to the
original PI controller (Section 2) rather than continuing to chase the
0.4A target on the state-space controller:

- **Phase 0-1**: cleanup and RLC-fit validation, as planned --
  `resid_rms_a` 0.030-0.034A across all four channels (~1.5% of the ~2A
  operating scale), fit judged trustworthy.
- **Phase 2**: `tools/build_freq_gain_header.py` + `src/FreqGainModel.h`
  implemented per Section 5.5's corrected derivation (`u_ff *=
  |Z(f)|/R`, clamped at `FREQ_GAIN_CORRECTION_MAX`). Wired into
  `CurrentBalanceController::computeTick()`.
- **Phase 3**: offline validation (`tools/validate_freq_gain_model.py`)
  confirmed the correction model is internally consistent (dips to exactly
  1.0 at each channel's fitted `f0`) but found the raw RLC-derived
  correction magnitude (7-10x at 60Hz) far more aggressive than what real
  closed-loop duty data implied was needed -- most likely because a
  per-channel scalar correction can't capture how much of the real
  compensation the allocator's cross-channel coupling already provides.
  `FREQ_GAIN_CORRECTION_MAX` was set conservatively to `3.0` for the first
  hardware pass rather than the raw model's implied value, consistent with
  this project's incremental-validation practice.
- **Phase 4** (hardware, `rmax=2.0A`): mixed, direction-asymmetric result.
  CW true whole-run peak spread improved modestly (2.413A -> 2.021A) but
  CCW got *worse* (1.868A -> 2.695A), with a new overshoot pattern
  appearing around 165-190Hz that hadn't been as severe before (A/B/C
  climbing past 3A). This showed the feedforward correction's interaction
  with the *fixed* LQID gains is itself direction-dependent -- expected
  per the plan's own caveat, but confirmed the fix is necessary, not
  sufficient, alone.
- **Phase 5**: `run_max_spread` (true whole-run, unmasked max spread)
  added to `tools/pid_metrics.py` and threaded into
  `tools/tune_lqr_hyperparams.py`'s cost function and CSV output,
  backward-compatible with existing parsers.
- **Phase 6** (automated HIL tuning, `{q_spread, r_duty}`, 16 hardware
  trials total across both directions): the tuner's own live safety-abort
  (`spread > 2.0A`, `current > 1.5*rmax`) initially tripped on literally
  the baseline trial for both directions -- these soft, search-level
  thresholds predated the frequency correction and were sized for a
  regime where ~2A spread meant "something is badly broken," not "a normal
  if suboptimal operating point" as it now was. Raised to `spread_limit=
  3.0A`, `current_margin=2.0x` (still far under firmware's real hardware
  limits, `I_MAX_A=12A` / `I_SOFT_LIMIT_A=10A`, both untouched) with
  explicit user confirmation before resuming. With room to search, the
  coordinate descent completed both directions but found essentially no
  improvement in the metric that matters: best `run_max_spread` was 2.39A
  (CW, baseline 2.53A) and 2.447A (CCW, baseline 2.34A -- i.e. flat/no
  real improvement). This **confirms** the plan's own prediction: `q_spread`
  and `r_duty` don't touch the mechanism actually producing the residual
  spread (feedforward error during the resonance crossing, plus its
  direction-asymmetric interaction with fixed LQID gains) -- no amount of
  retuning these two weights was ever going to close a ~2A gap to a 0.4A
  target.

**Decision point**: offered a third search parameter
(`FREQ_GAIN_CORRECTION_MAX`, requiring new tooling to patch a compile-time
constant per trial and another ~15-25 minute automated hardware round).
User declined and instead directed a full reversion to the original PI
controller (Section 2) as the final deployed state.

**Final state**: the board is flashed with `main_current_pid.cpp`
(`current_pid` PlatformIO env) -- the simple global-min/max PI(+D)
controller, no plant model, no feedforward, no state-space machinery. This
is, empirically, the best-performing controller measured across this
entire investigation: **0.447A** true whole-run peak spread, vs. every
state-space variant's best result of ~2.0-2.7A even after: coupling-aware
feedforward, a constrained duty allocator, directional anti-windup, a
frequency-gated integral, dual-core architecture (ruled out as a factor),
a frequency-scaled feedforward correction derived from first principles,
and 16 rounds of automated LQR-weight search. The state-space approach's
core modeling assumption -- that a single DC/steady-state gain matrix
(`Bd`, hence `FF`/`K_x`/`K_d`/`K_z`, hence the allocator's target) can
represent this plant well enough across a 1-210Hz sweep that crosses the
coils' own LC resonance -- turned out to require more correction machinery
(frequency-scheduled feedforward *and* frequency-scheduled feedback gains
*and* a frequency-aware allocator target, none of which were fully
implemented) than the investigation's remaining scope justified, given a
simple model-free PI controller already met the target comfortably. Left
as a concrete lesson for any future revisit: model-based control only pays
for itself once the model is accurate across the full operating envelope,
not just at a single characterization point -- a lesson which, absent this
full derivation, is easy to underestimate the cost of.

Left in the repo, not reverted (harmless, dead code paths unless
`state_space` is reflashed): `src/CurrentBalanceController.{h,cpp}`,
`src/FreqGainModel.h`, `src/rlc_gain_model.h`,
`tools/build_freq_gain_header.py`, `tools/validate_freq_gain_model.py`,
and the `run_max_spread` metric addition to `tools/pid_metrics.py` (a
strict improvement to the metrics tooling regardless of which controller
is used going forward).

---

## 7. Original forward plan (executed through Phase 6, see Section 6 above)

Full phased plan lives at
`~/.claude/plans/implement-pid-autotuner-using-harmonic-sundae.md` (Phase 0
cleanup of the diagnostic single-core build; Phase 1 validate the RLC fit's
residuals as a gain(f) predictor; Phase 2 derive and wire in the corrected
`|Z(f)|/R` feedforward scaling with an upper clamp, per Section 5.5's
corrected sign; Phase 3 offline sanity-check plots before hardware; Phase 4
hardware validation of the feedforward fix alone; Phase 5 add a true
whole-run `max(spread)` metric to `tools/pid_metrics.py` (does not exist
today -- current metrics are HOLD-only or resonance-band-restricted); Phase
6 extend the existing HIL tuner `tools/tune_lqr_hyperparams.py` (build-
flash-capture-with-live-safety-abort-and-guaranteed-safe-finalize,
confirmed reusable) to search `{q_spread, r_duty}` (and a third parameter
if needed) against `run_max_spread < 0.4A` as an explicit pass/fail gate,
rather than the relative-improvement-only scoring used in the prior sweep;
Phase 7 final end-to-end verification against the PI baseline).

**Key finding that reframes the prior automated-tuning attempt:** the one
existing HIL sweep (`tools/tune_lqr_hyperparams_runs/20260712_164809/`,
searching only `q_spread`/`r_duty`) found "no improvement over baseline" in
both directions. In light of Section 5, this is now understood as
expected, not a dead end: no LQR cost-weight retuning can compensate for a
feedforward that is wrong by a large, frequency-dependent, sign-varying
factor across most of the operating range -- the correction has to happen
at the model level (Section 5.5), not the gain-tuning level. Automated
`q_spread`/`r_duty` search is expected to become *useful* only after the
feedforward fix is in place.

---

## Appendix: symbol glossary

| symbol | meaning |
|---|---|
| `x`, `i` | coil current state vector `[i_A,i_B,i_C,i_D]`, Amps |
| `u`, `d` | duty command vector, percent, `[DUTY_MIN, DUTY_MAX] = [5,100]` |
| `L`, `R` | inductance (diag), resistance/coupling (full) matrices |
| `Kv` | duty% -> effective drive voltage gain, `(4/pi)*V_supply/100` |
| `A`, `B`, `C`, `D` | continuous-time state-space matrices (`D` here means the state-space feedthrough term, distinct from channel D) |
| `Ad`, `Bd` | zero-order-hold discretized `A`, `B` at `Ts` |
| `Ts` | control tick period, 50 ms |
| `FF` | feedforward matrix, `Bd^-1` |
| `K_x, K_d, K_z` | LQID state/derivative/integral feedback gain blocks |
| `Q`, `R_cost` | LQR state and control cost weights (`q_track, q_derivative, q_integral, r_duty` in code) |
| `z_integral` | discrete integral-of-error state |
| `derr_filt` | EMA-filtered discrete derivative of tracking error |
| `omega`, `f` | angular frequency (rad/s) and drive/rotation frequency (Hz), `omega=2*pi*f` |
| `Z(omega)` | complex series R-L-C impedance |
| `f0` | resonance frequency, `1/(2*pi*sqrt(LC))` |
| `FREQ_MODEL_VALID_HZ` | integral-gate threshold, 130 Hz |
| `ALLOCATOR_R_DUTY` | duty allocator's actuation-effort regularizer weight, 0.001 |

---

## 8. Generalizing the PI controller: ratio-tracking profiles (balance + tilt)

Following Section 6's reversion to the PI controller, the next need was to
support a second experiment type ("tilt": distinct per-coil target current
RATIOS, e.g. `k=[1.0,0.5,0.3,0.8]`, on coils that -- unlike the balance
rig -- are **independently driven**, no shared supply constraint) without
duplicating `main_current_pid.cpp`'s proven control law. No hardware was
available for this work (power off); everything here is verified offline
(native host tests + Python schema validation), with hardware validation
explicitly deferred.

### 8.1 Generalizing the anchor policy to ratios

`main_current_pid.cpp`'s policy: every tick, find the argmin channel
(lowest raw current), ramp its duty toward `DUTY_MAX` (the "anchor"), and
have every other channel track the anchor's raw current `i_min` via PI+D
with asymmetric anti-windup. This exists to maximize achievable current
under a **shared** constraint -- raising the group's floor is the only way
to raise everyone's ceiling.

Generalization (`lib/RatioCurrentController`): replace every raw-current
comparison with a ratio-normalized one. Define `nrm_i = i_meas_i /
ratio_i`; the anchor is `argmin_i(nrm_i)` (hysteresis compared in the same
normalized space); the shared magnitude every channel's target is scaled
from is `m = nrm_{idxAnchor}`, so channel `i`'s target becomes `m *
ratio_i`. With `ratio_i = 1` for all `i`, `nrm_i == i_meas_i` exactly
(division by `1.0f` is exact in IEEE-754), so this reduces to the original
formulas bit-for-bit -- confirmed by a native regression test that runs
both implementations side by side over 2000 ticks of synthetic,
deliberately-adversarial input (varying `dt`, periodic overcurrent spikes,
anchor-switch-inducing crossovers) and asserts equality to `1e-4`.

### 8.2 Independent mode: no shared constraint

For rigs with independent per-channel supply/driver headroom (no
argmin/anchor mechanism applies), every channel tracks `magnitude *
ratio_i` directly via the same PI+D formula, where `magnitude` is a single
shared reference-governor scalar (structurally analogous to
`CurrentBalanceController`'s `r_target` ramp, but without that class's
feedforward/allocator machinery -- deliberately not reused, since this
controller's whole purpose is to stay as simple as the proven PI
baseline). Two bugs were found and fixed via native testing before this
converged correctly:

**Bug 1 -- ungated ramp races ahead of PI lag.** The obvious governor rule
("ramp `magnitude` up while no channel is saturated, back off on
overcurrent") is exactly what the shared-constraint anchor uses -- but
there it's safe because the anchor ramps its OWN duty directly (inherently
self-limiting: duty is clamped every tick, so it can never itself exceed
`DUTY_MAX`). In independent mode, `magnitude` is a virtual CURRENT target
that must pass through a lagged PI loop before duty catches up. Checking
"is duty saturated *yet*" is a **lagging** indicator: with the original
gate, `magnitude` overshot to ~10.5 against a ~3.0A sustainable ceiling in
a native test (`gain=0.03` A/duty%, `dutyMax=100` => ceiling `3.0A`)
*before* any channel's clamped duty ever first read `>=100`, permanently
saturating every channel (every ratio-scaled target now exceeded what's
physically achievable, with no path to unwind). **Fix**: gate the ramp-up
on convergence, not just saturation -- `magnitude` only advances when
`!anySaturatedHigh && !anyOvercurrent && maxAbsErr < magnitudeSettleTolA`
(new `Config` field), i.e. "the whole group has already caught up to the
current target" before pushing it further. This keeps `magnitude` from
ever getting more than one step ahead of what's been verified achievable,
regardless of how aggressive the PI gains are.

**Bug 2 -- the settle-gate itself can deadlock at startup.** With Bug 1's
fix alone, a *second* failure appeared: `magnitude` got permanently stuck
at its very first ramp step. Cause: `reset()` warm-starts every channel at
a fixed duty (`50.0f`, matching `main_current_pid.cpp`'s `START_DUTY` --
correct for shared-constraint mode, regression-locked), but in independent
mode `magnitude` starts at `0`, so the first few targets are tiny while
duty starts at 50% -- a large, slow-to-resolve mismatch. Lowering the
independent-mode warm-start to `dutyMin` narrowed this gap but exposed the
underlying problem: once a channel's *early, still-small* target sits
below what `dutyMin` can physically produce (e.g. `dutyMin=5%` =>
`0.15A` floor, but the first magnitude step's target is `0.1A`), duty
pins at the floor, error can never shrink below `magnitudeSettleTolA`, and
`magnitude` can never advance to raise the target past the floor --
permanent deadlock, confirmed via native test before the fix. **Fix**:
`pidStep()` now reports whether anti-windup froze the integrator this tick
(`railLocked` out-param) -- a rail-locked channel is *already doing
everything physically possible*; its error is a saturation artifact, not
evidence the group hasn't converged, so it's **excluded** from the
settle-gate's `maxAbsErr` computation. This lets `magnitude` keep climbing
past an early, transiently-infeasible target instead of deadlocking on it,
while still gating on genuine (non-rail-locked) tracking error everywhere
else.

Both bugs were found by writing a **standalone native debug harness**
(compile `RatioCurrentController.cpp` directly with `g++`, no PlatformIO/
ESP32 toolchain, print per-tick `magnitude`/`duty`/`integrator` state) --
faster iteration than round-tripping through `pio test`'s Unity harness for
root-causing, then folded back into the permanent native test suite once
each fix was confirmed.

### 8.3 Why native (host) tests, not just offline Python

`RatioCurrentController` deliberately does **not** include `constants.h`
(which pulls in `driver/gpio.h`/`<Arduino.h>`/ESP-IDF headers) -- it owns
its own `kNumChannels` constant instead, specifically so it compiles and
links on a plain host toolchain with zero ESP32 dependencies. It lives in
`lib/RatioCurrentController/src/` (not `src/`, matching this project's
existing `PWMController`/`PWMSequencer`/etc. convention) specifically
because PlatformIO's `pio test` links `lib/` automatically into both the
firmware AND a native test env, whereas files under `src/` are excluded
from test builds by default (confirmed the hard way: an initial attempt
with the class under `src/` + a `build_src_filter` override compiled
clean but silently failed to *link* against the test binary). The native
env (`[env:native_pid]`) required moving the ESP32 board/framework/lib_deps
out of the shared `[env]` section into a named `[esp32_env]` extra-section
with explicit `extends = esp32_env` per real firmware env, since a plain
`[env]` is auto-inherited by every `[env:xxx]` including the native one.

The JSON profile loader (`src/PidProfile.cpp`) is intentionally **not**
covered by native tests -- it depends on SPIFFS/Serial (ESP32-only) and is
a thin I/O wrapper around ArduinoJson, not logic worth a native port.
Its schema is validated instead by `tools/validate_pid_profile.py` (plain
Python, checks required fields/ranges/channel completeness) -- run
automatically by `tools/stage_pid_profile.py` before any file reaches the
device.

### 8.4 Deferred: hardware validation

Per explicit instruction (power off, no more hardware this session), the
following remain unverified on real coils and are the concrete follow-up
once power is back:
1. Flash `pid_profile` env with `task_sequences/pid_profile_balance.json`
   (`ratios=[1,1,1,1]`, `mode=shared_constraint`), confirm true whole-run
   peak spread matches the proven `main_current_pid.cpp` baseline
   (`0.447A`) within noise -- the real-world confirmation of Section 8.1's
   offline bit-exact equivalence claim.
2. Flash with `task_sequences/pid_profile_tilt.json`, confirm commanded
   ratios are actually achieved in measured current and the independent-
   mode governor's settle-gated ramp behaves as designed under real
   (not simulated-symmetric-gain) plant dynamics.
3. Only after both pass, consider retiring `main_current_pid.cpp`/
   `[env:current_pid]` in favor of `main_pid_profile.cpp` with the balance
   profile as default.
