# Model

Tools for modeling the magnetically-driven spinning micro flying robot.

- `frequency_modulation_piecewise_polynomial_gui.m` — build a piecewise field-frequency command and simulate the robot's spin (phase-lock) response.
- `frequency_modulation_piecewise_vertical_gui.m` — same editor, plus altitude.
- `frequency_tracking_statespace_gui.m` — same editor again, but the drive is modeled as a
  series-RLC coil, so the torque margin and step-out ceiling become *outputs*; also prints
  the linearized `A`/`B` about hover.
- `lecture_notes.md` / `.pdf` — full derivations (dynamics, moment of inertia, state-space).
- `visual_servo.ipynb` — live-camera segmentation and SVD pose estimation.
- `flyingrobot_thick _rod2.STL` — robot geometry.

## Run

In MATLAB, from this folder:

```matlab
frequency_modulation_piecewise_polynomial_gui   % or ..._vertical_gui
frequency_tracking_statespace_gui
```

Base MATLAB only (`ode45`); no toolboxes. Every field has a hover tooltip.

## Which tool

- **Polynomial** — rotational only: does the robot's spin track the commanded field frequency, or step out?
- **Vertical** — adds the heave model `L/mg = (f_robot/f_hover)^2`, so you also see altitude. Adds two fields (below).
- **State-space** — the Vertical model with the torque margin removed. Instead of assuming a
  constant `tau_max`, it derives one from the coil and magnet: `tau_max(f) = m*B_max*R/|Z(f)|`
  for a series RLC. Use it to ask whether *this* coil can actually deliver the command —
  the margin and the step-out ceiling come out as results, not assumptions — and to read off
  the linearized `A`/`B`, eigenvalues, and phase-lock mode about hover.

## Fields

You build a command as a list of segments (table rows) and set a few global parameters.

### Global settings

| Field | Meaning |
|---|---|
| Torque margin | *(polynomial, vertical)* Headroom `M = max magnetic torque / drag at the first Start freq`. `M >= 1`. Sets the step-out ceiling `f_max = f0*sqrt(M)`. |
| Tolerance (Hz) | Hold pass/fail band: a Hold passes if its last-20% tail tracking error stays within this. |
| Auto-chain adjacent segments | On: each segment's Start is forced to the previous End, for one continuous command. |
| Initial vertical velocity (m/s) | *(vertical, state-space)* velocity at `t=0`, upward positive. |
| Lift = weight frequency (Hz) | *(vertical, state-space)* hover spin frequency `f_hover`, where lift = weight. |
| B_max (mT) | *(state-space)* field amplitude **at the LC resonance**, not at every frequency. |
| Magnet m (mA m^2) | *(state-space)* combined dipole moment of both magnets. `tau_max = m*B`. Default 3.6256 assumes N52; it is an assumption, not a measurement. |
| L (mH), C (uF), R (ohm) | *(state-space)* the series-RLC coil channel. `f_res` and `Q` are shown live next to them. |

### Segment table

| Column | Meaning |
|---|---|
| Type | `Hold`, `Polynomial`, or `Exponential`. |
| Start (Hz), End (Hz) | Segment endpoints. Hold ignores End. |
| Duration (s) | Segment length (> 0). |
| Order / exp k | Polynomial order (integer `>=1`: 1 linear, 2 quadratic, 3 cubic) or exponential curvature `k` (0 = linear). Ignored for Hold. |

Segment shapes, over normalized time `s = (t - t0)/duration` in `[0,1]`:

- **Hold** — constant at Start.
- **Polynomial** — `Start + (End-Start)*s^n`.
- **Exponential** — `Start + (End-Start)*(1-exp(-k*s))/(1-exp(-k))`, reaching End exactly.

### Buttons

`+ Hold / + Polynomial / + Exponential` add a segment. `Copy`, `Delete`, `Clear All`, `Move Up/Down` edit the table. `Preview Command` plots the command only; `Run Simulation` integrates the dynamics and reports stats. `Reset Example` restores defaults.

### Fixed-model readouts

`I_robot` (spin-axis inertia), `k_drag` (from `tau_drag = -k*f^2`), and the fit `R^2`. Fixed constants — see `lecture_notes.md` §3–§4.

### Outputs

Command vs. robot frequency; wrapped phase `delta = theta_field - theta_robot`; *(vertical)* altitude; and a results box with tracking error, torque, net phase turns, and per-Hold pass/fail.

## Dynamics

```
delta_dot   = 2*pi*f_field(t) - omega_robot
I*omega_dot = tau_max*sin(delta) - k_drag*f_robot*|f_robot|
```

Vertical adds `z_ddot = g*((f_robot/f_hover)^2 - 1)`. Derivations: `lecture_notes.md` §5 (swing equation, step-out), §6 (heave).

State-space carries the same equations as `x = [delta; omega; z; z_dot]`, `u = f_field`, and
replaces the constant `tau_max` with the coil's frequency response:

```
X(f)       = 2*pi*f*L - 1/(2*pi*f*C)
tau_max(f) = m*B_max * R/sqrt(R^2 + X(f)^2)
```

so the drive weakens away from resonance and vanishes at DC (the series cap blocks it) —
which is what limits spin-up from rest. Because `tau_max` now moves with the command, the
step-out ceiling has no closed form and is found numerically. Derivation: §7–§8.

## Caveats

- Command must stay under `f_max = f0*sqrt(M)`, or the robot steps out. *(State-space: under
  the numerically-found ceiling instead — and if `Effective margin at hover` prints below 1,
  the coil cannot hold hover at all.)*
- Tolerance checks the Hold tail only.
- Auto-chain overrides each segment's Start.
- Vertical model ignores vertical drag and ground contact.
