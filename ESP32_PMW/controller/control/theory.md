# Chapter 4. Control: pose in, coil commands out

*Stage 4 of the pipeline. Consumes: a position fix from
[chapter 3](../pose/theory.md), via `CameraSource`. Produces: field frequency and
lateral commands over serial to `src/main_flight.cpp`.*

*Two plants live here. Sections 1–11 are the altitude model that flies today: one input, one
axis, a uniform field. Sections 12–15 are the three-dimensional model behind the coil-array
GUI, which shares the robot but not much else, and which turns out to consume exactly the five
degrees of freedom the vision estimator produces.*

The plant is a spinning magnetic rotor, and the single most important fact about
it is that **frequency is the throttle**: lift scales with spin rate, and spin
rate is set by how fast the external field rotates. Everything here follows from
that one lever, and from the fact that the robot must stay phase-locked to the
field while you move it.

## Reading order

| # | file | what it does |
|---|---|---|
| 1 | `hover_model.py` | the plant: inertia, drag, thrust. Sections 3-7 of this chapter |
| 2 | `design_hover_lqr.py` | linearise, discretise, solve the DARE -> `hover_controller.json`. Section 8 |
| 3 | `simulate_hover.py` | closed loop against the *nonlinear* truth plant |
| 4 | `hover_controller_runner.py` | the real-time runner, and `CameraSource`: the seam |
| 5 | `z_track.py` | altitude tracking by frequency modulation; the torque budget |
| 6 | `link.py` | `SerialComm`: the host half of the firmware link |
| 6b | `takeoff_report.py` | per-attempt takeoff metrics: liftoff knee, capture verdict, L/C refit. Section 18 |
| 6c | `robust_cert.py` | exact gain/phase margins of the shipped gains, over mass and COM. Section 20 |
| 7 | `ai/design/spatial_model.py` | the 3-D plant: multi-coil field, spin axis, gradient force. Section 12 |
| 8 | `ai/design/spatial_mpc.py` | MPC over that plant, on the 5-DOF pose. Section 13 |
| 9 | `ai/design/simulate_spatial.py` | closed loop and the real-time viewer (`--live`) |
| 10 | `ai/design/open_loop_sweep.py` | is any fixed-current array passively stable? Section 14 |
| 12 | `ai/design/coil_geometry.py`, `drive_model.py` | array geometry and the drive. Section 15 |
| 13 | `ai/design/stability_cert.py`, `optimise_array.py` | certify one design, then search. Section 15 |
| 14 | `ai/matlab/` | the original interactive sims these models were ported from |

Sections 1–11 are the scalar altitude plant, one input and one axis. Sections 12–15 are the
three-dimensional one, and are self-contained apart from the inertia in section 3 and the
tilt geometry in section 11.

## 4.0 Safety comes before theory

**As of 2026-08-29 there is no watchdog anywhere. Nothing stops the coils on its
own.** The only things that de-energise them are the viser `stop` button,
`controller/control/safe_off.py`, and the GPIO14 reset button on the board. If the
host dies, the USB is unplugged, or a camera hangs, the coils stay driven until
someone presses that button; the bench supply's 10 A limit is the only other ceiling.

This was deliberate. Every automatic backstop was removed at the operator's
instruction: the firmware's 500 ms command-silence watchdog (reported not to work in
practice), the host's no-position-fix land, the armed-with-no-fix land, the step-out
land, and the never-reached-FLIGHT land. What remains is manual: the `stop` and `land`
buttons, SIGINT, and `stop` from a `finally` on every exit path including an unhandled
exception -- which every entry point that energises the coils still owes.

**The consequence is that the operator's stop button is now the safety system**, so it
has to be correct in a way it was not. The panel's `takeoff`/`estop`/`land` flags are
one-shot latches cleared on read, and the loop used to read them several times per tick
behind short-circuits -- which is why stop did nothing during spin-up, why a takeoff
press was sometimes swallowed, and why the coils sometimes restarted unasked. The rule
that fixes all of it, and the thing not to regress: **read each latch exactly once per
tick, into a local, at the top of one loop.** `controller/control/test_panel.py` is the
regression guard.

The original argument for the watchdogs still stands and is worth recording, since it
is the case for putting them back: the failure mode of a vision-driven loop is not a
crash, it is a *silence*, and a silent position source looks exactly like a perfectly
stationary robot.

## 4.0.1 The altitude law

`z_track.ZTracker` is a **positional** controller that inverts the lift law exactly,
$f = f_{hover}\sqrt{1 + a/g}$, with conditional anti-windup. It replaced an
incremental (velocity-form) PID with an asymmetric rate limit derived from the
torque budget; inverting the law is the better altitude controller because the
nonlinearity is known in closed form rather than something a gain has to absorb.

## 0. System description and modeling assumptions

The robot is a millimeter-scale rotor: a magnetized central body carrying four blades. A
**rotating external magnetic field** (generated by a coil array) drags the robot's in-plane
magnetization around; the resulting spin drives the blades, which produce aerodynamic
**thrust** (lift) and **drag torque**. Modulating the field's rotation frequency modulates
spin speed, hence thrust, hence altitude.

**Assumptions (A1–A7):**

- **A1: Coaxial alignment.** The robot's spin axis coincides with the field's rotation
  axis ($\hat z$). Justified by gyroscopic stiffness at 100–230 Hz spin rates and by the
  restoring out-of-plane magnetic torque, which makes the aligned state an attractor.
- **A2: Rigid body.** Blades do not flex; inertia is constant.
- **A3: Uniform field over the robot.** The coil array's working volume is
  Helmholtz-like: the field applies pure torque (no translational magnetic force, since
  force couples to $\nabla B \approx 0$), and the torque amplitude $\tau_{max}$ is
  position-independent.
- **A4: Linear magnetics.** Air-core coils (or cores well below saturation):
  $B \propto I_{coil}$, so $\tau_{max}$ is a constant set by the drive amplitude.
- **A5: Quadratic aerodynamics.** Blade-element Reynolds number is in the pressure-drag
  regime ($Re \sim 10\text{-}100$), so local aerodynamic forces scale with the *square*
  of local airspeed. Validated empirically (§4.3, $R^2$ reported by the GUI).
- **A6: Hover-referenced aerodynamic coefficients.** Drag and thrust coefficients are
  those measured/fitted near hover; inflow corrections for climb appear only as the
  first-order heave damping term (§6.3).
- **A7: Rigid stereo mount.** The two cameras hold their relative pose across both
  calibration and use ([§14](../calib/theory.md) (ch.2)). Introduced with the extrinsic calibration; irrelevant to
  §1–[§13](../pose/theory.md) (ch.3), which are monocular or camera-free.

---

## 1. Notation and unit conventions

| Symbol | Meaning | Units |
|---|---|---|
| $\theta_f,\ \theta_r$ | field / robot angle in the rotation plane (from an arbitrary fixed lab direction) | rad |
| $\delta = \theta_f - \theta_r$ | phase lag of robot behind field | rad |
| $\omega = \dot\theta_r$ | robot angular velocity | rad/s |
| $f = \omega/2\pi$ | robot spin frequency | Hz |
| $f_f(t)$ | commanded field rotation frequency (**the control input**) | Hz |
| $I$ | robot moment of inertia about spin axis | kg·m² |
| $\tau_{max} = mB$ | peak magnetic torque (dipole moment × field) | N·m |
| $k_d$ | drag-torque coefficient ($\tau_{drag} = -k_d f^2$) | N·m/Hz² |
| $k_T$ | thrust coefficient ($T = k_T f^2$) | N/Hz² |
| $M$ | torque margin $= \tau_{max}/(k_d f_0^2)$ at start frequency $f_0$ | – |
| $z,\ w=\dot z$ | altitude, climb rate | m, m/s |
| $m_R,\ g$ | robot mass, gravity | kg, m/s² |

**Radian/Hz convention.** Physics (Newton's law, torques, the ODE state) lives in
radians; commands, data fits, and plots live in Hz. Every factor of $2\pi$ marks a
border crossing: $\omega = 2\pi f$, $\dot\omega = 2\pi\dot f$.

---

## 2. Rotational kinematics: the phase variable

Fix lab axes with $\hat z$ along the field's rotation axis and any reference direction
$\hat x$ in the plane. The field vector and the robot's in-plane magnetization are

$$\mathbf B = B\,[\cos\theta_f,\ \sin\theta_f,\ 0], \qquad
  \mathbf m = m\,[\cos\theta_r,\ \sin\theta_r,\ 0].$$

Only the difference $\delta = \theta_f - \theta_r$ enters the dynamics (the reference
direction is arbitrary), so the natural kinematic chain is

$$\delta = \theta_f - \theta_r,\qquad
  \dot\delta = 2\pi f_f - \omega,\qquad
  \ddot\delta = 2\pi\dot f_f - \dot\omega .$$

$\dot\delta$ is the rate at which the gap opens; $\ddot\delta$ is the *relative* angular
acceleration (field minus robot). Working in $\delta$ is equivalent to riding in the
field's rotating frame and watching the robot swing relative to you.

---

## 3. Moment of inertia

From the CAD tensor in `newRobotPhysicalParams()`, ported as `spatial_model.robot_params()`.
About the total COM at $[1.927, 0, 0]$ mm:

$$I = \mathrm{diag}\big(3.3578,\; 2.0373,\; 2.0369\big)\times10^{-9}\ \text{kg}\cdot\text{m}^2$$

$I_s = 3.3578\times10^{-9}$ kg·m² is the polar moment about the spin axis (body $x$).
$I_t = 2.0373\times10^{-9}$ kg·m² is the transverse moment about a diameter. The rotor is
transversely symmetric to 0.02 %. Their ratio $I_t/I_s = 0.607$ is what §11.3 uses, and §9
tabulates both alongside $m_R$ and $m_{dip}$.

The central body sets $I_s$. The two magnets contribute 0.23 % of it, so $I_s$ tolerates a
wide error in their geometry. $m_{dip}$ does not, and the same CAD numbers set it.

> **Check the magnet CAD before trusting $m_{dip}$.** The two magnet cylinders sit 0.70 mm
> apart on one axis and each is 0.794 mm long, so as described they interpenetrate by
> 0.094 mm. Either the cylinder axis is not CAD $z$, or the spacing and length disagree.
> $m_{dip}$ scales every magnetic torque in §12.

---

## 4. Torques on the robot

### 4.1 Magnetic drive torque

A dipole in a field experiences $\boldsymbol\tau = \mathbf m \times \mathbf B$. With both
vectors in-plane (A1), the component along $\hat z$ is

$$\tau_z = m B \sin(\theta_f - \theta_r) = \tau_{max}\sin\delta .$$

Properties that shape the whole problem:

- **Restoring up to 90°:** more lag → more torque, but only until $\delta = \pi/2$ where
  torque peaks at $\tau_{max}$. Beyond 90° torque *falls* with lag → loss of
  synchronization ("step-out", pole slipping) becomes possible.
- **Linear in coil current** (A4): the coil array obeys superposition,
  $\mathbf B(p) = A(p)\,\mathbf I_{coil}$, so $\tau_{max} \propto I_{coil}$. Distance
  enters through the field map: approximately constant inside the designed uniform
  region, $\propto 1/r^3$ (dipole far field) outside; never linear in $r$.

**Margin parameterization.** Instead of specifying $B$ in tesla, define the margin at
the initial frequency $f_0$:

$$\tau_{max} = M \cdot k_d f_0^2, \qquad M \ge 1 .$$

$M$ measures torque headroom over the drag that must be overcome to spin at $f_0$.

### 4.2 Aerodynamic drag torque

Blade-element drag in the pressure-drag regime (A5) scales as the square of local airspeed,
and airspeed is proportional to $f$. Integrating $r\,dF$ over the blades collapses all
geometry and air properties into one coefficient:

$$\tau_{drag} = \tfrac12 \rho C_D (2\pi f)^2 \int_{blades} c(r)\, r^3\, dr
  \;\equiv\; k_d\, f^2 .$$

$k_d$ has units N·m/Hz². The dynamics write $-k_d f\,|f|$ so the torque always opposes
rotation.

### 4.3 Fitting $k_d$ from data

Through-origin least squares on the code's 23 measured points, 10–230 Hz, against
$\tau = -k_d f^2$:

$$k_d = \frac{\sum_i f_i^2\,(-\tau_i)}{\sum_i f_i^4}
  \;=\; 3.91\times10^{-10}\ \text{N}\cdot\text{m/Hz}^2 .$$

$R^2$ is the empirical check on the quadratic form itself: near unity justifies A5.

---

## 5. Phase (synchronization) dynamics

### 5.1 The two-state model

Newton's second law for rotation plus the phase kinematics give the model the GUI
integrates (the `rotationODE` anonymous function in `runSimulation`):

$$\boxed{\;
\begin{aligned}
\dot\delta &= 2\pi f_f(t) - \omega \\[2pt]
I\,\dot\omega &= \tau_{max}\sin\delta \;-\; k_d\,\frac{\omega}{2\pi}\left|\frac{\omega}{2\pi}\right|
\end{aligned}\;}$$

### 5.2 The swing equation

Eliminate $\omega = 2\pi f_f - \dot\delta$ using $\dot\omega = 2\pi\dot f_f - \ddot\delta$:

$$I(2\pi \dot f_f - \ddot\delta) = \tau_{max}\sin\delta - k_d\Big(f_f - \frac{\dot\delta}{2\pi}\Big)^2 .$$

Expanding the drag term to first order in $\dot\delta$ (robot near command speed),
$k_d(f_f - \dot\delta/2\pi)^2 \approx k_d f_f^2 - (k_d f_f/\pi)\,\dot\delta$, and
rearranging:

$$\boxed{\;
I\,\ddot\delta \;+\; \underbrace{\frac{k_d f_f}{\pi}}_{c\ \text{(damping)}}\dot\delta
\;+\; \underbrace{\tau_{max}\sin\delta}_{\text{restoring}}
\;=\; \underbrace{2\pi I \dot f_f}_{\text{inertial demand}}
\;+\; \underbrace{k_d f_f^2}_{\text{drag load}}
\;}$$

This is the **swing equation**: identical in form to a synchronous machine on a power
grid or a torque-driven pendulum. The right-hand side is the torque the field must
supply for perfect tracking; the robot "pays" for it by settling at whatever lag
$\delta$ makes $\tau_{max}\sin\delta$ match the demand.

### 5.3 Equilibria and their stability

For constant $f_f$ ($\dot f_f = 0$), equilibria satisfy

$$\sin\delta_{eq} = \frac{k_d f_f^2}{\tau_{max}} = \frac{1}{M}\Big(\frac{f_f}{f_0}\Big)^{\!2}
  \;\equiv\; s_e \in (0, 1].$$

Two solutions per revolution: $\delta_1 = \arcsin s_e$ and $\delta_2 = \pi - \delta_1$.
Linearizing the $(\delta, \omega)$ system, the Jacobian at an equilibrium is

$$J = \begin{bmatrix} 0 & -1 \\ \kappa/I & -c/I \end{bmatrix},
\qquad \kappa = \tau_{max}\cos\delta_{eq},
\qquad \lambda^2 + \tfrac{c}{I}\lambda + \tfrac{\kappa}{I} = 0 .$$

- $\delta_1 < \pi/2$: $\kappa > 0$ → **stable focus** (both eigenvalues in the left
  half-plane; complex because damping is weak). The operating point.
- $\delta_2 > \pi/2$: $\kappa < 0$ → **saddle**. Its stable manifold (the separatrix)
  bounds the basin of attraction; trajectories crossing it slip a pole ($\delta$ advances
  by $2\pi$). This is the geometric picture of step-out.

The GUI starts the simulation *at* the stable equilibrium:
$\delta(0) = \arcsin(1/M)$, $\omega(0) = 2\pi f_0$ (`deltaInitial` / `stateAtStart` in
`runSimulation`), so all observed transients are caused by the commanded maneuver.

### 5.4 Step-out ceiling

Lock requires $s_e \le 1$:

$$\boxed{\ f_{f} \le f_{max} = f_0\sqrt{M}\ }
\qquad (M{=}5,\ f_0{=}160\ \text{Hz} \Rightarrow f_{max} \approx 358\ \text{Hz}).$$

The margin is *not* constant: it erodes quadratically with commanded frequency:
$M_{eff}(f_f) = M (f_0/f_f)^2$.

### 5.5 Small-signal behavior: natural frequency and damping

Perturbing $\delta = \delta_{eq} + \varepsilon$ about the stable focus:

$$I\ddot\varepsilon + c\,\dot\varepsilon + \kappa\,\varepsilon = 0,
\qquad
\omega_n = \sqrt{\frac{\tau_{max}\cos\delta_{eq}}{I}},
\qquad
\zeta = \frac{c}{2\sqrt{I\,\tau_{max}\cos\delta_{eq}}} .$$

With the file's numbers ($M=5$, $f_0=160$ Hz → $\tau_{max} = 5.00\times10^{-5}$ N·m,
$\delta_{eq} = 11.5^\circ$) and the CAD inertia of §3:

$$\omega_n \approx 121\ \text{rad/s} \;(\approx 19.2\ \text{Hz}),
\qquad
\zeta \approx 0.0245 \;(Q \approx 20),
\qquad
t_{settle} \approx \frac{4}{\zeta\omega_n} \approx 1.35\ \text{s}.$$

**Interpretation:** the phase lock is a lightly-damped 19 Hz oscillator. Aerodynamic
drag provides almost no phase damping, so every ramp ending rings for ~20 cycles. This
is why the solver needs `MaxStep ≤ 2e-4 s` and why Hold segments need $\gtrsim 1.4$ s
to settle within tolerance.

### 5.6 Ramp feasibility (quasi-static torque budget)

Perfect tracking of a ramp requires the drive-side torque to fit under $\tau_{max}$:

$$2\pi I\,\dot f_f + k_d f_f^2 \le \tau_{max}
\quad\Longrightarrow\quad
\boxed{\ \dot f_{max}(f_f) = \frac{\tau_{max} - k_d f_f^2}{2\pi I}\ }$$

At $f_0 = 160$ Hz, $M = 5$: $\dot f_{max} \approx 1.9$ kHz/s: far above the ~100 Hz/s
ramps in the example schedule. The *practical* limit is usually dynamic, not
quasi-static: an abrupt ramp end deposits energy into the $Q \approx 20$ phase
oscillation, and if the swing peak carries $\delta$ across the separatrix (§5.3) the
robot steps out below $\dot f_{max}$. Smooth blends (higher-order polynomial,
exponential segments) exist precisely to taper $\dot f_f$ at boundaries.

---

## 6. Vertical (heave) dynamics

### 6.1 Blade-element thrust

The same element as §4.2 produces lift perpendicular to the local flow, giving

$$T = \tfrac12 \rho C_L (2\pi f)^2 \int c(r)\, r^2\, dr \;\equiv\; k_T f^2 .$$

Thrust integrates $r^2 c\,dr$ (force) while drag torque integrates $r^3 c\,dr$ (force ×
arm), so both scale as $f^2$. **Thrust and drag torque are inseparable**: their ratio
$T/\tau_{drag} = (C_L/C_D)(\int c r^2)/(\int c r^3)$ is fixed by geometry and independent
of $f$. Rotor power follows, $P = \tau_{drag}\,\omega \propto f^3$.

### 6.2 Vertical equation of motion and hover

Newton in $z$ (up positive):

$$m_R \ddot z = k_T f^2 - m_R\, g - D_z(\dot z).$$

Hover ($\ddot z = \dot z = 0$):

$$\boxed{\ f_h = \sqrt{\frac{m_R\, g}{k_T}}\ }
\qquad\Longleftrightarrow\qquad k_T = \frac{m_R\,g}{f_h^2}\ \text{(calibration from a hover test)}.$$

### 6.3 Heave damping from inflow

Climbing at $\dot z$ tilts the inflow angle at each blade element and cuts the angle of
attack, so thrust falls with climb rate. With lift slope $a = dC_L/d\alpha$:

$$\frac{\partial T}{\partial \dot z} = -\tfrac12 \rho\, a\,(2\pi f)\int c(r)\, r\, dr
\;\equiv\; -k_w f \;<\; 0 .$$

Heave therefore has **natural aerodynamic damping proportional to spin frequency**. Spin
rotors get altitude-rate stability for free:

$$\boxed{\ m_R \ddot z = k_T f^2 - m_R g - k_w f\,\dot z\ }$$

The same inflow effect perturbs blade drag, making $k_d$ weakly climb-dependent. That is
the price of assumption A6.

### 6.4 Linearization about hover: frequency → altitude

Let $f = f_h + \Delta f$, keep first order:

$$\Delta T = 2 k_T f_h\,\Delta f = \frac{2 m_R g}{f_h}\,\Delta f
\quad\Longrightarrow\quad
\ddot z = \frac{2g}{f_h}\,\Delta f - \frac{k_w f_h}{m_R}\,\dot z .$$

**The frequency command is the altitude control input with gain $2g/f_h$.**
At $f_h = 160$ Hz: $2g/f_h \approx 0.12\ \text{(m/s}^2)$ per Hz. The example schedule's
±20 Hz steps command ≈ ±2.5 m/s² (a quarter-g). As a transfer function, with heave time
constant $\tau_h = m_R/(k_w f_h)$:

$$\frac{\Delta z(s)}{\Delta f(s)} = \frac{2g/f_h}{s\,(s + 1/\tau_h)} .$$

---

### 6.5 Sensitivity

From 6.2, $T = k_T f^2$ and $m_R g = k_T f_h^2$, so the heave law carries $f_h$ as a
parameter:

$$\ddot z = g\left(\frac{f^2}{f_h^2} - 1\right).$$

Two partials at hover, $f = f_h$:

$$\frac{\partial \ddot z}{\partial f} = \frac{2g}{f_h},
\qquad
\frac{\partial \ddot z}{\partial f_h} = -\frac{2g}{f_h}.$$

They are equal and opposite. A relative error $\varepsilon$ in the parameter is therefore
indistinguishable from a relative error $-\varepsilon$ in the command, and no measurement of
$z$ separates them. Exactly, under $f_h \to f_h(1+\varepsilon)$,

$$\ddot z = g\left(\frac{f^2}{f_h^2(1+\varepsilon)^2} - 1\right),$$

so holding hover needs $f = f_h(1+\varepsilon)$: **the command carries the parameter error in
full.** The heave gain $2g/f_h$ is the exchange rate in both directions, which fixes the
precision to which $f_h$ must be known: to keep $|\ddot z|$ below some $a$, the relative error
must satisfy $|\varepsilon| \le a/(2g)$.

### 6.6 Frequency trim

Write the inverted law with an estimate $\hat f$ in place of $f_h$:

$$f_{cmd} = \hat f \sqrt{1 + \frac{a}{g}}, \qquad a \ge -g.$$

A parameter error can be absorbed in $a$ or in $\hat f$. The two are not equivalent.

**Trim in acceleration.** Hold $\hat f$ fixed and let $a$ carry it. Hovering against a true
$f_h(1+\varepsilon)$ requires $f_{cmd} = \hat f(1+\varepsilon)$, hence a constant

$$a_{bias} = g\left[(1+\varepsilon)^2 - 1\right] = g\left(2\varepsilon + \varepsilon^2\right).$$

This is permanent, and it is spent inside the actuator's own limits: with $a$ clamped to
$[-g,\,a_{max}]$ the usable upward range falls to $a_{max} - a_{bias}$, and the slew bound
$\dot a_{max}$ is reached $a_{bias}$ earlier in the climbing direction. Whatever integral gain
produces $a_{bias}$ sets how fast it arrives, never its size: the cost follows from where the
trim is stored, not from how it is tuned.

**Trim in frequency.** Let $\hat f$ carry the parameter and $a$ carry feedback alone:

$$a_{fb} = k_p e + k_v(\dot z_{ref} - \dot z), \qquad \dot{\hat f} = \gamma e,
\qquad e = z_{ref} - z.$$

At equilibrium $e = 0$, so $a_{fb} = 0$ and $\hat f \to f_h$ whatever $f_h$ is. The clamp range
is preserved in full and $a_{bias}$ does not exist.

For the estimator dynamics, expand about hover with $a_{fb}$ small. Writing
$\delta = \hat f - f_h$ and $f_{cmd} \approx \hat f\,(1 + a_{fb}/2g)$,

$$\ddot z \approx a_{fb} + \frac{2g}{f_h}\,\delta,$$

so $\delta$ enters as a disturbance acceleration through the same gain $2g/f_h$. With
$z_{ref} = 0$ the closed loop is third order,

$$\dddot z + k_v \ddot z + k_p \dot z + \frac{2g\gamma}{f_h} z = 0,$$

whose slow root, for small $\gamma$, is the trim pole

$$s \approx -\frac{2g\gamma}{f_h\,k_p}.$$

The position loop has $\omega_n = \sqrt{k_p}$ and $\zeta = k_v/2\sqrt{k_p}$. Keeping the trim
pole well below $\omega_n$ decouples the two: the estimator sees a settled position loop, and
the position loop sees a constant parameter.

---

## 7. The coupled nonlinear model

Stack the rotational and vertical dynamics. Thrust responds to the robot's **actual**
spin frequency $\omega/2\pi$ (not the command): the physically correct coupling:

**State** $x = [\delta,\ \omega,\ z,\ w]^\top$, **input** $u = f_f(t)$ (Hz):

$$\boxed{\;
\begin{aligned}
\dot\delta &= 2\pi u - \omega \\[2pt]
\dot\omega &= \frac{1}{I}\left[\tau_{max}\sin\delta - \frac{k_d}{4\pi^2}\,\omega|\omega|\right] \\[4pt]
\dot z &= w \\[2pt]
\dot w &= \frac{1}{m_R}\left[\frac{k_T}{4\pi^2}\,\omega^2 - m_R g - \frac{k_w}{2\pi}\,\omega\, w\right]
\end{aligned}\;}$$

**Cascade structure.** $(\delta, \omega)$ evolve independently of $(z, w)$; the vertical
subsystem is driven by $\omega$ alone. The timescales stack:

$$\underbrace{\text{phase lock}}_{\sim 18\ \text{Hz},\ \zeta \approx 0.02}
\;\gg\;
\underbrace{\text{rotor speed ramps}}_{\sim 1\text{-}10\ \text{Hz}}
\;\gg\;
\underbrace{\text{heave}}_{\sim 0.1\text{-}1\ \text{Hz}}$$

This separation legitimizes designing the layers independently: the altitude planner
treats $f$ as directly commandable *provided* it respects the inner loop's limits
($f \le f_0\sqrt{M}$, $\dot f \le \dot f_{max}$). The MATLAB GUI simulates exactly the
inner two states of this model.

---

## 8. Linearized state-space model

### 8.1 Trim point

Hover at frequency $f_h$ with the field commanding $u^{*} = f_h$:

$$x^{*} = \big[\ \delta^{*} = \arcsin\!\tfrac{k_d f_h^2}{\tau_{max}},\quad
\omega^{*} = 2\pi f_h,\quad z^{*} = \text{any},\quad w^{*} = 0\ \big]^\top .$$

### 8.2 Jacobians

Define perturbations $\tilde x = x - x^{*}$, $\tilde u = u - u^{*}$, and the shorthand

$$\kappa = \tau_{max}\cos\delta^{*} \ \ (\text{phase stiffness}),\qquad
c = \frac{k_d f_h}{\pi}\ \ (\text{phase damping}),\qquad
\tau_h = \frac{m_R}{k_w f_h}\ \ (\text{heave lag}).$$

Differentiating the §7 model at trim, with $k_T f_h^2 = m_R g$:

$$\boxed{\ \dot{\tilde x} = A\,\tilde x + B\,\tilde u,\qquad y = C\,\tilde x\ }$$

$$A = \begin{bmatrix}
0 & -1 & 0 & 0 \\[2pt]
\dfrac{\kappa}{I} & -\dfrac{c}{I} & 0 & 0 \\[8pt]
0 & 0 & 0 & 1 \\[2pt]
0 & \dfrac{g}{\pi f_h} & 0 & -\dfrac{1}{\tau_h}
\end{bmatrix},
\qquad
B = \begin{bmatrix} 2\pi \\ 0 \\ 0 \\ 0 \end{bmatrix},
\qquad
C = \begin{bmatrix} 0 & \tfrac{1}{2\pi} & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

with outputs $y = [\,f_{robot},\ z\,]^\top$ and $D = 0$. The lower-left block entry
$g/(\pi f_h)$ is the thrust sensitivity to actual spin speed; the zero column 3 reflects
that nothing depends on absolute altitude.

### 8.3 Eigenvalues (modes)

The block-triangular structure factors the characteristic polynomial:

$$\det(sI - A) = \underbrace{\Big(s^2 + \tfrac{c}{I}s + \tfrac{\kappa}{I}\Big)}_{\text{phase-lock mode}}
\cdot \underbrace{\Big(s + \tfrac{1}{\tau_h}\Big)}_{\text{heave mode}}
\cdot \underbrace{s}_{\text{altitude integrator}}$$

$$\lambda_{1,2} = -\zeta\omega_n \pm j\,\omega_n\sqrt{1-\zeta^2} \approx -2.96 \pm 121\,j\ \text{s}^{-1},
\qquad
\lambda_3 = -\frac{1}{\tau_h},
\qquad
\lambda_4 = 0 .$$

- $\lambda_{1,2}$: the 19.2 Hz, $\zeta \approx 0.025$ phase ringing (visible in every GUI run).
- $\lambda_3$: climb-rate settling, aerodynamic (inflow) damping.
- $\lambda_4$: altitude is an integrator: no feedback, marginally stable, as expected
  for open-loop height.

### 8.4 Input–output transfer functions

Command → robot frequency (from rows 1–2):

$$\frac{\tilde f_{robot}(s)}{\tilde u(s)} = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}
\qquad (\text{DC gain } 1:\ \text{the robot tracks the command exactly at steady state}).$$

Command → altitude (full cascade):

$$\frac{\tilde z(s)}{\tilde u(s)} =
\frac{2g}{f_h}\cdot
\frac{\omega_n^2}{\big(s^2 + 2\zeta\omega_n s + \omega_n^2\big)\big(s + 1/\tau_h\big)\, s}.$$

Steady climb rate per Hz of frequency offset:
$w_{ss}/\Delta f = (2g/f_h)\,\tau_h = 2 m_R g /(k_w f_h^2)$.

### 8.5 Structural properties

- **Controllability:** single input $u$ enters only $\dot\delta$, but the chain
  $u \to \delta \to \omega \to w \to z$ makes $(A, B)$ controllable for
  $\kappa, g, f_h \ne 0$ (generic hover). The robot is *underactuated*, one input,
  four states, but the cascade renders altitude controllable through frequency modulation.
- **Observability:** with $y = [f_{robot}, z]$, all four states are observable
  ($\delta$ is reconstructed from $\dot f_{robot}$ via the second state equation).
- **Validity boundary:** the linear model holds while (i) $\delta$ stays well inside
  the separatrix (no step-out: $f \le f_0\sqrt M$, $\dot f \le \dot f_{max}$),
  (ii) magnetics stay linear (below core saturation, drive near resonance so
  $\tau_{max}$ is frequency-flat), and (iii) climb rates are small enough that the
  hover-referenced $k_d, k_T$ apply.

---

## 9. Parameter summary (values from the code / defaults)

| Parameter | Value | Source |
|---|---|---|
| $I_s$ | $3.3578\times10^{-9}$ kg·m² | §3: CAD body $I_0$ + two magnets (one cylinder element each) |
| $I_t$ | $2.0373\times10^{-9}$ kg·m² ($I_t/I_s = 0.607$) | §3, CAD tensor |
| $m_R$ | $8.3566\times10^{-5}$ kg (819.8 µN) | §3, CAD |
| $m_{dip}$ | $3.6257\times10^{-3}$ A·m² | §3: two N52 cylinders, $2(B_r/\mu_0)V$ |
| $k_d$ | $3.91\times10^{-10}$ N·m/Hz² | least-squares fit, lines 26–54 |
| $M$ (default) | 5 | GUI "Torque margin" |
| $f_0$ (example) | 160 Hz | example schedule |
| $\tau_{max}$ | $5.00\times10^{-5}$ N·m | $M k_d f_0^2$ |
| $\delta_{eq}$ | $11.5^\circ$ | $\arcsin(1/M)$ |
| $\omega_n$ | 121 rad/s (19.2 Hz) | §5.5 |
| $\zeta$ | 0.0245 ($Q \approx 20$) | §5.5 |
| $t_{settle}$ | ≈ 1.35 s | $4/\zeta\omega_n$ |
| $f_{max}$ | ≈ 358 Hz | $f_0\sqrt M$ |
| $\dot f_{max}(f_0)$ | ≈ 1.9 kHz/s | §5.6 |
| $2g/f_h$ | ≈ 0.12 (m/s²)/Hz at $f_h{=}160$ Hz | §6.4 |
| $k_T,\ k_w$ | symbolic: calibrate via hover test ($k_T = m_R g/f_h^2$) | §6.2 |

Multi-coil array (§12), the `cross` preset at its GUI defaults:

| Parameter | Value | Source |
|---|---|---|
| coils | 16, as 4 channels of $2\times2$ | $R = 10.5$ mm, 650 turns, 1.25 A |
| channel centres | 37 mm radius, $2\times2$ pitch 21 mm | phases 0/270/180/90° |
| $\lvert B\rvert$ on axis | 4.58 mT at $z = 15.2$ mm | §12.1 |
| channel authority | rank 3, $\kappa = 3$–7 | §12.7 |
| field maximum | $z = 15.156$ mm on the axis | §12.6 |
| lock floor at 110 Hz | $\lvert B\rvert \ge 1.00$ mT (circular) | §12.3 |
| gradient-force stiffness | $\mathrm{diag}(+0.054, +0.054, -0.107)$ N/m | §12.6 |

---

## 10. Limitations and extensions

1. **Rotation-only attitude.** A1 removes tilt dynamics. Steering (tilting the field's
   rotation plane) and righting maneuvers (e.g. inverted takeoff) need full rigid-body
   attitude dynamics: Euler's equations + 3-D $\mathbf m \times \mathbf B$, where
   $\theta_f, \theta_r$ are no longer scalars. These two tilt DOF, and the precession
   they produce on a fast-spinning rotor, are developed in §11, and §12 builds the
   executable model.
2. **Constant $\tau_{max}$.** Real drives (resonant LC coil drive, cored coils) have
   frequency- and amplitude-dependent field: replace $\tau_{max} \to \tau_{max}(f_f)$
   from a measured current-vs-frequency sweep.
3. **Hover-referenced aerodynamics.** Aggressive climb changes inflow → $k_d(f, \dot z)$,
   $k_T(f, \dot z)$ from momentum theory; the swing equation's drag load then couples to
   the vertical state, breaking the one-way cascade weakly.
4. **Torque ripple.** Field harmonics (core saturation flat-topping, non-ideal coil
   phasing) inject ripple at $2f, 4f, \dots$; components landing near $\omega_n$ are
   amplified $Q \approx 20$ times.
5. **No lateral dynamics.** Position in the plane and precession-based translation (§11.5)
   are outside *this* model; §12 drops A3 and adds them, at the cost of a 16-coil field map.
   Wall and ground aerodynamic effects are outside both.

---

## 11. Tilt and precession dynamics (beyond A1)

§10.1 and §10.5 defer the two attitude DOF that A1 discards: the **tilt** of the robot's
spin axis $\hat n$ away from the field's rotation axis $\hat z$. On a body spinning at
100–230 Hz this tilt does not simply relax back to level: it **precesses**. This section
derives the tilt subsystem, quantifies why A1 is nonetheless safe for the altitude
problem, and connects it to the optical pose estimate produced by `pose/estimator.py`.

### 11.1 Gyroscopic stiffness: why misalignment precesses

The spin angular momentum along the body axis is large:

$$\mathbf L \approx I_s\,\omega\,\hat n,
\qquad I_s\omega\big|_{f=160} \approx (3.3578\times10^{-9})(2\pi\cdot160)
\approx 3.38\times10^{-6}\ \text{kg·m}^2/\text{s},$$

where $I_s = I$ (§3) is the polar (spin-axis) moment. A **transverse** magnetic torque
$\boldsymbol\tau_\perp$ (perpendicular to $\hat n$) does *not* tip the axis toward
$-\boldsymbol\tau_\perp$ as it would on a static body. For a fast gyro $|\mathbf L|$ is
nearly constant, so

$$\frac{d\hat n}{dt} \approx \frac{\boldsymbol\tau_\perp}{I_s\omega}$$

: the axis moves **along** $\boldsymbol\tau_\perp$, i.e. 90° from the naive "falling"
direction. A restoring torque therefore produces circulation of the axis (precession),
not direct un-tilting. This is the geometric heart of the whole subsystem.

### 11.2 The transverse magnetic torque (spin-averaged)

Tilt $\hat n$ from $\hat z$ by a small angle $\beta$ about the lab $\hat y$ axis,
$\hat n = [\sin\beta,\,0,\,\cos\beta]$. The in-plane magnetization (§2), body-fixed in the
blade plane and spinning at $\theta_r$, and the planar rotating field are

$$\mathbf m = m\,[\cos\theta_r\cos\beta,\ \sin\theta_r,\ -\cos\theta_r\sin\beta],
\qquad
\mathbf B = B\,[\cos\theta_f,\ \sin\theta_f,\ 0].$$

Then $\boldsymbol\tau = \mathbf m\times\mathbf B$ has axial component

$$\tau_z = \tau_{max}\big(\cos\beta\,\cos\theta_r\sin\theta_f - \sin\theta_r\cos\theta_f\big)
\ \xrightarrow{\beta\to0}\ \tau_{max}\sin\delta$$

(the §4.1 drive torque, recovered), and transverse components that, averaged over one fast
spin ($\theta_r = \theta_f - \delta$, using $\langle\cos(\theta_f-\delta)\cos\theta_f\rangle
= \tfrac12\cos\delta$ and $\langle\cos(\theta_f-\delta)\sin\theta_f\rangle=\tfrac12\sin\delta$),
give

$$\boxed{\ \langle\tau_y\rangle = -\tfrac12\,\tau_{max}\cos\delta\,\sin\beta
\ \equiv\ -\kappa_t\,\beta,
\qquad
\langle\tau_x\rangle = +\tfrac12\,\tau_{max}\sin\delta\,\sin\beta\ }$$

So the synchronized field supplies a genuine **tilt stiffness**

$$\kappa_t = \tfrac12\,\tau_{max}\cos\delta_{eq} \approx \tfrac12\,\kappa
\qquad(\text{half the phase-lock stiffness }\kappa=\tau_{max}\cos\delta_{eq}\text{ of §5.5/§8}),$$

plus a smaller cross-axis torque $\propto\sin\delta = 1/M$. This is the "restoring
out-of-plane magnetic torque" A1 invokes: now with an explicit coefficient.

### 11.3 Precession, nutation, and the role of damping

Collect the tilt into a complex variable $\chi = \beta_x + i\beta_y$. The symmetric-rotor
transverse equation (Euler's equations linearized about the fast spin) is

$$I_t\,\ddot\chi
\;+\; \underbrace{i\,I_s\omega\,\dot\chi}_{\text{gyroscopic}}
\;+\; \underbrace{c_t\,\dot\chi}_{\text{aero damping}}
\;+\; \underbrace{\kappa_t\,\chi}_{\text{magnetic}}
\;=\; \tau^{d}(t),$$

with $I_t$ the transverse moment about a diameter. From the CAD tensor (§3),
$I_t/I_s = 0.607$, so $I_s/I_t = 1.65$, not the 2 an ideal flat disc would give. The gyroscopic term splits the response into two widely separated modes:

$$\Omega_{\text{nut}} \approx \frac{I_s}{I_t}\,\omega = 1.65\,\omega
\ (165\text{–}379\ \text{Hz over the 100–230 Hz spin range, fast nutation}),
\qquad
\Omega_{\text{prec}} \approx \frac{\kappa_t}{I_s\omega}
= \frac{2.45\times10^{-5}}{3.38\times10^{-6}} \approx 7.3\ \text{rad/s}
\ (\approx 1.2\ \text{Hz, slow precession}).$$

**Damping is what makes alignment an attractor.** With $c_t = 0$, gyroscopic action plus
the restoring torque yield *steady coning at fixed $\beta$*: a tilted top circles forever,
it does not straighten. Only $c_t$ (aerodynamic drag on the wobbling blades, plus any
magnetic/eddy loss) spirals the cone inward to $\beta\to0$. A1's claim that "the aligned
state is an attractor" is therefore a statement about **dissipation**, not about the
magnetic torque alone.

### 11.4 Where tilt sits in the timescale stack

$$\underbrace{\text{nutation}}_{165\text{–}379\ \text{Hz}}
\;\gg\;
\underbrace{\text{phase lock}}_{\sim 19\ \text{Hz}}
\;\gg\;
\underbrace{\text{precession}}_{\sim 1.2\ \text{Hz}}
\;\sim\;
\underbrace{\text{heave}}_{\sim 0.1\text{–}1\ \text{Hz}}$$

Nutation is far above every other mode and is safely ignorable. **Precession, however,
lands right in the heave band.** Its *amplitude* stays small under small disturbances
(large $I_s\omega$ ⇒ stiff), which is exactly what licenses A1 for altitude control. But it
is **not** frequency-separated from heave: any real misalignment or steering transient that
excites tilt produces a ~1 Hz coning that both modulates thrust (§11.5) and appears
directly in the optical pose (§11.6). A1 is thus safe for *nominal* hover, but is a
genuine approximation the moment tilt is actually excited.

### 11.5 Coupling into thrust (why altitude barely notices)

A tilted rotor splits its thrust between vertical and lateral:

$$T_z = k_T f^2\cos\beta, \qquad T_{\text{lat}} = k_T f^2\sin\beta.$$

Vertical thrust loses only $O(\beta^2)$: negligible for the small $\beta$ that gyroscopic
stiffness enforces, so the §7 one-way cascade into heave survives to second order. The
**lateral** component is first order in $\beta$: it is precisely the steering handle the
scalar model omits (§10.5). Altitude is insensitive to tilt; *translation is made entirely
of it*.

### 11.6 Linear tilt block and the optical output

Appended to the §8 state $\tilde x = [\delta,\omega,z,w]^\top$, the tilt subsystem adds
$[\beta_x,\dot\beta_x,\beta_y,\dot\beta_y]$ as a block of the same shape as the phase-lock
block, but gyroscopically cross-coupled between the two axes:

$$I_t\ddot\beta_x + c_t\dot\beta_x + \kappa_t\beta_x - I_s\omega\,\dot\beta_y = \tau^{d}_x,
\qquad
I_t\ddot\beta_y + c_t\dot\beta_y + \kappa_t\beta_y + I_s\omega\,\dot\beta_x = \tau^{d}_y.$$

The disturbance/input torques $\tau^{d}$ are set by the commanded tilt of the field's
rotation plane: the steering input the scalar model has no room for. §12.5 derives
$\tau^{d}$ from the coil currents, and §12.8 gives the first-order form this block collapses
to once nutation is dropped, which is what the executable model integrates.

This subsystem is the natural plant for the optical sensor. `pose/estimator.py` returns the
disk's **normal vector**, to first order that is exactly $(\beta_x,\beta_y)$, so it adds
output rows

$$y_{\text{tilt}} = \begin{bmatrix}\beta_x\\ \beta_y\end{bmatrix} = C_t\,\tilde x,$$

which make the tilt states observable. The §8 model contains neither these states nor this
output, so **any** attempt to close a loop on the measured pose (attitude hold, steering,
disturbance rejection) requires exactly the block above. For pure altitude regulation the
pose estimate is instead a **validity monitor**: as long as it reads $\beta\approx0$, A1
holds and the §7–§8 model is the complete story.

---

## 12. The multi-coil spatial plant

§7's model is one-dimensional by construction: A3 makes the field uniform, so it applies pure
torque, and lateral motion has nowhere to come from. Drop A3 and the picture changes
completely. `matlab/MultiCoilBeamformingGUI_quickGeom_rigidTilt_coil22mm.m` simulates the
plant that results, and `ai/design/spatial_model.py` is its port. The differences are
structural, not incremental:

| | §7–§8 | this section |
|---|---|---|
| position | $z$ | $r \in \mathbb{R}^3$ |
| attitude | none (A1) | spin axis $s \in S^2$ |
| field | uniform, $\tau_{max}$ constant | 16 coils, spatially varying |
| phase lag | a dynamic state $\delta$ | algebraic, solved per step |
| lateral force | absent | tilt-thrust and gradient force |

The configuration manifold is $\mathbb{R}^3 \times S^2$: **five degrees of freedom, exactly
the five that [chapter 3](../pose/theory.md) proves the vision estimator measures**, and
neither has a sixth. Roll is unobservable optically and absent here, because cycle-averaging
has already integrated the spin phase away. §13 makes that correspondence the basis of a
controller.

The array is the GUI's `cross` preset, built by `quick_coils`: 16 air-core coils of radius
10.5 mm and 650 turns, grouped as four channels of $2\times2$. Channel centres sit at 37 mm
radius on the $\pm x, \pm y$ axes with a 21 mm internal pitch. Three details about it are
load-bearing:

**The phase progression is the rotation.** Channels are driven at 0°, 270°, 180°, 90° going
round the ring. Right and left are in antiphase and so contribute only to $\mathbf u$ below;
top and bottom are in antiphase with each other and quadrature to the first pair, so they
contribute only to $\mathbf v$. Two orthogonal, quadrature field components is exactly a
rotating field, and reversing the progression reverses the spin sense.

**One channel is one actuator.** The four coils of a channel share a phase and a current, so
however many coils there are physically, the array has four independent complex drives. §12.7
shows that is enough.

**The tilt is rigid.** `quick_coils(tilt_deg=...)` rotates each $2\times2$ group bodily about
its own tangential axis $\hat e_t = \hat z \times \hat e_r$, by Rodrigues:

$$R = \cos a\,\mathbb 1 + (1-\cos a)\,\hat e_t\hat e_t^\top + \sin a\,[\hat e_t]_\times,
\qquad \hat n_{ch} = R\hat z = \sin a\,\hat e_r + \cos a\,\hat z,$$

and the *same* $R$ carries the in-plane basis that places the four coil centres, so the group
stays coplanar instead of the normals tipping while the centres stay at $z=0$. The default
preset uses $a = 0$, which is what the measured coil table in the GUI reflects; the port
reproduces that table position-for-position (§12.10).

### 12.1 The field as two harmonic coefficients

Every coil is driven at the same frequency, so at a fixed point the field traces a closed
curve once per drive cycle. Sampling it in time would be wasteful and would alias; instead
note that a coil driven at $I_c\cos(\theta + \varphi_c)$ contributes a field linear in that
drive, so the total is exactly a first harmonic:

$$\boxed{\ \mathbf B(\theta) = \mathbf u\cos\theta + \mathbf v\sin\theta\ },
\qquad
\mathbf u = \sum_c \mathbf b_c(r)\, I_c\cos\varphi_c,
\qquad
\mathbf v = -\sum_c \mathbf b_c(r)\, I_c\sin\varphi_c,$$

with $\mathbf b_c(r)$ the field of coil $c$ per ampere. Two vectors, six numbers, and the
entire cycle is in them. Everything below is algebra on $\mathbf u$ and $\mathbf v$.

Two consequences worth naming, because both are load-bearing later:

- **The field is linear in the currents** (A4 again). $\mathbf u$ and $\mathbf v$ are linear
  maps of the drive phasors $z_c = I_c e^{i\varphi_c}$, so a controller evaluating many
  candidate current sets at one point pays for the geometry once. This is what makes an MPC
  horizon affordable (§13.4).
- **No sign ambiguity.** $\mathbf u, \mathbf v$ come from the drive, not from fitting an
  ellipse to samples, so there is no eigenvector sign to resolve. §12.4 depends on that.

`spatial_model.coil_basis` uses the point-dipole field for $\mathbf b_c$, matching the
MATLAB's `B_from_coil_stub`. Its accuracy is the weakest link in the whole chain, and §12.9
puts a number on it.

### 12.2 The polarization ellipse, exactly

$\mathbf B(\theta)$ traces the image of the unit circle under the $3\times2$ map
$A = [\,\mathbf u\ \ \mathbf v\,]$, which is an ellipse. Its semiaxes $a \ge b$ are therefore
$A$'s **singular values**: exact, with no fitting and no time sampling. Since only $a+b$ and
$ab$ are ever needed, the $2\times2$ Gram matrix suffices:

$$G = A^\top A = \begin{bmatrix} \mathbf u\cdot\mathbf u & \mathbf u\cdot\mathbf v \\
\mathbf u\cdot\mathbf v & \mathbf v\cdot\mathbf v\end{bmatrix},
\qquad
a^2, b^2 = \frac{\operatorname{tr}G \pm \sqrt{(\operatorname{tr}G)^2 - 4\det G}}{2}.$$

Closed form, and the port uses it rather than calling an SVD, because this sits in the
innermost loop of every prediction §13 rolls out. On the array's axis the ellipse is a
circle: $a = b = 4.58$ mT at $z = 15.2$ mm, 4.29 mT at $z = 20.1$ mm.

### 12.3 Drag balance in an elliptical field, and the lock floor

§5.3 balanced drag against $\tau_{max}\sin\delta$ for a circular field. For an ellipse the
cycle-averaged axial magnetic torque is $\tfrac12 m_{dip}(a+b)\sin\varphi$, so synchronous
lock requires

$$\boxed{\ \sin\varphi = \frac{2\,\tau_{drag}}{m_{dip}\,(a+b)}
= \frac{2\,k_d f^2}{m_{dip}\,(a+b)}\ }$$

At $a = b = B$ this collapses to $\sin\varphi = k_d f^2/(m_{dip}B) = \tau_{drag}/\tau_{max}$,
recovering §5.3 exactly. The elliptical generalization simply replaces the field amplitude by
the mean semiaxis $(a+b)/2$.

Unlike §5's $\delta$, $\varphi$ here is **not a dynamic state**. The model assumes the phase
loop has already settled, legitimate, since §11.4's stack puts phase lock at 19 Hz, an order
above everything this section cares about, and solves the balance algebraically each step.
The price is that the phase transient is invisible, so §5.6's ramp limits still bind and are
not enforced here.

**The lock floor.** No real $\varphi$ exists when the right-hand side exceeds 1, and that is
step-out. Rearranged, it is a condition on the *local field strength*:

$$a + b \;\ge\; \frac{2 k_d f^2}{m_{dip}}
\qquad\Longrightarrow\qquad
\lvert B\rvert \ \ge\ 1.00\ \text{mT at }110\ \text{Hz (circular, } k_d = 3\times10^{-10}).$$

Against the 4.3–4.6 mT actually available near the axis that is a factor of about 4.4 in
hand: but the floor scales as $f^2$, so the margin erodes fast: at 4.5 mT the required
$\sin\varphi$ reaches the 0.8 safety limit at 209 Hz and hits 1 at 233 Hz. **The workspace is
therefore a region whose shape depends on the commanded frequency**, not a fixed box. Both the MATLAB and
the port report the violation rather than clipping $\varphi$ to $\pi/2$: clipping would hide
loss of synchronization behind a plausible-looking trajectory, which is the one failure this
model exists to predict.

### 12.4 The directed rotation axis

Differentiating $\mathbf B(\theta)$ gives
$\mathbf B \times \tfrac{d\mathbf B}{d\theta} = \mathbf u \times \mathbf v$, so

$$\hat n = \frac{\mathbf u \times \mathbf v}{\lVert \mathbf u \times \mathbf v\rVert}$$

is the physical axis *and sense* of rotation. **Never flip it**: not toward $+\hat z$, not
toward the previous step's value, not toward the robot's spin axis. All three are tempting as
continuity rules and all three are wrong: the sign carries the rotation direction, and
flipping it silently reverses the modelled precession. Both the MATLAB and the port carry
this as a comment at the point of temptation.

### 12.5 The geometric phase reference and the averaged torque

The phase lag $\varphi$ has to be measured from *something*, and the natural-looking choice,
the $\cos\theta$ coefficient $\mathbf u$, is arbitrary, since it depends on where the drive's
phase origin was put. The model instead uses a reference fixed by geometry. Let $\hat s$ be
the robot's spin axis and $\hat n$ the field's, and define

$$\hat e_I = \widehat{\hat n \times \hat s}
\quad(\text{the intersection of the two rotation planes}),
\qquad
\hat e_R = \widehat{\hat s \times \hat e_I},
\qquad
\hat e_F = \widehat{\hat n \times \hat e_I}.$$

$\hat e_I$ is the line where the robot's plane of rotation meets the field's; $\hat e_R$ lies
in the robot's plane pointing toward the projection of $\hat n$, so it is the *aligning*
direction. The reference phase $\lambda$ is the field phase at which
$\mathbf B(\lambda)$ points along $+\hat e_I$, solved from
$\mathbf B(\lambda)\cdot\hat e_F = 0$, i.e. $\lambda = \operatorname{atan2}(-u_F, v_F)$ with
the branch chosen by $\mathbf B(\lambda)\cdot\hat e_I > 0$. The moment then lags by $\varphi$
from that reference:

$$\mathbf m(\theta) = m_{dip}\big[\cos(\theta - \lambda - \varphi)\,\hat e_I
+ \sin(\theta - \lambda - \varphi)\,\hat e_R\big].$$

Because both $\mathbf m$ and $\mathbf B$ are first harmonics, the cycle average is exact from
their coefficients: $\langle\cos^2\rangle = \langle\sin^2\rangle = \tfrac12$ and
$\langle\sin\cos\rangle = 0$ give

$$\boxed{\ \langle\boldsymbol\tau\rangle
= \tfrac12\big(\mathbf m_{\cos}\times\mathbf u + \mathbf m_{\sin}\times\mathbf v\big)\ }$$

with the component along $\hat s$ removed, because the model holds the spin rate constant and
air drag absorbs the axial impulse. What remains is the transverse torque that tilts the axis:
§11.2's $\kappa_t$, now computed rather than assumed.

At exact alignment $\hat s \parallel \hat n$ the two planes coincide and $\hat e_I$ is
undefined. The transverse torque genuinely vanishes there, so torque returns zero rather than
inventing a basis; the gradient force still needs an in-plane reference and falls back to the
ellipse's major axis, which only fixes an otherwise arbitrary phase origin.

### 12.6 The gradient force, and why Earnshaw decides its sign

Non-uniform field means $\nabla B \ne 0$, so the dipole feels a force as well as a torque.
Holding the instantaneous moment fixed through the spatial derivative, as $\mathbf F =
\nabla(\mathbf m\cdot\mathbf B)$ requires, and averaging over a cycle:

$$\langle F_i\rangle = \tfrac12\Big(\mathbf m_{\cos}\cdot\partial_i\mathbf u
+ \mathbf m_{\sin}\cdot\partial_i\mathbf v\Big).$$

The MATLAB evaluates $\partial_i\mathbf u$ by central differences, six extra field evaluations
per step plus a step-size knob the user has to choose. The point dipole has a closed-form
gradient, so the port uses it:

$$\frac{\partial B_i}{\partial r_j} = \frac{3\mu_0}{4\pi \lvert d\rvert^5}
\left[(\mathbf m\cdot \mathbf d)\,\delta_{ij} + d_i m_j + m_i d_j
- \frac{5 (\mathbf m\cdot \mathbf d)\, d_i d_j}{\lvert \mathbf d\rvert^2}\right],
\qquad \mathbf d = \mathbf r - \mathbf c.$$

It is symmetric and traceless, which is $\nabla\times\mathbf B = \nabla\cdot\mathbf B = 0$
away from the sources, and the port asserts both. Removing the knob is incidental; cutting
field evaluations sevenfold is what makes §13's horizon affordable.

**The force is not a perturbation.** The GUI defaults it off and its name invites treating it
as a correction, but on this array it reaches **44.5 % of the robot's weight** at $z = 20$ mm,
pulling down. Its structure is more interesting than its size. Differentiating about the
on-axis field maximum at $z = 15.156$ mm gives

$$\frac{\partial F_i}{\partial r_j}
= \mathrm{diag}\big(+0.0536,\ +0.0536,\ -0.1071\big)\ \text{N/m},
\qquad \operatorname{tr} = -1\times10^{-6}\ \text{N/m}.$$

The trace vanishes to one part in $10^5$, and it must: $\nabla\cdot\mathbf F =
\nabla^2(\mathbf m\cdot\mathbf B) = \mathbf m\cdot\nabla^2\mathbf B = 0$. So the vertical trap
(negative, restoring, stiffness 0.107 N/m) **forces** a lateral anti-trap of exactly half that
in each horizontal direction. This is Earnshaw's theorem doing its work: no static field can
trap in all three directions, and here the price of a good vertical trap is a lateral
expulsion of $+13.0$ % of weight at 2 mm off axis.

That anti-trap is a $+0.0536$ N/m stiffness on an $8.36\times10^{-5}$ kg robot: an unstable
pole at $\sqrt{k/m} = 25.3$ rad/s, **4.03 Hz, doubling every 27 ms**. The field-tilt actuator
that would have to fight it slews the spin axis at $\tfrac12\tau_{max}/(I_s\omega) \approx
3.2$ rad/s, about **0.51 Hz**. Eight times too slow. With the gradient force included, this
array cannot hold the robot laterally, and no controller changes that, because the ratio is
set by Laplace's equation and the magnet's dipole moment rather than by the control law.

§14 reopens this with the coil array tilted and the frequency dropped, which does make the
static lateral stiffness restoring, and shows why that is still not enough.

### 12.7 Channel authority, and inverting it

The four channels give $2\times4 = 8$ real knobs; the field at a point is 6 numbers. Summing
each channel's coils gives a $3\times4$ basis $\mathbf B_{ch}(r)$ in tesla per amp, and

$$\mathbf u = \mathbf B_{ch}\,\mathrm{Re}\,\mathbf z, \qquad
\mathbf v = -\mathbf B_{ch}\,\mathrm{Im}\,\mathbf z,$$

which is block diagonal: the real parts set $\mathbf u$, the imaginary parts set $\mathbf v$,
independently. Numerically $\mathbf B_{ch}$ has **rank 3 everywhere in the workspace**, with
singular values $(2.42,\ 2.42,\ 0.36)$ mT/A on axis and condition number 3–7. So the array
commands the rotation axis, the amplitude and the ellipticity of the local field completely
and independently. The weak singular direction is the one that tilts the rotation plane,
unsurprising, since all sixteen coils are coplanar with $+\hat z$ normals, so tilt costs
about seven times the current that in-plane field does.

Full rank means the map inverts. To command a circular field of amplitude $B_0$ about an axis
$\hat n$, build an orthonormal pair $(\hat e_1, \hat e_2)$ spanning $\hat n^\perp$ with
$\hat e_1 \times \hat e_2 = \hat n$, and solve

$$\mathbf z = \mathbf B_{ch}^{+}(B_0\hat e_1) - i\,\mathbf B_{ch}^{+}(B_0\hat e_2),
\qquad \mathbf B_{ch}^{+} = \mathbf B_{ch}^\top(\mathbf B_{ch}\mathbf B_{ch}^\top)^{-1}.$$

The realized axis matches the command to better than $10^{-3}$ degrees, and the ellipse is
circular to machine precision. The pseudo-inverse picks the minimum-norm solution in the
one-dimensional null space, which, since ohmic loss is $\tfrac12\sum_c R_c\lvert I_c\rvert^2$
, is simultaneously the **minimum-dissipation** current set. The energy term in §13's cost is
therefore already at its optimum for any commanded field, and only trades against amplitude.

### 12.8 The eight-state plant

Collecting §12.1–12.7, with state $x = [\,\mathbf r,\ \mathbf v,\ \hat s\,]$ and the drive
$(\mathbf z, f)$:

$$\boxed{\;
\begin{aligned}
\dot{\mathbf r} &= \mathbf v \\[2pt]
\dot{\mathbf v} &= \underbrace{g\Big(\tfrac{f}{f_h}\Big)^{2}\hat s}_{\text{lift along the spin axis}}
- g\hat z + \frac{\langle\mathbf F\rangle}{m_R} - \beta\mathbf v \\[4pt]
I_s\,\omega\,\dot{\hat s} &= \langle\boldsymbol\tau\rangle_{\perp \hat s}
\end{aligned}\;}$$

The third line is the first-order, fast-spin limit of §11.6's block: nutation at $1.65\omega$
(165–379 Hz) is far above everything else and is dropped, which turns two tilt states per axis
into one. **Lift points along $\hat s$, not along $\hat z$**: that single fact is the whole
lateral actuator, and it is the first-principles content of the `k_lat` seed guess in
`control/hover_model.py`.

Ordering inside a step matters and is preserved in the port: the spin axis is updated *before*
lift and gradient force are evaluated, so translation uses one consistent $\hat s$.

**What the model has no dissipation for.** §11.3 already warned that alignment is an attractor
only because of $c_t$, and that with $c_t = 0$ the transverse torque yields steady coning at
fixed tilt rather than straightening. This model sets $c_t = 0$. The consequence is sharper
than "lightly damped": run open-loop with the currents held fixed, and the robot leaves a
20 mm box within about a second **from any starting offset**, including 10 µm, with a growth
time constant of 0.175 s. The mechanism is precisely §11.1's: displacing the robot tilts the
local field axis, the spin axis answers by *precessing* 90° from the restoring direction
instead of aligning with it, and a restoring torque delivered through a 90° phase shift is a
growing spiral. Measured directly, commanding a field tilt of 0.1 in $+x$ moves the spin axis
into $+y$ at 72° from the command after 50 ms.

A plausible $c_t$ does not rescue it. §11.3's decay rate $\kappa_t c_t/(I_s\omega)^2$ with
$c_t$ of order the *axial* drag coefficient gives a time constant near 100 s, three orders of
magnitude too slow. `spatial_model.Plant` exposes `align_tau` for experiments and defaults it
to zero, matching the MATLAB. Treat it exactly as `hover_model.py` treats `k_lat`: a number to
identify on the rig, never a result.

§13.2 shows this is not the end of the story, because the instability belongs to the
*fixed-current* plant, and a controller does not hold its currents fixed. §14 asks the
opposite question, whether any fixed-current array is passively stable, and bounds the answer.

### 12.9 What the model gets wrong

Three items, in descending order of size.

1. **The point-dipole coil.** `B_from_coil_stub` is a far-field form used at $R/\lvert d\rvert
   \approx 0.2$–$0.3$. Against the exact on-axis loop field it overestimates by **13.9 % at
   35 mm** (the nearest coils to the working point) and **6.0 % at 53 mm** (the farthest).
   Because the error varies with distance it does not cancel between coils: it reweights near
   against far, which is what sets the field *direction*, so it biases $\hat n$ and every
   torque downstream, not merely the magnitude. It dwarfs every geometric detail in §3.
   Biot–Savart for a circular loop (elliptic integrals) is a drop-in replacement and
   is the first thing to fix before trusting any of this against hardware.
2. **No phase dynamics.** §12.3 solves $\varphi$ algebraically, so the $Q \approx 20$ swing of
   §5.5 and the ramp limits of §5.6 are invisible. Feasible here is not feasible there.
3. **A stale default.** The GUI's initial spin axis, `s0 = [-5.5557e-4, ~0, 0.99999985]`, is
   documented in the file as "the local directed `n_rot` at the recommended initial point". It
   is not, for the coil table shipped beside it: the port reproduces that table
   position-for-position and phase-for-phase, and the axis at $(2, 0, 20.084)$ mm tilts
   $+0.0488$ in $x$, not $-0.00056$. The stored value fits the same array at $z \approx 15.03$
   mm: within 0.12 mm of the field maximum at 15.156 mm, which is where the axis tilt crosses
   zero and where passive hover would have been tested. So `z0` was later changed from about
   15 mm to 20.084 mm and `s0` was never regenerated. Regenerate it; do not trust the
   constant.

### 12.10 Validating the port

MATLAB does not run in this environment, so `spatial_model.py` cannot be checked by diffing
trajectories against the original. It is pinned by invariants instead, all of which run on:

```bash
uv run python ai/design/spatial_model.py
```

They cover geometry, the field and its analytic gradient, the torque and force structure, and
the two defining behaviours of §12.8: on-axis hover drifts under 1 nm in 2 s, off-axis it
steps out in 0.39 s. Read `_self_check` for what each one asserts.

Two are worth knowing about because they reach outside the port. The GUI documents its default
start point as the local $\hat n$, so recomputing it is a free regression test against the
original implementation, and it is what exposed the stale constant in §12.9. The circular-field
limit lands on $\sin\delta = k_d f^2/\tau_{max}$, which §5.3 derived independently, so §12.3
and §5.3 stand or fall together.

**What none of this catches** is a shared assumption. Every check is internal to the
point-dipole field, so if `B_from_coil_stub` is wrong then the port is faithfully, verifiably
wrong in the same way as the MATLAB. §12.9 item 1 is why that matters, and only Biot–Savart or
hardware settles it.

---

## 13. Model-predictive control on the 5-DOF pose

The question this section answers: given what `pose/estimator.py` measures, can §12's plant be
flown by MPC? Yes, and the reasons it works are not the ones the formulation suggests.

### 13.1 The estimator is a complete sensor

§12's configuration manifold is $\mathbb{R}^3\times S^2$. Chapter 3's estimator returns
position and normal, and proves roll is exactly unobservable: [§21](../pose/theory.md)
(ch.3) shows rotating the rim about its own normal maps the image to itself. That is not a
gap to be worked around: **§12's model has no roll state either**, because cycle-averaging
integrated the spin phase away in §12.5. Sensor and model have the same five degrees of
freedom for the same reason.

Rates come from `pose/filter.py`: two constant-velocity Kalman filters, one on position and
one on the normal as a 3-vector, giving 3.5 mm/s velocity RMSE against 64 mm/s for a raw
finite difference. So the full eight-state $x = [\mathbf r, \mathbf v, \hat s]$ is measured,
and **no observer is needed beyond what already exists**.

Three practical gaps, all small and all real:

- **The normal's sign is not observable**, and §12 needs the *directed* spin axis: $\hat s$
  and $-\hat s$ are different states with opposite precession. Resolve it from the field,
  which sets the spin sense: take the branch with $\hat s\cdot\hat n > 0$. `PoseFilter`
  already maintains sign continuity frame to frame, so this is a one-time choice at takeoff.
- **$\dot{\hat s}$ is computed but not returned.** `PoseFilter.update` hands back
  `(xyz, velocity, normal)` and drops `self.nrm.rate`, which is the normal's rate the filter
  already estimated. One line.
- **The frames are not connected.** The estimator reports millimetres in the datum frame set
  by `calib/zeroing.py`; §12 is SI in the coil frame. The rigid transform between them exists
  nowhere in this repository. Nothing here can touch hardware until it does.

### 13.2 The instability belongs to the currents, not the robot

§12.8's open-loop result, gone in under a second from a 10 µm offset, looks like it settles
the question in the negative. It does not, and the reason is worth stating plainly because it
inverts the obvious reading.

That experiment held the coil currents fixed. A controller does not. Re-solving §12.7's
allocation for the *measured* position each step makes the commanded field axis vertical
wherever the robot actually is, which removes the position-to-tilt coupling **outright**.
Linearizing the re-allocating loop about hover (`spatial_mpc.linearize`, numerical Jacobians
through the allocation) gives

$$\lambda = \{-0.78 \pm 3.43j,\ \ -0.2,\ -0.2,\ -0.2,\ \ 0,\ 0,\ 0\}\ \text{s}^{-1}$$

: a damped precession, three modes at the translational damping $\beta$, three integrators for
position, and **no unstable eigenvalue at all**.

So the hard part of this plant is not stabilizing an unstable rotor. It is that the actuator
must be recomputed from the measurement every step: hold the currents fixed and no gain can
save it; re-solve them and the plant is merely marginally stable. That is a statement about
where the loop has to close, and it is the single most useful thing in this section.

It also means an LQR suffices for stabilization. What MPC adds is constraints, and §13.3 is
about those.

### 13.3 Formulation

**State** $x = [\mathbf r\ (\text{m}),\ \mathbf v\ (\text{m/s}),\ \hat s]$, eight degrees of
freedom. **Input** $u = [\,\text{tilt}_x,\ \text{tilt}_y,\ f\,]$: the commanded field axis
$\hat n \propto (\text{tilt}_x, \text{tilt}_y, 1)$ and the drive frequency, with §12.7's
allocation turning the first two into currents. The honest input is the four channel phasors,
nine real numbers; the reduction is lossless because §12.7's rank-3 result says the axis is
all that the tilt knobs can independently set, and the allocation recovers the phasors.

**Cost.** Quadratic, each term normalized by the value at which it should count as one unit of
bad: Bryson's rule, the convention `design_hover_lqr.py` already uses:

$$J = \sum_{k}\Big[
\lVert \mathbf r_k - \mathbf r^{ref}\rVert^2_{Q_r} +
\lVert \mathbf v_k\rVert^2_{Q_v} +
\lVert \mathbf a_k\rVert^2_{Q_a} +
\lVert \mathbf j_k\rVert^2_{Q_j} +
\lVert \mathbf I_k\rVert^2_{R_I} +
\lVert \Delta u_k\rVert^2_{R_\Delta}\Big]
+ \text{terminal}$$

| term | normalizer | why that number |
|---|---|---|
| position | 2 mm | the band `simulate_hover.plot` already draws |
| velocity | 50 mm/s | ch.3's test trajectories run 7.6–24.4 mm/s |
| acceleration | 2.5 m/s² | §6.4: a ±20 Hz step is a quarter g |
| jerk | 25 m/s³ | one acceleration unit per 0.1 s |
| coil current | $I_{max}$ | dimensionless duty; $\tfrac12\sum R_c\lvert I_c\rvert^2$ in watts once $R_c$ is measured |
| input slew | $\dot f_{max}T_s$ | soft companion to the hard slew bound |

Acceleration and jerk cost nothing extra to evaluate: the plant already produces
$\mathbf a_k$ at every predicted step, and $\mathbf j_k = (\mathbf a_{k+1}-\mathbf a_k)/T_s$
is one subtraction. Physically
$\mathbf j \approx \tfrac{2gf}{f_h^2}\dot f\,\hat s + g(f/f_h)^2\dot{\hat s}$, so the two
sources of jerk are frequency slew and tilt rate: the same two things $R_\Delta$ reaches.
Both are kept, because $R_\Delta$ shapes the *command* and $Q_j$ shapes the *trajectory*, and
over a spatially varying field those differ.

Two tensions are structural rather than tuning artifacts, and are better stated than tuned
away:

1. **Energy fights lock.** Lowering the current lowers $(a+b)$, which raises the required
   $\sin\varphi$ of §12.3 and pushes toward step-out. The constraint sets the floor; the
   energy term only spends what is left above it.
2. **Smoothness fights speed.** Exactly the trade `design_hover_lqr.py` records for the
   altitude loop, where the tighter designs limit-cycled against latency and 0.78 Hz poles
   were the answer.

**Constraints**, and this is what a gain cannot express:

- **Lock margin** $2k_d f^2 \le m_{dip}(a+b)$. Under §12.7's fixed-amplitude allocation
  $a+b = 2B_0$ everywhere, so it collapses to a *bound* on $f$ alone, 208.6 Hz at
  $B_0 = 4.5$ mT with an 0.8 safety factor, which the solver enforces exactly and for free.
  It does not collapse entirely: $B_0$ is only achievable while the allocation stays inside
  the current ceiling, and far off axis it does not, so the residual coupling stays as a
  penalty on the realized ratio.
- **Current ceiling** per channel, enforced by saturating the allocation: keep the axis and
  the shape, give up amplitude. Without this the pseudo-inverse "achieves" $B_0$ anywhere by
  demanding unbounded current, and a prediction that leaves the array never loses lock.
- **Frequency bounds and slew**, reusing `design_hover_lqr.py`'s 200 Hz/s.
- **Workspace box** where the field map is trusted.

### 13.4 Real-time iteration, and what it costs

One SQP pass per control step, warm-started from §13.2's LQR solution, with the inputs free
for two steps and reverting to trim after. Three details each fixed a failure that looked like
a modelling problem:

- **Warm-start from the LQR, not from trim.** Doing nothing is a deep local minimum here: a
  tilt costs position immediately, because precession sends the robot 90° from the command
  (§12.8), and only pays later. Started at trim, SLSQP found no descent direction at all and
  the controller sat still while the robot flew away.
- **Pad the horizon tail with trim, not with the last block.** Ordinary move-blocking holds
  the final input to the horizon's end; a constant field tilt drives a constant precession, so
  any nonzero tail spirals out and scores as infeasible. Every nonzero input then looked
  identical to step-out, leaving exactly one feasible point.
- **Scale the decision variables.** A tilt of $10^{-2}$ and a frequency of $10^{2}$ differ by
  four orders of magnitude, and SLSQP's single absolute finite-difference step suits neither.
  At the default $1.5\times10^{-8}$ the tilt gradient was rounding noise off a 60-step Euler
  rollout.

Detuning matters as much as it does for the altitude loop. At input weight $\times1$ the
fastest closed-loop pole is 27.6 rad/s, which holds at a 20 ms control step and diverges at
50 ms; at $\times50$ it is 8.3 rad/s and stable across 10–50 ms.

Measured, at 20 Hz with a 0.4 s horizon and one SQP iteration:

| | |
|---|---|
| solve time | 44 ms median, 47 ms p95, against a 50 ms budget |
| real-time factor | 1.2× (13 s of flight in 10.7 s wall) |
| hold, from 13.15 mm out | 0.1 µm after 6 s |
| stepped target | settled 1–2 µm |
| peak acceleration / jerk | 0.25 m/s², 3.9 m/s³ |
| peak current, peak lock ratio | 1.66 A of 3.0, 0.23 of 1.0 |

Affordability comes from §12: the analytic field gradient removes six field evaluations per
step, `use_grad = False` skips the gradient tensor entirely, and replacing `np.cross` with an
explicit 3-vector form removed 65 % of the model's runtime. Together the plant step went
245 µs to 60 µs, which is the difference between running and not.

Cycle-averaging also makes the sample rate generous rather than tight. The model's fastest
retained mode is precession at about 1.2 Hz; nutation was dropped in §12.8 and the spin phase
in §12.1. The sub-millisecond step the MATLAB's `dt` recommendation asks for is a property of
explicit Euler, not of the physics.

### 13.5 What this does not yet show

The controller is running against its own model, so the tracking numbers measure the control
law and nothing else. Four things stand between here and hardware, in the order they will
bite:

1. **The field model** (§12.9), at 6–14 % and biasing direction, not just magnitude.
2. **The frames** (§13.1): the datum-to-coil transform does not exist.
3. **Latency.** Chapter 4's existing finding, keep the loop under about 0.8 Hz or it
   limit-cycles, is about actuation delay through the serial link, not solver speed, and
   nothing here simulates it. MPC is the right tool for it (`PoseFilter.predict_ahead` exists
   precisely to hand a controller a current estimate), but that has not been demonstrated.
4. **The gradient force.** Turned on, §12.6's Earnshaw argument says this array cannot hold
   lateral position at all, by a factor of eight in bandwidth. The demonstrations above run
   with it off, which is the GUI's default and is defensible only while the robot stays where
   the gradient is weak. **Whether the real rig is in that regime is the first thing to
   measure**, and if it is not, the answer is a different coil geometry rather than a better
   controller.

Finally, the optimality claim is local and should stay that way. The dynamics are bilinear in
the moment and the field gradient, so the problem is nonconvex, one SQP iteration per step
gives no global guarantee, and the warm start is doing much of the work. What the MPC
contributes over the LQR that seeds it is constraint handling, and that is the honest summary
of its role.

### 13.6 Running it

`ai/design/simulate_spatial.py` drives the loop in two modes over the same plant and controller.

**Headless** (the default) runs a scripted target sequence, hold, 6 mm across, 4 mm up, then
a diagonal, writes `spatial_mpc_sim.png`, and asserts the settling, constraint and
solve-time bounds tabulated in §13.4. This is the reproducible path and the one to run after
touching anything in §12.

**`--live`** animates it in real time. Click the top-down panel to move the target in $x$ and
$y$; a slider sets $z$. Two choices in it are worth knowing:

- The plant advances by however much **wall-clock** time has actually passed since the last
  frame, capped so a stall cannot take one enormous step. A slow solve therefore shows up as
  the simulated clock slowing down, never as a trajectory that quietly stops being real time.
  The achieved factor is printed in the title.
- Loss of lock stops the animation and says so, rather than raising through the caller. That
  matches what the runner has to do on hardware, where the response to divergence is a land
  command and the trace is the thing you want to keep.

**Reading the panels.** The controller is running against its own model, the predictor and
the plant are literally the same code, so the position traces measure the control law and
contain no information about robustness. The two panels that do carry information are the
**lock margin**, which is how close the robot is to §12.3's step-out, and the **channel
currents**, which are what the amplifier has to deliver. A run that tracks beautifully while
riding either limit has not solved the problem. In the shipped scenario both stay far off:
0.23 of 1.0 on lock, 1.66 A of 3.0 on current, with field tilts of about 1°.

---

---

## 14. Passive open-loop stability, and the bandwidth that decides it

§12.6 stated the negative result: the gradient force traps vertically, Earnshaw makes it expel
laterally at exactly half the rate, and the field-tilt actuator that would fight the expulsion
is eight times too slow. An earlier write-up (removed; this section supersedes it) reopened
the question with a sweep over channel
radius $d$, outward coil tilt $\theta$ and drive frequency $f$, and found geometries where the
quasi-static lateral stiffness goes negative. This section derives what that quantity is a
limit of, why it is not sufficient, and what the sufficient condition actually is.

`ai/design/open_loop_sweep.py` is the executable form. It reuses §12's model unchanged.

### 14.1 Where the lateral restoring force comes from

Move the robot off axis by $x$. Two forces answer, and they oppose each other.

**The gradient force expels.** §12.6: $C_{grad} = \partial F_x/\partial r_x > 0$, forced
positive by $\operatorname{tr}\nabla\mathbf F = 0$ whenever the axial trap
$\partial F_z / \partial r_z$ is negative.

**The tilted lift restores.** The local rotation axis $\hat n$ is no longer vertical off
centre, and §12.8's lift points along the spin axis $\hat s$, not along $\hat z$. If the axis
tips *inward* as the robot moves out, the lift acquires an inward component:

$$C_{tilt} = L\,G_n, \qquad L = m_R g\left(\frac{f}{f_h}\right)^2,
\qquad G_n = \frac{\partial n_x}{\partial r_x}.$$

$G_n$ is zero on the untilted array at its field maximum, which is why §12.6 saw only the
expulsion. Tilting the coils outward and dropping $f$ moves the trim height off that null and
makes $G_n$ strongly negative: $-58.9\ \mathrm{m^{-1}}$ at $d = 46$ mm, $\theta = 15^\circ$,
$f = 92$ Hz. The sum

$$C_{net} = C_{tilt} + C_{grad}$$

is then genuinely negative, and that earlier reading took it for passive lateral trapping.

**Why the frequency has to drop.** In this model lift depends on $f$ alone, so the axial
equilibrium is set entirely by the gradient force having to make up the difference:

$$\langle F_{grad,z}\rangle(z_{eq}) = m_R g\left(1 - (f/f_h)^2\right).$$

At $f < f_h$ that is positive, so $z_{eq}$ sits *below* the on-axis field maximum, where the
gradient pulls up, and the magnetic field carries about 30 % of the weight. That is also the
only reason an axial trap exists at all: nothing else in the model depends on $z$.

### 14.2 The lateral transfer function

$C_{net}$ is a static stiffness, and the path it describes is not static. Collect the lateral
position into $\xi = r_x + i r_y$ and the spin-axis tilt into $\chi = s_x + i s_y$. Four-fold
symmetry of the array makes the field-axis tilt isotropic, $\nu = G_n \xi$, with no cross term.
Translation from §12.8, and alignment from §11.6 with nutation dropped:

$$m_R\ddot\xi + m_R\beta\dot\xi = C_{grad}\,\xi + L\,\chi,
\qquad
(c_t + i I_s\omega)\,\dot\chi = \kappa_t(\nu - \chi).$$

The alignment block is a first-order lag with a **complex** pole, and the $i$ is §11.1's
precession: the axis moves *along* the torque, not toward it.

$$\frac{\chi}{\nu} = \frac{p}{s + p},
\qquad p = \frac{\kappa_t}{c_t + i I_s\omega}.$$

Closing the loop gives a cubic with complex coefficients, which is correct rather than sloppy:
the gyroscopic term splits forward whirl from backward whirl, so the two lateral axes do not
decouple into a real conjugate pair.

$$\boxed{\ m_R s^3 + m_R(\beta + p)\,s^2 + (m_R\beta p - C_{grad})\,s - C_{net}\,p = 0\ }$$

Three limits, and each is a design rule.

**At $s \to 0$** the constant term is $-C_{net}\,p$. So $C_{net} < 0$ is the DC stiffness
condition and says nothing whatever about any other frequency. It is exactly, and only, what
that earlier sweep computed.

**At $|s| \gg |p|$** the cubic degenerates to $s^2 - C_{grad}/m_R$: the tilt path has been
rolled off and the anti-trap acts alone, with growth rate

$$\lambda_{grad} = \sqrt{C_{grad}/m_R}.$$

**The pole cannot be moved past a ceiling.** Since $c_t \ge 0$,

$$\lvert p\rvert = \frac{\kappa_t}{\lvert c_t + i I_s\omega\rvert}
\;\le\; \frac{\kappa_t}{I_s\omega} \;\equiv\; \Omega_{align},$$

with equality at $c_t = 0$. **Alignment damping cannot buy bandwidth, it can only lose it.**
This is worth stating plainly because §11.3 and §12.8 both point at $c_t$ as the missing
ingredient, and for *alignment as an attractor* they are right. For *speed* they are not: the
bound is gyroscopic, set by the spin angular momentum, and no dissipation, and no control law,
touches it.

### 14.3 The criterion

Putting the two together, a necessary condition for the tilt mechanism to reach the
instability it is meant to cancel:

$$\boxed{\
R \;\equiv\; \frac{\Omega_{align}}{\lambda_{grad}}
= \frac{m_{dip}\,(a+b)\cos\varphi}{4\,I_s\,(2\pi f)}\Big/\sqrt{\frac{C_{grad}}{m_R}}
\;>\; 1\ }$$

$R$ replaces $C_{net}$ as the screening quantity. Measured at that sweep's three candidates
and at the §12 baseline, under the exact loop field of §14.5:

| $d$, $\theta$, $f$ | $z_{eq}$ | $C_{net}$ | $\Omega_{align}$ | $\lambda_{grad}$ | $R$ | predicted root | plant |
|---|---|---|---|---|---|---|---|
| 46 mm, 15°, 92 Hz | 18.39 mm | $-0.0157$ | 1.08 rad/s | 14.7 rad/s | **0.073** | $+14.6$ /s | escapes at 0.54 s |
| 47 mm, 10°, 94 Hz | 17.17 mm | $-0.0111$ | 1.18 rad/s | 14.3 rad/s | **0.082** | $+14.3$ /s | escapes at 0.55 s |
| 49 mm, 14°, 98 Hz | 17.82 mm | $-0.0355$ | 0.59 rad/s | 10.9 rad/s | **0.054** | $+11.0$ /s | escapes at 0.81 s |
| 37 mm, 0°, 110 Hz | 15.16 mm | $+0.0536$ | 3.49 rad/s | 25.3 rad/s | 0.138 | $+25.3$ /s | §12.8 |

The last two columns are the check that matters. The cubic's rightmost root is compared
against the growth rate the *full nonlinear plant* shows when integrated from 10 µm off axis,
$\ln(1000)/t_{escape}$: $+14.6$ against $+12.8$, $+14.3$ against $+12.5$, $+11.0$ against
$+8.5$. Agreement to 15–30 % on a reduced-order model with a complex pole is enough to trust
the structure, and without it §14.2 would be algebra with no plant attached.

Note also that the geometry sweep made $R$ **worse than the baseline**, not better. Pushing
$d$ out to 46–49 mm is what makes $G_n$ negative, and it is the same move that drops the
working field from 4.58 mT to 1.0–1.5 mT, which costs $\Omega_{align}$ linearly while buying
$\lambda_{grad}$ only as a square root.

### 14.4 Which knob moves $R$, and by how much

Since $C_{grad} \propto m_{dip} B''$ and $\kappa_t \propto m_{dip} B$,

$$R \;\propto\; \frac{\sqrt{m_{dip}\,m_R}}{I_s\,f}\cdot\frac{B}{\sqrt{B''}}
\;\propto\; \frac{\sqrt{m_{dip}\,m_R}\;\sqrt{I_{amp}}\;\ell}{I_s\,f},$$

with $\ell$ the array's length scale. Drive current enters as $\sqrt{\ }$ only, because it
raises the anti-trap at the same time as the tilt stiffness. From $R = 0.073$:

| knob | predicted | measured (§14.7 stage D) | what $R = 1$ would take |
|---|---|---|---|
| drive current $I_{amp}$ | $+1/2$ | non-monotone, see below | nothing available |
| drive frequency $f$ | $-1$ | $-1$ | 6.7 Hz, far below hover |
| dipole moment $m_{dip}$ | $+1/2$ | $+0.685$ | $30\times$ the magnet volume |
| spin inertia $I_s$ | $-1$ | $-1.000$ | $10\times$ lighter rotor at fixed $m_{dip}$ |
| uniform rotor scale $k$ | $-3/2$ | $-1.578$ | $k = 0.215$, a $4.6\times$ smaller robot |

$I_s$ comes out at exactly $-1$, as it must: it enters nowhere but $\Omega_{align}$. $m_{dip}$
and mass overshoot and undershoot their predicted $+1/2$ because both move the trim height, so
neither is a clean one-parameter family; the uniform scale is, and it lands within 5 % of the
predicted $-3/2$. The last row is the only knob that is not absurd, and it is not a parameter
of the drive: it is a different robot. Under a uniform scale $m_{dip}, m_R \propto k^3$,
$I_s \propto k^5$, and $f_h \propto k^{-1/2}$ because blade thrust goes as $f^2 D^4$ against a
weight going as $k^3$. Two consequences make the family clean: $\lambda_{grad}$ is
$k$-independent, since $C_{grad}$ and $m_R$ both go as $k^3$, and $z_{eq}$ is $k$-invariant for
the same reason. Only $\Omega_{align}$ moves, as $k^{-3/2}$. Step-out gets easier too,
$\sin\varphi \propto k$.

So the honest reading of that result: **the tilt mechanism is real, the geometry
does move $C_{net}$ through zero, and none of it is reachable by tuning the drive.** What it
buys is a factor of $\sqrt{}$ where a factor of 14 is needed. The lever that works is rotor
size, and §13.2's answer, re-solve the currents from the measurement, remains the one that
works on this hardware.

### 14.5 Three thresholds the stiffness screen does not see

**Step-out.** §12.3's lock floor $\sin\varphi = 2 k_d f^2 / (m_{dip}(a+b))$ is a condition on
the *local* field, and pushing the coils out to $d = 49$ mm drops it to 1.04 mT. That point
runs at $\sin\varphi = 0.766$, essentially at the 0.8 safety limit, so the geometry with the
most negative $C_{net}$ is nearly disqualified on step-out alone. It is not, however, the
binding constraint globally: of the 1707 gridded points with adequate clearance, step-out
rejects 64 and $C_{net} \ge 0$ rejects 1483 (§14.7).

**Clearance.** Outward tilt raises the inner coils of each $2\times2$ group by
$\tfrac12\,\text{pitch}\,\sin\theta$, and the coil bodies are 22 mm long. Trim heights of
17–19 mm are not automatically above them.

**The Laplace trace.** $\operatorname{tr}\nabla\mathbf F = 0$ holds for a *fixed* moment, and
the port asserts it: at the untilted baseline $2C_{grad} + \partial F_z/\partial r_z$ closes to
$1\times10^{-8}$ N/m. At the tilted operating points it leaves $+0.0054$ N/m, about 15 % of the
axial stiffness. That residual is not an error. Recomputing the derivative with the moment
frozen at its on-axis value closes the trace again, to $1.5\times10^{-8}$, which identifies the
whole residual as the moment re-orienting with position. It is worth knowing about, because it
is the one term in the model that Earnshaw does not constrain, and therefore the only place a
static lateral trap could hide. It is 15 % of the wrong sign's worth, so it does not, but the
sweep reports it per point rather than absorbing it.

### 14.6 The field model, and what fixing it changed

§12.9 item 1 said the point-dipole stub was the first thing to fix, and this section is where it
mattered, since $G_n$ is a statement about field *direction*.
`spatial_model.loop_basis` now provides the exact circular-loop field via elliptic integrals,
selected by `Coils.model = "loop"`, and the sweep runs every point under both. The verdict does
not move, $R$ changes by under 20 %, but $C_{net}$ does: at 46 mm/15°/92 Hz it goes from
$-0.0157$ to $-0.0067$ N/m, and $z_{eq}$ from 18.39 to 18.92 mm. **The dipole stub overstates
the lateral restoring by a factor of 2.3.** Any future search over coil positions has to run on
the loop field.

### 14.7 What the sweep found

Stage A grids $d \in [34, 60]$ mm, $\theta \in [0, 30]^\circ$, $f \in [60, 130]$ Hz under both
field models. Under the loop field, 3112 of those points have a stable axial trim at all, 1643
of those clear step-out and the coil bodies, and **none of them is laterally stable**: the
rightmost root of §14.2's cubic is positive at every one, from $+8.6$ to $+20.4\ \mathrm{s^{-1}}$.

Read that as a statement about this *family*, not about coil arrays. It is a two-parameter,
coplanar, four-fold symmetric family, and §15 shows that opening three specific freedoms inside
it produces designs that are asymptotically stable and certifiable.

The interesting part is why, and it is sharper than "$R$ is small everywhere".

$$\textbf{The two conditions pull the trim height in opposite directions.}$$

$C_{net} < 0$ needs the robot **below** the on-axis field maximum, where $\hat n$ tips inward;
that means $f < f_h$ and pushing $d$ out to 41 to 55 mm, which is exactly what drops the working
field to 0.92 to 1.83 mT and collapses $\Omega_{align}$. $R$ is largest **above** the maximum,
at $f = 130$ Hz and $z_{eq} = 29.9$ mm, where the field is flat enough that
$\lambda_{grad}$ falls faster than $\Omega_{align}$ does: $R = 0.281$ there, the best anywhere.
But above the maximum the axis tips the wrong way and $C_{net} = +0.043$ N/m. The best point
that actually restores is $d = 48$ mm, $\theta = 2^\circ$, $f = 95$ Hz, and it has $R = 0.097$.

`stability_map.png` shows this as one picture: $R$ falls monotonically to the right across the
$(d,\theta)$ plane while $C_{net}$ only crosses zero to the right of the black contour.

**Two corrections to how that reads.** First, the regions are not *disjoint*: 215 gridded points
have $C_{net} < 0$ while clearing step-out and the coil bodies. What is true is weaker and is the
whole point, namely that they overlap only where both are weak, $R$ running from 0.043 to 0.097
across that entire region against 0.138 for the untilted baseline. Second, the opposition is
structural only for **coaxial** arrays. The $G_n$ sign flip sits within 0.2 mm of the on-axis
field maximum at $0$ to $2^\circ$ of tilt, and separates from it by more than 10 mm at $20$ to
$30^\circ$: at $d = 43$ mm and $\theta = 30^\circ$ the maximum is at 6.0 mm and the flip at
16.6 mm. Tilt already begins to prise the two apart, which is the thread §15 pulls.

**Frequency.** The upper bound is not step-out. $C_{net}$ crosses zero at 95 Hz for most
geometries and 105 Hz at $30^\circ$ tilt, while $\sin\varphi$ there is only 0.41 to 0.65. A
second, harder ceiling sits just above: past about 130 Hz **no axial equilibrium exists at all**,
because lift then exceeds weight and the gradient force cannot pull down hard enough anywhere.

**Amplitude does not help.** Raising the drive from 1.25 A to 8 A moves $R$ from 0.281 to 0.256,
non-monotonically, because the extra field pulls the trim point down into a steeper gradient as
fast as it raises $\kappa_t$. $C_{net}$ meanwhile goes from $+0.043$ to $+0.285$ N/m. The
$\sqrt{I_{amp}}$ of §14.4 is an upper bound that the moving trim does not even deliver.

**Phase precision is not a limiting factor**, which is worth recording because it was a
candidate. A quadrature error $\varepsilon$ gives an ellipse of aspect exactly
$b/a = \tan(45^\circ - \varepsilon/2)$, but $\kappa_t$ and the lock margin both depend on the
*mean* semiaxis $(a+b)/2$, which is second order in $\varepsilon$: at $30^\circ$ of error the
field is visibly elliptical ($b/a = 0.577$) and yet $(a+b)/2$ has fallen 4 % and $R$ 9 %.

**The trace audit, over the whole grid.** With the moment frozen the residual
$2C_{grad} + \partial F_z/\partial r_z$ stays under $3.8\times10^{-7}$ N/m at all 3112 points,
and with the moment re-solved its median is $3\times10^{-4}$ N/m. §14.5's reading holds
everywhere, not just at the three points it was checked on.

### 14.8 Correspondence with the implementation

| quantity | where |
|---|---|
| exact loop field, FD gradient | `spatial_model.loop_field`, `loop_basis` |
| trim solve, scanning past step-out | `open_loop_sweep.trim` |
| $G_n$, $C_{grad}$, $C_{tilt}$, $C_{net}$, the trace audit | `open_loop_sweep.evaluate` |
| §14.2's cubic | `open_loop_sweep.lateral_roots` |
| §14.4's exponents, measured | `open_loop_sweep.stage_d`, `fit_exponent`, `solve_scale` |
| §14.7's grid, and the figures | `open_loop_sweep.run`, `results/open_loop_sweep/` |
| the pinned numbers in §14.3 | `open_loop_sweep._self_check` |

```bash
uv run python ai/design/open_loop_sweep.py --self-check
uv run python ai/design/open_loop_sweep.py            # about 25 minutes
uv run python ai/design/open_loop_sweep.py --reuse    # cached stage A, seconds
```

---

## 15. Opening the geometry: two rings, a fifth channel, and a certificate

§14 ended on a negative result: nothing in the `quick_coils` family is laterally stable. That
family is coplanar, four-fold symmetric, and two parameters wide. This section opens three
freedoms inside it, and the result changes sign: designs that are asymptotically stable, carry a
Lyapunov certificate, and hold position in the nonlinear plant for as long as they are run.

The three freedoms are not arbitrary. Each one answers a specific obstruction §14 identified.

### 15.1 Where the rotating field actually comes from

Take a coil at radius $r$ from the array axis with its own axis along $\hat z$, and evaluate its
field at an on-axis point $(0,0,z)$. With $\mathbf m = m\hat z$ and
$\mathbf d = (-r, 0, z - z_c)$, the dipole form gives

$$B_x = \frac{\mu_0}{4\pi}\,\frac{3(\mathbf m\cdot\mathbf d)\,d_x}{\lvert d\rvert^5}
= \frac{\mu_0}{4\pi}\,\frac{3\,m\,(z - z_c)(-r)}{\lvert d\rvert^5}.$$

$$\boxed{\ \text{The transverse field on the array axis is proportional to } z - z_c.\ }$$

Three consequences follow immediately, and all three are load-bearing.

**A coil plane level with the robot drives nothing.** At $z = z_c$ the transverse field is
exactly zero, so a ring coplanar with the working point contributes no rotation whatsoever. The
existing array works only because the robot hovers 20 mm above it. `coil_geometry` pins this:
the in-plane field measures $1.6\times10^{-18}$ T.

**A symmetric two-ring split cancels the drive.** Put half the coils at $z_c - g/2$ and half at
$z_c + g/2$ with the robot at the midpoint, and the two contributions carry opposite signs of
$z - z_c$ and cancel. The textbook Helmholtz move, applied naively here, turns the field off.

**The upper ring must be reverse-wound.** Driven in antiphase, the sign flip in the drive
cancels the sign flip in $z - z_c$ and the two rings *add*. That costs nothing: it is a winding
reversal, not a fifth driver. Measured at $r = 37$ mm with the rings at $z = -10$ and $+50$ mm
and the robot at 20 mm, against the 16 coplanar coils of §12:

| array | $B$ | $B''$ | $B^2/\lvert B''\rvert$ |
|---|---|---|---|
| 16 coplanar, $d = 37$ mm | 4.448 mT | $-7.550$ | $2.62\times10^{-6}$ |
| the same 16, split into two antiphase rings | 2.737 mT | $-0.813$ | $9.21\times10^{-6}$ |

Field strength falls, curvature falls nine times faster. Since §14.4 gives
$R \propto \sqrt{B^2/\lvert B''\rvert}$, that is a factor 1.9 on $R$ for no hardware at all.

### 15.2 The fifth channel, and why the obvious version of it is inert

**A steady field does exactly nothing to this robot.** Not to a tolerance: exactly. The
cycle-averaged force and torque are linear in $\mathbf B$, and §12.5 makes the moment of a
synchronous rotor a pure first harmonic, so $\langle\mathbf m\rangle = 0$ and

$$\langle\boldsymbol\tau\rangle = \langle\mathbf m\rangle\times\mathbf B_{dc} = 0,
\qquad
\langle\mathbf F\rangle = \nabla\big(\langle\mathbf m\rangle\cdot\mathbf B_{dc}\big) = 0.$$

The same orthogonality kills every harmonic $n \ne 1$, so a fifth channel at any frequency other
than $f$ is inert too. `spatial_model._self_check` pins this as bitwise equality: a 5.9 mT steady
field changes the acceleration and the torque by zero.

**It becomes decisive the moment the rotor carries an axial moment.** Give it
$m_z = \varepsilon\,m_{dip}$ along the spin axis, $\varepsilon$ of order a few per cent, by
tilting the magnets or adding a third. The two couplings are then exactly orthogonal:

$$m_{dip}\ (\text{blade plane}) \leftrightarrow \text{the rotating field only},
\qquad
m_z\ (\text{spin axis}) \leftrightarrow \text{the steady field only},$$

the first because $\langle\mathbf m_{rot}\rangle = 0$, the second because
$\langle\mathbf B_{rot}\rangle = 0$.

That orthogonality is what breaks §14's deadlock. There, the axial trap and the drive field were
**the same knob**: flattening the field to raise $\Omega_{align}$ destroyed the trap, which is
the saddle-node the sweep ran into, and keeping the trap kept the anti-trap that sets
$\lambda_{grad}$. Earnshaw ties them because both act on $m_{dip}$.

With a steady channel they separate. Tune the rotating geometry so $B''\approx 0$, which §15.1
does, and let the steady gradient acting on $m_z$ supply the axial trap on its own. Earnshaw
still applies to the steady part, and the port asserts its trace closes, but the anti-trap it
buys scales with the **small** $m_z$ instead of the large $m_{dip}$. The criterion collapses to

$$\boxed{\ R = \sqrt2\,\frac{\Omega_{align}}{\omega_z} > 1
\qquad\Longleftrightarrow\qquad \omega_z < \sqrt2\,\Omega_{align}\ }$$

with $\omega_z$ the axial trap's natural frequency, now a free design parameter rather than a
consequence. **The price is a soft, slow axial mode**, and that trade is the deliverable.

The steady field also adds a tilt stiffness $m_z B_{dc}$ alongside $\tfrac12 m_{dip}B_{ac}$,
worth a few per cent at $\varepsilon = 3\%$. A bonus, not the mechanism.

**Where the two coil sets go, and why they do not fight.** The rotating rings must be axially
offset from the robot, by §15.1. The steady ring wants the opposite: a ring's on-axis $B_z$ peaks
in its own plane, which is exactly where an axial trap has to sit. So the steady ring goes level
with the robot, where §15.1 guarantees it contributes no rotating field at all. The two
requirements put the two coil sets in different places, which is the reason one array can serve
both.

### 15.3 Four-fold symmetry is optimal, not merely convenient

It is tempting to break symmetry to gain freedom. Earnshaw forbids the gain. The lateral
Jacobian satisfies $C_{xx} + C_{yy} = k_z$ with $k_z$ fixed by whatever axial trap is wanted, and
what destabilises the robot is the **stiffest** lateral direction, $\max(C_{xx}, C_{yy})$. Since

$$\max(C_{xx}, C_{yy}) \;\ge\; \tfrac12\,(C_{xx} + C_{yy}) = \tfrac12 k_z,$$

with equality if and only if $C_{xx} = C_{yy}$, breaking symmetry can only make the worst
direction worse. The four-fold family is therefore the right place to search, and the search
space is four times smaller for it. `coil_geometry.free_array` exists to test the claim
numerically rather than to rely on it.

### 15.4 The optimiser, and what it optimises

The objective is not $R$. $R$ and the static stiffness $C_{net}$ are two separate necessary
conditions and neither implies the other: the two-ring array of §15.1 reaches $R = 0.55$ while
still being laterally *expelling*. Optimising either alone is gameable. So the objective is the
stability margin of the **full eight-state plant**, linearised at its own trim with the coil
currents held fixed:

$$\sigma = -\max\operatorname{Re}\lambda(A), \qquad A = \texttt{stability\_cert.linearize}.$$

Held fixed is the whole point. `spatial_mpc.linearize` differentiates *through* the current
allocator and so answers the closed-loop question, which §13.2 already settles. Sanity check on
the reduced model: at §14's operating point the eight-state margin is $+13.98\ \mathrm{s^{-1}}$
against the cubic's $+14.63$, a ratio of 0.96, so §14.2 was an honest summary.

`optimise_array` searches with `differential_evolution`, population-based sampling followed by an
L-BFGS-B polish. One detail is worth recording because it inverted the result. The feasible
region is a thin sliver bounded by the camera cone, and the best designs sit on its edge;
differential evolution's mutation walks out of that sliver and does not return. A 3960-evaluation
run returned $-1.17\ \mathrm{s^{-1}}$ where a 33-evaluation local polish of the same seed
returned $+0.067$. The module therefore polishes from the seed as well as from the global answer
and keeps the better. **Global optimality is not claimed**, and cannot be for a problem this
non-convex; the run reports the spread across independent restarts instead.

### 15.5 Certification

A Hurwitz linearisation gives local asymptotic stability of the nonlinear system by Lyapunov's
indirect method, and $V = x^\top P x$ from $A^\top P + PA = -Q$ is a genuine Lyapunov function on
a neighbourhood. `stability_cert.certify` solves it and refuses to return anything when $A$ is
not Hurwitz, so a certificate cannot be produced for a design that does not deserve one.

One structural warning decides what is provable at all. With `align_tau = 0` the tilt block
carries no dissipation, so damping has to arrive through the translational coupling or not at
all. `min_align_tau` reports the smallest alignment damping that makes a given design Hurwitz.
That number is a rig identification, never a result, exactly as §12.8 says of `align_tau` and
`hover_model.py` says of `k_lat`.

### 15.6 The drive: rail-limited, and why the square wave is already right

Measured per channel, from `calculations/coil_capacitors.xlsx`: $R = 12.6$ to $18\ \Omega$,
$L = 6.64$ to $6.85$ mH. At 92 Hz that is $X_L = 3.9\ \Omega$ against $R = 15\ \Omega$, so
$Q = 0.26$ and

- **the chain is essentially purely resistive.** Series resonance cancels $X_L$ and recovers
  3.3 %. It only begins to pay above about 350 Hz, where $X_L \approx R$. The 447 µF that would
  resonate 92 Hz is not worth fitting.
- **the binding limit is the supply, not heat.** 12 V across 15 $\Omega$ delivers 0.99 A of peak
  fundamental against a 2 A continuous rating, a factor of two in hand thermally and none at all
  electrically. Worth noting that the 1.25 A the model has always assumed needs about 15 V, not
  12.

**The first-harmonic theorem.** The cycle averages of §12.5 and §12.6 project onto $n = 1$:
$\langle\cos\theta\cos n\theta\rangle = 0$ for $n \ne 1$. So the waveform *shape* within a cycle
is irrelevant to both the torque and the force, and only the fundamental phasor matters. Every
harmonic dissipates $I_n^2R/2$ and contributes exactly nothing.

That settles the waveform question in a direction worth stating plainly. A bipolar square puts
$4/\pi = 1.27$ times its amplitude into the fundamental, so **under a voltage limit it beats a
sine by 27 %**. Under a thermal limit a sine would win by 19 %, because 19 % of a square's heating
sits in harmonics that average to nothing. The rail is what binds, so **the square drive already
in the firmware is the correct choice** and should not be changed. A trapezoid interpolates and
wins under neither.

Two places where a non-sinusoidal drive does earn something, both reachable with the existing
firmware and no new code:

- **envelope modulation** near the precession pole, through `addCarrier*RampTask`, as a route to
  the alignment damping §15.5 needs;
- **burst duty-cycling**, because the rotor's spin-down time constant $I_s\omega/(k_d f^2)$ is
  0.764 s at 92 Hz, seventy field periods, so it carries through an off time on inertia alone.

### 15.7 What was found

A design that is asymptotically stable, certified, and holds up nonlinearly:

| | |
|---|---|
| rotating rings | $r = 36.7$ mm at $z = -11.0$ and $+50.7$ mm, upper reverse-wound, pitch 21 mm |
| steady ring | $r = 60$ mm at $z = 19.9$ mm, 2.0 A |
| drive | $f = 110$ Hz, 0.97 A set by the 12 V rail, 28.4 W ohmic |
| rotor | $m_z = 3\%$ of $m_{dip}$ |
| trim | $z_{eq} = 19.92$ mm, $B = 2.10$ mT, $\sin\varphi = 0.478$ |

$$\lambda = \{-0.067 \pm 0.182j,\ -0.071 \pm 0.295j,\ -0.100 \pm 0.492j,\ -0.844 \pm 1.285j\}$$

All eight in the left half plane, so $A$ is Hurwitz and `certify` returns a $P$. Run open loop
with the currents fixed from 100 µm off axis, the nonlinear plant **holds for the full 6 s**,
ending 48.8 µm off axis with $z$ within 3 µm of trim. Against §14, where every design left a
10 mm box in about half a second, that is a change of kind and not of degree.

**Five channels, four driven, no new driver boards.** The upper ring is a winding reversal on the
existing four; the steady ring draws constant current and needs a bench supply, not an H-bridge.

### 15.8 What this does not settle

1. **The margin is small and the axial trap is soft.** $\sigma = 0.067\ \mathrm{s^{-1}}$ is a
   15 s time constant, and $\omega_z = 0.50$ rad/s is 0.08 Hz. This is stability, not authority.
2. **Camera clearance is 0.2 degrees.** The design sits on that constraint, and the constraint
   itself assumes the stereo pair sights *between* the coil arms rather than along them. Along
   them it fails by 8.4 degrees. Confirm the rig's actual azimuths before trusting it.
3. **It needs a robot change.** Without $m_z$ the fifth channel is inert, by §15.2.
4. **The field model is unvalidated against hardware.** There is no field measurement anywhere in
   this repository, and the coils carry a ferrite core the model does not represent, so absolute
   magnitudes are the least trustworthy numbers here and the ratios are the most. A single
   Hall-probe axial scan remains the highest-value missing measurement in the project.

### 15.9 Correspondence with the implementation

| quantity | where |
|---|---|
| $B \propto (z - z_c)$, the antiphase two-ring, constraints | `coil_geometry.py` |
| the steady field, and the axial moment it acts on | `spatial_model.dc_field_at`, `Plant._dc_terms` |
| fixed-current linearisation, Lyapunov, region of attraction | `stability_cert.py` |
| rail limit, resonance, waveform fundamentals | `drive_model.py` |
| the search, and the Pareto front | `optimise_array.py` |

```bash
uv run python ai/design/coil_geometry.py --self-check
uv run python ai/design/drive_model.py --self-check
uv run python ai/design/stability_cert.py --self-check
uv run python ai/design/optimise_array.py --self-check
uv run python ai/design/optimise_array.py
```

## 16. What the measurement noise does to the loop

§11 settles that this stage needs no observer of its own: `pose/filter.py` already
delivers the full eight-state $x = [r, v, \hat s]$, so the controller reads a state
rather than reconstructing one. That is still true, and nothing here adds a second
Kalman gain. What it does add is the number that was missing from three separate
decisions in this chapter.

The measurement is now characterised on the bench rather than assumed --
[ch. 3 §18](../pose/theory.md) derives the model and `pose/noise.py` fits it. Two of
its outputs reach this stage, and they are not interchangeable.

**$\sigma_{\text{total}}$, the whole scatter**, is what the pose filter carries as
$R$. It is dominated by a drift with a correlation time of order hundreds of
milliseconds, which no filter removes and which therefore lands in the loop as a
slowly varying position bias -- an error the integrator will faithfully chase.

**$\sigma_{\text{white}}$, the frame-to-frame part**, is the only component that
reaches a *rate*. A first difference subtracts two samples that share the drift, so
it cancels; what survives is the white part alone, amplified by $1/\Delta t$ and
then low-passed. With $a = \Delta t/(\tau+\Delta t)$,

$$\sigma_v = \frac{\sigma_w\sqrt2}{\Delta t}\sqrt{\frac{a}{2-a}}$$

### 16.1 The three constants this fixes

**`z_track.TAU_ZDOT_S` and `predictor.TAU_VEL_S`** are the same time constant and
must move together. The equation above prices any candidate $\tau$, and
`noise.NoiseModel.tau_for_velocity_sigma` inverts it. The bound worth knowing is
that raising $\tau$ past the measured correlation time averages over samples that
are not independent: it buys lag and nearly no noise. At 80 ms against a correlation
time of order 400 ms, both constants already sit inside that regime, so the lever
here is smaller than it looks.

**`z_track.STEPOUT_ZDOT_MPS`** used to be justified in a comment as "16 sigma" of a
$\dot z$ noise of 9 mm/s, itself derived from an assumed 0.5 mm per frame that
nothing had measured. Tripping step-out spuriously commands full torque at a robot
that is flying correctly, so the margin has to be real. `check_stepout_margin`
recomputes the multiple from whatever model is on disk and `demo()` asserts it
against `STEPOUT_MIN_SIGMA`; with no calibration recorded it says the threshold is
unjustified rather than passing quietly.

**The Bryson weights in `design_hover_lqr`** put $1/(0.010)^2$ on $x$, meaning 10 mm
of lateral deviation is worth one unit of cost. A measurement whose own noise is a
large fraction of that makes the controller spend authority chasing noise, so the
design now warns when the measured lateral $\sigma$ exceeds a tenth of the weight.

### 16.2 Correspondence with the implementation

| Model element | Code |
|---|---|
| Rate noise from the white component | `noise.NoiseModel.velocity_sigma_mm_s` |
| Choosing $\tau$ from a noise budget | `noise.NoiseModel.tau_for_velocity_sigma` |
| The shared rate time constant | `z_track.TAU_ZDOT_S`, `predictor.TAU_VEL_S` |
| Step-out margin, checked against the measurement | `z_track.check_stepout_margin`, `z_track.demo` |
| Bryson weight sanity against measured noise | `design_hover_lqr._noise_provenance`, `design` |
| What noise a gain file was tuned against | the `noise` block in `control/hover_controller.json` |

The gain file records the noise it assumed. Gains that cannot be audited against the
conditions they were tuned for are the same failure as a constant with an expired
reason attached, which [ch. 3 §16.26](../pose/theory.md) had already paid for twice.

## 17. Starting from rest: capture, and two ideas that do not work

Sections 4 and 5 model a rotor that is *already* synchronised, and so does every
simulation in this repository -- the GUI starts at $\delta(0)=\arcsin(1/M)$,
$\omega(0)=2\pi f_0$ (section 5.3). Capture from rest is unmodelled, and it is the thing
that actually fails: takeoff locks at 5 Hz on some runs and not others.

**Intermittency is the diagnosis.** A torque shortfall fails the same way every time. A
capture failure fails at random, because whether a stationary rotor is caught by a field
that is already rotating depends on the angle it happens to be resting at. Starting the
ramp at $f_{start}$ asks the rotor to absorb $f_{start}$ of slip instantly at $t=0$; from
the swing equation, $\dot\delta = 2\pi f_f - \omega$ with $\omega(0)=0$, so $\delta$ runs
at $2\pi f_{start}$ and there is only a torque *pulse* of duration $\sim 1/(2 f_{start})$
in which to be caught before the field runs away. Capture is therefore a race, decided by
the initial angle.

**Align-and-ramp removes the race.** Hold a *static* field first. The rotor is pulled to
the one stable angle of that field, so its initial condition stops being random. Then
begin rotating from exactly that angle with $f \to 0^+$: initial slip is zero and capture
is by construction, not by luck. This is the classical open-loop start for a PMSM without
position feedback, and it needs no sensing whatsoever.

The hold must outlast the ring-down, not merely the movement. Section 5.5 puts the phase
mode at $\omega_n \approx 121$ rad/s (19.2 Hz) with $\zeta \approx 0.0245$ -- about 20
cycles, so several hundred milliseconds.

**It does not work on this hardware, and the reason is the same series capacitor that
shapes everything else in section 18.** Align-and-ramp was implemented as an `ALIGN` state
in `src/main_flight.cpp` and measured on 2026-08-29: the entire hold logged $I = 0.00$ A on
all four channels. A static field is DC, a capacitor in series with each coil blocks DC by
construction, and a coil carrying no current pulls the rotor nowhere. The alignment was not
weak, it was absent -- the rotor sat wherever it already was and the ramp started from the
same random angle as before.

The argument above is not wrong; it is inapplicable. The classical PMSM open-loop start
assumes a coil that can be driven at DC, and this one cannot. Reviving it needs hardware:
a bypass across the series capacitor during start-up, or a switched bank. The `ALIGN` state
has been deleted rather than left at `ALIGN_MS = 0`, because a disabled state that reads as
implemented is worse than no state at all. What survives is the one mechanism that does
work -- `PwmController::setGlobalFrequency`'s `leavingDc` branch resumes rotation at the
angle being held -- which is still what makes a start from $f \to 0^+$ continuous.

### 17.1 Why the coils cannot be run at resonance at every rotation rate

The obvious wish is to drive the coils near their electrical resonance -- where current,
and so torque, is greatest -- regardless of how fast the field is turning. In this
topology that is not available, and the reason is structural rather than a limitation of
the firmware.

Each coil carries a square wave at the field frequency with a fixed phase offset equal to
its azimuth, $i_k(t) = I\sin(2\pi f t - \alpha_k)$, so

$$\mathbf{B}(t) = \sum_k A\, i_k(t)\, \hat u_k \propto (\cos 2\pi f t,\ \sin 2\pi f t)$$

a vector of constant magnitude rotating at $f$. **The coil current alternates at exactly
the field rotation rate.** Resonance is a property of the electrical frequency, so the
two are the same number and cannot be chosen independently.

The natural workaround -- hold the electrical frequency at resonance $f_c$ and rotate
slowly by modulating each coil's amplitude, $A_k(t) = \cos(2\pi f_s t - \alpha_k)$ --
fails for a different reason:

$$\mathbf{B}(t) = \sin(2\pi f_c t)\sum_k A_k(t)\hat u_k = \sin(2\pi f_c t)\,\mathbf{B}_s(t)$$

This is a slowly-rotating vector multiplied by a scalar that **changes sign every half
cycle** of the carrier. The rotor cannot respond at $f_c$, so it sees the carrier-cycle
average of the torque, which is zero. The rotor buzzes; it does not turn. Decoupling
rotation rate from electrical frequency needs hardware -- a switched capacitor bank, or a
bypass across the series capacitor during start-up -- not a drive waveform.

### 17.2 Why field-oriented control is not available here

FOC would end the question of synchronism outright: reference the field to the *measured*
rotor angle and step-out becomes impossible by construction. It is blocked by sensing, not
by software.

Commutation needs **signed per-phase current, sampled synchronously with the PWM**. The
VNH5019 `CS` pin gives an unsigned magnitude, and it is sampled asynchronously at about
1 kHz (`PwmController::_serviceCurrentLoop`, gated on `dtSenseMs >= 1.0f`) -- roughly 7
samples per electrical cycle at 150 Hz. That is enough to regulate *amplitude*, which is
what `CurrentBalanceController` does, and not enough to resolve an angle.

The only rotor-angle observable in the system is vision, and section 12 of `pose/theory.md`
shows why it cannot serve: roll about the spin axis is unobservable from the rim, and the
blades alias above $\text{fps}/8$. `controller/pose/spin.py` reads blade phase where it is
valid and refuses to answer where it is not. Angle-referenced control therefore waits on
either a rotor-angle sensor or per-phase current sensing; neither exists on this board.

**Section 22 does not overturn this, and the distinction is the whole reason it is a
separate section.** The lock-in in `coil_probe.cpp` does extract a current angle from the
same unsigned `CS` pin -- but off the flight path, at one held frequency, averaged over
hundreds of cycles, to produce two calibration constants per channel. Commutation needs the
angle *signed, every cycle, in real time*, and none of the three things this section rules
out are made available by averaging. What 22 buys is a better open-loop command, not a
closed loop on rotor angle.

## 18. The spin-up ramp, and what it can and cannot explain

The ramp from rest to hover is the only fully open-loop part of a flight: nothing measures
the rotor while it runs, so every failure inside it is inferred after the fact from what
the field was doing when the robot stopped following. This section computes what the ramp
is allowed to do, compares that against what it actually does, and finds the two disagree
in a way that rules out the obvious explanation.

The reported symptom, 2026-08-29: **the robot steps out consistently at 40-60 Hz, and the
frequency then jumps to about 100 Hz.**

### 18.1 The two constraints, and they are not the same constraint

A ramp $f(t)$ from rest has to satisfy two different things, at two different moments.

**Capture, at $t = 0$.** The field starts at $f_{start}$ against a rotor at $\omega = 0$, so
the swing equation $\dot\delta = 2\pi f_f - \omega$ gives $\dot\delta(0) = 2\pi f_{start}$:
the whole starting frequency appears instantly as slip. There is a single torque pulse of
duration $\sim 1/(2 f_{start})$ in which to be caught. Equating the impulse available to the
momentum needed gives the pull-in frequency of section 17,

$$f_{pull} = \sqrt{\frac{\tau_{max}}{4\pi J}}$$

implemented as `z_track.pull_in_hz`. Capture requires $f_{start} < f_{pull}$.

**Following, for $t > 0$.** Once locked, the rotor must supply drag *and* angular
acceleration out of the same torque budget, at a load angle bounded by $s_{lim}$:

$$k_{drag} f^2 + 2\pi J \dot f \;\le\; s_{lim}\,\tau_{max}(f)
\qquad\Longrightarrow\qquad
\dot f_{max}(f) = \frac{s_{lim}\,\tau_{max}(f) - k_{drag} f^2}{2\pi J}$$

implemented as `TorqueLimits.f_dot_max`. Note $\tau_{max}$ is a function of $f$, not a
constant: the coil is a series RLC and its current -- and so its torque -- rises with
frequency below resonance. This is why the ceiling is a curve and not a number.

### 18.2 The EASE curve has one closed-form property that matters

The firmware ramps with `TaskMode::EASE`, the symmetric sigmoid

$$\sigma(t) = \frac{t^k}{t^k + (1-t)^k}, \qquad f(t) = f_0 + \Delta f \,\sigma(t)$$

Differentiating at the midpoint, where $t^k = (1-t)^k = 2^{-k}$:

$$\sigma'(1/2) = \frac{u'v - uv'}{(u+v)^2}\bigg|_{t=1/2}
= \frac{k\,2^{2-2k}}{2^{2-2k}} = k$$

**The peak ramp rate is exactly $k$ times the average rate**, $\dot f_{peak} = k\,\Delta f/T$,
and it occurs at the midpoint of the ramp in time -- which for a sigmoid is the midpoint in
frequency too. $k = 1$ is exactly linear. This is the whole tuning surface of the curve, and
it is why `k` is exposed on the wire (`takeoff=<start>:<end>:<ms>:<k>`) rather than compiled in.

For $3 \to 160$ Hz over 40 s at $k=2$:

| $t/T$ | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 |
|---|---|---|---|---|---|---|---|
| $f$, Hz | 12.2 | 27.4 | 51.3 | 81.5 | 111.7 | 135.6 | 150.8 |

At $k=1$ the same endpoints give a straight 34.4 / 50.1 / 65.8 / 81.5 / 97.2 / 112.9 / 128.6.

### 18.3 The margins, computed

**Recomputed 2026-08-30** against `TorqueLimits` as it now stands (`F_STEPOUT_HZ = 225`
MEASURED, `RESISTANCE_OHM = 6.9`, `CAPACITANCE_F = 400` uF, `s_lim = 0.8`, $k_{drag}$ from
`hover_model.fit_k_drag`). The figures this section carried until then were computed
against `f_stepout = 190` and `R = 1.7`, both of which the same changeset replaced -- so
every number below moved, and the conclusions moved with two of them:

| $f$, Hz | 4 | 6 | 10 | 20 | 30 | 50 | 80 | 120 | 160 |
|---|---|---|---|---|---|---|---|---|---|
| $\dot f_{max}$, Hz/s | 51.7 | 77.1 | 126.7 | 240.5 | 335.3 | 461.3 | 519.8 | 446.6 | 267.4 |
| commanded, Hz/s | 0.72 | 1.37 | 2.29 | 3.96 | 5.20 | 6.89 | 7.8 | 7.4 | ~1 |
| margin | 72x | 56x | 55x | 61x | 64x | 67x | 67x | 60x | large |

**The ramp is nowhere near its following limit -- roughly 40x clear at every point,
including the 40-60 Hz band where the failure is reported.** For the mid-ramp slope to be
the binding constraint at 50 Hz, $\tau_{max}$ would have to be overestimated by a factor of
about 45.

Capture is a different story:

$$\tau_{max}(3\ \text{Hz}) = 1.03\times10^{-6}\ \text{N m}
\quad\Longrightarrow\quad f_{pull} = 4.94\ \text{Hz}$$

against a 3.0 Hz start. **A margin of 1.65x, and it is still the tightest number anywhere
on the ramp by a factor of forty** -- the ranking survived the constant change even though
both numbers moved (it was 4.17 Hz and 1.4x at `f_stepout = 190`).

And `z_track.py` contains **two torque models that disagree about where capture is even
possible.** `takeoff_start_hz` takes torque linear in $f$ from an anchor
$\tau_{ref} = 5 k_{drag} f_{160}^2$ (the 5 is a bare margin factor with no citation) and
puts the crossing $f_{pull}(f) = f$ at **7.9 Hz**. `TorqueLimits.tau_max(f)`, built from
`coil_gain(f)` and anchored on `F_STEPOUT_HZ`, now puts the same crossing at **8.07 Hz**.

**The two models used to disagree by 1.4x; at the measured `F_STEPOUT_HZ = 225` they
agree to within 2%.** That is the single largest change in this section, and it retires
the argument that follows from the disagreement -- there is no longer a "failing side of
one model and the limit of the other":

| ramp start | $f_{pull}/f_{start}$ by `tau_max` | capturable by `takeoff_start_hz` |
|---|---|---|
| 2.8 Hz | 1.70x | yes |
| 3.0 Hz | 1.65x | yes |
| 5.0 Hz | 1.27x | yes |
| 6.0 Hz | 1.16x | yes |
| 7.9 Hz | 1.01x -- marginal | marginal (this is its crossing) |

**Every run before 2026-08-29 started at 7.9 Hz**, because the host asked
`model_start_hz` -> `takeoff_start_hz` for the number and sent `takeoff=7.9:150`. At the
retired `f_stepout = 190` that start computed as 0.85x -- on the failing side -- which is
what motivated dropping it. At the measured 225 it computes as 1.01x: **marginal rather
than doomed.** So the 7.9 Hz start is no longer an explanation for those failures, and
this section no longer offers one; what it offers is a start with real margin. The ramp
start is now `constants.RAMP_START_HZ = 2.0`, reached through `ramp.DEFAULT`'s first
segment, and the model call is gone from the host. The firmware carries no start of its
own any more: `RAMP_START_HZ` and the `takeoff=` form that used it were deleted
2026-08-31, so there is exactly one number and the host owns it.

Starting low also costs nothing that matters. The old worry was that a low start pushes with
almost nothing (0.2 Hz drew a measured 0.00 A), but 18.3 shows the following margin is ~40x
throughout, so time spent climbing out of the weak band is not time the rotor can fall behind
in. And 3.0 Hz sits just under the 25 fps alias limit, which is the only reason the capture
question is answerable at all. Because $f_{pull} \propto \sqrt{\tau_{max}}$,
capture fails if $\tau_{max}$ is overestimated by 2.7x. That is a wider door than the 1.9x
this said at `f_stepout = 190`, and the anchor is no longer a guess: `F_STEPOUT_HZ = 225`
is labelled MEASURED, from ramps to 240 and 230 Hz where lift peaked near 180-210 and
collapsed in the 220-240 bin. The residual uncertainty has moved off the step-out and onto
$C$ -- see 18.6, and `f_ceiling()` is now 201.2 Hz, not the ~167 the code comment quoted.

### 18.4 What this rules out, and what it leaves

The hypothesis this section was written to test -- that the S-curve's steep middle
coincides with the 40-60 Hz band and tears the rotor out of sync -- **is not supported**.
The coincidence is real (the steepest part of a $k=2$ ramp does sit near the reported
band) but the magnitude is wrong by a factor of forty. Recorded here as a negative result
so it is not re-derived.

Three explanations survive, and they are distinguishable by measurement rather than by
argument:

1. **Capture never happened.** The rotor is never caught at 3 Hz, so it never turns at all.
   The field then sweeps past a stationary rotor, and an observer watching the telemetry
   sees the drive frequency climb through 40-60 Hz and on to 100 while nothing flies. On
   this reading the "step-out frequency" is not a property of the rotor at all; it is
   simply where someone notices. Section 17's intermittency argument predicts exactly this,
   and the blade tracking already reports the rotor dead at 1.2 Hz and turning by 2.8 Hz --
   which puts `RAMP_START_HZ = 3.0` barely on the right side of the line.
2. **$\tau_{max}$ is overestimated.** Everything above is anchored on `F_STEPOUT_HZ`, a
   guess. If the true torque scale is smaller, capture fails first (it needs only 1.9x) and
   the following margin shrinks with it.
3. **The target was below the hover point** -- certainly true, and independent of any model.
   See 18.5.

The discriminator is the rotor, not the altitude -- but the sensor available is weaker
than one would like, and its weakness is asymmetric in a way that happens to be useful.

`f_hz` in the flight log is the *commanded field*. `spin` is `pose/spin.py`'s
`SpinWitness.turning`, and it is deliberately **not** a rate: above $\text{fps}/8$ the
blade phase aliases and the rate is unrecoverable, so the class returns a three-state
motion latch instead of a confident wrong number. The asymmetry is the point:

- **`turning` is provable at any speed.** A median per-frame phase step above threshold
  means the rotor moved, whatever the aliasing does to the magnitude.
- **`stopped` is only emitted below $\text{fps}/8$.** At a multiple of $\text{fps}/4$ a
  spinning rotor renders an identical frame every time, so a still phase is evidence of
  nothing unless the commanded field is known to be below the alias limit.
- **blank is unknown**, and must never be read as stopped.

With `stereo_frames` building the witness at 25 fps the alias limit is 3.125 Hz -- which
sits a hair above `RAMP_START_HZ = 3.0`. So the sensor is decisive in *both* directions
exactly over the pull-in window, which is the one place 18.3 says the margin is thin, and
decisive in *one* direction (motion, not stillness) for the rest of the ramp. Hypothesis 1
is therefore falsifiable as stated: if the rotor is never captured, the opening rows read a
confident `stopped`.

What no camera in this rig can do is measure rotor *rate* through 40-160 Hz. Even
`spin.probe()`'s 210 fps single-camera path aliases above $210/8 \approx 26$ Hz. The
strongest mid-ramp statement available is a lower bound: the highest $f$ at which a
`turning` row appears. `takeoff_report.py` reports that bound and the three-state capture
verdict, and does not attempt to infer a step-out frequency from blanks.

**The instrument was inoperative until 2026-08-29, and nobody noticed.** `turning` returns
`False` only inside `if self.field_hz is not None and self.field_hz <= self.alias_limit_hz`,
and `live_viz.stereo_frames` called `w.update(t, frame, ellipse)` with no `field_hz` at all.
So the confident-standstill branch was unreachable in every run this rig has ever flown: the
witness could report TURNING or unknown, never STOPPED. The capture question was not
unanswered, it was unanswerable. The runner now assigns `tick.spin.field_hz = link.freq`
each tick -- the frequency the *firmware reports*, not the one the host intended, so the
gate stays shut until telemetry actually arrives.

The alias limit also needs the true frame rate, and the honest value is not the camera's.
The pose pipeline is compute-bound rather than clocked -- segmenting a stereo pair costs
about 50 ms against a sensor that can deliver 119 fps -- so passing the capture rate would
inflate $\text{fps}/8$ roughly fivefold and manufacture confident `stopped` verdicts across
a band where nothing is resolvable. That is the dangerous direction to be wrong in. The rate
is therefore measured in the loop as an EWMA of the frame interval and pushed onto the
witness, which self-calibrates and errs safe: a dropped frame lengthens the interval, lowers
the estimate, and narrows the band in which `stopped` will be committed.

**This leaves a narrow but real measurement window at the start of the ramp**, bounded on
three sides by numbers that are now all known:

$$\underbrace{2.8\ \text{Hz}}_{\text{torque floor: rotor dead below}}
\;<\; \underbrace{3.0\ \text{Hz}}_{\texttt{RAMP\_START\_HZ}}
\;<\; \underbrace{3.1\ \text{Hz}}_{\text{alias limit at 25 fps}}
\;<\; \underbrace{4.2\ \text{Hz}}_{f_{pull}}$$

The ramp start has to clear the torque floor (blade tracking puts the rotor dead at 1.2 Hz
and turning by 2.8 Hz) and stay under $f_{pull}$ for capture. It should *also* stay under
the alias limit for the first samples, or the one measurement that settles hypothesis 1 is
never taken. At 25 fps all four constraints are satisfiable and 3.0 Hz sits inside the
window, with about 0.1 Hz to spare on the measurement side. If the achieved pipeline rate
turns out lower than 25 fps the alias limit falls below the ramp start and the window
closes -- in which case lower the first segment's start frequency in `ramp.DEFAULT`
toward 2.9, which improves the capture margin and the measurability at the same time. Do
not widen it by lying to the witness about fps.

### 18.5 A ramp that stops below the hover point cannot lift the robot

The one defect here that needs no model. Takeoff is observed at about 160 Hz. The firmware
carried `HOVER_HZ = 150.0f`, and used that single constant for two different jobs: the
default ramp target, *and* the upper clamp on the `takeoff=` argument. So a host asking for
`takeoff=3:160` was silently given `takeoff=3:150`, and no amount of ramp shaping or loop
tuning could have produced a flight. It also poisoned identification downstream, because
`_anchor_controller` read back the frequency the ramp reached and re-centred the whole
controller on it -- anchoring the loop to the number where the ramp was cut off rather than
to a hover point.

Two constants doing two jobs is the failure. The default target and the ceiling are now
separate concerns: the target is a default (`HOVER_HZ = 160.0f`), and there is no ceiling
at all -- the host owns every real limit, per section 4's division of labour.

### 18.6 The capacitor that is not where the model puts it

`z_track.py` carries $L = 1.4$ mH (confirmed) and $C = 400\ \mu$F (fitted). Together:

$$f_0 = \frac{1}{2\pi\sqrt{LC}} = 213\ \text{Hz}, \qquad
Q = \frac{1}{R}\sqrt{\frac{L}{C}} = 0.27$$

Both numbers are hard to reconcile with the rest of the repository. $Q = 0.27$ is
R-dominated -- there is no resonance to speak of -- whereas `S_LIM = 0.8` is justified
against "the $Q \sim 22$ phase swing at ramp ends", and takeoff is reported to happen at
160 Hz "around the resonance of the coils". $L$ is the trusted value, so $C$ is the
suspect, and its provenance says why: it was fitted from a 0.2-60 Hz ramp, the one band
where the capacitive term dominates and the inductive term barely participates. The fit
could not have constrained $L$, and a $C$ fitted with $L$ unconstrained is only as good as
the band it was taken in.

A ramp to 160 Hz is the first data this rig has through 100-200 Hz, which is exactly the
band that separates the two. `takeoff_report.py` refits both from $|I|(f)$ over the ramp
and reports the implied $f_0$ and $Q$ beside the current constants. Until that lands, the
resonance is unknown and any statement that 160 Hz is "at resonance" is unsupported.

**Superseded twice over as of 2026-09-01, and still unresolved.** The capacitor bank was
doubled to 800 µF, so every fitted $C$ above is void -- $f_0 \propto C^{-1/2}$, and the
nominal moves to 150 Hz. And the question is now known to be *per channel*: 22.2 shows the
four channels each have their own $(f_0, Q)$, that only those two numbers are identifiable
from $|I|(f)$ at all, and that their spread is what puts the coil currents out of phase with
each other. `takeoff_report.metrics` fits all four separately under `phase_fits`, and
`coil_phase.fit_channel` **refuses** a band whose peak sits at an edge -- which is the
specific failure that produced the 400 µF above, now guarded rather than merely regretted.

### 18.7 Correspondence with the implementation

| Quantity | Where |
|---|---|
| $\sigma(t)$, EASE, $\sigma'(1/2) = k$ | `PwmSequencer::applyCurve`, `TaskMode::EASE` |
| ramp schedule, 25 ms grid | `PwmSequencer::compile`, `run` |
| the profile: shape, validation, `seq=` encoding | `controller/control/ramp.py` -- **the only place a ramp is defined** |
| the numbers it is built from | `controller/control/constants.py`, `RAMP_*` |
| `seq=clear` / `seq=ramp:...` / `seq=go` | `dispatch` in `src/main_flight.cpp` |
| the profile a run actually flew | `# ramp: <label>`, first line of that run's CSV |
| $\dot f_{max}(f)$ | `z_track.TorqueLimits.f_dot_max` |
| $f_{pull}$ | `z_track.pull_in_hz`, bracketed by `takeoff_start_hz` |
| $f$ commanded vs rotor turning | `f_hz` vs `spin` in `results/takeoff/*.csv` -- `spin` is the witness latch (`turning`/`stopped`/blank), deliberately NOT a rate |
| liftoff knee, capture verdict, $L$/$C$ refit | `controller/control/takeoff_report.py` |

**`seq=` is the only ramp path, and the second one was deleted rather than repaired.**
Until 2026-08-31 the firmware carried its own spin-up: a bare `takeoff` command with
`RAMP_START_HZ` / `HOVER_HZ` / `CLIMB_MS` compiled in, plus a `takeoff=<a>:<b>:<c>:<d>`
parser. Three defects came with it, and all three were properties of the path existing at
all rather than of any one line:

1. The ramp compiled in `setup()` was **dead**. `PwmSequencer::compile` only records
   initial state and does nothing until `start()`, and every start path re-added and
   re-compiled first.
2. The `takeoff=` parser was **unused** by the host -- a strict subset of `seq=`, one
   segment, always `TaskMode::EASE`.
3. A bare `takeoff` ramped **linear, k=1**, because the parser's default `p[3] = 1.0f`
   disagreed with the `2.0f` in the dead `setup()` copy. By 18.3 that crosses the ~4.9 Hz
   pull-in window in 0.2 s and never captures the rotor: the command spun the field and
   left the rotor sitting still.

Deleting the path removed all three at once and took `HOVER_HZ`, `CLIMB_MS` and
`RAMP_START_HZ` with it, so no frequency is defined in two places. The cost, accepted
deliberately: **the board can no longer be spun up from a serial monitor without the
Python host.** `seq=clear` / `seq=ramp:` / `seq=go` is the only way in, and
`link._note_drive` no longer counts `takeoff` as drive -- if a second spin-up path is
ever added it must be re-added there, or its heat goes unaccounted.

The general lesson is the one worth keeping: two ways to express the same ramp will drift,
and the copy nobody runs is the one that drifts furthest. It had a different curve
exponent from its twin fifteen lines away, and nothing caught it because nothing ran it.

### 18.8 MEASURED 2026-08-29: the rotor DOES track to 170 Hz -- and still does not fly

This section first concluded that the rotor pulls out at 25 Hz and the rig was
torque-starved by an order of magnitude. **That was wrong**, and the way it was wrong is
worth keeping, because the same mistake is easy to repeat.

**The retracted result.** A stepped sweep -- hold a frequency, probe, jump to the next --
showed clean tracking to 20 Hz and OSCILLATING by 27 Hz. Read as a pull-out limit, it
implied 2.4% of the thrust needed at 160 Hz. But an abrupt step between two *held*
frequencies is not the same experiment as a ramp: it demands the whole slip instantly,
exactly the capture race of 18.1, and it fails far below the frequency a smooth ramp
sustains. Ramp-rate tests then showed 40 Hz reached comfortably at 4-19 Hz/s and lost only
at 76 Hz/s, which already contradicted the 25 Hz figure.

**The test that settled it, and it uses the blur as evidence rather than fighting it.**
Above roughly 50 Hz the blades smear within one 4.8 ms exposure -- 90 degrees of rotation,
a full blade pitch -- so `blade_phase` finds no harmonic and `probe` returns nothing. That
silence is itself a measurement:

| condition | samples in 4 s | inference |
|---|---|---|
| coils off, rotor still | 408 | the probe works |
| **holding 170 Hz** | **0** | blades smeared, so rotor **> ~50 Hz** |

and the coast-down confirms it directly. Cutting the drive at 170 Hz and probing for 6 s
caught the rotor decelerating through the measurable band -- 14.3 Hz at 2.6 s, 10.7 at 3.8,
4.7 at 4.9, 2.35 at 5.7 -- 1187 samples of a real spin-down. A rotor that had been sitting
still reads a flat zero instead. **The rotor is synchronised at 170 Hz.**

The deceleration is also informative. Aerodynamic drag alone gives $\dot f \propto -f^2$,
so $1/f$ would rise linearly; it does not. The measured rate is closer to constant,
-3 to -5 Hz/s across the band, which is the signature of a **Coulomb friction** torque
rather than aerodynamic load. That is a bearing, not the air.

**So torque is not the problem, and neither is the ramp, the start frequency, the ramp
shape nor the controller.** The rotor spins where it is told, in the right direction
(18.9), and produces no measurable lift: $z_{max} = 0.1$-$0.3$ mm against a 0.07 mm pad
scatter, across four ramps to 160, 170 and 200 Hz. The open question moved from *does it
spin* to *why does spinning produce no thrust* -- blade pitch, blade area, rotor mass, or a
mechanical restraint on the body.

**Two instrument lessons, both already written down and both still catching people.**
`SpinWitness.turning` reported `turning` at every frequency, which is correct and useless:
it is a motion latch, and a rotor rocking 10 degrees is moving (18.4). And a sweep stepped
from a standing start at 5 Hz reads OSCILLATING at every later frequency, because pull-in
is ~4.2 Hz and every step after the first inherits a stationary rotor. Capture low, ramp
smoothly, and use the blur.

**An attempt to measure blade drag from the coast-down FAILED. Recorded so it is not
retried the same way.** The idea was sound -- with the drive cut, the only torques left are
aerodynamic and friction, so the spin-down curve is the load -- and a first pass appeared to
give $k_{drag} \approx 5.9\times10^{-11}$, some 7x under
`hover_model.fit_k_drag`'s $3.91\times10^{-10}$. **That number is withdrawn.** A dedicated
run releasing from 170, 120 and 80 Hz, with the camera held open so $t=0$ was exact, showed
the method does not work:

| t after release (s) | from 170 Hz | from 120 Hz | from 80 Hz |
|---|---|---|---|
| 0.2 | -- | 29.6 | 2.4 |
| 0.4 | 32.6 | 2.9 | 2.4 |
| 0.8 | 2.6 | 3.3 | 25.9 |
| 1.5 | 33.2 | 21.9 | 8.4 |
| 3.0 | 16.3 | 30.0 | 21.1 |

Three releases from wildly different speeds give interleaved, non-monotonic garbage. The
cause is the instrument, not the rotor: above $\text{fps}/8 \approx 26$ Hz the blade phase
aliases, `np.unwrap` then picks the smallest step consistent with each pair of frames, and
differentiating that yields a number that looks like a rate and is not one. **A plausible
decreasing sequence out of `gradient(unwrap(phase))` is not evidence** -- the earlier
"14.3, 10.7, 4.7, 2.35 Hz" coast came from the same computation and cannot be trusted
either.

What survives is the part that does not depend on a rate at all: **holding 170 Hz yields
zero blade samples against 408 on a still rotor**, and an absence of signal is unambiguous
where an aliased rate is not. The blades are smeared, so the rotor is turning fast. The
rotor tracks; the aerodynamic coefficient remains unmeasured on this rig.

Measuring it properly needs an instrument that resolves rate above 26 Hz, and this rig has
none: exposure is not controllable through OpenCV on macOS (AVFoundation returns -1 and
every `set` fails), and the 320x240 / 364 fps mode is a sensor crop in which the rotor
overflows the frame and blade strength collapses from 0.545 to 0.033. A tachometer, an
optical interrupter, or a camera moved back far enough to use the cropped mode would each
close it.

**Coil measurements that survived the retraction.** Current peaks near 170 Hz -- 0.95 A at
20-40 Hz, 4.08 A at 160-180, falling to 2.93 A by 200-220 -- so series resonance is about
174 Hz, not the 213 Hz that $L = 1.4$ mH with $C = 400\,\mu$F predicts, and driving above
~180 Hz *reduces* torque. The best fit from the 120 s ramp is $L = 2.49$ mH,
$C = 334\,\mu$F, $f_0 = 174$ Hz, $Q = 0.40$. Separately, at $f > 140$ Hz the four channels
drew 3.39, 4.70, 4.73 and 3.77 A at equal commanded duty -- a 39% spread, so the rotating
field is measurably elliptical, and this is the first real evidence for re-enabling
`CurrentBalanceController`.

**The $L$/$C$ pair here is void as of 2026-09-01**, when the bank went to 800 µF: $f_0$
scales as $C^{-1/2}$, so this 174 Hz peak maps to 112 Hz against the 334 µF fitted here, or
138 Hz against the write-up's selected 500 µF. That those two disagree is the same
under-constrained fit 18.6 describes, and neither may be written into `constants.py` as a
measurement. $Q = 0.40$ is the closest thing to a measured $Q$ this project has, and 22.3
sets it against the three other values in the tree, which span 0.09 to 22. The 39 % current
spread survives the bank change untouched -- it is an amplitude fact, not a resonance one.

### 18.9 The reappearance test: a rate-free way to bound rotor speed

Every attempt in 18.8 to measure rotor rate above $\text{fps}/8 \approx 26$ Hz failed,
because aliasing turns `gradient(unwrap(phase))` into confident noise. The way out is to
stop asking for a rate and ask a question the instrument can actually answer.

**The blades smear at a known speed.** With a 4-blade rotor the phase pattern repeats every
90 degrees, and the exposure is one frame period, 4.8 ms at 210 fps. The signal disappears
when the rotor turns a full blade pitch within one exposure:

$$f_{smear} = \frac{360/B}{360\,t_{exp}} = \frac{1}{B\,t_{exp}} \approx 52\ \text{Hz}$$

So **presence or absence of blade signal is a clean threshold detector at 52 Hz**, and an
absence cannot be aliased the way a rate can.

**The test.** Cut the drive and time how long until blade signal reappears. A rotor released
at $f_0$ must coast from $f_0$ down to $f_{smear}$ first, and the torque that implies is
$\tau = 2\pi I (f_0 - f_{smear})/t_{reappear}$ -- a number to compare against what the
system could possibly supply. Measured 2026-08-29:

| released at | signal returns | implied braking torque |
|---|---|---|
| 170 Hz | 0.06 s | $4.1\times10^{-5}$ N m |
| 120 Hz | 0.11 s | $1.3\times10^{-5}$ N m |
| 80 Hz | 0.01 s | $5.8\times10^{-5}$ N m |

Peak available *drive* torque is of order $10^{-6}$ N m. These demand ten to a hundred times
that, from a rotor that is coasting with no drive at all. **The rotor was never at the
commanded frequency.** The times are also unordered in $f_0$, which is what you see when the
rotor sits at the same speed in every case: just above 52 Hz, where the signal returns almost
at once.

**So the rotor saturates near 52-60 Hz however hard the field is driven**, and that finally
explains the flights. Lift goes as $f^2$, so a rotor pinned at ~55 Hz against a hover point
near 160 Hz delivers $(55/160)^2 \approx 12\%$ of the thrust needed -- and $z_{max}$ across
ramps to 160, 170, 200 and 250 Hz was 0.1 to 1.0 mm against 0.07 mm of pad scatter.

**The mechanism is friction, not the drive.** The rotor rides a bearing on the takeoff rod.
A Coulomb torque there caps rotor speed independently of drive frequency -- which is exactly
the signature of a ceiling that does not move when the field is pushed from 80 to 250 Hz --
and the same friction opposes the vertical motion directly. It also explains why the drive
side kept looking healthy: capture works, tracking to 20 Hz is exact, and the coils reach
their resonance current. None of that is the binding constraint.

Two remedies need no hardware change. Reducing the friction itself is the direct one. The
other is **dither**: `dither=<hz>:<pct>` in `src/main_flight.cpp` modulates the collective,
and so the axial force, at a few tens of Hz. It shakes field *strength*, not direction, so
it adds an axial ripple without steering the disk -- the standard way to break stiction.
Off by default.

### 18.10 Why optical rotor-rate measurement is dead above 26 Hz here

Three separate attempts to measure rotor rate above $\text{fps}/8 \approx 26$ Hz failed,
each in a different way, and all three produced numbers that looked like data. Recorded
together so the fourth attempt is not made.

**1. Aliased rate.** `gradient(unwrap(phase))` above the alias limit yields a plausible
decreasing sequence that is pure noise -- see 18.8.

**2. Readable-or-not as a binary.** `MIN_STRENGTH = 0.15` is a low bar, and a partly
smeared rotor still clears it. A demo video settled it: at 20 Hz blade strength is 0.683
with blades clearly resolved; at 90 Hz it is 0.268 -- visibly smeared into a near-uniform
disc -- yet still counted "readable", which made a fast rotor look slow.

**3. Strength as a continuous smear proxy. This one is the trap worth naming.** Strength is
not monotonic in rotor speed, because a $B$-fold blade pattern is **strobed** by the frame
rate: it looks frozen whenever the rotor rate is a multiple of $\text{fps}/B$, and smeared
between those nodes. At 210 fps with 4 blades the comb sits at 52.5, 105, 157 Hz. Measured:

| field Hz | 20 | 40 | 60 | 80 | 100 | 130 |
|---|---|---|---|---|---|---|
| strength | 0.690 | 0.541 | **0.020** | 0.188 | **0.218** | **0.010** |
| nearest node | 0 | 52.5 | 52.5 | 105 | 105 | 105 |

Strength collapses between nodes (60 Hz, 130 Hz) and **recovers** near them (100 Hz, close
to the 105 Hz node). A rising strength therefore means the rotor moved *closer to a strobe
node*, not that it slowed down. Any monotonic reading of this curve is wrong.

**What is left.** Absence of signal remains trustworthy -- it cannot be aliased upward --
so the 52 Hz smear threshold still gives a one-sided bound (18.9). Everything else needs a
different instrument. The cheapest fix is **one asymmetric mark on a single blade**: it
makes the pattern period 360 degrees instead of $360/B$, moving the alias limit from
$\text{fps}/8$ to $\text{fps}/2 = 105$ Hz and collapsing the strobe comb to a single node,
which would cover the whole flight band. Failing that, a tachometer or an optical
interrupter. Exposure control is not an option: AVFoundation on macOS returns -1 for both
exposure properties and rejects every `set`.

**Consequence for the experiment programme:** rotor speed is not observable in the flight
band, so **$z$ is the only metric**. Ramp profile, duration and target frequency are swept
against lift directly.

### 18.11 The takeoff-to-closed-loop handover

Four phases, and the seam is phase 3.

1. **Datum wait.** `takeoff=True` does not command the ramp. The host holds it until
   `PRIME_FIXES` consecutive position fixes, so the ramp never begins before the estimator
   has the robot -- a ramp commanded at $t=0$ once logged no usable samples at all.
2. **Open-loop ramp.** The host sends one `takeoff=<start>:<end>:<ms>:<k>` and then goes
   passive: while `state != FLIGHT` it drains telemetry, logs and draws, and commands
   nothing. The firmware's sequencer owns the trajectory.
3. **The latch.** `seq.isDone()` sets FLIGHT; the host sees `state=2`, sends `throttle=`,
   and anchors the controller (below).
4. **Closed loop**, gated on the operator's `armed` checkbox *and* `enable_freq_cmd`.

**What keeps it smooth.** Nothing is automatic -- `armed` starts off and `mag max` starts
at 0, so a human chooses the moment. The frequency slew limit bounds the rate of change at
`freq_slew * ts` (6.7 Hz per tick at 200 Hz/s and 30 Hz). Anti-windup is conditional, and
slew deliberately does not freeze the integrator because it is a transient rather than a
clamp. Underneath, `PwmController::setGlobalFrequency` applies a phase-continuity
correction, so a frequency change never jumps the field angle out from under the rotor --
which is what makes step changes survivable at all. The state predictor bridges dropouts so
a lost frame does not enter the loop as a step.

**What does not keep it smooth, without help.** The gains carry `f_hover` from their design
point, and the control law is $u = [0, f_{hover}] + u_{ff} - Kx$, with `prev_f_field`
initialised to the same constant. So the first closed-loop command is the *design*
frequency, not the frequency the ramp actually reached. Ramping to 170 Hz and then arming
would walk the drive down toward the design point at the full slew rate. `_anchor` closes
this by setting `f_hover` and `prev_f_field` -- and `ZTracker.f_hat` -- from the frequency
the firmware reports at the latch.

**Anchor the trim, not the band.** An earlier version of this also narrowed
`freq_min`/`freq_max` to $\pm 25\%$ of the reached frequency. That band is a control
envelope, not a physical limit, and it was removed with the other caps -- but removing it
took the trim anchoring with it, which is a different thing wearing the same name. The trim
is where the loop *starts*; the band is how far it may *go*. Only the first belongs to the
handover.

### 18.13 MEASURED 2026-09-01: the departure is a growing whirl, not a push

Three runs with the transparent pole, all open loop, all ending with the robot leaving the
tracked volume. The operator's description was "very non-vertical ... more like it spun
out", and the pose log agrees: over the last 0.3 s of tracking the robot moves further
sideways than up, then vanishes.

**It is not a static lateral force.** The departure azimuth, measured against the pad
centre, was different every run:

| run | departure azimuth | field at departure |
|---|---|---|
| 07:49:56 | 160.3 deg | 104 Hz |
| 07:59:13 | 4.4 deg | 117 Hz |
| 08:01:21 | 276.5 deg | 89 Hz |

A fixed coil imbalance would push the same way every time. This does not, so trimming a
constant lateral bias cannot be the fix.

**It is a whirl.** Tracking the azimuth of the displacement rather than its endpoint, the
robot *orbits* while its radius grows:

| run | radius, last 0.3 s | azimuth swept | whirl rate |
|---|---|---|---|
| 07:49:56 | 0.16 -> 2.24 mm | +25.6 deg | +0.24 rev/s |
| 07:59:13 | 0.14 -> 1.89 mm | -111.8 deg | **-1.05 rev/s** |
| 08:01:21 | 0.08 -> 2.30 mm | +535.5 deg | **+5.03 rev/s** |

Run 3 completed one and a half orbits while spiralling out. Run 2 whirled the *other way*.
The radius e-folds in roughly 90 ms in all three. This is a conical whirl about the takeoff
rod whose amplitude grows until the robot escapes, and the escape direction is simply
wherever it is in the orbit when amplitude wins -- which is why the azimuths look random.

**Why ramp shape did not matter.** 18.2's dwell argument predicted that passing through the
unstable band faster would help, and the operator proposed exactly that. Tested: 30 s ramp
(2.17 s dwell in 90-120 Hz) against 10 s (0.72 s). `f_liftoff` and `f_break` were identical
at 98 and 116 Hz. Three times less exposure changed nothing, because the growth time is
~90 ms and every ramp spends far longer than that anywhere near onset. **The dwell-time
hypothesis is dead.** A first reading of those two runs suggested the departure was instead
locked to 116 Hz; the third run broke at 88 Hz and killed that too. Both hypotheses died on
small-n evidence, which is the standing lesson of this section.

**Why the controller cannot be expected to fix it.** 19 records that the 10 Hz structural
mode is not dampable by a loop whose closed-loop poles sit at 0.16-0.78 Hz. The whirl
measured here reaches 5 rev/s, six to thirty times faster than those poles, and 19.1 shows
the binding constraint is the 65 Hz pose pipeline rather than the command clock -- so a
faster control clock does not buy it back either. Arming lateral is worth doing to learn
whether a given frequency is below whirl onset, but it should not be expected to suppress
the whirl itself.

**QUALIFIED the same day, by the operator watching the 85 Hz run.** Their description was
"enough for the robot to leave the flying pad, but not enough lift to hover -- it just
fell", and the pose log agrees: the run ends at z = -14.5 mm, *below* the pad, not above
it. So at 85 Hz the failure is **insufficient thrust**, and the azimuth sweep tabulated
above is at least partly a robot falling and tumbling off the rod rather than a bearing
whirl growing under it. The orbital numbers are real measurements; the *whirl*
interpretation of them is not established, and this section should not be read as though
it were. What is solid: the departure direction is not repeatable, so a static lateral
bias is not the cause.

Two failure modes may coexist, and the runs so far cannot separate them: above ~100 Hz the
robot is flung (operator: "still got flung out"), at 85 Hz it lifts and falls back. If so
the rig has no operating window -- too little thrust below, instability above.

**The thrust ceiling has a known mechanical cause.** 17 records that the rotor saturates
just above ~52 Hz however hard the field is driven, because of friction at the takeoff rod.
Lift goes as *rotor* speed squared, not field frequency, so once the rotor stops tracking
the field, raising the field buys nothing -- which is exactly the measured trend here:
z_max fell monotonically from 8.7 mm at an 85 Hz target to 0.9 mm at 210 Hz. That makes rod
friction, not ramp shape and not the controller, the first thing to attack. If the rotor
cannot be made to spin faster than ~60 Hz, no ramp and no controller will hover this rig
as built.

### 18.14 MEASURED 2026-09-01: the thrust axis is tilted ~4.6 deg, and that is the problem

The best-tracked liftoff of the day (`results/takeoff/20260901_082621.csv`, 6.6 % lost,
2233 pad samples at 0.05 mm scatter) lifted at 126 Hz and peaked 7.1 mm. Differentiating
the pose with a Savitzky-Golay filter (0.15 s window, quadratic, resampled to 200 Hz) over
the liftoff window gives:

| | |
|---|---|
| vertical velocity | -0.3 mm/s |
| vertical acceleration | -0.006 g, i.e. **T/mg = 0.994** |
| lateral velocity | **68.3 mm/s** |
| lateral acceleration | 0.079 g |

**The rig has the thrust to hover and is pointing it sideways.** Vertical thrust balances
weight almost exactly while the robot accelerates laterally at 0.08 g. Taking the ratio,
`atan(0.079/0.994)` puts the thrust axis **4.6 deg off vertical**.

The tilt is not noise, and it is not a disturbance: it **grows with thrust**, which is the
signature of a fixed angular offset rather than a random push.

| field | lateral accel | implied tilt |
|---|---|---|
| 124 Hz | 0.009 g | 0.5 deg |
| 124 Hz | 0.044 g | 2.6 deg |
| 134 Hz | 0.086 g | 4.8 deg |
| 134 Hz | 0.097 g | 5.7 deg |

Candidate causes, none yet distinguished: centre of mass offset from the spin axis, blade
plane not perpendicular to that axis, or the robot seated at an angle on the rod. All three
are mechanical and none is addressable from the host.

**RETRACTION of 18.13's whirl reading.** 18.13 reported the departure as a growing conical
whirl at up to 5 rev/s and concluded, citing 19, that the control loop was six to thirty
times too slow to damp it. **The spectrum does not support that.** A Hann-windowed FFT of
the lateral position over the last 3 s before departure puts essentially all the power
below 2 Hz -- peaks at 0.67, 1.00 and 1.67 Hz, which at that window length is the
resolution floor, i.e. slow drift. There is no 5 Hz whirl and no excitation of the 10 Hz
structural mode. The "1.5 orbits in 0.3 s" measured in 18.13 is a robot tumbling after it
had already left the pad, not an instability growing before it.

This reverses the practical conclusion. A sub-2 Hz drift against closed-loop poles at
0.16-0.78 Hz is marginal but reachable, so **lateral control is worth considerably more
than 18.13 claimed**. The recurring error in both 18.13 and this correction is the same
one 18.8 records: reading a mechanism off a handful of samples taken after the event of
interest, rather than measuring it during.

**What blocks testing it.** Across five runs on 2026-09-01 the lateral loop never armed
once, because `hover_controller_runner` gates arming on the firmware reporting FLIGHT and
the robot always departed before `seq.isDone()`. The restriction is host-side only:
`main_flight.cpp` runs `applyMixer()` in `case SPINUP` as well as `case FLIGHT`, so `az=`
and `mag=` are honoured throughout the ramp. Arming during the climb would put the
controller in the loop before the robot leaves the pad, which is where a 4.6 deg tilt and
a 1 Hz drift have to be met.

### 18.15 MEASURED 2026-09-01: the robot sits crooked, by 1.6 deg toward ~146 deg

18.14 inferred a ~4.6 deg thrust tilt from accelerations. The rotor normal is now logged
directly (`tilt_deg`, `tilt_az_deg`, added to `constants.CSV_COLUMNS` the same day -- the
estimator had produced `Pose.theta_deg`/`phi_deg` every frame since the start and nothing
had ever written them down). Measured through a full ramp:

| field | tilt | tilt azimuth | azimuth concentration |
|---|---|---|---|
| 0-5 Hz, no thrust | **1.46 deg** | 189 deg | 0.63 |
| 20-50 Hz | 2.27 deg | 150 deg | 0.96 |
| 50-80 Hz | 2.42 deg | 156 deg | 0.95 |
| 100-110 Hz | **3.08 deg** | 143 deg | **0.97** |

Two separable components. **A static ~1.5 deg that is there before any meaningful thrust**
-- the robot is seated crooked on the rod -- and a thrust-proportional part that roughly
doubles it by 105 Hz. The azimuth converges on ~145-155 deg and stays there: concentration
0.97 means a fixed direction, not a precession and not a wander.

The static part is confirmed independently by `mixer_sign.py`: four 2 s holds, commanding
`az` at 0/90/180/270 deg with `mag=0.15` at a seated 70 Hz, returned tilt azimuths of
145.9, 142.5, 151.2 and 146.1 deg -- the same direction every time, magnitude fixed near
1.6 deg. Four independent measurements agreeing within 9 deg.

**The datum-zeroing objection, and why the growth survives it.** `calibrate_zero` sets the
datum from a reference *pose*, not from gravity, and warns that a near-face-on reference
recovers the axis about 10 deg off (10.4 deg at 0, 2.3 deg by 10). So the *absolute* tilt
carries the datum's own axis error and 1.5 deg is inside it. The *growth* with thrust does
not: a datum error is a constant rotation, so it offsets every reading equally and cancels
in the difference. 1.46 -> 3.08 deg is real however the zero was taken.

**NEGATIVE RESULT: that mixer_sign run cannot calibrate the mixer, by construction.**
`HOLD_HZ = 70` was chosen below the 98-126 Hz liftoff band so the robot would stay on the
pad. Seated on the rod it is mechanically constrained and cannot tilt in response to `az=`
at all, so the four holds measured its resting tilt four times rather than the disk's
response. The safety margin removed the degree of freedom being measured. `applyMixer`'s
"Verify sign on rig" is therefore STILL unverified, and no feedforward trim should be built
until it is -- a trim on the wrong sign doubles the tilt rather than cancelling it.

**What this points at.** A software trim spends lateral authority on every flight to cancel
something a mechanical adjustment removes once, and the direction is now known well enough
to check on the bench: ~146 deg, which is between coils B (90) and C (180), nearer C.

### 18.16 Why the mixer's authority may not be measurable on this rig

`applyMixer`'s "Verify sign on rig" is still unverified after two attempts, and the reason
is structural rather than a bad script.

To measure whether `az=`/`mag=` moves the rotor normal, the robot must be free enough to
tilt and present enough to measure. Those two conditions do not overlap:

| hold | outcome | why no measurement |
|---|---|---|
| 70 Hz | seated hard on the rod | constrained; four holds at az 0/90/180/270 returned the SAME tilt azimuth (145.9, 142.5, 151.2, 146.1 deg) -- its resting tilt, measured four times |
| 115 Hz | departed during the ramp | gone before the sweep began; operator confirmed "the frequency was too high, it took off/spun off" |

Liftoff has been measured between 98 and 126 Hz across runs on the same day, so the band
where the robot is unloaded but still on the pad is a few Hz wide and moves between runs.
That is the same narrow window 18.14 records between lift and departure.

**Useful consequence for whoever tries next.** `az` pointed exactly at a coil axis is
already individual amplitude control -- `applyMixer` uses `max(0, cos(az - COIL_AZ))`, so
the target coil drops by `MIX_GAIN*mag`, both neighbours sit at `cos(+-90) = 0`, and the
opposite coil's `cos(180) = -1` is clamped away. No per-channel firmware command is needed
to drop one coil at a time, which is worth knowing before anyone adds one.

**If the window proves unusable**, the remaining routes are: identify during the ramp with
short perturbations rather than holds, accepting that the disk may not settle; or give up
on characterising lateral authority in flight and remove the tilt at its source. 18.15
measures that source as a 1.6 deg seating offset toward ~146 deg, which needs no rig time
to investigate and no lateral authority to fix.

### 18.17 MEASURED 2026-09-01: f_hover is ~101 Hz, and the f^2 law dies above ~110 Hz

`z_track`'s plant law is `z_ddot = g*((f_robot/f_hover)^2 - 1)`, so `f_hover` is
recoverable from any flight that logs both z and f: `f_hover = f / sqrt(1 + z_ddot/g)`.
Differentiating z with Savitzky-Golay (0.15 s, quadratic, resampled to 200 Hz) over the
samples that are clear of the pad, across four runs:

| run | f where z_ddot = 0 |
|---|---|
| 2->210 / 30 s | 90 Hz |
| capture + 8->210 / 20 s | 114 Hz |
| capture + 8->120 / 14 s | 107 Hz |
| capture + 8->210, armed on ramp | 95 Hz |

**Measured f_hover = ~101 Hz.** Against `design_hover_lqr.f_hover` = 160, `z_track`'s
starting `f_hat` = 190, and `RAMP_TARGET_HZ` = 210. 18.12 argued those three numbers were
different quantities and all correct; that argument stands, but **all three sit roughly
twice the frequency at which this rig actually balances its weight**.

Thrust goes as f^2, so a 210 Hz target commands about `(210/101)^2 = 4.3x` weight. With the
4.6 deg tilt of 18.14 that surplus becomes `4.3*sin(4.6 deg) = 0.35 g` of lateral force --
which is why every high-target run left the tracked volume in under a second, and why
LOWER targets measured MORE lift (8.7 mm at an 85 Hz target against 0.9 mm at 210). The rig
was never thrust-starved. It was grossly over-driven.

**And the f^2 law itself does not hold above ~110 Hz.** Binning the same run by height:

| z above pad | f | 1 + z_ddot/g |
|---|---|---|
| 0.3-1 mm | 104 Hz | 1.0000 |
| 1-3 mm | 124 Hz | 1.0018 |
| 6-20 mm | 134 Hz | **0.9731** |

Thrust is flat then FALLING where the model predicts a 66 % rise from 104 to 134 Hz. A fit
of `1 + z_ddot/g` against `f^2` returns a negative slope, so `f_hover` cannot even be
extracted that way over this band. The tilt does not explain it: `cos(3 deg) = 0.9986`,
against a measured 2.7 % drop.

The likely cause is the rotor ceasing to track the field somewhere near 110-125 Hz, after
which driving faster does not spin it faster. NOT ESTABLISHED: the 6-20 mm bin is where the
robot was departing, and a tumbling robot's z_ddot is not thrust. 62 samples. Distinguishing
these needs a run that HOLDS a frequency in 100-130 Hz while airborne, which is exactly the
band 18.16 records as hard to hold.

**Operator observation the same day, which agrees:** with coil A dropped the robot "flew off
in a much more stable manner", and "by keeping the drone vertical, the drone is able to take
off more stably and at a lower frequency". Both follow directly from the numbers above.

### 18.18 MEASURED 2026-09-01: the field is 29 % stronger toward 151 deg, and that is the tilt

`probe=` (a lock-in on the coil current, IDLE-only, ~1.6 s per point) measured per-coil
amplitude and phase against the commanded reference at five frequencies. 8 s of drive
total:

| f | amplitude dipole | direction | amp spread | phase spread | strongest |
|---|---|---|---|---|---|
| 60 Hz | 2.0 % | 104 deg | 5 % | 3.4 deg | C |
| 90 Hz | 7.1 % | 139 deg | 15 % | 4.2 deg | C |
| 120 Hz | 15.7 % | **153 deg** | 21 % | 11.7 deg | C |
| 150 Hz | 22.3 % | **151 deg** | 27 % | 13.4 deg | C |
| 174 Hz | 28.9 % | **151 deg** | 33 % | 11.5 deg | C |

**The direction converges on ~151 deg above 120 Hz, and 18.15 measures the rotor leaning at
146 deg.** Five degrees apart, from two instruments sharing nothing: a lock-in on coil
current, and the optical rotor normal. 20.3 argued that the only surviving candidate cause
was a lab-frame field asymmetry, and that a field asymmetry is host-addressable; this is
that asymmetry, measured.

**Both grow with drive, which is the second half of the agreement.** The imbalance goes
2 % -> 29 % from 60 to 174 Hz, and the tilt goes 1.46 -> 3.08 deg from rest to 105 Hz. The
frequency dependence is diagnostic on its own: R dominates the channel impedance at low
frequency, where the coils look matched, and the per-coil L/C spread takes over approaching
the ~174 Hz series resonance (18.8 fits f_0 = 174 Hz, Q = 0.40). So this is a REACTIVE
mismatch, not a resistive one, which is why it is invisible on a DC or low-frequency check.

**The trim, at the frequency actually flown:**

| f | resultant weak direction | strength |
|---|---|---|
| 150 Hz | 151 deg | 0.19 |
| 174 Hz | 151 deg | 0.24 |

`applyMixer` weakens the coils facing `az`, and B/C are the strong side, so the correction
points AT 151 deg rather than at `tilt + 180`. Both 0 deg and 315 deg have been flown and
151 deg has not.

**RETRACTION: the 18 deg dipole measured the same day from `CS` telemetry is withdrawn.**
Averaging `I[A]` from the periodic telemetry at 90 Hz gave 0.49 A at 18 deg with coil A
strongest; the lock-in at the same frequency gives 139 deg with coil C strongest. The
lock-in is the instrument to believe, and 17.2 says why: `CS` is an unsigned magnitude
sampled asynchronously at about 1 kHz, roughly 7 samples per electrical cycle at 150 Hz,
"enough to regulate amplitude and not enough to resolve an angle". It is not enough to
resolve an amplitude ratio between channels either, once the sampling aliases. Coherence on
the lock-in was 0.90.

**What an amplitude trim cannot reach.** The phase spread is 13.4 deg at 150 Hz and grows
with frequency alongside the amplitude spread. A rotating field needs the four channels at
0/90/180/270 deg *relative*, so a phase error distorts the field into an ellipse whatever
the amplitudes do, and `applyMixer` has no phase authority at all. Correcting it needs a
per-coil phase offset in firmware -- which `PwmController::setPhaseOffset` already
supports, so the mechanism exists and only the calibration is missing.

### 18.12 Three "hover frequencies", and why they disagree

Three files carried a number for where this rig flies, all dated 2026-08-29, spread over
50 Hz. They are three different quantities and all three are right:

| Value | Where | What it actually is |
|---|---|---|
| 160 Hz | `design_hover_lqr.f_hover` | The **linearisation point** for the gains. `_anchor` overwrites the trim at runtime with the frequency the ramp reached, so this only has to be close enough for the linearisation. |
| 190 Hz | `z_track.F_HOVER_HZ` | The tracker's **starting** $\hat f$, which it then adapts. Liftoff measured at 180-185, lift peaking 190-210. |
| 210 Hz | `constants.RAMP_TARGET_HZ` | The **ramp target**, and the best measured so far -- this is the one that flew. |

Only the last is "where it hovers". The others were labelled as though they were, which is
how the same day's measurements came to look like a contradiction. None of them needed
changing; the labels did.

## 19. Loop rate: what it cost, and what it bought

Written 2026-08-30. The 10 Hz structural mode from 14 is not dampable by a loop whose
closed-loop poles sit at 0.16-0.78 Hz and which steps at an effective 20 Hz. The work
below was aimed at 200 Hz. It reached 200 Hz of **command** and 65 Hz of **measurement**,
and the second number is the one that matters.

### 19.1 The decomposition nobody had written down

Mean command age is $T_s/2$. Against the slowest closed-loop pole at 0.78 Hz:

| | $T_s/2$ | phase lag at 0.78 Hz |
|---|---|---|
| control at 30 Hz | 16.7 ms | 4.7 deg |
| control at 200 Hz | 2.5 ms | 0.7 deg |
| pose pipeline at 20 Hz | 50 ms | 14 deg |
| pose pipeline at 65 Hz | 15 ms | 4.3 deg |

**The 200 Hz control clock is worth about 4 degrees. The pipeline is worth about 10.**
That ranking is the whole reason the effort went into segmentation rather than into the
clock, and it is worth re-deriving before anyone spends a session raising the clock again.

### 19.2 The pipeline, measured

Replay of `results/flights/2026-08-29_231418`, first 250 frames (246 solved), through
`live_viz.from_recording(viz=NullViz(), speed=0)`. Median ms per stereo pair:

| | segment | estimate | wall | rate |
|---|---|---|---|---|
| baseline | 27.8 | 16.8 | 48.0 | 20.8 Hz |
| \+ plate-response cache fixed | 21.6 | 17.7 | 41.3 | 24.2 Hz |
| \+ evidence map windowed on the previous ellipse | 16.0 | 18.1 | 35.5 | 28.2 Hz |
| \+ the two views in parallel | 8.9 | 17.1 | 28.1 | 35.6 Hz |
| \+ centre-cal displacement cached | 8.9 | 12.3 | 23.2 | 43.0 Hz |
| \+ 640x400 | 3.0 | 10.5 | 15.4 | 64.8 Hz |
| \+ rim shape cached on the normal | 3.2 | 9.3 | 14.5 | 69.0 Hz |
| \+ plate cache made per-camera (19.5) | 3.1 | 9.2 | 14.1 | 70.7 Hz |
| \+ `REFINE_TOL` 1e-5 -> 1e-4 | 3.2 | 7.4 | 12.4 | 80.6 Hz |
| \+ batched Jacobian (19.4) | 3.0 | 5.9 | **10.6** | **93.9 Hz** |

That last column includes decoding the recording's mp4 inline, which the live loop does
not -- `sources.MonoCamera` decodes on its grabber thread. The cost the control loop
actually pays is `est.update`, which `StereoPose` already times: **3.2 ms segment + 6.2 ms
solve = 9.4 ms, i.e. ~107 Hz.**

**A caveat on all of these, found late.** `from_recording` hardcodes `backgrounds="running"`
while the live `stereo_frames` defaults to the *saved* plates. On this recording the
running plate walks onto a station-keeping robot -- the failure `background.RunningPlate`'s
own docstring and `demo_video.py`'s comment both warn about -- so `segment()` returns
`None` on **100% of frames at every resolution**, and the pose comes from the tracked
`_prev_ellipse` seed plus the joint image-mode solve. That is a designed fallback and it
produces good fits (the demo overlay puts the rim on the rim, 1.9 mm discrepancy), but it
means the numbers above under-represent the mask path, which bails early instead of
completing. The *relative* improvements are sound -- every row ran the same path -- and the
solve, which is 2/3 of the budget, is unaffected. The absolute segmentation figure would be
larger on a session with a good plate. Re-measure with saved plates before quoting 107 Hz
as the live rate.

Solve count was 246/250 at every step, so none of it was bought by dropping frames. The
same code at 1280x800 is 22.4 ms and 44.7 Hz, so a little over half the total win is the
resolution and the rest is arithmetic that was being repeated.

**Where the remaining time is.** `refine` is 9.2 of it, and `refine` is dominated by
its *numerical Jacobian*: `least_squares` reports ~9.6 residual evaluations a frame, but
`evidence` is entered ~47 times, because 5 parameters differenced two-point cost 5 extra
evaluations each and those are not counted in `nfev`. **Roughly 80% of the solve is
finite differences.** An analytic Jacobian would take that ~47 back toward ~10 -- it needs
the image gradient of the evidence map times the pixel-vs-pose derivative, which is
tractable and is the single largest remaining item. It is not attempted here: it changes
the descent direction, and the diff was already large.

The two cheap consequences of that structure were taken. `_rim_points` recomputes the
tangent basis and two outer products per evaluation, but **three of the five perturbations
move only the centre**, which shifts the rim rigidly -- so the normal-dependent part is
cached (`_rim_shape`, keyed on the normal's bytes: exact, verified bit-identical over 3000
random poses). And `_tangent_basis` no longer calls `np.cross`: identical arithmetic, but
for a 3-vector `np.cross` spends most of its time in axis bookkeeping, which showed up as
124k `normalize_axis_tuple` calls in the profile.

**The plate-response cache had never once hit.** `segment.ring_weight` cached the plate's
41x41 opening on `id(img)`, with a comment asserting the estimator passes the same array
every frame. It does not: `background.RunningPlate.update` returns
`self.bg.astype(np.uint8)`, a fresh array every call, and a `RunningPlate` is the live
default. So the cache paid the 2.6 ms per view it existed to avoid, every frame, and the
`id()` key held a reference to each dead plate so the ids could not even be recycled. It
is keyed on the plate's own frame counter now. Note that a *stable* id would have been
worse than the miss: it would have pinned a response to a plate that moves.

**The ROI was already written and wired to nothing.** `ring_weight` has taken an `roi`
argument, with `_clamp_roi` and a measured "0.37 ms on 450x450 against 2.64 ms
full-frame" in its own docstring, for as long as it has existed. No caller ever passed
one -- while `stereo._prev_ellipse` was already carrying the previous frame's ellipse per
camera, used only as a fallback seed. The window is that ellipse's major axis times
`ROI_MARGIN`, squared (the rim rotates between frames, and a box that hugs the minor axis
clips it when it does), and absent or failed tracking falls back to the whole frame -- a
tracker that cannot re-acquire is worse than a slow one.

### 19.3 Getting past 90 Hz: the solve's stopping rule, not its arithmetic

Once `refine` is 70% of the pair, the question is not how fast an evaluation is but how
many there are. Four knobs, all measured on the same 250 frames:

| setting | Hz (`update`) | nfev | `refine_rms_px` | `discrepancy_mm` | solved | pose shift |
|---|---|---|---|---|---|---|
| `xtol=ftol=gtol` 1e-5 | 81 | 9 | 4.7468 | 1.4619 | 246 | -- |
| **1e-4** | **98** | **7** | **4.7552** | **1.4619** | **246** | **0.127 mm** |
| 1e-3 | 114 | 3 | 4.7554 | 1.4619 | 246 | 0.157 mm |
| `max_iter` 6 or 4 | 71 | 9 | -- | -- | 246 | 0.000 mm |
| `x_scale=1.0` | 73 | 9 | -- | -- | 246 | 0.133 mm |

`max_iter` does nothing because the cap is never reached -- the seed is good and the solve
converges in about seven evaluations, so `MAX_REFINE_ITER` has been protecting against a
case that does not occur. **The stopping tolerance was the only real knob**, and 1e-4 pays
17 Hz for 0.18% of a 4.7 px residual, with `discrepancy_mm` -- an independent cross-view
consistency measure, and the one that would expose a genuinely worse fit -- unchanged to
four decimals. 1e-3 keeps paying, but the pose keeps moving while the rms stops following,
which is the signature of stopping early rather than converging. That is where it stops.

**The finite-difference step went the other way, which is worth recording.** The intuition
was that scipy's default relative step (~1.5e-8) is far below the smoothness scale of a
bilinearly-sampled, Gaussian-blurred evidence map, so the Jacobian would be float noise and
a larger step would converge faster. Measured, a larger step converges to a *better*
residual and takes *more* iterations:

| `diff_step` | Hz | nfev | `refine_rms_px` |
|---|---|---|---|
| default (~1.5e-8) | 81 | 9 | 4.7468 |
| 1e-6 | 66 | 12 | 4.7662 |
| 1e-5 | 43 | 17 | 4.5879 |
| 1e-4 | 36 | 20 | 4.4241 |
| 1e-3 | 34 | 19 | 4.4305 |

So the default step is not noise-limited, and a coarser Jacobian is a **quality** knob
pointing away from speed: 7% better rms for 2.3x the time. Left alone. Noted because the
opposite is a natural thing to assume and it costs a session to find out.

### 19.4 The numerical Jacobian, and why a GPU is the wrong instrument

With `REFINE_TOL` at 1e-4 the solve still enters `evidence` about five times per reported
`nfev`, because five parameters differenced two-point cost five extra evaluations an
iteration. The instinct is to make the arithmetic faster. **The arithmetic is not the
cost.** Projecting and distorting one view's samples, measured on this rig:

| points | time | per point |
|---|---|---|
| 45 | 22.9 us | 508 ns |
| 180 | 40.9 us | 227 ns |
| 900 | 132.6 us | 147 ns |
| 3600 | 457.5 us | 127 ns |

That is **~15 us of fixed per-call overhead plus ~120 ns a point** -- consistent with the
earlier `sample_n` sweep, where cutting the point count 4x bought only 27% of the time. The
solve is bound by the *number of numpy and cv2 calls on small arrays*, not by floating-point
work.

Two things follow. First, **the fix is to batch, not to accelerate**: the five perturbations
are evaluated in one pass over `5n` points instead of five passes over `n` (`evidence_many`,
and a `jac` supplied to `least_squares` instead of letting it difference). Five 180-point
calls cost 205 us; one 900-point call costs 133. Measured end to end, the solve goes 7.6 ->
6.2 ms, `est.update` 10.1 -> 9.4 ms.

Getting scipy's step rule exactly right mattered more than expected: it uses
`rel * sign(x) * max(1, |x|)`, and dropping the **sign** steps the negative parameters the
wrong way. With the sign, median `refine_rms_px` is 4.7569 against scipy's own 4.7552
(0.036%) with `discrepancy_mm` identical; without it, 4.7661. The pose still moves 0.137 mm,
which is the flat-optimum behaviour seen throughout this chapter rather than a worse fit.

Second, **a GPU kernel would make this slower, not faster.** The arrays are 180x2 doubles,
about 3 kB; a kernel launch plus a round trip is 5-10 us against a whole evaluation that
costs tens of microseconds on the CPU. GPUs win when arithmetic intensity is high and the
data is already resident, and here it is neither: the evidence maps would have to be
uploaded every frame, and the solve is a sequential trust region -- each iteration depends
on the last, so there is nothing to run wide. The only genuinely parallel axis is the five
Jacobian columns, and that is exactly what batching already exploits on the CPU for free.
The honest GPU-shaped opportunity is upstream, in segmentation, where full-frame morphology
and connected components are data-parallel -- but segmentation is now 3.0 ms of a 9.4 ms
pair, so the ceiling on that whole direction is 3 ms.

**The remaining structural item is still an analytic Jacobian** -- image gradient of the
evidence map times the pixel-vs-pose derivative -- which removes the five extra evaluations
rather than batching them. Not attempted *here*; done in **19.12**, where it turned out to
be an accuracy result first and a speed result only through the stopping tolerance, because
the forward-difference step this section trusted is noise.

### 19.5 What was measured and NOT taken

Two speedups were rejected on measurement, and the measurements are the point:

**Fewer rim samples in `refine`.** `refine` already takes `sample_n`; nobody passes it.
Dropping 180 to 120 buys 4 ms and moves the pose 0.19 mm median; 45 buys 8 ms for 0.26 mm.
Against a bias floor of 0.185-0.274 mm that is not free. The principled objection is
Nyquist: the evidence map is Gaussian-blurred at sigma = 3, so features are 6-9 px wide,
and 120 samples around a 450 px ring is one every 12 px. 180 is properly sampled and 120
is not. **Rejected.**

**320x240.** 70.9 Hz against 64.8 for 640x400 -- 6 Hz more -- for a per-axis bias of
0.205/0.169 mm, over the floor rather than under it. 640x400 costs 0.110/0.119/0.065 mm
spread against the 0.119 mm `pose/theory.md` 315-327 predicts, which is the rare case of a
prediction landing exactly. **Rejected: twice the error for 6 Hz.**

One was taken **with** a measured cost. The centre-cal displacement in `refine`'s evidence
residual cost two conic decompositions per view per evaluation -- 222 `cone_from_circle`
calls a frame, more than the projection it corrects -- and `least_squares` spends 6 of
every 7 evaluations on finite differences at effectively the same pose. Caching it on a
quantised pose collapses those. It costs 0.14 mm median, because the Jacobian loses the
displacement's own derivative; tightening the quantum to 1e-6 mm does not recover it, so
collapsing the finite differences IS the effect. What justifies keeping it: `refine_rms_px`
moves 5.112 to 5.126, 0.3% on a 5 px residual, with `discrepancy_mm` and `margin`
unchanged. The optimum is flat over that 0.14 mm. 17.0 -> 12.5 ms.

### 19.6 The control clock, and why it is a thread

`stereo_frames` is a generator, so capture, segmentation, triangulation and the control
body all ran on one thread and the controller could not step faster than a pair took to
segment. The pose source now runs on its own thread behind a drop-oldest single slot --
the same shape as `sources.MonoCamera`, and for the same stated reason: for feedback a
stale pose is worse than no pose, so a queue that buffers backlog would be actively
harmful. **The counter, not the slot, is what says "new".**

The control loop stays on the main thread. That is not a preference:

- `signal.signal` can only be installed on, and delivered to, the main thread. SIGINT's
  `land()` running on main while a worker sits mid-`link.send` puts two threads on one
  pyserial handle and half-writes a `stop`.
- The generator's `close()` runs from the producer's own `finally`. Calling it from the
  consumer while the producer is suspended inside it raises "generator already executing".
- The existing `finally` already ordered `land()` -> `send("stop")` -> close. `_PoseFeed`
  duck-types onto that with no edit, which keeps the coils de-energising *before* anything
  joins a thread that may be blocked in a 2 s camera read. Preserve that order.

One rule decides every site in the loop: **anything that consumes a frame is gated on
`fresh`; anything that commands the coils runs on the clock.** Ungating a frame consumer
is not a tidiness point. At 500 Hz one held fix satisfies `PRIME_FIXES = 15` in 30 ms and
ramps the coils on a single frame.

The predictor advances from `t_pred`, its own clock, not from the last control step --
they are different rates now. Setting `t_pred` to `tick.t`, the shutter stamp, means the
next step propagates the pose forward by its full pipeline age. That is the latency
compensation `filter.predict_ahead` was written for and never wired to, and here it is
free. `StatePredictor` rather than `predict_ahead` because it is the model-forward, which
`predictor.py`'s own assert shows beats a kinematic coast at every gap length -- and
because `filt` lives on the pose thread now, so reading it from the control loop would be
a torn-state hazard for no gain.

Measured end to end against a 65 Hz source: **199.8 Hz**.

### 19.7 Parallelising the views made the pipeline non-reproducible

Found by accident, while checking that an "exact" change really was exact: **two identical
replays of the same recording disagreed by up to 0.34 mm** (median 0, p95 0.24) -- most
frames bit-identical, a minority flipping. Forcing the view pool to one worker made it
bit-identical again, so the two-worker pool from 19.2 introduced it.

The per-view segmentation output was identical between runs; the *evidence maps* were not.
`segment.ring_weight`'s plate-response cache was one module-global dict that cleared itself
whenever it exceeded four entries. With two views on two threads, one view's insert could
evict the other's live entry -- and because the cache holds a response computed against a
particular plate generation, a mid-block recompute produced the response for a *different*
generation. **Which plate the map was built against therefore depended on thread
interleaving.** Not a crash, not a wrong answer, just an answer that would not reproduce.

Fixed by giving each caller its own slot, replaced in place, with no eviction rule at all:
the threads never touch the same key, each get/set is a single GIL-atomic dict op, and the
cache is bounded by the number of cameras rather than by a magic 4.

Two things worth carrying forward. **A shared cache with a global eviction rule is not
thread-safe just because every individual operation is** -- the eviction is what couples
the callers, and it coupled two views that were otherwise independent. And
**reproducibility is a measurement instrument**: this was invisible until a change that
should have moved nothing was measured and appeared to move 0.39 mm. Every accuracy figure
in 19.2 and 19.3 was taken before the fix and therefore carries up to ~0.24 mm of this
noise on top of the effect being measured. They were re-taken after it. The resolution
comparison in particular came out unchanged -- per-axis spread 0.118/0.117/0.066 mm
against the 0.119 mm predicted -- which is why the conclusion stands.

### 19.8 The gains at 200 Hz are a null change

Discrete LQR converges to the continuous-time LQR as $T_s \to 0$, so with $Q$ and $R$
fixed the closed loop is invariant and only $K$ walks toward its continuous limit:

| rate | closed-loop pole rates, Hz |
|---|---|
| 30 | 0.162 0.162 0.552 0.552 0.780 0.780 |
| 100 | 0.162 0.162 0.552 0.552 0.780 0.780 |
| 200 | 0.162 0.162 0.552 0.552 0.780 0.780 |
| 400 | 0.162 0.162 0.552 0.552 0.780 0.780 |

$K[0]$ goes from `[58.31 17.47 0 0 42.72 0]` to `[65.23 18.99 0 0 48.83 0]`, about 12%.
The pole check at `design_hover_lqr.py` passes with 43x margin instead of 6.4x. Everything
downstream re-derives from `ts`: `freq_slew * ts` is still 200 Hz/s, the integrator is
still per-second, `VelocityEstimator`'s alpha comes from its 5 Hz cutoff.

`_noise_provenance` records a velocity sigma 2.8x larger (0.94 -> 2.63 mm/s on z),
because differentiating pose noise at 5 ms rather than 33 ms amplifies it as
$\sqrt{\text{rate}}$. **That number is provenance only -- it does not feed the design --
and it is pessimistic**, because it assumes an independent measurement every 5 ms and
there is not one: between poses `StatePredictor.predict` moves position by exactly
$v\,dt$, so the differencer sees its own velocity back and injects nothing. The real
defect is narrower: on the step a fix lands, `VelocityEstimator.update` divides the
innovation by `self.ts` rather than the true pose interval, an over-large kick partly
cancelled by the smaller alpha. Not fixed pre-emptively. If the bench shows chatter on the
rate channel, the ladder is: drop the clock to ~2x the measured pose rate; then thread a
real `dt` through `VelocityEstimator.update`; then raise `tau`.

### 19.9 The phase-lock check had been measuring the sample grid

Scenario c (plant hovering at 143 Hz against a design at 160) appeared to regress from
46.6 deg to 73.5 deg on the 200 Hz redesign. It had not. `simulate_hover` integrates the
plant finely -- `solve_ivp`, `max_step = ts/8` -- but **recorded** `delta` once per control
step, so the metric measured whatever $T_s$ happened to sample. Recording the peak across
the sub-step instead:

| control rate | 30 | 50 | 60 | 80 | 100 | 130 | 200 |
|---|---|---|---|---|---|---|---|
| scenario c, deg (before) | 46.6 | 72.2 | 74.4 | 69.6 | 72.1 | 73.8 | 73.5 |
| scenario c, deg (after) | 74.2 | 74.2 | 74.4 | 74.4 | 74.3 | 74.3 | 74.3 |

**The excursion was always 74 degrees.** The 30 Hz simulation stepped over the peak, and a
gate at 60 deg had been passing a scenario that violated it for as long as the gate has
existed. The 200 Hz design is not a regression; the measurement got honest.

The gate now separates the two questions it had conflated: lock is *physically* lost at 90
deg, where $\sin\delta$ peaks and the rotor slips a pole, and 60 deg is the margin we
want. Above 60 reports THIN, above 90 fails. Scenario c reports THIN, which is true and
which a permanently red check would have taught nobody.

### 19.10 Serial at 200 Hz: deadbands, not baud

Three commands a step at ~29 bytes is 4.4-5.8 kB/s, 38-50% of 115200 8N1, with 29 bytes
taking 2.5 ms to shift out of a 5 ms budget. Raising the baud and coalescing the commands
are both reflashes. Extending the `AZ_DEADBAND_DEG` pattern that already existed is five
lines of Python: 1 deg of azimuth, 0.005 of magnitude, 0.05 Hz against a 200 Hz/s slew
ceiling. Measured on a 65 Hz source at 200 Hz control: **92 B/s, 0.8% of the line.**

`RESEND_S = 0.5` is not a heartbeat for its own sake. With deadbands in and no firmware
watchdog and no ack, a corrupted `mag=` line would otherwise stand until the next value
crossed a deadband -- which in a steady hover is never.

### 19.11 What this does not buy

At 65 Hz pose the Nyquist limit is 32 Hz and the 10 Hz mode gets six samples a cycle. That
is enough to **observe** it and log it honestly. It is not enough to damp it: the closed
loop is still at 0.78 Hz, thirteen times slower than the mode.

The prize is one step further on and deliberately not taken here.
`design_hover_lqr.py` detunes the Bryson weights to 0.78 Hz explicitly *for latency
robustness* -- "tighter designs pass on paper and fail in closed loop, at 1.33 Hz through
a ~2.2 Hz limit cycle from one frame of latency". That justification was written against a
50 ms pipeline. At 15 ms it largely evaporates and a tighter design becomes admissible.
**That is a separate, measured change, and bundling it into the diff that made it possible
would have made both unreviewable.**

### 19.12 The analytic Jacobian, and the finite-difference step that was noise

Written 2026-09-01, closing 19.4's "remaining structural item".

The analytic Jacobian is in, as `stereo.refine.jac_analytic`. For
`residual = sqrt(max(ref - E, 0))` the chain is

$$\frac{\partial\,\text{res}}{\partial p}
 = -\frac{1}{2\,\text{res}}\;\nabla_{\!\text{pix}} w \;
   \frac{\partial\,\text{pix}}{\partial\,\text{rim}}\;
   \frac{\partial\,\text{rim}}{\partial p}$$

Only two factors are still differenced, and neither touches the image: `d(rim)/d(normal)`
in 3-space (two cached `_rim_shape` lookups) and `d(raw pixel)/d(ideal pixel)`, a smooth
polynomial with no image data in it, batched over `3n` points in one call. The three
centre columns of `d(rim)/dp` are exactly the identity -- the same fact `_rim_shape` was
already cached on -- and the pinhole term is closed form off the camera-frame point
`_project_ideal` had already computed.

**It did not do what it was expected to do.** The prediction in 19.4 was speed: take ~47
`evidence` entries a frame back toward ~10. Measured at the shipped `REFINE_TOL` of 1e-4
it went the other way -- `nfev` **7 -> 10** and `est.update` **6.9 -> 9.3 ms** -- while
`refine_rms_px` **improved 5.1642 -> 4.7890** with `discrepancy_mm` identical to four
decimals. That is not a slower Jacobian. That is a *better* one, and 19.3 had already
written down what a better Jacobian looks like on this problem without recognising it:

> a coarser Jacobian is a **quality** knob pointing away from speed: 7% better rms for
> 2.3x the time.

**The claim to retract is 19.3's "the default step is not noise-limited".** It is.
Measured columnwise cosine between the analytic Jacobian and forward differences at
scipy's own step:

| reference step | cosine per column |
|---|---|
| `rel = 1e-6` | 0.95 0.98 1.00 0.96 0.97 |
| `rel = 1.5e-8` (scipy's default) | 0.13 0.22 0.79 0.31 0.12 |

At `1.5e-8` relative on a centre near 300 mm the perturbation is ~4.5e-5 px, and
`segment.sample_map` casts its coordinates to **float32**, whose resolution near a
300 px coordinate is ~3e-5 px. **The step is smaller than the pixel coordinate it
perturbs**, so the difference is float32 rounding, not signal. The shipped solve has been
descending on noise that happens to be uncorrelated enough to terminate early.

The cleanest demonstration is synthetic and is now the self-check in `stereo._self_check`.
Plant a pose, render its rim into two noiseless evidence maps, seed 1.5 mm and 0.05 rad
away, and recover it:

| Jacobian | median error | max |
|---|---|---|
| analytic | **0.030 mm** | 0.52 mm |
| batched forward differences | 2.33 mm | 3.53 mm |

Fifty times worse, on a scene with no noise, no plate and no occlusion. Nothing but the
derivative is different between those two rows.

**The speed comes back through the tolerance, because a tolerance is only meaningful
against the Jacobian that produced the gradient.** `REFINE_TOL` 1e-4 was tuned in 19.3
against the noisy one. Re-swept against the analytic one, on the same 250 frames:

| | est ms | nfev | `refine_rms_px` | `discrepancy_mm` | `union_coverage` | solved |
|---|---|---|---|---|---|---|
| batched FD, 1e-4 (was shipped) | 6.9 | 7 | 5.1642 | 0.4641 | 0.8611 | 246 |
| analytic, 1e-4 | 7.8 | 10 | 4.7890 | 0.4641 | 0.8278 | 246 |
| **analytic, 1e-3** | **4.1** | **4** | **4.9089** | **0.4641** | **0.8778** | **246** |
| analytic, 3e-3 | 3.2 | 3 | 5.1027 | 0.4641 | 0.8833 | 246 |
| analytic, 1e-2 | 2.4 | 2 | 5.4185 | 0.4641 | 0.9000 | 246 |

`REFINE_TOL_ANALYTIC = 1e-3` is taken. It needs no trade argued for it: against the
baseline it is **40% faster and better on every quality axis at once** -- rms lower,
coverage higher, `discrepancy_mm` unmoved, 246 solved. The pose moves 0.24 mm median
against a bias floor of 0.185-0.274 mm, which is the flat-optimum behaviour seen
throughout this chapter rather than a worse fit.

3e-3 is still better than the baseline and is left available. **1e-2 is where it stops**:
`refine_rms_px` crosses the baseline and the pose shift jumps 0.25 -> 0.34 mm while the
rms stops following, which is 19.3's own signature of stopping early rather than
converging.

The tolerance is a **separate constant**, not a retune of `REFINE_TOL`, because
`mode="ellipse"` still runs on `jac="2-point"` and must keep the tolerance that was
measured against it.

**One negative result worth the line it costs.** The first implementation took the image
gradient from `cv2.Sobel` on the evidence map. That is the gradient of the wrong function:
`sample_map` reads **bilinearly**, so the surface the solver descends is piecewise-linear
between pixel centres, while Sobel is a 3x3-smoothed estimate of a different one. Swapping
it for a central difference *of the sampled field* -- all five reads batched into one
remap over `5n` points -- is the consistent thing to do and is what ships. It changed the
columnwise cosines by less than 1e-4, because `RING_BLUR_SIGMA = 3` leaves the map smooth
enough that the two agree anyway. Consistent for a reason that did not turn out to matter,
kept because the next person to widen the blur would find that it does.

Reproducibility (19.7's instrument) holds: two replays are bit-identical across all 246
poses and every non-timing column.

**What is still open.** `est.update` is now ~3.2 ms segment + ~4.1 ms solve. Segmentation
and the solve are within a factor of two of each other for the first time, so the next
honest look is at segmentation -- and that is the one data-parallel stage, which is where
19.4 said the GPU-shaped opportunity actually lives. The ceiling on it is still ~3 ms.

### 19.13 500 Hz, and the wire that made the clock moot

Written 2026-09-01. The clock went to 500 Hz because it was asked for. **19.1's ranking
still holds and nothing here overturns it**: against the 0.78 Hz closed loop, 200 Hz of
command was worth 0.7 deg of phase lag and 500 Hz takes that to 0.28. The clock was never
the expensive term. What follows is what it actually cost to hold, and the one thing found
along the way that *was* worth more than the clock.

**The wire was the larger delay, and nobody had put it next to the clock.** 19.10 measured
the serial line as a *bandwidth* problem and solved it with deadbands: 92 B/s, 0.8% of
115200. But 29 bytes at 115200 8N1 is **2.5 ms on the wire**, and that is latency, not
bandwidth. Set beside 19.1's table:

| | mean command age | wire | total |
|---|---|---|---|
| 200 Hz, 115200 | 2.5 ms | 2.5 ms | 5.0 ms |
| 500 Hz, 115200 | 1.0 ms | 2.5 ms | 3.5 ms |
| **500 Hz, 921600** | **1.0 ms** | **0.31 ms** | **1.3 ms** |

Raising the clock alone buys 1.5 ms of the 5.0; the baud buys 2.2 ms more for a one-line
change on each side. 19.10 deferred it only because it was a reflash, and reflashes were
allowed again on 2026-09-01. `SERIAL_BAUD` in `src/constants.h` and
`link.SerialComm.BAUD` must match, and **a mismatch is silent** -- there is no handshake
and no ack, so the firmware simply never parses a command and the coils hold their last
value. The deadbands stay: they now exist for the latency of a line and because every
byte not sent is a byte that cannot arrive corrupted.

**What a 2 ms period actually broke.** Four things, none of them the arithmetic:

- **The pacing sleep was capped at a flat 2 ms**, which *is* the period at 500 Hz. Every
  sleep overshot the next step and the grid resynced on every tick. Now `ts / 4`.
- **Two "nothing was sent" log lines ran every tick** -- the disarmed and
  `enable_freq_cmd=False` paths, i.e. the default ones -- into a line-buffered file. 500
  flushes a second to say nothing changed. Rate-limited to `WITHHELD_LOG_S = 0.5`, the
  same period as `RESEND_S`, since both answer "is this thing still alive".
- **The CSV row was line-buffered**, one flush a tick. Block-buffered now; `fh.close()`
  in the existing `finally` flushes the tail on every exit that runs Python at all.
- **`drain()` read one byte per syscall.** A 120-byte telemetry line cost 120 syscalls in
  whichever tick it landed in. One read of `in_waiting` now, which means the framer must
  handle several lines in one buffer and a CRLF split across two reads -- both covered by
  `link.demo()`.

**Measured, 10 s against a real-time 100 Hz pose source:**

| design | achieved | dt med | dt p95 | dt max | overrun |
|---|---|---|---|---|---|
| 200 Hz | 198.4 Hz | 5.00 ms | 5.17 ms | 70.67 ms | 0.2% |
| **500 Hz** | **498.5 Hz** | **2.00 ms** | **2.08 ms** | **16.70 ms** | **0.1%** |
| 1000 Hz | 989.4 Hz | 1.00 ms | 1.04 ms | 13.69 ms | 0.3% |

500 Hz is *cleaner* than 200 was. The rare tens-of-ms outlier is a GC or scheduler stall
and is not rate-dependent -- which is precisely why the loop now **counts overruns and
prints dt percentiles on exit**. It used to resync in silence, so a loop that never made
its period looked exactly like one that always did. macOS sleep granularity (~1 ms) was
expected to force a busy-wait and did not; that rung is still available and unclimbed.

**The gains are the null change 19.8 predicted.** At 500 Hz the closed-loop poles are
0.162 / 0.552 / 0.780 Hz -- identical to 30, 100, 200 and 400. Only `K` walks, to
`[66.01 19.16 0 0 49.53 0]`.

**The defect that stopped being deferrable.** 19.8's ladder put "thread a real `dt`
through `VelocityEstimator.update`" second, behind dropping the clock. Raising the clock
instead makes it worse in exact proportion: the estimator divided a fix's jump by the
*control* period, so it reported a rate too large by `pose_interval / ts` -- 2x at 200 Hz,
**5x at 500**. Fixed properly rather than scaled: a tick with no new fix passes `dt=None`
and the estimate is **held**, because between fixes the position comes from
`StatePredictor`, which moved it by exactly `v * dt`, so re-differencing returns the
velocity that produced it -- no information, only filter lag. `alpha` follows `dt` too, or
the rate dependence comes back from the other side. `simulate_hover._check_velocity_estimator`
is the guard: constant truth, fixes every k-th tick, k from 2 to 20, at both 5 ms and 2 ms.

**One thing the rate change silently broke, found by looking.** `Scenario.latency_frames`
counted *control steps*, so "1 frame of latency" meant 5 ms at 200 Hz and 2 ms at 500.
Raising the clock made every latency scenario easier while claiming to test the same
thing -- the same class of error as 19.9's phase-lock check measuring its own sample grid.
It is `latency_s` now, in seconds, and scenario b carries 19.1's measured 15 ms. All seven
scenarios pass at 500 Hz with the honest number.

### 19.14 The tighter design 19.11 promised, and why it is not taken

Written 2026-09-01. 19.11 named a prize:

> `design_hover_lqr.py` detunes the Bryson weights to 0.78 Hz explicitly *for latency
> robustness* [...] That justification was written against a 50 ms pipeline. At 15 ms it
> largely evaporates and a tighter design becomes admissible.

**Measured, it does not.** The sweep is one knob -- `design(authority=)`, which scales
Bryson's `u_max` so `r = 1/(u_max * a)^2`; larger means a bigger command is acceptable,
which buys bandwidth. Every scenario, at the honest 15 ms of 19.1:

| `authority` | fastest pole | at 15 ms | what fails |
|---|---|---|---|
| **1.0 (shipped)** | **0.780 Hz** | **PASS** | -- |
| 1.15 | 0.836 | PASS | -- |
| 1.30 | 0.889 | PASS | -- |
| 1.50 | 0.955 | PASS | -- |
| 1.75 | 1.032 | FAIL | `d_klat4.0x`: mag saturated 41% of the run (limit 20%) |
| 2.0 | 1.103 | FAIL | `d_klat4.0x`: saturated 51%, and settling 2.08 mm against 2.0 |
| 4.0 | 3.011 | FAIL | `b`: the noise-plus-latency scenario |

**The ceiling is not latency.** It is `d_klat4.0x` -- the scenario where the plant's
lateral gain is four times the design's -- and it fails on **actuator saturation**, which
is a statement about authority against an uncertain plant and has nothing to do with delay.
Latency only becomes the binding term above `authority = 4`, well past where the loop has
already saturated. So 19.11's argument was right about latency and wrong about the
conclusion: removing the latency objection does not make a tighter design admissible,
because latency was not what was holding it.

**The honest headroom is 0.78 -> 0.955 Hz**, 22%, and it is still 10x slower than the 10 Hz
structural mode from 14. It does not change what the loop can damp. It is available
(`authority=1.5`) and **not taken**, for a reason the table makes plain: the binding
scenario exists because `k_lat = 0.05` is a **seed, not a measurement** -- the code says so
at the parameter. The margin being spent is the margin that exists precisely because
nobody has measured the plant gain. Spending it to buy 22% of a bandwidth that is still an
order of magnitude short of the mode is the wrong trade.

**Measure `k_lat` first.** That is a better session than this one was: it collapses the
`d_klat` family from a 0.25x-4x uncertainty sweep to a number, and whatever headroom is
real would then be visible instead of insured against.

**One thing worth recording about the design as it stands.** At 25 ms, `d_klat4.0x` fails
at `authority = 1.0` too. The shipped controller's latency margin on the gain-mismatch
scenario is therefore between 15 and 25 ms -- narrower than "detuned for latency
robustness" suggests, and worth knowing before anything lengthens the pipeline again.

### 19.15 The pipeline in C++

Written 2026-09-03; the chapter is `pose/theory.md` 21. Everything `est.update` does per
frame was ported to C++ and held to the Python to rounding: same 246 of 250 frames on
the bench replay, identical trust-region iteration counts on every one, 4e-7 mm at p95.
The solve went **4.1 -> 0.4 ms** -- 19.4's diagnosis, that it was bound by the count of
numpy and cv2 calls and not by arithmetic, in full. Segmentation barely moved, being
OpenCV in both cores. Wall 8.8 -> 5.3 ms a pair (113 -> 189 Hz), which
against 19.1's table is a fraction of a degree of phase at 0.78 Hz; 19.14 stands. Holding
the port to the reference also found three things the reference was doing unrecorded --
the cv2 wheel's `remap` stopped quantising when it moved to OpenCV 5, the analytic
Jacobian is a float32 computation, and one float32 ulp of `fitEllipseDirect` moves the
solve by 0.4 mm on 5% of frames -- all in 21.2.

## 20. Robust stability: what mass and a centre-of-mass offset can and cannot do

18.14 measured a thrust axis 4.6 deg off vertical, growing with thrust, and named three
candidate causes without distinguishing them: a centre-of-mass offset from the spin axis, a
blade plane not perpendicular to that axis, or the robot seated at an angle on the rod. This
section asks the control question that sits behind them -- over what range of mass and
thrust-axis error is the shipped controller still asymptotically stable? -- and finds on the
way that **spin-averaging rules out the first two candidates by two orders of magnitude.**

The answer is exact rather than sampled, because this plant is unusually cooperative.

### 20.1 The uncertainty is rank-one, and the closed loop is affine in it

`hover_model.linearize_reduced` returns a state matrix with **no uncertain entries at all** --
a double-integrator pair -- and puts every parameter into two scalars of $B$:

$$b_\ell = g\,k_{lat}, \qquad b_v = \frac{2g}{f_h}.$$

$A$ is nilpotent blockwise, so the ZOH discretisation is exact and *linear* in each scalar,
$B_{d,i} = [\,T_s^2/2,\ T_s\,]^\top b$, and `augment_integrators` appends rows that contain no
parameter either. The shipped $K$ has off-block entries of order $10^{-13}$, so the six-state
augmented loop splits into two three-state SISO loops, each of the form

$$\boxed{\ M_i(b) \;=\; A_{a,i} \;-\; b\,\mathbf v\,\mathbf k_i^\top,
\qquad \mathbf v = [\,T_s^2/2,\ T_s,\ 0\,]^\top,\ A_{a,i}\ \text{parameter-free}.\ }$$

A **rank-one, exactly affine** perturbation. Its characteristic polynomial is
$\chi_{ol}(z) + b\,N(z)$ with $N(z) = \mathbf k_i^\top\operatorname{adj}(zI - A_{a,i})\mathbf v$:
a textbook root locus in $b$. So the certified set of gains is a genuine stability boundary,
found by bisection to machine precision, and **no LMI, no polytopic over-bound and no norm
bound is needed** -- none of the conservatism those tools carry applies here.

`robust_cert._self_check` verifies the affineness numerically rather than trusting this
paragraph, and checks that rebuilding through the design code reproduces the shipped
closed-loop poles (0.162, 0.552, 0.780 Hz) to $10^{-9}$.

| axis | certified gain multiplier | down | up |
|---|---|---|---|
| lateral | $[0.0802,\ 106.46]$ | 21.9 dB | 40.5 dB |
| vertical | $[0.1097,\ 156.60]$ | 19.2 dB | 43.9 dB |

### 20.2 Mass enters one scalar, and thrust binds long before stability

The 1-D plant never forms $k_T$: lift is written $g\big((f/f_h)^2-1\big)$, so mass appears
*only* through $f_h = \sqrt{m_R g/k_T}$, and

$$b_v = \frac{2g}{f_h} \;\propto\; \frac{1}{\sqrt{m_R}}
\qquad\Longrightarrow\qquad
\frac{m}{m_{nom}} = \left(\frac{b_{v,nom}}{b_v}\right)^{2}.$$

A heavier robot is a *lower* loop gain. The map is monotone, so the certified interval in
$b_v$ maps to a certified interval in mass with no slack introduced:

$$\frac{m}{m_{nom}} \in [\,4.1\times10^{-5},\ 83.1\,].$$

**That number is true and useless, which is the finding.** $f_h$ scales as $\sqrt{m}$, so a
mass ratio of only $(f_{stepout}/f_h)^2 = (225/160)^2 = 1.98$ already drives the required
hover frequency onto `F_STEPOUT_HZ`. **Thrust and step-out bind before stability does, by a
factor of 42.** Mass uncertainty is not a stability problem on this rig and no robustness
effort should be spent on it; 6.5's exchange rate -- a relative error $\varepsilon$ in $f_h$
is indistinguishable from $-\varepsilon$ in the command -- remains the operative statement,
and it is about *authority*, not stability.

### 20.3 A centre-of-mass offset cannot produce a steady tilt

This is the part 18.14 needed and did not have.

**A free body feels no gravity torque about its own centre of mass.** Once the robot is off
the pad, gravity acting at the COM has zero moment arm about the COM, so a COM offset cannot
tilt it directly. The only path is the **thrust line**: thrust acts at the blades'
aerodynamic centre, and if the COM is displaced from that line by $d$ transverse to the spin
axis, the moment about the COM is

$$\tau_0 = d\,T = d\,m_R g\,\frac{T}{m_R g}.$$

Proportional to thrust -- which is exactly 18.14's measured signature, and the reason that
section reads the growth with thrust as evidence *for* a fixed geometric offset.

**But $d$ is body-fixed, and the body spins.** In the lab frame that torque rotates at
$\omega$, so it drives 11.3's tilt block at the spin frequency itself:

$$I_t\ddot\chi + (iI_s\omega + c_t)\dot\chi + \kappa_t\chi = \tau_0 e^{i\omega t}
\qquad\Longrightarrow\qquad
\chi(t) = \frac{\tau_0\,e^{i\omega t}}{\kappa_t - (I_s+I_t)\omega^2 + i c_t\omega}.$$

The response is a **synchronous coning whose mean over one revolution is zero**, not a steady
tilt. It is off resonance by construction: nutation sits at $(I_s/I_t)\omega = 1.65\,\omega$
and never at $\omega$, so nothing amplifies it. At 126 Hz the inertia term
$(I_s+I_t)\omega^2 = 3.38\times10^{-3}$ exceeds $\kappa_t = 2.45\times10^{-5}$ by 138x, so
$\kappa_t$ drops out and

$$\boxed{\ |\chi| \;\simeq\; \frac{m_R g\,d}{(I_s+I_t)\,\omega^2}\ }$$

Dropping $c_t$ makes this an **upper** bound on the response (12.8 sets it to zero; 11.3 shows
a plausible value is three orders too small to matter). Evaluated:

| $f$ | coning per mm of offset | offset needed for 4.6 deg |
|---|---|---|
| 126 Hz (measured liftoff) | 0.0140 deg | **329 mm** |
| 160 Hz (design point) | 0.0086 deg | 532 mm |
| 210 Hz (ramp target) | 0.0050 deg | 918 mm |

**A COM offset is ruled out.** Producing the measured 4.6 deg would take a 329 mm offset on a
robot a few millimetres across -- wrong by two and a half orders of magnitude. A generous
1 mm offset buys 0.014 deg, which is below the pose estimator's own scatter.

**The same argument kills the second candidate.** A blade plane tilted by $\gamma$ from the
spin axis is also body-fixed, so its thrust vector sweeps a cone at $\omega$ and its lateral
component averages to zero over a revolution, leaving only an $O(\gamma^2)$ loss of vertical
thrust. Any body-fixed asymmetry is annihilated by spin-averaging; that is what spinning is
*for*.

**What survives is a lab-frame tilt of the spin axis**, 18.14's third candidate, and the
azimuth data already says so. `mixer_sign.py` measured the rotor normal sitting 1.5 deg off
the datum axis at rest and growing to 3.1 deg by 105 Hz **in a fixed azimuth near 150 deg,
with concentration 0.97**. A body-fixed defect would present an azimuth advancing at the spin
rate; a concentration of 0.97 about one lab direction is a lab-frame misalignment. And a spin
axis tilted by $\beta$ in the lab gives lateral acceleration $g(T/m_Rg)\sin\beta$, growing
with thrust just as the measurement shows, with no body-fixed offset required.

**This is a more hopeful reading than 18.14's.** That section concluded "all three are
mechanical and none is addressable from the host". If the surviving mechanism is instead a
lab-frame tilt that *grows with drive* -- 1.5 deg at rest, 3.1 deg by 105 Hz -- then part of
it is a drive-dependent field asymmetry, and a field asymmetry **is** addressable from the
host, through the same mixer trim `mixer_sign.py` was written to calibrate. The test that
separates them is already specified there: a seating angle is present at zero drive, a field
asymmetry is not.

### 20.4 The three channels a COM offset does open

Recorded so the ruling-out above is not read as "COM offset is harmless":

1. **A constant lateral acceleration bias.** The augmented integrator on $x$ rejects it in
   steady state, so it is not a stability question -- but it is spent inside `mag_max`, and
   6.6 prices exactly this trade for the vertical axis. 18.14's 0.079 g is 8 % of a
   one-$g$ authority budget. `simulate_hover.Scenario.dist_accel` is the channel for it and
   is currently set non-zero nowhere.
2. **A once-per-rev component at 50-230 Hz.** Against closed-loop poles at 0.162-0.780 Hz
   this is two decades out of band. It is a vibration question, not a control question.
3. **A rotation of the lateral input direction.** The only one of the three that can
   destabilise, and the subject of 20.5.

### 20.5 Input rotation is exactly a phase margin

The runner drives **both** lateral axes with the same $K$ row 0, so the two axes are
identical. Collect them into $\xi = x + iy$ as 14.2 does: a common rotation $\psi$ between
the commanded lateral direction and the realised one is then exactly a **complex loop gain**
$b_\ell\,\kappa\,e^{i\psi}$ on the same three-state loop. The certified set is a region in the
complex gain plane -- its real-axis extent is a gain margin, its angular extent a phase
margin -- and because $A_{a,i}$, $\mathbf v$ and $\mathbf k$ are real, the region is symmetric
under conjugation, so $|\psi|$ is the whole story.

$$\boxed{\ |\psi| < 69.4^\circ\ \text{at nominal gain}\ }$$

| $\psi$ | 0 | 15 | 30 | 45 | 60 | 67.5 |
|---|---|---|---|---|---|---|
| $\kappa_{lo}$ | 0.080 | 0.137 | 0.224 | 0.366 | 0.637 | 0.900 |
| $\kappa_{hi}$ | 106.5 | 102.8 | 92.1 | 75.0 | 52.7 | 39.9 |

**This is the margin that is actually short.** 12.8 measured, on the spatial model, that
"commanding a field tilt of 0.1 in $+x$ moves the spin axis into $+y$ at 72 deg from the
command after 50 ms". 72 deg is **outside** the certified 69.4 deg, by 2.6 deg. The two
numbers come from different models and should not be over-read as a prediction of failure,
but they are the same quantity, they are the same size, and the gain-margin columns show the
region collapsing fast past 60 deg: at $\psi = 67.5^\circ$ the tolerable gain band has
narrowed from $[0.08,\,106]$ to $[0.90,\,39.9]$, and its lower edge has climbed to within
10 % of nominal. A loop that is 21.9 dB from its low-gain boundary at $\psi = 0$ is 0.9 dB
from it at 67.5 deg.

**The correction is free, which is the practical point.** A known rotation is removed by
rotating the command, and `mixer_sign.py` already prints the required trim in exactly that
form ("a trim must use `az = P + 180 - off`"). The 69.4 deg is therefore tolerance on the
*residual* after calibration, which is ample. Uncalibrated, it is not. **Run `mixer_sign.py`
to conclusion before arming the lateral loop** -- it has never been run to conclusion, and
across five runs on 2026-09-01 the lateral loop never armed once.

### 20.6 What this does not show

13.5 states the trap this section has to avoid: certifying a linearised model against itself
measures the control law and nothing else. Four limits, in the order they bite:

1. **Two of the parameters are not measured.** $k_{lat}$ is still the seed guess of
   `hover_model.py:70`, and the mixer rotation is unmeasured. A wide certified region is
   permission to arm the loop and identify them, not evidence the rig sits inside it.
2. **Latency is not in the certificate.** 19.14 measured `d_klat4.0x` failing at 25 ms while
   passing at 15 ms, so the gain-mismatch margin is bounded by *latency*, not by the root
   locus computed here. The 106x upper gain figure is a statement about an undelayed loop and
   must not be quoted without that qualification.
3. **The certificate holds $K$ fixed while perturbing the plant.** That is the right question,
   but `_anchor` re-trims $f_h$ at runtime to the frequency the ramp actually reached, which
   *shrinks* the effective mass error -- so the mass result is conservative in the safe
   direction.
4. **A Gaussian is never fully certified.** Unbounded support means no box covers every draw.
   What is certified is a box plus the probability mass it carries, which is why
   `robust_cert.as_sigmas` reports a miss probability alongside every half-width and never
   returns zero.

### 20.7 Correspondence with the implementation

| Quantity | Where |
|---|---|
| $M_i(b) = A_{a,i} - b\,\mathbf v\,\mathbf k_i^\top$ | `robust_cert.axis_loop`, built through the design code so it cannot drift |
| certified gain interval | `robust_cert.gain_interval`, bisection on the root locus |
| complex-gain region, phase margin | `robust_cert.complex_region`, `robust_cert.phase_margin` |
| $m/m_{nom} = (b_{nom}/b)^2$ | `robust_cert.mass_interval` |
| $|\chi| = m_R g d/((I_s+I_t)\omega^2)$ | `robust_cert.coning_from_com` |
| box-to-sigma inversion | `robust_cert.as_sigmas` |
| the rotation this bounds | `mixer_sign.py`, the rig measurement that anchors it |

```bash
uv run python controller/control/robust_cert.py     # self-check, then the report
uv run python controller/control/simulate_hover.py  # scenario d must stay consistent with it
```

The certified lateral band $[0.080,\ 106]$ contains scenario d's $0.25\times$, $1\times$ and
$4\times$ $k_{lat}$, and all three PASS in the nonlinear simulation -- a falsification test
the certificate could have failed and did not.

---

## 22. The commanded phase is not the current phase

### 22.1 The objection

Raised in review, 2026-09-01, against the write-up's claim of $\pm 0.5^\circ$ phase
precision:

> That is the *commanded* phase. What the robot sees is the *current* phase, and it is set
> by the RLC network. Near resonance the current phase detunes as
> $\Delta\theta \approx -2Q\,(\Delta\omega/\omega_0)$, and a $\pm 20\%$ electrolytic
> tolerance moves $f_r$ by 10 %, so the channel-to-channel phase error is many times the
> claimed precision -- and it is analog, not digital.

**The structure of this is correct and the conclusion stands.** The commanded phase is not
the quantity the rotor responds to, the difference is per channel, and nothing in this
project has ever measured it. It enters the tilt budget the same way an amplitude imbalance
does, so a tilt attributed to coil asymmetry currently has an uncontrolled contribution
mixed in.

Three things about it need correcting, and one of them is worse than the objection.

### 22.2 The current phase needs exactly two numbers per channel

For a series RLC the current phase relative to the driving voltage is
$\theta = \arctan\big((\omega L - 1/\omega C)/R\big)$. Substituting
$\omega_0 = 1/\sqrt{LC}$ and $Q = \tfrac{1}{R}\sqrt{L/C}$:

$$\boxed{\ \theta_k(f) = \arctan\!\left(Q_k\left(\frac{f}{f_{0,k}}
   - \frac{f_{0,k}}{f}\right)\right)\ }$$

$R$, $L$, $C$ and the drive amplitude have all left. Expanding about $f_{0}$ recovers the
reviewer's $2Q\,\Delta f/f_0$ as the small-detuning limit, so this is the same statement
without the linearisation.

**Two numbers per channel is what makes the measurement possible on this hardware**, and the
reason is the term that vanished. The absolute gain of the CS path is the least trustworthy
quantity on the board:

| | |
|---|---|
| VNH5019 sense ratio | $K = 4670$ to $10110$, $\pm 30\%$ part to part (`hw_references/VNH5019_CS.png`) |
| CS drift | $\pm 19\%$ over temperature, same table |
| `R4`, CS to ADC | **no value in the BOM** (`docs/PCB_Design_Documentation.md` 157) |
| `SENS` $\approx 15.3$ A/V | empirical, and $\sim 3.2\times$ what a 1 k$\Omega$ load implies -- unexplained |

An *amplitude* calibration inherits every one of those. A *phase* calibration inherits none
of them, because a real scalar gain drops out of an argument. The same cancellation is why
`coil_phase.fit_channel` solves for $(A, f_0, Q)$ from the shape of $|I|(f)$ and throws $A$
away, and why its self-check asserts that scaling the input current by $3.17$ moves the
fitted $f_0$ by less than $10^{-3}$ Hz.

A second consequence, less obvious: **report the phase relative to the four-channel mean,
never absolutely.** The CS path has its own delay -- $t_{DSENSE}$ is 20-50 µs, which is
1.4-3.6 deg at 200 Hz -- and it is common to four identical channels, so it cancels in a
difference and does not cancel in an absolute number. Nothing is lost, because what tilts
$\hat n$ is the channel-to-channel difference in the first place.

### 22.3 The magnitude is wrong, because $Q$ is not known to within a factor of five

The review computed $Q = 1.29$ from $R = 1.3\ \Omega$. That number is not invented -- it is
the write-up's own measured coil DC resistance (§3.2). But `constants.py` carries
$R = 6.9\ \Omega$ for the series channel, and the two give:

| $R$ | source | $Q$ at 800 µF | provenance |
|---|---|---|---|
| $1.3\ \Omega$ | write-up 3.2 | 1.02 | measured, coil DC only |
| $6.9\ \Omega$ | `constants.py` | 0.19 | measured, series channel incl. driver |
| $15\ \Omega$ | `constants.py` | 0.09 | measured per channel, the driver's view |

and 18.6 already records a fourth, $Q \sim 22$, quoted in the justification for
`S_LIM = 0.8` and supported by nothing. **Four values spanning two and a half orders of
magnitude, all of them in this repository.** 18.6 closed with "the resonance is unknown and
any statement that 160 Hz is 'at resonance' is unsupported"; that is still true, and it is
now the binding obstacle to answering the review at all.

The capacitor bank was doubled to **800 µF per coil array on 2026-09-01**, which moots every
fitted constant taken before that date -- $f_0 \propto C^{-1/2}$, so the measured 174 Hz
peak scales to 112 Hz against the fitted 334 µF or 138 Hz against the write-up's selected
500 µF. Those two do not agree either, which is the same disease. With the confirmed
$L = 1.4$ mH the nominal is $f_0 = 150$ Hz.

Predicted spread at 190 Hz, over the range the unknowns actually span:

| bank | effective tol. | $Q = 0.19$ | $Q = 1.02$ |
|---|---|---|---|
| one cap | 20 % | 4.5 deg | 19.9 deg |
| five in parallel | 8.9 % | 2.0 deg | 8.8 deg |
| eight in parallel | 7.1 % | 1.6 deg | 6.9 deg |

peak-to-peak, in degrees. **A parallel bank averages its own tolerance down by $\sqrt N$**,
which the objection did not account for and which works in the rig's favour: the write-up
describes the 500 µF bank as five 100 µF parts in parallel, so the effective tolerance is
nearer 9 % than 20 %.

The honest summary is that **the answer is somewhere between 1.6 and 20 degrees**, that the
range is set by an $R$ this project has measured three incompatible ways, and that no
argument settles it. One measurement does.

### 22.4 The digital claim is worse than the objection assumed

The $\pm 0.5^\circ$ figure comes from `writeup` 3.5, where it describes the multi-node
controller/client hardware synchronisation: a controller node emits a zero-shift calibration
signal and clients pick it up on an interrupt. **That path is compiled out.**
`platformio.ini` sets `-D USE_SYNC=0`; `SYNC_LATENCY_US` and `SYNC_AS_SERVER` are dead
config, and `PwmController::setPhase` early-returns entirely under
`USE_SYNC && SYNC_AS_SERVER`.

What ships instead is plain GPIO toggling out of a periodic `esp_timer` at **25 µs**
(`PwmController.cpp:146`). LEDC drives only the 20 kHz carrier and never the phase; there is
no MCPWM anywhere in the tree. One tick is therefore the real quantum:

| $f$ | 100 Hz | 150 | 174 | 210 | 300 |
|---|---|---|---|---|---|
| one tick, deg | 0.90 | 1.35 | 1.57 | 1.89 | 2.70 |

A finer truncation exists downstream -- `startUs` is computed to 1 µs, 0.06 deg at 174 Hz
(`PwmController.cpp:281`) -- but the 25 µs ISR is what binds. **So the claim is off by 3-4x
on its own terms, before any analog effect, and it describes a build that has never flown.**

This matters for the fix as well as the write-up. A trim finer than the quantum buys
nothing, so the quantum is the floor on any residual this rig can report, and `coil_phase.report`
prints the two side by side rather than quoting a residual that the hardware cannot express.

### 22.5 Measuring it: a lock-in at $2f$ on an unsigned pin

Three properties of the existing current-sense path each destroy phase on their own: the ADC
is paced at ~1 kHz (`PwmController::_serviceCurrentLoop`), a 50 ms EMA smooths it
(`current_sense.cpp`), and `driveTelemetry` prints at 2 Hz. `coil_probe.cpp` bypasses all
three -- raw reads, no filter, accumulated in quadrature and reported once.

**The CS pin is unsigned**, mirroring whichever high-side FET is sourcing, so a sinusoidal
coil current arrives full-wave rectified. $|\sin x|$ has no component at its own rate:

$$|\sin x| = \frac{2}{\pi} - \frac{4}{\pi}\sum_{k\ge 1}\frac{\cos 2kx}{4k^2-1}
  = \frac{2}{\pi} - \frac{4}{3\pi}\cos 2x - \dots$$

so the lock-in runs at $2f$ and halves the resulting angle, recovering $\theta_k$ modulo
$180^\circ$. That ambiguity is harmless against a spread of a few degrees -- a channel
genuinely half a turn out is an amplitude fault, and `coil_balance.py` is the tool for that.
The leading minus sign matters and is not cosmetic: the reference sits half a turn from the
current, so $\arg$ returns $2\psi + \pi$, and dropping the $\pi$ would bias every channel by
90 deg.

**The mux skew is removed by construction, not by correction.** The ESP32 ADC needs a
throwaway read after switching pins, so the four channels are visited in sequence; at 200 Hz
each 50 µs of skew is 3.6 deg, which is the entire effect being measured. Rather than
subtract a nominal offset, every sample carries a timestamp taken next to its own
conversion, so the skew never enters the accumulators at all.

**This does not reopen field-oriented control.** 17.2 rules FOC out because commutation
needs *signed* per-phase current sampled synchronously *every cycle in real time*, and
concludes that "angle-referenced control waits on either a rotor-angle sensor or per-phase
current sensing; neither exists on this board." That remains true. What is done here is a
*calibration*: one frequency, held for a second, coherently averaged over hundreds of cycles,
off the flight path, producing two constants. It answers a question 17.2 was not asking, and
it hands the loop no angle it could fly on.

**Two independent routes, deliberately not averaged.** The shape of $|I|(f)$ over a ramp
gives $(f_0, Q)$ per channel from magnitudes alone; the `probe=` lock-in gives $\theta$ per
channel directly, from angles alone. They share no arithmetic, so agreement is the check that
the lock-in is measuring the coil and not its own reference. `coil_phase.compare` prints both
and never blends them, for the reason 18.6 gives about averaging numbers that disagree.

### 22.6 The trim, and why it must depend on frequency

$\theta_k$ changes sign through resonance and swings tens of degrees across a 2-210 Hz ramp,
so a single constant offset is wrong nearly everywhere it is applied.
`PwmController::setPhaseTrim` therefore stores $(f_{0,k}, Q_k)$ and re-evaluates the closed
form on every frequency change, caching the result outside the ISR spinlock -- four `atanf`
calls with interrupts disabled would be a large fraction of the 25 µs tick.

The base phase is kept separately from the corrected one (`_basePhaseDeg` against
`_phaseOffsetsPct`) so that repeated frequency changes cannot accumulate a correction into
the command. `setPhase` keeps meaning "the phase I want the *current* at", which is why
`PwmSequencer` needs no knowledge that a trim exists.

**A half-filled table is refused.** `setPhaseTrim` disarms unless all four channels carry a
positive $f_0$ and $Q$, because three trimmed channels and one raw is a larger asymmetry
than trimming none -- and an uncalibrated build must drive the commanded phase raw rather
than a guessed correction.

### 22.7 What this does not settle

**No measurement has been taken yet.** Everything above 22.4 is a prediction bracketed by an
unresolved $R$; `COIL_F0_HZ` and `COIL_Q` in `src/drive_common.h` are all zero, the trim is
disarmed, and the residual this section exists to report does not have a number. Until a
probe sweep runs on the 800 µF bank, the correct statement about channel-to-channel phase is
"between 1.6 and 20 deg, unmeasured" -- which is still a better claim than $\pm 0.5^\circ$,
because it is true.

Four further limits:

1. **The ramp must pass through resonance.** $f_0$ and $Q$ are separable only from a band
   that brackets the peak. A ramp stopping short still fits, and fits confidently -- that is
   exactly how 400 µF entered `constants.py` (18.6: "fitted in the one band where it is the
   only thing visible"). `coil_phase.fit_channel` refuses when the peak sits at an edge, and
   its self-check asserts the refusal on a band truncated below $f_0$. The most recent run,
   `20260901_092547`, reaches only 109 Hz and cannot support this fit.
2. **The rotor is present during a probe.** The measurement is of the electrical network;
   back-EMF from a turning rotor is not modelled. The balance-loop comment
   (`tilt_ccw_nodisk`) reports CS as blind to the disk, which suggests the coupling is small,
   but "suggests" is the strength of that claim.
3. **A phase trim does not touch the amplitude imbalance.** The 39 % current spread of 18.8
   and the 1.63 A dipole at 145 deg of `coil_balance.py` are a separate defect with a
   separate trim. Whether phase or amplitude dominates the measured 1.6 deg lean is precisely
   what is not yet known, and correcting one does not test the other.
4. **The residual floor is the 25 µs tick**, not the lock-in. Below ~1.7 deg at 190 Hz there
   is nothing the commanded phase can express.

### 22.8 Where this lands in the tilt budget

20.5 certifies $|\psi| < 69.4^\circ$ of input rotation, and observes that the margin is
already the short one: 12.8's measured 72 deg sits outside it. A per-channel current-phase
error is not the same rotation as $\psi$ -- it is a distortion of the field rather than a
rigid rotation of the command -- but it spends from the same budget, and it does so
*uncalibrated*, which is the condition 20.5 says the 69.4 deg is not ample for. That is the
argument for measuring it rather than bounding it: 20.5's tolerance is on the residual after
calibration, and there has been no calibration.

### 22.9 Correspondence with the implementation

| quantity | where |
|---|---|
| $\theta_k(f) = \arctan(Q_k(f/f_{0,k} - f_{0,k}/f))$ | `coil_phase.theta`, and `PwmController::_recomputeTrim` |
| $(f_0, Q)$ from the shape of $\lvert I\rvert(f)$ | `coil_phase.fit_channel`; refuses an edge peak |
| $(f_0, Q)$ from measured angles | `coil_phase.fit_probe` |
| the two routes, side by side | `coil_phase.compare` -- never averaged |
| per-run per-channel fit | `takeoff_report.metrics`, key `phase_fits` |
| the $2f$ lock-in, mux skew, coherence | `lib/PwmController/src/coil_probe.cpp` |
| `probe=<hz>[:<ms>[:<duty>]]`, IDLE only | `cmdProbe` in `src/main_flight.cpp` |
| the answer line the host parses | `link.parse_probe`, `link.ProbePoint` |
| `probe=` counted as drive (heat) | `link.SerialComm._note_drive` |
| the measured constants | `COIL_F0_HZ` / `COIL_Q`, `src/drive_common.h` -- zero until measured |
| arming, and the refusal of a partial table | `PwmController::setPhaseTrim` |
| the commanded-phase quantum | `esp_timer_start_periodic(_periodicTimer, 25)`, `PwmController.cpp:146` |

```bash
uv run python controller/control/coil_phase.py             # self-check, no hardware
uv run python controller/control/coil_phase.py --measure   # probe sweep, ENERGISES
uv run python controller/control/coil_phase.py --fit results/takeoff/<run>.csv
```

## 23. The channel-0 amplitude sweep, and why its photographs can be trusted

`spiffs_data/tilt.json` holds coils B, C and D at 100% carrier and steps coil A through
20, 40, 60, 80 and 100%, 2.5 s at each, after a 30 s EASE ramp to 150 Hz. It is run by
`main_tilt.cpp` (`pio run -e tilt -t upload`, then `-t uploadfs`) and filmed by
`controller/control/tilt_sweep.py`. Three things about it are decisions, not details.

### 23.1 The sweep is in current, not in duty, and that is why the balancer stays on

`main_tilt` calls `enableCurrentBalance()`, and the instinct from the calibration rigs
(§`platformio.ini`, the PASSTHROUGH note) is that a loop which equalises the four
currents must erase a deliberate asymmetry. It does not, because the balancer is not an
equaliser. `CurrentBalanceController` normalises each channel's *commanded carrier* into
a ratio $r_i = c_i / \max_j c_j$ and drives channel $i$ toward $m\,r_i$, where $m$ is the
magnitude set by the reference channels -- the ones within `refBandPct` of the top
carrier. Stepping A to 20% therefore makes it a follower with $r_A = 0.2$ and asks the PI
for one fifth of the current the held channels carry.

The distinction matters for what the photographs mean. Open loop, a carrier of 20% is a
statement about a *duty cycle*, and the current it produces depends on that coil's
resistance, its capacitor, and how far the drive frequency sits from that channel's own
$f_0$ -- all four differ, and §22 is the argument that the last of these is unknown to a
factor of five on this rig. Closed loop, 20% is a statement about *current*, which is
the quantity the field is proportional to. So the sweep's independent variable is
$I_A/I_{B,C,D} \in \{0.2, \ldots, 1.0\}$, and the lean angle read off each still can be
plotted against a number that means the same thing on any day.

The calibration rigs run PASSTHROUGH for the opposite reason: they measure the coupling
*from* duty *to* current, so closing that loop would measure the loop.

### 23.2 The label line is the only event, and no offset is fitted

`main_tilt` parses no serial. Nothing is commanded, there is no run CSV, and so none of
the markers §19 and `sync.py` rely on exist -- there is no `<> csv t=0`, because there is
no host clock origin to record. What there is instead is `label=<name>`, printed by
`main_tilt`'s `loop()` whenever `JsonPwmSequencer::stepLabel()` changes. On the label
change, not the step change: `resolution_ms` is 25, so a 2.5 s hold compiles to ~100
queue entries that all carry one label.

`tilt_sweep.run` reads the cameras and the serial port in one process, so that line is
stamped with the same `time.monotonic()` as the frames beside it, and the alignment is
again free rather than estimated (`sync.py`'s opening argument applies unchanged). The
still for each step is cut `SETTLE_BACK_S` before the *next* label rather than after the
current one: the rotor spends most of a 2.5 s hold swinging into the new lean, and it is
the settled attitude that is the measurement.

### 23.3 A take that dropped a frame cannot be cut, and is refused

`FlightWriter` writes a row to `frames.csv` for every stereo read, but hands the frame to
a bounded queue and *counts a drop* rather than blocking when the encoder falls behind --
correctly, since a flight outranks its film. The consequence for cutting stills is that
row $n$ of `frames.csv` and frame $n$ of the mp4 are the same frame only until the first
drop, and thereafter differ by the running drop count, which is not recorded per frame.

Every still after that point would be a genuine photograph of the wrong duty cycle, with
nothing downstream able to notice: the images are all of the same robot in the same rig,
and only the filename claims which step it is. `tilt_sweep.stills` therefore reads
`meta.json` and refuses outright when `dropped > 0`. Refusing costs a re-shoot; the
alternative costs a plot whose x-axis is wrong by one step and which looks fine.

### 23.4 The normal as a tilt vector about the rest attitude

`tilt_report.lean_table` reports the disc normal $\hat n$ (sign resolved against the mast,
§23 preamble) not as an angle but as a tilt vector in the plane perpendicular to the rest
datum $\hat u$. With $\theta = \arccos(\hat n \cdot \hat u)$ and the unit lean direction
$\hat l = (\hat n - (\hat n\cdot\hat u)\hat u)/\lVert\cdot\rVert$, the two components are
$\theta\,(\hat l\cdot\hat e_1)$ and $\theta\,(\hat l\cdot\hat e_2)$, so their hypotenuse is
$\theta$ exactly and the azimuth is their `atan2`. That is the same $(\theta, \phi)$ split
`estimator._angles_from_normal` makes about world $+z$, moved onto the physical zero --
world $+z$ is camera A's optical axis and means nothing here (§23 preamble, `stereo_rig.json`).

The basis is a naming choice, not a measurement: $\hat e_1$ is camera A's $x$ projected onto
the rest plane and $\hat e_2 = \hat u \times \hat e_1$. Camera A's $x$ is the one direction
on this rig that is fixed and can be pointed at; nothing else about the choice matters, and
a different $\hat e_1$ only rotates the azimuth by a constant. The azimuth is nan under
0.5 deg of tilt for the reason `_angles_from_normal` gives: `atan2(0, 0)` is noise, not a
direction. Per-frequency plots draw every solved frame as a point with no line through
them -- below ~60 Hz the blades strobe (`body_angle.py` header), and a line through a
strobing rotor draws structure the data does not have.

### 23.5 One folder per take, and the frames outside a point are kept

`controller/report.py` runs the offline pass in one command. It adds no measurement: the
solve, the mast pass, the datum and the two-sensor table are 23.1-23.4 unchanged, and the
script is an order plus two things those stages did not produce.

The first is the **command record**. `sweep.log` was read only for its `label=` lines
(23.2); its 2 Hz telemetry -- the commanded frequency and the four duties -- was read for
the drop instant and then thrown away. It is now `telemetry.csv` in full, and every change
in it is `events.csv`: each label, and each duty step over 0.5 %. Frequency is deliberately
NOT evented. It ramps continuously, so every sample is a change and an event file of 464
rows says nothing; it is a column and a trace instead. Every rule on the timeline plot has
a row in `events.csv`, which is the property that makes a marked plot checkable.

The second is that `lean_table` now keeps the frames **outside** every frequency point,
with `freq_hz` -1. Those are the ramps and the coils-off rest windows, and their tilt is
the datum's own scatter measured against itself: on drone 1 it sits near zero between
points and steps to 20 deg inside them, which is the cheapest available check that the
datum is a datum and not an artefact of the window it was cut from. The per-point plots
and `summary` select on `freq_hz`, so they never see those rows.

## Appendix A: Correspondence with the MATLAB implementation

Two model families were ported *into* Python. The four 1-D files (three `*_gui.m`
plus `frequency_tracking_statespace_sim.m`) map onto §1–§11, and
`MultiCoilBeamformingGUI_quickGeom_rigidTilt_coil22mm.m` maps onto §12–§13 as
`ai/design/spatial_model.py`. Function and variable names are kept across the port, so the two
read side by side and `grep` finds the counterpart.

Three departures from the multi-coil original, each argued at its own section: the field
gradient is analytic rather than a six-point central difference (§12.6), the geometric phase
basis is computed once per step and shared by torque and force (§12.5), and `align_tau` exists,
defaulting to the MATLAB's zero, so §12.8's missing dissipation is an explicit parameter rather
than a silent one.

The four 1-D files carry an older robot's inertia and simulate a different machine; §3 has the
current value. The controller has no MATLAB counterpart at all: `spatial_mpc.py` and
`simulate_spatial.py` are new.

One file goes the other way. `matlab/open_loop_two_ring.m` is a back-port of §15's
two-ring result out of Python and into MATLAB, self-contained because the helpers
inside the multi-coil GUI file are local to it. It has **never been executed** --
MATLAB was not available where it was written -- so `matlab/two_ring_fixture.csv`
ships beside it as the Python model's output: any disagreement past 1e-9 is a
porting bug in the `.m` file, not a physics difference. Python remains the source
of truth.

## Appendix B: Why $\dot f_{max}$ and $f_{max}$ are the two feasibility axes

The torque budget $\tau_{max} \ge 2\pi I \dot f_f + k_d f_f^2$ is a single inequality in
two variables $(f_f, \dot f_f)$: its boundary is a downward parabola in the
$(f_f^2, \dot f_f)$ plane. $f_{max} = f_0\sqrt M$ is its $\dot f_f = 0$ intercept;
$\dot f_{max}(f)$ is a horizontal slice. Any commanded trajectory must remain inside
this region *with dynamic margin* for the $Q \approx 20$ phase swing excited at segment
boundaries: a conservative rule of thumb is to keep the quasi-static demand below
$\tau_{max}\sin(\pi/2 - \Delta)$ with a swing allowance $\Delta$ of a few tens of
degrees, or to shape $\dot f_f$ continuously (higher-order polynomial / exponential
segments) so little swing energy is injected.

## 24. Holding the seated rotor upright from the camera, and dropping a coil against that

### 24.1 Why this is possible on the mast and not in the air

The free rotor of 11.6 and 12.8 answers a field tilt with a precession 72 deg off the
command after 50 ms, undamped at 1.2 Hz, inside the heave band. On the mast none of that
applies: the rod reacts, the rotor cannot precess, and a duty step on one coil moves the
axis by 3-5 deg in about 0.3 s and stays there (`results/tilt_sweep/*/normal_angle_f*.png`,
2026-09-02). With the plate segmenter, the mast prior and the disc-mast fusion
(`pose/theory.md` 20), the axis is read at 2-3 deg per frame and ~0.5 deg over 0.25 s, with
the rod found in 87-95 % of frames -- the rejection of the pose normal as a feedback signal
in `attitude.py` was measured on the rim estimator in free flight and does not carry over.
So `control/tilt_servo.py` closes a loop, but a slow one: an integrator alone, pole at
0.3 Hz, under the 0.8 Hz latency rule (19.11) and far below the 4 Hz wobble, which it
averages rather than fights. It is an auto-trim, not a stabiliser.

### 24.2 The Jacobian is measured, and that is the sign certification

18.16 failed twice to measure the mixer's sign because the seated and airborne windows
never overlapped. Seated is the operating point here, so each coil is probed in turn --
duty 100 -> 80 for 0.6 s -- and the lean response is that coil's column of $J$ (deg per %
of drop), sign included. A column under three times the pre-probe scatter refuses the run.
Then $u \leftarrow u + K_i\,\Delta t\,J^{+}\,\bar\ell$ on four duties, clamped 40-100 with a
40 % cap on the total drop; the pseudo-inverse spends the drops where they act, and a coil
whose sign is "wrong" is simply a coil with a negative column. The self-check flips one and
converges. `duty=A:B:C:D` in `main_flight.cpp` is what the loop drives: four independent
carrier ceilings the single-lobe az/mag mixer cannot express, counted as drive by
`link._note_drive`.

### 24.3 Thirty percent of what

23.1 is right that the sweep's independent variable ought to be a current ratio and that
30 % of duty is not 30 % of current *across coils* (39 % spread at equal duty). Within one
coil at one frequency, though, the drive is a linear RLC and $I \propto$ duty, so once the
loop has converged and is FROZEN, setting coil A to $0.30\,d_{\mathrm{bal},A}$ is exactly
30 % of its balanced current. The 2 Hz `I[A]:` telemetry ratio is logged as the check, by
magnitude, because channel A's sense reads negative on this board (`main_tilt.cpp`). The
freeze is not optional: an unfrozen trim cancels the experiment it is meant to prepare.

### 24.4 What it cannot do, stated once

Zero phase authority (18.18): the 13 deg per-coil phase spread makes an elliptical field
no amplitude trim removes. The disc and the mast disagree by a systematic 5 deg, so a bias
in either becomes a bias in "upright" until one is shown to carry it. And nothing here
transfers to free flight -- the seated Jacobian is a property of the rod.
