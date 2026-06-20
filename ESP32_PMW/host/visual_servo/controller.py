"""Height controller: PD + gravity feedforward -> commanded rotation frequency.

The height plant is a double integrator (frequency -> lift -> accel -> vel -> pos),
so a PD controller with a gravity feedforward (F_HOVER, the equilibrium frequency
where lift = weight) is the right first controller. Kd damping is essential because
magnetic hover is lightly damped. Output is clamped to [F_MIN, F_MAX] and
rate-limited; the integrator (if used) has anti-windup.
"""
from config import ControlGains, DEFAULT_GAINS


class HeightController:
    def __init__(self, gains: ControlGains = DEFAULT_GAINS):
        self.g = gains
        self._integral = 0.0
        self._last_cmd = gains.f_hover

    def reset(self):
        self._integral = 0.0
        self._last_cmd = self.g.f_hover

    @staticmethod
    def _clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def update(self, z_ref_mm, z_mm, zdot_mm_s, dt):
        """Compute the commanded frequency (Hz).

        z_ref_mm : target height
        z_mm     : measured/filtered height (None -> hold last command)
        zdot_mm_s: filtered vertical velocity (for damping)
        dt       : timestep (s), for the integrator and slew limit
        """
        g = self.g
        if z_mm is None:
            return self._last_cmd  # lost marker: hold; firmware watchdog backstops

        err = z_ref_mm - z_mm
        deriv = -(zdot_mm_s if zdot_mm_s is not None else 0.0)

        # Integral with anti-windup clamp on its CONTRIBUTION (in Hz).
        i_term = 0.0
        if g.ki > 0.0:
            self._integral += err * dt
            i_term = self._clamp(g.ki * self._integral, -g.i_clamp, g.i_clamp)
            # back-calculate the integral state to the clamped value
            if g.ki != 0.0:
                self._integral = i_term / g.ki

        u = g.f_hover + g.kp * err + g.kd * deriv + i_term
        u = self._clamp(u, g.f_min, g.f_max)

        # Host-side slew limit (firmware also slews; this keeps logs honest).
        if dt > 0.0:
            max_step = g.max_slew_hz_s * dt
            u = self._clamp(u, self._last_cmd - max_step, self._last_cmd + max_step)

        self._last_cmd = u
        return u
