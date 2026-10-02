import casadi as ca
import numpy as np

# 1. COIL CENTERS (3D ndarray) & GEOMETRY
coil_centers = np.array(
    [
        [0.03, 0.00, 0.00],
        [0.00, 0.03, 0.00],
        [-0.03, 0.00, 0.00],
        [0.00, -0.03, 0.00],
    ],
    dtype=float,
)

# Initial orientations pointing along +z axis
init_orientations = np.tile([0.0, 0.0, 1.0], (len(coil_centers), 1))

R_coil = 0.0105  # Coil radius [m]
H_coil = 0.0150  # Coil height [m]
margin = 0.0010  # Physical clearance [m]

R_min = 2 * R_coil + margin
H_min = H_coil + margin
R_bore = 0.0150  # Central bore radius limit [m]

# Target points along central Z-axis
z_pts = np.linspace(0.01, 0.05, 5)
r_targets = np.column_stack([np.zeros_like(z_pts), np.zeros_like(z_pts), z_pts])

# 2. OPTIMIZATION SETUP
opti = ca.Opti()

# Decision Variables: 3D Positions (Nx3) AND 3D Orientations (Nx3)
p = opti.variable(len(coil_centers), 3)
n = opti.variable(len(coil_centers), 3)

# Initial guesses
opti.set_initial(p, coil_centers)
opti.set_initial(n, init_orientations)

# Bounds on positions
opti.subject_to(opti.bounded(-0.10, p, 0.10))

# Constraint 1: Unit vector norm constraint for orientation (||n_i||^2 = 1)
for i in range(len(coil_centers)):
    opti.subject_to(ca.sumsqr(n[i, :]) == 1.0)

# Constraint 2: Central bore clearance (keep coils off central axis)
for i in range(len(coil_centers)):
    r_axis_sq = p[i, 0] ** 2 + p[i, 1] ** 2
    opti.subject_to(r_axis_sq >= R_bore**2)

# Constraint 3: Cylindrical collision avoidance between coil pairs
for i in range(len(coil_centers)):
    for j in range(i + 1, len(coil_centers)):
        dx = p[i, 0] - p[j, 0]
        dy = p[i, 1] - p[j, 1]
        dz = p[i, 2] - p[j, 2]

        cyl_dist = (dx**2 + dy**2) / (R_min**2) + (dz**2) / (H_min**2)
        opti.subject_to(cyl_dist >= 1.0)

# 3. FULL 3D DIPOLE MAGNETIC FIELD OBJECTIVE
# B = (3 * (r · n) * r - r^2 * n) / r^5
total_relative_Bz = 0

for target in r_targets:
    target_mx = ca.MX(target).T
    for i in range(len(coil_centers)):
        dr = target_mx - p[i, :]
        dist_sq = ca.sumsqr(dr) + 1e-6

        # Dot product: (dr · n_i)
        dot_dr_n = ca.dot(dr, n[i, :])

        # Z-component of general 3D dipole field
        Bz_i = (3.0 * dot_dr_n * dr[2] - dist_sq * n[i, 2]) / (dist_sq**2.5)
        total_relative_Bz += Bz_i

opti.minimize(-total_relative_Bz)

# 4. SOLVER WITH DEBUG FALLBACK
opti.solver(
    "ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 300}
)

try:
    sol = opti.solve()
    opt_p = sol.value(p)
    opt_n = sol.value(n)
    obj_val = -sol.value(total_relative_Bz)
    print("Optimization Succeeded!\n")
except Exception:
    opt_p = opti.debug.value(p)
    opt_n = opti.debug.value(n)
    obj_val = -opti.debug.value(total_relative_Bz)
    print("Solver stopped early (retrieved latest debug values):\n")

# 5. RESULTS
print("Optimized Positions (m):\n", opt_p)
print("\nOptimized Orientations (Unit Vectors [nx, ny, nz]):\n", opt_n)

# Convert orientation vectors to tilt angle relative to +Z axis (degrees)
tilts_deg = np.degrees(np.arccos(np.clip(opt_n[:, 2], -1.0, 1.0)))
print("\nTilt Angles from +Z (deg):\n", tilts_deg)
print(f"\nRelative Field Objective Value: {obj_val:.4f}")