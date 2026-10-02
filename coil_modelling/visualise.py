import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 1. PARAMETERS & GEOMETRY
# ==============================================================================
spacing = 0.026  # Grid spacing (2.6 cm)
grid_coords = np.array([-1.5, -0.5, 0.5, 1.5]) * spacing  # [-3.9, -1.3, 1.3, 3.9] cm
positions = np.array([[x, y, 0.0] for x in grid_coords for y in grid_coords])

# Cluster mapping (4x4 grid split into 4 2x2 clusters)
cluster_map = np.array(
    [
        (0 if i >= 2 else 1) if j >= 2 else (3 if i >= 2 else 2)
        for i in range(4)
        for j in range(4)
    ]
)
cluster_phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

I0 = 1.25  # Current (A)
K_dipole = 5.629e-5  # Dipole constant T*m^3/A
sigma_rot = np.radians(0.5)  # Pose rotation noise (rad)
sigma_trans = 0.0002  # Pose translation noise (m)


# ==============================================================================
# 2. COIL NORMAL GENERATOR
# ==============================================================================
def get_coil_normals(topology, theta_deg=0.0, z_focal=0.045):
    normals = []
    cluster_centers = np.zeros((4, 2))
    for c_id in range(4):
        cluster_centers[c_id] = np.mean(
            positions[cluster_map == c_id, :2], axis=0
        )

    for k, p in enumerate(positions):
        c_id = cluster_map[k]
        if topology == "baseline":
            normals.append([0.0, 0.0, 1.0])
        elif topology == "cluster_tilt":
            c_ctr = cluster_centers[c_id]
            r_dir = -c_ctr / np.linalg.norm(c_ctr)
            rad = np.radians(theta_deg)
            nx = r_dir[0] * np.sin(rad)
            ny = r_dir[1] * np.sin(rad)
            nz = np.cos(rad)
            normals.append([nx, ny, nz])
        elif topology == "focal_convergent":
            v = np.array([-p[0], -p[1], z_focal - p[2]])
            normals.append(v / np.linalg.norm(v))

    return np.array(normals)


# ==============================================================================
# 3. EXACT LOSS FUNCTIONS
# ==============================================================================
def compute_field_components(normals, r_eval):
    """Computes quadrature field components u and v at point r_eval."""
    u, v = np.zeros(3), np.zeros(3)
    for k in range(16):
        p, n = positions[k], normals[k]
        psi = cluster_phases[cluster_map[k]]
        rk = r_eval - p
        dk = np.linalg.norm(rk)
        bk = K_dipole * (3 * np.dot(rk, n) * rk / dk**5 - n / dk**3)
        u += I0 * bk * np.cos(psi)
        v += -I0 * bk * np.sin(psi)
    return u, v


def loss_directional_noise(normals, N_mc=60):
    """L_noise: Expectation of angular deviation of rotation normal under SE(3) pose noise."""
    roi_coords = np.linspace(-0.02, 0.02, 3)
    z_eval = 0.02
    abs_noise_list = []

    for rx in roi_coords:
        for ry in roi_coords:
            r = np.array([rx, ry, z_eval])
            u_nom, v_nom = compute_field_components(normals, r)
            w_nom = np.cross(u_nom, v_nom)
            n_rot_nom = w_nom / np.linalg.norm(w_nom)

            n_rot_trials = []
            for _ in range(N_mc):
                u_pert, v_pert = np.zeros(3), np.zeros(3)
                for k in range(16):
                    p_nom, n_nom = positions[k], normals[k]
                    psi = cluster_phases[cluster_map[k]]

                    dp = np.random.normal(0, sigma_trans, 3)
                    dw = np.random.normal(0, sigma_rot, 3)
                    p_p = p_nom + dp
                    n_p = n_nom + np.cross(dw, n_nom)
                    n_p /= np.linalg.norm(n_p)

                    rk = r - p_p
                    dk = np.linalg.norm(rk)
                    bk = K_dipole * (
                        3 * np.dot(rk, n_p) * rk / dk**5 - n_p / dk**3
                    )
                    u_pert += I0 * bk * np.cos(psi)
                    v_pert += -I0 * bk * np.sin(psi)

                w_pert = np.cross(u_pert, v_pert)
                n_rot_trials.append(w_pert / np.linalg.norm(w_pert))

            delta_n = np.array(n_rot_trials) - n_rot_nom
            abs_noise_list.append(np.mean(np.sum(np.abs(delta_n), axis=1)))

    return np.mean(abs_noise_list)


def loss_field_magnitude(normals):
    """L_mag: Inverse magnetic field strength in target region (penalizes attenuation)."""
    roi_coords = np.linspace(-0.02, 0.02, 3)
    z_eval = 0.02
    mag_list = []

    for rx in roi_coords:
        for ry in roi_coords:
            r = np.array([rx, ry, z_eval])
            u, v = compute_field_components(normals, r)
            b_rms = np.sqrt(0.5 * (np.linalg.norm(u) ** 2 + np.linalg.norm(v) ** 2))
            mag_list.append(b_rms)

    mean_b = np.mean(mag_list)
    return 1.0 / (mean_b * 1e4 + 1e-6)  # Scaled inverse Gauss


def loss_ellipticity_ortho(normals):
    """L_ortho: Deviation from circular polarization in the rotating field plane."""
    roi_coords = np.linspace(-0.02, 0.02, 3)
    z_eval = 0.02
    ellip_list = []

    for rx in roi_coords:
        for ry in roi_coords:
            r = np.array([rx, ry, z_eval])
            u, v = compute_field_components(normals, r)
            norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
            diff = np.abs(norm_u - norm_v) / (norm_u + norm_v + 1e-9)
            dot_ortho = np.abs(np.dot(u, v)) / (norm_u * norm_v + 1e-9)
            ellip_list.append(diff + dot_ortho)

    return np.mean(ellip_list)


def compute_total_loss(normals, w1=1.0, w2=0.15, w3=0.5, N_mc=60):
    """L_total = w1 * L_noise + w2 * L_mag + w3 * L_ortho."""
    l_noise = loss_directional_noise(normals, N_mc=N_mc)
    l_mag = loss_field_magnitude(normals)
    l_ortho = loss_ellipticity_ortho(normals)

    l_total = w1 * l_noise + w2 * l_mag + w3 * l_ortho
    return l_total, l_noise, l_mag, l_ortho


# ==============================================================================
# 4. LOSS LANDSCAPE GENERATION (PARAMETER SWEEP)
# ==============================================================================
# Sweep grid over tilt angle theta (0 to 20 deg) and focal height z_focal (2.5 to 8.5 cm)
theta_vals = np.linspace(0.0, 20.0, 10)
z_focal_vals = np.linspace(0.025, 0.085, 10)

L_noise_grid = np.zeros((len(z_focal_vals), len(theta_vals)))
L_mag_grid = np.zeros((len(z_focal_vals), len(theta_vals)))
L_ortho_grid = np.zeros((len(z_focal_vals), len(theta_vals)))
L_total_grid = np.zeros((len(z_focal_vals), len(theta_vals)))

print("Computing loss landscapes across design parameter space...")
for i, z_f in enumerate(z_focal_vals):
    for j, th in enumerate(theta_vals):
        # Combined hybrid topology parameterization
        normals_focal = get_coil_normals("focal_convergent", z_focal=z_f)
        normals_cluster = get_coil_normals("cluster_tilt", theta_deg=th)
        normals = 0.5 * (normals_focal + normals_cluster)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        l_tot, l_n, l_m, l_o = compute_total_loss(normals, N_mc=40)
        L_noise_grid[i, j] = l_n
        L_mag_grid[i, j] = l_m
        L_ortho_grid[i, j] = l_o
        L_total_grid[i, j] = l_tot

# Find optimal design from parameter sweep
min_idx = np.unravel_index(np.argmin(L_total_grid), L_total_grid.shape)
opt_z_focal = z_focal_vals[min_idx[0]]
opt_theta = theta_vals[min_idx[1]]
opt_normals = 0.5 * (
    get_coil_normals("focal_convergent", z_focal=opt_z_focal)
    + get_coil_normals("cluster_tilt", theta_deg=opt_theta)
)
opt_normals /= np.linalg.norm(opt_normals, axis=1, keepdims=True)

print(
    f"\nOptimal Design Found: theta = {opt_theta:.2f}°, z_focal = {opt_z_focal*100:.2f} cm"
)


# ==============================================================================
# 5. VISUALIZE LOSS LANDSCAPES
# ==============================================================================
TH, ZF = np.meshgrid(theta_vals, z_focal_vals * 100)  # Convert z_focal to cm

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

# 1. Noise Loss Landscape
c1 = axs[0, 0].contourf(TH, ZF, L_noise_grid, 20, cmap="viridis")
fig.colorbar(c1, ax=axs[0, 0])
axs[0, 0].set_title(
    r"Directional Noise Loss $\mathcal{L}_{\mathrm{noise}}$", fontweight="bold"
)
axs[0, 0].set_xlabel("Cluster Tilt Angle $\\theta$ [deg]")
axs[0, 0].set_ylabel("Focal Height $z_{\\mathrm{focal}}$ [cm]")

# 2. Field Magnitude Loss Landscape
c2 = axs[0, 1].contourf(TH, ZF, L_mag_grid, 20, cmap="plasma")
fig.colorbar(c2, ax=axs[0, 1])
axs[0, 1].set_title(
    r"Field Attenuation Loss $\mathcal{L}_{\mathrm{mag}}$", fontweight="bold"
)
axs[0, 1].set_xlabel("Cluster Tilt Angle $\\theta$ [deg]")
axs[0, 1].set_ylabel("Focal Height $z_{\\mathrm{focal}}$ [cm]")

# 3. Ellipticity Loss Landscape
c3 = axs[1, 0].contourf(TH, ZF, L_ortho_grid, 20, cmap="magma")
fig.colorbar(c3, ax=axs[1, 0])
axs[1, 0].set_title(
    r"Field Polarization Loss $\mathcal{L}_{\mathrm{ortho}}$",
    fontweight="bold",
)
axs[1, 0].set_xlabel("Cluster Tilt Angle $\\theta$ [deg]")
axs[1, 0].set_ylabel("Focal Height $z_{\\mathrm{focal}}$ [cm]")

# 4. Total Combined Loss Landscape
c4 = axs[1, 1].contourf(TH, ZF, L_total_grid, 25, cmap="coolwarm")
fig.colorbar(c4, ax=axs[1, 1])
axs[1, 1].scatter(
    [opt_theta],
    [opt_z_focal * 100],
    color="yellow",
    s=120,
    edgecolors="black",
    marker="*",
    label="Optimal Minimum",
)
axs[1, 1].set_title(
    r"Total Cost Function $\mathcal{L}_{\mathrm{total}}$", fontweight="bold"
)
axs[1, 1].set_xlabel("Cluster Tilt Angle $\\theta$ [deg]")
axs[1, 1].set_ylabel("Focal Height $z_{\\mathrm{focal}}$ [cm]")
axs[1, 1].legend(loc="upper right")

plt.tight_layout()
plt.show()


# ==============================================================================
# 6. BENCHMARKING OPTIMAL DESIGN VS BASELINE & CLUSTER TILTS
# ==============================================================================
benchmarks = {
    "Baseline (0° Tilt)": get_coil_normals("baseline"),
    "Cluster Tilt (5°)": get_coil_normals("cluster_tilt", theta_deg=5.0),
    "Cluster Tilt (10°)": get_coil_normals("cluster_tilt", theta_deg=10.0),
    f"Optimal ({opt_theta:.1f}°, {opt_z_focal*100:.1f}cm)": opt_normals,
}

results = {}
print("\n" + "=" * 70)
print(f"{'Topology':<30} | {'L_noise':<10} | {'L_mag':<10} | {'L_ortho':<10} | {'L_total':<10}")
print("=" * 70)

for name, norm_set in benchmarks.items():
    l_tot, l_n, l_m, l_o = compute_total_loss(norm_set, N_mc=150)
    results[name] = (l_n, l_m, l_o, l_tot)
    print(
        f"{name:<30} | {l_n:<10.4f} | {l_m:<10.4f} | {l_o:<10.4f} | {l_tot:<10.4f}"
    )
print("=" * 70)


# ==============================================================================
# 7. BENCHMARK COMPARISON BAR CHART
# ==============================================================================
labels = list(results.keys())
l_noise_vals = [results[k][0] for k in labels]
l_mag_vals = [results[k][1] for k in labels]
l_ortho_vals = [results[k][2] for k in labels]
l_tot_vals = [results[k][3] for k in labels]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - 1.5 * width, l_noise_vals, width, label=r"Noise ($\mathcal{L}_{noise}$)", color="crimson")
ax.bar(x - 0.5 * width, l_mag_vals, width, label=r"Field Atten. ($\mathcal{L}_{mag}$)", color="darkorange")
ax.bar(x + 0.5 * width, l_ortho_vals, width, label=r"Ellipticity ($\mathcal{L}_{ortho}$)", color="mediumpurple")
ax.bar(x + 1.5 * width, l_tot_vals, width, label=r"Total ($\mathcal{L}_{total}$)", color="teal")

ax.set_ylabel("Loss Score (Lower is better)")
ax.set_title("Benchmarking Topology Performance Metrics", fontweight="bold", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend()
ax.grid(True, axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()