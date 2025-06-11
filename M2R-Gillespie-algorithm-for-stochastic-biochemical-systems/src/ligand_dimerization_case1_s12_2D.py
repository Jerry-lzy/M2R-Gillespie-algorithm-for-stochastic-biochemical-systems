import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
from gillespie import Reaction, my_gillespie  # noqa

k1 = 0.002   # R1: S1 + S2 → 2 S1
k2 = 0.0015  # R2: 2 S1 → S1 + S2


# R1: S1 + S2 → 2 S1
R1 = Reaction(
    reactants={0: 1, 1: 1},
    products={0: 2},
    rate=lambda s: k1
)

# R2: 2 S1 → S1 + S2
R2 = Reaction(
    reactants={0: 2},
    products={0: 1, 1: 1},
    rate=lambda s: k2
)

reactions = [R1, R2]


# RRE
def ode_system(x, t): # noqa
    S1, S2 = x # noqa
    flux_R1 = k1 * S1 * S2 # noqa
    flux_R2 = k2 * (S1 * (S1 - 1) / 2) # noqa

    dS1 = + flux_R1 - flux_R2 # noqa

    dS2 = - flux_R1 + flux_R2 # noqa

    return [dS1, dS2]


# SSA trajectory
def sample_stationary_S1_S2(  # noqa
    S1_0, S2_0, t_max, burn_in, dt_sample # noqa
):
    times, traj = my_gillespie(
        reactions,
        [S1_0, S2_0],
        t_max
    )
    sample_times = np.arange(burn_in, t_max, dt_sample)

    # S1 and S2
    S1_traj = traj[:, 0]  # noqa
    S2_traj = traj[:, 1]  # noqa

    S1_samples = np.interp(sample_times, times, S1_traj)  # noqa
    S2_samples = np.interp(sample_times, times, S2_traj)  # noqa

    # Round to integer
    all_S1 = np.round(S1_samples).astype(int)  # noqa
    all_S2 = np.round(S2_samples).astype(int)  # noqa

    return all_S1, all_S2


# Plot joint stationary distribution
def plot_joint_stationary_S1_S2(S1_0, S2_0,t_max=5000.0,burn_in=2500.0,dt_sample=1.0): # noqa
    all_S1, all_S2 = sample_stationary_S1_S2( # noqa
        S1_0, S2_0,
        t_max, burn_in, dt_sample
    )
    S1_min, S1_max = np.min(all_S1), np.max(all_S1) # noqa
    S2_min, S2_max = np.min(all_S2), np.max(all_S2) # noqa

    # set bin
    x_edges = np.arange(S1_min, S1_max + 2) - 0.5
    y_edges = np.arange(S2_min, S2_max + 2) - 0.5

    # Plot 2D Graph
    H, xedges, yedges = np.histogram2d( # noqa
        all_S1, all_S2,
        bins=[x_edges, y_edges],
        density=True
    )
    # H.shape = (len(x_edges)-1, len(y_edges)-1)

    plt.figure(figsize=(6, 5))
    X, Y = np.meshgrid(xedges, yedges, indexing='xy')  # noqa
    plt.pcolormesh(
        X, Y, H.T,
        cmap="jet",
        shading="auto"
    )
    cbar = plt.colorbar()
    cbar.set_label("Probability density φ(S1,S2)")
    plt.xlabel("number of S1 molecules")
    plt.ylabel("number of S2 molecules")
    plt.title("Joint stationary distribution φ(S1,S2)")
    plt.tight_layout()
    plt.show()


S1_0, S2_0 = 100, 50
plot_joint_stationary_S1_S2(
    S1_0, S2_0,
    t_max=5000.0,
    burn_in=2500.0,
    dt_sample=1.0
)
