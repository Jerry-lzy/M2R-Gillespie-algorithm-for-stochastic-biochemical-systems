import numpy as np # noqa
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gillespie import Reaction, my_gillespie

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


# steady distribution from SSA
def sample_stationary_S1_S2(S1_0, S2_0, t_max, burn_in, dt_sample, n_runs): # noqa
    all_S1 = [] # noqa
    all_S2 = [] # noqa
    for _ in range(n_runs):
        times, traj = my_gillespie(reactions, [S1_0, S2_0], t_max)
        sample_times = np.arange(burn_in, t_max, dt_sample)

        S1_traj = traj[:, 0] # noqa
        S2_traj = traj[:, 1] # noqa
        S1_samples = np.interp(sample_times, times, S1_traj) # noqa
        S2_samples = np.interp(sample_times, times, S2_traj) # noqa

        all_S1.append(np.round(S1_samples).astype(int))
        all_S2.append(np.round(S2_samples).astype(int))

    all_S1 = np.hstack(all_S1) # noqa
    all_S2 = np.hstack(all_S2) # noqa
    return all_S1, all_S2


# 5. Plot stationary histograms
def plot_stationary_distribution_S1_S2(S1_0, S2_0, t_max,burn_in=500.0, dt_sample=1.0,n_runs=50): # noqa

    all_S1, all_S2 = sample_stationary_S1_S2( # noqa
        S1_0, S2_0, t_max, burn_in, dt_sample, n_runs
    )

    # Determine bins
    S1_min, S1_max = all_S1.min(), all_S1.max() # noqa
    S2_min, S2_max = all_S2.min(), all_S2.max() # noqa
    bins_S1 = np.arange(S1_min, S1_max + 2) - 0.5 # noqa
    bins_S2 = np.arange(S2_min, S2_max + 2) - 0.5 # noqa

    # Solve ODE
    t_grid = np.linspace(0, t_max, 1000)
    ode_sol = odeint(ode_system, [S1_0, S2_0], t_grid)
    S1_ss = ode_sol[-1, 0] # noqa
    S2_ss = ode_sol[-1, 1] # noqa

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=False, constrained_layout=True) # noqa

    # Plot for S1
    ax1.hist(all_S1, bins=bins_S1, density=True,
             alpha=0.6, edgecolor="black", linewidth=0.5)
    ax1.axvline(S1_ss, color="red", linestyle="--", linewidth=1.5,
                label=f"ODE mean ≈ {S1_ss:.1f}")
    ax1.set_xlabel("S1 molecule count")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"Stationary distribution of S1 S1_0=100") # noqa
    ax1.legend(fontsize="small")

    # Plot for S2
    ax2.hist(all_S2, bins=bins_S2, density=True,
             alpha=0.6, edgecolor="black", linewidth=0.5)
    ax2.axvline(S2_ss, color="red", linestyle="--", linewidth=1.5,
                label=f"ODE mean ≈ {S2_ss:.1f}")
    ax2.set_xlabel("S2 molecule count")
    ax2.set_ylabel("Probability")
    ax2.set_title(f"Stationary distribution of S2 S2_0=0") # noqa
    ax2.legend(fontsize="small")

    plt.tight_layout()
    plt.show()


S1_0, S2_0 = 100, 0
t_final = 2000.0

plot_stationary_distribution_S1_S2(
    S1_0, S2_0,
    t_max=t_final,
    burn_in=1000.0,
    dt_sample=1.0,
    n_runs=100
)
