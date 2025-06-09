import numpy as np # noqa
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gillespie import Reaction, my_gillespie

k1 = 0.001  # A + A → C
k2 = 0.01  # A + B → D
k3 = 1.2   # ∅ → A
k4 = 1.0   # ∅ → B

# A + A → C
R1 = Reaction(
    reactants={0: 2},
    products={2: 1},
    rate=lambda s: k1
)

# A + B → D
R2 = Reaction(
    reactants={0: 1, 1: 1},
    products={3: 1},
    rate=lambda s: k2
)

# ∅ → A
R3 = Reaction(
    reactants={},
    products={0: 1},
    rate=lambda s: k3
)

# ∅ → B
R4 = Reaction(
    reactants={},
    products={1: 1},
    rate=lambda s: k4
)

reactions = [R1, R2, R3, R4]


# Deterministic ODE system
def ode_system(x, t): # noqa
    A, B, C, D = x # noqa
    dA = k3 # noqa
    dA -= 2 * k1 * (A * (A - 1) / 2)
    dA -= k2 * A * B

    dB = k4 # noqa
    dB -= k2 * A * B

    dC = k1 * (A * (A - 1) / 2) # noqa

    dD = k2 * A * B # noqa
    return [dA, dB, dC, dD]


# Function of stationary data
def sample_stationary_A_B(A0, B0, C0, D0, t_max, burn_in, dt_sample, n_runs): # noqa
    all_A, all_B = [], [] # noqa

    for run in range(n_runs):
        times, traj = my_gillespie(reactions, [A0, B0, C0, D0], t_max)

        sample_times = np.arange(burn_in, t_max, dt_sample)

        A_traj = traj[:, 0] # noqa
        B_traj = traj[:, 1] # noqa
        A_samples = np.interp(sample_times, times, A_traj) # noqa
        B_samples = np.interp(sample_times, times, B_traj) # noqa

        # Round to integer
        all_A.append(np.round(A_samples).astype(int))
        all_B.append(np.round(B_samples).astype(int))

    all_A = np.hstack(all_A) # noqa
    all_B = np.hstack(all_B) # noqa
    return all_A, all_B


# Plot stationary histograms
def plot_stationary_distribution(A0, B0, C0, D0, t_max, # noqa
                                 burn_in=500.0, dt_sample=1.0,
                                 n_runs=50):
    all_A, all_B = sample_stationary_A_B( # noqa
        A0, B0, C0, D0, t_max, burn_in, dt_sample, n_runs
    )

    # bins for A and B
    A_min, A_max = all_A.min(), all_A.max() # noqa
    B_min, B_max = all_B.min(), all_B.max() # noqa
    bins_A = np.arange(A_min, A_max + 2) - 0.5 # noqa
    bins_B = np.arange(B_min, B_max + 2) - 0.5 # noqa

    # Solve ODE
    t_grid = np.linspace(0, t_max, 1000)
    ode_sol = odeint(ode_system, [A0, B0, C0, D0], t_grid)
    A_ss = ode_sol[-1, 0] # noqa
    B_ss = ode_sol[-1, 1] # noqa

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(6, 8), sharex=False, constrained_layout=True) # noqa

    # Plot A
    axA.hist(all_A, bins=bins_A, density=True,
             alpha=0.6, edgecolor="black", linewidth=0.5)
    axA.axvline(A_ss, color="red", linestyle="--", linewidth=1.5,
                label=f"Deterministic ≈ {A_ss:.1f}")
    axA.set_xlabel("A molecule count")
    axA.set_ylabel("Probability")
    axA.set_title(f"Stationary distribution of A, A0 = 0") # noqa
    axA.legend(fontsize="small")

    # Plot B
    axB.hist(all_B, bins=bins_B, density=True,
             alpha=0.6, edgecolor="black", linewidth=0.5)
    axB.axvline(B_ss, color="red", linestyle="--", linewidth=1.5,
                label=f"Deterministic ≈ {B_ss:.1f}")
    axB.set_xlabel("B molecule count")
    axB.set_ylabel("Probability")
    axB.set_title(f"Stationary distribution of B B0 = 0") # noqa
    axB.legend(fontsize="small")

    plt.tight_layout()
    plt.show()


A0, B0, C0, D0 = 0, 0, 0, 0 # noqa
t_final = 2000.0

plot_stationary_distribution(
    A0, B0, C0, D0,
    t_max=t_final,
    burn_in=1000.0,
    dt_sample=1.0,
    n_runs=100
)
