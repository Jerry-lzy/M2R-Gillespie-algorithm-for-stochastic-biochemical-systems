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


# 3. Deterministic ODE system
def ode_system(x, t): # noqa
    S1, S2 = x # noqa
    flux_R1 = k1 * S1 * S2 # noqa
    flux_R2 = k2 * (S1 * (S1 - 1) / 2) # noqa

    dS1 = + flux_R1 - flux_R2 # noqa

    dS2 = - flux_R1 + flux_R2 # noqa

    return [dS1, dS2]


# 4. Plot function
def plot_ligand_dimerization(S1_0, S2_0, t_max, n_ssa=5): # noqa
    # Solve deterministic ODE
    t_grid = np.linspace(0, t_max, 500)
    ode_sol = odeint(ode_system, [S1_0, S2_0], t_grid)

    # subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
    species = ["S1", "S2"] # noqa
    colors = plt.cm.tab10(np.arange(n_ssa))

    # Plot for S1
    # deterministic
    ax1.plot(t_grid, ode_sol[:, 0],
             "--k", linewidth=1.5, label="Deterministic")
    # SSA runs
    for i in range(n_ssa):
        times, traj = my_gillespie(reactions, [S1_0, S2_0], t_max)
        ax1.step(times, traj[:, 0],
                 where="post",
                 color=colors[i],
                 alpha=0.7,
                 label=(f"SSA run {i+1}"))
    ax1.set_ylabel("S1 count")
    ax1.set_title("Ligand-mediated dimerization: S1(t) at S1_0 = 100")
    ax1.legend(fontsize="small", loc="upper right")

    # Plot for S2
    ax2.plot(t_grid, ode_sol[:, 1],
             "--k", linewidth=1.5, label="Deterministic")
    for i in range(n_ssa):
        # simulate SSA for S2
        times, traj = my_gillespie(reactions, [S1_0, S2_0], t_max)
        ax2.step(times, traj[:, 1],
                 where="post",
                 color=colors[i],
                 alpha=0.7,
                 label=(f"SSA run {i+1}"))
    ax2.set_ylabel("S2 count")
    ax2.set_title("Ligand-mediated dimerization: S2(t)")
    ax2.legend(fontsize="small", loc="upper right")

    # Final formatting
    ax2.set_xlabel("Time")
    plt.tight_layout()
    plt.show()


S1_0 = 100 # noqa
S2_0 = 50
t_final = 100.0

plot_ligand_dimerization(S1_0, S2_0, t_final, n_ssa=3)
