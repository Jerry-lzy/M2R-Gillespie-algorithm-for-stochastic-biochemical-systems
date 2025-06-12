import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
from scipy.integrate import odeint  # noqa
from gillespie import Reaction, my_gillespie  # noqa
from scipy.interpolate import interp1d

k1 = 0.5
k2 = 0.2
k3 = 0.1
k4 = 0.05


# R1: D → D + R
R1 = Reaction(
    reactants={0: 1},
    products={0: 1, 1: 1},
    rate=lambda s: k1
)

# R2: R → R + P
R2 = Reaction(
    reactants={1: 1},
    products={1: 1, 2: 1},
    rate=lambda s: k2
)

# R3: R → ∅
R3 = Reaction(
    reactants={1: 1},
    products={},
    rate=lambda s: k3
)

# R4: P → ∅
R4 = Reaction(
    reactants={2: 1},
    products={},
    rate=lambda s: k4
)

reactions = [R1, R2, R3, R4]


# RRE
def ode_system(x, t):  # noqa
    D, R, P = x  # noqa

    dD = 0.0 # noqa
    dR = + k1 * D # noqa
    dR -= k3 * R

    dP = + k2 * R # noqa
    dP -= k4 * P

    return [dD, dR, dP]


#  SSA vs. ODE for D, R, P (vertical layout)
def plot_gene_expression(D0, R0, P0, t_max, n_ssa=5): # noqa
    # Solve ODE
    t_grid = np.linspace(0, t_max, 500)
    ode_sol = odeint(ode_system, [D0, R0, P0], t_grid)

    ssa_results = []
    for _ in range(n_ssa):
        times, traj = my_gillespie(reactions, [D0, R0, P0], t_max)
        ssa_results.append((times, traj))


    fig, (axD, axR, axP) = plt.subplots(3, 1, figsize=(6, 12), sharex=True) # noqa
    species = ["D", "R", "P"] # noqa
    colors = plt.cm.tab10(np.arange(n_ssa))

    axD.plot(
        t_grid, ode_sol[:, 0],
        "--k", linewidth=1.5, label="Deterministic"
    )
    ssa_interp = np.zeros((n_ssa, len(t_grid), 3))
    for i, (times, traj) in enumerate(ssa_results):
        for j in range(3):  # species index: D, R, P
            f = interp1d(times, traj[:, j], kind='previous', bounds_error=False, # noqa
                         fill_value=(traj[0, j], traj[-1, j]))
            ssa_interp[i, :, j] = f(t_grid)

    ssa_mean = np.mean(ssa_interp, axis=0)  # noqa
    # Step 4: Plot
    fig, (axD, axR, axP) = plt.subplots(3, 1, figsize=(6, 12), sharex=True) # noqa
    colors = plt.cm.tab10(np.arange(n_ssa))

    # Plot D
    axD.plot(t_grid, ode_sol[:, 0], "--k", linewidth=1.5, label="Deterministic") # noqa
    for i, (times, traj) in enumerate(ssa_results):
        axD.step(times, traj[:, 0], where="post", color=colors[i], alpha=0.6, label=f"SSA run {i+1}") # noqa
    axD.plot(t_grid, ssa_mean[:, 0], color="red", linewidth=2, label="SSA mean") # noqa
    axD.set_ylabel("D count")
    axD.set_title("Gene expression: DNA D(t)")
    axD.legend(fontsize="small", loc="lower right")

    # Plot R
    axR.plot(t_grid, ode_sol[:, 1], "--k", linewidth=1.5, label="Deterministic") # noqa
    for i, (times, traj) in enumerate(ssa_results):
        axR.step(times, traj[:, 1], where="post", color=colors[i], alpha=0.6, label=f"SSA run {i+1}") # noqa
    axR.plot(t_grid, ssa_mean[:, 1], color="red", linewidth=2, label="SSA mean") # noqa
    axR.set_ylabel("R count")
    axR.set_title("Gene expression: mRNA R(t)")
    axR.legend(fontsize="small", loc="lower right")

    # Plot P
    axP.plot(t_grid, ode_sol[:, 2], "--k", linewidth=1.5, label="Deterministic") # noqa
    for i, (times, traj) in enumerate(ssa_results):
        axP.step(times, traj[:, 2], where="post", color=colors[i], alpha=0.6, label=f"SSA run {i+1}") # noqa
    axP.plot(t_grid, ssa_mean[:, 2], color="red", linewidth=2, label="SSA mean") # noqa
    axP.set_ylabel("P count")
    axP.set_title("Gene expression: Protein P(t)")
    axP.legend(fontsize="small", loc="lower right")

    axP.set_xlabel("Time")
    plt.tight_layout()
    plt.show()


D0, R0, P0 = 1, 0, 0
t_final = 100.0

plot_gene_expression(D0, R0, P0, t_final, n_ssa=5)
