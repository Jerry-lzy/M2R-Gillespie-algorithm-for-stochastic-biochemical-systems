import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
from scipy.integrate import odeint  # noqa
from gillespie import Reaction, my_gillespie  # noqa


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

    fig, (axD, axR, axP) = plt.subplots(3, 1, figsize=(6, 12), sharex=True) # noqa
    species = ["D", "R", "P"] # noqa
    colors = plt.cm.tab10(np.arange(n_ssa))

    # Plot D(t)
    axD.plot(
        t_grid, ode_sol[:, 0],
        "--k", linewidth=1.5, label="Deterministic"
    )
    for i in range(n_ssa):
        times, traj = my_gillespie(
            reactions, [D0, R0, P0], t_max
        )
        axD.step(
            times, traj[:, 0],
            where="post",
            color=colors[i],
            alpha=0.7,
            label=f"SSA run {i+1}"
        )
    axD.set_ylabel("D count")
    axD.set_title("Gene expression: D(t)")
    axD.legend(fontsize="small", loc="upper right")

    # Plot R(t)
    axR.plot(
        t_grid, ode_sol[:, 1],
        "--k", linewidth=1.5, label="Deterministic"
    )
    for i in range(n_ssa):
        times, traj = my_gillespie(
            reactions, [D0, R0, P0], t_max
        )
        axR.step(
            times, traj[:, 1],
            where="post",
            color=colors[i],
            alpha=0.7,
            label=f"SSA run {i+1}"
        )
    axR.set_ylabel("R count")
    axR.set_title("Gene expression: R(t)")
    axR.legend(fontsize="small", loc="upper right")

    # Plot P(t)
    axP.plot(
        t_grid, ode_sol[:, 2],
        "--k", linewidth=1.5, label="Deterministic"
    )
    for i in range(n_ssa):
        times, traj = my_gillespie(
            reactions, [D0, R0, P0], t_max
        )
        axP.step(
            times, traj[:, 2],
            where="post",
            color=colors[i],
            alpha=0.7,
            label=f"SSA run {i+1}"
        )
    axP.set_ylabel("P count")
    axP.set_title("Gene expression: P(t)")
    axP.legend(fontsize="small", loc="upper right")

    axP.set_xlabel("Time")
    plt.tight_layout()
    plt.show()


D0, R0, P0 = 1, 0, 0
t_final = 100.0

plot_gene_expression(D0, R0, P0, t_final, n_ssa=5)
