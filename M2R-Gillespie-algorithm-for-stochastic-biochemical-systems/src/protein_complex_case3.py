import numpy as np  # noqa
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gillespie import Reaction, my_gillespie

# This code is quite similar to protein_complex_second.py
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


# To get final dA, dB, dC, dD
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

def plot_complex_formation_degradation(A0, B0, C0, D0, t_max, n_ssa=5): # noqa
    t_grid = np.linspace(0, t_max, 500)
    ode_sol = odeint(ode_system, [A0, B0, C0, D0], t_grid)

    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=True)
    species = ["A", "B", "C", "D"]
    colors = plt.cm.tab10(np.arange(n_ssa))

    for idx, ax in enumerate(axes):
        # Plot rre equation
        ax.plot(t_grid, ode_sol[:, idx], "--k", linewidth=1.5, label="Deterministic") # noqa

        # SSA trajectories
        for i in range(n_ssa):
            times, traj = my_gillespie(
                reactions, [A0, B0, C0, D0], t_max
            )
            ax.step(
                times,
                traj[:, idx],
                where="post",
                color=colors[i],
                alpha=0.7,
                label=(f"SSA run {i+1}" if idx == 0 else None)
            )

        ax.set_ylabel(f"{species[idx]} count")
        if idx == 0:
            ax.legend(fontsize="small", loc="upper right")

        if species[idx] in ["C", "D"]:
            ax.text(
                0.5, 0.9,
                "That is what we are not interested in",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
                alpha=0.7
            )

    axes[-1].set_xlabel("Time")
    plt.suptitle("Complex formation + degradation at A0 = 50, B0 = 0")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


A0 = 50
B0 = 0
C0 = 0
D0 = 0
t_final = 100.0
plot_complex_formation_degradation(A0, B0, C0, D0, t_final, n_ssa=5)
