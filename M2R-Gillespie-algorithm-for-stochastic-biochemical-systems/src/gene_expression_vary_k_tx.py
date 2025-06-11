import numpy as np  # noqa
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gillespie import Reaction, my_gillespie


gamma_m = 0.1    # mRNA decay rate R → ∅
k_tl = 0.2    # translation rate R → R + P
gamma_p = 0.05   # protein decay rate P → ∅


def ode_system(x, t, k_tx): # noqa
    D, R, P = x # noqa
    dD = 0.0 # noqa
    dR = k_tx * D - gamma_m * R # noqa
    dP = k_tl * R - gamma_p * P # noqa
    return [dD, dR, dP]


def plot_gene_expression_vary_k_tx( # noqa
    k_tx_list, D0=1, R0=0, P0=0, # noqa
    t_max=100.0, n_ssa=5
):
    t_grid = np.linspace(0, t_max, 500)
    fig, axes = plt.subplots(
        2, len(k_tx_list),
        figsize=(4*len(k_tx_list), 8),
        sharex='col',
        sharey='row',
        constrained_layout=True
    )

    for j, k_tx in enumerate(k_tx_list):
        R1 = Reaction({0:1}, {0:1, 1:1}, rate=lambda s, k_tx=k_tx: k_tx) # noqa
        R2 = Reaction({1:1}, {1:1, 2:1}, rate=lambda s: k_tl) # noqa
        R3 = Reaction({1:1}, {}, rate=lambda s: gamma_m) # noqa
        R4 = Reaction({2:1}, {}, rate=lambda s: gamma_p) # noqa
        reactions = [R1, R2, R3, R4]

        # deterministic solution
        ode_sol = odeint(ode_system, [D0, R0, P0], t_grid, args=(k_tx,))

        axR = axes[0, j] # noqa
        axP = axes[1, j] # noqa

        # --- mRNA row ---
        axR.plot(t_grid, ode_sol[:, 1],
                 "--k", linewidth=1.5, label="Deterministic mRNA")
        for i in range(n_ssa):
            times, traj = my_gillespie(reactions, [D0, R0, P0], t_max)
            axR.step(times, traj[:, 1],
                     where="post",
                     color=f"C{i}", alpha=0.7,
                     label=f"SSA run {i+1}")
        axR.set_title(f"k_tx = {k_tx}")
        if j == 0:
            axR.set_ylabel("mRNA count")
            axR.legend(fontsize="small", loc="upper right")
        axR.tick_params(labelbottom=True)

        # --- protein row ---
        axP.plot(t_grid, ode_sol[:, 2],
                 "--k", linewidth=1.5, label="Deterministic protein")
        for i in range(n_ssa):
            times, traj = my_gillespie(reactions, [D0, R0, P0], t_max)
            axP.step(times, traj[:, 2],
                     where="post",
                     color=f"C{i}", linestyle=":",
                     alpha=0.7, label=f"SSA run {i+1}")
        if j == 0:
            axP.set_ylabel("Protein count")
            axP.legend(fontsize="small", loc="upper right")
        axP.set_xlabel("Time")
        axP.tick_params(labelbottom=True)

    plt.suptitle("Gene expression: varying transcription rate k_tx")
    plt.show()


if __name__ == "__main__":
    plot_gene_expression_vary_k_tx(
        k_tx_list=[0.1, 0.5, 1.0],
        D0=1, R0=0, P0=0,
        t_max=100.0,
        n_ssa=5
    )
