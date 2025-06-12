import numpy as np  # noqa
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from gillespie import Reaction, my_gillespie

gamma_m = 0.1
k_tl = 0.2
gamma_p = 0.05

def ode_system(x, t, k_tx):
    D, R, P = x
    dD = 0.0
    dR = k_tx * D - gamma_m * R
    dP = k_tl * R - gamma_p * P
    return [dD, dR, dP]

def plot_gene_expression_vary_k_tx(k_tx_list, D0=1, R0=0, P0=0, t_max=100.0, n_ssa=5):
    t_grid = np.linspace(0, t_max, 500)
    fig, axes = plt.subplots(
        2, len(k_tx_list),
        figsize=(4 * len(k_tx_list), 8),
        sharex='col',
        sharey='row',
        constrained_layout=True
    )

    for j, k_tx in enumerate(k_tx_list):
        # Define reactions
        R1 = Reaction({0: 1}, {0: 1, 1: 1}, rate=lambda s, k_tx=k_tx: k_tx)
        R2 = Reaction({1: 1}, {1: 1, 2: 1}, rate=lambda s: k_tl)
        R3 = Reaction({1: 1}, {}, rate=lambda s: gamma_m)
        R4 = Reaction({2: 1}, {}, rate=lambda s: gamma_p)
        reactions = [R1, R2, R3, R4]

        # ODE solution
        ode_sol = odeint(ode_system, [D0, R0, P0], t_grid, args=(k_tx,))

        # Run SSA simulations
        ssa_results = []
        for _ in range(n_ssa):
            times, traj = my_gillespie(reactions, [D0, R0, P0], t_max)
            ssa_results.append((times, traj))

        # Interpolate SSA to common grid
        ssa_interp = np.zeros((n_ssa, len(t_grid), 3))
        for i, (times, traj) in enumerate(ssa_results):
            for s in range(3):
                f = interp1d(times, traj[:, s], kind='previous', bounds_error=False,
                             fill_value=(traj[0, s], traj[-1, s]))
                ssa_interp[i, :, s] = f(t_grid)
        ssa_mean = np.mean(ssa_interp, axis=0)

        # Plot mRNA
        axR = axes[0, j]
        axR.plot(t_grid, ode_sol[:, 1], '--k', linewidth=1.5, label="Deterministic")
        for i, (times, traj) in enumerate(ssa_results):
            axR.step(times, traj[:, 1], where='post', color=f'C{i}', alpha=0.7,
                     label=f'SSA run {i+1}' if j == 0 else None)
        axR.plot(t_grid, ssa_mean[:, 1], color='red', linewidth=2,
                 label='SSA mean' if j == 0 else None)
        axR.set_title(f'$k_{{tx}} = {k_tx}$')
        if j == 0:
            axR.set_ylabel('mRNA count')
            axR.legend(fontsize='small', loc='upper right')
        axR.tick_params(labelbottom=True)

        # Plot protein
        axP = axes[1, j]
        axP.plot(t_grid, ode_sol[:, 2], '--k', linewidth=1.5, label="Deterministic")
        for i, (times, traj) in enumerate(ssa_results):
            axP.step(times, traj[:, 2], where='post', color=f'C{i}', alpha=0.7,
                     label=f'SSA run {i+1}' if j == 0 else None)
        axP.plot(t_grid, ssa_mean[:, 2], color='red', linewidth=2,
                 label='SSA mean' if j == 0 else None)
        if j == 0:
            axP.set_ylabel('Protein count')
            axP.legend(fontsize='small', loc='upper right')
        axP.set_xlabel('Time')
        axP.tick_params(labelbottom=True)

    plt.suptitle("Gene expression: varying transcription rate $k_{tx}$")
    plt.show()


if __name__ == "__main__":
    plot_gene_expression_vary_k_tx(
        k_tx_list=[0.1, 0.5, 1.0],
        D0=1, R0=0, P0=0,
        t_max=100.0,
        n_ssa=5
    )
