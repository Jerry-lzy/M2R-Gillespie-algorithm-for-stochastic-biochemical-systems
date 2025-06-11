import numpy as np  # noqa
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gillespie import Reaction, my_gillespie


k_tx = 0.5    # transcription rate D → D + R
k_tl = 0.2    # translation rate R → R + P
gamma_p = 0.05  # protein degradation rate P → ∅


def ode_system(x, t, gamma_m): # noqa
    D, R, P = x # noqa
    dD = 0.0 # noqa
    dR = k_tx * D - gamma_m * R # noqa
    dP = k_tl * R - gamma_p * P # noqa
    return [dD, dR, dP]


def plot_gene_expression_vary_gamma_m( # noqa
    gamma_m_list, D0=1, R0=0, P0=0, # noqa
    t_max=100.0, n_ssa=5
):
    n = len(gamma_m_list)
    t_grid = np.linspace(0, t_max, 500)
    fig, axes = plt.subplots(
        2, n,
        figsize=(4*n, 8),
        sharex='col',
        sharey='row',
        constrained_layout=True
    )

    for j, gamma_m in enumerate(gamma_m_list):
        R1 = Reaction({0:1}, {0:1,1:1}, rate=lambda s: k_tx)  # noqa
        R2 = Reaction({1:1}, {1:1,2:1}, rate=lambda s: k_tl) # noqa
        R3 = Reaction({1:1}, {},            rate=lambda s, gm=gamma_m: gm)  # noqa
        R4 = Reaction({2:1}, {},            rate=lambda s: gamma_p)   # noqa
        reactions = [R1, R2, R3, R4]

        # deterministic solution
        ode_sol = odeint(ode_system, [D0, R0, P0], t_grid, args=(gamma_m,))

        axR = axes[0, j] # noqa
        axP = axes[1, j] # noqa

        axR.plot(t_grid, ode_sol[:, 1], '--k', lw=2, label='Deterministic')
        for i in range(n_ssa):
            times, traj = my_gillespie(reactions, [D0, R0, P0], t_max)
            axR.step(times, traj[:, 1], where='post',
                     color=f'C{i}', alpha=0.7,
                     label=f'SSA {i+1}')
        axR.set_title(f'γₘ={gamma_m}')
        if j == 0:
            axR.set_ylabel('mRNA count')
            axR.legend(fontsize='small', loc='upper right')

        axP.plot(t_grid, ode_sol[:, 2], '--k', lw=2, label='Deterministic')
        for i in range(n_ssa):
            times, traj = my_gillespie(reactions, [D0, R0, P0], t_max)
            axP.step(times, traj[:, 2], where='post',
                     color=f'C{i}', alpha=0.7,
                     label=f'SSA {i+1}')
        if j == 0:
            axP.set_ylabel('Protein count')
        axP.set_xlabel('Time')
        if j == 0:
            axP.legend(fontsize='small', loc='upper right')

    plt.suptitle('Gene expression (varying protein degradation rate γₘ)')
    plt.show()


if __name__ == '__main__':
    plot_gene_expression_vary_gamma_m(
        gamma_m_list=[0.05, 0.1, 0.2],
        D0=1, R0=0, P0=0,
        t_max=100.0,
        n_ssa=5
    )
