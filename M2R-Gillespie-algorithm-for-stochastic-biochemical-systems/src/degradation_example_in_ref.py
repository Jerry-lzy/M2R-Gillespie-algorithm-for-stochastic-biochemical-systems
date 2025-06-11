import numpy as np  # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie


def plot_degradation_two_ks( # noqa
    k_list, X0=100, t_max=6.0, n_ssa=2 # noqa
):  # noqa
    t = np.linspace(0, t_max, 200)
    fig, axes = plt.subplots(
        1, len(k_list),
        figsize=(5 * len(k_list), 4),
        sharey=True,
        constrained_layout=True
    )
    if len(k_list) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_list):
        # analytic
        mean = X0 * np.exp(-k * t)
        std  = np.sqrt(X0 * np.exp(-k * t) * (1 - np.exp(-k * t))) # noqa
        ax.plot(t, mean, 'r-',  lw=1.5, label='Analytic mean')
        ax.plot(t, mean+ std,'r--', lw=1.0, label='Analytic ±1 std') # noqa
        ax.plot(t, mean- std,'r--', lw=1.0) # noqa

        # SSA runs
        for i in range(n_ssa):
            R = Reaction( # noqa
                reactants={0: 1},
                products={},
                rate=lambda s, k=k: k
            )
            times, traj = my_gillespie([R], [X0], t_max)
            ax.step(
                times, traj[:, 0],
                where='post',
                alpha=0.7,
                label=(f'SSA run {i+1}')
            )

        ax.set_title(f'k = {k}')
        ax.set_xlabel('time')
        if ax is axes[0]:
            ax.set_ylabel('count')
        ax.legend(fontsize='small', loc='upper right')

    fig.suptitle('Degradation S→∅')

    plt.show()


if __name__ == "__main__":
    plot_degradation_two_ks(
        k_list=[0.5, 2.0],
        X0=100,
        t_max=6.0,
        n_ssa=2
    )
