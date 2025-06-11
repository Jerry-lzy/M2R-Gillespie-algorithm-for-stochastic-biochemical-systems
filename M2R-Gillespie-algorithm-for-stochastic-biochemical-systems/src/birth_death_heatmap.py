import numpy as np  # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie


def plot_bd_heatmap(k1_vals, k2_vals, X0, # noqa
                    t_max, burn_in, dt_sample, n_runs=50):
    M, N = len(k1_vals), len(k2_vals) # noqa
    means = np.zeros((M, N))

    for i, k1_ in enumerate(k1_vals):
        for j, k2_ in enumerate(k2_vals):
            birth = Reaction({}, {0: 1}, rate=lambda s, k2_=k2_: k2_)
            death = Reaction({0: 1}, {},   rate=lambda s, k1_=k1_: k1_)
            reactions = [birth, death]

            times, traj = my_gillespie(reactions, [X0], t_max)
            sample_times = np.arange(burn_in, t_max, dt_sample)
            vals = np.interp(sample_times, times, traj[:, 0])
            means[i, j] = np.mean(vals)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        k2_vals,
        k1_vals,
        means,
        shading='auto',
        cmap='viridis'
    )
    cbar = plt.colorbar()
    cbar.set_label('Mean molecules count')
    plt.xlabel('k2 (birth rate)')
    plt.ylabel('k1 (death rate)')
    plt.title(f'Steady-state mean (X0={X0})')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__": # noqa
    k1, k2, X0 = 0.1, 1.0, 0

    k1_vals = np.linspace(0.05, 0.2, 10)
    k2_vals = np.linspace(0.5, 2.0, 10)
    plot_bd_heatmap(k1_vals, k2_vals, X0,
                    t_max=2000, burn_in=1000, dt_sample=1.0, n_runs=50)
