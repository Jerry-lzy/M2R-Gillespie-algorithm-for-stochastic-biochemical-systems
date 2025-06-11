import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
from gillespie import Reaction, my_gillespie  # noqa


def sample_mean_S1_S2(k1, k2, # noqa
                     S1_0, S2_0, # noqa
                     t_max, burn_in, dt_sample): # noqa
    # build reactions
    R1 = Reaction( # noqa
        reactants={0: 1, 1: 1},
        products={0: 2},
        rate=lambda s, k1=k1: k1
    )
    R2 = Reaction( # noqa
        reactants={0: 2},
        products={0: 1, 1: 1},
        rate=lambda s, k2=k2: k2
    )
    reactions = [R1, R2]

    # run SSA
    times, traj = my_gillespie(reactions, [S1_0, S2_0], t_max)

    # sample in steady‐state window
    sample_times = np.arange(burn_in, t_max, dt_sample)
    S1_traj = traj[:, 0] # noqa
    S2_traj = traj[:, 1] # noqa
    S1_samples = np.interp(sample_times, times, S1_traj) # noqa
    S2_samples = np.interp(sample_times, times, S2_traj) # noqa

    return S1_samples.mean(), S2_samples.mean()


def plot_ligand_heatmap(k1_vals, k2_vals, # noqa
                        S1_0=100, S2_0=50, # noqa
                        t_max=500.0,
                        burn_in=250.0,
                        dt_sample=5.0,
                        n_runs=5): # noqa
    M, N = len(k1_vals), len(k2_vals) # noqa
    mean_S1 = np.zeros((M, N)) # noqa
    mean_S2 = np.zeros((M, N)) # noqa

    for i, k1 in enumerate(k1_vals):
        for j, k2 in enumerate(k2_vals):
            s1, s2 = 0.0, 0.0
            for _ in range(n_runs):
                m1, m2 = sample_mean_S1_S2(
                    k1, k2,
                    S1_0, S2_0,
                    t_max, burn_in, dt_sample
                )
                s1 += m1
                s2 += m2
            mean_S1[i, j] = s1 / n_runs
            mean_S2[i, j] = s2 / n_runs

    # create meshgrid for plotting
    K2, K1 = np.meshgrid(k2_vals, k1_vals) # noqa

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.pcolormesh(
        K2, K1, mean_S1,
        shading='auto', cmap='viridis'
    )
    fig.colorbar(im1, ax=ax1, label='Mean S1')
    ax1.set_xlabel('k2 (dissociation rate)')
    ax1.set_ylabel('k1 (association rate)')
    ax1.set_title('Steady-state mean S1')

    im2 = ax2.pcolormesh(
        K2, K1, mean_S2,
        shading='auto', cmap='magma'
    )
    fig.colorbar(im2, ax=ax2, label='Mean S2')
    ax2.set_xlabel('k2 (dissociation rate)')
    ax2.set_ylabel('k1 (association rate)')
    ax2.set_title('Steady-state mean S2')

    fig.suptitle('Ligand_heatmap')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    k1_vals = np.linspace(0.001, 0.01, 10)
    k2_vals = np.linspace(0.001, 0.02, 10)

    plot_ligand_heatmap(
        k1_vals, k2_vals,
        S1_0=100, S2_0=50,
        t_max=500.0,
        burn_in=250.0,
        dt_sample=5.0,
        n_runs=5
    )
