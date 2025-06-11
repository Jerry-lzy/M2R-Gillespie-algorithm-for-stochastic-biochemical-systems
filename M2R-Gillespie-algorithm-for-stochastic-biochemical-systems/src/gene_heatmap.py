import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie

k_tx = 0.5
gamma_m = 0.1   # mRNA decay
k_tl = 0.2
gamma_p = 0.05  # protein decay

# reactions
gene_tx = Reaction(reactants={},    products={0: 1}, rate=lambda state: k_tx) # noqa
mRNA_deg = Reaction(reactants={0: 1}, products={}, rate=lambda state: gamma_m)  # noqa
translation = Reaction(reactants={0: 1}, products={0: 1, 1: 1}, rate=lambda state: k_tl) # noqa
protein_deg = Reaction(reactants={1: 1}, products={}, rate=lambda state: gamma_p) # noqa
initial = [0, 0]
t_max = 200


def steady_state_mean(k_tx, k_tl, t_max, burn_in, dt_sample):  # noqa 
    reactions = [gene_tx, mRNA_deg, translation, protein_deg]

    times, traj = my_gillespie(reactions, [1, 0], t_max)
    sample_times = np.arange(burn_in, t_max, dt_sample)

    m_samples = np.interp(sample_times, times, traj[:, 0])
    p_samples = np.interp(sample_times, times, traj[:, 1])

    return m_samples.mean(), p_samples.mean()


if __name__ == "__main__": # noqa 
    k_tx_vals = np.linspace(0.1, 1.0, 20)
    k_tl_vals = np.linspace(0.05, 0.5, 20)

    M = len(k_tx_vals)
    N = len(k_tl_vals)
    mean_mRNA = np.zeros((M, N))  # noqa 
    mean_prot = np.zeros((M, N))

    for i, k_tx in enumerate(k_tx_vals):
        for j, k_tl in enumerate(k_tl_vals):
            mm, mp = steady_state_mean(
                k_tx, k_tl,
                t_max=200.0,
                burn_in=100.0,
                dt_sample=1.0
            )
            mean_mRNA[i, j] = mm
            mean_prot[i, j] = mp

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # mRNA
    im1 = ax1.pcolormesh(
        k_tl_vals, k_tx_vals, mean_mRNA,
        shading="auto", cmap="viridis"
    )
    fig.colorbar(im1, ax=ax1, label="mean mRNA")
    ax1.set_xlabel("translation rate")
    ax1.set_ylabel("transcription rate")
    ax1.set_title("Steady-state mean mRNA")

    # Protein
    im2 = ax2.pcolormesh(
        k_tl_vals, k_tx_vals, mean_prot,
        shading="auto", cmap="magma"
    )
    fig.colorbar(im2, ax=ax2, label="mean protein")
    ax2.set_xlabel("translation rate")
    ax2.set_ylabel("transcription rate")
    ax2.set_title("Steady-state mean protein")

    plt.tight_layout()
    plt.show()
