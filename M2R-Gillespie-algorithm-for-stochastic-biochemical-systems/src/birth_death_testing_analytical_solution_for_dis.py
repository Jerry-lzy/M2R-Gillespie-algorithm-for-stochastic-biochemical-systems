import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie
from scipy.special import comb, factorial

# analytic solution of pn
def analytic_birth_death_pn(n_array, t, k1, k2, n0): # noqa
    lam = (k2 / k1) * (1 - np.exp(-k1 * t))
    pref = np.exp(-lam)
    pn = np.zeros_like(n_array, dtype=float)
    for idx, n in enumerate(n_array):
        total = 0.0
        for k in range(min(n0, n) + 1):
            binom = comb(n0, k) * (1 - np.exp(-k1 * t))**(n0 - k) * np.exp(-k1 * k * t) # noqa
            pois = lam**(n - k) / factorial(n - k)
            total += binom * pois
        pn[idx] = pref * total
    return pn

def plot_birth_death_histogram_dense(k1, k2, X0, t_max, # noqa
                                     n_runs=50,
                                     burn_in=50000.0,
                                     dt_sample=0.5,
                                     bin_width=0.2):
    birth = Reaction({}, {0: 1}, rate=lambda s: k2)
    death = Reaction({0: 1}, {}, rate=lambda s: k1)
    reactions = [birth, death]

    all_samples = []
    for _ in range(n_runs):
        times, traj = my_gillespie(reactions, [X0], t_max)
        sample_times = np.arange(burn_in, t_max, dt_sample)
        sample_vals = np.interp(sample_times, times, traj[:, 0])
        all_samples.append(sample_vals)
    all_samples = np.hstack(all_samples)

    n_min = int(np.floor(all_samples.min()))
    n_max = int(np.ceil(all_samples.max()))
    n_vals = np.arange(n_min, n_max + 1)

    # bins from n_min-0.5 to n_max+0.5 with width=bin_width
    edges = np.arange(n_min - 0.5, n_max + 0.5 + bin_width, bin_width)

    plt.figure(figsize=(8, 5))
    plt.hist(all_samples,
             bins=edges,
             density=True,
             alpha=0.6,
             edgecolor='black',
             linewidth=0.3)

    pn = analytic_birth_death_pn(n_vals, t_max, k1, k2, X0)
    plt.plot(n_vals, pn, 'ro-', markersize=4, label='Analytic p_n')

    plt.xticks(n_vals)
    plt.xlabel('Number of molecules')
    plt.ylabel('Probability')
    plt.title(f'Steady-state distribution (bin width={bin_width})')
    plt.legend()
    plt.tight_layout()
    plt.show()


# parameters
k1 = 0.1
k2 = 1.0
X0 = 0
t_max = 100000.0

plot_birth_death_histogram_dense(k1, k2, X0, t_max)
