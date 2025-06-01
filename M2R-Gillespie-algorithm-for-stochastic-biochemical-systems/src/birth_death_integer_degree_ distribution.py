import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie
from scipy.special import comb, factorial

def analytic_birth_death_pn(n_array, t, k1, k2, n0): # noqa
    lam = (k2 / k1) * (1 - np.exp(-k1 * t))
    prefactor = np.exp(-lam)
    pn = np.zeros_like(n_array, dtype=float)
    for idx, n in enumerate(n_array):
        total = 0.0
        for k in range(min(n0, n) + 1):
            binom_term = comb(n0, k) * (1 - np.exp(-k1 * t))**(n0 - k) * np.exp(-k1 * k * t) # noqa
            pois_term  = lam**(n - k) / factorial(n - k) # noqa
            total += binom_term * pois_term
        pn[idx] = prefactor * total
    return pn

def plot_birth_death_histogram(k1, k2, X0, t_max, # noqa
                               n_runs=20,
                               burn_in=5000.0,
                               dt_sample=1.0):
    birth = Reaction(reactants={}, products={0: 1}, rate=lambda s: k2)
    death = Reaction(reactants={0: 1}, products={}, rate=lambda s: k1)
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

    # Use fixed bin edges at integer ±0.5, width=1
    bins = n_vals - 0.5

    plt.figure(figsize=(8, 5))
    # draw bars with width=1 and no gaps
    counts, _, patches = plt.hist(all_samples,
                                  bins=np.append(bins, n_max + 0.5),
                                  density=True,
                                  alpha=0.6,
                                  edgecolor='black',
                                  linewidth=0.5)
    for bar in patches:
        bar.set_width(1.0)

    pn = analytic_birth_death_pn(n_vals, t_max, k1, k2, X0)
    plt.plot(n_vals, pn, 'ro-', markersize=4, label='Analytic p_n')

    plt.xticks(n_vals)                      # show every integer on x
    plt.xlabel('Number of molecules')
    plt.ylabel('Probability')
    plt.title('Steady-state distribution (binned at integers)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# parameters
k1 = 0.1 # noqa
k2 = 1.0
X0 = 0
t_max = 20000.0

plot_birth_death_histogram(k1, k2, X0, t_max)
