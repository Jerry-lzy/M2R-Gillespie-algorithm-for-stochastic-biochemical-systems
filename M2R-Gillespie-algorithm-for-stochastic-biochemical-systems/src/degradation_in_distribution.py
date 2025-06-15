import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie
from scipy.special import comb

# Parameters for the degradation reaction
k = 1.0
X0 = 20
t_max = 6.0
n_runs = 500

# Define the degradation SSA:
R = Reaction(reactants={0: 1}, products={}, rate=lambda state: k)

# Analytic distribution at time t:
#  p_n(t) = C(X0, n) * (e^{-k t})^n * (1 - e^{-k t})^{X0 - n}
def analytic_degradation_pn(n_array, t, k, X0): # noqa
    p_survive = np.exp(-k * t)
    p_die = 1 - p_survive
    pn = comb(X0, n_array) * (p_survive**n_array) * (p_die**(X0 - n_array))
    return pn


final_counts = []
for _ in range(n_runs):
    times, traj = my_gillespie([R], [X0], t_max)
    final_counts.append(traj[-1, 0])

final_counts = np.array(final_counts)


n_min = 0
n_max = X0
bins = np.arange(n_min, n_max + 2) - 0.5

plt.figure(figsize=(8, 5))
plt.hist(final_counts,
         bins=bins,
         density=True,
         alpha=0.6,
         edgecolor='black',
         linewidth=0.5,
         label=f'SSA ')


n_vals = np.arange(n_min, n_max + 1)
pn = analytic_degradation_pn(n_vals, t_max, k, X0)
plt.plot(n_vals, pn, 'ro-', markersize=4, label='Analytic Binomial')

plt.xlabel('Number of molecules n')
plt.ylabel('Probability')
plt.title(f'Degradation Stationary distribution at t = {t_max}')
plt.xticks(n_vals[::10])
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
