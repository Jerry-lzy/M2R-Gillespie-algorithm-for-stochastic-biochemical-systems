import numpy as np # noqa
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gillespie import Reaction, my_gillespie
from scipy.special import comb, factorial

def analytic_birth_death_mean_var(t, k1, k2, X0): # noqa
    """ # noqa
    Compute mean and variance of the birth-death process:
      mean(t) = (k2/k1)*(1 - exp(-k1*t)) + X0*exp(-k1*t)
      var(t)  = X0*exp(-k1*t)*(1 - exp(-k1*t)) + (k2/k1)*(1 - exp(-k1*t))
    """
    expm = np.exp(-k1 * t)
    mean = (k2 / k1) * (1 - expm) + X0 * expm
    var = X0 * expm * (1 - expm) + (k2 / k1) * (1 - expm)
    return mean, var

def analytic_birth_death_pn(n_array, t, k1, k2, n0): # noqa
    """ # noqa
    Compute the probability p_n(t) that calulate by my mates.
    """
    lam = (k2 / k1) * (1 - np.exp(-k1 * t))
    prefactor = np.exp(-lam)
    pn = np.zeros_like(n_array, dtype=float)

    for idx, n in enumerate(n_array):
        total = 0.0
        max_k = min(n0, n)
        for k in range(max_k + 1):
            binom_term = comb(n0, k) * (1 - np.exp(-k1 * t))**(n0 - k) * np.exp(-k1 * k * t) # noqa
            pois_term = lam**(n - k) / factorial(n - k)
            total += binom_term * pois_term
        pn[idx] = prefactor * total

    return pn

def plot_birth_death(k1, k2, X0, t_max, n_trajectories=200): # noqa
    birth = Reaction(reactants={}, products={0: 1},
                     rate=lambda state: k2)
    death = Reaction(reactants={0: 1}, products={},
                     rate=lambda state: k1)
    reactions = [birth, death]

    # Single SSA trajectory vs ODE & analytic
    times, traj = my_gillespie(reactions, [X0], t_max)

    # ODE
    t_grid = np.linspace(0, t_max, 300)
    x_ode = odeint(lambda x, t: k2 - k1*x, X0, t_grid).flatten()

    # analytic mean & std……
    mean, var = analytic_birth_death_mean_var(t_grid, k1, k2, X0)
    std = np.sqrt(var)

    plt.figure(figsize=(8, 5))
    plt.step(times, traj[:, 0], where='post', alpha=0.6, label='SSA traj')
    plt.plot(t_grid, x_ode, 'k-', label='ODE sol')
    plt.plot(t_grid, mean,  'r-', label='Analytic mean')
    plt.plot(t_grid, mean+std, 'r--', label='Analytic ±1 std')
    plt.plot(t_grid, mean-std, 'r--')
    plt.legend(); plt.show() # noqa

    # histogram as empirical distribution and analytic distribution
    final_counts = []
    for _ in range(n_trajectories):
        _, traj_s = my_gillespie(reactions, [X0], t_max)
        final_counts.append(traj_s[-1, 0])
    bins = np.arange(min(final_counts), max(final_counts)+2) - 0.5
    plt.hist(final_counts, bins=bins, density=True, alpha=0.5,
             label=f'SSA final dist (N={n_trajectories})')

    n_vals = np.arange(min(final_counts), max(final_counts)+1)
    pn = analytic_birth_death_pn(n_vals, t_max, k1, k2, X0)
    plt.plot(n_vals, pn, 'ro-', label='Analytic p_n')
    plt.legend()
    plt.show()

k1 = 0.1 # noqa
k2 = 1.0
X0 = 0
t_max = 50.0
n_trajectories = 200

plot_birth_death(k1, k2, X0, t_max, n_trajectories)
