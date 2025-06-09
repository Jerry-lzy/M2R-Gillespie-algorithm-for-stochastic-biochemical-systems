import numpy as np # noqa
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gillespie import Reaction, my_gillespie

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

def plot_birth_death1(k1, k2, X0, t_max, n_trajectories=200): # noqa  
    birth = Reaction(reactants={}, products={0: 1}, rate=lambda state: k2)
    death = Reaction(reactants={0: 1}, products={}, rate=lambda state: k1)
    reactions = [birth, death]

    # Prepare ODE
    t_grid = np.linspace(0, t_max, 300)
    x_ode = odeint(lambda x, t: k2 - k1*x, X0, t_grid).flatten()

    # analytic mean & std
    mean, var = analytic_birth_death_mean_var(t_grid, k1, k2, X0)
    std = np.sqrt(var)

    plt.figure(figsize=(8, 5))

    max_ssa = 0
    for run_idx in range(2):
        times, traj = my_gillespie(reactions, [X0], t_max)
        plt.step(times, traj[:, 0], where='post', alpha=0.7, label=f'SSA run {run_idx+1}') # noqa
        max_ssa = max(max_ssa, traj[:, 0].max())

    #  ODE
    plt.plot(t_grid, x_ode, 'k-', linewidth=1.5, label='Deterministic')
    max_ode = x_ode.max()

    plt.plot(t_grid, mean, 'r-', linewidth=1.5, label='Analytic mean')
    plt.plot(t_grid, mean + std, 'r--', linewidth=1.0, label='Analytic ±1 std')
    plt.plot(t_grid, mean - std, 'r--', linewidth=1.0)
    max_meanstd = (mean + std).max()

    y_max = int(np.ceil(max(max_ssa, max_ode, max_meanstd)))
    plt.yticks(np.arange(0, y_max + 1, 1))

    plt.xlim(0, t_max)
    plt.xlabel('Time')
    plt.ylabel('Molecule count')
    plt.title(f'Birth-Death: k1={k1}, k2={k2}, X₀={X0} (two SSA runs)')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()


k1 = 0.1
k2 = 1.0
X0 = 0
t_max = 100.0
n_trajectories = 200000

plot_birth_death1(k1, k2, X0, t_max, n_trajectories)
