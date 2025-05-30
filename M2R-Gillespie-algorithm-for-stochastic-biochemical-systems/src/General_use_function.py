import numpy as np # noqa
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from gillespie import my_gillespie

# run SSA and return times, trajectory
def run_ssa(reactions, initial, t_max): # noqa
    return my_gillespie(reactions, initial, t_max)

# solve ODE: func(y, t)
def run_ode(func, initial, t_grid): # noqa
    return odeint(func, initial, t_grid)

def plot_compare(times, traj, t_grid, ode_sol, species_idx=0, # noqa
                 labels={'ssa': 'SSA', 'ode': 'ODE'}, title=''):
    plt.step(times, traj[:, species_idx], where='post', label=labels['ssa'])
    plt.plot(t_grid, ode_sol[:, species_idx], '-', label=labels['ode'])
    plt.xlabel('time')
    plt.ylabel('count')
    plt.title(title)
    plt.legend()
    plt.show()
