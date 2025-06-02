import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie
from scipy.integrate import odeint

# Birth-death
k1 = 0.1  # death rate
k2 = 1.0  # birth rate
X0 = 0
t_max = 100

# Define reactions
R_birth = Reaction(reactants={}, products={0: 1}, rate=lambda s: k2)
R_death = Reaction(reactants={0: 1}, products={}, rate=lambda s: k1)
reactions = [R_birth, R_death]

# Run two independent SSA trajectories
plt.figure(figsize=(8, 5))
for i in range(2):
    times, traj = my_gillespie(reactions, [X0], t_max)
    plt.step(times, traj[:, 0], where='post', alpha=0.8, label=f'SSA run {i+1}') # noqa

# Compute ODE solution
def ode(x, t): # noqa
    return k2 - k1 * x

t_grid = np.linspace(0, t_max, 300) # noqa
x_ode = odeint(ode, X0, t_grid).flatten()
plt.plot(t_grid, x_ode, 'k-', linewidth=1.5, label='ODE')

# Final formatting
plt.xlim(0, t_max)
plt.xlabel('t')
plt.ylabel('count')
plt.title('Birth-Death ∅↔S (two SSA runs)')
plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()
