import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie
from scipy.integrate import odeint

# Case 2: birth-death
k1 = 0.1  # death rate
k2 = 1.0  # birth rate

R_birth = Reaction({}, {0: 1}, rate=k2)
R_death = Reaction({0: 1}, {}, rate=k1)
initial = [0]

# simulate SSA
times, traj = my_gillespie([R_birth, R_death], initial, t_max=100)

# ODE solution: dX/dt = k2 - k1*X
def ode(x, t): # noqa
    return k2 - k1 * x

# time grid
t = np.linspace(0, 100, 200) # noqa
x_ode = odeint(ode, initial, t)

# plot
plt.step(times, traj[:, 0], where='post', label='SSA')
plt.plot(t, x_ode, '-', label='ODE')
plt.xlabel('t')
plt.ylabel('count')
plt.title('Birth-Death ∅↔S')
plt.legend()
plt.show()
