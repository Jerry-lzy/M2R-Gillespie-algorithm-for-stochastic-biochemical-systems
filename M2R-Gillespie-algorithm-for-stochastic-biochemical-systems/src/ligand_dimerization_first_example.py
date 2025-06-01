import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie
from scipy.integrate import odeint

# Case 3: Dimerization
k = 0.01

# species 0 = S, 1 = S2
R_dim = Reaction(reactants={0: 2}, products={1: 1}, rate=lambda state: k) # noqa
initial = [100, 0]

t_max = 200

# simulate SSA
times, traj = my_gillespie([R_dim], initial, t_max=t_max)

# To approch a solution
def ode(y, t): # noqa
    S, S2 = y # noqa
    # Using the equation dS/dt = -2*k * C(S,2)
    rate = k * S*(S-1)/2
    dS = -2 * rate # noqa
    dS2 = +1 * rate # noqa
    return [dS, dS2]

# solve ODE
t = np.linspace(0, t_max, 300) # noqa
sol = odeint(ode, initial, t)

# plot
plt.step(times, traj[:, 0], where='post', alpha=0.5, label='SSA [S]')
plt.plot(t, sol[:, 0], '-', label='ODE [S]')
plt.step(times, traj[:, 1], where='post', alpha=0.5, label='SSA [S2]')
plt.plot(t, sol[:, 1], '-', label='ODE [S2]')
plt.xlabel('t')
plt.ylabel('count')
plt.title('Dimerization 2S->S2')
plt.legend()
plt.show()
