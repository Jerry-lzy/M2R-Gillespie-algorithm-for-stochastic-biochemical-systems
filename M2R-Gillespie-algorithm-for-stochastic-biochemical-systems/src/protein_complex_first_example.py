import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie
from scipy.integrate import odeint

# Case 4: Protein complex formation
k = 0.005  # complex formation rate

# species 0 = A, 1 = B, 2 = C
R_form = Reaction(reactants={0: 1, 1: 1}, products={2: 1}, rate=lambda state: k) # noqa
initial = [50, 30, 0]  # [A], [B], [C]

t_max = 100
# simulate SSA
times, traj = my_gillespie([R_form], initial, t_max)

# ODE model
def ode(y, t): # noqa
    A, B, C = y # noqa
    rate = k * A * B
    dA = -rate # noqa
    dB = -rate # noqa
    dC = +rate # noqa
    return [dA, dB, dC]

# solve ODE
t = np.linspace(0, t_max, 200) # noqa
sol = odeint(ode, initial, t)

# plot
plt.figure(figsize=(8, 5))
plt.step(times, traj[:, 0], where='post', alpha=0.5, label='SSA [A]')
plt.plot(t, sol[:, 0], '-', label='ODE [A]')
plt.step(times, traj[:, 1], where='post', alpha=0.5, label='SSA [B]')
plt.plot(t, sol[:, 1], '-', label='ODE [B]')
plt.step(times, traj[:, 2], where='post', alpha=0.5, label='SSA [C]')
plt.plot(t, sol[:, 2], '-', label='ODE [C]')
plt.xlabel('t')
plt.ylabel('count')
plt.title('Complex formation A+B->C')
plt.legend()
plt.show()
