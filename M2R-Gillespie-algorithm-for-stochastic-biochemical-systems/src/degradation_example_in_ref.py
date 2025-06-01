import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie

# Case 1: degradation
k = 1.0
R = Reaction(reactants={0: 1}, products={}, rate=lambda state: k)
initial = [100]
t_max=6 # noqa

# analytic mean and std
t = np.linspace(0, t_max, 200)
mean = initial[0] * np.exp(-k * t)
std = np.sqrt(initial[0] * np.exp(-k * t) * (1 - np.exp(-k * t)))

plt.figure(figsize=(8, 5))
for run_idx in range(2):
    times, traj = my_gillespie([R], initial, t_max)
    plt.step(times, traj[:, 0], where='post', alpha=0.7, label=f'SSA run {run_idx+1}') # noqa

# plot analytic mean and ±1
plt.plot(t, mean,  'r-',  linewidth=1.5, label='Analytic mean')
plt.plot(t, mean+std, 'r--', linewidth=1.0, label='Analytic ±1')
plt.plot(t, mean-std, 'r--', linewidth=1.0)

plt.xlabel('t')
plt.ylabel('count')
plt.title('Degradation S->∅')
plt.legend()
plt.show()
