import numpy as np # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie

# Case 1: degradation
k = 1.0
R = Reaction({0: 1}, {}, rate=1.0)
initial = [100]
t_max=6 # noqa

# simulate
times, traj = my_gillespie([R], initial, t_max)

# analytic mean and std
t = np.linspace(0, t_max, 200)
mean = initial[0] * np.exp(-k * t)
std = np.sqrt(initial[0] * np.exp(-k * t) * (1 - np.exp(-k * t)))

# plot
plt.step(times, traj[:, 0], where='post', label='SSA')
plt.plot(t, mean, '-', label='mean')
plt.plot(t, mean+std, '--', label='+1σ')
plt.plot(t, mean-std, '--', label='-1σ')
plt.xlabel('t')
plt.ylabel('count')
plt.title('Degradation S->∅')
plt.legend()
plt.show()
