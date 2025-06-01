import numpy as np  # noqa
import matplotlib.pyplot as plt
from gillespie import Reaction, my_gillespie
from scipy.integrate import odeint

# Case 5: Gene expression
# species 0 = mRNA, 1 = Protein
k_tx = 0.5
gamma_m = 0.1   # mRNA decay
k_tl = 0.2
gamma_p = 0.05  # protein decay

# reactions
gene_tx = Reaction(reactants={},    products={0: 1}, rate=lambda state: k_tx) # noqa
mRNA_deg = Reaction(reactants={0: 1}, products={}, rate=lambda state: gamma_m)  # noqa
translation = Reaction(reactants={0: 1}, products={0: 1, 1: 1}, rate=lambda state: k_tl) # noqa
protein_deg = Reaction(reactants={1: 1}, products={}, rate=lambda state: gamma_p) # noqa
initial = [0, 0]
t_max = 200

# SSA sim
times, traj = my_gillespie(
    [gene_tx, mRNA_deg, translation, protein_deg],
    initial, t_max
)

# To approch a solution
def ode(y, t):  # noqa
    m, p = y
    dm = k_tx - gamma_m*m
    dp = k_tl*m - gamma_p*p
    return [dm, dp]

# solve ODE
t = np.linspace(0, t_max, 300)  # noqa
sol = odeint(ode, initial, t)

# plot
plt.figure(figsize=(8, 5))
# mRNA
plt.step(times, traj[:, 0], where='post', alpha=0.5, label='SSA mRNA')
plt.plot(t, sol[:, 0], '-', label='ODE mRNA')
# Protein
plt.step(times, traj[:, 1], where='post', alpha=0.5, label='SSA protein')
plt.plot(t, sol[:, 1], '-', label='ODE protein')

plt.xlabel('time')
plt.ylabel('count')
plt.title('Gene expression')
plt.legend()
plt.show()
