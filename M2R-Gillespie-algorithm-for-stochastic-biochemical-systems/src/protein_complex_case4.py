import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from gillespie import Reaction, my_gillespie

# Reaction constants
k1 = 0.001  # A + A → C
k2 = 0.01   # A + B → D
k3 = 1.2    # ∅ → A
k4 = 1.0    # ∅ → B

# Define reactions
R1 = Reaction({0: 2}, {2: 1}, rate=lambda s: k1)
R2 = Reaction({0: 1, 1: 1}, {3: 1}, rate=lambda s: k2)
R3 = Reaction({}, {0: 1}, rate=lambda s: k3)
R4 = Reaction({}, {1: 1}, rate=lambda s: k4)
reactions = [R1, R2, R3, R4]

# Deterministic ODE system
def ode_system(x, t):
    A, B, C, D = x
    dA = k3 - k1 * A * (A - 1) - k2 * A * B
    dB = k4 - k2 * A * B
    dC = k1 * A * (A - 1) / 2
    dD = k2 * A * B
    return [dA, dB, dC, dD]

# Plotting function for A, B with SSA mean
def plot_AB_with_mean(A0, B0, C0, D0, t_max=100.0, n_ssa=5):
    t_grid = np.linspace(0, t_max, 500)
    ode_sol = odeint(ode_system, [A0, B0, C0, D0], t_grid)

    # Run SSA simulations
    ssa_results = []
    for _ in range(n_ssa):
        times, traj = my_gillespie(reactions, [A0, B0, C0, D0], t_max)
        ssa_results.append((times, traj))

    # Interpolate all trajectories onto t_grid
    ssa_interp = np.zeros((n_ssa, len(t_grid), 4))
    for i, (times, traj) in enumerate(ssa_results):
        for j in range(4):
            f = interp1d(times, traj[:, j], kind='previous',
                         bounds_error=False,
                         fill_value=(traj[0, j], traj[-1, j]))
            ssa_interp[i, :, j] = f(t_grid)
    ssa_mean = np.mean(ssa_interp, axis=0)

    # Plot A and B
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    species = ["A", "B"]
    colors = plt.cm.tab10(np.arange(n_ssa))

    for idx, ax in enumerate(axes):
        ax.plot(t_grid, ode_sol[:, idx], "--k", lw=1.5, label="Deterministic")
        for i, (times, traj) in enumerate(ssa_results):
            ax.step(times, traj[:, idx], where="post", color=colors[i], alpha=0.6,
                    label=f"SSA run {i+1}" if idx == 0 else None)
        ax.plot(t_grid, ssa_mean[:, idx], color="red", lw=2,
                label="SSA mean" if idx == 0 else None)

        ax.set_ylabel(f"{species[idx]} count")
        if idx == 0:
            ax.legend(fontsize="small", loc="upper right")

    axes[-1].set_xlabel("Time")
    plt.suptitle(f"Protein complex formation: A₀={A0}, B₀={B0}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



A0 = 0
B0 = 30
C0 = 0
D0 = 0
t_final = 100.0
plot_AB_with_mean(A0, B0, C0, D0, t_final, n_ssa=5)
