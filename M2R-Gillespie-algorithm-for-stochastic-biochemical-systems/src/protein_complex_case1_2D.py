import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
from scipy.integrate import odeint  # noqa
from gillespie import Reaction, my_gillespie  # noqa


k1 = 0.001  # A + A → C
k2 = 0.01   # A + B → D
k3 = 1.2    # ∅ → A
k4 = 1.0    # ∅ → B


# R1: A + A → C
R1 = Reaction(
    reactants={0: 2},
    products={2: 1},
    rate=lambda s: k1
)

# R2: A + B → D
R2 = Reaction(
    reactants={0: 1, 1: 1},
    products={3: 1},
    rate=lambda s: k2
)

# R3: ∅ → A
R3 = Reaction(
    reactants={},
    products={0: 1},
    rate=lambda s: k3
)

# R4: ∅ → B
R4 = Reaction(
    reactants={},
    products={1: 1},
    rate=lambda s: k4
)

reactions = [R1, R2, R3, R4]


# RRE
def ode_system(x, t): # noqa
    A, B, C, D = x # noqa
    dA = k3 # noqa
    dA -= 2 * k1 * (A * (A - 1) / 2)
    dA -= k2 * A * B

    dB = k4 # noqa
    dB -= k2 * A * B

    dC = k1 * (A * (A - 1) / 2) # noqa

    dD = k2 * A * B # noqa
    return [dA, dB, dC, dD]


def sample_stationary_A_B( # noqa
    A0, B0, C0, D0, # noqa 
    t_max, burn_in, dt_sample
):  # noqa
    times, traj = my_gillespie(
        reactions,
        [A0, B0, C0, D0],
        t_max
    )
    # times: array of event times
    # traj:  shape = (num_events+1, 4), columns = [A, B, C, D]

    # build sampling grid in [burn_in, t_max)
    sample_times = np.arange(burn_in, t_max, dt_sample)

    # extract A‐ and B‐trajectories
    A_traj = traj[:, 0] # noqa
    B_traj = traj[:, 1] # noqa

    # step‐wise interpolation at sample_times
    A_samples = np.interp(sample_times, times, A_traj) # noqa
    B_samples = np.interp(sample_times, times, B_traj) # noqa

    # round to nearest integer
    all_A = np.round(A_samples).astype(int) # noqa
    all_B = np.round(B_samples).astype(int) # noqa

    return all_A, all_B


# Plot joint stationary distribution
def plot_joint_stationary_AB(A0, B0, C0, D0,t_max=5000.0,burn_in=2500.0,dt_sample=1.0): # noqa
    all_A, all_B = sample_stationary_A_B( # noqa
        A0, B0, C0, D0,
        t_max, burn_in, dt_sample
    )

    A_min, A_max = np.min(all_A), np.max(all_A) # noqa
    B_min, B_max = np.min(all_B), np.max(all_B) # noqa

    # set bin
    x_edges = np.arange(A_min, A_max + 2) - 0.5
    y_edges = np.arange(B_min, B_max + 2) - 0.5

    H, xedges, yedges = np.histogram2d( # noqa
        all_A, all_B,
        bins=[x_edges, y_edges],
        density=True
    )
    # H.shape = (len(x_edges)-1, len(y_edges)-1)

    plt.figure(figsize=(6, 5))
    X, Y = np.meshgrid(xedges, yedges, indexing='xy') # noqa
    plt.pcolormesh(
        X, Y, H.T,
        cmap="jet",
        shading="auto"
    )
    cbar = plt.colorbar()
    cbar.set_label("Probability density φ(A,B)")
    plt.xlabel("number of A molecules")
    plt.ylabel("number of B molecules")
    plt.title("Joint stationary distribution φ(A,B)")
    plt.tight_layout()
    plt.show()


A0, B0, C0, D0 = 0, 0, 0, 0
plot_joint_stationary_AB(
    A0, B0, C0, D0,
    t_max=5000.0,
    burn_in=2500.0,
    dt_sample=1.0
)
