import numpy as np # noqa
from scipy.special import comb


class Reaction: # noqa
    # reactants/products: dict {species_index: count}
    def __init__(self, reactants, products, rate): # noqa
        self.reactants = reactants
        self.products = products
        self.rate = rate

    def propensity(self, state): # noqa
        # compute reaction rate
        p = self.rate(state)
        for i, v in self.reactants.items():
            if state[i] < v:
                return 0
            p *= comb(state[i], v, exact=True)
        return p


def my_gillespie(reactions, initial_state, t_max): # noqa
    # basic Gillespie SSA
    rng = np.random.default_rng()
    state = np.array(initial_state, int)
    n = len(state)

    changes = []
    for r in reactions:
        delta = np.zeros(n, int)
        for i, v in r.products.items():
            delta[i] += v
        for i, v in r.reactants.items():
            delta[i] -= v
        changes.append(delta)

    t = 0
    times = [t]
    traj = [state.copy()]

    while t < t_max:
        rates = np.array([r.propensity(state) for r in reactions])
        total = rates.sum()
        if total == 0:
            break

        # next event time
        tau = rng.exponential(1/total)
        t += tau
        if t > t_max:
            break

        # pick reaction
        r = rng.random() * total
        idx = np.searchsorted(rates.cumsum(), r)
        state = state + changes[idx]
        times.append(t)
        traj.append(state.copy())

    return np.array(times), np.vstack(traj)
