
import numpy as np
from scipy.ndimage import gaussian_filter

# ============================================================
# Stage 2 Simulation — Minimal Developmental Gain Asymmetry
# ============================================================

GRID_SIZE = 20
NUM_TRIALS = 8
NUM_GENERATIONS = 20
POP_SIZE = 20

# Developmental gains (2% asymmetry)
GAIN_L = 0.98
GAIN_R = 1.02

# Filter constraints
MIN_A = 0.1

rng = np.random.default_rng()


def generate_environment():
    g = rng.normal(0, 1, (GRID_SIZE, GRID_SIZE))
    g = gaussian_filter(g, sigma=1.0, mode="reflect")

    cx, cy = rng.integers(1, GRID_SIZE - 2, size=2)
    patch = rng.uniform(1.5, 2.0)
    g[cx - 1:cx + 2, cy - 1:cy + 2] += patch
    return g


def gaussian_filter_width(img, width):
    if width < MIN_A:
        width = MIN_A
    return gaussian_filter(img, sigma=width, mode="reflect")


def compute_mismatch(img, width):
    filt = gaussian_filter_width(img, width)
    return np.sum(np.abs(img - filt))


def compute_fitness(a_L, a_R, lam):
    RTs = []
    for _ in range(NUM_TRIALS):
        img = generate_environment()

        mL = compute_mismatch(img, a_L)
        mR = compute_mismatch(img, a_R)

        mL *= GAIN_L
        mR *= GAIN_R

        D = (mR - mL) + lam * (abs(mR) - abs(mL))
        RT = 1.0 / (abs(D) + 1e-6)
        RTs.append(RT)

    return -np.mean(RTs)  # maximize fitness


def create_individual():
    return np.array([
        rng.uniform(1.4, 1.6),  # a_L
        rng.uniform(1.4, 1.6),  # a_R
        rng.uniform(0.2, 0.4)   # λ
    ])


def mutate(ind):
    ind = ind.copy()
    ind += rng.normal(0, 0.05, size=3)
    ind[0] = max(ind[0], MIN_A)
    ind[1] = max(ind[1], MIN_A)
    return ind


def recombine(p1, p2):
    return (p1 + p2) / 2.0


def evolve_lineage():
    pop = np.array([create_individual() for _ in range(POP_SIZE)])

    for gen in range(NUM_GENERATIONS):
        fitnesses = np.array([
            compute_fitness(ind[0], ind[1], ind[2]) for ind in pop
        ])

        idx = np.argsort(fitnesses)[::-1]
        parents = pop[idx[:POP_SIZE // 2]]

        new_pop = []
        for _ in range(POP_SIZE):
            p1, p2 = parents[rng.integers(len(parents), size=2)]
            child = recombine(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        pop = np.array(new_pop)

    final_fitnesses = np.array([
        compute_fitness(ind[0], ind[1], ind[2]) for ind in pop
    ])
    best = pop[np.argmax(final_fitnesses)]
    return best, final_fitnesses.max()


def run_stage2(n_lineages=20):
    results = []
    for i in range(n_lineages):
        best, fit = evolve_lineage()
        aL, aR, lam = best
        delta = aR - aL
        results.append((aL, aR, lam, delta, fit))
        print(f"Lineage {i+1}: aL={aL:.3f}, aR={aR:.3f}, Δ={delta:.3f}")

    return results


if __name__ == "__main__":
    results = run_stage2(n_lineages=20)
    print("\n=== Stage 2 Completed ===")
    for i, (aL, aR, lam, delta, fit) in enumerate(results):
        print(f"{i+1}: Δ={delta:.3f}, fitness={fit:.5f}")
