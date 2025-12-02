
import numpy as np
from scipy.ndimage import gaussian_filter

# ============================================================
# Stage 3 Simulation — Evolutionary Stability of Polarity
# Includes Stage 2 seeding logic
# ============================================================

GRID_SIZE = 20
NUM_TRIALS = 8
POP_SIZE = 20
MIN_A = 0.1

# Developmental gains
GAIN_L_STAGE2 = 0.98
GAIN_R_STAGE2 = 1.02

GAIN_L_STAGE3 = 0.98
GAIN_R_STAGE3 = 1.02

rng = np.random.default_rng()


# ----------------- Environment and basic operations ----------------- #

def generate_environment():
    """Generate a 20x20 environment with global structure and local anomaly."""
    g = rng.normal(0, 1, (GRID_SIZE, GRID_SIZE))
    g = gaussian_filter(g, sigma=1.0, mode="reflect")
    cx, cy = rng.integers(1, GRID_SIZE - 2, size=2)
    patch = rng.uniform(1.5, 2.0)
    g[cx - 1:cx + 2, cy - 1:cy + 2] += patch
    return g


def gaussian_filter_width(img, width):
    """Apply Gaussian smoothing with lower-bound on width."""
    if width < MIN_A:
        width = MIN_A
    return gaussian_filter(img, sigma=width, mode="reflect")


def compute_mismatch(img, width):
    """Compute absolute-difference mismatch between original and filtered grid."""
    filt = gaussian_filter_width(img, width)
    return np.sum(np.abs(img - filt))


def compute_fitness(a_L, a_R, lam, gain_L, gain_R, num_trials=NUM_TRIALS):
    """Compute fitness (negative mean RT) for given parameters and gains."""
    RTs = []
    for _ in range(num_trials):
        img = generate_environment()
        mL = compute_mismatch(img, a_L)
        mR = compute_mismatch(img, a_R)

        mL *= gain_L
        mR *= gain_R

        D = (mR - mL) + lam * (abs(mR) - abs(mL))
        RT = 1.0 / (abs(D) + 1e-6)
        RTs.append(RT)
    return -np.mean(RTs)


# -------------------------- GA primitives --------------------------- #

def create_individual():
    """Create a new individual [a_L, a_R, lambda] from symmetric priors."""
    return np.array([
        rng.uniform(1.4, 1.6),   # a_L
        rng.uniform(1.4, 1.6),   # a_R
        rng.uniform(0.2, 0.4)    # λ
    ])


def mutate(ind, sigma=0.05):
    """Gaussian mutation with lower-bounded filter widths."""
    ind = ind.copy()
    ind += rng.normal(0, sigma, size=3)
    ind[0] = max(ind[0], MIN_A)
    ind[1] = max(ind[1], MIN_A)
    return ind


def recombine(p1, p2):
    """Arithmetic mean recombination."""
    return (p1 + p2) / 2.0


# ------------------------ Stage 2 seeding --------------------------- #

def evolve_lineage_stage2(num_generations=20, gain_L=GAIN_L_STAGE2, gain_R=GAIN_R_STAGE2):
    """Evolve one Stage 2 lineage and return best individual and its fitness."""
    pop = np.array([create_individual() for _ in range(POP_SIZE)])

    for _ in range(num_generations):
        fitnesses = np.array([
            compute_fitness(ind[0], ind[1], ind[2], gain_L, gain_R)
            for ind in pop
        ])
        idx = np.argsort(fitnesses)[::-1]
        parents = pop[idx[:POP_SIZE // 2]]

        new_pop = []
        for _ in range(POP_SIZE):
            p1, p2 = parents[rng.integers(len(parents), size=2)]
            child = recombine(p1, p2)
            child = mutate(child, sigma=0.05)
            new_pop.append(child)
        pop = np.array(new_pop)

    final_fitnesses = np.array([
        compute_fitness(ind[0], ind[1], ind[2], gain_L, gain_R)
        for ind in pop
    ])
    best = pop[np.argmax(final_fitnesses)]
    return best, final_fitnesses.max()


def get_stage2_seeds(n_lineages=20):
    """Run Stage 2 for n_lineages and return best seeds."""
    seeds = []
    for i in range(n_lineages):
        best, fit = evolve_lineage_stage2()
        aL, aR, lam = best
        delta = aR - aL
        print(f"[Stage2 seed] Lineage {i+1}: aL={aL:.3f}, aR={aR:.3f}, Δ={delta:.3f}, fitness={fit:.5f}")
        seeds.append(best)
    return np.array(seeds)


# -------------------------- Stage 3 logic --------------------------- #

def evolve_stage3_from_seed(seed,
                            num_generations=30,
                            mutation_sigma=0.05,
                            num_trials=NUM_TRIALS):
    """
    Evolve a Stage 3 lineage starting from a Stage 2 best seed.
    Returns:
        history_delta: list of Δ = a_R - a_L for best individual per generation
        reversed_flag: True if polarity reversal occurred after divergence threshold
    """
    # Initialize population around seed
    pop = []
    for _ in range(POP_SIZE):
        ind = seed + rng.normal(0, mutation_sigma, size=3)
        ind[0] = max(ind[0], MIN_A)
        ind[1] = max(ind[1], MIN_A)
        pop.append(ind)
    pop = np.array(pop)

    history_delta = []
    divergence_threshold = 0.3
    crossed = False
    initial_sign = None
    reversed_flag = False

    for _ in range(num_generations):
        fitnesses = np.array([
            compute_fitness(ind[0], ind[1], ind[2], GAIN_L_STAGE3, GAIN_R_STAGE3, num_trials=num_trials)
            for ind in pop
        ])
        idx = np.argsort(fitnesses)[::-1]
        best = pop[idx[0]]
        aL, aR, lam = best
        delta = aR - aL
        history_delta.append(delta)

        if abs(delta) >= divergence_threshold:
            if not crossed:
                crossed = True
                initial_sign = np.sign(delta)
            else:
                if np.sign(delta) != initial_sign:
                    reversed_flag = True

        parents = pop[idx[:POP_SIZE // 2]]
        new_pop = []
        for _ in range(POP_SIZE):
            p1, p2 = parents[rng.integers(len(parents), size=2)]
            child = recombine(p1, p2)
            child = mutate(child, sigma=mutation_sigma)
            new_pop.append(child)
        pop = np.array(new_pop)

    return np.array(history_delta), reversed_flag


def run_stage3_short(n_lineages=20,
                     num_generations=30,
                     mutation_sigma=0.05,
                     num_trials=NUM_TRIALS):
    """
    Main Stage 3 run: 20 lineages, 30 generations (as in manuscript).
    """
    seeds = get_stage2_seeds(n_lineages=n_lineages)
    all_hist = []
    any_reversal = False
    final_deltas = []
    min_post_div = []

    for i, seed in enumerate(seeds):
        history_delta, reversed_flag = evolve_stage3_from_seed(
            seed,
            num_generations=num_generations,
            mutation_sigma=mutation_sigma,
            num_trials=num_trials
        )
        all_hist.append(history_delta)
        any_reversal = any_reversal or reversed_flag

        # Collect stats after divergence threshold
        divergence_threshold = 0.3
        post = history_delta[np.abs(history_delta) >= divergence_threshold]
        if len(post) > 0:
            min_post_div.append(np.min(np.abs(post)))
        final_deltas.append(history_delta[-1])

        print(f"[Stage3 short] Lineage {i+1}: final Δ={history_delta[-1]:.3f}, reversed={reversed_flag}")

    all_hist = np.array(all_hist)
    final_deltas = np.array(final_deltas)
    min_post_div = np.array(min_post_div) if len(min_post_div) > 0 else np.array([])

    print("\n=== Stage 3 (short run) summary ===")
    print(f"Any reversals: {any_reversal}")
    if len(min_post_div) > 0:
        print(f"Min |Δ| after divergence: {min_post_div.min():.3f}")
        print(f"Final Δ range: {final_deltas.min():.3f} to {final_deltas.max():.3f}")

    return all_hist, any_reversal, final_deltas, min_post_div


def run_stage3_long(n_lineages=10,
                    num_generations=200,
                    mutation_sigma=0.05,
                    num_trials=NUM_TRIALS):
    """
    Extended Stage 3 run: 10 lineages, 200 generations for long-run stability.
    """
    seeds = get_stage2_seeds(n_lineages=n_lineages)
    all_hist = []
    any_reversal = False
    final_deltas = []
    min_post_div = []

    for i, seed in enumerate(seeds):
        history_delta, reversed_flag = evolve_stage3_from_seed(
            seed,
            num_generations=num_generations,
            mutation_sigma=mutation_sigma,
            num_trials=num_trials
        )
        all_hist.append(history_delta)
        any_reversal = any_reversal or reversed_flag

        divergence_threshold = 0.3
        post = history_delta[np.abs(history_delta) >= divergence_threshold]
        if len(post) > 0:
            min_post_div.append(np.min(np.abs(post)))
        final_deltas.append(history_delta[-1])

        print(f"[Stage3 long] Lineage {i+1}: final Δ={history_delta[-1]:.3f}, reversed={reversed_flag}")

    all_hist = np.array(all_hist)
    final_deltas = np.array(final_deltas)
    min_post_div = np.array(min_post_div) if len(min_post_div) > 0 else np.array([])

    print("\n=== Stage 3 (long run) summary ===")
    print(f"Any reversals: {any_reversal}")
    if len(min_post_div) > 0:
        print(f"Min |Δ| after divergence: {min_post_div.min():.3f}")
        print(f"Final Δ range: {final_deltas.min():.3f} to {final_deltas.max():.3f}")

    return all_hist, any_reversal, final_deltas, min_post_div


if __name__ == "__main__":
    # Example usage: run short Stage 3 protocol
    print("Running Stage 3 short protocol (20 lineages, 30 generations)...")
    run_stage3_short()

    # Uncomment below to run long protocol:
    # print("\nRunning Stage 3 long protocol (10 lineages, 200 generations)...")
    # run_stage3_long()
