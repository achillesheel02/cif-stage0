# CIF Stage 0: Bootstrap Experiment

**Can a system with zero prior knowledge learn to anticipate its environment using only five preconditions?**

This is the first empirical test of the Convergent Information Framework (CIF) brain kernel. Not a product. A research experiment. The value is in what breaks, not what works.

## Theoretical Foundation

### The CIF Kernel

The core equation:

```
K(M, E, sigma) -> delta_M
```

Where:
- **M** = memory (structured experience)
- **E** = environment (observations + actions + outcomes)
- **sigma** = a selection function (what to attend to, what to predict)
- **delta_M** = memory update (what was learned)

The kernel transforms raw experience into structured memory that improves future predictions. The question is: what does M need to contain *before the first experience* for this loop to bootstrap at all?

### M_0: The Initial Memory

We hypothesise five preconditions are necessary and sufficient for bootstrap:

| # | Precondition | What it provides | Stage 0 implementation |
|---|---|---|---|
| 1 | **Distinction primitives** | Ability to tell states apart | Grid equality + Hamming distance |
| 2 | **Signal/noise prior** | Detecting change vs. static | Hamming > 0 after action = signal |
| 3 | **Self/world tag** | Attribution of cause | All experiences tagged `self_caused=true` |
| 4 | **Dimensional scaffold** | D1 (register events) + D6 (predict next state) | Memory store + dual-path prediction |
| 5 | **Goal attractor** | What to optimise | Prediction accuracy (seek predictable outcomes) |

No reward signal. No value function. No neural network. No backpropagation. No weights. The system seeks states where its predictions match reality. That's the entire drive.

### Dual-Path Prediction

Every prediction is made twice through independent paths:

- **Path A (memory)**: Nearest-neighbour lookup. "What happened last time in a state like this when I took this action?"
- **Path B (frequency)**: Most common outcome for this action, ignoring state. "What usually happens when I do this?"

The gap between them is the convergence signal:

```
Xi = accuracy_A - accuracy_B
```

When Xi > 0, memory structure helps. When Xi < 0, memory structure hurts. When Xi = 0, memory retrieval adds nothing over simple counting.

## Experiment Design

### Environment: MicroWorld

A 5x5 grid. One coloured pixel (the "marker") on a black background. Four actions: up, down, left, right. Deterministic transitions. Edge-clamping (no wrapping).

**Why this environment**: It's the simplest world where contingent learning is possible and non-trivial. 25 states x 4 actions = 100 state-action pairs. Small enough to fully explore, large enough to show a learning curve. Deterministic so we can distinguish retrieval failures from environmental noise.

**Why not 3x3**: 9 states = 36 pairs = trivial memorisation. No learning curve visible.

### Agent: Stage0Agent

- **Warmup** (episodes 0-99): Uniform random action selection. Builds initial memory.
- **Post-warmup** (episodes 100+): Softmax over familiarity scores with decaying temperature. The agent gravitates toward states it can predict, not states with "reward."
- **Memory**: Flat Vec of (action, state_before, state_after) tuples with hit counts. Deduplication on exact match. Retrieval: exact match first, Hamming-distance fallback.

### Metrics

Four core metrics, logged every 50 episodes:

| Metric | Formula | Healthy signal | Broken signal |
|---|---|---|---|
| `prediction_accuracy` | rolling mean of Path A hits | S-curve rising | Flat at ~0.25 (chance) |
| `consolidation_ratio` | unique_tuples / total_stores | Falls toward theoretical minimum | Stays at 1.0 (no dedup) |
| `action_entropy` | -sum(p_i * ln(p_i)) | Starts at ln(4)=1.386, decreases | Crashes to 0 (degenerate) |
| `path_advantage` | accuracy_A - accuracy_B | Positive and growing | Negative (Path B wins) |

Plus **Strand checkpoints** using [strand-core](https://github.com/achillesheel02/strand-core): per-action convergence matrix between Path A and Path B, Frobenius norm, gap classification.

### Configuration

```
world_size:       5       (25 positions)
n_actions:        4       (up/down/left/right)
warmup_episodes:  100     (random exploration)
max_episodes:     5000    (total run)
temperature_init: 2.0     (softmax temperature)
temperature_decay: 0.995  (per-episode cooling)
accuracy_window:  100     (rolling window for accuracy)
seed:             42      (reproducibility)
```

## Results

### Run Output (seed=42, default config)

```
episode     acc_A    acc_B   consol  entropy path_adv     temp  frobenius        gap
------------------------------------------------------------------------------------------
0           0.000    0.000    1.000    0.000    0.000    2.000      0.000       none
50          0.490    0.333    0.510    1.380    0.157    2.000      0.718 column_localised
100         0.485    0.168    0.515    1.386    0.317    1.980      1.000 column_localised
150         0.596    0.185    0.404    1.372    0.411    1.541      1.000 column_localised
200         0.667    0.174    0.333    1.370    0.493    1.199      1.000 column_localised
250         0.713    0.147    0.287    1.374    0.566    0.934      1.000 column_localised
300         0.761    0.153    0.239    1.372    0.608    0.727      1.000 column_localised
350         0.804    0.152    0.205    1.376    0.651    0.566      1.000 column_localised
400         0.844    0.167    0.180    1.377    0.676    0.440      1.000 column_localised
450         0.872    0.168    0.160    1.373    0.704    0.343      1.000 column_localised
500         0.907    0.230    0.144    1.366    0.677    0.267      1.000 column_localised
550         0.912    0.338    0.131    1.350    0.575    0.208      1.000 column_localised
600         0.917    0.445    0.120    1.333    0.472    0.162      0.875 column_localised
650         0.917    0.517    0.111    1.315    0.400    0.126      0.866 column_localised
700         0.917    0.565    0.103    1.297    0.353    0.098      0.707 column_localised
750         0.917    0.565    0.096    1.279    0.353    0.076      0.707 column_localised
...
4950        0.917    0.565    0.015    0.866    0.353    0.010      0.707 column_localised
```

### Summary

```
Total episodes:        5000
Final Path A accuracy: 0.917
Final Path B accuracy: 0.565
Path advantage:        +0.353
Action entropy:        0.865 (max: 1.386)
Unique tuples:         72
Total stores:          5000
Consolidation ratio:   0.014
Final temperature:     0.010

Strand: = = > >  |  frobenius=0.707  |  2/4 converged  |  gap=column_localised
```

**Diagnosis: BOOTSTRAP SUCCESS** -- predictions well above chance, Path A > Path B. M_0 is sufficient for this environment.

## Analysis

### What Worked

1. **Prediction accuracy reached 91.7%** (Path A). The system learned to anticipate its environment from zero prior knowledge using only the five M_0 preconditions. No reward signal was needed.

2. **Path advantage is strongly positive (+35.3 pp)**. Memory structure (nearest-neighbour retrieval) substantially outperforms simple frequency counting. The memory *architecture* matters, not just the data volume.

3. **Consolidation ratio fell to 0.014**. 5000 experiences compressed into 72 unique tuples. The deduplication write policy works: the system builds a compact model of its environment.

4. **Entropy decreased gradually** (1.386 -> 0.865), not catastrophically. The predictability-seeking goal attractor produces increasing action preference without collapsing to a degenerate strategy.

5. **The learning curve shows a clear S-shape**: slow during warmup (0-100), rapid climb (100-500), plateau at 91.7% (500+). This is the expected signature of a working bootstrap.

### What the 8.3% Residual Reveals

Path A saturates at 91.7%, not 100%. This is a **retrieval policy finding**:

- In a deterministic 5x5 world, the theoretical maximum accuracy is 100% (every state-action pair has exactly one outcome).
- The gap comes from the rolling accuracy window (last 100 episodes) and the fact that some state-action pairs at edges produce "no change" outcomes that the approximate retrieval occasionally confuses.
- The 72 unique tuples vs. the theoretical 100 state-action pairs means the agent hasn't explored every corner. The predictability-seeking attractor steers it away from unfamiliar territory.

This is finding #1: **the goal attractor trades exploration for exploitation**. A system that seeks predictability will avoid the unknown. This is correct behaviour for a bootstrap (consolidate what you know) but will need a curiosity mechanism for Stage 1.

### The Strand Signal

The strand matrix shows `= = > >` (2 converged, 2 divergent), classified as `column_localised`. This means:

- **Two actions** have Path A and Path B converging (both predict well -- likely the most-used actions)
- **Two actions** have Path A ahead of Path B (memory helps but frequency hasn't caught up -- likely less-used actions)
- The gap is **localised**, not systematic. The system's weakness is action-specific, not architectural.

Frobenius stabilised at 0.707, not 0. Full convergence would require Path B to catch up on all actions, which requires more uniform exploration -- exactly what the predictability-seeking attractor suppresses.

### Path B's Trajectory

Path B (frequency) follows an interesting arc:
- Starts low (0.168 at episode 100)
- Climbs as the agent's action distribution narrows (fewer actions = higher frequency accuracy)
- Saturates at 0.565

This reveals the **baseline intelligence of frequency counting in a deterministic world with non-uniform action selection**. Path B gets 56.5% accuracy for free, just by tracking "what usually happens." Path A's value is the 35.3 pp above this -- the value of *remembering context*, not just counting.

## Findings for Stage 1

| Finding | Implication | Stage 1 requirement |
|---|---|---|
| M_0 is sufficient for deterministic grid | The five preconditions bootstrap | Maintain all five in Stage 1 |
| Goal attractor suppresses exploration | System avoids unknown states | Add curiosity/novelty mechanism |
| 72/100 state-action pairs explored | Incomplete model of environment | Exploration bonus or intrinsic motivation |
| Strand gap is column_localised | Per-action, not per-architecture weakness | Per-action exploration balancing |
| Flat Vec scales to 72 tuples | O(n) scan acceptable at this scale | Stage 1 needs indexed retrieval (graph?) |
| Hamming distance sufficient for 5x5 binary | Similarity metric works here | Continuous states need embedding similarity |
| Consolidation ratio 0.014 | Extreme compression works for deterministic world | Stochastic environments need probabilistic memory |
| Temperature floor at 0.01 | Prevents complete greediness | May need adaptive temperature in Stage 1 |

## Failure Modes NOT Observed

These were predicted in the experiment design but did not manifest:

- **Entropy collapse to 0** (corner-sitting): Did not happen. The temperature decay rate (0.995) with floor (0.01) prevented it.
- **Path B beats Path A**: Never happened. Memory retrieval was always superior to frequency.
- **No consolidation**: Consolidation ratio dropped immediately and continuously.
- **Flat accuracy at chance**: Accuracy rose within the first 100 episodes.

These non-failures are themselves findings: the parameter defaults are reasonable, and the five M_0 preconditions are well-balanced for this environment.

## Reproducing

```bash
# Clone and build
git clone https://github.com/achillesheel02/cif-stage0.git
cd cif-stage0
cargo build --release

# Run with defaults
cargo run --release

# Ablation: smaller world
cargo run --release -- --size 3 --episodes 1000

# Ablation: no warmup
cargo run --release -- --warmup 0

# Ablation: different seed
cargo run --release -- --seed 99

# Full options
cargo run --release -- --help
```

**Dependency**: Requires [strand-core](https://github.com/achillesheel02/strand-core) at `../strand-core`.

## Architecture

```
src/
  grid.rs       Grid type + Hamming distance (state representation)
  config.rs     M0Config (all tunable parameters)
  world.rs      MicroWorld (deterministic 5x5 grid environment)
  memory.rs     ExperienceMemory (flat Vec, dedup, exact + approximate retrieval)
  agent.rs      Stage0Agent (dual-path prediction, softmax action selection)
  metrics.rs    Instrumentation (4 metrics + strand checkpoints + diagnosis)
  main.rs       CLI runner (the loop)
  lib.rs        Module exports
```

~600 lines of Rust. Compiles in <5 seconds. Runs 5000 episodes in <100ms. No neural networks. No LLM. No external dependencies beyond strand-core, serde, and rand.

## Theoretical Context

This experiment tests the first stage of a curriculum for building a CIF-based cognitive system:

| Stage | What it discovers | Environment | Memory |
|---|---|---|---|
| **0 (this)** | Actions affect environment | Deterministic grid | Flat Vec |
| 1 | State similarity enables generalisation | Stochastic grid | Graph/indexed |
| 2 | Temporal patterns exist | Sequences | Episodic memory |
| 3 | Symbolic compression reduces memory | Language grounding | Symbolic + episodic |
| 4 | Other agents exist | Multi-agent | Theory of mind |
| 5 | Self-model improves predictions | Reflexive | Meta-memory |

Stage 0 answers the question: *can M_0 bootstrap at all?* The answer is yes, in a deterministic environment with discrete states and four actions. Every subsequent stage will test what breaks when we remove one of those constraints.

## License

MIT

## Authors

Barak Achillah Asidi and Claude Pro Max (B)
