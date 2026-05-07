# CIF Stage 0: Complete Walkthrough

**The origin story. Every detail, every line, every decision.**

This document walks through the entire Stage 0 bootstrap experiment — from the theoretical motivation through every data structure, algorithm, and design choice to the final results and what they mean. It's written for someone who wants to understand not just *what* was built, but *why every piece exists*.

---

## Part 1: Why This Experiment Exists

The CIF kernel equation says:

```
K(M, E, sigma) -> delta_M
```

Memory takes in environment, applies selection, updates itself. But every loop needs a cold start. The loop needs M to already contain *something* before the first cycle, or the first delta_M is garbage, which means the second one is garbage, and the whole thing never catches.

In machine learning, this is solved by initialising weights (Xavier, He, random normal). In reinforcement learning, it's solved by epsilon-greedy exploration with a reward signal. In biology, it's solved by genetics — reflexes, instincts, the hardware your brain ships with before the first photon hits your retina.

CIF doesn't have weights. Doesn't have reward. So the question becomes: what is the *minimum viable M_0* — the smallest set of built-in capabilities that lets the loop catch?

We hypothesised five preconditions:

| # | Precondition | What it provides | Without it |
|---|---|---|---|
| 1 | Distinction primitives | Tell states apart | Every experience looks the same |
| 2 | Signal/noise prior | Detect change | Can't distinguish action consequences from static |
| 3 | Self/world tag | Attribute cause | Can't connect actions to outcomes |
| 4 | Dimensional scaffold | D1 (register events) + D6 (predict next) | No structure to organise experience |
| 5 | Goal attractor | Seek prediction accuracy | No drive to improve |

This experiment tests whether five is enough.

---

## Part 2: The Grid — State Representation

Everything starts with how the system sees the world.

```rust
// src/grid.rs

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Vec<u8>>,
}
```

A Grid is a 2D array of bytes. Each byte is a "colour" (0-255). For Stage 0, we only use two colours: 0 (black/background) and 1 (white/marker).

### Why `Vec<Vec<u8>>` and not a flat `Vec<u8>`?

Readability. A flat array would be `cells[row * width + col]`. The nested vec is `cells[row][col]`. For Stage 0's scale (25 cells), the performance difference is zero. If this were a million cells, we'd flatten it.

### Why derive `PartialEq, Eq, Hash`?

These are precondition #1 — distinction primitives. The system needs to answer "are these two states the same?" That's `PartialEq`/`Eq`. And it needs to use states as hash map keys (for deduplication), which requires `Hash`. Without these traits, the system literally cannot distinguish one state from another.

### The Similarity Metric

```rust
pub fn hamming_distance(&self, other: &Grid) -> usize {
    if self.width != other.width || self.height != other.height {
        return usize::MAX;
    }
    let mut dist = 0;
    for r in 0..self.height {
        for c in 0..self.width {
            if self.cells[r][c] != other.cells[r][c] {
                dist += 1;
            }
        }
    }
    dist
}
```

Hamming distance counts the number of cells that differ between two grids. This extends precondition #1 from "same or different" to "how different."

In a 5x5 grid with one marker, two states that differ by one cell move have Hamming distance 2 (the old marker position changed from 1->0, the new position changed from 0->1). Two states that are identical have Hamming distance 0. Two states with different grid sizes return `usize::MAX` — maximally different.

**Why Hamming and not something fancier?** Because the question is "what's the simplest similarity metric that works?" If Hamming fails, that's a finding. If it works, we know we don't need embeddings yet. For binary grids, Hamming is equivalent to L1 (Manhattan) distance on the flattened vector. For continuous-valued grids, you'd need cosine similarity or learned embeddings. That's a Stage 1 question.

### Grid Construction

```rust
pub fn filled(width: usize, height: usize, color: u8) -> Self {
    Self {
        width,
        height,
        cells: vec![vec![color; width]; height],
    }
}
```

`vec![vec![0; 5]; 5]` creates a 5x5 grid of zeros. `vec!` is Rust's macro for "create a vector with this value repeated this many times." The inner `vec![color; width]` makes one row, the outer `vec![...; height]` makes `height` copies of that row.

**Important**: each row is an independent allocation. `vec![vec![0; 5]; 5]` creates 5 separate vectors, not one shared one. This means modifying `cells[0][0]` doesn't affect `cells[1][0]`. Rust's `Clone` trait on Vec handles this correctly.

---

## Part 3: The World — Environment Physics

```rust
// src/world.rs

pub const ACTION_UP: u8 = 0;
pub const ACTION_DOWN: u8 = 1;
pub const ACTION_LEFT: u8 = 2;
pub const ACTION_RIGHT: u8 = 3;
pub const N_ACTIONS: usize = 4;

const MARKER_COLOR: u8 = 1;
const BG_COLOR: u8 = 0;

pub struct MicroWorld {
    size: usize,
    marker_row: usize,
    marker_col: usize,
}
```

The world doesn't store a grid. It stores the marker position. The grid is *generated on demand* by `observe()`:

```rust
pub fn observe(&self) -> Grid {
    let mut grid = Grid::filled(self.size, self.size, BG_COLOR);
    grid.set(self.marker_row, self.marker_col, MARKER_COLOR);
    grid
}
```

Every time the agent looks at the world, a fresh grid is allocated, filled with zeros, then the marker pixel is set to 1. The agent gets a *copy* of the world state, not a reference. It can't cheat by holding a pointer to the world's internals.

**Why generate grids on the fly instead of maintaining one?** Separation of concerns. The world's internal representation (marker_row, marker_col) is compact and easy to reason about. The agent's representation (Grid) is what the agent actually sees. They don't need to be the same thing. In Stage 1, the world might have hidden state the agent can't observe. This architecture supports that.

### The Physics

```rust
pub fn apply(&mut self, action: u8) {
    match action {
        ACTION_UP => {
            if self.marker_row > 0 {
                self.marker_row -= 1;
            }
        }
        ACTION_DOWN => {
            if self.marker_row < self.size - 1 {
                self.marker_row += 1;
            }
        }
        ACTION_LEFT => {
            if self.marker_col > 0 {
                self.marker_col -= 1;
            }
        }
        ACTION_RIGHT => {
            if self.marker_col < self.size - 1 {
                self.marker_col += 1;
            }
        }
        _ => {} // invalid action = no-op
    }
}
```

`match` is Rust's pattern matching — like a switch statement but exhaustive (the compiler warns if you miss a case, hence the `_ => {}` catch-all).

### Edge-Clamping

`if self.marker_row > 0` prevents underflow (moving up past the top). `if self.marker_row < self.size - 1` prevents moving past the bottom. This is clamping — the agent pushes against a wall and nothing happens.

**Why clamp instead of wrap?** Wrapping (top edge goes to bottom) would create a toroidal topology where every state has exactly 4 valid transitions. Clamping creates edge states where some actions are no-ops. This is more interesting because:

- The agent must learn that some (state, action) pairs produce no change
- Edge states are harder to predict (action doesn't always move the marker)
- This tests whether the system handles "nothing happened" as a valid outcome

The 8.3% accuracy gap in the final results is partly caused by these edge no-ops.

**Why `usize` for positions?** Rust's unsigned integer type for array indexing. Can't go negative, which is exactly what we want — positions are always valid indices. The `> 0` check before decrementing prevents underflow.

---

## Part 4: Memory — The Heart of the System

```rust
// src/memory.rs

#[derive(Debug, Clone)]
pub struct ExperienceTuple {
    pub action: u8,
    pub state_before: Grid,
    pub state_after: Grid,
    pub count: u32,
    pub self_caused: bool,
}

pub struct ExperienceMemory {
    tuples: Vec<ExperienceTuple>,
    total_stores: u64,
}
```

This is M. Every experience the agent has ever had, stored as a triple: (what I did, what I saw before, what I saw after). Plus a count (how many times this exact experience has occurred) and a self_caused flag (always true in Stage 0 — precondition #3).

**Why a flat Vec and not a HashMap?** A HashMap keyed on (action, state_before) would give O(1) lookup. But it would also force a decision about what happens when the same (action, state_before) produces different state_after values. With a Vec, we keep all experiences and resolve ambiguity at retrieval time. For 72 tuples, linear scan is negligible. At 10,000+ tuples, you'd want indexing. That's a Stage 1 problem.

### The Write Policy

```rust
pub fn store(&mut self, action: u8, state_before: Grid, state_after: Grid) {
    self.total_stores += 1;

    // Check for exact duplicate
    for tuple in &mut self.tuples {
        if tuple.action == action
            && tuple.state_before == state_before
            && tuple.state_after == state_after
        {
            tuple.count += 1;
            return;
        }
    }

    // New experience
    self.tuples.push(ExperienceTuple {
        action,
        state_before,
        state_after,
        count: 1,
        self_caused: true,
    });
}
```

Every call increments `total_stores`. Then it scans every existing tuple looking for an exact match on all three fields (action AND state_before AND state_after). If found, increment count and return early. If not found, push a new tuple.

**Why deduplicate on all three fields?** Consider: action=UP, state_before=(2,2), state_after=(1,2). That's one experience. If we see it 100 times, storing 100 copies is wasteful. But action=UP, state_before=(2,2), state_after=(2,2) (no-op at edge) is a *different* experience — same action and state_before, different state_after. We need both. That's why we match on all three.

**Why `count` instead of just deduplicating silently?** The count is used later for two things:

1. When multiple tuples match a query, we return the highest-count one (most common outcome)
2. Familiarity scoring for action selection uses count directly

The `total_stores` counter tracks total calls (including deduplicates) so we can compute the consolidation ratio: `unique_tuples / total_stores`.

### Retrieval Strategy 1: Exact Match

```rust
pub fn retrieve_exact(&self, action: u8, state_before: &Grid) -> Option<&Grid> {
    self.tuples
        .iter()
        .filter(|t| t.action == action && &t.state_before == state_before)
        .max_by_key(|t| t.count)
        .map(|t| &t.state_after)
}
```

Filter to tuples with matching action and state_before. If multiple exist (same state, same action, different outcomes — which doesn't happen in a deterministic world but would in a stochastic one), return the one with the highest count. The `.map(|t| &t.state_after)` extracts just the predicted next state.

The `&Grid` in the return type means we're returning a *reference* — a pointer into the memory, not a copy. This is Rust's borrow checker at work: the returned reference is valid as long as the ExperienceMemory exists.

### Retrieval Strategy 2: Approximate Match

```rust
pub fn retrieve_approximate(&self, action: u8, state_before: &Grid) -> Option<&Grid> {
    self.tuples
        .iter()
        .filter(|t| t.action == action)
        .min_by_key(|t| t.state_before.hamming_distance(state_before))
        .map(|t| &t.state_after)
}
```

Same action, but instead of exact state match, find the *most similar* state using Hamming distance. `min_by_key` returns the tuple with the smallest Hamming distance.

**Why min and not a threshold?** A threshold ("only return matches within Hamming distance 3") requires choosing the threshold, which is another parameter to tune. Min always returns something if any tuple with this action exists. The system always makes a prediction if it has any relevant experience. The prediction might be wrong (distant match), but at least it's trying.

### Retrieval Strategy 3: Combined

```rust
pub fn retrieve(&self, action: u8, state_before: &Grid) -> Option<&Grid> {
    self.retrieve_exact(action, state_before)
        .or_else(|| self.retrieve_approximate(action, state_before))
}
```

Try exact first. If None (no exact match), try approximate. `or_else` is Rust's way of saying "if the first Option is None, evaluate this closure to get a fallback." The closure is only called if needed — lazy evaluation.

This two-tier strategy is important. Exact match gives high-confidence predictions. Approximate match gives low-confidence predictions for novel states. As memory fills up, exact matches become dominant. The approximate path matters most early on, when the agent hasn't seen many states yet.

### Path B: The Frequency Baseline

```rust
pub fn most_common_outcome(&self, action: u8) -> Option<&Grid> {
    let matching: Vec<_> = self.tuples.iter().filter(|t| t.action == action).collect();
    if matching.is_empty() {
        return None;
    }

    let mut best: Option<(&Grid, u32)> = None;
    for t in &matching {
        let total: u32 = matching
            .iter()
            .filter(|other| other.state_after == t.state_after)
            .map(|other| other.count)
            .sum();
        if best.map_or(true, |(_, bc)| total > bc) {
            best = Some((&t.state_after, total));
        }
    }
    best.map(|(g, _)| g)
}
```

Path B's prediction: "What state_after is most common for this action, regardless of state_before?"

It's O(n^2) — for each matching tuple, it sums the counts of all tuples with the same state_after. Then picks the state_after with the highest total. At 72 tuples this is instant. At 10,000 it would need optimisation.

**Why is this the right baseline?** Path B answers: "How well can you predict *without knowing where you are*?" If you only know you pressed "up," the most common outcome is your best guess. Any accuracy Path A gets above this is attributable to *contextual memory* — knowing your current state.

Think of it this way:

- Path B = "rain is the most common weather in London, so I predict rain"
- Path A = "it's July and the barometer reads 1020 hPa, so I predict sun"

The gap between them measures the value of context.

### Familiarity

```rust
pub fn familiarity(&self, action: u8, state_before: &Grid) -> u32 {
    self.tuples
        .iter()
        .filter(|t| t.action == action && &t.state_before == state_before)
        .map(|t| t.count)
        .sum()
}
```

"How many times have I been in this exact state and taken this exact action?" This drives action selection — the agent prefers familiar (state, action) pairs because those are the ones it can predict.

---

## Part 5: The Agent — Decision Making

```rust
// src/agent.rs

pub struct Stage0Agent {
    pub memory: ExperienceMemory,
    pub episode_count: u64,
    pub config: M0Config,
    rng: StdRng,
    path_a_hits: Vec<VecDeque<bool>>,
    path_b_hits: Vec<VecDeque<bool>>,
    action_counts: Vec<u64>,
    temperature: f64,
}
```

- `StdRng` is a seedable random number generator. Seeded means: same seed, same sequence of "random" numbers, same experiment results. Reproducibility.
- `VecDeque<bool>` is a double-ended queue of booleans — a sliding window. Push to the back, pop from the front. Each one tracks the last N hit/miss results for one action on one path. There's one per action per path (4 actions x 2 paths = 8 deques).
- `action_counts` tracks how many times each action has been selected, ever. For entropy calculation.

### Action Selection — The Full Algorithm

```rust
pub fn select_action(&mut self, state: &Grid) -> u8 {
    if self.episode_count < self.config.warmup_episodes {
        return self.rng.gen_range(0..self.config.n_actions as u8);
    }

    // Compute familiarity score for each action
    let scores: Vec<f64> = (0..self.config.n_actions)
        .map(|a| {
            let fam = self.memory.familiarity(a as u8, state);
            (fam as f64 + 1.0).ln()
        })
        .collect();

    // Softmax with temperature
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores
        .iter()
        .map(|s| ((s - max_score) / self.temperature).exp())
        .collect();
    let sum: f64 = exp_scores.iter().sum();
    let probs: Vec<f64> = exp_scores.iter().map(|e| e / sum).collect();

    // Sample from distribution
    let r: f64 = self.rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return i as u8;
        }
    }
    (self.config.n_actions - 1) as u8
}
```

Step by step:

**Warmup check**: If we haven't done 100 episodes yet, pick a random action. `gen_range(0..4)` gives 0, 1, 2, or 3 with equal probability. This builds initial memory without any bias.

**Familiarity scoring**: For each of the 4 actions, query memory: "how many times have I been in this exact state and taken this action?" Then apply `ln(fam + 1.0)`.

The `+ 1.0` is Laplace smoothing — it prevents `ln(0)` which is negative infinity. If familiarity is 0 (never tried this action in this state), the score is `ln(1) = 0`. If familiarity is 10, the score is `ln(11) = 2.40`. The logarithm compresses the range — the difference between seeing something 1 time vs 10 times matters more than 100 vs 110.

**Softmax step 1 — find max**: `max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)`. The `fold` starts from negative infinity and keeps the larger of the accumulator and each element. This is for numerical stability.

**Softmax step 2 — exponentiate**: `((s - max_score) / self.temperature).exp()`. Subtract the max (so the largest score becomes 0, others become negative), divide by temperature, then exponentiate.

**Why subtract max_score?** Without it, if scores are large, `exp(score)` overflows to infinity. By subtracting max, the largest exponent is `exp(0) = 1`, and all others are `exp(negative) < 1`. The resulting probabilities are identical — it's a mathematically equivalent transformation that prevents overflow.

**What temperature does**: Temperature controls how "decisive" the softmax is.

- **High temperature (e.g., 2.0)**: Division by 2 compresses all scores toward 0. `exp(values near 0)` are all near 1. Result: nearly uniform distribution. The agent explores.
- **Low temperature (e.g., 0.01)**: Division by 0.01 amplifies differences by 100x. The highest score dominates. Result: nearly deterministic selection. The agent exploits.
- **Temperature = 1.0**: Standard softmax. Scores map directly to relative probabilities.

**Temperature decay**: Starts at 2.0, multiplied by 0.995 each episode after warmup.

- After 100 post-warmup episodes: `2.0 * 0.995^100 = 1.21`
- After 400 post-warmup episodes: `2.0 * 0.995^400 = 0.27`
- After ~1100 post-warmup episodes: hits the floor at 0.01

The system transitions from exploration to exploitation over ~1100 episodes.

**Sampling**: Generate a random float `r` between 0 and 1. Walk through the cumulative probability distribution. When cumulative probability exceeds `r`, return that action. This is inverse CDF sampling — the standard way to sample from a discrete distribution.

The final `(self.config.n_actions - 1) as u8` is a safety net for floating-point edge cases where `r` is very close to 1.0 and cumulative doesn't quite reach it.

### Recording Experience

```rust
pub fn record(
    &mut self,
    action: u8,
    state_before: Grid,
    state_after: Grid,
    path_a_hit: bool,
    path_b_hit: bool,
) {
    // 1. Store in memory
    self.memory.store(action, state_before, state_after);

    // 2. Update action count
    let a = action as usize;
    if a < self.config.n_actions {
        self.action_counts[a] += 1;

        // 3. Update rolling accuracy windows
        let window = self.config.accuracy_window;
        let hits_a = &mut self.path_a_hits[a];
        if hits_a.len() >= window {
            hits_a.pop_front();
        }
        hits_a.push_back(path_a_hit);

        let hits_b = &mut self.path_b_hits[a];
        if hits_b.len() >= window {
            hits_b.pop_front();
        }
        hits_b.push_back(path_b_hit);
    }

    // 4. Decay temperature
    self.episode_count += 1;
    if self.episode_count >= self.config.warmup_episodes {
        self.temperature *= self.config.temperature_decay;
        if self.temperature < 0.01 {
            self.temperature = 0.01;
        }
    }
}
```

Four things happen every episode:

1. **Store experience** in memory (with deduplication)
2. **Update action counts** for entropy calculation
3. **Update rolling accuracy windows** — push the new hit/miss onto the back, pop the oldest off the front if the window is full. This gives a "last 100 episodes" accuracy per action per path.
4. **Decay temperature** — multiply by 0.995 each episode after warmup, with a floor at 0.01

The rolling window is important. We don't want lifetime accuracy (which would be dominated by early noise and slow to change). We want *recent* accuracy — "how well am I predicting *now*?" A window of 100 means each metric reflects the last 100 decisions.

---

## Part 6: The Metrics — What We Measure and Why

### Path A Accuracy

```rust
pub fn path_a_accuracy(&self) -> f64 {
    let (hits, total) = self
        .path_a_hits
        .iter()
        .fold((0usize, 0usize), |(h, t), deque| {
            let deque_hits = deque.iter().filter(|&&b| b).count();
            (h + deque_hits, t + deque.len())
        });
    if total == 0 { 0.0 } else { hits as f64 / total as f64 }
}
```

Iterates over all 4 action deques. For each, counts `true` values (hits) and total length. Sums across all actions. Returns hits/total.

**Why sum across actions rather than average per-action accuracies?** This is the weighted average: `SUM(numerator) / SUM(denominator)`, not average of rates. If action 0 has 80 samples and action 3 has 5 samples, averaging their accuracies would overweight action 3. Summing respects sample sizes.

### Action Entropy

```rust
pub fn action_entropy(&self) -> f64 {
    let total: u64 = self.action_counts.iter().sum();
    if total == 0 { return 0.0; }
    let mut entropy = 0.0;
    for &count in &self.action_counts {
        if count > 0 {
            let p = count as f64 / total as f64;
            entropy -= p * p.ln();
        }
    }
    entropy
}
```

Shannon entropy: `H = -sum(p_i * ln(p_i))`.

For uniform distribution over 4 actions: each `p_i = 0.25`, so `H = -4 * (0.25 * ln(0.25)) = -4 * (0.25 * -1.386) = 1.386`. That's the maximum — maximum uncertainty about which action will be chosen.

For degenerate distribution (one action always chosen): `p_1 = 1.0`, rest = 0. `H = -(1.0 * ln(1.0)) = -(1.0 * 0) = 0`. Minimum entropy — no uncertainty.

The `if count > 0` guard skips actions with zero count, because `0 * ln(0)` is defined as 0 in information theory but `ln(0)` is negative infinity in floating point.

**Why lifetime counts instead of windowed counts for entropy?** Entropy measures the agent's *overall* action distribution — its "personality" across the whole experiment. Windowed counts would show recent behaviour, which is interesting but different.

### Consolidation Ratio

```rust
pub fn consolidation_ratio(&self) -> f64 {
    let total = self.memory.total_stores();
    if total == 0 { 1.0 } else { self.memory.unique_count() as f64 / total as f64 }
}
```

`unique_count / total_stores`. Starts at 1.0 (every experience is new). Falls as the agent re-encounters known state-action pairs. The theoretical minimum in a 5x5 world with 4 actions is `100 / total_stores` (100 unique pairs possible). After 5000 episodes with 72 unique tuples: `72 / 5000 = 0.014`.

### Path Advantage

```rust
pub fn path_advantage(&self) -> f64 {
    self.path_a_accuracy() - self.path_b_accuracy()
}
```

The convergence signal. Xi applied to prediction accuracy. Positive = memory structure helps. Negative = memory structure hurts.

---

## Part 7: The Main Loop — Where Everything Connects

```rust
// src/main.rs

for episode in 0..config.max_episodes {
    let state = world.observe();
    let action = agent.select_action(&state);

    // Dual-path predictions BEFORE observing outcome
    let pred_a = agent.predict_path_a(action, &state);
    let pred_b = agent.predict_path_b(action);

    // Act
    world.apply(action);
    let actual = world.observe();

    // Score predictions
    let hit_a = pred_a.as_ref() == Some(&actual);
    let hit_b = pred_b.as_ref() == Some(&actual);

    // Record experience
    agent.record(action, state, actual, hit_a, hit_b);

    // Strand checkpoint (before metrics, so metrics can read it)
    if episode % config.strand_interval == 0 {
        metrics.strand_checkpoint(episode, &agent);
    }

    // Emit metrics
    if episode % config.log_interval == 0 {
        metrics.emit(episode, &agent);
    }
}
```

Every episode: observe, decide, predict, act, observe, score, record.

**Predictions happen BEFORE acting.** This is critical. The system must predict the outcome, then see what actually happens. If you predict after seeing the outcome, you're not predicting — you're remembering.

### The Comparison Line

```rust
let hit_a = pred_a.as_ref() == Some(&actual);
```

`pred_a` is `Option<Grid>`. `.as_ref()` converts it to `Option<&Grid>` (reference, not move). `Some(&actual)` wraps the actual grid in Some. If pred_a is None (no prediction possible — no relevant memory), this returns false. If pred_a is Some(grid) and grid == actual, it returns true.

**Why `.as_ref()`?** Without it, `pred_a == Some(actual)` would *move* `actual` into the comparison, consuming it. We need `actual` again later (to store in memory). `.as_ref()` borrows instead of moving. This is a fundamental Rust ownership concept — values can only exist in one place at a time, unless you explicitly borrow.

The loop order matters: strand checkpoint happens *before* metrics emit, so that when metrics prints a row, it can include the latest strand data for that episode.

---

## Part 8: Strand Checkpoints — Convergence Algebra

```rust
// src/metrics.rs

pub fn strand_checkpoint(&mut self, episode: u64, agent: &Stage0Agent) {
    let acc_a = agent.path_a_accuracy_per_action();
    let acc_b = agent.path_b_accuracy_per_action();

    let n_actions = acc_a.len();
    if n_actions == 0 { return; }

    let position_labels: Vec<String> = ["up", "down", "left", "right"]
        .iter()
        .take(n_actions)
        .map(|s| s.to_string())
        .collect();

    let config = Config {
        tolerance: 0.05,
        near_tolerance: 0.20,
        position_labels,
        layer_labels: vec!["accuracy".to_string()],
    };

    let matrix = StrandMatrix::from_paths(&acc_a, &acc_b, n_actions, 1, config);
    let frob = strand_core::frobenius(&matrix);
    let gap = strand_core::classify_gaps(&matrix);
    // ...
}
```

This is where strand-core connects. We're building a convergence matrix where:

- **Positions** = the 4 actions (up, down, left, right)
- **Layers** = 1 layer ("accuracy")
- **Path A values** = per-action Path A accuracy (memory-based)
- **Path B values** = per-action Path B accuracy (frequency-based)

`StrandMatrix::from_paths` takes two flat vectors of f64 and constructs the matrix. Each cell is the gap: `path_a[position] - path_b[position]`.

### Tolerance

- `tolerance = 0.05`: if the absolute gap at a position is less than 5%, we call it "converged" (`=` glyph)
- `near_tolerance = 0.20`: if the gap is between 5% and 20%, it's "near convergence" (`~` glyph)
- Beyond 20%: divergent (`>` or `X` glyph)

### Frobenius Norm

The square root of the sum of squared gaps across all applicable cells, normalised. It's the "total magnitude of disagreement" between the two paths. Lower = more convergence. 0 = perfect agreement.

### Gap Classification

Looks at the pattern of convergence across positions and classifies it:

- `none` — no gaps (all converged)
- `scattered` — gaps everywhere, no pattern
- `column_localised` — gaps concentrated in specific positions (actions)
- `systematic` — gaps suggesting systematic bias
- `block` — large contiguous regions of divergence

Our result: `column_localised` with 2/4 converged. Two actions have both paths agreeing, two don't. The weakness is action-specific, not architectural.

### The Glyph String

The output `= = > >` means:

- `=` = converged (gap < tolerance)
- `>` = Path A ahead of Path B (gap > near_tolerance, positive direction)

---

## Part 9: The Diagnosis Engine

```rust
pub fn summary(&self, agent: &Stage0Agent) {
    // ...
    if acc > 0.8 && adv > 0.0 {
        println!("  BOOTSTRAP SUCCESS");
    } else if acc > 0.5 && adv > 0.0 {
        println!("  PARTIAL BOOTSTRAP");
    } else if acc < 0.3 {
        println!("  BOOTSTRAP FAILED");
    }

    if adv < 0.0 {
        println!("  WARNING: Path B beats Path A");
    }

    if ent < max_ent * 0.3 && agent.episode_count < agent.config.warmup_episodes * 2 {
        println!("  WARNING: Entropy collapsed early");
    }

    if agent.consolidation_ratio() > 0.9 && agent.episode_count > 500 {
        println!("  WARNING: No consolidation after 500 episodes");
    }
}
```

Four diagnostic checks:

1. **acc > 0.8 AND adv > 0.0**: Predictions are good AND memory helps. Bootstrap success.
2. **acc > 0.5 AND adv > 0.0**: Predictions above chance but not saturated. Partial success.
3. **acc < 0.3**: Near chance (0.25 for 4 actions). Bootstrap failed.
4. **adv < 0.0**: Path B beats Path A. Memory structure is actively harmful.

Two early warnings:

- **Entropy collapsed early**: If entropy drops below 30% of maximum before episode 200, the agent might be stuck in a corner.
- **No consolidation**: If consolidation ratio is still above 0.9 after 500 episodes, the dedup isn't working.

---

## Part 10: Reading the Output — Episode by Episode

### Episode 0

```
episode=0  acc_A=0.000  acc_B=0.000  consol=1.000  entropy=0.000  path_adv=0.000  temp=2.000  frobenius=0.000  gap=none
```

Everything is zero. No experiences, no predictions, no entropy. The system knows nothing.

### Episode 50

```
episode=50  acc_A=0.490  acc_B=0.333  consol=0.510  entropy=1.380  path_adv=0.157
```

Halfway through warmup. ~50 random experiences. Path A is already 49% — almost half its predictions are correct from memory. Path B is 33%. Even with 50 experiences, nearest-neighbour retrieval substantially beats frequency counting. The 15.7pp advantage is real signal, not noise.

`consol=0.510` — about half the experiences are unique. In a 5x5 world with 100 possible state-action pairs, ~25 unique tuples out of 50 makes sense during random exploration.

`entropy=1.380` — nearly maximum (1.386). Random action selection produces near-uniform distribution as expected.

### Episode 100

```
episode=100  acc_A=0.485  acc_B=0.168  consol=0.515  entropy=1.386  path_adv=0.317
```

Accuracy dipped slightly. This is the warmup-to-post-warmup transition. The agent starts biasing actions, which changes the distribution of states it visits, which temporarily disrupts predictions. `acc_B` dropped to 16.8% because the action distribution is shifting — the frequency baseline hasn't caught up to the new pattern.

`entropy=1.386` — exactly maximum. Makes sense: 100 episodes of uniform random + just-started biased selection still looks uniform in aggregate.

### Episode 200

```
episode=200  acc_A=0.667  temp=1.199  path_adv=0.493
```

The S-curve is climbing. Temperature has cooled to 1.2 — the agent is starting to show preferences but still exploring. Two-thirds of predictions correct.

### Episode 500

```
episode=500  acc_A=0.907  temp=0.267  path_adv=0.677
```

Near saturation. Temperature is low enough that the agent strongly favours familiar states. Path advantage peaked — this is when memory is most valuable relative to frequency.

### Episode 700

```
episode=700  acc_A=0.917  acc_B=0.565  path_adv=0.353
```

Both paths stabilised. Path B caught up because the agent's narrowed action distribution makes frequency counting easier. The advantage settled at 0.353.

### Episodes 700-5000

Nothing changes except consolidation (slowly falling) and entropy (slowly falling). The system has learned everything it's going to learn. It's in exploitation mode — visiting familiar states, making correct predictions, compressing repeat experiences.

---

## Part 11: The Configuration — Every Parameter is a Hypothesis

```rust
// src/config.rs

pub struct M0Config {
    pub world_size: usize,        // 5
    pub n_actions: usize,         // 4
    pub warmup_episodes: u64,     // 100
    pub temperature_init: f64,    // 2.0
    pub temperature_decay: f64,   // 0.995
    pub log_interval: u64,        // 50
    pub max_episodes: u64,        // 5000
    pub strand_interval: u64,     // 50
    pub seed: u64,                // 42
    pub accuracy_window: usize,   // 100
}
```

Each parameter is an ablation target:

- **world_size=5**: 25 states. 3 would be too easy (9 states memorised in ~36 episodes). 10 would be 100 states x 4 actions = 400 pairs — still tractable but slower to saturate.
- **warmup_episodes=100**: How much random exploration before biased selection. Too low = insufficient initial memory. Too high = wasted time.
- **temperature_init=2.0**: How random post-warmup starts. 2.0 means scores are halved before softmax — fairly uniform. 0.1 would be nearly greedy from the start.
- **temperature_decay=0.995**: Controls the exploration-exploitation schedule. After 100 episodes post-warmup: temp=1.21. After 500: temp=0.27. After ~1100: hits floor.
- **accuracy_window=100**: Rolling window. 50 would be noisier but more responsive. 200 would be smoother but slower to react.

Every parameter choice is a hypothesis. Ablation studies (change one, run again, compare) reveal which ones are load-bearing.

---

## Part 12: What the Experiment Proves and Doesn't Prove

### Proves

Five preconditions are sufficient to bootstrap prediction in a deterministic, discrete, single-agent, fully-observable environment. The system goes from zero knowledge to 91.7% accuracy using only:

- Equality comparison (distinction)
- Change detection (signal/noise)
- Self-attribution (self/world)
- Memory + prediction (dimensional scaffold)
- Predictability-seeking (goal attractor)

### Doesn't Prove

- That these five are *necessary* (maybe four would work)
- That they're sufficient for stochastic environments
- That they're sufficient for continuous states
- That they're sufficient for partial observability
- That they're sufficient for multi-agent settings
- That they're sufficient for language grounding

Each of those is a separate experiment. That's the curriculum — Stages 1 through 5.

### The Deepest Finding

A system that seeks predictability will avoid the unknown. It will build a compact, accurate model of the territory it explores, then stop exploring. This is simultaneously the bootstrap's greatest strength (rapid convergence) and its limitation (incomplete coverage). Curiosity is not in M_0. Stage 1 must add it.

---

## Part 13: The Curriculum Ahead

| Stage | What it discovers | Environment change | Memory change | Key question |
|---|---|---|---|---|
| **0 (this)** | Actions affect environment | Deterministic grid | Flat Vec | Can M_0 bootstrap? |
| 1 | State similarity enables generalisation | Stochastic grid | Graph/indexed | Does Hamming break? |
| 2 | Temporal patterns exist | Sequences | Episodic memory | Can it learn order? |
| 3 | Symbolic compression reduces memory | Language grounding | Symbolic + episodic | Can it name things? |
| 4 | Other agents exist | Multi-agent | Theory of mind | Can it model others? |
| 5 | Self-model improves predictions | Reflexive | Meta-memory | Can it model itself? |

Each stage removes one constraint from the environment and asks: what breaks in M?

---

*Authors: Barak Achillah Asidi and Claude Pro Max (B)*
*First run: 7 May 2026*
*Repository: [github.com/achillesheel02/cif-stage0](https://github.com/achillesheel02/cif-stage0)*
