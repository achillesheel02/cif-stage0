/// Stage0Agent — the CIF kernel at bootstrap.
///
/// M_0 priors wired in:
/// 1. Distinction primitives — Grid equality (via ExperienceMemory)
/// 2. Signal/noise prior — change detection (Hamming > 0 = signal)
/// 3. Self/world tag — all experiences tagged self_caused=true
/// 4. Dimensional scaffold — D1 (register events), D6 (predict next state)
/// 5. Goal attractor — prediction accuracy (seek predictable outcomes)
///
/// Dual-path prediction:
///   Path A (memory): nearest-neighbour lookup
///   Path B (frequency): most common outcome per action

use crate::config::M0Config;
use crate::grid::Grid;
use crate::memory::ExperienceMemory;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

pub struct Stage0Agent {
    pub memory: ExperienceMemory,
    pub episode_count: u64,
    pub config: M0Config,
    rng: StdRng,
    /// Per-action rolling accuracy for Path A (memory-based).
    path_a_hits: Vec<VecDeque<bool>>,
    /// Per-action rolling accuracy for Path B (frequency-based).
    path_b_hits: Vec<VecDeque<bool>>,
    /// Per-action total selections (for entropy calculation).
    action_counts: Vec<u64>,
    /// Current temperature.
    temperature: f64,
}

impl Stage0Agent {
    pub fn new(config: M0Config) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let window = config.accuracy_window;
        let n = config.n_actions;
        Self {
            memory: ExperienceMemory::new(),
            episode_count: 0,
            temperature: config.temperature_init,
            config,
            rng,
            path_a_hits: (0..n).map(|_| VecDeque::with_capacity(window)).collect(),
            path_b_hits: (0..n).map(|_| VecDeque::with_capacity(window)).collect(),
            action_counts: vec![0; n],
        }
    }

    /// Select an action given current state.
    ///
    /// Before warmup: uniform random.
    /// After warmup: softmax over familiarity scores (prediction-accuracy seeking).
    pub fn select_action(&mut self, state: &Grid) -> u8 {
        if self.episode_count < self.config.warmup_episodes {
            return self.rng.gen_range(0..self.config.n_actions as u8);
        }

        // Compute score: blend familiarity (exploit) with inverse familiarity (explore)
        let cw = self.config.curiosity_weight;
        let scores: Vec<f64> = (0..self.config.n_actions)
            .map(|a| {
                let fam = self.memory.familiarity(a as u8, state);
                let ln_fam = (fam as f64 + 1.0).ln();
                (1.0 - cw) * ln_fam + cw * (-ln_fam)
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

    /// Path A prediction: nearest-neighbour memory lookup.
    pub fn predict_path_a(&self, action: u8, state: &Grid) -> Option<Grid> {
        self.memory.retrieve(action, state).cloned()
    }

    /// Path B prediction: most common outcome for this action.
    pub fn predict_path_b(&self, action: u8) -> Option<Grid> {
        self.memory.most_common_outcome(action).cloned()
    }

    /// Record an experience and update running accuracy.
    pub fn record(
        &mut self,
        action: u8,
        state_before: Grid,
        state_after: Grid,
        path_a_hit: bool,
        path_b_hit: bool,
    ) {
        // Store in memory
        self.memory
            .store(action, state_before, state_after);

        // Update action count
        let a = action as usize;
        if a < self.config.n_actions {
            self.action_counts[a] += 1;

            // Update rolling accuracy windows
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

        // Decay temperature
        self.episode_count += 1;
        if self.episode_count >= self.config.warmup_episodes {
            if self.config.adaptive_temperature {
                let expected = (self.config.world_size * self.config.world_size * self.config.n_actions) as f64;
                let coverage = self.memory.unique_count() as f64 / expected;
                if coverage >= self.config.coverage_gate {
                    self.temperature *= self.config.temperature_decay;
                }
            } else {
                self.temperature *= self.config.temperature_decay;
            }
            if self.temperature < 0.01 {
                self.temperature = 0.01;
            }
        }
    }

    // ── Metrics ────────────────────────────────────────────────────────

    /// Overall Path A accuracy (across all actions, rolling window).
    pub fn path_a_accuracy(&self) -> f64 {
        let (hits, total) = self
            .path_a_hits
            .iter()
            .fold((0usize, 0usize), |(h, t), deque| {
                let deque_hits = deque.iter().filter(|&&b| b).count();
                (h + deque_hits, t + deque.len())
            });
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Overall Path B accuracy (across all actions, rolling window).
    pub fn path_b_accuracy(&self) -> f64 {
        let (hits, total) = self
            .path_b_hits
            .iter()
            .fold((0usize, 0usize), |(h, t), deque| {
                let deque_hits = deque.iter().filter(|&&b| b).count();
                (h + deque_hits, t + deque.len())
            });
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Per-action Path A accuracy.
    pub fn path_a_accuracy_per_action(&self) -> Vec<f64> {
        self.path_a_hits
            .iter()
            .map(|deque| {
                if deque.is_empty() {
                    0.0
                } else {
                    deque.iter().filter(|&&b| b).count() as f64 / deque.len() as f64
                }
            })
            .collect()
    }

    /// Per-action Path B accuracy.
    pub fn path_b_accuracy_per_action(&self) -> Vec<f64> {
        self.path_b_hits
            .iter()
            .map(|deque| {
                if deque.is_empty() {
                    0.0
                } else {
                    deque.iter().filter(|&&b| b).count() as f64 / deque.len() as f64
                }
            })
            .collect()
    }

    /// Memory consolidation ratio: unique_tuples / total_stores.
    pub fn consolidation_ratio(&self) -> f64 {
        let total = self.memory.total_stores();
        if total == 0 {
            1.0
        } else {
            self.memory.unique_count() as f64 / total as f64
        }
    }

    /// Shannon entropy of action distribution.
    pub fn action_entropy(&self) -> f64 {
        let total: u64 = self.action_counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let mut entropy = 0.0;
        for &count in &self.action_counts {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Path advantage: accuracy_A - accuracy_B.
    pub fn path_advantage(&self) -> f64 {
        self.path_a_accuracy() - self.path_b_accuracy()
    }

    /// Current temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Average prediction confidence across unique (action, state_before) pairs.
    pub fn avg_prediction_confidence(&self) -> f64 {
        let mut seen: Vec<(u8, &Grid)> = Vec::new();
        let mut sum = 0.0;
        let mut count = 0usize;

        for tuple in self.memory.iter_tuples() {
            if !seen.iter().any(|(a, s)| *a == tuple.action && *s == &tuple.state_before) {
                seen.push((tuple.action, &tuple.state_before));
                sum += self.memory.prediction_confidence(tuple.action, &tuple.state_before);
                count += 1;
            }
        }

        if count == 0 { 1.0 } else { sum / count as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::MicroWorld;

    #[test]
    fn test_random_action_during_warmup() {
        let config = M0Config {
            warmup_episodes: 1000,
            seed: 42,
            ..M0Config::default()
        };
        let mut agent = Stage0Agent::new(config);
        let state = Grid::filled(5, 5, 0);

        // During warmup, actions should be distributed roughly uniformly
        let mut counts = [0u32; 4];
        for _ in 0..1000 {
            let a = agent.select_action(&state);
            counts[a as usize] += 1;
        }
        // Each action should appear at least 150 times out of 1000
        for &c in &counts {
            assert!(c > 150, "action count {} too low for uniform", c);
        }
    }

    #[test]
    fn test_accuracy_starts_zero() {
        let agent = Stage0Agent::new(M0Config::default());
        assert_eq!(agent.path_a_accuracy(), 0.0);
        assert_eq!(agent.path_b_accuracy(), 0.0);
    }

    #[test]
    fn test_record_updates_accuracy() {
        let mut agent = Stage0Agent::new(M0Config::default());
        let before = Grid::filled(5, 5, 0);
        let after = Grid::filled(5, 5, 0);

        agent.record(0, before.clone(), after.clone(), true, false);
        assert!(agent.path_a_accuracy() > 0.0);
        assert_eq!(agent.path_b_accuracy(), 0.0);
    }

    #[test]
    fn test_entropy_uniform() {
        let mut agent = Stage0Agent::new(M0Config::default());
        let state = Grid::filled(5, 5, 0);
        let after = Grid::filled(5, 5, 0);

        // Record equal counts for all actions
        for action in 0..4u8 {
            for _ in 0..25 {
                agent.record(action, state.clone(), after.clone(), false, false);
            }
        }
        let entropy = agent.action_entropy();
        // ln(4) ≈ 1.386
        assert!((entropy - 4.0f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_consolidation_ratio_decreases() {
        let mut agent = Stage0Agent::new(M0Config::default());
        let before = Grid::filled(5, 5, 0);
        let after = Grid::filled(5, 5, 0);

        // Same experience repeated
        for _ in 0..10 {
            agent.record(0, before.clone(), after.clone(), true, true);
        }
        // 1 unique tuple / 10 total stores = 0.1
        assert!((agent.consolidation_ratio() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_integration_with_world() {
        let config = M0Config {
            max_episodes: 200,
            warmup_episodes: 50,
            seed: 123,
            ..M0Config::default()
        };
        let mut world = MicroWorld::new(config.world_size);
        let mut agent = Stage0Agent::new(config);

        for _ in 0..200 {
            let state = world.observe();
            let action = agent.select_action(&state);
            let pred_a = agent.predict_path_a(action, &state);
            let pred_b = agent.predict_path_b(action);

            let _ = world.apply(action);
            let actual = world.observe();

            let hit_a = pred_a.as_ref() == Some(&actual);
            let hit_b = pred_b.as_ref() == Some(&actual);

            agent.record(action, state, actual, hit_a, hit_b);
        }

        // After 200 episodes in a 5x5 deterministic world,
        // Path A should be doing better than chance
        assert!(
            agent.path_a_accuracy() > 0.3,
            "Path A accuracy {} too low after 200 episodes",
            agent.path_a_accuracy()
        );
    }

    // ── Stage 1 tests ────────────────────────────────────────────────

    #[test]
    fn test_curiosity_zero_matches_stage0() {
        // curiosity_weight=0 should produce identical scores to Stage 0
        let config = M0Config {
            curiosity_weight: 0.0,
            warmup_episodes: 0,
            seed: 42,
            ..M0Config::default()
        };
        let mut agent = Stage0Agent::new(config);
        let state = Grid::filled(5, 5, 0);
        let after = Grid::filled(5, 5, 0);
        // Build some memory
        for action in 0..4u8 {
            for _ in 0..10 {
                agent.record(action, state.clone(), after.clone(), false, false);
            }
        }
        // All actions have equal familiarity, so with cw=0, distribution should be uniform-ish
        let mut counts = [0u32; 4];
        for _ in 0..400 {
            let a = agent.select_action(&state);
            counts[a as usize] += 1;
        }
        for &c in &counts {
            assert!(c > 50, "cw=0 action count {} too low for uniform-ish", c);
        }
    }

    #[test]
    fn test_adaptive_temp_gates_on_coverage() {
        let config = M0Config {
            adaptive_temperature: true,
            coverage_gate: 0.5,
            warmup_episodes: 0,
            temperature_init: 2.0,
            temperature_decay: 0.99,
            seed: 42,
            ..M0Config::default()
        };
        let mut agent = Stage0Agent::new(config);
        let state = Grid::filled(5, 5, 0);
        let after = Grid::filled(5, 5, 0);

        // Record a single tuple — coverage = 1/100 = 0.01 < 0.5
        agent.record(0, state.clone(), after.clone(), true, true);
        // Temperature should NOT have decayed (coverage too low)
        assert!(
            (agent.temperature() - 2.0).abs() < 0.01,
            "temp {} should be ~2.0 when coverage < gate",
            agent.temperature()
        );
    }

    #[test]
    fn test_stochastic_bootstrap() {
        let config = M0Config {
            max_episodes: 500,
            warmup_episodes: 100,
            noise: 0.2,
            seed: 42,
            ..M0Config::default()
        };
        let mut world = MicroWorld::with_noise(config.world_size, config.noise, config.seed);
        let mut agent = Stage0Agent::new(config);

        for _ in 0..500 {
            let state = world.observe();
            let action = agent.select_action(&state);
            let pred_a = agent.predict_path_a(action, &state);
            let pred_b = agent.predict_path_b(action);

            let _ = world.apply(action);
            let actual = world.observe();

            let hit_a = pred_a.as_ref() == Some(&actual);
            let hit_b = pred_b.as_ref() == Some(&actual);

            agent.record(action, state, actual, hit_a, hit_b);
        }

        assert!(
            agent.path_a_accuracy() > 0.3,
            "Path A accuracy {} too low with 20% noise after 500 episodes",
            agent.path_a_accuracy()
        );
    }
}
