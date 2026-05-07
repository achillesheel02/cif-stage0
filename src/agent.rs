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
use crate::other::OtherPolicy;
use crate::rules::RuleSet;
use crate::self_model::SelfMemory;
use crate::temporal::TemporalMemory;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

pub struct Stage0Agent {
    pub memory: ExperienceMemory,
    pub temporal_memory: TemporalMemory,
    pub rule_set: RuleSet,
    pub episode_count: u64,
    pub config: M0Config,
    rng: StdRng,
    /// Per-action rolling accuracy for Path R (rule-based).
    path_r_hits: Vec<VecDeque<bool>>,
    /// Per-action rolling accuracy for Path T (temporal context).
    path_t_hits: Vec<VecDeque<bool>>,
    /// Per-action rolling accuracy for Path A (memory-based).
    path_a_hits: Vec<VecDeque<bool>>,
    /// Per-action rolling accuracy for Path B (frequency-based).
    path_b_hits: Vec<VecDeque<bool>>,
    /// Per-action total selections (for entropy calculation).
    action_counts: Vec<u64>,
    /// Current temperature.
    temperature: f64,
    /// State-conditioned other-action observations: (state, action_B, count).
    other_observations: Vec<(Grid, u8, u32)>,
    /// Global frequency count of B's actions (Path O-B baseline).
    other_action_freq: Vec<u64>,
    /// Rolling accuracy for other-prediction (Path O-A).
    path_o_hits: Vec<VecDeque<bool>>,
    /// Rolling accuracy for other-frequency baseline (Path O-B).
    path_o_freq_hits: Vec<VecDeque<bool>>,
    /// Factored self-memory (Stage 5). None when self_model_enabled=false.
    self_memory: Option<SelfMemory>,
    /// Per-action rolling accuracy for Path S (self-memory).
    path_s_hits: Vec<VecDeque<bool>>,
    /// Rolling accuracy for best-path selection.
    best_path_hits: VecDeque<bool>,
}

impl Stage0Agent {
    pub fn new(config: M0Config) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let window = config.accuracy_window;
        let n = config.n_actions;
        let context_len = config.context_len;
        let world_size = config.world_size;
        let self_memory = if config.self_model_enabled {
            Some(SelfMemory::new())
        } else {
            None
        };
        Self {
            memory: ExperienceMemory::new(),
            temporal_memory: TemporalMemory::new(context_len),
            rule_set: RuleSet::new(world_size),
            episode_count: 0,
            temperature: config.temperature_init,
            config,
            rng,
            path_r_hits: (0..n).map(|_| VecDeque::with_capacity(window)).collect(),
            path_t_hits: (0..n).map(|_| VecDeque::with_capacity(window)).collect(),
            path_a_hits: (0..n).map(|_| VecDeque::with_capacity(window)).collect(),
            path_b_hits: (0..n).map(|_| VecDeque::with_capacity(window)).collect(),
            action_counts: vec![0; n],
            other_observations: Vec::new(),
            other_action_freq: vec![0; n],
            path_o_hits: (0..n).map(|_| VecDeque::with_capacity(window)).collect(),
            path_o_freq_hits: (0..n).map(|_| VecDeque::with_capacity(window)).collect(),
            self_memory,
            path_s_hits: (0..n).map(|_| VecDeque::with_capacity(window)).collect(),
            best_path_hits: VecDeque::with_capacity(window),
        }
    }

    /// Path R prediction: rule-based generative prediction.
    /// When rules_enabled=false, delegates to Path A (backward compat).
    pub fn predict_rule(&self, action: u8, state: &Grid) -> Option<Grid> {
        if !self.config.rules_enabled || self.rule_set.rule_count() == 0 {
            return self.predict_path_a(action, state);
        }
        self.rule_set.predict(action, state)
    }

    /// Path T prediction: temporal context-aware memory lookup.
    /// When context_len=0, delegates to Path A (backward compat).
    pub fn predict_temporal(&self, action: u8, state: &Grid, context: &[(u8, Grid)]) -> Option<Grid> {
        if self.config.context_len == 0 {
            return self.predict_path_a(action, state);
        }
        self.temporal_memory.retrieve(context, action, state).cloned()
    }

    /// Select an action given current state and context.
    ///
    /// Before warmup: uniform random.
    /// After warmup: softmax over familiarity scores (prediction-accuracy seeking).
    /// Context is accepted but not used for action selection (temporal informs prediction, not action choice).
    pub fn select_action(&mut self, state: &Grid, _context: &[(u8, Grid)]) -> u8 {
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
        path_r_hit: bool,
        path_t_hit: bool,
        path_a_hit: bool,
        path_b_hit: bool,
        context: &[(u8, Grid)],
        hit_s: bool,
        hit_best: bool,
    ) {
        // Store in all memories
        self.memory
            .store(action, state_before.clone(), state_after.clone());
        if let Some(ref mut sm) = self.self_memory {
            sm.store(action, state_before.clone(), state_after.clone());
        }
        self.temporal_memory
            .store(context, action, state_before, state_after);

        // Update action count
        let a = action as usize;
        if a < self.config.n_actions {
            self.action_counts[a] += 1;

            // Update rolling accuracy windows
            let window = self.config.accuracy_window;

            let hits_r = &mut self.path_r_hits[a];
            if hits_r.len() >= window {
                hits_r.pop_front();
            }
            hits_r.push_back(path_r_hit);

            let hits_t = &mut self.path_t_hits[a];
            if hits_t.len() >= window {
                hits_t.pop_front();
            }
            hits_t.push_back(path_t_hit);

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

            let hits_s = &mut self.path_s_hits[a];
            if hits_s.len() >= window {
                hits_s.pop_front();
            }
            hits_s.push_back(hit_s);
        }

        // Best-path rolling accuracy (not per-action)
        {
            let window = self.config.accuracy_window;
            if self.best_path_hits.len() >= window {
                self.best_path_hits.pop_front();
            }
            self.best_path_hits.push_back(hit_best);
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

    /// Overall Path T accuracy (across all actions, rolling window).
    pub fn path_t_accuracy(&self) -> f64 {
        let (hits, total) = self
            .path_t_hits
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

    /// Per-action Path T accuracy.
    pub fn path_t_accuracy_per_action(&self) -> Vec<f64> {
        self.path_t_hits
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

    /// Path advantage: accuracy_A - accuracy_B (memory value).
    pub fn path_advantage(&self) -> f64 {
        self.path_a_accuracy() - self.path_b_accuracy()
    }

    /// Temporal advantage: accuracy_T - accuracy_A (temporal context value).
    pub fn temporal_advantage(&self) -> f64 {
        self.path_t_accuracy() - self.path_a_accuracy()
    }

    /// Overall Path R accuracy (across all actions, rolling window).
    pub fn path_r_accuracy(&self) -> f64 {
        let (hits, total) = self
            .path_r_hits
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

    /// Per-action Path R accuracy.
    pub fn path_r_accuracy_per_action(&self) -> Vec<f64> {
        self.path_r_hits
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

    /// Rule advantage: accuracy_R - accuracy_A (value of symbolic compression).
    pub fn rule_advantage(&self) -> f64 {
        self.path_r_accuracy() - self.path_a_accuracy()
    }

    /// Extract rules from experience memory.
    pub fn extract_rules(&mut self) {
        self.rule_set.extract(&self.memory);
    }

    /// Current temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    // ── Stage 4: Theory of Mind ────────────────────────────────────────

    /// Predict B's next action from current state (Path O-A, state-conditioned).
    pub fn predict_other_action(&self, state: &Grid) -> Option<u8> {
        if self.config.other_policy == OtherPolicy::None || self.other_observations.is_empty() {
            return None;
        }

        // Exact match first
        let mut best_action: Option<u8> = None;
        let mut best_count: u32 = 0;
        for (s, a, c) in &self.other_observations {
            if s == state && *c > best_count {
                best_action = Some(*a);
                best_count = *c;
            }
        }
        if best_action.is_some() {
            return best_action;
        }

        // Approximate: find closest state, return its MLE action
        let mut min_dist = usize::MAX;
        for (s, a, c) in &self.other_observations {
            let d = state.hamming_distance(s);
            if d < min_dist || (d == min_dist && *c > best_count) {
                min_dist = d;
                best_action = Some(*a);
                best_count = *c;
            }
        }
        best_action
    }

    /// Predict B's most frequent action overall (Path O-B, frequency baseline).
    pub fn predict_other_freq(&self) -> Option<u8> {
        let total: u64 = self.other_action_freq.iter().sum();
        if total == 0 {
            return None;
        }
        let (best_idx, _) = self
            .other_action_freq
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)?;
        Some(best_idx as u8)
    }

    /// Record an observation of B's action.
    pub fn record_other(&mut self, state: &Grid, other_action: u8, hit_o: bool, hit_o_freq: bool, my_action: u8) {
        // Update frequency
        if (other_action as usize) < self.other_action_freq.len() {
            self.other_action_freq[other_action as usize] += 1;
        }

        // Dedup store
        let mut found = false;
        for (s, a, c) in &mut self.other_observations {
            if s == state && *a == other_action {
                *c += 1;
                found = true;
                break;
            }
        }
        if !found {
            self.other_observations.push((state.clone(), other_action, 1));
        }

        // Update rolling accuracy
        let a = my_action as usize;
        if a < self.config.n_actions {
            let window = self.config.accuracy_window;
            let hits = &mut self.path_o_hits[a];
            if hits.len() >= window {
                hits.pop_front();
            }
            hits.push_back(hit_o);

            let freq_hits = &mut self.path_o_freq_hits[a];
            if freq_hits.len() >= window {
                freq_hits.pop_front();
            }
            freq_hits.push_back(hit_o_freq);
        }
    }

    /// Path O-A accuracy (state-conditioned other-prediction).
    pub fn path_o_accuracy(&self) -> f64 {
        let (hits, total) = self
            .path_o_hits
            .iter()
            .fold((0usize, 0usize), |(h, t), deque| {
                let deque_hits = deque.iter().filter(|&&b| b).count();
                (h + deque_hits, t + deque.len())
            });
        if total == 0 { 0.0 } else { hits as f64 / total as f64 }
    }

    /// Path O-B accuracy (frequency baseline other-prediction).
    pub fn other_freq_accuracy(&self) -> f64 {
        let (hits, total) = self
            .path_o_freq_hits
            .iter()
            .fold((0usize, 0usize), |(h, t), deque| {
                let deque_hits = deque.iter().filter(|&&b| b).count();
                (h + deque_hits, t + deque.len())
            });
        if total == 0 { 0.0 } else { hits as f64 / total as f64 }
    }

    /// Xi_other = path_o_accuracy - other_freq_accuracy (value of state-conditioned model).
    pub fn other_advantage(&self) -> f64 {
        self.path_o_accuracy() - self.other_freq_accuracy()
    }

    /// Number of stored other-agent observations.
    pub fn other_observation_count(&self) -> usize {
        self.other_observations.len()
    }

    // ── Stage 5: Reflexive Self-Model ──────────────────────────────────

    /// Path S prediction: self-memory lookup (self-only grid).
    /// When self_model_enabled=false, returns None (no data).
    pub fn predict_self(&self, action: u8, state: &Grid) -> Option<Grid> {
        self.self_memory.as_ref()?.retrieve(action, state).cloned()
    }

    /// Overall Path S accuracy (self-memory, rolling window).
    pub fn path_s_accuracy(&self) -> f64 {
        let (hits, total) = self
            .path_s_hits
            .iter()
            .fold((0usize, 0usize), |(h, t), deque| {
                let deque_hits = deque.iter().filter(|&&b| b).count();
                (h + deque_hits, t + deque.len())
            });
        if total == 0 { 0.0 } else { hits as f64 / total as f64 }
    }

    /// Xi_self = path_s - path_a (value of factored self-model).
    pub fn self_advantage(&self) -> f64 {
        self.path_s_accuracy() - self.path_a_accuracy()
    }

    /// Select the best prediction from all available paths.
    /// Uses rolling accuracy to pick the most trusted path.
    pub fn best_prediction(&self, action: u8, state: &Grid, context: &[(u8, Grid)]) -> (Option<Grid>, &'static str) {
        let candidates: Vec<(f64, Option<Grid>, &'static str)> = vec![
            (self.path_r_accuracy(), self.predict_rule(action, state), "R"),
            (self.path_t_accuracy(), self.predict_temporal(action, state, context), "T"),
            (self.path_s_accuracy(), self.predict_self(action, state), "S"),
            (self.path_a_accuracy(), self.predict_path_a(action, state), "A"),
            (self.path_b_accuracy(), self.predict_path_b(action), "B"),
        ];

        candidates
            .into_iter()
            .filter(|(_, pred, _)| pred.is_some())
            .max_by(|(a, _, _), (b, _, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, pred, name)| (pred, name))
            .unwrap_or((None, "A"))
    }

    /// Best-path rolling accuracy.
    pub fn best_path_accuracy(&self) -> f64 {
        if self.best_path_hits.is_empty() {
            0.0
        } else {
            let hits = self.best_path_hits.iter().filter(|&&b| b).count();
            hits as f64 / self.best_path_hits.len() as f64
        }
    }

    /// Xi_reflexive = best_path - path_a (value of meta-selection).
    pub fn reflexive_advantage(&self) -> f64 {
        self.best_path_accuracy() - self.path_a_accuracy()
    }

    /// Self-memory unique tuple count (0 if disabled).
    pub fn self_unique_count(&self) -> usize {
        self.self_memory.as_ref().map_or(0, |sm| sm.unique_count())
    }

    /// Self-memory consolidation ratio.
    pub fn self_consolidation_ratio(&self) -> f64 {
        match &self.self_memory {
            Some(sm) => {
                let total = sm.total_stores();
                if total == 0 { 1.0 } else { sm.unique_count() as f64 / total as f64 }
            }
            None => 0.0,
        }
    }

    /// Compare predicted vs actual on self-marker only (strip color=2).
    pub fn self_hit(predicted: &Option<Grid>, actual: &Grid) -> bool {
        match predicted {
            Some(p) => p.strip_color(2) == actual.strip_color(2),
            None => false,
        }
    }

    // ── Stage 6: Goal-Directed Planning ─────────────────────────────────

    /// Predict for planning rollouts.
    /// Priority: Rules (generative, works on imagined states) > Self-model > Memory.
    /// Excludes temporal (needs real context, unavailable during rollout).
    pub fn predict_for_planning(&self, action: u8, state: &Grid) -> Option<Grid> {
        if self.config.rules_enabled && self.rule_set.rule_count() > 0 {
            if let Some(pred) = self.rule_set.predict(action, state) {
                return Some(pred);
            }
        }
        if let Some(ref sm) = self.self_memory {
            if let Some(pred) = sm.retrieve(action, state) {
                return Some(pred.clone());
            }
        }
        self.memory.retrieve(action, state).cloned()
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
            let a = agent.select_action(&state, &[]);
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

        agent.record(0, before.clone(), after.clone(), true, true, true, false, &[], false, false);
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
                agent.record(action, state.clone(), after.clone(), false, false, false, false, &[], false, false);
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
            agent.record(0, before.clone(), after.clone(), true, true, true, true, &[], false, false);
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
            let action = agent.select_action(&state, &[]);
            let pred_a = agent.predict_path_a(action, &state);
            let pred_b = agent.predict_path_b(action);

            let _ = world.apply(action);
            let actual = world.observe();

            let hit_a = pred_a.as_ref() == Some(&actual);
            let hit_b = pred_b.as_ref() == Some(&actual);

            agent.record(action, state, actual, hit_a, hit_a, hit_a, hit_b, &[], false, false);
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
                agent.record(action, state.clone(), after.clone(), false, false, false, false, &[], false, false);
            }
        }
        // All actions have equal familiarity, so with cw=0, distribution should be uniform-ish
        let mut counts = [0u32; 4];
        for _ in 0..400 {
            let a = agent.select_action(&state, &[]);
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
        agent.record(0, state.clone(), after.clone(), true, true, true, true, &[], false, false);
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
            let action = agent.select_action(&state, &[]);
            let pred_a = agent.predict_path_a(action, &state);
            let pred_b = agent.predict_path_b(action);

            let _ = world.apply(action);
            let actual = world.observe();

            let hit_a = pred_a.as_ref() == Some(&actual);
            let hit_b = pred_b.as_ref() == Some(&actual);

            agent.record(action, state, actual, hit_a, hit_a, hit_a, hit_b, &[], false, false);
        }

        assert!(
            agent.path_a_accuracy() > 0.3,
            "Path A accuracy {} too low with 20% noise after 500 episodes",
            agent.path_a_accuracy()
        );
    }

    // ── Stage 2 tests ────────────────────────────────────────────────

    #[test]
    fn test_context_zero_t_equals_a() {
        // With context_len=0, Path T should delegate to Path A
        let config = M0Config {
            context_len: 0,
            warmup_episodes: 0,
            seed: 42,
            ..M0Config::default()
        };
        let mut agent = Stage0Agent::new(config);
        let before = Grid::filled(5, 5, 0);
        let mut after = Grid::filled(5, 5, 0);
        after.set(0, 0, 1);

        agent.record(0, before.clone(), after.clone(), true, true, true, false, &[], false, false);

        let pred_a = agent.predict_path_a(0, &before);
        let pred_t = agent.predict_temporal(0, &before, &[]);
        assert_eq!(pred_a, pred_t);
    }

    #[test]
    fn test_four_way_tracking() {
        let config = M0Config {
            context_len: 1,
            warmup_episodes: 0,
            seed: 42,
            ..M0Config::default()
        };
        let mut agent = Stage0Agent::new(config);
        let state = Grid::filled(5, 5, 0);
        let after = Grid::filled(5, 5, 0);

        // Record with all four paths having different outcomes
        agent.record(0, state.clone(), after.clone(), true, false, false, false, &[], false, false);
        agent.record(0, state.clone(), after.clone(), false, true, false, false, &[], false, false);
        agent.record(0, state.clone(), after.clone(), false, false, true, false, &[], false, false);
        agent.record(0, state.clone(), after.clone(), false, false, false, true, &[], false, false);

        // path_r: 1/4, path_t: 1/4, path_a: 1/4, path_b: 1/4
        assert!((agent.path_r_accuracy() - 0.25).abs() < 0.01);
        assert!((agent.path_t_accuracy() - 0.25).abs() < 0.01);
        assert!((agent.path_a_accuracy() - 0.25).abs() < 0.01);
        assert!((agent.path_b_accuracy() - 0.25).abs() < 0.01);
        assert!(agent.temporal_advantage().abs() < 0.01);
        assert!(agent.rule_advantage().abs() < 0.01);
    }

    // ── Stage 3 tests ────────────────────────────────────────────────

    #[test]
    fn test_rules_disabled_r_equals_a() {
        let config = M0Config {
            rules_enabled: false,
            warmup_episodes: 0,
            seed: 42,
            ..M0Config::default()
        };
        let mut agent = Stage0Agent::new(config);
        let mut before = Grid::filled(5, 5, 0);
        before.set(2, 2, 1);
        let mut after = Grid::filled(5, 5, 0);
        after.set(1, 2, 1);

        agent.record(0, before.clone(), after.clone(), true, true, true, false, &[], false, false);

        let pred_r = agent.predict_rule(0, &before);
        let pred_a = agent.predict_path_a(0, &before);
        assert_eq!(pred_r, pred_a);
    }

    // ── Stage 4 tests ────────────────────────────────────────────────

    #[test]
    fn test_predict_other_none_without_other() {
        let agent = Stage0Agent::new(M0Config::default());
        let state = Grid::filled(5, 5, 0);
        assert_eq!(agent.predict_other_action(&state), None);
    }

    #[test]
    fn test_predict_other_learns_fixed() {
        let mut config = M0Config::default();
        config.other_policy = crate::other::OtherPolicy::Fixed;
        let mut agent = Stage0Agent::new(config);
        let state = Grid::filled(5, 5, 0);

        // Record B always going UP (action 0) in this state
        for _ in 0..10 {
            agent.record_other(&state, 0, true, true, 0);
        }

        let pred = agent.predict_other_action(&state);
        assert_eq!(pred, Some(0)); // Should predict UP
    }

    #[test]
    fn test_self_hit_ignores_b() {
        let mut pred = Grid::filled(5, 5, 0);
        pred.set(2, 2, 1);
        pred.set(3, 3, 2); // B at wrong position

        let mut actual = Grid::filled(5, 5, 0);
        actual.set(2, 2, 1);
        actual.set(4, 4, 2); // B at different position

        // Self markers match (both at 2,2), so self_hit should be true
        assert!(Stage0Agent::self_hit(&Some(pred), &actual));
    }

    #[test]
    fn test_path_o_tracking() {
        let mut config = M0Config::default();
        config.other_policy = crate::other::OtherPolicy::Fixed;
        let mut agent = Stage0Agent::new(config);
        let state = Grid::filled(5, 5, 0);

        agent.record_other(&state, 0, true, true, 0);
        agent.record_other(&state, 0, true, false, 0);
        agent.record_other(&state, 0, false, false, 0);

        // 2/3 hits for path_o, 1/3 for freq
        assert!((agent.path_o_accuracy() - 2.0 / 3.0).abs() < 0.01);
        assert!((agent.other_freq_accuracy() - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_rules_generalize_to_unseen_state() {
        let config = M0Config {
            rules_enabled: true,
            warmup_episodes: 0,
            seed: 42,
            ..M0Config::default()
        };
        let mut agent = Stage0Agent::new(config);

        // Teach: UP from (2,2) → (1,2)
        let mut before = Grid::filled(5, 5, 0);
        before.set(2, 2, 1);
        let mut after = Grid::filled(5, 5, 0);
        after.set(1, 2, 1);
        agent.record(0, before.clone(), after.clone(), true, true, true, false, &[], false, false);

        // Extract rules
        agent.extract_rules();

        // Predict UP from (4, 0) — never seen in memory
        let mut unseen = Grid::filled(5, 5, 0);
        unseen.set(4, 0, 1);
        let mut expected = Grid::filled(5, 5, 0);
        expected.set(3, 0, 1);

        let pred_r = agent.predict_rule(0, &unseen);
        let pred_a = agent.predict_path_a(0, &unseen);

        // Rules should predict correctly via generalization
        assert_eq!(pred_r, Some(expected.clone()));
        // Memory retrieval returns approximate match (not the correct answer)
        assert_ne!(pred_a, Some(expected));
    }

    // ── Stage 5 tests ────────────────────────────────────────────────

    #[test]
    fn test_self_model_disabled_no_effect() {
        let config = M0Config {
            self_model_enabled: false,
            ..M0Config::default()
        };
        let agent = Stage0Agent::new(config);
        assert_eq!(agent.path_s_accuracy(), 0.0);
        assert_eq!(agent.self_unique_count(), 0);
        assert!(agent.predict_self(0, &Grid::filled(5, 5, 0)).is_none());
    }

    #[test]
    fn test_self_model_strips_other() {
        let config = M0Config {
            self_model_enabled: true,
            other_policy: OtherPolicy::Random,
            warmup_episodes: 0,
            seed: 42,
            ..M0Config::default()
        };
        let mut agent = Stage0Agent::new(config);

        // Store: self at (2,2)→(1,2), other at (4,4)
        let mut before = Grid::filled(5, 5, 0);
        before.set(2, 2, 1);
        before.set(4, 4, 2);
        let mut after = Grid::filled(5, 5, 0);
        after.set(1, 2, 1);
        after.set(4, 3, 2);

        agent.record(0, before.clone(), after.clone(), true, true, true, false, &[], true, true);

        // Query with same self-pos but DIFFERENT other-pos should still match
        let mut query = Grid::filled(5, 5, 0);
        query.set(2, 2, 1);
        query.set(0, 0, 2); // other in completely different spot

        let pred = agent.predict_self(0, &query);
        assert!(pred.is_some());
        // Result should be self-only (no color=2)
        assert_eq!(pred.unwrap().find_other(), None);
    }

    #[test]
    fn test_best_prediction_selects_highest() {
        let config = M0Config {
            self_model_enabled: true,
            warmup_episodes: 0,
            seed: 42,
            ..M0Config::default()
        };
        let mut agent = Stage0Agent::new(config);
        let state = Grid::filled(5, 5, 0);
        let after = Grid::filled(5, 5, 0);

        // Feed hits to Path S only (hit_s=true, path_a=false, path_b=false)
        for _ in 0..10 {
            agent.record(0, state.clone(), after.clone(), false, false, false, false, &[], true, true);
        }

        // Path S should have highest accuracy
        assert!(agent.path_s_accuracy() > agent.path_a_accuracy());

        let (_, name) = agent.best_prediction(0, &state, &[]);
        assert_eq!(name, "S");
    }
}
