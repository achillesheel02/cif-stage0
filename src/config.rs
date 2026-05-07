/// M_0 configuration — the tunable substrate.
///
/// Each parameter is a hypothesis about what the bootstrap needs.
/// Ablation: remove one, see what breaks.

#[derive(Debug, Clone)]
pub struct M0Config {
    /// Grid dimension (default 5 → 25 positions → 100 state-action pairs).
    pub world_size: usize,
    /// Number of actions (default 4: up/down/left/right).
    pub n_actions: usize,
    /// Episodes of pure random exploration before action bias kicks in.
    pub warmup_episodes: u64,
    /// Initial softmax temperature for action selection (higher = more random).
    pub temperature_init: f64,
    /// Per-episode temperature multiplier (< 1.0 = cooling).
    pub temperature_decay: f64,
    /// Print metrics every N episodes.
    pub log_interval: u64,
    /// Total episodes to run.
    pub max_episodes: u64,
    /// Strand checkpoint every N episodes.
    pub strand_interval: u64,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Rolling window size for accuracy tracking.
    pub accuracy_window: usize,
    /// Action noise probability [0.0, 1.0]. 0.0 = deterministic (Stage 0).
    pub noise: f64,
    /// Curiosity weight for action selection [0.0, 1.0]. 0.0 = pure familiarity (Stage 0).
    pub curiosity_weight: f64,
    /// Enable coverage-gated temperature decay. false = fixed decay (Stage 0).
    pub adaptive_temperature: bool,
    /// Coverage fraction required before temperature decay begins (only if adaptive_temperature).
    pub coverage_gate: f64,
    /// Enable hidden drift in the world (Stage 2).
    pub drift_enabled: bool,
    /// Steps per drift phase. 4 phases = 1 full cycle. Default 10 → 40-step cycle.
    pub drift_period: u64,
    /// Temporal context window size K. 0 = no temporal memory (Stage 0/1 compat).
    pub context_len: usize,
    /// Enable rule extraction (Stage 3).
    pub rules_enabled: bool,
    /// Re-extract rules every N episodes.
    pub rule_interval: u64,
}

impl Default for M0Config {
    fn default() -> Self {
        Self {
            world_size: 5,
            n_actions: 4,
            warmup_episodes: 100,
            temperature_init: 2.0,
            temperature_decay: 0.995,
            log_interval: 50,
            max_episodes: 5000,
            strand_interval: 50,
            seed: 42,
            accuracy_window: 100,
            noise: 0.0,
            curiosity_weight: 0.0,
            adaptive_temperature: false,
            coverage_gate: 0.5,
            drift_enabled: false,
            drift_period: 10,
            context_len: 0,
            rules_enabled: false,
            rule_interval: 100,
        }
    }
}
