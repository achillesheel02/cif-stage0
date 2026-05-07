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
        }
    }
}
