/// CIF Stage 0 — Bootstrap Experiment
///
/// The tightest possible loop: agent + environment + dual-path prediction.
/// Run, observe, learn what breaks.
///
/// Usage:
///   cargo run --release                    # default config
///   cargo run --release -- --episodes 10000  # more episodes
///   cargo run --release -- --size 3          # smaller world
///   cargo run --release -- --seed 99         # different RNG seed

use cif_stage0::agent::Stage0Agent;
use cif_stage0::config::M0Config;
use cif_stage0::grid::Grid;
use cif_stage0::metrics::Metrics;
use cif_stage0::world::MicroWorld;
use std::collections::VecDeque;

fn main() {
    let config = parse_args();

    let stage = if config.drift_enabled || config.context_len > 0 {
        "Stage 2 — Temporal Bootstrap"
    } else if config.noise > 0.0 || config.curiosity_weight > 0.0 || config.adaptive_temperature {
        "Stage 1 — Stochastic Bootstrap"
    } else {
        "Stage 0 — Bootstrap Experiment"
    };
    println!("CIF {} ", stage);
    println!("World: {}x{} | Actions: {} | Episodes: {} | Warmup: {} | Seed: {}",
        config.world_size, config.world_size, config.n_actions,
        config.max_episodes, config.warmup_episodes, config.seed);
    if config.noise > 0.0 || config.curiosity_weight > 0.0 || config.adaptive_temperature {
        println!("Noise: {:.2} | Curiosity: {:.2} | Adaptive temp: {} | Coverage gate: {:.2}",
            config.noise, config.curiosity_weight, config.adaptive_temperature, config.coverage_gate);
    }
    if config.drift_enabled || config.context_len > 0 {
        println!("Drift: {} | Drift period: {} | Context K: {}",
            config.drift_enabled, config.drift_period, config.context_len);
    }
    println!();

    let mut world = MicroWorld::with_drift(
        config.world_size, config.noise, config.seed,
        config.drift_enabled, config.drift_period,
    );
    let mut agent = Stage0Agent::new(config.clone());
    let mut metrics = Metrics::new();
    let mut history: VecDeque<(u8, Grid)> = VecDeque::new();

    for episode in 0..config.max_episodes {
        let state = world.observe();
        let context: Vec<(u8, Grid)> = history.iter().cloned().collect();
        let action = agent.select_action(&state, &context);

        // Three-way predictions BEFORE observing outcome
        let pred_t = agent.predict_temporal(action, &state, &context);
        let pred_a = agent.predict_path_a(action, &state);
        let pred_b = agent.predict_path_b(action);

        // Act
        let _executed = world.apply(action);
        let actual = world.observe();

        // Score predictions
        let hit_t = pred_t.as_ref() == Some(&actual);
        let hit_a = pred_a.as_ref() == Some(&actual);
        let hit_b = pred_b.as_ref() == Some(&actual);

        // Update history (BEFORE record consumes state)
        let state_for_history = state.clone();

        // Record experience
        agent.record(action, state, actual, hit_t, hit_a, hit_b, &context);

        // Maintain history buffer
        history.push_back((action, state_for_history));
        if history.len() > config.context_len {
            history.pop_front();
        }

        // Strand checkpoint (before metrics, so metrics can read it)
        if episode % config.strand_interval == 0 {
            metrics.strand_checkpoint(episode, &agent);
        }

        // Emit metrics
        if episode % config.log_interval == 0 {
            metrics.emit(episode, &agent);
        }
    }

    metrics.summary(&agent);
}

fn parse_args() -> M0Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = M0Config::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--episodes" | "-n" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.max_episodes = val.parse().unwrap_or(config.max_episodes);
                }
            }
            "--size" | "-s" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.world_size = val.parse().unwrap_or(config.world_size);
                }
            }
            "--warmup" | "-w" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.warmup_episodes = val.parse().unwrap_or(config.warmup_episodes);
                }
            }
            "--seed" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.seed = val.parse().unwrap_or(config.seed);
                }
            }
            "--log-interval" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.log_interval = val.parse().unwrap_or(config.log_interval);
                }
            }
            "--temp" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.temperature_init = val.parse().unwrap_or(config.temperature_init);
                }
            }
            "--noise" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.noise = val.parse().unwrap_or(config.noise);
                }
            }
            "--curiosity" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.curiosity_weight = val.parse().unwrap_or(config.curiosity_weight);
                }
            }
            "--adaptive-temp" => {
                config.adaptive_temperature = true;
            }
            "--coverage-gate" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.coverage_gate = val.parse().unwrap_or(config.coverage_gate);
                }
            }
            "--drift" => {
                config.drift_enabled = true;
            }
            "--drift-period" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.drift_period = val.parse().unwrap_or(config.drift_period);
                }
            }
            "--context" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.context_len = val.parse().unwrap_or(config.context_len);
                }
            }
            "--help" | "-h" => {
                println!("Usage: cif-stage0 [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --episodes, -n    Total episodes (default: 5000)");
                println!("  --size, -s        World grid size (default: 5)");
                println!("  --warmup, -w      Warmup episodes before action bias (default: 100)");
                println!("  --seed            RNG seed (default: 42)");
                println!("  --log-interval    Print metrics every N episodes (default: 50)");
                println!("  --temp            Initial temperature (default: 2.0)");
                println!("  --noise           Action noise probability [0.0-1.0] (default: 0.0)");
                println!("  --curiosity       Curiosity weight [0.0-1.0] (default: 0.0)");
                println!("  --adaptive-temp   Enable coverage-gated temperature decay");
                println!("  --coverage-gate   Coverage fraction before decay starts (default: 0.5)");
                println!("  --drift           Enable hidden drift (Stage 2)");
                println!("  --drift-period N  Steps per drift phase (default: 10, full cycle = 4N)");
                println!("  --context K       Temporal context window size (default: 0)");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    config
}
