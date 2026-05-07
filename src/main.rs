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
use cif_stage0::metrics::Metrics;
use cif_stage0::world::MicroWorld;

fn main() {
    let config = parse_args();

    println!("CIF Stage 0 — Bootstrap Experiment");
    println!("World: {}x{} | Actions: {} | Episodes: {} | Warmup: {} | Seed: {}",
        config.world_size, config.world_size, config.n_actions,
        config.max_episodes, config.warmup_episodes, config.seed);
    println!();

    let mut world = MicroWorld::new(config.world_size);
    let mut agent = Stage0Agent::new(config.clone());
    let mut metrics = Metrics::new();

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
            "--help" | "-h" => {
                println!("Usage: cif-stage0 [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --episodes, -n  Total episodes (default: 5000)");
                println!("  --size, -s      World grid size (default: 5)");
                println!("  --warmup, -w    Warmup episodes before action bias (default: 100)");
                println!("  --seed          RNG seed (default: 42)");
                println!("  --log-interval  Print metrics every N episodes (default: 50)");
                println!("  --temp          Initial temperature (default: 2.0)");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    config
}
