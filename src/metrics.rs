/// Instrumentation for Stage 0 — the 4 core metrics + strand checkpoints.
///
/// This is the debugger for M_0. Every number here answers a question
/// about whether the bootstrap is working.

use crate::agent::Stage0Agent;
use strand_core::{Config, StrandMatrix};

/// A single metrics snapshot.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub episode: u64,
    pub accuracy: f64,
    pub consolidation: f64,
    pub entropy: f64,
    pub path_advantage: f64,
    pub temperature: f64,
    pub unique_tuples: usize,
    pub confidence: f64,
}

/// A strand checkpoint — dual-path convergence across actions.
#[derive(Debug, Clone)]
pub struct StrandCheckpoint {
    pub episode: u64,
    pub frobenius: f64,
    pub converged_count: usize,
    pub applicable_cells: usize,
    pub gap_class: String,
    pub glyph_string: String,
}

pub struct Metrics {
    pub snapshots: Vec<Snapshot>,
    pub strand_checkpoints: Vec<StrandCheckpoint>,
    header_printed: bool,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            strand_checkpoints: Vec::new(),
            header_printed: false,
        }
    }

    /// Emit a metrics snapshot to stdout.
    pub fn emit(&mut self, episode: u64, agent: &Stage0Agent) {
        let snap = Snapshot {
            episode,
            accuracy: agent.path_a_accuracy(),
            consolidation: agent.consolidation_ratio(),
            entropy: agent.action_entropy(),
            path_advantage: agent.path_advantage(),
            temperature: agent.temperature(),
            unique_tuples: agent.memory.unique_count(),
            confidence: agent.avg_prediction_confidence(),
        };

        if !self.header_printed {
            println!(
                "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
                "episode", "acc_A", "acc_B", "consol", "entropy", "path_adv", "temp", "conf", "frobenius", "gap"
            );
            println!("{}", "-".repeat(100));
            self.header_printed = true;
        }

        // Find matching strand checkpoint if any
        let (frob_str, gap_str) = self
            .strand_checkpoints
            .iter()
            .rev()
            .find(|sc| sc.episode == episode)
            .map(|sc| (format!("{:.3}", sc.frobenius), sc.gap_class.clone()))
            .unwrap_or_else(|| ("-".into(), "-".into()));

        println!(
            "{:<8} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10}",
            snap.episode,
            snap.accuracy,
            agent.path_b_accuracy(),
            snap.consolidation,
            snap.entropy,
            snap.path_advantage,
            snap.temperature,
            snap.confidence,
            frob_str,
            gap_str,
        );

        self.snapshots.push(snap);
    }

    /// Build a strand checkpoint — dual-path convergence per action.
    pub fn strand_checkpoint(&mut self, episode: u64, agent: &Stage0Agent) {
        let acc_a = agent.path_a_accuracy_per_action();
        let acc_b = agent.path_b_accuracy_per_action();

        let n_actions = acc_a.len();
        if n_actions == 0 {
            return;
        }

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

        let checkpoint = StrandCheckpoint {
            episode,
            frobenius: frob.normalised_applicable,
            converged_count: frob.converged_count,
            applicable_cells: frob.applicable_cells,
            gap_class: gap.label().to_string(),
            glyph_string: matrix.render().trim().to_string(),
        };

        self.strand_checkpoints.push(checkpoint);
    }

    /// Print a summary at the end of the run.
    pub fn summary(&self, agent: &Stage0Agent) {
        println!("\n{}", "=".repeat(90));
        println!("STAGE 0 SUMMARY");
        println!("{}", "=".repeat(90));
        println!(
            "Total episodes:        {}",
            agent.episode_count
        );
        println!(
            "Final Path A accuracy: {:.3}",
            agent.path_a_accuracy()
        );
        println!(
            "Final Path B accuracy: {:.3}",
            agent.path_b_accuracy()
        );
        println!(
            "Path advantage:        {:.3}",
            agent.path_advantage()
        );
        println!(
            "Action entropy:        {:.3} (max: {:.3})",
            agent.action_entropy(),
            (agent.config.n_actions as f64).ln()
        );
        println!(
            "Unique tuples:         {}",
            agent.memory.unique_count()
        );
        println!(
            "Total stores:          {}",
            agent.memory.total_stores()
        );
        println!(
            "Consolidation ratio:   {:.3}",
            agent.consolidation_ratio()
        );
        println!(
            "Final temperature:     {:.4}",
            agent.temperature()
        );

        // Strand summary
        if let Some(last) = self.strand_checkpoints.last() {
            println!(
                "\nFinal strand: {} | frobenius={:.3} | {}/{} converged | gap={}",
                last.glyph_string,
                last.frobenius,
                last.converged_count,
                last.applicable_cells,
                last.gap_class,
            );
        }

        // Diagnosis
        println!("\nDIAGNOSIS:");
        let acc = agent.path_a_accuracy();
        let adv = agent.path_advantage();
        let ent = agent.action_entropy();
        let max_ent = (agent.config.n_actions as f64).ln();

        if acc > 0.8 && adv > 0.0 {
            println!("  BOOTSTRAP SUCCESS — predictions well above chance, Path A > Path B.");
            println!("  M_0 is sufficient for this environment. Ready for Stage 1.");
        } else if acc > 0.5 && adv > 0.0 {
            println!("  PARTIAL BOOTSTRAP — predictions above chance but not saturated.");
            println!("  Memory structure helps (Path A > B). May need more episodes or richer retrieval.");
        } else if acc < 0.3 {
            println!("  BOOTSTRAP FAILED — predictions near chance.");
            println!("  Check: memory retrieval, similarity metric, or write policy.");
        }

        if adv < 0.0 {
            println!("  WARNING: Path B beats Path A — memory structure is worse than frequency.");
            println!("  Retrieval policy may be fundamentally wrong.");
        }

        let conf = agent.avg_prediction_confidence();
        if conf < 0.95 {
            println!(
                "  STOCHASTIC ENVIRONMENT — avg confidence {:.1}% indicates noisy transitions.",
                conf * 100.0
            );
            if acc > 0.5 {
                println!("  Path A accuracy {:.1}% despite noise suggests M_0 generalises.", acc * 100.0);
            }
        }

        if ent < max_ent * 0.3 && agent.episode_count < agent.config.warmup_episodes * 2 {
            println!("  WARNING: Entropy collapsed early — possible degenerate strategy.");
            println!("  Check: edge-clamping + predictability-seeking = corner-sitting?");
        }

        if agent.consolidation_ratio() > 0.9 && agent.episode_count > 500 {
            println!("  WARNING: No consolidation after 500 episodes.");
            println!("  Distinction primitives may not be grouping experiences.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_metrics_new() {
        let m = Metrics::new();
        assert!(m.snapshots.is_empty());
        assert!(m.strand_checkpoints.is_empty());
    }
}
