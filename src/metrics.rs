/// Instrumentation for Stage 0 — the 4 core metrics + strand checkpoints.
///
/// This is the debugger for M_0. Every number here answers a question
/// about whether the bootstrap is working.

use crate::agent::Stage0Agent;
use crate::other::OtherPolicy;
use crate::world::MicroWorld;
use strand_core::{Config, StrandMatrix};

/// A single metrics snapshot.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub episode: u64,
    pub accuracy_r: f64,
    pub accuracy_t: f64,
    pub accuracy: f64,
    pub consolidation: f64,
    pub entropy: f64,
    pub path_advantage: f64,
    pub xi_rule: f64,
    pub xi_temporal: f64,
    pub xi_total: f64,
    pub temperature: f64,
    pub unique_tuples: usize,
    pub temporal_tuples: usize,
    pub rule_count: usize,
    pub rule_confidence: f64,
    pub confidence: f64,
    pub accuracy_o: f64,
    pub xi_other: f64,
    pub self_accuracy: f64,
    pub accuracy_s: f64,
    pub xi_self: f64,
    pub best_path_accuracy: f64,
    pub xi_reflexive: f64,
    pub self_unique_tuples: usize,
    pub self_consolidation: f64,
    pub goals_reached: u64,
    pub avg_steps_per_goal: f64,
    pub navigation_efficiency: f64,
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
    pub fn emit(&mut self, episode: u64, agent: &Stage0Agent, world: &MicroWorld) {
        let snap = Snapshot {
            episode,
            accuracy_r: agent.path_r_accuracy(),
            accuracy_t: agent.path_t_accuracy(),
            accuracy: agent.path_a_accuracy(),
            consolidation: agent.consolidation_ratio(),
            entropy: agent.action_entropy(),
            path_advantage: agent.path_advantage(),
            xi_rule: agent.rule_advantage(),
            xi_temporal: agent.temporal_advantage(),
            xi_total: agent.path_r_accuracy() - agent.path_b_accuracy(),
            temperature: agent.temperature(),
            unique_tuples: agent.memory.unique_count(),
            temporal_tuples: agent.temporal_memory.unique_count(),
            rule_count: agent.rule_set.rule_count(),
            rule_confidence: agent.rule_set.avg_confidence(),
            confidence: agent.avg_prediction_confidence(),
            accuracy_o: agent.path_o_accuracy(),
            xi_other: agent.other_advantage(),
            self_accuracy: 0.0,
            accuracy_s: agent.path_s_accuracy(),
            xi_self: agent.self_advantage(),
            best_path_accuracy: agent.best_path_accuracy(),
            xi_reflexive: agent.reflexive_advantage(),
            self_unique_tuples: agent.self_unique_count(),
            self_consolidation: agent.self_consolidation_ratio(),
            goals_reached: world.goals_reached(),
            avg_steps_per_goal: world.avg_steps_per_goal(),
            navigation_efficiency: world.avg_navigation_efficiency(),
        };

        if !self.header_printed {
            if agent.config.adaptive_strategy {
                println!(
                    "{:<8} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
                    "episode", "goals", "effic", "acc_R", "acc_A", "plan%", "greedy%", "expl%",
                    "consol", "entropy", "gap"
                );
                println!("{}", "-".repeat(110));
            } else if agent.config.goal_enabled {
                println!(
                    "{:<8} {:>6} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
                    "episode", "goals", "avg_steps", "effic", "acc_R", "acc_A", "acc_B",
                    "consol", "entropy", "frobenius", "gap"
                );
                println!("{}", "-".repeat(115));
            } else if agent.config.self_model_enabled {
                println!(
                    "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
                    "episode", "acc_S", "acc_A", "acc_B", "xi_self", "xi_refl", "best_acc",
                    "consol_S", "consol_A", "entropy", "frobenius", "gap"
                );
                println!("{}", "-".repeat(120));
            } else if agent.config.other_policy != OtherPolicy::None {
                println!(
                    "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
                    "episode", "acc_A", "acc_B", "acc_O", "acc_Ofrq", "xi_other", "xi_mem",
                    "consol", "entropy", "frobenius", "gap"
                );
                println!("{}", "-".repeat(110));
            } else if agent.config.rules_enabled {
                println!(
                    "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
                    "episode", "acc_R", "acc_T", "acc_A", "acc_B", "xi_rule", "xi_temp", "xi_mem",
                    "consol", "entropy", "temp", "frobenius", "gap"
                );
                println!("{}", "-".repeat(130));
            } else if agent.config.context_len > 0 {
                println!(
                    "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
                    "episode", "acc_T", "acc_A", "acc_B", "xi_temp", "xi_mem", "xi_tot",
                    "consol", "entropy", "temp", "frobenius", "gap"
                );
                println!("{}", "-".repeat(120));
            } else {
                println!(
                    "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
                    "episode", "acc_A", "acc_B", "consol", "entropy", "path_adv", "temp", "conf", "frobenius", "gap"
                );
                println!("{}", "-".repeat(100));
            }
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

        if agent.config.adaptive_strategy {
            let (pc, gc, ec) = agent.strategy_counts();
            let total = (pc + gc + ec).max(1) as f64;
            println!(
                "{:<8} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>7.1}% {:>7.1}% {:>7.1}% {:>8.3} {:>8.3} {:>10}",
                snap.episode,
                snap.goals_reached,
                snap.navigation_efficiency,
                snap.accuracy_r,
                snap.accuracy,
                100.0 * pc as f64 / total,
                100.0 * gc as f64 / total,
                100.0 * ec as f64 / total,
                snap.consolidation,
                snap.entropy,
                gap_str,
            );
        } else if agent.config.goal_enabled {
            println!(
                "{:<8} {:>6} {:>10.1} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10}",
                snap.episode,
                snap.goals_reached,
                snap.avg_steps_per_goal,
                snap.navigation_efficiency,
                snap.accuracy_r,
                snap.accuracy,
                agent.path_b_accuracy(),
                snap.consolidation,
                snap.entropy,
                frob_str,
                gap_str,
            );
        } else if agent.config.self_model_enabled {
            println!(
                "{:<8} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10}",
                snap.episode,
                snap.accuracy_s,
                snap.accuracy,
                agent.path_b_accuracy(),
                snap.xi_self,
                snap.xi_reflexive,
                snap.best_path_accuracy,
                snap.self_consolidation,
                snap.consolidation,
                snap.entropy,
                frob_str,
                gap_str,
            );
        } else if agent.config.other_policy != OtherPolicy::None {
            println!(
                "{:<8} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10}",
                snap.episode,
                snap.accuracy,
                agent.path_b_accuracy(),
                snap.accuracy_o,
                agent.other_freq_accuracy(),
                snap.xi_other,
                snap.path_advantage,
                snap.consolidation,
                snap.entropy,
                frob_str,
                gap_str,
            );
        } else if agent.config.rules_enabled {
            println!(
                "{:<8} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10}",
                snap.episode,
                snap.accuracy_r,
                snap.accuracy_t,
                snap.accuracy,
                agent.path_b_accuracy(),
                snap.xi_rule,
                snap.xi_temporal,
                snap.path_advantage,
                snap.consolidation,
                snap.entropy,
                snap.temperature,
                frob_str,
                gap_str,
            );
        } else if agent.config.context_len > 0 {
            println!(
                "{:<8} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10}",
                snap.episode,
                snap.accuracy_t,
                snap.accuracy,
                agent.path_b_accuracy(),
                snap.xi_temporal,
                snap.path_advantage,
                snap.xi_total,
                snap.consolidation,
                snap.entropy,
                snap.temperature,
                frob_str,
                gap_str,
            );
        } else {
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
        }

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
    pub fn summary(&self, agent: &Stage0Agent, world: &MicroWorld) {
        let stage_label = if agent.config.goal_enabled {
            "STAGE 6 SUMMARY"
        } else if agent.config.self_model_enabled {
            "STAGE 5 SUMMARY"
        } else if agent.config.other_policy != OtherPolicy::None {
            "STAGE 4 SUMMARY"
        } else if agent.config.rules_enabled {
            "STAGE 3 SUMMARY"
        } else if agent.config.context_len > 0 || agent.config.drift_enabled {
            "STAGE 2 SUMMARY"
        } else if agent.config.noise > 0.0 || agent.config.curiosity_weight > 0.0 || agent.config.adaptive_temperature {
            "STAGE 1 SUMMARY"
        } else {
            "STAGE 0 SUMMARY"
        };
        println!("\n{}", "=".repeat(90));
        println!("{}", stage_label);
        println!("{}", "=".repeat(90));
        println!(
            "Total episodes:        {}",
            agent.episode_count
        );

        if agent.config.rules_enabled {
            println!(
                "Final Path R accuracy: {:.3}",
                agent.path_r_accuracy()
            );
        }
        if agent.config.context_len > 0 {
            println!(
                "Final Path T accuracy: {:.3}",
                agent.path_t_accuracy()
            );
        }
        println!(
            "Final Path A accuracy: {:.3}",
            agent.path_a_accuracy()
        );
        println!(
            "Final Path B accuracy: {:.3}",
            agent.path_b_accuracy()
        );
        if agent.config.rules_enabled {
            println!(
                "Xi rule (R-A):         {:.3}",
                agent.rule_advantage()
            );
        }
        if agent.config.context_len > 0 {
            println!(
                "Xi temporal (T-A):     {:.3}",
                agent.temporal_advantage()
            );
        }
        if agent.config.rules_enabled || agent.config.context_len > 0 {
            println!(
                "Xi memory (A-B):       {:.3}",
                agent.path_advantage()
            );
        }
        if agent.config.rules_enabled {
            println!(
                "Xi total (R-B):        {:.3}",
                agent.path_r_accuracy() - agent.path_b_accuracy()
            );
        } else if agent.config.context_len > 0 {
            println!(
                "Xi total (T-B):        {:.3}",
                agent.path_t_accuracy() - agent.path_b_accuracy()
            );
        } else {
            println!(
                "Path advantage:        {:.3}",
                agent.path_advantage()
            );
        }
        if agent.config.rules_enabled {
            println!(
                "Rules extracted:       {}",
                agent.rule_set.rule_count()
            );
            println!(
                "Rule confidence:       {:.3}",
                agent.rule_set.avg_confidence()
            );
        }
        if agent.config.other_policy != OtherPolicy::None {
            println!(
                "Path O accuracy:       {:.3}",
                agent.path_o_accuracy()
            );
            println!(
                "Path O-freq accuracy:  {:.3}",
                agent.other_freq_accuracy()
            );
            println!(
                "Xi other (O-Ofreq):    {:.3}",
                agent.other_advantage()
            );
            println!(
                "Other observations:    {}",
                agent.other_observation_count()
            );
        }
        if agent.config.self_model_enabled {
            println!(
                "Path S accuracy:       {:.3}",
                agent.path_s_accuracy()
            );
            println!(
                "Xi self (S-A):         {:.3}",
                agent.self_advantage()
            );
            println!(
                "Best-path accuracy:    {:.3}",
                agent.best_path_accuracy()
            );
            println!(
                "Xi reflexive (best-A): {:.3}",
                agent.reflexive_advantage()
            );
            println!(
                "Self-memory tuples:    {}",
                agent.self_unique_count()
            );
            println!(
                "Self-memory consol:    {:.3}",
                agent.self_consolidation_ratio()
            );
        }
        println!(
            "Action entropy:        {:.3} (max: {:.3})",
            agent.action_entropy(),
            (agent.config.n_actions as f64).ln()
        );
        println!(
            "Unique tuples:         {}",
            agent.memory.unique_count()
        );
        if agent.config.context_len > 0 {
            println!(
                "Temporal tuples:       {}",
                agent.temporal_memory.unique_count()
            );
        }
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
            if agent.config.context_len == 0 {
                println!("  M_0 is sufficient for this environment. Ready for Stage 1.");
            }
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

        // Temporal diagnosis
        if agent.config.context_len > 0 {
            let xi_t = agent.temporal_advantage();
            if xi_t > 0.05 && agent.config.drift_enabled {
                println!(
                    "  TEMPORAL PATTERN DETECTED — Xi_temporal={:.3}, context K={} captures drift.",
                    xi_t, agent.config.context_len
                );
            }
            if agent.config.drift_enabled && xi_t < 0.02 {
                println!("  WARNING: drift present but temporal memory shows no advantage (Xi_temporal={:.3}).", xi_t);
                println!("  Context window K={} may be too short to infer drift phase.", agent.config.context_len);
            }
            if !agent.config.drift_enabled && xi_t > 0.02 {
                println!("  WARNING: false positive temporal signal (Xi_temporal={:.3}) in Markovian world.", xi_t);
            }
        }

        // Rule diagnosis
        if agent.config.rules_enabled {
            let xi_r = agent.rule_advantage();
            let acc_r = agent.path_r_accuracy();
            if xi_r > 0.05 {
                println!("  RULE COMPRESSION EFFECTIVE — Xi_rule={:.3}, rules outperform raw memory.", xi_r);
            }
            if acc_r > 0.95 && agent.config.noise == 0.0 {
                println!("  SYMBOLIC COMPRESSION NEAR-OPTIMAL — acc_R={:.3} in deterministic world.", acc_r);
            }
            if agent.config.context_len > 0 && xi_r > agent.temporal_advantage() {
                println!("  RULES OUTPERFORM TEMPORAL CONTEXT — Xi_rule={:.3} > Xi_temporal={:.3}.", xi_r, agent.temporal_advantage());
            }
            if agent.config.drift_enabled && xi_r < -0.05 {
                println!("  RULES DEGRADE UNDER DRIFT — Xi_rule={:.3}. MLE delta corrupted by phase mixing.", xi_r);
            }
        }

        // Other-agent diagnosis
        if agent.config.other_policy != OtherPolicy::None {
            let acc_o = agent.path_o_accuracy();
            let xi_o = agent.other_advantage();
            if acc_o > 0.9 {
                println!("  OTHER-MODEL STRONG — acc_O={:.3}, reliable theory of B.", acc_o);
            } else if acc_o > 0.5 {
                println!("  OTHER-MODEL DISCOVERED — acc_O={:.3}, M_0 predicts B above chance.", acc_o);
            } else if acc_o > 0.3 {
                println!("  OTHER-MODEL WEAK — acc_O={:.3}, some signal but noisy.", acc_o);
            }
            if xi_o > 0.1 {
                println!("  STATE-CONDITIONED MODEL OUTPERFORMS FREQUENCY — Xi_other={:.3}.", xi_o);
            }
            if acc < 0.5 && agent.config.noise == 0.0 && !agent.config.drift_enabled {
                println!("  SELF-MODEL DEGRADED — B's presence fragments memory.");
            }
        }

        // Self-model diagnosis
        if agent.config.self_model_enabled {
            let xi_s = agent.self_advantage();
            let acc_s = agent.path_s_accuracy();
            if xi_s > 0.05 {
                println!("  SELF-MODEL EFFECTIVE — Xi_self={:.3}, factored memory outperforms full grid.", xi_s);
            }
            let xi_r = agent.reflexive_advantage();
            if xi_r > 0.05 {
                println!("  REFLEXIVE SELECTION EFFECTIVE — Xi_reflexive={:.3}, path selection outperforms fixed Path A.", xi_r);
            }
            let self_tup = agent.self_unique_count();
            let full_tup = agent.memory.unique_count();
            if full_tup > 0 && self_tup < full_tup / 2 {
                println!("  SELF-MEMORY COMPRESSES — {} tuples vs {} full ({:.0}x reduction).",
                    self_tup, full_tup, full_tup as f64 / self_tup.max(1) as f64);
            }
            if acc_s > 0.8 && acc < 0.3 {
                println!("  SELF-MODEL RECOVERS — factored out other agent's noise (acc_S={:.3} vs acc_A={:.3}).", acc_s, acc);
            }
        }

        // Goal + navigation diagnosis
        if agent.config.goal_enabled {
            let goals = world.goals_reached();
            let eff = world.avg_navigation_efficiency();
            let avg_steps = world.avg_steps_per_goal();
            println!(
                "Goals reached:         {}",
                goals
            );
            if goals > 0 {
                println!(
                    "Avg steps per goal:    {:.1}",
                    avg_steps
                );
                println!(
                    "Navigation efficiency: {:.3}",
                    eff
                );
            }
            if agent.config.plan_enabled {
                println!("Planning: model-based BFS (depth {})", agent.config.plan_depth);
            } else if agent.config.greedy_enabled {
                println!("Planning: greedy (Manhattan distance)");
            } else {
                println!("Planning: none (familiarity-based action selection)");
            }
            if goals > 0 && eff > 0.9 {
                println!("  NAVIGATION NEAR-OPTIMAL — efficiency={:.3}, planner finds efficient paths.", eff);
            } else if goals > 0 && eff > 0.5 {
                println!("  NAVIGATION FUNCTIONAL — efficiency={:.3}, room for improvement.", eff);
            } else if goals > 0 && eff < 0.5 && agent.config.plan_enabled {
                println!("  PLANNER WEAK — efficiency={:.3}, model may be inaccurate or depth too shallow.", eff);
            }
            if goals == 0 && agent.config.max_episodes > 500 {
                println!("  NO GOALS REACHED — check plan_depth or model coverage.");
            }
        }

        // Adaptive strategy diagnosis (Stage 7)
        if agent.config.adaptive_strategy {
            let (pc, gc, ec) = agent.strategy_counts();
            let total = (pc + gc + ec).max(1) as f64;
            println!(
                "Strategy: plan={:.1}% greedy={:.1}% explore={:.1}% (gate={:.2}, model_conf={:.3})",
                100.0 * pc as f64 / total,
                100.0 * gc as f64 / total,
                100.0 * ec as f64 / total,
                agent.config.confidence_gate,
                agent.rule_pos_accuracy(),
            );
            if pc as f64 / total > 0.8 {
                println!("  MODEL CONFIDENT — planning dominates ({:.1}%), model accuracy sufficient.", 100.0 * pc as f64 / total);
            } else if gc as f64 / total > 0.5 {
                println!("  MODEL UNCERTAIN — greedy fallback dominates ({:.1}%), model below confidence gate.", 100.0 * gc as f64 / total);
            }
        }
        if agent.config.recency_rules {
            println!("Recency rules: window={}, rule confidence={:.3}",
                agent.config.recency_window,
                agent.rule_set.avg_confidence());
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
