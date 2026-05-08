/// RuleSet — Stage 3 symbolic compression.
///
/// Extracts movement rules from raw experience: (action, at_edge) → position delta.
/// Rules are generative predictions — they construct the predicted grid from the
/// current position + delta, rather than retrieving a stored grid. This means
/// rules predict correctly for states never seen before.

use std::collections::HashMap;

use crate::grid::Grid;
use crate::memory::ExperienceMemory;
use crate::world::{ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP};

#[derive(Debug, Clone)]
pub struct MovementRule {
    pub action: u8,
    pub at_edge: bool,
    pub delta_row: i32,
    pub delta_col: i32,
    pub count: u32,
    pub total: u32,
}

pub struct RuleSet {
    rules: Vec<MovementRule>,
    world_size: usize,
}

fn is_at_edge(action: u8, row: usize, col: usize, size: usize) -> bool {
    match action {
        ACTION_UP => row == 0,
        ACTION_DOWN => row == size - 1,
        ACTION_LEFT => col == 0,
        ACTION_RIGHT => col == size - 1,
        _ => false,
    }
}

impl RuleSet {
    pub fn new(world_size: usize) -> Self {
        Self {
            rules: Vec::new(),
            world_size,
        }
    }

    /// Extract movement rules from experience memory.
    /// For each (action, at_edge) pair, finds the most common position delta.
    pub fn extract(&mut self, memory: &ExperienceMemory) {
        // Count deltas per (action, at_edge)
        let mut delta_counts: HashMap<(u8, bool, i32, i32), u32> = HashMap::new();
        let mut totals: HashMap<(u8, bool), u32> = HashMap::new();

        for tuple in memory.iter_tuples() {
            let pos_before = tuple.state_before.find_marker();
            let pos_after = tuple.state_after.find_marker();

            if let (Some((rb, cb)), Some((ra, ca))) = (pos_before, pos_after) {
                let dr = ra as i32 - rb as i32;
                let dc = ca as i32 - cb as i32;
                let edge = is_at_edge(tuple.action, rb, cb, self.world_size);

                *delta_counts
                    .entry((tuple.action, edge, dr, dc))
                    .or_insert(0) += tuple.count;
                *totals.entry((tuple.action, edge)).or_insert(0) += tuple.count;
            }
        }

        // For each (action, at_edge), take the delta with the highest count
        let mut best: HashMap<(u8, bool), (i32, i32, u32)> = HashMap::new();
        for ((action, edge, dr, dc), count) in &delta_counts {
            let key = (*action, *edge);
            let entry = best.entry(key).or_insert((0, 0, 0));
            if *count > entry.2 {
                *entry = (*dr, *dc, *count);
            }
        }

        self.rules.clear();
        for ((action, at_edge), (dr, dc, count)) in &best {
            let total = totals[&(*action, *at_edge)];
            self.rules.push(MovementRule {
                action: *action,
                at_edge: *at_edge,
                delta_row: *dr,
                delta_col: *dc,
                count: *count,
                total,
            });
        }
    }

    /// Predict the next state by applying the matching rule.
    /// Returns None if no rule exists for this (action, edge_condition).
    pub fn predict(&self, action: u8, state: &Grid) -> Option<Grid> {
        let (row, col) = state.find_marker()?;
        let at_edge = is_at_edge(action, row, col, self.world_size);

        let rule = self
            .rules
            .iter()
            .find(|r| r.action == action && r.at_edge == at_edge)?;

        let new_row = (row as i32 + rule.delta_row)
            .max(0)
            .min(self.world_size as i32 - 1) as usize;
        let new_col = (col as i32 + rule.delta_col)
            .max(0)
            .min(self.world_size as i32 - 1) as usize;

        let mut predicted = state.clone();
        predicted.set(row, col, 0);          // clear old agent position
        predicted.set(new_row, new_col, 1);  // place agent at new position
        Some(predicted)
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Extract movement rules from raw (action, state_before, state_after) tuples.
    /// Each tuple has implicit count=1. Used for recency-weighted extraction.
    pub fn extract_from_raw(&mut self, tuples: &[(u8, Grid, Grid)], world_size: usize) {
        let mut delta_counts: HashMap<(u8, bool, i32, i32), u32> = HashMap::new();
        let mut totals: HashMap<(u8, bool), u32> = HashMap::new();

        for (action, state_before, state_after) in tuples {
            let pos_before = state_before.find_marker();
            let pos_after = state_after.find_marker();

            if let (Some((rb, cb)), Some((ra, ca))) = (pos_before, pos_after) {
                let dr = ra as i32 - rb as i32;
                let dc = ca as i32 - cb as i32;
                let edge = is_at_edge(*action, rb, cb, world_size);

                *delta_counts
                    .entry((*action, edge, dr, dc))
                    .or_insert(0) += 1;
                *totals.entry((*action, edge)).or_insert(0) += 1;
            }
        }

        let mut best: HashMap<(u8, bool), (i32, i32, u32)> = HashMap::new();
        for ((action, edge, dr, dc), count) in &delta_counts {
            let key = (*action, *edge);
            let entry = best.entry(key).or_insert((0, 0, 0));
            if *count > entry.2 {
                *entry = (*dr, *dc, *count);
            }
        }

        self.rules.clear();
        for ((action, at_edge), (dr, dc, count)) in &best {
            let total = totals[&(*action, *at_edge)];
            self.rules.push(MovementRule {
                action: *action,
                at_edge: *at_edge,
                delta_row: *dr,
                delta_col: *dc,
                count: *count,
                total,
            });
        }
    }

    /// Average confidence across rules: mean(count/total).
    pub fn avg_confidence(&self) -> f64 {
        if self.rules.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .rules
            .iter()
            .map(|r| r.count as f64 / r.total as f64)
            .sum();
        sum / self.rules.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ExperienceMemory;

    fn grid_at(size: usize, r: usize, c: usize) -> Grid {
        let mut g = Grid::filled(size, size, 0);
        g.set(r, c, 1);
        g
    }

    #[test]
    fn test_extract_single_tuple() {
        let mut mem = ExperienceMemory::new();
        // UP from (2,2) → (1,2)
        mem.store(ACTION_UP, grid_at(5, 2, 2), grid_at(5, 1, 2));

        let mut rules = RuleSet::new(5);
        rules.extract(&mem);

        assert_eq!(rules.rule_count(), 1);
        let rule = &rules.rules[0];
        assert_eq!(rule.action, ACTION_UP);
        assert!(!rule.at_edge);
        assert_eq!(rule.delta_row, -1);
        assert_eq!(rule.delta_col, 0);
    }

    #[test]
    fn test_extract_mle_wins() {
        let mut mem = ExperienceMemory::new();
        // UP from interior: mostly (-1,0), sometimes (0,0) due to noise
        for _ in 0..8 {
            mem.store(ACTION_UP, grid_at(5, 2, 2), grid_at(5, 1, 2));
        }
        for _ in 0..2 {
            mem.store(ACTION_UP, grid_at(5, 2, 2), grid_at(5, 2, 2)); // no movement
        }

        let mut rules = RuleSet::new(5);
        rules.extract(&mem);

        let rule = rules
            .rules
            .iter()
            .find(|r| r.action == ACTION_UP && !r.at_edge)
            .unwrap();
        assert_eq!(rule.delta_row, -1);
        assert_eq!(rule.delta_col, 0);
        assert_eq!(rule.count, 8);
        assert_eq!(rule.total, 10);
    }

    #[test]
    fn test_predict_interior() {
        let mut mem = ExperienceMemory::new();
        mem.store(ACTION_UP, grid_at(5, 2, 2), grid_at(5, 1, 2));

        let mut rules = RuleSet::new(5);
        rules.extract(&mem);

        // Predict UP from (3, 1) — a state never stored in memory
        let state = grid_at(5, 3, 1);
        let predicted = rules.predict(ACTION_UP, &state);
        assert_eq!(predicted, Some(grid_at(5, 2, 1)));
    }

    #[test]
    fn test_predict_edge() {
        let mut mem = ExperienceMemory::new();
        // UP at top edge: (0,2) → (0,2) (no movement)
        mem.store(ACTION_UP, grid_at(5, 0, 2), grid_at(5, 0, 2));

        let mut rules = RuleSet::new(5);
        rules.extract(&mem);

        let state = grid_at(5, 0, 3);
        let predicted = rules.predict(ACTION_UP, &state);
        assert_eq!(predicted, Some(grid_at(5, 0, 3)));
    }

    #[test]
    fn test_predict_empty_returns_none() {
        let rules = RuleSet::new(5);
        let state = grid_at(5, 2, 2);
        assert_eq!(rules.predict(ACTION_UP, &state), None);
    }

    #[test]
    fn test_confidence_deterministic() {
        let mut mem = ExperienceMemory::new();
        for _ in 0..10 {
            mem.store(ACTION_UP, grid_at(5, 2, 2), grid_at(5, 1, 2));
        }

        let mut rules = RuleSet::new(5);
        rules.extract(&mem);

        assert!((rules.avg_confidence() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confidence_noisy() {
        let mut mem = ExperienceMemory::new();
        for _ in 0..7 {
            mem.store(ACTION_UP, grid_at(5, 2, 2), grid_at(5, 1, 2));
        }
        for _ in 0..3 {
            mem.store(ACTION_UP, grid_at(5, 2, 2), grid_at(5, 2, 2));
        }

        let mut rules = RuleSet::new(5);
        rules.extract(&mem);

        assert!((rules.avg_confidence() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_extract_from_raw_basic() {
        let tuples = vec![
            (ACTION_UP, grid_at(5, 2, 2), grid_at(5, 1, 2)),
            (ACTION_UP, grid_at(5, 3, 1), grid_at(5, 2, 1)),
            (ACTION_DOWN, grid_at(5, 1, 1), grid_at(5, 2, 1)),
        ];

        let mut rules = RuleSet::new(5);
        rules.extract_from_raw(&tuples, 5);

        assert_eq!(rules.rule_count(), 2); // UP interior + DOWN interior
        let up_rule = rules.rules.iter().find(|r| r.action == ACTION_UP).unwrap();
        assert_eq!(up_rule.delta_row, -1);
        assert_eq!(up_rule.delta_col, 0);
    }

    #[test]
    fn test_extract_from_raw_matches_extract() {
        // Same data through both paths should produce same rules
        let mut mem = ExperienceMemory::new();
        let raw = vec![
            (ACTION_UP, grid_at(5, 2, 2), grid_at(5, 1, 2)),
            (ACTION_DOWN, grid_at(5, 2, 2), grid_at(5, 3, 2)),
            (ACTION_LEFT, grid_at(5, 2, 2), grid_at(5, 2, 1)),
            (ACTION_RIGHT, grid_at(5, 2, 2), grid_at(5, 2, 3)),
        ];
        for (a, sb, sa) in &raw {
            mem.store(*a, sb.clone(), sa.clone());
        }

        let mut rules_mem = RuleSet::new(5);
        rules_mem.extract(&mem);

        let mut rules_raw = RuleSet::new(5);
        rules_raw.extract_from_raw(&raw, 5);

        assert_eq!(rules_mem.rule_count(), rules_raw.rule_count());
        // Both should predict the same for any state
        let state = grid_at(5, 3, 3);
        for action in 0..4u8 {
            assert_eq!(rules_mem.predict(action, &state), rules_raw.predict(action, &state));
        }
    }

    #[test]
    fn test_edge_detection() {
        assert!(is_at_edge(ACTION_UP, 0, 2, 5));
        assert!(!is_at_edge(ACTION_UP, 1, 2, 5));
        assert!(is_at_edge(ACTION_DOWN, 4, 2, 5));
        assert!(!is_at_edge(ACTION_DOWN, 3, 2, 5));
        assert!(is_at_edge(ACTION_LEFT, 2, 0, 5));
        assert!(!is_at_edge(ACTION_LEFT, 2, 1, 5));
        assert!(is_at_edge(ACTION_RIGHT, 2, 4, 5));
        assert!(!is_at_edge(ACTION_RIGHT, 2, 3, 5));
    }
}
