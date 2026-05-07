/// ExperienceMemory — the Stage 0 memory store.
///
/// Stores (action, state_before, state_after) tuples.
/// Write policy: deduplicate, increment count on repeat.
/// Retrieval: exact match first, then approximate (Hamming distance).
///
/// This is M at its most primitive. No embeddings, no graph structure,
/// no selectivity. What breaks here tells us what M_0 is missing.

use crate::grid::Grid;

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

impl ExperienceMemory {
    pub fn new() -> Self {
        Self {
            tuples: Vec::new(),
            total_stores: 0,
        }
    }

    /// Store an experience. Deduplicates: if exact match exists, increment count.
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
            self_caused: true, // always true in Stage 0
        });
    }

    /// Retrieve: exact match on (action, state_before).
    /// Returns state_after from the highest-count matching tuple.
    pub fn retrieve_exact(&self, action: u8, state_before: &Grid) -> Option<&Grid> {
        self.tuples
            .iter()
            .filter(|t| t.action == action && &t.state_before == state_before)
            .max_by_key(|t| t.count)
            .map(|t| &t.state_after)
    }

    /// Retrieve: approximate match on (action, state_before).
    /// Finds the tuple with same action and minimum Hamming distance on state_before.
    /// Returns None if no tuples exist for this action.
    pub fn retrieve_approximate(&self, action: u8, state_before: &Grid) -> Option<&Grid> {
        self.tuples
            .iter()
            .filter(|t| t.action == action)
            .min_by_key(|t| t.state_before.hamming_distance(state_before))
            .map(|t| &t.state_after)
    }

    /// Retrieve: exact first, then approximate fallback.
    pub fn retrieve(&self, action: u8, state_before: &Grid) -> Option<&Grid> {
        self.retrieve_exact(action, state_before)
            .or_else(|| self.retrieve_approximate(action, state_before))
    }

    /// Path B: most common state_after for this action (ignores state_before).
    /// "What usually happens when I do this action?"
    pub fn most_common_outcome(&self, action: u8) -> Option<&Grid> {
        // Group by state_after, sum counts
        let matching: Vec<_> = self.tuples.iter().filter(|t| t.action == action).collect();
        if matching.is_empty() {
            return None;
        }

        // Find state_after with highest total count
        // (Simple O(n^2) — fine for Stage 0 scale)
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

    /// How many times has this exact (action, state_before) pair been seen?
    pub fn familiarity(&self, action: u8, state_before: &Grid) -> u32 {
        self.tuples
            .iter()
            .filter(|t| t.action == action && &t.state_before == state_before)
            .map(|t| t.count)
            .sum()
    }

    /// Number of unique tuples stored.
    pub fn unique_count(&self) -> usize {
        self.tuples.len()
    }

    /// Total store() calls (including deduplicates).
    pub fn total_stores(&self) -> u64 {
        self.total_stores
    }

    /// Number of unique tuples for a given action.
    pub fn count_for_action(&self, action: u8) -> usize {
        self.tuples.iter().filter(|t| t.action == action).count()
    }

    /// Prediction confidence for (action, state_before).
    /// Ratio of highest-count outcome to total observations.
    /// 1.0 = deterministic, <1.0 = stochastic, 0.0 = no data.
    pub fn prediction_confidence(&self, action: u8, state_before: &Grid) -> f64 {
        let matching: Vec<&ExperienceTuple> = self
            .tuples
            .iter()
            .filter(|t| t.action == action && &t.state_before == state_before)
            .collect();

        if matching.is_empty() {
            return 0.0;
        }

        let total: u32 = matching.iter().map(|t| t.count).sum();
        let max_count: u32 = matching.iter().map(|t| t.count).max().unwrap_or(0);

        max_count as f64 / total as f64
    }

    /// Iterate over all stored tuples.
    pub fn iter_tuples(&self) -> impl Iterator<Item = &ExperienceTuple> {
        self.tuples.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_at(size: usize, r: usize, c: usize) -> Grid {
        let mut g = Grid::filled(size, size, 0);
        g.set(r, c, 1);
        g
    }

    #[test]
    fn test_store_and_retrieve_exact() {
        let mut mem = ExperienceMemory::new();
        let before = grid_at(3, 1, 1);
        let after = grid_at(3, 0, 1);
        mem.store(0, before.clone(), after.clone());

        let result = mem.retrieve_exact(0, &before);
        assert_eq!(result, Some(&after));
    }

    #[test]
    fn test_dedup_increments_count() {
        let mut mem = ExperienceMemory::new();
        let before = grid_at(3, 1, 1);
        let after = grid_at(3, 0, 1);
        mem.store(0, before.clone(), after.clone());
        mem.store(0, before.clone(), after.clone());
        mem.store(0, before.clone(), after.clone());

        assert_eq!(mem.unique_count(), 1);
        assert_eq!(mem.total_stores(), 3);
        assert_eq!(mem.familiarity(0, &before), 3);
    }

    #[test]
    fn test_retrieve_approximate() {
        let mut mem = ExperienceMemory::new();
        // Store: action 0 at (1,1) → (0,1)
        let before = grid_at(3, 1, 1);
        let after = grid_at(3, 0, 1);
        mem.store(0, before.clone(), after.clone());

        // Query: action 0 at (1,2) — close but not exact
        let query = grid_at(3, 1, 2);
        assert!(mem.retrieve_exact(0, &query).is_none());
        let result = mem.retrieve_approximate(0, &query);
        assert_eq!(result, Some(&after)); // returns closest match
    }

    #[test]
    fn test_most_common_outcome() {
        let mut mem = ExperienceMemory::new();
        let outcome_a = grid_at(3, 0, 0);
        let outcome_b = grid_at(3, 0, 1);

        // outcome_a seen 3 times, outcome_b seen 1 time
        for _ in 0..3 {
            mem.store(0, grid_at(3, 1, 0), outcome_a.clone());
        }
        mem.store(0, grid_at(3, 1, 1), outcome_b.clone());

        let common = mem.most_common_outcome(0);
        assert_eq!(common, Some(&outcome_a));
    }

    #[test]
    fn test_no_match_returns_none() {
        let mem = ExperienceMemory::new();
        let query = grid_at(3, 1, 1);
        assert!(mem.retrieve(0, &query).is_none());
        assert!(mem.most_common_outcome(0).is_none());
    }

    // ── Stage 1 tests ────────────────────────────────────────────────

    #[test]
    fn test_confidence_deterministic() {
        let mut mem = ExperienceMemory::new();
        let before = grid_at(3, 1, 1);
        let after = grid_at(3, 0, 1);
        for _ in 0..10 {
            mem.store(0, before.clone(), after.clone());
        }
        assert!((mem.prediction_confidence(0, &before) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confidence_stochastic() {
        let mut mem = ExperienceMemory::new();
        let before = grid_at(3, 1, 1);
        let after_a = grid_at(3, 0, 1);
        let after_b = grid_at(3, 2, 1);
        for _ in 0..7 {
            mem.store(0, before.clone(), after_a.clone());
        }
        for _ in 0..3 {
            mem.store(0, before.clone(), after_b.clone());
        }
        assert!((mem.prediction_confidence(0, &before) - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confidence_no_data() {
        let mem = ExperienceMemory::new();
        let before = grid_at(3, 1, 1);
        assert!((mem.prediction_confidence(0, &before)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_different_actions_isolated() {
        let mut mem = ExperienceMemory::new();
        let state = grid_at(3, 1, 1);
        let after_up = grid_at(3, 0, 1);
        let after_down = grid_at(3, 2, 1);

        mem.store(0, state.clone(), after_up.clone());
        mem.store(1, state.clone(), after_down.clone());

        assert_eq!(mem.retrieve_exact(0, &state), Some(&after_up));
        assert_eq!(mem.retrieve_exact(1, &state), Some(&after_down));
        assert!(mem.retrieve_exact(2, &state).is_none());
    }
}
