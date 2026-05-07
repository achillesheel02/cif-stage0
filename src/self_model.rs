/// SelfMemory — factored memory that strips other-agent markers.
///
/// Wraps ExperienceMemory but removes color=2 (other agent) from all
/// grids before store and retrieve. This means:
/// - B's position never enters self-memory
/// - Memory deduplicates on self-position only
/// - Retrieval matches on self-state regardless of B's location
///
/// When no other agent exists, strip_color(2) is a no-op,
/// so SelfMemory degenerates to ExperienceMemory (backward compat).

use crate::grid::Grid;
use crate::memory::ExperienceMemory;

pub struct SelfMemory {
    inner: ExperienceMemory,
    strip_color: u8,
}

impl SelfMemory {
    pub fn new() -> Self {
        Self {
            inner: ExperienceMemory::new(),
            strip_color: 2,
        }
    }

    /// Store an experience with other-agent markers stripped.
    pub fn store(&mut self, action: u8, state_before: Grid, state_after: Grid) {
        let sb = state_before.strip_color(self.strip_color);
        let sa = state_after.strip_color(self.strip_color);
        self.inner.store(action, sb, sa);
    }

    /// Retrieve: strips query state, then exact + approximate fallback.
    pub fn retrieve(&self, action: u8, state_before: &Grid) -> Option<&Grid> {
        let sq = state_before.strip_color(self.strip_color);
        self.inner.retrieve(action, &sq)
    }

    /// Familiarity on self-only state.
    pub fn familiarity(&self, action: u8, state_before: &Grid) -> u32 {
        let sq = state_before.strip_color(self.strip_color);
        self.inner.familiarity(action, &sq)
    }

    /// Most common outcome for this action (ignores state_before).
    pub fn most_common_outcome(&self, action: u8) -> Option<&Grid> {
        self.inner.most_common_outcome(action)
    }

    pub fn unique_count(&self) -> usize {
        self.inner.unique_count()
    }

    pub fn total_stores(&self) -> u64 {
        self.inner.total_stores()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_self_only(size: usize, r: usize, c: usize) -> Grid {
        let mut g = Grid::filled(size, size, 0);
        g.set(r, c, 1);
        g
    }

    fn grid_with_other(size: usize, sr: usize, sc: usize, or: usize, oc: usize) -> Grid {
        let mut g = Grid::filled(size, size, 0);
        g.set(sr, sc, 1);
        g.set(or, oc, 2);
        g
    }

    #[test]
    fn test_strip_removes_other_from_store() {
        let mut sm = SelfMemory::new();
        let before = grid_with_other(5, 2, 2, 4, 4);
        let after = grid_with_other(5, 1, 2, 4, 3);
        sm.store(0, before.clone(), after.clone());

        // Query with self-only state should find it
        let query = grid_self_only(5, 2, 2);
        let result = sm.retrieve(0, &query);
        assert!(result.is_some());
        // Result should be self-only (no color=2)
        assert_eq!(result.unwrap().find_other(), None);
    }

    #[test]
    fn test_dedup_ignores_b_position() {
        let mut sm = SelfMemory::new();
        // Same self-position, different B positions
        for b_col in 0..5 {
            let before = grid_with_other(5, 2, 2, 4, b_col);
            let after = grid_with_other(5, 1, 2, 4, b_col);
            sm.store(0, before, after);
        }
        // All should dedup to 1 unique tuple
        assert_eq!(sm.unique_count(), 1);
        assert_eq!(sm.total_stores(), 5);
    }

    #[test]
    fn test_no_other_degenerates() {
        let mut sm = SelfMemory::new();
        let mut em = ExperienceMemory::new();

        let before = grid_self_only(5, 2, 2);
        let after = grid_self_only(5, 1, 2);

        sm.store(0, before.clone(), after.clone());
        em.store(0, before.clone(), after.clone());

        assert_eq!(sm.unique_count(), em.unique_count());
        assert_eq!(sm.total_stores(), em.total_stores());
        assert_eq!(sm.retrieve(0, &before), em.retrieve(0, &before));
    }

    #[test]
    fn test_random_b_doesnt_fragment() {
        let mut sm = SelfMemory::new();
        let mut em = ExperienceMemory::new();

        // 20 episodes: self at (2,2)→(1,2), B wanders in bottom row (no overlap)
        for i in 0..20 {
            let b_col = i % 5;
            let before = grid_with_other(5, 2, 2, 4, b_col);
            let after = grid_with_other(5, 1, 2, 4, b_col);
            sm.store(0, before.clone(), after.clone());
            em.store(0, before, after);
        }

        // SelfMemory should have 1 unique tuple (same self-state every time)
        assert_eq!(sm.unique_count(), 1);
        // ExperienceMemory should have many (different B positions)
        assert!(em.unique_count() > 3);
    }
}
