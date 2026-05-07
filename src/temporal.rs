/// TemporalMemory — Stage 2 context-window memory.
///
/// Stores (context, action, state_before, state_after) tuples where context
/// is the last K (action, state) pairs. When context_len=0, behaves
/// identically to ExperienceMemory (backward compat).

use crate::grid::Grid;

#[derive(Debug, Clone)]
pub struct TemporalTuple {
    pub context: Vec<(u8, Grid)>,
    pub action: u8,
    pub state_before: Grid,
    pub state_after: Grid,
    pub count: u32,
}

pub struct TemporalMemory {
    tuples: Vec<TemporalTuple>,
    total_stores: u64,
    context_len: usize,
}

/// Distance between two context sequences.
/// Action mismatch = 25 (full grid penalty). State distance = Hamming.
/// Length mismatch = 25 per missing step.
fn context_distance(a: &[(u8, Grid)], b: &[(u8, Grid)]) -> usize {
    let len = a.len().min(b.len());
    let mut dist = 0usize;
    for i in 0..len {
        if a[i].0 != b[i].0 {
            dist += 25;
        }
        dist += a[i].1.hamming_distance(&b[i].1);
    }
    dist += a.len().abs_diff(b.len()) * 25;
    dist
}

impl TemporalMemory {
    pub fn new(context_len: usize) -> Self {
        Self {
            tuples: Vec::new(),
            total_stores: 0,
            context_len,
        }
    }

    /// Store an experience with context. Deduplicates on exact match.
    pub fn store(
        &mut self,
        context: &[(u8, Grid)],
        action: u8,
        state_before: Grid,
        state_after: Grid,
    ) {
        self.total_stores += 1;
        let ctx: Vec<(u8, Grid)> = context
            .iter()
            .rev()
            .take(self.context_len)
            .rev()
            .cloned()
            .collect();

        // Check for exact duplicate
        for tuple in &mut self.tuples {
            if tuple.action == action
                && tuple.state_before == state_before
                && tuple.state_after == state_after
                && tuple.context.len() == ctx.len()
                && tuple.context.iter().zip(ctx.iter()).all(|(a, b)| a.0 == b.0 && a.1 == b.1)
            {
                tuple.count += 1;
                return;
            }
        }

        self.tuples.push(TemporalTuple {
            context: ctx,
            action,
            state_before,
            state_after,
            count: 1,
        });
    }

    /// Retrieve: exact context match first, then approximate fallback.
    /// Returns state_after from highest-count matching tuple.
    pub fn retrieve(
        &self,
        context: &[(u8, Grid)],
        action: u8,
        state_before: &Grid,
    ) -> Option<&Grid> {
        let ctx: Vec<(u8, Grid)> = context
            .iter()
            .rev()
            .take(self.context_len)
            .rev()
            .cloned()
            .collect();

        // Exact match
        let exact = self
            .tuples
            .iter()
            .filter(|t| {
                t.action == action
                    && t.state_before == *state_before
                    && t.context.len() == ctx.len()
                    && t.context.iter().zip(ctx.iter()).all(|(a, b)| a.0 == b.0 && a.1 == b.1)
            })
            .max_by_key(|t| t.count)
            .map(|t| &t.state_after);

        if exact.is_some() {
            return exact;
        }

        // Approximate: match action, minimize context + state distance
        self.tuples
            .iter()
            .filter(|t| t.action == action)
            .min_by_key(|t| {
                t.state_before.hamming_distance(state_before) + context_distance(&t.context, &ctx)
            })
            .map(|t| &t.state_after)
    }

    pub fn unique_count(&self) -> usize {
        self.tuples.len()
    }

    pub fn total_stores(&self) -> u64 {
        self.total_stores
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
    fn test_context_zero_like_experience_memory() {
        let mut mem = TemporalMemory::new(0);
        let before = grid_at(3, 1, 1);
        let after = grid_at(3, 0, 1);
        mem.store(&[], 0, before.clone(), after.clone());

        let result = mem.retrieve(&[], 0, &before);
        assert_eq!(result, Some(&after));
        assert_eq!(mem.unique_count(), 1);
    }

    #[test]
    fn test_exact_context_match() {
        let mut mem = TemporalMemory::new(2);
        let s0 = grid_at(3, 1, 1);
        let s1 = grid_at(3, 0, 1);
        let s2 = grid_at(3, 0, 2);
        let after = grid_at(3, 1, 2);

        let ctx = vec![(0u8, s0.clone()), (1u8, s1.clone())];
        mem.store(&ctx, 2, s2.clone(), after.clone());

        let result = mem.retrieve(&ctx, 2, &s2);
        assert_eq!(result, Some(&after));
    }

    #[test]
    fn test_different_context_different_prediction() {
        let mut mem = TemporalMemory::new(1);
        let state = grid_at(3, 1, 1);
        let after_a = grid_at(3, 0, 1);
        let after_b = grid_at(3, 2, 1);

        let ctx_a = vec![(0u8, grid_at(3, 0, 0))];
        let ctx_b = vec![(1u8, grid_at(3, 2, 2))];

        // Same (action=0, state) but different context → different outcomes
        for _ in 0..5 {
            mem.store(&ctx_a, 0, state.clone(), after_a.clone());
            mem.store(&ctx_b, 0, state.clone(), after_b.clone());
        }

        assert_eq!(mem.retrieve(&ctx_a, 0, &state), Some(&after_a));
        assert_eq!(mem.retrieve(&ctx_b, 0, &state), Some(&after_b));
    }

    #[test]
    fn test_approximate_context_fallback() {
        let mut mem = TemporalMemory::new(1);
        let state = grid_at(3, 1, 1);
        let after = grid_at(3, 0, 1);

        let ctx_stored = vec![(0u8, grid_at(3, 0, 0))];
        mem.store(&ctx_stored, 0, state.clone(), after.clone());

        // Query with a slightly different context (different state in context)
        let ctx_query = vec![(0u8, grid_at(3, 0, 1))];
        let result = mem.retrieve(&ctx_query, 0, &state);
        assert_eq!(result, Some(&after)); // approximate match should find it
    }

    #[test]
    fn test_dedup_with_context() {
        let mut mem = TemporalMemory::new(1);
        let state = grid_at(3, 1, 1);
        let after = grid_at(3, 0, 1);
        let ctx = vec![(0u8, grid_at(3, 0, 0))];

        for _ in 0..5 {
            mem.store(&ctx, 0, state.clone(), after.clone());
        }
        assert_eq!(mem.unique_count(), 1);
        assert_eq!(mem.total_stores(), 5);
    }

    #[test]
    fn test_context_distance_identical() {
        let ctx = vec![(0u8, grid_at(3, 1, 1))];
        assert_eq!(context_distance(&ctx, &ctx), 0);
    }

    #[test]
    fn test_context_distance_action_mismatch() {
        let a = vec![(0u8, grid_at(3, 1, 1))];
        let b = vec![(1u8, grid_at(3, 1, 1))];
        assert_eq!(context_distance(&a, &b), 25);
    }

    #[test]
    fn test_context_distance_length_mismatch() {
        let a = vec![(0u8, grid_at(3, 1, 1)), (1u8, grid_at(3, 0, 1))];
        let b = vec![(0u8, grid_at(3, 1, 1))];
        // 1 matching pair (dist=0) + 1 length penalty (25)
        assert_eq!(context_distance(&a, &b), 25);
    }
}
