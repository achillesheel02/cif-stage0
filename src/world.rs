/// MicroWorld — the Stage 0 environment.
///
/// A tiny deterministic grid world. One marker pixel moves on a background.
/// Properties: contingent, low-dimensional, low-noise, tight-loop.
/// The system must discover that its actions reliably move the marker.

use crate::grid::Grid;
use crate::other::{OtherAgent, OtherPolicy};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const ACTION_UP: u8 = 0;
pub const ACTION_DOWN: u8 = 1;
pub const ACTION_LEFT: u8 = 2;
pub const ACTION_RIGHT: u8 = 3;
pub const N_ACTIONS: usize = 4;

const MARKER_COLOR: u8 = 1;
const BG_COLOR: u8 = 0;

pub struct MicroWorld {
    size: usize,
    marker_row: usize,
    marker_col: usize,
    noise: f64,
    rng: StdRng,
    drift_enabled: bool,
    drift_period: u64,
    drift_step: u64,
    other: Option<OtherAgent>,
}

impl MicroWorld {
    /// Create a new world with the marker at center. Deterministic (Stage 0).
    pub fn new(size: usize) -> Self {
        Self::with_noise(size, 0.0, 42)
    }

    /// Create a new world with action noise (Stage 1).
    pub fn with_noise(size: usize, noise: f64, seed: u64) -> Self {
        Self::with_drift(size, noise, seed, false, 10)
    }

    /// Create a new world with optional noise and drift (Stage 2).
    pub fn with_drift(size: usize, noise: f64, seed: u64, drift_enabled: bool, drift_period: u64) -> Self {
        let center = size / 2;
        Self {
            size,
            marker_row: center,
            marker_col: center,
            noise,
            rng: StdRng::seed_from_u64(seed.wrapping_add(1)),
            drift_enabled,
            drift_period,
            drift_step: 0,
            other: None,
        }
    }

    /// Create a world with another agent (Stage 4).
    pub fn with_other(
        size: usize, noise: f64, seed: u64,
        drift_enabled: bool, drift_period: u64,
        other_policy: OtherPolicy, other_seed: u64, patrol_period: u64,
    ) -> Self {
        let mut world = Self::with_drift(size, noise, seed, drift_enabled, drift_period);
        if other_policy != OtherPolicy::None {
            world.other = Some(OtherAgent::new(size, other_policy, other_seed, patrol_period));
        }
        world
    }

    /// Observe the current state as a Grid.
    pub fn observe(&self) -> Grid {
        let mut grid = Grid::filled(self.size, self.size, BG_COLOR);
        grid.set(self.marker_row, self.marker_col, MARKER_COLOR);
        if let Some(ref other) = self.other {
            let (or, oc) = other.pos();
            if grid.get(or, oc) == Some(BG_COLOR) {
                grid.set(or, oc, 2);
            }
        }
        grid
    }

    /// Apply a single movement in the given direction (clamped at edges).
    fn apply_movement(&mut self, direction: u8) {
        match direction {
            ACTION_UP => {
                if self.marker_row > 0 {
                    self.marker_row -= 1;
                }
            }
            ACTION_DOWN => {
                if self.marker_row < self.size - 1 {
                    self.marker_row += 1;
                }
            }
            ACTION_LEFT => {
                if self.marker_col > 0 {
                    self.marker_col -= 1;
                }
            }
            ACTION_RIGHT => {
                if self.marker_col < self.size - 1 {
                    self.marker_col += 1;
                }
            }
            _ => {}
        }
    }

    /// Current drift direction: right → down → left → up cycle.
    fn drift_direction(&self) -> u8 {
        let phase = (self.drift_step % (4 * self.drift_period)) / self.drift_period;
        [ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT, ACTION_UP][phase as usize]
    }

    /// Apply an action. Returns the actually-executed action.
    /// With noise > 0, the intended action may be replaced by a random one.
    /// With drift enabled, a hidden force also moves the marker after the action.
    pub fn apply(&mut self, action: u8) -> u8 {
        let executed = if self.noise > 0.0 && self.rng.gen::<f64>() < self.noise {
            self.rng.gen_range(0..N_ACTIONS as u8)
        } else {
            action
        };

        self.apply_movement(executed);

        if self.drift_enabled {
            self.apply_movement(self.drift_direction());
            self.drift_step += 1;
        }

        if let Some(ref mut other) = self.other {
            let b_action = other.decide(self.marker_row, self.marker_col);
            other.apply_movement(b_action);
        }

        executed
    }

    /// Reset marker to center.
    pub fn reset(&mut self) {
        let center = self.size / 2;
        self.marker_row = center;
        self.marker_col = center;
    }

    /// Current marker position.
    pub fn marker_pos(&self) -> (usize, usize) {
        (self.marker_row, self.marker_col)
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn other_last_action(&self) -> Option<u8> {
        self.other.as_ref().map(|o| o.last_action())
    }

    pub fn other_pos(&self) -> Option<(usize, usize)> {
        self.other.as_ref().map(|o| o.pos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let w = MicroWorld::new(5);
        assert_eq!(w.marker_pos(), (2, 2));
        let g = w.observe();
        assert_eq!(g.get(2, 2), Some(1));
        assert_eq!(g.get(0, 0), Some(0));
    }

    #[test]
    fn test_move_up() {
        let mut w = MicroWorld::new(5);
        w.apply(ACTION_UP);
        assert_eq!(w.marker_pos(), (1, 2));
    }

    #[test]
    fn test_move_down() {
        let mut w = MicroWorld::new(5);
        w.apply(ACTION_DOWN);
        assert_eq!(w.marker_pos(), (3, 2));
    }

    #[test]
    fn test_clamp_at_top() {
        let mut w = MicroWorld::new(5);
        for _ in 0..10 {
            w.apply(ACTION_UP);
        }
        assert_eq!(w.marker_pos(), (0, 2));
    }

    #[test]
    fn test_clamp_at_bottom() {
        let mut w = MicroWorld::new(5);
        for _ in 0..10 {
            w.apply(ACTION_DOWN);
        }
        assert_eq!(w.marker_pos(), (4, 2));
    }

    #[test]
    fn test_clamp_at_left() {
        let mut w = MicroWorld::new(5);
        for _ in 0..10 {
            w.apply(ACTION_LEFT);
        }
        assert_eq!(w.marker_pos(), (2, 0));
    }

    #[test]
    fn test_clamp_at_right() {
        let mut w = MicroWorld::new(5);
        for _ in 0..10 {
            w.apply(ACTION_RIGHT);
        }
        assert_eq!(w.marker_pos(), (2, 4));
    }

    #[test]
    fn test_deterministic() {
        let mut w1 = MicroWorld::new(5);
        let mut w2 = MicroWorld::new(5);
        let actions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT];
        for &a in &actions {
            let _ = w1.apply(a);
            let _ = w2.apply(a);
        }
        assert_eq!(w1.observe(), w2.observe());
    }

    #[test]
    fn test_reset() {
        let mut w = MicroWorld::new(5);
        let _ = w.apply(ACTION_UP);
        let _ = w.apply(ACTION_LEFT);
        w.reset();
        assert_eq!(w.marker_pos(), (2, 2));
    }

    #[test]
    fn test_grid_changes_after_action() {
        let mut w = MicroWorld::new(5);
        let before = w.observe();
        let _ = w.apply(ACTION_UP);
        let after = w.observe();
        assert_ne!(before, after);
        assert_eq!(before.hamming_distance(&after), 2);
    }

    #[test]
    fn test_grid_unchanged_at_edge() {
        let mut w = MicroWorld::new(5);
        for _ in 0..5 {
            let _ = w.apply(ACTION_UP);
        }
        let before = w.observe();
        let _ = w.apply(ACTION_UP);
        let after = w.observe();
        assert_eq!(before, after);
    }

    // ── Stage 1 tests ────────────────────────────────────────────────

    #[test]
    fn test_noise_zero_is_deterministic() {
        let mut w1 = MicroWorld::with_noise(5, 0.0, 42);
        let mut w2 = MicroWorld::with_noise(5, 0.0, 42);
        let actions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT];
        for &a in &actions {
            assert_eq!(w1.apply(a), w2.apply(a));
        }
        assert_eq!(w1.observe(), w2.observe());
    }

    #[test]
    fn test_apply_returns_executed_action() {
        let mut w = MicroWorld::new(5);
        assert_eq!(w.apply(ACTION_UP), ACTION_UP);
        assert_eq!(w.apply(ACTION_DOWN), ACTION_DOWN);
        assert_eq!(w.apply(ACTION_LEFT), ACTION_LEFT);
        assert_eq!(w.apply(ACTION_RIGHT), ACTION_RIGHT);
    }

    #[test]
    fn test_noise_one_produces_different_actions() {
        let mut w = MicroWorld::with_noise(5, 1.0, 42);
        let mut different = false;
        for _ in 0..100 {
            let executed = w.apply(ACTION_UP);
            if executed != ACTION_UP {
                different = true;
                break;
            }
        }
        assert!(different, "noise=1.0 should replace at least one action in 100 tries");
    }

    #[test]
    fn test_noise_reproducible() {
        let mut w1 = MicroWorld::with_noise(5, 0.5, 99);
        let mut w2 = MicroWorld::with_noise(5, 0.5, 99);
        for _ in 0..50 {
            assert_eq!(w1.apply(ACTION_UP), w2.apply(ACTION_UP));
        }
    }

    // ── Stage 2 tests ────────────────────────────────────────────────

    #[test]
    fn test_drift_disabled_is_stage01() {
        let mut w1 = MicroWorld::with_noise(5, 0.0, 42);
        let mut w2 = MicroWorld::with_drift(5, 0.0, 42, false, 10);
        let actions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT];
        for &a in &actions {
            assert_eq!(w1.apply(a), w2.apply(a));
        }
        assert_eq!(w1.observe(), w2.observe());
    }

    #[test]
    fn test_drift_moves_marker_extra() {
        // drift_period=10, phase 0 = right. Moving up should also drift right.
        let mut w = MicroWorld::with_drift(5, 0.0, 42, true, 10);
        assert_eq!(w.marker_pos(), (2, 2));
        w.apply(ACTION_UP); // up + drift right
        assert_eq!(w.marker_pos(), (1, 3)); // row-1, col+1
    }

    #[test]
    fn test_drift_cycle() {
        // Full cycle: 4 phases × drift_period=1 → 4 steps to complete
        let mut w = MicroWorld::with_drift(5, 0.0, 42, true, 1);
        // Start at (2,2). Stay still (clamp doesn't matter for tracking phase).
        // Step 0: phase 0 = right. Agent does nothing meaningful, let's use up.
        // We'll track drift_direction through the phases.
        let w_test = MicroWorld::with_drift(5, 0.0, 42, true, 1);
        assert_eq!(w_test.drift_direction(), ACTION_RIGHT); // phase 0
        w.apply(ACTION_UP); // drift_step becomes 1
        // After step, drift_step=1, phase=1=down
        assert_eq!(w.drift_step, 1);
    }

    #[test]
    fn test_drift_clamps_at_edges() {
        // Drift right at right edge should clamp
        let mut w = MicroWorld::with_drift(5, 0.0, 42, true, 10);
        // Move to right edge
        for _ in 0..5 {
            w.apply(ACTION_RIGHT); // agent right + drift right
        }
        assert_eq!(w.marker_pos().1, 4); // col clamped at 4
    }

    #[test]
    fn test_drift_deterministic() {
        let mut w1 = MicroWorld::with_drift(5, 0.0, 42, true, 5);
        let mut w2 = MicroWorld::with_drift(5, 0.0, 42, true, 5);
        let actions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT,
                       ACTION_UP, ACTION_UP, ACTION_RIGHT, ACTION_DOWN];
        for &a in &actions {
            assert_eq!(w1.apply(a), w2.apply(a));
        }
        assert_eq!(w1.observe(), w2.observe());
    }

    // ── Stage 4 tests ────────────────────────────────────────────────

    #[test]
    fn test_no_other_is_stage0_3() {
        let mut w1 = MicroWorld::with_drift(5, 0.0, 42, false, 10);
        let mut w2 = MicroWorld::with_other(5, 0.0, 42, false, 10, OtherPolicy::None, 137, 5);
        let actions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT];
        for &a in &actions {
            assert_eq!(w1.apply(a), w2.apply(a));
        }
        assert_eq!(w1.observe(), w2.observe());
    }

    #[test]
    fn test_other_appears_on_grid() {
        let w = MicroWorld::with_other(5, 0.0, 42, false, 10, OtherPolicy::Fixed, 137, 5);
        let grid = w.observe();
        assert!(grid.find_other().is_some(), "Other agent should appear on grid");
    }

    #[test]
    fn test_other_moves_after_apply() {
        let mut w = MicroWorld::with_other(5, 0.0, 42, false, 10, OtherPolicy::Fixed, 137, 5);
        let pos_before = w.other_pos().unwrap();
        w.apply(ACTION_UP);
        let pos_after = w.other_pos().unwrap();
        // Fixed = UP, so B should have moved up (from (4,4) to (3,4))
        assert_ne!(pos_before, pos_after);
        assert_eq!(pos_after.0, pos_before.0 - 1);
    }

    #[test]
    fn test_overlap_a_priority() {
        let mut w = MicroWorld::with_other(5, 0.0, 42, false, 10, OtherPolicy::Chase, 137, 5);
        // Chase B toward A until they overlap. A at center (2,2), B starts at (4,4).
        for _ in 0..20 {
            w.apply(ACTION_UP); // A moves up, but B chases
        }
        let grid = w.observe();
        // A's marker should be visible at A's position
        assert_eq!(grid.get(w.marker_pos().0, w.marker_pos().1), Some(1));
    }
}
