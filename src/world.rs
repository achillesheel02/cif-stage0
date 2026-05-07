/// MicroWorld — the Stage 0 environment.
///
/// A tiny deterministic grid world. One marker pixel moves on a background.
/// Properties: contingent, low-dimensional, low-noise, tight-loop.
/// The system must discover that its actions reliably move the marker.

use crate::grid::Grid;
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
}

impl MicroWorld {
    /// Create a new world with the marker at center. Deterministic (Stage 0).
    pub fn new(size: usize) -> Self {
        Self::with_noise(size, 0.0, 42)
    }

    /// Create a new world with action noise (Stage 1).
    pub fn with_noise(size: usize, noise: f64, seed: u64) -> Self {
        let center = size / 2;
        Self {
            size,
            marker_row: center,
            marker_col: center,
            noise,
            rng: StdRng::seed_from_u64(seed.wrapping_add(1)),
        }
    }

    /// Observe the current state as a Grid.
    pub fn observe(&self) -> Grid {
        let mut grid = Grid::filled(self.size, self.size, BG_COLOR);
        grid.set(self.marker_row, self.marker_col, MARKER_COLOR);
        grid
    }

    /// Apply an action. Returns the actually-executed action.
    /// With noise > 0, the intended action may be replaced by a random one.
    pub fn apply(&mut self, action: u8) -> u8 {
        let executed = if self.noise > 0.0 && self.rng.gen::<f64>() < self.noise {
            self.rng.gen_range(0..N_ACTIONS as u8)
        } else {
            action
        };

        match executed {
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
}
