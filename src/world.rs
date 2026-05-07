/// MicroWorld — the Stage 0 environment.
///
/// A tiny deterministic grid world. One marker pixel moves on a background.
/// Properties: contingent, low-dimensional, low-noise, tight-loop.
/// The system must discover that its actions reliably move the marker.

use crate::grid::Grid;

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
}

impl MicroWorld {
    /// Create a new world with the marker at center.
    pub fn new(size: usize) -> Self {
        let center = size / 2;
        Self {
            size,
            marker_row: center,
            marker_col: center,
        }
    }

    /// Observe the current state as a Grid.
    pub fn observe(&self) -> Grid {
        let mut grid = Grid::filled(self.size, self.size, BG_COLOR);
        grid.set(self.marker_row, self.marker_col, MARKER_COLOR);
        grid
    }

    /// Apply an action. Deterministic. Clamps at edges.
    pub fn apply(&mut self, action: u8) {
        match action {
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
            _ => {} // invalid action = no-op
        }
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
            w1.apply(a);
            w2.apply(a);
        }
        assert_eq!(w1.observe(), w2.observe());
    }

    #[test]
    fn test_reset() {
        let mut w = MicroWorld::new(5);
        w.apply(ACTION_UP);
        w.apply(ACTION_LEFT);
        w.reset();
        assert_eq!(w.marker_pos(), (2, 2));
    }

    #[test]
    fn test_grid_changes_after_action() {
        let mut w = MicroWorld::new(5);
        let before = w.observe();
        w.apply(ACTION_UP);
        let after = w.observe();
        assert_ne!(before, after);
        // Exactly 2 cells differ (old marker pos → bg, new marker pos → marker)
        assert_eq!(before.hamming_distance(&after), 2);
    }

    #[test]
    fn test_grid_unchanged_at_edge() {
        let mut w = MicroWorld::new(5);
        // Move to top edge
        for _ in 0..5 {
            w.apply(ACTION_UP);
        }
        let before = w.observe();
        w.apply(ACTION_UP); // clamped — should not change
        let after = w.observe();
        assert_eq!(before, after);
    }
}
