/// OtherAgent — Stage 4 non-player entity.
///
/// Agent B has a fixed policy (not learning). From A's perspective,
/// B is structured environment dynamics — like drift but potentially learnable.

use crate::world::{ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtherPolicy {
    None,
    Random,
    Fixed,
    Patrol,
    Chase,
    Flee,
}

pub struct OtherAgent {
    pub row: usize,
    pub col: usize,
    policy: OtherPolicy,
    step: u64,
    patrol_period: u64,
    last_action: u8,
    rng: StdRng,
    size: usize,
}

impl OtherAgent {
    pub fn new(size: usize, policy: OtherPolicy, seed: u64, patrol_period: u64) -> Self {
        Self {
            row: size - 1,
            col: size - 1,
            policy,
            step: 0,
            patrol_period,
            last_action: 0,
            rng: StdRng::seed_from_u64(seed),
            size,
        }
    }

    /// Choose an action based on policy. A's position needed for Chase/Flee.
    pub fn decide(&mut self, a_row: usize, a_col: usize) -> u8 {
        let action = match self.policy {
            OtherPolicy::None => ACTION_UP,
            OtherPolicy::Random => self.rng.gen_range(0..4),
            OtherPolicy::Fixed => ACTION_UP,
            OtherPolicy::Patrol => {
                let phase = (self.step / self.patrol_period) % 4;
                [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT][phase as usize]
            }
            OtherPolicy::Chase => self.chase_action(a_row, a_col),
            OtherPolicy::Flee => self.flee_action(a_row, a_col),
        };
        self.step += 1;
        self.last_action = action;
        action
    }

    fn chase_action(&self, a_row: usize, a_col: usize) -> u8 {
        let dr = a_row as i32 - self.row as i32;
        let dc = a_col as i32 - self.col as i32;
        // Prefer vertical, then horizontal
        if dr < 0 {
            ACTION_UP
        } else if dr > 0 {
            ACTION_DOWN
        } else if dc < 0 {
            ACTION_LEFT
        } else if dc > 0 {
            ACTION_RIGHT
        } else {
            // Already at A's position — stay (no-op, pick UP which will clamp or not)
            ACTION_UP
        }
    }

    fn flee_action(&self, a_row: usize, a_col: usize) -> u8 {
        let dr = a_row as i32 - self.row as i32;
        let dc = a_col as i32 - self.col as i32;
        // Move away: opposite of chase, prefer direction with more room
        if dr.abs() >= dc.abs() {
            if dr <= 0 {
                ACTION_DOWN
            } else {
                ACTION_UP
            }
        } else if dc <= 0 {
            ACTION_RIGHT
        } else {
            ACTION_LEFT
        }
    }

    pub fn apply_movement(&mut self, direction: u8) {
        match direction {
            ACTION_UP => {
                if self.row > 0 {
                    self.row -= 1;
                }
            }
            ACTION_DOWN => {
                if self.row < self.size - 1 {
                    self.row += 1;
                }
            }
            ACTION_LEFT => {
                if self.col > 0 {
                    self.col -= 1;
                }
            }
            ACTION_RIGHT => {
                if self.col < self.size - 1 {
                    self.col += 1;
                }
            }
            _ => {}
        }
    }

    pub fn last_action(&self) -> u8 {
        self.last_action
    }

    pub fn pos(&self) -> (usize, usize) {
        (self.row, self.col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_all_actions() {
        let mut b = OtherAgent::new(5, OtherPolicy::Random, 42, 5);
        let mut seen = [false; 4];
        for _ in 0..100 {
            let a = b.decide(2, 2);
            b.apply_movement(a);
            seen[a as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "Random should produce all 4 actions");
    }

    #[test]
    fn test_fixed_always_up() {
        let mut b = OtherAgent::new(5, OtherPolicy::Fixed, 42, 5);
        for _ in 0..10 {
            let a = b.decide(2, 2);
            assert_eq!(a, ACTION_UP);
        }
    }

    #[test]
    fn test_patrol_cycles() {
        let mut b = OtherAgent::new(5, OtherPolicy::Patrol, 42, 2);
        let expected = [ACTION_UP, ACTION_UP, ACTION_RIGHT, ACTION_RIGHT,
                       ACTION_DOWN, ACTION_DOWN, ACTION_LEFT, ACTION_LEFT];
        for &exp in &expected {
            let a = b.decide(2, 2);
            assert_eq!(a, exp);
        }
    }

    #[test]
    fn test_chase_approaches() {
        let mut b = OtherAgent::new(5, OtherPolicy::Chase, 42, 5);
        // B starts at (4,4), A at (0,0)
        let start_dist = 4 + 4;
        for _ in 0..3 {
            let a = b.decide(0, 0);
            b.apply_movement(a);
        }
        let end_dist = (b.row as i32).abs() + (b.col as i32).abs();
        assert!(end_dist < start_dist, "Chase should reduce Manhattan distance");
    }

    #[test]
    fn test_flee_retreats() {
        let mut b = OtherAgent::new(5, OtherPolicy::Flee, 42, 5);
        // B starts at (4,4), A at (2,2). B should try to move away.
        let start_dist = 2 + 2;
        let a = b.decide(2, 2);
        b.apply_movement(a);
        let end_dist = ((b.row as i32) - 2).abs() + ((b.col as i32) - 2).abs();
        assert!(end_dist >= start_dist, "Flee should maintain or increase Manhattan distance");
    }

    #[test]
    fn test_edge_clamping() {
        let mut b = OtherAgent::new(5, OtherPolicy::Fixed, 42, 5);
        // Fixed = UP. B starts at (4,4). Move UP 10 times — should clamp at row 0.
        for _ in 0..10 {
            let a = b.decide(2, 2);
            b.apply_movement(a);
        }
        assert_eq!(b.row, 0);
        assert_eq!(b.col, 4);
    }
}
