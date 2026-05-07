/// Planner — Stage 6 goal-directed planning.
///
/// Uses the agent's learned model as a simulator to evaluate action sequences.
/// BFS over positions: for each reachable position at depth d, try all 4 actions,
/// predict the resulting state, keep the shortest path to each unique position.
/// Returns the first action of the path that terminates closest to the goal.

use crate::agent::Stage0Agent;
use crate::grid::{Grid, manhattan};
use crate::world::{ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, N_ACTIONS};
use std::collections::{HashMap, VecDeque};

pub struct PlanResult {
    pub action: u8,
    pub predicted_distance: usize,
    pub depth_explored: usize,
    pub states_evaluated: usize,
    pub reached_goal: bool,
}

/// BFS over positions using the agent's model as simulator.
///
/// At each depth, expands all frontier states by trying all 4 actions.
/// Deduplicates by position (keeps shortest path to each position).
/// Early terminates when goal position is found.
pub fn plan_bfs(
    agent: &Stage0Agent,
    state: &Grid,
    goal: (usize, usize),
    max_depth: usize,
) -> PlanResult {
    let start_pos = match state.find_marker() {
        Some(pos) => pos,
        None => return PlanResult {
            action: ACTION_UP,
            predicted_distance: usize::MAX,
            depth_explored: 0,
            states_evaluated: 0,
            reached_goal: false,
        },
    };

    // Check if already at goal
    if start_pos == goal {
        return PlanResult {
            action: ACTION_UP,
            predicted_distance: 0,
            depth_explored: 0,
            states_evaluated: 0,
            reached_goal: true,
        };
    }

    // BFS state: (grid, first_action, depth)
    let mut queue: VecDeque<(Grid, u8, usize)> = VecDeque::new();
    // visited: position → (first_action, distance_to_goal)
    let mut visited: HashMap<(usize, usize), (u8, usize)> = HashMap::new();
    visited.insert(start_pos, (ACTION_UP, manhattan(start_pos, goal)));
    let mut states_evaluated: usize = 0;

    // Seed BFS with depth-1 expansions from start state
    for a in 0..N_ACTIONS as u8 {
        if let Some(pred) = agent.predict_for_planning(a, state) {
            states_evaluated += 1;
            if let Some(pos) = pred.find_marker() {
                if pos == goal {
                    return PlanResult {
                        action: a,
                        predicted_distance: 0,
                        depth_explored: 1,
                        states_evaluated,
                        reached_goal: true,
                    };
                }
                let dist = manhattan(pos, goal);
                if !visited.contains_key(&pos) {
                    visited.insert(pos, (a, dist));
                    if max_depth < 2 {
                        // Don't enqueue if we won't explore further
                    } else {
                        queue.push_back((pred, a, 1));
                    }
                }
            }
        }
    }
    let mut max_depth_reached: usize = 1;

    // Continue BFS for deeper depths
    while let Some((grid, first_action, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        for a in 0..N_ACTIONS as u8 {
            if let Some(pred) = agent.predict_for_planning(a, &grid) {
                states_evaluated += 1;
                if let Some(pos) = pred.find_marker() {
                    if pos == goal {
                        return PlanResult {
                            action: first_action,
                            predicted_distance: 0,
                            depth_explored: depth + 1,
                            states_evaluated,
                            reached_goal: true,
                        };
                    }
                    if !visited.contains_key(&pos) {
                        let dist = manhattan(pos, goal);
                        visited.insert(pos, (first_action, dist));
                        if depth + 1 < max_depth {
                            queue.push_back((pred, first_action, depth + 1));
                        }
                    }
                }
            }
        }
        if depth + 1 > max_depth_reached {
            max_depth_reached = depth + 1;
        }
    }

    // Find the visited position closest to goal
    let (best_action, best_dist) = visited
        .values()
        .min_by_key(|(_, d)| *d)
        .map(|(a, d)| (*a, *d))
        .unwrap_or((ACTION_UP, manhattan(start_pos, goal)));

    PlanResult {
        action: best_action,
        predicted_distance: best_dist,
        depth_explored: max_depth_reached,
        states_evaluated,
        reached_goal: false,
    }
}

/// Greedy baseline: pick the action minimising Manhattan distance to goal.
/// Simulates each action with edge clamping (no model needed).
pub fn greedy_action(pos: (usize, usize), goal: (usize, usize), grid_size: usize) -> u8 {
    let candidates = [
        (ACTION_UP,    (pos.0.saturating_sub(1), pos.1)),
        (ACTION_DOWN,  (std::cmp::min(pos.0 + 1, grid_size - 1), pos.1)),
        (ACTION_LEFT,  (pos.0, pos.1.saturating_sub(1))),
        (ACTION_RIGHT, (pos.0, std::cmp::min(pos.1 + 1, grid_size - 1))),
    ];

    candidates
        .iter()
        .min_by_key(|(_, new_pos)| manhattan(*new_pos, goal))
        .map(|(a, _)| *a)
        .unwrap_or(ACTION_UP)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_toward_goal() {
        // Agent at (2,2), goal at (0,2) → should go UP
        assert_eq!(greedy_action((2, 2), (0, 2), 5), ACTION_UP);
        // Agent at (2,2), goal at (4,2) → should go DOWN
        assert_eq!(greedy_action((2, 2), (4, 2), 5), ACTION_DOWN);
        // Agent at (2,2), goal at (2,0) → should go LEFT
        assert_eq!(greedy_action((2, 2), (2, 0), 5), ACTION_LEFT);
        // Agent at (2,2), goal at (2,4) → should go RIGHT
        assert_eq!(greedy_action((2, 2), (2, 4), 5), ACTION_RIGHT);
    }

    #[test]
    fn test_greedy_at_goal() {
        // Agent at goal — any action is fine, should not panic
        let a = greedy_action((2, 2), (2, 2), 5);
        assert!(a < N_ACTIONS as u8);
    }

    #[test]
    fn test_greedy_at_edge() {
        // Agent at (0,0), goal at (0,0) — at edge, at goal
        let a = greedy_action((0, 0), (4, 4), 5);
        // Should go DOWN or RIGHT (both reduce distance equally)
        assert!(a == ACTION_DOWN || a == ACTION_RIGHT);
    }

    #[test]
    fn test_greedy_diagonal() {
        // Agent at (0,0), goal at (4,4) — diagonal
        let a = greedy_action((0, 0), (4, 4), 5);
        // Both DOWN and RIGHT reduce distance by 1
        assert!(a == ACTION_DOWN || a == ACTION_RIGHT);
    }

    #[test]
    fn test_plan_empty_model() {
        // Agent with no experience — planner should still return an action
        use crate::config::M0Config;
        let agent = Stage0Agent::new(M0Config::default());
        let mut state = Grid::filled(5, 5, 0);
        state.set(2, 2, 1);
        let result = plan_bfs(&agent, &state, (0, 0), 3);
        // Should return something (falls back to start position distance)
        assert!(result.action < N_ACTIONS as u8);
        assert!(!result.reached_goal);
    }

    #[test]
    fn test_plan_with_rules() {
        // Agent with rules should find optimal path
        use crate::config::M0Config;
        use crate::world::MicroWorld;

        let mut config = M0Config::default();
        config.rules_enabled = true;
        let mut agent = Stage0Agent::new(config);
        let mut world = MicroWorld::new(5);

        // Train: run 200 episodes to build experience, then extract rules
        for i in 0..200 {
            let state = world.observe();
            let action = (i % 4) as u8;
            let pred_a = agent.predict_path_a(action, &state);
            let pred_b = agent.predict_path_b(action);
            let _exec = world.apply(action);
            let actual = world.observe();
            let hit_a = pred_a.as_ref() == Some(&actual);
            let hit_b = pred_b.as_ref() == Some(&actual);
            agent.record(action, state, actual, hit_a, hit_a, hit_a, hit_b, &[], false, false);
        }
        agent.extract_rules();

        // Now plan: agent at (2,2), goal at (0,0)
        let state = world.observe();
        let result = plan_bfs(&agent, &state, (0, 0), 5);
        // With rules, the planner should find a path
        assert!(result.states_evaluated > 0);
        // The action should move toward (0,0) — UP or LEFT
        assert!(result.action == ACTION_UP || result.action == ACTION_LEFT);
    }

    #[test]
    fn test_plan_depth_1() {
        // At depth 1, planner is essentially greedy-with-model
        use crate::config::M0Config;
        use crate::world::MicroWorld;

        let mut config = M0Config::default();
        config.rules_enabled = true;
        let mut agent = Stage0Agent::new(config);
        let mut world = MicroWorld::new(5);

        // Train
        for i in 0..200 {
            let state = world.observe();
            let action = (i % 4) as u8;
            let pred_a = agent.predict_path_a(action, &state);
            let pred_b = agent.predict_path_b(action);
            let _exec = world.apply(action);
            let actual = world.observe();
            let hit_a = pred_a.as_ref() == Some(&actual);
            let hit_b = pred_b.as_ref() == Some(&actual);
            agent.record(action, state, actual, hit_a, hit_a, hit_a, hit_b, &[], false, false);
        }
        agent.extract_rules();

        let state = world.observe();
        let result = plan_bfs(&agent, &state, (0, 0), 1);
        assert_eq!(result.depth_explored, 1);
    }
}
