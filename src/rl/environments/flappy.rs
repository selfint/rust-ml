use std::convert::TryInto;
use std::fmt::Display;

use ndarray::prelude::*;
use ndarray_rand::rand::{thread_rng, Rng};

use crate::rl::prelude::*;

#[derive(Clone)]
pub struct FlappyEnvironment {
    size: usize,
    hole_size: usize,
    player: (usize, usize),
    player_vel: isize,
    walls: Vec<(usize, usize)>,
    done: bool,
}

impl FlappyEnvironment {
    pub fn new(size: usize, hole_size: usize) -> Self {
        FlappyEnvironment {
            size,
            hole_size,
            player: (size / 3, size / 2),
            player_vel: 0,
            walls: vec![],
            done: false,
        }
    }

    /// update player position and return true if player is out-of-bounds
    fn update_player(&mut self, action: &DiscreteAction) -> bool {
        if self.player_vel <= 2 && action.0 == 1 {
            self.player_vel += 2
        } else if self.player_vel >= -2 {
            self.player_vel -= 1;
        }

        let new_player_y = self.player.1 as isize + self.player_vel;

        // update player pos only if it didn't crash
        if 0 <= new_player_y && new_player_y < self.size as isize {
            self.player.1 = new_player_y.try_into().unwrap();
            false
        } else {
            true
        }
    }

    fn update_walls(&mut self) {
        self.walls = self
            .walls
            .iter()
            .filter(|&(wx, _wy)| *wx > 0)
            .map(|&(wx, wy)| (wx - 1, wy))
            .collect();

        match self.walls.last() {
            Some(last_wall) => {
                if last_wall.0 < self.size - 8 {
                    self.spawn_wall()
                }
            }
            None => self.spawn_wall(),
        }
    }

    fn spawn_wall(&mut self) {
        let mut rng = thread_rng();
        let hole = rng.gen_range(0..(self.size - self.hole_size));

        let mut walls: Vec<(usize, usize)> = (0..self.size)
            .filter_map(|w| {
                if w < hole || w > hole + self.hole_size  {
                    Some((self.size - 1, w))
                } else {
                    None
                }
            })
            .collect();
        self.walls.append(&mut walls);
    }

    fn get_tile_vec(&self, row: usize, col: usize) -> Vec<f32> {
        let wall_tile = if self.walls.contains(&(row, col)) {
            1.
        } else {
            0.
        };
        let player_tile = if (row, col) == self.player { 1. } else { 0. };

        vec![wall_tile, player_tile]
    }

    fn check_wall_collision(&self) -> bool {
        self.walls.contains(&self.player)
    }

    fn player_on_wall_column(&self) -> bool {
        self.walls.iter().any(|&(wx, _wy)| wx == self.player.0)
    }
}

impl Environment<DiscreteAction> for FlappyEnvironment {
    fn reset(&mut self) {
        self.player = (self.size / 3, self.size / 2);
        self.player_vel = 0;
        self.walls = vec![];
        self.done = false;
    }

    fn observe(&self) -> State {
        let mut state = Vec::with_capacity(self.observation_space());

        for row in 0..self.size {
            for col in 0..self.size {
                state.append(&mut self.get_tile_vec(row, col));
            }
        }

        arr1(&state)
    }

    fn step(&mut self, action: &DiscreteAction) -> Reward {
        let border_crash = self.update_player(&action);

        self.update_walls();

        if !self.done {
            let wall_crash = self.check_wall_collision();
            self.done |= wall_crash || border_crash;

            if border_crash {
                -2.
            } else if wall_crash {
                -1.
            } else if self.player_on_wall_column() {
                1.
            } else {
                0.
            }
        } else {
            0.
        }
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn max_reward(&self) -> Reward {
        200.
    }

    fn action_space(&self) -> usize {
        2
    }

    fn observation_space(&self) -> usize {
        self.size * self.size * 2
    }
}

impl Display for FlappyEnvironment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.observe();
        let mut tiles = Vec::with_capacity(self.observation_space() / 2);
        for t in (0..self.observation_space()).step_by(2) {
            let (wall, player) = (state[t], state[t + 1]);
            tiles.push(if player > 0. {
                if wall > 0. {
                    '💀'
                } else {
                    '🐦'
                }
            } else if wall > 0. {
                '🧱'
            } else {
                '🟦'
            });
        }

        // transpose tiles
        let tiles: String = Array2::from_shape_vec((self.size, self.size), tiles)
            .expect("failed to digest tiles into matrix")
            .reversed_axes()
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if i % self.size == 0 {
                    let mut string = String::from('\n');
                    string.push(t);
                    string
                } else {
                    String::from(t)
                }
            })
            .collect();

        // reverse tile lines
        let tiles = tiles.lines().rev().collect::<Vec<&str>>();
        let info = format!("player: {:?}", self.player);
        let mut screen = tiles;
        screen.push(&info);

        f.write_str(&screen.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_action_kills_player() {
        let mut env = FlappyEnvironment::new(10, 3);
        for _ in 0..1000 {
            env.step(&DiscreteAction(0));
            if env.is_done() {
                break;
            }
        }

        assert!(env.is_done());
    }

    #[test]
    fn test_constant_action_kills_player() {
        let mut env = FlappyEnvironment::new(10, 3);
        for _ in 0..1000 {
            env.step(&DiscreteAction(1));
            if env.is_done() {
                break;
            }
        }

        assert!(env.is_done());
    }
}
