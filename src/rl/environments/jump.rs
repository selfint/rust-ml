use crate::rl::environment::Action;
use crate::rl::environment::Environment;
use ndarray::prelude::*;
use ndarray_rand::rand::{thread_rng, Rng};
use std::convert::TryInto;

struct JumpEnvironment {
    size: usize,
    player: (usize, usize),
    walls: Vec<(usize, usize)>,
    ground: usize,
    player_vel: isize,
    done: bool,
}

impl JumpEnvironment {
    pub fn new(size: usize) -> Self {
        let ground = size / 4;
        Self {
            size,
            player: (ground, ground + 1),
            walls: vec![],
            ground,
            player_vel: 0,
            done: false,
        }
    }

    fn spawn_wall(&mut self) {
        let mut rng = thread_rng();
        let wx = self.size - 1;
        let wy1 = rng.gen_range((self.ground + 1)..(self.size - 1));
        let wy2 = rng.gen_range((self.ground + 1)..(self.size - 1));

        self.walls.push((wx, wy1));
        self.walls.push((wx, wy2));
    }

    fn update_player(&mut self, action: &Action) {
        if let Action::Discrete(a) = action {
            if *a == 1 && self.player.1 == self.ground + 1 {
                self.player_vel = 2;
            }
        }

        self.player.1 = (self.player.1 as isize - self.player_vel)
            .try_into()
            .expect("tried to decrease player y below 0");
        if self.player.1 != self.ground + 1 {
            self.player_vel -= 1;
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
                if last_wall.0 < self.size - 5 {
                    self.spawn_wall()
                }
            }
            None => self.spawn_wall(),
        }
    }

    fn check_collision(&self) -> bool {
        self.walls.contains(&self.player)
    }

    fn player_on_wall_column(&self) -> bool {
        self.walls.iter().any(|&(wx, _wy)| wx == self.player.0)
    }
}

impl Environment for JumpEnvironment {
    fn reset(&mut self) {
        self.player = (self.ground, self.ground + 1);
        self.player_vel = 0;
        self.walls = vec![];
        self.done = false;
    }

    fn observe(&self) -> Array1<f32> {
        let mut state = Vec::with_capacity(self.observation_space());
        for row in 0..self.size {
            for col in 0..self.size {
                // ground tile
                state.push(if row == self.ground { 1. } else { 0. });

                // wall tile
                state.push(if self.walls.contains(&(row, col)) {
                    1.
                } else {
                    0.
                });

                // player tile
                state.push(if (row, col) == self.player { 1. } else { 0. });
            }
        }

        arr1(&state)
    }

    fn step(&mut self, action: &Action) -> f32 {
        self.update_player(action);
        self.update_walls();

        if !self.done {
            let crashed = self.check_collision();
            self.done |= crashed;

            if crashed {
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

    fn max_reward(&self) -> f32 {
        200.
    }

    fn action_space(&self) -> (Action, Action) {
        (Action::Discrete(0), Action::Discrete(1))
    }

    fn observation_space(&self) -> usize {
        self.size * self.size * 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_action_kills_player() {
        let mut env = JumpEnvironment::new(5);
        for _ in 0..100 {
            env.step(&Action::Discrete(0));
        }

        assert!(env.is_done());
    }

    #[test]
    fn test_constant_action_kills_player() {
        let mut env = JumpEnvironment::new(10);
        for _ in 0..1000 {
            env.step(&Action::Discrete(1));
        }

        assert!(env.is_done());
    }
}
