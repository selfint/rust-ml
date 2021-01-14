use crate::rl::environment::Action;
use crate::rl::environment::Environment;
use ndarray::prelude::*;
use ndarray_rand::rand::{thread_rng, Rng};

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
}

impl Environment for JumpEnvironment {
    fn reset(&mut self) {
        unimplemented!()
    }

    fn observe(&self) -> Array1<f32> {
        unimplemented!()
    }

    fn step(&mut self, action: &Action) -> f32 {
        if let Action::Discrete(a) = action {
            if *a == 1 && self.player.1 == self.ground + 1 {
                self.player_vel = 2;
            }
        }

        self.walls = self
            .walls
            .iter()
            .filter(|&(wx, _wy)| *wx > 0)
            .map(|&(wx, wy)| (wx - 1, wy))
            .collect();

        if let Some(last_wall) = self.walls.last() {
            if last_wall.0 < self.size - 5 {
                let mut rng = thread_rng();
                let wx = self.size - 1;
                let wy1 = rng.gen_range((self.ground + 1)..(self.size - 1));
                let wy2 = rng.gen_range((self.ground + 1)..(self.size - 1));

                self.walls.push((wx, wy1));
                self.walls.push((wx, wy2));
            }
        } else {
            let mut rng = thread_rng();
            let wx = self.size - 1;
            let wy1 = rng.gen_range((self.ground + 1)..(self.size - 1));
            let wy2 = wy1 + 1;

            self.walls.push((wx, wy1));
            self.walls.push((wx, wy2));
        }

        if !self.done {
            let crashed = self.walls.contains(&self.player);
            self.done |= crashed;

            if crashed {
                -1.
            } else if self.walls.iter().any(|&(wx, _wy)| wx == self.player.0) {
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
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_action_kills_player() {
        let mut env = JumpEnvironment::new(5);
        for _ in 0..100 {
            eprintln!("walls={:?}, player={:?}", env.walls, env.player);
            env.step(&Action::Discrete(0));
        }

        assert!(env.is_done());
    }
}
