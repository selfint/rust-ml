use crate::rl::prelude::*;

pub trait Environment<A: Action>: Clone {
    fn reset(&mut self);
    fn observe(&self) -> State;
    fn step(&mut self, action: &A) -> Reward;
    fn is_done(&self) -> bool;
    fn max_reward(&self) -> Reward;
    fn action_space(&self) -> usize;
    fn observation_space(&self) -> usize;
}
