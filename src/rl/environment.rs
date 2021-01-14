use ndarray::prelude::*;

pub enum Action {
    Discrete(usize),
    Continuous(Array1<f32>)
}

pub trait Environment {
    fn reset(&mut self);
    fn observe(&self) -> Array1<f32>;
    fn step(&mut self, action: &Action) -> f32;
    fn is_done(&self) -> bool;
    fn max_reward(&self) -> f32;
}

