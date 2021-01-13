use ndarray::prelude::*;


pub trait Environment {
    type Action;

    fn reset(&mut self);
    fn observe(&self) -> Array1<f32>;
    fn step(&mut self, action: &Self::Action) -> f32;
    fn is_done(&self) -> bool;
    fn max_reward(&self) -> f32;
}

