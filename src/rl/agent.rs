use ndarray::prelude::*;

use crate::environment::Action;

pub trait Agent {
    fn act(&self, state: &Array1<f32>) -> Action;
}
