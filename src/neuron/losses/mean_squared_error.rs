use ndarray::Array1;

use crate::neuron::losses::{Loss, LossTrait};

#[derive(Clone, Debug)]
pub struct MSE;

pub fn mse(prediction: &Array1<f32>, expected: &Array1<f32>) -> f32 {
    (prediction - expected).map(|e| e * e).mean().unwrap()
}

impl LossTrait for MSE {
    fn loss(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> f32 {
        mse(prediction, expected)
    }
}

impl MSE {
    pub fn new() -> Loss {
        Box::new(Self)
    }
}
