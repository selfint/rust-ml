use ndarray::Array1;

use crate::neuron::losses::{Loss, LossTrait};

#[derive(Clone, Debug)]
pub struct MSE;

pub fn mse_loss(prediction: &Array1<f32>, expected: &Array1<f32>) -> f32 {
    (prediction - expected).map(|e| e * e).mean().unwrap()
}

pub fn mse_derivative(prediction: &Array1<f32>, expected: &Array1<f32>) -> f32 {
    (1. / 2. * prediction.len() as f32) * (prediction - expected).sum()
}

impl LossTrait for MSE {
    fn loss(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> f32 {
        mse_loss(prediction, expected)
    }

    fn derivative(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> f32 {
        mse_derivative(prediction, expected)
    }
}

impl MSE {
    pub fn new() -> Loss {
        Box::new(Self)
    }
}
