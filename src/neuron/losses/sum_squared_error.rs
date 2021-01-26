use ndarray::Array1;

use crate::neuron::losses::{Loss, LossTrait};

#[derive(Clone, Debug)]
pub struct SSE;

pub fn sse_loss(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    (prediction - expected).map(|e| e * e)
}

pub fn sse_derivative(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    2. * (prediction - expected)
}

impl LossTrait for SSE {
    fn loss(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
        sse_loss(prediction, expected)
    }

    fn derivative(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
        sse_derivative(prediction, expected)
    }
}

impl SSE {
    pub fn new() -> Loss {
        Box::new(Self)
    }
}
