use ndarray::Array1;

use crate::neuron::losses::{Loss, LossTrait};

#[derive(Clone, Debug)]
pub struct CCE;

pub fn cce_loss(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    // can we only calculate the negative log of the prediction for the correct index?
    // e.g. pred = [1,2,3] exp = [1,0,0] => loss = -ln(1)
    -1. * prediction.map(|x| x.ln()) * expected
}

pub fn cce_derivative(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    -1. * expected / prediction
}

impl LossTrait for CCE {
    fn loss(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
        cce_loss(prediction, expected)
    }

    fn derivative(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
        cce_derivative(prediction, expected)
    }
}

impl CCE {
    pub fn new() -> Loss {
        Box::new(Self)
    }
}
