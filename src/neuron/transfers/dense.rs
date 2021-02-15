use ndarray::prelude::*;

use crate::neuron::transfers::TransferTrait;

#[derive(Clone, Debug)]
pub struct Dense;

pub fn dense_transfer(
    weights: &Array2<f32>,
    biases: &Array1<f32>,
    input: &Array1<f32>,
) -> Array1<f32> {
    weights.dot(input) + biases
}

impl TransferTrait for Dense {
    fn transfer(
        &self,
        weights: &Array2<f32>,
        biases: &Array1<f32>,
        input: &Array1<f32>,
    ) -> Array1<f32> {
        dense_transfer(weights, biases, input)
    }
}

impl Dense {
    pub fn new() -> Box<dyn TransferTrait> {
        Box::new(Self)
    }
}
