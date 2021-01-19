use ndarray::prelude::*;

use crate::neuron::transfers::TransferTrait;

#[derive(Clone, Debug)]
pub struct FullyConnected;

pub fn fully_connected_transfer(
    weights: &Array2<f32>,
    biases: &Array1<f32>,
    input: &Array1<f32>,
) -> Array1<f32> {
    weights.dot(input) + biases
}

impl TransferTrait for FullyConnected {
    fn transfer(
        &self,
        weights: &Array2<f32>,
        biases: &Array1<f32>,
        input: &Array1<f32>,
    ) -> Array1<f32> {
        fully_connected_transfer(weights, biases, input)
    }
}

impl FullyConnected {
    pub fn new() -> Box<dyn TransferTrait> {
        Box::new(Self)
    }
}
