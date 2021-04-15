use ndarray::Array1;

use crate::neuron::networks::NetworkStruct;

pub trait OptimizeOnce {
    fn optimize_once(
        &self,
        network: &mut NetworkStruct,
        input: &Array1<f32>,
        expected: &Array1<f32>,
        learning_rate: f32,
    );
}

pub trait OptimizeBatch {
    fn optimize_batch(
        &self,
        network: &mut NetworkStruct,
        batch_inputs: &[Array1<f32>],
        batch_expected: &[Array1<f32>],
        learning_rate: f32,
    );
}
