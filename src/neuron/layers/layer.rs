use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::neuron::activations::ActivationStruct;
use crate::neuron::transfers::TransferStruct;

#[derive(Debug, Clone)]
pub struct LayerStruct {
    outputs: usize,
    inputs: usize,
    transfer_fn: TransferStruct,
    activation_fn: ActivationStruct,
    weights: Array2<f32>,
    biases: Array1<f32>,
    input_value: Option<Array1<f32>>,
    transfer_value: Option<Array1<f32>>,
    activation_value: Option<Array1<f32>>,
}

impl LayerStruct {
    pub fn new(
        outputs: usize,
        inputs: usize,
        transfer_fn: TransferStruct,
        activation_fn: ActivationStruct,
    ) -> Self {
        let distribution = Uniform::new(-0.01, 0.01);
        Self {
            outputs,
            inputs,
            transfer_fn,
            activation_fn,
            weights: Array2::random((outputs, inputs), distribution),
            biases: Array1::random(outputs, distribution),
            input_value: None,
            transfer_value: None,
            activation_value: None,
        }
    }

    pub fn input_size(&self) -> usize {
        self.inputs
    }
    pub fn output_size(&self) -> usize {
        self.outputs
    }
    pub fn get_weights(&self) -> &Array2<f32> {
        &self.weights
    }
    pub fn get_weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }
    pub fn get_biases(&self) -> &Array1<f32> {
        &self.biases
    }
    pub fn get_biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.biases
    }
    pub fn get_input(&self) -> Option<&Array1<f32>> {
        self.input_value.as_ref()
    }
    pub fn set_input(&mut self, input: Array1<f32>) {
        self.input_value = Some(input);
    }
    pub fn get_transfer(&self) -> Option<&Array1<f32>> {
        self.transfer_value.as_ref()
    }
    pub fn set_transfer(&mut self, transfer: Array1<f32>) {
        self.transfer_value = Some(transfer);
    }
    pub fn get_activation(&self) -> Option<&Array1<f32>> {
        self.activation_value.as_ref()
    }
    pub fn set_activation(&mut self, activation: Array1<f32>) {
        self.activation_value = Some(activation);
    }

    pub fn apply_transfer(&self, inputs: &Array1<f32>) -> Array1<f32> {
        self.transfer_fn
            .transfer(&self.weights, &self.biases, inputs)
    }

    pub fn apply_activation(&self, transfer: &Array1<f32>) -> Array1<f32> {
        self.activation_fn.activate(transfer)
    }
    pub fn apply_derivation(&self, transfer: &Array1<f32>) -> Array1<f32> {
        self.activation_fn.derive(transfer)
    }

    pub fn forward(&self, inputs: &Array1<f32>) -> Array1<f32> {
        self.apply_activation(&self.apply_transfer(inputs))
    }

    pub fn forward_cached(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let transfer = self.apply_transfer(input);
        let activation = self.apply_activation(&transfer);

        self.set_input(input.clone());
        self.set_transfer(transfer);
        self.set_activation(activation.clone());

        activation
    }
}

pub trait Layer: Clone {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

    fn get_weights(&self) -> &Array2<f32>;
    fn get_weights_mut(&mut self) -> &mut Array2<f32>;

    fn get_biases(&self) -> &Array1<f32>;
    fn get_biases_mut(&mut self) -> &mut Array1<f32>;

    fn apply_transfer(&self, input: &Array1<f32>) -> Array1<f32>;
    fn apply_activation(&self, transfer: &Array1<f32>) -> Array1<f32>;

    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.apply_activation(&self.apply_transfer(input))
    }
}

pub trait Cached: Layer {
    fn get_input(&self) -> Option<&Array1<f32>>;
    fn get_transfer(&self) -> Option<&Array1<f32>>;
    fn get_activation(&self) -> Option<&Array1<f32>>;

    fn cache_input(&mut self, input: Array1<f32>);
    fn cache_transfer(&mut self, transfer: Array1<f32>);
    fn cache_activation(&mut self, activation: Array1<f32>);

    fn apply_activation_derivative(&self, transfer: &Array1<f32>) -> Array1<f32>;

    fn forward_cached(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let transfer = self.apply_transfer(input);
        let activation = self.apply_activation(&transfer);

        self.cache_input(input.clone());
        self.cache_transfer(transfer);
        self.cache_activation(activation.clone());

        activation
    }
}

#[cfg(test)]
mod tests {
    use crate::neuron::activations::linear;
    use crate::neuron::transfers::dense;

    use super::*;

    #[test]
    fn test_layer() {
        let layer = LayerStruct::new(3, 2, dense(), linear());
        let output = layer.forward(&arr1(&[1., 0.]));
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_sizes() {
        let layer = LayerStruct::new(3, 2, dense(), linear());
        assert_eq!(layer.input_size(), 2);
        assert_eq!(layer.output_size(), 3);
    }
}
