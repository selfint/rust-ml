use ndarray::prelude::*;


pub trait NeuronLayer: Clone {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

    fn get_weights(&self) -> &Array2<f32>;
    fn get_weights_mut(&mut self) -> &mut Array2<f32>;

    fn get_biases(&self) -> &Array1<f32>;
    fn get_biases_mut(&mut self) -> &mut Array1<f32>;

    fn apply_transfer(&self, input: &Array1<f32>) -> Array1<f32>;
    fn apply_activation(&self, transfer: &Array1<f32>) -> Array1<f32>;

    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.apply_activation(&self.apply_transfer(input))
    }
}

pub trait Cached: NeuronLayer {
    fn get_input(&self) -> Option<&Array1<f32>>;
    fn get_transfer(&self) -> Option<&Array1<f32>>;
    fn get_activation(&self) -> Option<&Array1<f32>>;

    fn cache_input(&mut self, input: Array1<f32>);
    fn cache_transfer(&mut self, transfer: Array1<f32>);
    fn cache_activation(&mut self, activation: Array1<f32>);

    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
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
    use super::*;
    use crate::neuron::activations::Linear;
    use crate::neuron::layers::Layer;
    use crate::neuron::transfers::FullyConnected;

    #[test]
    fn test_relu_layer() {
        let mut layer = Layer::new(3, 2, FullyConnected::new(), Linear::new());
        let output = layer.forward(&arr1(&[1., 0.]));
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn size_test() {
        let layer = Layer::new(3, 2, FullyConnected::new(), Linear::new());
        assert_eq!(layer.input_size(), 2);
        assert_eq!(layer.output_size(), 3);
    }
}
