use ndarray::prelude::*;

use crate::neuron::layers::NeuronLayer;
use crate::neuron::networks::Network;
use crate::neuron::networks::Regression;

#[derive(Debug, Clone)]
pub struct StandardFeedForwardNetwork<L: NeuronLayer> {
    layers: Vec<L>,
}

impl<L: NeuronLayer> StandardFeedForwardNetwork<L> {
    pub fn new(layers: Vec<L>) -> StandardFeedForwardNetwork<L> {
        assert!(!layers.is_empty(), "network must have at least one layer");

        StandardFeedForwardNetwork { layers }
    }
}

impl<L: NeuronLayer> Network<L> for StandardFeedForwardNetwork<L> {
    fn len(&self) -> usize {
        self.layers.len()
    }

    fn shape(&self) -> Vec<usize> {
        let mut shape = vec![self.layers[0].input_size()];
        for layer in self.layers.iter() {
            shape.push(layer.output_size());
        }

        shape
    }

    fn get_weights(&self) -> Vec<&Array2<f32>> {
        self.layers.iter().map(|l| l.get_weights()).collect()
    }

    fn get_biases(&self) -> Vec<&Array1<f32>> {
        self.layers.iter().map(|l| l.get_biases()).collect()
    }

    fn get_weights_mut(&mut self) -> Vec<&mut Array2<f32>> {
        self.layers
            .iter_mut()
            .map(|l| l.get_weights_mut())
            .collect()
    }

    fn get_biases_mut(&mut self) -> Vec<&mut Array1<f32>> {
        self.layers.iter_mut().map(|l| l.get_biases_mut()).collect()
    }

    fn get_layers(&self) -> &Vec<L> {
        &self.layers
    }

    fn get_layers_mut(&mut self) -> &mut Vec<L> {
        &mut self.layers
    }
}

impl<L: NeuronLayer> Regression<L> for StandardFeedForwardNetwork<L> {
    fn predict(&self, input: &Array1<f32>) -> Array1<f32> {
        self.layers
            .iter()
            .fold(input.clone(), |prev_layer_output, layer| {
                layer.forward(&prev_layer_output)
            })
    }
}

#[cfg(test)]
mod tests {
    use crate::neuron::activations::{Linear, ReLu, Sigmoid};
    use crate::neuron::layers::Layer;
    use crate::neuron::transfers::FullyConnected;

    use super::*;

    #[test]
    fn test_network_predict() {
        let l1 = Layer::new(3, 2, FullyConnected::new(), ReLu::new());
        let l2 = Layer::new(4, 3, FullyConnected::new(), Sigmoid::new());
        let l3 = Layer::new(1, 4, FullyConnected::new(), Linear::new());

        let network = StandardFeedForwardNetwork::new(vec![l1, l2, l3]);

        let input = [0., 1.];
        let output = network.predict(&arr1(&input));
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_network_is_cloneable() {
        let l1 = Layer::new(3, 2, FullyConnected::new(), ReLu::new());
        let l2 = Layer::new(4, 3, FullyConnected::new(), Sigmoid::new());
        let l3 = Layer::new(1, 4, FullyConnected::new(), Linear::new());

        let layers = vec![l1, l2, l3];
        let network1 = StandardFeedForwardNetwork::new(layers);
        let network2 = network1.clone();

        let input = [0., 1.];
        let output1 = network1.predict(&arr1(&input));
        let output2 = network2.predict(&arr1(&input));

        assert!(
            (output1 - output2).map(|&x| x.abs()).sum() <= f32::EPSILON,
            "cloned networks returns different outputs"
        )
    }
}
