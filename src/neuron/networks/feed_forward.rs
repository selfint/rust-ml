use crate::neuron::layers::LayerTrait;
use crate::neuron::networks::Network;
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct FeedForwardNetwork {
    layers: Vec<Box<dyn LayerTrait>>,
}

impl FeedForwardNetwork {
    pub fn new(layers: Vec<Box<dyn LayerTrait>>) -> Self {
        FeedForwardNetwork { layers }
    }
}

impl Network for FeedForwardNetwork {
    fn shape(&self) -> Vec<usize> {
        self.layers.iter().map(|l| l.output_size()).collect()
    }

    fn get_weights(&self) -> Vec<&Array2<f32>> {
        self.layers.iter().map(|l| l.get_weights()).collect()
    }

    fn get_biases(&self) -> Vec<&Array1<f32>> {
        self.layers.iter().map(|l| l.get_biases()).collect()
    }

    fn predict(&self, input: &Array1<f32>) -> Array1<f32> {
        self.layers
            .iter()
            .fold(input.clone(), |prev_layer_output, layer| {
                layer.forward(&prev_layer_output)
            })
    }

    fn get_layers(&self) -> &Vec<Box<dyn LayerTrait>> {
        &self.layers
    }

    fn get_layers_mut(&mut self) -> &mut Vec<Box<dyn LayerTrait>> {
        &mut self.layers
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::layers::{ReLuLayer, SigmoidLayer, SoftmaxLayer};

    #[test]
    fn test_network_predict() {
        let l1 = Box::new(ReLuLayer::new(3, 2));
        let l2 = Box::new(SigmoidLayer::new(4, 3));
        let l3 = Box::new(SoftmaxLayer::new(1, 4));

        let network = FeedForwardNetwork::new(vec![l1, l2, l3]);

        let input = [0., 1.];
        let output = network.predict(&arr1(&input));
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_network_is_cloneable() {
        let l1 = Box::new(ReLuLayer::new(3, 2));
        let l2 = Box::new(SigmoidLayer::new(4, 3));
        let l3 = Box::new(SoftmaxLayer::new(1, 4));

        let layers: Vec<Box<dyn LayerTrait>> = vec![l1, l2, l3];
        let network1 = FeedForwardNetwork::new(layers.clone());
        let network2 = FeedForwardNetwork::new(layers);

        let input = [0., 1.];
        let output1 = network1.predict(&arr1(&input));
        let output2 = network2.predict(&arr1(&input));

        assert!(
            (output1 - output2).map(|&x| x.abs()).sum() <= f32::EPSILON,
            "cloned networks returns different outputs"
        )
    }
}
