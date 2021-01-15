use crate::layer::FeedForwardLayer;
use ndarray::prelude::*;
use std::fmt::Debug;

#[derive(Debug)]
pub struct FeedForwardNetwork {
    pub layers: Vec<Box<dyn FeedForwardLayer>>,
}

pub trait NeuralNetworkTrait: Debug {
    fn shape(&self) -> Vec<usize>;
    fn get_weights(&self) -> Vec<&Array2<f32>>;
    fn get_biases(&self) -> Vec<&Array1<f32>>;
}

pub trait FeedForwardNetworkTrait: NeuralNetworkTrait {
    fn predict(&self, input: &Array1<f32>) -> Array1<f32>;
}

impl FeedForwardNetwork {
    pub fn new(layers: Vec<Box<dyn FeedForwardLayer>>) -> Self {
        FeedForwardNetwork { layers }
    }
}

impl NeuralNetworkTrait for FeedForwardNetwork {
    fn shape(&self) -> Vec<usize> {
        self.layers.iter().map(|l| l.output_size()).collect()
    }

    fn get_weights(&self) -> Vec<&Array2<f32>> {
        self.layers.iter().map(|l| l.get_weights()).collect()
    }

    fn get_biases(&self) -> Vec<&Array1<f32>> {
        self.layers.iter().map(|l| l.get_biases()).collect()
    }
}

impl FeedForwardNetworkTrait for FeedForwardNetwork {
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
    use super::*;
    use crate::layers::{ReLuLayer, SigmoidLayer, SoftmaxLayer};

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
}
