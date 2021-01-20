use crate::neuron::layers::Cached;
use crate::neuron::networks::{NetworkTrait, CachedNetworkTrait};
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct CachedNetwork<L: Cached> {
    layers: Vec<L>,
}

impl<L: Cached> CachedNetwork<L> {
    pub fn new(layers: Vec<L>) -> CachedNetwork<L> {
        assert!(!layers.is_empty(), "network must have at least one layer");

        CachedNetwork { layers }
    }
}

impl<L: Cached> NetworkTrait<L> for CachedNetwork<L> {
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

impl<L: Cached> CachedNetworkTrait<L> for CachedNetwork<L> {
    fn predict_cached(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.layers
            .iter_mut()
            .fold(input.clone(), |prev_layer_output, layer| {
                layer.forward_cached(&prev_layer_output)
            })
    }
}

