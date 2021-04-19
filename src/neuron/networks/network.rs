use ndarray::prelude::*;

use crate::neuron::layers::Layer;

#[derive(Debug, Clone)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn shape(&self) -> Vec<usize> {
        let mut shape = vec![self.layers[0].input_size()];
        for layer in self.layers.iter() {
            shape.push(layer.output_size());
        }

        shape
    }

    pub fn get_weights(&self) -> Vec<&Array2<f32>> {
        self.layers.iter().map(|l| l.get_weights()).collect()
    }

    pub fn get_biases(&self) -> Vec<&Array1<f32>> {
        self.layers.iter().map(|l| l.get_biases()).collect()
    }

    pub fn get_weights_mut(&mut self) -> Vec<&mut Array2<f32>> {
        self.layers
            .iter_mut()
            .map(|l| l.get_weights_mut())
            .collect()
    }

    pub fn get_biases_mut(&mut self) -> Vec<&mut Array1<f32>> {
        self.layers.iter_mut().map(|l| l.get_biases_mut()).collect()
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    pub fn get_layers_mut(&mut self) -> &mut Vec<Layer> {
        &mut self.layers
    }
    pub fn predict(&self, input: &Array1<f32>) -> Array1<f32> {
        self.layers
            .iter()
            .fold(input.clone(), |prev_layer_output, layer| {
                layer.forward(&prev_layer_output)
            })
    }
    pub fn predict_cached(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.layers
            .iter_mut()
            .fold(input.clone(), |prev_layer_output, layer| {
                layer.forward_cached(&prev_layer_output)
            })
    }
}
