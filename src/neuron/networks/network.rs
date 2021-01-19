use crate::neuron::layers::LayerTrait;
use ndarray::prelude::*;
use std::fmt::Debug;

pub trait Network: Debug + CloneableNetwork {
    fn shape(&self) -> Vec<usize>;
    fn get_weights(&self) -> Vec<&Array2<f32>>;
    fn get_weights_mut(&mut self) -> Vec<&mut Array2<f32>>;
    fn get_biases(&self) -> Vec<&Array1<f32>>;
    fn get_biases_mut(&mut self) -> Vec<&mut Array1<f32>>;
    fn get_layers(&self) -> &Vec<Box<dyn LayerTrait>>;
    fn get_layers_mut(&mut self) -> &mut Vec<Box<dyn LayerTrait>>;
    fn predict(&self, input: &Array1<f32>) -> Array1<f32>;
}

pub trait CloneableNetwork {
    fn clone_network(&self) -> Box<dyn Network>;
}

impl<T: 'static + Network + Clone> CloneableNetwork for T {
    fn clone_network(&self) -> Box<dyn Network> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Network> {
    fn clone(&self) -> Box<dyn Network> {
        self.clone_network()
    }
}
