use ndarray::prelude::*;

use crate::neuron::layers::{Cached, NeuronLayer};

pub trait NetworkTrait<L: NeuronLayer>: Clone {
    fn len(&self) -> usize;
    fn shape(&self) -> Vec<usize>;
    fn get_weights(&self) -> Vec<&Array2<f32>>;
    fn get_weights_mut(&mut self) -> Vec<&mut Array2<f32>>;
    fn get_biases(&self) -> Vec<&Array1<f32>>;
    fn get_biases_mut(&mut self) -> Vec<&mut Array1<f32>>;
    fn get_layers(&self) -> &Vec<L>;
    fn get_layers_mut(&mut self) -> &mut Vec<L>;
}

pub trait FeedForwardNetworkTrait<L: NeuronLayer>: NetworkTrait<L> {
    fn predict(&self, input: &Array1<f32>) -> Array1<f32>;
}

pub trait CachedNetworkTrait<L: NeuronLayer + Cached>: NetworkTrait<L> {
    fn predict_cached(&mut self, input: &Array1<f32>) -> Array1<f32>;
}
