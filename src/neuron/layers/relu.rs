use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::layer::LayerTrait;

#[derive(Clone, PartialEq, Debug)]
pub struct ReLuLayer {
    output_size: usize,
    input_size: usize,
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl ReLuLayer {
    pub fn new(output_size: usize, input_size: usize) -> Self {
        let distribution = Uniform::new(-0.01, 0.01);
        ReLuLayer {
            output_size,
            input_size,
            weights: Array2::random((output_size, input_size), distribution),
            biases: Array1::random(output_size, distribution),
        }
    }
}

impl LayerTrait for ReLuLayer {
    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    fn get_weights(&self) -> &Array2<f32> {
        &self.weights
    }

    fn get_biases(&self) -> &Array1<f32> {
        &self.biases
    }

    fn get_weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }

    fn get_biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.biases
    }

    fn activate(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|&x| if x > 0. { x } else { 0. })
    }
}
