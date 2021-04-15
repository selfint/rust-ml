use ndarray::prelude::*;
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_stats::QuantileExt;

use crate::neuron::layers::LayerStruct;
use crate::neuron::networks::NetworkStruct;
use crate::rl::prelude::*;
use crate::rl::trainers::genetic_algorithm::Evolve;

/// An `Agent` with a `Network` that can `Evolve` and perform `DiscreteAction`s or
/// `ContinuousAction`
#[derive(Clone)]
pub struct NeuroEvolutionAgent {
    network: NetworkStruct,
}

impl NeuroEvolutionAgent {
    pub fn new(network: NetworkStruct) -> Self {
        Self { network }
    }

    fn crossover_weights(
        &self,
        new_layer: &mut LayerStruct,
        other_layer: &LayerStruct,
        rng: &mut ndarray_rand::rand::prelude::ThreadRng,
    ) {
        let layer_weights = new_layer.get_weights_mut();
        let other_weights = other_layer.get_weights();
        for dst in 0..layer_weights.len_of(Axis(0)) {
            for src in 0..layer_weights.len_of(Axis(1)) {
                if rng.gen_bool(0.5) {
                    layer_weights[[dst, src]] = other_weights[[dst, src]];
                }
            }
        }
    }

    fn crossover_biases(
        &self,
        new_layer: &mut LayerStruct,
        other_layer: &LayerStruct,
        rng: &mut ndarray_rand::rand::prelude::ThreadRng,
    ) {
        let layer_biases = new_layer.get_biases_mut();
        let other_biases = other_layer.get_biases();
        for dst in 0..layer_biases.len() {
            if rng.gen_bool(0.5) {
                layer_biases[dst] = other_biases[dst];
            }
        }
    }
}

impl Agent<DiscreteAction> for NeuroEvolutionAgent {
    fn act(&mut self, state: &State) -> DiscreteAction {
        DiscreteAction(self.network.predict(state).argmax().unwrap())
    }
}

impl Agent<ContinuousAction> for NeuroEvolutionAgent {
    fn act(&mut self, state: &State) -> ContinuousAction {
        ContinuousAction(self.network.predict(state))
    }
}

impl Evolve for NeuroEvolutionAgent {
    /// mutate weights and biases of agent's network
    fn mutate(&mut self, mutation_rate: f64) {
        let mut rng = thread_rng();
        for layer_weights in self.network.get_weights_mut() {
            for weight in layer_weights {
                if rng.gen_bool(mutation_rate) {
                    *weight = rng.gen_range(-1.0..1.0);
                }
            }
        }

        for layer_biases in self.network.get_biases_mut() {
            for bias in layer_biases {
                if rng.gen_bool(mutation_rate) {
                    *bias = rng.gen_range(-1.0..1.0);
                }
            }
        }
    }

    /// crossover agent's network with other's network
    fn crossover(&self, other: &Self) -> Self {
        let mut rng = thread_rng();
        let mut new_network = self.network.clone();
        let new_layers = new_network.get_layers_mut();
        let other_layers = other.network.get_layers();

        for (new_layer, other_layer) in new_layers.iter_mut().zip(other_layers.iter()) {
            self.crossover_biases(new_layer, other_layer, &mut rng);
            self.crossover_weights(new_layer, other_layer, &mut rng);
        }

        Self::new(new_network)
    }
}
