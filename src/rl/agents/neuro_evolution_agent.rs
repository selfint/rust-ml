use std::marker::PhantomData;

use ndarray::prelude::*;
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_stats::QuantileExt;

use crate::neuron::layers::Layer;
use crate::neuron::networks::Regression;
use crate::rl::prelude::*;
use crate::rl::trainers::neuro_evolution_internals::Evolve;

#[derive(Clone)]
pub struct NeuroEvolutionAgent<N, L>
where
    L: Layer,
    N: Regression<L>,
{
    network: N,
    phantom: PhantomData<L>,
}

impl<N, L> NeuroEvolutionAgent<N, L>
where
    L: Layer,
    N: Regression<L>,
{
    pub fn new(network: N) -> Self {
        Self {
            network,
            phantom: PhantomData,
        }
    }
}

impl<N, L> Agent<DiscreteAction> for NeuroEvolutionAgent<N, L>
where
    L: Layer,
    N: Regression<L>,
{
    fn act(&mut self, state: &State) -> DiscreteAction {
        DiscreteAction(self.network.predict(state).argmax().unwrap())
    }
}

impl<N, L> Agent<ContinuousAction> for NeuroEvolutionAgent<N, L>
where
    L: Layer,
    N: Regression<L>,
{
    fn act(&mut self, state: &State) -> ContinuousAction {
        ContinuousAction(self.network.predict(state))
    }
}

impl<N, L> Evolve for NeuroEvolutionAgent<N, L>
where
    L: Layer,
    N: Regression<L>,
{
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
            // crossover biases
            {
                let layer_biases = new_layer.get_biases_mut();
                let other_biases = other_layer.get_biases();
                for dst in 0..layer_biases.len() {
                    if rng.gen_bool(0.5) {
                        layer_biases[dst] = other_biases[dst];
                    }
                }
            }

            // crossover weights
            {
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
        }

        Self::new(new_network)
    }
}
