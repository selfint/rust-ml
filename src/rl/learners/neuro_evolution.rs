use crate::rl::agent::Agent;
use crate::rl::agents::network_agent::NetworkAgent;
use crate::rl::environment::Environment;
use crate::rl::learner::Learner;
use ndarray::prelude::*;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_rand::rand_distr::WeightedIndex;
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

// Allows for learning using a genetic algorithm
pub trait Evolve {
    fn mutate(&mut self);
    fn crossover(&self, other: &Self) -> Self;
}

impl Evolve for NetworkAgent {
    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let random_layer = self.network.get_layers_mut().choose_mut(&mut rng).unwrap();

        if rng.gen_bool(0.5) {
            // mutate weight
            let layer_weights = random_layer.get_weights_mut();
            let weight_src = rng.gen_range(0..layer_weights.len_of(Axis(1)));
            let weight_dst = rng.gen_range(0..layer_weights.len_of(Axis(0)));
            layer_weights[[weight_dst, weight_src]] = rng.gen_range(-0.01..0.01);
        } else {
            // mutate bias
            let layer_biases = random_layer.get_biases_mut();
            let bias = rng.gen_range(0..layer_biases.len());
            layer_biases[bias] = rng.gen_range(-0.01..0.01);
        }
    }

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

pub struct NeuroEvolutionLearner<A: Evolve + Agent> {
    agents: Vec<A>,
}

impl<A: Evolve + Agent> NeuroEvolutionLearner<A> {
    pub fn new(agents: Vec<A>) -> Self {
        Self { agents }
    }

    /// use scores to generate new generation using survival of the fittest
    fn new_generation(&mut self, scores: &[f32]) {
        assert_eq!(
            scores.len(),
            self.agents.len(),
            "scores length must match agent amount"
        );

        let scores = arr1(scores);
        let min_score = *scores.min().expect("failed to get min score");
        let max_score = *scores.min().expect("failed to get max score");
        let weights = match min_score == max_score {
            true => Array1::ones(scores.len()),
            false => scores - min_score,
        };
        let weighted_dist = WeightedIndex::new(&weights).unwrap();

        let mut new_generation = vec![];
        let mut rng = thread_rng();
        for _ in 0..self.agents.len() {
            let parents_indices: Vec<usize> =
                (&mut rng).sample_iter(&weighted_dist).take(2).collect();
            let a0 = &self.agents[parents_indices[0]];
            let a1 = &self.agents[parents_indices[1]];
            let mut child = a0.crossover(a1);
            child.mutate();

            new_generation.push(child);
        }

        self.agents = new_generation;
    }
}

impl<A: Evolve + Agent> Learner for NeuroEvolutionLearner<A> {
    fn master<E: Environment>(
        &mut self,
        env: &E,
        epochs: usize,
        params: Option<&HashMap<&str, f32>>,
    ) {
        let params = params.expect("expected Some params, got None");

        // get params
        let agent_amount = *params
            .get("agent_amount")
            .expect("expected 'agent_amount' key");
        assert!(
            ((agent_amount as usize) as f32) - agent_amount <= f32::EPSILON,
            "couldn't losslessly convert agent_amount to usize"
        );
        let agent_amount = agent_amount as usize;

        // create training environments
        let mut environments: Vec<E> = vec![env.clone(); agent_amount];

        // run epochs
        let max_reward = environments[0].max_reward();
        for _e in 0..epochs {
            let mut scores = vec![];

            // evaluate each agent
            for (agent, environment) in self.agents.iter().zip(environments.iter_mut()) {
                let mut score: f32 = 0.;
                while !environment.is_done() && score < max_reward {
                    let action = agent.act(&environment.observe());
                    let reward = environment.step(&action);
                    score += reward;
                }
                scores.push(score);
            }

            // spawn new generation
            self.new_generation(&scores);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Environment;
    use crate::environments::jump::JumpEnvironment;
    use crate::neuron::layer::LayerTrait;
    use crate::neuron::layers::{ReLuLayer, SigmoidLayer, SoftmaxLayer};
    use crate::neuron::networks::feed_forward::FeedForwardNetwork;
    use crate::rl::agents::network_agent::NetworkAgent;
    use crate::rl::Action;

    #[test]
    fn test_neuro_evolution_learner() {
        let env = JumpEnvironment::new(10);
        let agent_amount = 10;
        let env_observation_space = env.observation_space();
        let (env_min_action, env_max_action) = env.action_space();
        let env_max_action = match env_max_action {
            Action::Discrete(a) => a,
            _ => panic!("expected Discrete action, got something else"),
        };
        let env_min_action = match env_min_action {
            Action::Discrete(a) => a,
            _ => panic!("expected Discrete action, got something else"),
        };

        let env_action_space = env_max_action - env_min_action;
        let network_layers: Vec<Box<dyn LayerTrait>> = vec![
            Box::new(ReLuLayer::new(10, env_observation_space)),
            Box::new(SigmoidLayer::new(5, 10)),
            Box::new(SoftmaxLayer::new(env_action_space, 5)),
        ];
        let agents: Vec<NetworkAgent> =
            vec![
                NetworkAgent::new(Box::new(FeedForwardNetwork::new(network_layers)));
                agent_amount
            ];
        let mut learner = NeuroEvolutionLearner::new(agents);
        let mut params = HashMap::with_capacity(1);
        params.insert("agent_amount", 10.);
        learner.master(&env, 10, Some(&params));
    }
}
