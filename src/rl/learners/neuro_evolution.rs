use crate::rl::agent::Agent;
use crate::rl::agents::network_agent::NetworkAgent;
use crate::rl::environment::Environment;
use crate::rl::learner::Learner;
use crate::rl::{Param, Reward};
use ndarray::prelude::*;
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_rand::rand_distr::WeightedIndex;
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

// Allows for learning using a genetic algorithm
pub trait Evolve {
    fn mutate(&mut self, mutation_rate: f64);
    fn crossover(&self, other: &Self) -> Self;
}

impl Evolve for NetworkAgent {
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

pub struct NeuroEvolutionLearner<A: Evolve + Agent> {
    agent: A,
    agent_amount: usize,
    mutation_rate: f64,
}

impl<A: Evolve + Agent> NeuroEvolutionLearner<A> {
    /// use scores to generate new generation using survival of the fittest
    fn new_generation(&mut self, old_generation: Vec<&A>, scores: &Array1<Reward>) -> Vec<A> {
        assert_eq!(
            scores.len(),
            self.agent_amount,
            "scores length must match agent amount"
        );

        let min_score = scores.min().expect("failed to get min score");
        let max_score = scores.min().expect("failed to get max score");
        let weights = match min_score == max_score {
            true => Array1::ones(scores.len()),
            false => scores - *min_score,
        };
        let weighted_dist = WeightedIndex::new(&weights).unwrap();

        let mut new_generation = vec![];
        let mut rng = thread_rng();
        for _ in 0..self.agent_amount {
            let parents_indices: Vec<usize> =
                (&mut rng).sample_iter(&weighted_dist).take(2).collect();
            let a0 = &old_generation[parents_indices[0]];
            let a1 = &old_generation[parents_indices[1]];
            let mut child = a0.crossover(a1);
            child.mutate(self.mutation_rate);

            new_generation.push(child);
        }

        new_generation
    }
}

impl<A: Evolve + Agent> Learner<A> for NeuroEvolutionLearner<A> {
    fn new(agent: A, params: Option<&HashMap<&str, Param>>) -> Self {
        let params = params.expect("expected Some params, got None");

        // parse params
        let agent_amount = if let Param::Usize(agent_amount) = *params
            .get("agent_amount")
            .expect("expected 'agent_amount' key")
        {
            agent_amount
        } else {
            panic!("expected agent_amount to be Usize")
        };

        let mutation_rate = if let Param::Float(mutation_rate) = *params
            .get("mutation_rate")
            .expect("expected 'mutation_rate' key")
        {
            mutation_rate as f64
        } else {
            panic!("expected mutation_rate to be Float")
        };

        Self {
            agent,
            agent_amount,
            mutation_rate,
        }
    }

    fn master<E: Environment>(&mut self, env: &E, epochs: usize, verbose: bool) -> A {
        // create multiple agents and an env for each one
        let mut agents = vec![self.agent.clone(); self.agent_amount];
        let mut best_agent = self.agent.clone();
        let mut envs = vec![env.clone(); self.agent_amount];

        // run epochs
        let max_reward = env.max_reward();
        for e in 0..epochs {
            let mut scores = Array1::zeros(self.agent_amount);

            // evaluate each agent
            for (i, (agent, environment)) in agents.iter().zip(envs.iter_mut()).enumerate() {
                let mut score = 0.;
                environment.reset();

                while !environment.is_done() && score < max_reward {
                    let action = agent.act(&environment.observe());
                    let reward = environment.step(&action);
                    score += reward;
                }

                scores[i] = score;
            }

            if verbose {
                println!(
                    "epoch {} | max score: {} | avg scores: {} | min score: {}",
                    e,
                    scores.max().unwrap(),
                    scores.sum() / self.agent_amount as f32,
                    scores.min().unwrap()
                );
            }

            // HILLCLIMBING: save the best agent from each generation
            best_agent = agents[scores.argmax().unwrap()].clone();

            // spawn new generation
            agents = self.new_generation(agents.iter().collect(), &scores);
            agents[0] = best_agent.clone();
        }

        best_agent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::layer::LayerTrait;
    use crate::neuron::layers::{ReLuLayer, SigmoidLayer, SoftmaxLayer};
    use crate::neuron::networks::feed_forward::FeedForwardNetwork;
    use crate::rl::agents::network_agent::NetworkAgent;
    use crate::rl::environment::Environment;
    use crate::rl::environments::jump::JumpEnvironment;

    #[test]
    fn test_neuro_evolution_learner() {
        let env = JumpEnvironment::new(10);
        let env_observation_space = env.observation_space();
        let env_action_space = env.action_space();
        let network_layers: Vec<Box<dyn LayerTrait>> = vec![
            Box::new(ReLuLayer::new(10, env_observation_space)),
            Box::new(SigmoidLayer::new(5, 10)),
            Box::new(SoftmaxLayer::new(env_action_space, 5)),
        ];
        let agent = NetworkAgent::new(Box::new(FeedForwardNetwork::new(network_layers)));
        let mut params = HashMap::with_capacity(1);
        params.insert("agent_amount", Param::Usize(10));
        params.insert("mutation_rate", Param::Float(0.01));
        let mut learner = NeuroEvolutionLearner::new(agent, Some(&params));
        let epochs = 10;
        learner.master(&env, epochs, false);
    }
}
