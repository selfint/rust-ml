use crate::neuron::layer::FeedForwardLayer;
use crate::neuron::network::{FeedForwardNetwork, FeedForwardNetworkTrait};
use crate::rl::agent::Agent;
use crate::rl::environment::{Action, Environment};
use crate::rl::learner::Learner;
use ndarray::prelude::*;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_rand::rand_distr::WeightedIndex;
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

pub struct NeuroEvolutionAgent {
    network: FeedForwardNetwork,
}

pub trait Evolve {
    fn mutate(&mut self);
    fn crossover(&self, other: &Self) -> Self;
}

impl NeuroEvolutionAgent {
    pub fn new(layers: Vec<Box<dyn FeedForwardLayer>>) -> Self {
        Self {
            network: FeedForwardNetwork::new(layers),
        }
    }

    pub fn new_multiple(agents_layers: Vec<Vec<Box<dyn FeedForwardLayer>>>) -> Vec<Self> {
        let mut agents: Vec<Self> = Vec::with_capacity(agents_layers.len());

        for layers in agents_layers {
            agents.push(NeuroEvolutionAgent::new(layers));
        }

        agents
    }
}

impl Agent for NeuroEvolutionAgent {
    fn act(&self, state: &Array1<f32>) -> Action {
        Action::Discrete(self.network.predict(state).argmax().unwrap())
    }
}

impl Evolve for NeuroEvolutionAgent {
    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let layer = self
            .network
            .layers
            .choose_mut(&mut rng)
            .expect("failed to choose random layer to mutate");

        if rng.gen_bool(0.5) {
            let layer_weights = layer.get_weights_mut();
            let weight_source = rng.gen_range(0..layer_weights.len_of(Axis(0)));
            let weight_dest = rng.gen_range(0..layer_weights.len_of(Axis(1)));
            layer_weights[[weight_source, weight_dest]] = rng.gen_range(-0.01..0.01);
        } else {
            let layer_biases = layer.get_biases_mut();
            let bias = rng.gen_range(0..layer_biases.len_of(Axis(0)));
            layer_biases[bias] = rng.gen_range(-0.01..0.01);
        }
    }

    fn crossover(&self, other: &Self) -> Self {
        todo!()
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
        for e in 0..epochs {
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
    use crate::neuron::layers::{ReLuLayer, SigmoidLayer, SoftmaxLayer};

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
        let mut agents_layers: Vec<Vec<Box<dyn FeedForwardLayer>>> =
            Vec::with_capacity(agent_amount);
        for _ in 0..agent_amount {
            let layers: Vec<Box<dyn FeedForwardLayer>> = vec![
                Box::new(ReLuLayer::new(10, env_observation_space)),
                Box::new(SigmoidLayer::new(5, 10)),
                Box::new(SoftmaxLayer::new(env_action_space, 5)),
            ];
            agents_layers.push(layers);
        }

        let agents: Vec<NeuroEvolutionAgent> = NeuroEvolutionAgent::new_multiple(agents_layers);
        let mut learner = NeuroEvolutionLearner::new(agents);
        let mut params = HashMap::with_capacity(1);
        params.insert("agent_amount", 10.);
        learner.master(&env, 10, Some(&params));
    }
}
