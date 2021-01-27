use ndarray::prelude::*;
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_rand::rand_distr::WeightedIndex;
use ndarray_stats::QuantileExt;

use crate::rl::prelude::*;

// Allows for learning using a genetic algorithm
pub trait Evolve {
    fn mutate(&mut self, mutation_rate: f64);
    fn crossover(&self, other: &Self) -> Self;
}

pub struct GeneticAlgorithmLearner {
    agent_amount: usize,
    mutation_rate: f64,
}

impl GeneticAlgorithmLearner {
    pub fn new(agent_amount: usize, mutation_rate: f64) -> Self {
        Self {
            agent_amount,
            mutation_rate,
        }
    }

    /// use scores to generate new generation using survival of the fittest
    fn new_generation<AC, AG>(
        &mut self,
        old_generation: Vec<&AG>,
        scores: &Array1<Reward>,
    ) -> Vec<AG>
    where
        AC: Action,
        AG: Agent<AC> + Evolve,
    {
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

impl<AC, AG> Trainer<AC, AG> for GeneticAlgorithmLearner
where
    AC: Action,
    AG: Agent<AC> + Evolve,
{
    fn train<'a, E: Environment<AC>>(
        &mut self,
        agent: &'a mut AG,
        env: &E,
        epochs: usize,
        verbose: bool,
    ) -> &'a mut AG {
        // create multiple agents and an env for each one
        let mut agents = vec![agent.clone(); self.agent_amount];
        let best_agent = agent;
        let mut envs = vec![env.clone(); self.agent_amount];

        // run epochs
        let max_reward = env.max_reward();
        for e in 0..epochs {
            let mut scores = Array1::zeros(self.agent_amount);

            // evaluate each agent
            for (i, (agent, environment)) in agents.iter_mut().zip(envs.iter_mut()).enumerate() {
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
            *best_agent = agents[scores.argmax().unwrap()].clone();

            // spawn new generation
            agents = self.new_generation(agents.iter().collect(), &scores);
            agents[0] = best_agent.clone();
        }

        best_agent
    }
}

#[cfg(test)]
mod tests {
    use crate::neuron::activations::{Linear, ReLu, Sigmoid};
    use crate::neuron::layers::StandardLayer;
    use crate::neuron::networks::StandardFeedForwardNetwork;
    use crate::neuron::transfers::Dense;
    use crate::rl::agents::NeuroEvolutionAgent;
    use crate::rl::environments::JumpEnvironment;
    use crate::rl::trainers::genetic_algorithm::GeneticAlgorithmLearner;

    use super::*;

    #[test]
    fn test_neuro_evolution_learner() {
        let env = JumpEnvironment::new(10);
        let env_observation_space = env.observation_space();
        let env_action_space = env.action_space();
        let l1 = StandardLayer::new(3, env_observation_space, Dense::new(), ReLu::new());
        let l2 = StandardLayer::new(4, 3, Dense::new(), Sigmoid::new());
        let l3 = StandardLayer::new(env_action_space, 4, Dense::new(), Linear::new());
        let network_layers = vec![l1, l2, l3];
        let mut agent = NeuroEvolutionAgent::new(StandardFeedForwardNetwork::new(network_layers));
        let agent_amount = 10;
        let mutation_rate = 0.01;
        let mut learner = GeneticAlgorithmLearner::new(agent_amount, mutation_rate);
        let epochs = 10;
        learner.train(&mut agent, &env, epochs, false);
    }
}
