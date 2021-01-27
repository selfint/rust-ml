use std::{thread, time};

use rust_ml::neuron::activations::{Linear, ReLu, Sigmoid};
use rust_ml::neuron::layers::StandardLayer;
use rust_ml::neuron::networks::StandardFeedForwardNetwork;
use rust_ml::neuron::transfers::Dense;
use rust_ml::rl::agents::NeuroEvolutionAgent;
use rust_ml::rl::environments::JumpEnvironment;
use rust_ml::rl::prelude::*;
use rust_ml::rl::trainers::GeneticAlgorithm;

fn main() {
    let env_size = 7;
    let env = JumpEnvironment::new(env_size);

    // build network
    let env_action_space = env.action_space();
    let env_observation_space = env.observation_space();
    let layers = vec![
        StandardLayer::new(3, env_observation_space, Dense::new(), ReLu::new()),
        // bring a bazooka to a knife fight
        StandardLayer::new(4, 3, Dense::new(), Sigmoid::new()),
        StandardLayer::new(env_action_space, 4, Dense::new(), Linear::new()),
    ];
    let network = StandardFeedForwardNetwork::new(layers);

    // build agent with network
    let mut agent = NeuroEvolutionAgent::new(network);

    // train learner
    let epochs = 1000;
    let agent_amount = 20;
    let mutation_rate = 0.01;
    let mut learner = GeneticAlgorithm::new(agent_amount, mutation_rate);
    learner.train(&mut agent, &env, epochs, true);

    // show trained agent
    let mut env = JumpEnvironment::new(env_size);
    let mut score = 0.;
    while !env.is_done() {
        let action = agent.act(&env.observe());
        score += env.step(&action);

        // reset cursor position
        print!("\x1B[2J\x1B[1;1H");
        println!("{}\nscore: {}", env, score);
        thread::sleep(time::Duration::from_millis(100));
    }
}
