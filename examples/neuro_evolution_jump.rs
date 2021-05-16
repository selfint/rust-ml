use std::{thread, time};

use rust_ml::neuron::activations::{linear, relu, sigmoid};
use rust_ml::neuron::layers::Layer;
use rust_ml::neuron::networks::Network;
use rust_ml::neuron::transfers::dense;
use rust_ml::rl::agents::NeuroEvolutionAgent;
use rust_ml::rl::environments::JumpEnvironment;
use rust_ml::rl::prelude::*;
use rust_ml::rl::trainers::genetic_algorithm::GeneticAlgorithm;

fn main() {
    let env_size = 7;
    let env = JumpEnvironment::new(env_size);

    // build network
    let env_action_space = env.action_space();
    let env_observation_space = env.observation_space();
    let layers = vec![
        Layer::new(3, env_observation_space, dense(None), relu()),
        // bring a bazooka to a knife fight
        Layer::new(4, 3, dense(None), sigmoid()),
        Layer::new(env_action_space, 4, dense(None), linear()),
    ];
    let network = Network::new(layers);

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
