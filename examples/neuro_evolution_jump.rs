use std::{thread, time};

use rust_ml::neuron::layers::{ReLuLayer, SigmoidLayer, SoftmaxLayer};
use rust_ml::neuron::networks::FeedForwardNetwork;
use rust_ml::rl::agents::NeuroEvolutionAgent;
use rust_ml::rl::environments::JumpEnvironment;
use rust_ml::rl::learners::NeuroEvolutionLearner;
use rust_ml::rl::prelude::*;

fn main() {
    let env_size = 7;
    let env = JumpEnvironment::new(env_size);

    // build network
    let action_space = env.action_space();
    let observation_space = env.observation_space();
    let network = FeedForwardNetwork::new(vec![
        Box::new(SigmoidLayer::new(10, observation_space)),
        // bring a bazooka to a knife fight
        Box::new(ReLuLayer::new(5, 10)),
        Box::new(SoftmaxLayer::new(action_space, 5)),
    ]);

    // build agent with network
    let mut agent = NeuroEvolutionAgent::new(Box::new(network));

    // train learner
    let epochs = 50;
    let agent_amount = 20;
    let mutation_rate = 0.01;
    let mut learner = NeuroEvolutionLearner::new(agent_amount, mutation_rate);
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
