use std::collections::HashMap;
use std::{thread, time};

use rust_ml::neuron::layers::{ReLuLayer, SigmoidLayer, SoftmaxLayer};
use rust_ml::neuron::networks::feed_forward::FeedForwardNetwork;
use rust_ml::rl::agent::Agent;
use rust_ml::rl::agents::network_agent::NetworkAgent;
use rust_ml::rl::environment::Environment;
use rust_ml::rl::environments::jump::JumpEnvironment;
use rust_ml::rl::learner::Learner;
use rust_ml::rl::learners::neuro_evolution::NeuroEvolutionLearner;
use rust_ml::rl::Param;

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
    let agent = NetworkAgent::new(Box::new(network));

    // set learner parameters
    let mut params = HashMap::with_capacity(2);
    params.insert("agent_amount", Param::Usize(20));
    params.insert("mutation_rate", Param::Float(0.01));

    let mut learner = NeuroEvolutionLearner::new(agent, Some(&params));

    // train learner
    let epochs = 50;
    let trained_agent = learner.master(&env, epochs, true);

    // show trained agent
    let mut env = JumpEnvironment::new(env_size);
    let mut score = 0.;
    while !env.is_done() {
        let action = trained_agent.act(&env.observe());
        score += env.step(&action);

        // reset cursor position
        print!("\x1B[2J\x1B[1;1H");
        println!("{}\nscore: {}", env, score);
        thread::sleep(time::Duration::from_millis(100));
    }
}
