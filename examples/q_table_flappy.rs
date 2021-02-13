use rust_ml::rl::prelude::*;

use std::{thread, time};

use rust_ml::neuron::activations::ReLu;
use rust_ml::neuron::layers::Dense;
use rust_ml::neuron::networks::StandardFeedForwardNetwork;

use rust_ml::rl::agents::NeuroEvolutionAgent;
use rust_ml::rl::agents::QTableAgent;
use rust_ml::rl::environments::FlappyEnvironment;
use rust_ml::rl::trainers::genetic_algorithm::GeneticAlgorithm;
use rust_ml::rl::trainers::q_learning::{QFunction, QLearner};

fn main() {
    // build training environment
    let env_size = 10;
    let env = FlappyEnvironment::new(env_size);

    // Q learning
    let qagent = QTableAgent::new(env.action_space());

    let epsilon_decay_rate = 0.99;
    let gamma = 0.9;
    let learning_rate = 0.8;
    let qtrainer = QLearner::new(epsilon_decay_rate, learning_rate, gamma);

    let q = (qagent, qtrainer);

    // Neuro evolution
    let network = StandardFeedForwardNetwork::new(vec![
        Dense::new(5, env.observation_space(), ReLu::new()),
        Dense::new(env.action_space(), 5, ReLu::new()),
    ]);
    let nagent = NeuroEvolutionAgent::new(network);

    let agent_amount = 50;
    let mutation_rate = 0.3;
    let ntrainer = GeneticAlgorithm::new(agent_amount, mutation_rate);

    let n = (nagent, ntrainer);

    // train agents
    let (mut agent, mut trainer) = n;

    let epochs = 1000;
    trainer.train(&mut agent, &env, epochs, true);

    // show trained agents
    loop {
        let mut test_env = FlappyEnvironment::new(env_size);
        let mut score = 0.;
        while !test_env.is_done() {
            let state = test_env.observe();
            //let action_values = agent.get_action_values(&state);
            let action = agent.act(&state);
            score += test_env.step(&action);

            // reset cursor position
            print!("\x1B[2J\x1B[1;1H");
            println!("{}\nscore: {}", test_env, score);
            //println!("action: {} | action values: {}", action.0, action_values.to_string());
            thread::sleep(time::Duration::from_millis(100));
        }
    }
}
