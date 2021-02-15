use rust_ml::rl::prelude::*;

use std::{thread, time};

use ndarray::prelude::*;

use rust_ml::neuron::activations::{LeakyReLu, Sigmoid, Softplus};
use rust_ml::neuron::layers::Dense;
use rust_ml::neuron::networks::StandardFeedForwardNetwork;

use rust_ml::rl::agents::NeuroEvolutionAgent;
use rust_ml::rl::agents::QTableAgent;
use rust_ml::rl::environments::FlappyEnvironment;
use rust_ml::rl::trainers::genetic_algorithm::GeneticAlgorithm;
use rust_ml::rl::trainers::q_learning::{QFunction, QLearner};

fn main() {
    // build training environment
    let env_size = 12;
    let hole_size = 3;
    let env = FlappyEnvironment::new(env_size, hole_size);

    let mut agent = QTableAgent::new(env.action_space());

    let epsilon_decay_rate = 0.999;
    let gamma = 0.9;
    let learning_rate = 0.3;
    let mut trainer = QLearner::new(epsilon_decay_rate, learning_rate, gamma);

    let epochs = 20000;
    trainer.train(&mut agent, &env, epochs, true);

    // show trained agents
    loop {
        let mut test_env = env.clone();
        let mut score = 0.;

        while !test_env.is_done() {
            let state = test_env.observe();
            let action = agent.act(&state);
            score += test_env.step(&action);

            // reset cursor position
            print!("\x1B[2J\x1B[1;1H");

            println!("{}\nscore: {} | action: {}", test_env, score, action.0);
            thread::sleep(time::Duration::from_millis(100));
        }

        thread::sleep(time::Duration::from_millis(300));
    }
}
