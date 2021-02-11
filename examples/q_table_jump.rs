use std::{thread, time};
use rust_ml::rl::prelude::*;

use rust_ml::rl::agents::QTableAgent;
use rust_ml::rl::trainers::q_learning::QLearner;
use rust_ml::rl::environments::JumpEnvironment;

fn main() {
    // build training environment
    let env_size = 7;
    let env = JumpEnvironment::new(env_size);

    // build agent
    let mut agent = QTableAgent::new(env.action_space());

    // build trainer
    let epsilon_decay_rate = 0.9;
    let learning_rate = 0.1;
    let gamma = 0.1;
    let mut trainer = QLearner::new(epsilon_decay_rate, learning_rate, gamma);

    // train agent
    let epochs = 10;
    trainer.train(&mut agent, &env, epochs, true);

    // show trained agent
    let mut test_env = JumpEnvironment::new(env_size);
    let mut score = 0.;
    while !test_env.is_done() {
        let action = agent.act(&test_env.observe());
        score += test_env.step(&action);

        // reset cursor position
        print!("\x1B[2J\x1B[1;1H");
        println!("{}\nscore: {}", test_env, score);
        thread::sleep(time::Duration::from_millis(100));
    }
}
