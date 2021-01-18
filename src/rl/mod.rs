pub mod agent;
pub mod agents;
pub mod environment;
pub mod environments;
pub mod learner;
pub mod learners;

use ndarray::prelude::*;

/// Actions that agents can take in an environment
pub enum Action {
    Discrete(usize),
    Continuous(Array1<f32>),
}

/// How an environment is observed
pub type State = Array1<f32>;

/// How an environment rewards an agent for an action
pub type Reward = f32;

/// Parameter for learners
pub enum Param {
    Float(f32),
    Int(i32),
    Usize(usize),
    Bool(bool),
    Str(String),
}
