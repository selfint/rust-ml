pub mod agent;
pub mod agents;
pub mod environment;
pub mod environments;
pub mod learner;
pub mod learners;

use ndarray::prelude::*;

pub enum ActionEnum {
    Discrete(usize),
    Continuous(Array1<f32>),
}

/// Actions that agents can take in an environment
pub type Action = ActionEnum;

/// How an environment is observed
pub type State = Array1<f32>;

/// How an environment rewards an agent for an action
pub type Reward = f32;
