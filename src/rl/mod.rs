pub mod agents;
pub mod environments;
pub mod learners;
pub mod prelude;

use ndarray::Array1;

/// Actions that agents can take in an environment
pub enum Action {
    Discrete(usize),
    Continuous(Array1<f32>),
}

/// How an environment is observed
pub type State = Array1<f32>;

/// How an environment rewards an agent for an action
pub type Reward = f32;
