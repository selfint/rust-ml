mod learner;
mod neuro_evolution;

pub use learner::Learner;
pub use neuro_evolution::NeuroEvolutionLearner;
pub mod neuro_evolution_internals {
    pub use super::neuro_evolution::Evolve;
}
