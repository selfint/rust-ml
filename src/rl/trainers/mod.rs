pub use genetic_algorithm::GeneticAlgorithm;
pub use trainer::Trainer;

mod genetic_algorithm;
mod trainer;

pub mod neuro_evolution_internals {
    pub use super::genetic_algorithm::Evolve;
}
