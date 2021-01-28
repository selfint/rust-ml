use crate::rl::prelude::*;

/// Train an Agent in any Environment
pub trait Trainer<AC: Action, AG: Agent<AC>> {
    fn train<'a, E: Environment<AC>>(
        &mut self,
        agent: &'a mut AG,
        env: &E,
        epochs: usize,
        verbose: bool,
    ) -> &'a mut AG;
}
