use crate::rl::{Action, State};

pub trait Agent: Clone {
    fn act(&self, state: &State) -> Action;
}
