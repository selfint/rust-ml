use crate::rl::{Action, State};

pub trait Agent {
    fn act(&self, state: &State) -> Action;
}
