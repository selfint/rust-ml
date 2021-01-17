use std::collections::HashMap;

use crate::environment::Environment;

pub trait Learner {
    fn master<E: Environment>(
        &mut self,
        env: &E,
        epochs: usize,
        params: Option<&HashMap<&str, f32>>,
    );
}
