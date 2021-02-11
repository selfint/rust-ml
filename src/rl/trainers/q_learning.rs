use crate::rl::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand::thread_rng;
use ndarray_rand::rand::Rng;

pub trait QFunction {
    fn get_action_values(&mut self, state: &State) -> Array1<Reward>;
    fn update_q_function(
        &mut self,
        state: &State,
        action: &DiscreteAction,
        reward: &Reward,
        next_state: &State,
        learning_rate: f32,
        gamma: f32,
    );
}

pub struct QLearner {
    epsilon: f64,
    epsilon_decay_rate: f64,
    learning_rate: f32,
    gamma: f32,
}

impl QLearner {
    pub fn new(epsilon_decay_rate: f64, learning_rate: f32, gamma: f32) -> Self {
        QLearner {
            epsilon: 1.,
            epsilon_decay_rate,
            learning_rate,
            gamma,
        }
    }
}

impl<A> Trainer<DiscreteAction, A> for QLearner
where
    A: Agent<DiscreteAction> + QFunction,
{
    fn train<'a, E: Environment<DiscreteAction>>(
        &mut self,
        agent: &'a mut A,
        env: &E,
        epochs: usize,
        verbose: bool,
    ) -> &'a mut A {
        let mut rng = thread_rng();
        let mut training_env = env.clone();
        let env_action_space = training_env.action_space();
        let max_score = training_env.max_reward();
        for e in 0..epochs {
            let mut score = 0.;
            training_env.reset();

            while !training_env.is_done() && score < max_score {
                let state = training_env.observe();
                let action = if rng.gen_bool(self.epsilon) {
                    DiscreteAction(rng.gen_range(0..env_action_space))
                } else {
                    agent.act(&state)
                };

                let reward = training_env.step(&action);
                let next_state = training_env.observe();
                agent.update_q_function(
                    &state,
                    &action,
                    &reward,
                    &next_state,
                    self.learning_rate,
                    self.gamma,
                );

                self.epsilon *= self.epsilon_decay_rate;
                score += reward
            }

            if verbose {
                println!("epoch {}: score={}", e, score);
            }
        }

        agent
    }
}

#[cfg(test)]
mod tests {
    use crate::rl::agents::QTableAgent;
    use crate::rl::environments::JumpEnvironment;

    use super::*;

    #[test]
    fn test_q_learner() {
        let env = JumpEnvironment::new(10);
        let env_action_space = env.action_space();
        let mut agent = QTableAgent::new(env_action_space);
        let epsilon_decay_rate = 0.9;
        let learning_rate = 0.1;
        let gamma = 0.1;
        let mut learner = QLearner::new(epsilon_decay_rate, learning_rate, gamma);
        let epochs = 10;
        learner.train(&mut agent, &env, epochs, false);
    }
}
