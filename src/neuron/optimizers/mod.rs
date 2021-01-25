pub use optimizer::{OptimizeRegressorBatch, OptimizeRegressorOnce};
pub use stochastic_gradient_descent::SGD;

mod optimizer;
mod stochastic_gradient_descent;
