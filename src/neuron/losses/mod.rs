pub use categorical_cross_entropy::{cce, cce_derivative, cce_loss};
pub use mean_squared_error::{mse, mse_derivative, mse_loss};
pub use sum_squared_error::{sse, sse_derivative, sse_loss};

pub use loss::Loss;

mod categorical_cross_entropy;
mod loss;
mod mean_squared_error;
mod sum_squared_error;
