mod loss;
mod mean_squared_error;
mod sum_squared_error;

pub use loss::{Loss, LossTrait};
pub use mean_squared_error::{mse_loss, MSE};
pub use sum_squared_error::{sse_loss, SSE};
