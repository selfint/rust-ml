pub use cached_network::CachedNetwork;
pub use feed_forward::StandardFeedForwardNetwork;
pub use network::{CachedRegression, Network, Regression};

mod cached_network;
mod feed_forward;
mod network;
