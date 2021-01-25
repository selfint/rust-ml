pub use cached_network::CachedNetwork;
pub use feed_forward::StandardFeedForwardNetwork;
pub use network::{
    CachedClassification, CachedRegression, Classification, NetworkTrait, Regression,
};

mod cached_network;
mod feed_forward;
mod network;
