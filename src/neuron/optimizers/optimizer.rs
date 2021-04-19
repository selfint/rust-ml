use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use crate::neuron::{losses::Loss, networks::Network};

pub trait OptimizeOnce {
    fn optimize_once(
        &self,
        network: &mut Network,
        input: &Array1<f32>,
        expected: &Array1<f32>,
        learning_rate: f32,
    );
}

pub trait OptimizeBatch {
    fn get_loss(&self) -> &Loss;

    fn optimize_batch(
        &self,
        network: &mut Network,
        batch_inputs: &[Array1<f32>],
        batch_expected: &[Array1<f32>],
        learning_rate: f32,
    );

    fn optimize(
        &self,
        network: &mut Network,
        train: &(Vec<Array1<f32>>, Vec<Array1<f32>>),
        test: &(Vec<Array1<f32>>, Vec<Array1<f32>>),
        learning_rate: f32,
        batch_size: usize,
        epochs: usize,
    ) {
        let (train_x, train_y) = train;
        let batches = train_x.len() / batch_size;
        for e in 0..epochs {
            // split data into batches
            for b in 0..batches {
                print!(
                    "batch {}/{} ({:.2}%)                  \r",
                    b,
                    batches,
                    (b as f32 / batches as f32) * 100.
                );
                let batch_inputs = &train_x[b..(b + batch_size)];
                let batch_expected = &train_y[b..(b + batch_size)];

                self.optimize_batch(network, batch_inputs, batch_expected, learning_rate);
            }

            print_network_score(&network, e, train, test, self.get_loss());
        }
    }
}

fn print_network_score(
    network: &Network,
    epoch: usize,
    train: &(Vec<Array1<f32>>, Vec<Array1<f32>>),
    test: &(Vec<Array1<f32>>, Vec<Array1<f32>>),
    loss: &Loss,
) {
    let (train_x, train_y) = train;
    let (test_x, test_y) = test;
    let train_samples = train_x.len();
    let test_samples = test_x.len();
    let mut train_loss = 0.;
    let mut train_mistakes = 0.;
    for (input, expected) in train_x.iter().zip(train_y.iter()) {
        let prediction = network.predict(input);
        if prediction.argmax().unwrap() != expected.argmax().unwrap() {
            train_mistakes += 1.;
        }

        train_loss += loss.loss(&prediction, expected).sum();
    }

    let mut test_loss = 0.;
    let mut test_mistakes = 0.;
    for (input, expected) in test_x.iter().zip(test_y.iter()) {
        let prediction = network.predict(input);
        if prediction.argmax().unwrap() != expected.argmax().unwrap() {
            test_mistakes += 1.;
        }

        test_loss += loss.loss(&prediction, expected).sum();
    }

    println!(
        "epoch {} | train loss: {:.4} accuracy: {:.2}% | test loss: {:.4} accuracy: {:.2}%",
        epoch,
        train_loss / (train_samples as f32),
        (1. - (train_mistakes / (train_samples as f32))) * 100.,
        test_loss / (test_samples as f32),
        (1. - (test_mistakes / (test_samples as f32))) * 100.,
    );
}
