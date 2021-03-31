use std::error::Error;

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use rust_ml::neuron::activations::Softplus;
use rust_ml::neuron::layers::CachedLayer;
use rust_ml::neuron::losses::{cce_loss, CCE};
use rust_ml::neuron::networks::{CachedNetwork, Regression};
use rust_ml::neuron::optimizers::{OptimizeBatch, SGD};
use rust_ml::neuron::transfers::Dense;

const MNIST_TRAIN_PATH: &str = "/home/tom/Documents/Datasets/MNIST/mnist_train.csv";
const MNIST_TEST_PATH: &str = "/home/tom/Documents/Datasets/MNIST/mnist_test.csv";

fn read_dataset(
    path: &str,
    rows: &mut Vec<Array1<f32>>,
    labels: &mut Vec<Array1<f32>>,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    for csv_row in rdr.records() {
        let record = csv_row?
            .iter()
            .map(|pixel| pixel.parse::<u8>().unwrap())
            .collect::<Vec<u8>>();

        // one-hot encode label
        let mut label = Array1::zeros(10);
        label[record[0] as usize] = 1.;
        //let label = arr1(&[record[0] as f32 / 10.]);

        // skip label and normalize row data
        // to be between 0 and 1
        let row = arr1(
            &record
                .iter()
                .skip(1)
                .map(|&x| (x as f32) / 255.)
                .collect::<Vec<f32>>(),
        );

        labels.push(label);
        rows.push(row);
    }
    Ok(())
}

fn read_training_data() -> Result<
    (
        Vec<Array1<f32>>,
        Vec<Array1<f32>>,
        Vec<Array1<f32>>,
        Vec<Array1<f32>>,
    ),
    Box<dyn Error>,
> {
    let mut train_rows = Vec::with_capacity(60000);
    let mut train_labels = Vec::with_capacity(60000);
    let mut test_rows = Vec::with_capacity(10000);
    let mut test_labels = Vec::with_capacity(10000);

    read_dataset(MNIST_TRAIN_PATH, &mut train_rows, &mut train_labels)?;
    read_dataset(MNIST_TEST_PATH, &mut test_rows, &mut test_labels)?;
    Ok((train_rows, train_labels, test_rows, test_labels))
}

fn print_network_score(
    network: &CachedNetwork<CachedLayer>,
    e: i32,
    sample_train_x: &[Array1<f32>],
    sample_train_y: &[Array1<f32>],
    sample_test_x: &[Array1<f32>],
    sample_test_y: &[Array1<f32>],
) {
    let train_samples = sample_train_x.len();
    let test_samples = sample_test_x.len();
    let mut train_loss = 0.;
    let mut train_mistakes = 0.;
    for (input, expected) in sample_train_x.iter().zip(sample_train_y.iter()) {
        let prediction = network.predict(input);
        if prediction.argmax().unwrap() != expected.argmax().unwrap() {
            train_mistakes += 1.;
        }

        train_loss += cce_loss(&prediction, expected).sum();
    }

    let mut test_loss = 0.;
    let mut test_mistakes = 0.;
    for (input, expected) in sample_test_x.iter().zip(sample_test_y.iter()) {
        let prediction = network.predict(input);
        if prediction.argmax().unwrap() != expected.argmax().unwrap() {
            test_mistakes += 1.;
        }

        test_loss += cce_loss(&prediction, expected).sum();
    }

    println!(
        "epoch {} | train loss: {} accuracy: {}% | test loss: {} accuracy: {}%",
        e,
        train_loss / (train_samples as f32),
        (1. - (train_mistakes / (train_samples as f32))) * 100.,
        test_loss / (test_samples as f32),
        (1. - (test_mistakes / (test_samples as f32))) * 100.,
    );
}

fn main() {
    // parameters
    let train_samples = 60_000;
    let test_samples = 10_000;
    let epochs = 100_000_000;
    let batch_size = 100;
    let mut learning_rate = 3.;
    let learning_rate_decay = 0.8;
    let min_learning_rate = 0.001;

    let (train_x, train_y, test_x, test_y) = read_training_data().expect("failed to load datasets");
    let sample_train_x = &train_x[0..train_samples];
    let sample_train_y = &train_y[0..train_samples];
    let sample_test_x = &test_x[0..test_samples];
    let sample_test_y = &test_y[0..test_samples];

    let mut network = CachedNetwork::new(vec![
        CachedLayer::new(10, 784, Dense::new(), Softplus::new()),
        CachedLayer::new(10, 10, Dense::new(), Softplus::new()),
    ]);

    let optimizer = SGD::new(CCE::new());

    let batches = sample_train_x.len() / batch_size;

    for e in 0..epochs {
        // split data into batches
        for b in 0..batches {
            let batch_inputs = &sample_train_x[b..(b + batch_size)];
            let batch_expected = &sample_train_y[b..(b + batch_size)];

            optimizer.optimize_batch(&mut network, batch_inputs, batch_expected, learning_rate);
        }

        if e % 10 == 0 {
            print_network_score(&network, e, sample_train_x, sample_train_y, sample_test_x, sample_test_y);
        }

        // decrease the learning rate gradually
        if e % 10 == 0 && learning_rate > min_learning_rate {
            learning_rate *= learning_rate_decay;
        }

        // make sure we don't learn too slowly
        if learning_rate < min_learning_rate {
            learning_rate = min_learning_rate;
        }
    }

    let mut test_loss = 0.;
    let mut test_mistakes = 0.;
    for (input, expected) in sample_test_x.iter().zip(sample_test_y.iter()) {
        let prediction = network.predict(input);
        if prediction.argmax().unwrap() != expected.argmax().unwrap() {
            test_mistakes += 1.;
        }

        test_loss += cce_loss(&prediction, expected).sum();
    }

    println!(
        "final test loss: {} accuracy: {}%",
        test_loss / (test_samples as f32),
        (1. - (test_mistakes / (test_samples as f32))) * 100.,
    );
    println!("trained network: {:?}", network);
}
