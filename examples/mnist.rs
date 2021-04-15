use std::error::Error;

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use rust_ml::neuron::activations::{leaky_relu, linear};
use rust_ml::neuron::layers::LayerStruct;
use rust_ml::neuron::losses::{cce, cce_loss};
use rust_ml::neuron::networks::NetworkStruct;
use rust_ml::neuron::optimizers::{OptimizeBatch, SGD};
use rust_ml::neuron::transfers::dense;

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
    network: &NetworkStruct,
    epoch: i32,
    train_x: &[Array1<f32>],
    train_y: &[Array1<f32>],
    test_x: &[Array1<f32>],
    test_y: &[Array1<f32>],
) {
    let train_samples = train_x.len();
    let test_samples = test_x.len();
    let mut train_loss = 0.;
    let mut train_mistakes = 0.;
    for (input, expected) in train_x.iter().zip(train_y.iter()) {
        let prediction = network.predict(input);
        if prediction.argmax().unwrap() != expected.argmax().unwrap() {
            train_mistakes += 1.;
        }

        train_loss += cce_loss(&prediction, expected).sum();
    }

    let mut test_loss = 0.;
    let mut test_mistakes = 0.;
    for (input, expected) in test_x.iter().zip(test_y.iter()) {
        let prediction = network.predict(input);
        if prediction.argmax().unwrap() != expected.argmax().unwrap() {
            test_mistakes += 1.;
        }

        test_loss += cce_loss(&prediction, expected).sum();
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

fn main() {
    // parameters
    let epochs = 100_000_000;
    let batch_size = 50;
    let learning_rate = 0.05;

    // load data
    println!("Loading MNIST dataset");
    let (train_x, train_y, test_x, test_y) = read_training_data().expect("failed to load datasets");
    println!("Loaded training data: {} rows", train_x.len());
    println!("Loaded test data: {} rows", test_x.len());

    // build network and optimizer
    println!("building network and optimizer");
    let mut network = NetworkStruct::new(vec![
        LayerStruct::new(128, 784, dense(), leaky_relu()),
        LayerStruct::new(10, 128, dense(), linear()),
    ]);
    let optimizer = SGD::new(cce());

    // training loop
    println!("beginning training loop");
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

            optimizer.optimize_batch(&mut network, batch_inputs, batch_expected, learning_rate);
        }

        print_network_score(&network, e, &train_x, &train_y, &test_x, &test_y);
    }

    // show final results
    let mut test_loss = 0.;
    let mut test_mistakes = 0.;
    for (input, expected) in test_x.iter().zip(test_y.iter()) {
        let prediction = network.predict(input);
        if prediction.argmax().unwrap() != expected.argmax().unwrap() {
            test_mistakes += 1.;
        }

        test_loss += cce_loss(&prediction, expected).sum();
    }

    println!(
        "final test loss: {} accuracy: {}%",
        test_loss,
        (1. - (test_mistakes / (test_x.len() as f32))) * 100.,
    );
    println!("trained network: {:?}", network);
}
