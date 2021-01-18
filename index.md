# Rust-ML

A simple machine learning library written in Rust, for supervised, unsupervised and 
reinforcement learning.

## Contents

- **Neuron**: A simple neural network library.
- **RL**: Implementations of RL algorithms and training environments.

## Neuron

CPU based neural network library implemented using [ndarray](https://github.com/rust-ndarray/ndarray).


It won't outperform Tensorflow but it should still be very fast.

## RL

Reinforcement learning library containing Agents, Learners and Environments. 


A **Learner** teaches an **Agent** to master an **Environment**.
All agents, learners and environments are designed to be easily swappable. For
example a QAgent can interact with a Jump environment and learn using a QLearner,
and that same agent can interact with a Bird environment and learn using a
NeuroEvolutionLearner.
