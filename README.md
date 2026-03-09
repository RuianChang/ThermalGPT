# ThermalGPT: Learning Quantum Measurement Statistics with Transformers

This repository demonstrates a machine learning pipeline for learning and generating quantum measurement statistics from large-scale physics simulations.

The project explores how a **Transformer-based generative model** can learn conditional measurement distributions of a subsystem given measurement outcomes of its surrounding environment (bath). The generated measurement data can then be used for **classical shadow tomography** to reconstruct subsystem quantum states.

This work combines **machine learning, computational physics, and statistical reconstruction**, and serves as an example of applying ML techniques to physics-based modeling problems.

---

# Project Overview

In many-body quantum systems, measurements on the environment collapse the global quantum state into a **projected ensemble**. The resulting subsystem state depends on the measurement outcomes of the surrounding bath.

The goal of this project is to train a machine learning model to learn the mapping:

bath measurement outcomes → subsystem measurement statistics

Once trained, the model can generate new measurement samples that reproduce the correct statistical behavior of the subsystem.

These generated measurements can then be used to reconstruct quantum states using **classical shadow tomography**.

---

# Pipeline

The workflow implemented in this repository follows three steps:

1. **Train a Transformer model** to learn conditional measurement distributions from simulated quantum measurement datasets.

2. **Generate measurement samples** from the trained model conditioned on bath measurement outcomes.

3. **Use classical shadow tomography** on the generated measurement data to reconstruct the subsystem density matrix and compare it with theoretical predictions.

This pipeline demonstrates that machine learning can reproduce physically meaningful measurement statistics and recover quantum observables.

---

# Files in this Repository

## 1. Transformer Training

**File**

ThermalGPT_QB=12.py

This script trains a Transformer-based generative model to learn measurement statistics from simulation datasets.

The model learns the conditional distribution

P(b₁, b₂ | z, P₁, P₂)

where

- z : measurement outcomes of bath qubits  
- P : measurement basis of subsystem qubits (X, Y, Z)  
- b : measurement outcomes

Key features:

- Transformer architecture implemented in **PyTorch**
- **GPU acceleration (CUDA)** for efficient training
- Autoregressive modeling of measurement outcomes
- Training on large-scale physics simulation datasets

The trained model can predict subsystem measurement outcomes conditioned on bath measurements and measurement bases.

---

## 2. Generating Measurement Samples

**File**

Data_generation_seen_z.ipynb

This notebook demonstrates how to use the trained Transformer model to generate measurement samples.

The workflow includes:

- loading the trained Transformer model
- conditioning on a fixed bath measurement outcome
- generating subsystem measurement results
- comparing generated measurement distributions with theoretical predictions

This step verifies that the learned model correctly reproduces the measurement statistics of the physical system.

---

## 3. Classical Shadow Reconstruction

**File**

Classical_Shadows.ipynb

The generated measurement samples are used to perform **classical shadow tomography**.

This notebook:

- reconstructs subsystem density matrices from measurement outcomes
- compares reconstructed states with theoretical projected states
- benchmarks results against Haar random ensemble predictions

This demonstrates that machine-learning-generated measurement data can reproduce correct physical observables.

---

# Scientific Motivation

In many-body quantum systems, measurements on part of the system collapse the global wavefunction into a **projected ensemble**.

A key question is whether the statistical behavior of a subsystem can be predicted from limited information about the environment.

This project explores whether machine learning models can learn the relationship

environment measurements → subsystem statistics

and reproduce correct quantum observables through generated measurement data.

The work connects machine learning with concepts from **quantum statistical mechanics and thermalization**.

---

# Technologies Used

- Python
- PyTorch
- GPU acceleration (CUDA)
- Transformer architectures
- Classical shadow tomography
- Numerical physics simulations

---

# Applications

This project illustrates how machine learning can be applied to physics-based simulation problems, including:

- learning statistical behavior of complex physical systems
- generative modeling for measurement data
- surrogate modeling of physics simulations
- data-driven reconstruction of physical observables

These ideas are broadly relevant to **computational physics, simulation science, and machine learning for physical systems**.

---

# Author

Rui-An Chang  
Ph.D. Candidate in Physics  
University of Texas at Austin

Research interests: computational physics, machine learning for physical systems, and large-scale numerical simulation.
