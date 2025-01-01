# Full Adder Power and Delay Optimization using Neural Networks

## Overview
This project uses machine learning to model and optimize the **average power** and **propagation delay** of a CMOS full adder circuit. 
The workflow involves generating simulation data using **LTspice**, training a **neural network** with TensorFlow/Keras, 
and performing parameter optimization using techniques like **differential evolution**.

### Key Features
- **Data Generation**: Automates input parameter generation using **Latin Hypercube Sampling (LHS)**.
- **Neural Network Training**: Models the relationship between $\(V_{DD}, W_n, C_L\)$ and the circuit's power and delay.
- **Parameter Optimization**: Finds optimal \(V_{DD}, W_n, C_L\) values to minimize power and delay.
- **Visualization**: Provides tools for analyzing predictions and trade-offs.

---

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
   - [1. Generate Data](#1-generate-data)
   - [2. Train Model](#2-train-model)
   - [3. Optimize Parameters](#3-optimize-parameters)
4. [Results](#results)
5. [Contributing](#contributing)
6. [License](#license)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/full-adder-optimization.git
   cd full-adder-optimization
