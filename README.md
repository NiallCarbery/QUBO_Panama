# QUBO_Panama

# QUBO Panama Canal Ship Scheduling

This project implements a Quadratic Unconstrained Binary Optimization (QUBO) model to optimize the scheduling of ships through the Panama Canal. The goal is to minimize the water usage cost while adhering to various constraints such as ship scheduling, lock capacities, and tandem lockage lengths.

## Features

- **Random Instance Generation**: Generates random ship parameters (benefits, lengths) and lock types for each time slot.
- **QUBO Model Building**: Constructs a QUBO model incorporating various constraints and objectives.
- **Solution Evaluation**: Evaluates candidate solutions based on water cost, penalties, and other metrics.
- **Simulated Annealing**: Uses simulated annealing to find feasible and optimized solutions.
- **Visualization**: Plots the optimized water usage cost against the baseline cost for different instance sizes.

## Code Overview

### utils.py
Contains utility functions for generating ship parameters, lock types, calculating water costs, and printing results.

### run.py
Main script to run the QUBO model instances, evaluate solutions, and plot results.

### QAOA.py
Implements the QAOA algorithm for solving the QUBO problem using quantum circuits.

### parameters.py
Defines global tuning parameters used across the project.

### make_qubo.py
Builds the QUBO matrix for the ship scheduling problem.

### evaluate.py
Evaluates candidate solutions for the QUBO problem, checking constraints and calculating penalties.

### data_processing.py
Processes input data to generate ship data for the QUBO model.

## Dependencies and Requirements

To run the code, you need the following dependencies:

- Python 3.7 or higher
- numpy
- pandas
- matplotlib
- qiskit
- dimod

You can install the required packages using pip:

```sh
pip install numpy pandas matplotlib qiskit dimod
```

## Usage

1. **Generate Ship Parameters**: Randomly generates benefits and lengths for a given number of ships.
2. **Generate Lock Types**: Generates lock types for each time slot.
3. **Build QUBO Model**: Constructs the QUBO model with constraints and objectives.
4. **Run Optimization**: Uses simulated annealing to find the best feasible solution.
5. **Evaluate Solutions**: Evaluates the solutions based on water cost, penalties, and other metrics.
6. **Plot Results**: Plots the optimized water usage cost against the baseline cost for different instance sizes.

## Example

To run the script and see the results, simply execute the `run.py` file. The script will iterate over different instance sizes, run the optimization, and plot the results.

## QUBO Panama QAOA

This script implements a Quantum Approximate Optimization Algorithm (QAOA) for ship scheduling through the Panama Canal using Quadratic Unconstrained Binary Optimization (QUBO).

## Example Usage Notebook

An example usage of the QUBO Panama model can be found in the Jupyter notebook `example_usage.ipynb` located in the `test` directory. This notebook demonstrates how to import the necessary modules, run the QUBO model, and visualize the results.