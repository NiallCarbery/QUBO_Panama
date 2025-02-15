# QUBO_Panama

# QUBO Panama Canal Ship Scheduling

This project implements a Quadratic Unconstrained Binary Optimization (QUBO) model to optimize the scheduling of ships through the Panama Canal. The goal is to minimize the water usage cost while adhering to various constraints such as ship scheduling, lock capacities, and tandem lockage lengths.

## Features

- **Random Instance Generation**: Generates random ship parameters (benefits, lengths) and lock types for each time slot.
- **QUBO Model Building**: Constructs a QUBO model incorporating various constraints and objectives.
- **Solution Evaluation**: Evaluates candidate solutions based on water cost, penalties, and other metrics.
- **Simulated Annealing**: Uses simulated annealing to find feasible and optimized solutions.
- **Visualization**: Plots the optimized water usage cost against the baseline cost for different instance sizes.

## Files

- `POC.py`: Main script containing all the functions and logic for generating instances, building the QUBO model, evaluating solutions, and running the optimization.

## Usage

1. **Generate Ship Parameters**: Randomly generates benefits and lengths for a given number of ships.
2. **Generate Lock Types**: Generates lock types for each time slot.
3. **Build QUBO Model**: Constructs the QUBO model with constraints and objectives.
4. **Run Optimization**: Uses simulated annealing to find the best feasible solution.
5. **Evaluate Solutions**: Evaluates the solutions based on water cost, penalties, and other metrics.
6. **Plot Results**: Plots the optimized water usage cost against the baseline cost for different instance sizes.

## Example

To run the script and see the results, simply execute the `POC.py` file. The script will iterate over different instance sizes, run the optimization, and plot the results.

## QUBO Panama QAOA

This script implements a Quantum Approximate Optimization Algorithm (QAOA) for ship scheduling through the Panama Canal using Quadratic Unconstrained Binary Optimization (QUBO).
