import numpy as np
import matplotlib.pyplot as plt
import dimod
import os
import sys

from utils import *
from make_qubo import build_qubo
from evaluate import evaluate_solution
from parameters import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath("."), "../data")))

from data_processing import generate_ship_data


def run_instance(num_ships, num_time_slots, NUM_READS=10):
    ship_data = generate_ship_data(num_ships)
    L = ship_data["Length (m)"].to_numpy()
    B = ship_data["Benefit"].to_numpy()
    lock_types = generate_lock_types(num_time_slots)
    
    Q = build_qubo(B, L, lock_types)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=NUM_READS)

    feasible = []
    infeasible_count = 0
    infeasibility_reasons = []

    for sample, energy in sampleset.data(["sample", "energy"]):
        (
            comp_energy,
            water_cost,
            total_benefit,
            total_penalty,
            tandem_count,
            cross_count,
            infeasibile_reason,
        ) = evaluate_solution(sample, B, L, lock_types)

        if total_penalty == 0:
            feasible.append((
                sample,
                water_cost,
                total_benefit,
                tandem_count,
                cross_count
            ))

        if infeasibile_reason:  # Check if infeasibility_reasons list is not empty
            infeasibility_reasons.append(infeasibile_reason)
            infeasible_count += 1

    if feasible:
        best_solution = min(feasible, key=lambda x: x[1])
        best_water_cost = best_solution[1]
        best_sample = best_solution[0]
        best_tandem = best_solution[3]
        best_cross = best_solution[4]
    else:
        best_water_cost = None
        best_sample = None
        best_tandem = 0
        best_cross = 0

    baseline_usage = baseline_water_usage(lock_types, num_time_slots)
    
    return (
        best_water_cost,
        baseline_usage,
        B,
        L,
        lock_types,
        best_sample,
        len(feasible),
        infeasible_count,
        best_tandem,
        best_cross,
        infeasibility_reasons,
    )


# -----------------------------
# Iterate over instance sizes and graph the results.
# -----------------------------
def iteration_run(instance_sizes=list(range(3, 10, 2)), NUM_READS=10):
    """
    Iterate over each run printing the runs of the reults while also tracking the infeasibility reasons.
    """
    # List Tracking
    infeasibility_reasons_list = []
    best_water_costs = []
    baseline_costs = []
    feasible_counts = []
    infeasible_counts = []
    tandem_counts = []
    cross_fill_counts = []

    for n in instance_sizes:
        # Set number of time slots equal to number of ships.
        T = n
        instance_results = run_instance(n, T, NUM_READS)
        best_water_costs.append(
            instance_results[0] if instance_results[0] is not None else np.nan
        )
        baseline_costs.append(instance_results[1])
        feasible_counts.append(instance_results[6])
        infeasible_counts.append(instance_results[7])
        tandem_counts.append(instance_results[8])
        cross_fill_counts.append(instance_results[9])
        infeasibility_reasons_list.append(instance_results[10])

        print_results(n, T, instance_results)

    return instance_sizes, best_water_costs, baseline_costs


def plot(instance_sizes, best_water_costs, baseline_costs):
    """
    Plot the water usage for all time slots versus the number of time slots used in optimal solution.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        instance_sizes,
        best_water_costs,
        marker="o",
        label="Optimized (Best) Water Usage",
    )
    plt.plot(
        instance_sizes,
        baseline_costs,
        marker="x",
        linestyle="--",
        label="Baseline Water Usage (One ship per slot)",
    )
    plt.xlabel("Number of Ships (Transits)")
    plt.ylabel("Total Water Usage Cost")
    plt.title("Water Usage vs. Number of Ships")
    plt.legend()
    plt.grid(True)
    plt.show()
