import numpy as np
import matplotlib.pyplot as plt
import dimod

from utils import *
from make_qubo import build_qubo
from evaluate import evaluate_solution
from parameters import *


def run_instance(num_ships, num_time_slots, NUM_READS=10):
    B, L = generate_ship_parameters(num_ships)
    lock_types = generate_lock_types(num_time_slots)
    Q = build_qubo(B, L, lock_types)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=NUM_READS)

    feasible = []
    infeasible_count = 0
    infeasibility_reasons = []
    T = num_time_slots
    for sample, energy in sampleset.data(["sample", "energy"]):
        valid = True
        reasons = []
        # Check ship-once constraint.
        for i in range(num_ships):
            if sum(sample[i * T + t] for t in range(T)) != 1:
                valid = False
                reasons.append(
                    f"Ship {i} scheduled {sum(sample[i * T + t] for t in range(T))} times"
                )
                break
        # Check time-slot capacity.
        for t in range(T):
            if sum(sample[i * T + t] for i in range(num_ships)) > 2:
                valid = False
                reasons.append(
                    f"Time slot {t} has {sum(sample[i * T + t] for i in range(num_ships))} ships"
                )
                break
        # Check tandem lockage length constraint.
        for t in range(T):
            scheduled = [i for i in range(num_ships) if sample[i * T + t] == 1]
            if len(scheduled) == 2:
                total_length = sum(L[i] for i in scheduled)
                if total_length > get_lock_length(lock_types[t]):
                    valid = False
                    reasons.append(
                        f"Time slot {t} exceeds lock length by {total_length - get_lock_length(lock_types[t])} meters"
                    )
                    break
        if valid:
            comp_energy, water_cost, tot_benefit, pen, tandem_count, cross_count, _ = (
                evaluate_solution(sample, B, L, lock_types)
            )
            feasible.append(
                (
                    sample,
                    comp_energy,
                    water_cost,
                    tot_benefit,
                    pen,
                    tandem_count,
                    cross_count,
                )
            )
        else:
            infeasible_count += 1
            infeasibility_reasons.append(reasons)

    if feasible:
        best_solution = min(feasible, key=lambda x: x[1])
        best_water_cost = best_solution[2]
        best_sample = best_solution[0]
        best_tandem = best_solution[5]
        best_cross = best_solution[6]
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
def iteration_run(instance_sizes = list(range(3, 10, 2)), NUM_READS=10):
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
        instance_results = np.array(run_instance(n, T, NUM_READS))
        best_water_costs.append(instance_results[0] if instance_results[0] is not None else np.nan)
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
