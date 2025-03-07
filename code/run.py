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
                cross_count,
                energy
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


def assign_ships_to_slots(B, L, lock_types):
    """
    B: (unused parameter in this snippet)
    L: list of ship lengths
    lock_types: list of lock types for each time slot, e.g.,
                ['Panamax', 'NeoPanamax', 'Panamax', ...]
                
    The function returns an assignment dictionary where keys correspond to the 
    flattened (ship_id, slot) indices and a 1 means the ship is assigned to that slot.
    """
    num_ships = len(L)
    num_time_slots = len(lock_types)
    
    # Initialize a flattened assignment dictionary.
    assignment = {idx: 0 for idx in range(num_ships * num_time_slots)}
    assigned = [False] * num_ships

    # Create lists of time slot indices by lock type.
    neo_slots = [i for i, lt in enumerate(lock_types) if lt == 'NeoPanamax']
    panamax_slots = [i for i, lt in enumerate(lock_types) if lt.startswith('Panamax')]
    
    # Process slots in descending order (if you want to fill later slots first).
    neo_slots.sort(reverse=True)
    panamax_slots.sort(reverse=True)

    # Sort ships in descending order of length.
    ships_sorted = sorted(range(num_ships), key=lambda i: L[i], reverse=True)

    # --- Step 1: Assign all ships longer than 294m to NeoPanamax locks ---
    # For each available NeoPanamax slot, assign one unassigned ship that is longer than 294m.
    # (Note: if there are more such ships than slots, extra ships will remain unassigned.)
    used_neo_slots = []  # track which slots we used in this step
    for slot in neo_slots[:]:  # iterate on a copy of neo_slots
        for i in ships_sorted:
            if not assigned[i] and L[i] > 294:
                assignment[i * num_time_slots + slot] = 1
                assigned[i] = True
                used_neo_slots.append(slot)
                break  # move to the next neo slot
    # Remove the used slots since they are no longer available for pairing.
    remaining_neo_slots = [slot for slot in neo_slots if slot not in used_neo_slots]

    # --- Step 2: For remaining NeoPanamax locks, try to assign two ships as a pair ---
    for slot in remaining_neo_slots:
        pair_found = False
        # Look at the list of still unassigned ships.
        unassigned_ships = [i for i in ships_sorted if not assigned[i]]
        n = len(unassigned_ships)
        # Try every distinct pair.
        for a in range(n):
            for b in range(a+1, n):
                i, j = unassigned_ships[a], unassigned_ships[b]
                if L[i] + L[j] < 366:  # condition: combined length less than 366 meters
                    # Assign both ships to this slot.
                    assignment[i * num_time_slots + slot] = 1
                    assignment[j * num_time_slots + slot] = 1
                    assigned[i] = True
                    assigned[j] = True
                    pair_found = True
                    break
            if pair_found:
                break
        # Optional: if no pair is found, you might assign one individual ship (if it fits the lock).
        if not pair_found:
            for i in unassigned_ships:
                if L[i] < get_lock_length('NeoPanamax'):
                    assignment[i * num_time_slots + slot] = 1
                    assigned[i] = True
                    break
                    
    # --- Step 3: Assign remaining ships to Panamax locks one by one ---
    for slot in panamax_slots:
        for i in ships_sorted:
            if not assigned[i] and L[i] <= get_lock_length('Panamax'):
                assignment[i * num_time_slots + slot] = 1
                assigned[i] = True
                break

    return assignment



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