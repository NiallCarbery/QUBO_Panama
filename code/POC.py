import numpy as np
import matplotlib.pyplot as plt
import dimod

from utils import generate_ship_parameters, generate_lock_types, get_lock_length, print_timetable, baseline_water_usage
from make_Qubo import build_qubo
from Evaluate import evaluate_solution
from parameters import *

# -----------------------------
# Function to run a single instance.
# -----------------------------
def run_instance(num_ships, num_slots):
    B, L = generate_ship_parameters(num_ships)
    lock_types = generate_lock_types(num_slots)
    Q = build_qubo(B, L, lock_types)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=NUM_READS)
    
    feasible = []
    infeasible_count = 0
    infeasibility_reasons = []
    T = num_slots
    for sample, energy in sampleset.data(['sample', 'energy']):
        valid = True
        reasons = []
        # Check ship-once constraint.
        for i in range(num_ships):
            if sum(sample[i * T + t] for t in range(T)) != 1:
                valid = False
                reasons.append(f"Ship {i} scheduled {sum(sample[i * T + t] for t in range(T))} times")
                break
        # Check time-slot capacity.
        for t in range(T):
            if sum(sample[i * T + t] for i in range(num_ships)) > 2:
                valid = False
                reasons.append(f"Time slot {t} has {sum(sample[i * T + t] for i in range(num_ships))} ships")
                break
        # Check tandem lockage length constraint.
        for t in range(T):
            scheduled = [i for i in range(num_ships) if sample[i * T + t] == 1]
            if len(scheduled) == 2:
                total_length = sum(L[i] for i in scheduled)
                if total_length > get_lock_length(lock_types[t]):
                    valid = False
                    reasons.append(f"Time slot {t} exceeds lock length by {total_length - get_lock_length(lock_types[t])} meters")
                    break
        if valid:
            comp_energy, water_cost, tot_benefit, pen, tandem_count, cross_count, _ = evaluate_solution(
                sample, B, L, lock_types)
            feasible.append((sample, comp_energy, water_cost, tot_benefit, pen, tandem_count, cross_count))
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
    baseline_usage = baseline_water_usage(lock_types, num_ships)
    return best_water_cost, baseline_usage, B, L, lock_types, best_sample, len(feasible), infeasible_count, best_tandem, best_cross, infeasibility_reasons

# -----------------------------
# Iterate over instance sizes and graph the results.
# -----------------------------

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
    best_cost, baseline_usage, B_inst, L_inst, lock_types, best_sample, feas_count, infeas_count, tandem_count, cross_count, infeasibility_reasons = run_instance(n, T)
    best_water_costs.append(best_cost if best_cost is not None else np.nan)
    baseline_costs.append(baseline_usage)
    feasible_counts.append(feas_count)
    infeasible_counts.append(infeas_count)
    tandem_counts.append(tandem_count)
    cross_fill_counts.append(cross_count)
    infeasibility_reasons_list.append(infeasibility_reasons)
    
    print(f"Instance with {n} ships (and {T} time slots):")
    print(f"  Optimized total water cost = {best_cost}")
    print(f"  Baseline total water cost   = {baseline_usage}")
    print(f"  Number of feasible solutions: {feas_count}")
    print(f"  Number of infeasible solutions: {infeas_count}")
    print(f"  Tandem lockages used: {tandem_count}")
    print(f"  Cross fills applied: {cross_count}")
    print(f"  Length of ships: {L_inst}")
    if infeasibility_reasons:
        print(f"  Infeasibility reasons: {infeasibility_reasons}")
    if best_sample is not None:
        print("  Timetable for best solution:")
        print_timetable(best_sample, n, T, lock_types)
    print("\n" + "-"*50 + "\n")

plt.figure(figsize=(10, 6))
plt.plot(instance_sizes, best_water_costs, marker='o', label='Optimized (Best) Water Usage')
plt.plot(instance_sizes, baseline_costs, marker='x', linestyle='--', label='Baseline Water Usage (One ship per slot)')
plt.xlabel("Number of Ships (Transits)")
plt.ylabel("Total Water Usage Cost")
plt.title("Water Usage vs. Number of Ships")
plt.legend()
plt.grid(True)
plt.show()
