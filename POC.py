import random
import numpy as np
import matplotlib.pyplot as plt
import dimod
import itertools

# -----------------------------
# Global tuning parameters
# -----------------------------
lambda_ship = 20          # Penalty for not scheduling a ship exactly once
lambda_conflict = 10    # Penalty for exceeding two ships per time slot
lambda_water = 2.0        # Weight for water cost in overall objective
lambda_length = 20      # Penalty coefficient for tandem lockage length violations
lambda_tandem = 10.0      # Reward for valid tandem lockage
lambda_crossfill = 3.0  # Reward for cross filling
penalty_infeasible = 1000 # Heavy penalty for infeasible assignments
crossfill_factor = 0.3    # Factor for cross filling
instance_sizes = list(range(3, 32, 2))  # Instance sizes (number of ships)
NUM_READS = 100         # Number of reads for the SimulatedAnnealingSampler

# -----------------------------
# Random instance generator functions
# -----------------------------
def generate_ship_parameters(num_ships):
    """
    Generate random benefits, water usages, and lengths for num_ships.
    Benefits: random integer between 8 and 12.
    Water usage: random integer between 2 and 6.
    Ship lengths: random integer between 20 and 40 meters.
    """
    B = [random.randint(8, 12) for _ in range(num_ships)]
    L = [random.randint(25, 60) for _ in range(num_ships)]
    return B, L

def generate_lock_types(num_slots):
    """
    Generate lock types for each time slot.
    The first 80% of the slots are Panamax locks (alternating "Panamax_A" and "Panamax_B"),
    and the remaining are NeoPanamax locks.
    """
    lock_types = []
    panamax_slots = int(0.7 * num_slots)
    for t in range(panamax_slots):
        if t % 2 == 0:
            lock_types.append("Panamax_A")
        else:
            lock_types.append("Panamax_B")
    for t in range(num_slots - panamax_slots):
        lock_types.append("NeoPanamax")
    return lock_types

def get_lock_length(lock_type):
    """
    Return the available lock length for the given lock type.
    """
    if lock_type.startswith("Panamax"):
        return 55
    elif lock_type.startswith("NeoPanamax"):
        return 70
    else:
        raise ValueError("Unknown lock type")

def water_cost_for_slot(lock_type, count):
    """
    Compute the baseline water cost for a time slot based on:
      - 0 ships: cost = 0.
      - 1 or 2 ships: cost = baseline water cost (30 for Panamax, 80 for NeoPanamax).
      - More than 2 ships: heavy penalty.
    """
    if count == 0:
        return 0
    elif count in [1, 2]:
        if lock_type.startswith("Panamax"):
            return 30
        elif lock_type.startswith("NeoPanamax"):
            return 80
    else:
        return penalty_infeasible * count

# -----------------------------
# QUBO model builder
# -----------------------------
def build_qubo(B, L, lock_types, lambda_ship, lambda_conflict, lambda_length):
    """
    Build the QUBO for ship scheduling.
    Decision variable: x[i,t] = 1 if ship i is scheduled in time slot t.
    
    The QUBO includes:
      - Reward term: subtract ship benefit.
      - Ship-once constraint.
      - Time-slot capacity constraint.
      - Tandem lockage length constraint: For each time slot, for each pair (i, j) of ships,
        if their combined length exceeds the available lock length, add a penalty term.
    """
    N = len(B)
    T = len(lock_types)
    Q = {}
    
    # Linear reward: subtract benefit for each assignment.
    for i in range(N):
        for t in range(T):
            idx = i * T + t
            Q[(idx, idx)] = Q.get((idx, idx), 0) - B[i]
    
    # Ship-once constraint: each ship must be scheduled exactly once.
    for i in range(N):
        indices = [i * T + t for t in range(T)]
        for idx in indices:
            Q[(idx, idx)] = Q.get((idx, idx), 0) + lambda_ship - 2 * lambda_ship
        for idx1, idx2 in itertools.combinations(indices, 2):
            key = (min(idx1, idx2), max(idx1, idx2))
            Q[key] = Q.get(key, 0) + 2 * lambda_ship
    
    # Time-slot capacity constraint: at most two ships per time slot.
    for t in range(T):
        for i, j in itertools.combinations(range(N), 2):
            idx_i = i * T + t
            idx_j = j * T + t
            Q[(min(idx_i, idx_j), max(idx_i, idx_j))] = Q.get((min(idx_i, idx_j), max(idx_i, idx_j)), 0) + 2 * lambda_conflict
    
    # Tandem lockage length constraint: penalize if two ships in the same slot exceed lock length.
    for t in range(T):
        available_length = get_lock_length(lock_types[t])
        for i, j in itertools.combinations(range(N), 2):
            if L[i] + L[j] > available_length:
                idx_i = i * T + t
                idx_j = j * T + t
                excess = L[i] + L[j] - available_length
                penalty_value = lambda_length * excess
                key = (min(idx_i, idx_j), max(idx_i, idx_j))
                Q[key] = Q.get(key, 0) + penalty_value

    # Water usage cost integration.
    for t in range(T):
        # Add quadratic coupling for ships in the same slot.
        for i, j in itertools.combinations(range(N), 2):
            idx_i = i * T + t
            idx_j = j * T + t
            key = (min(idx_i, idx_j), max(idx_i, idx_j))
            Q[key] = Q.get(key, 0) + 2 * lambda_water
        # Add linear terms for each ship in slot t.
        for i in range(N):
            idx = i * T + t
            Q[(idx, idx)] = Q.get((idx, idx), 0) - 3 * lambda_water

    # Reward valid tandem lockage.
    for t in range(T):
        available_length = get_lock_length(lock_types[t])
        for i, j in itertools.combinations(range(N), 2):
            if L[i] + L[j] <= available_length:
                idx_i = i * T + t
                idx_j = j * T + t
                key = (min(idx_i, idx_j), max(idx_i, idx_j))
                Q[key] = Q.get(key, 0) - lambda_tandem


    return Q

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_solution(sample, B, L, lock_types, lambda_water, lambda_length):
    """
    Evaluate a candidate solution.
    For each time slot:
      - Compute water cost.
      - Check if exactly 2 ships are scheduled; if so, check whether their combined length exceeds
        the lock’s available length and add a penalty if it does.
    
    Also checks ship-once and time-slot capacity constraints.
    
    Returns a tuple:
      (computed_energy, total_water_cost, total_benefit, total_penalty, tandem_count, cross_fill_count, infeasibility_reasons)
    """
    N = len(B)
    T = len(lock_types)
    total_penalty = 0
    tandem_count = 0
    cross_fill_count = 0
    infeasibility_reasons = []

    # Ship-once constraint.
    for i in range(N):
        s_sum = sum(sample[i * T + t] for t in range(T))
        if s_sum != 1:
            total_penalty += lambda_ship * (s_sum - 1) ** 2
            infeasibility_reasons.append(f"Ship {i} scheduled {s_sum} times")

    # Time-slot capacity constraint.
    for t in range(T):
        ships_in_slot = sum(sample[i * T + t] for i in range(N))
        if ships_in_slot > 2:
            total_penalty += lambda_conflict * (ships_in_slot - 2) ** 2
            infeasibility_reasons.append(f"Time slot {t} has {ships_in_slot} ships")

    total_benefit = 0
    for i in range(N):
        for t in range(T):
            if sample[i * T + t] == 1:
                total_benefit += B[i]

    total_water_cost = 0
    # Evaluate each time slot.
    for t in range(T):
        scheduled_ships = [i for i in range(N) if sample[i * T + t] == 1]
        count = len(scheduled_ships)
        current_lock = lock_types[t]
        cost_t = water_cost_for_slot(current_lock, count)
        if count == 2:
            tandem_count += 1
            total_length = sum(L[i] for i in scheduled_ships)
            available_length = get_lock_length(current_lock)
            if total_length > available_length:
                excess = total_length - available_length
                total_penalty += lambda_length * excess
                infeasibility_reasons.append(f"Time slot {t} exceeds lock length by {excess} meters")
        if t > 0:
            prev_count = sum(sample[i * T + (t-1)] for i in range(N))
            prev_lock = lock_types[t-1]
            if prev_lock.startswith("Panamax") and lock_types[t].startswith("Panamax") and (prev_count == 2 or prev_count == 1):
                cost_t *= (crossfill_factor)
                cross_fill_count += 1
        total_water_cost += cost_t

    computed_energy = -total_benefit + total_penalty + lambda_water * total_water_cost
    return computed_energy, total_water_cost, total_benefit, total_penalty, tandem_count, cross_fill_count, infeasibility_reasons

# -----------------------------
# Baseline water usage: worst-case where each ship transits alone.
# -----------------------------
def baseline_water_usage(lock_types, num_ships):
    cost = 0
    # Now, the number of time slots equals num_ships.
    for t in range(num_ships):
        cost += water_cost_for_slot(lock_types[t], 1)
    return cost

# -----------------------------
# Function to print the timetable.
# -----------------------------
def print_timetable(sample, num_ships, num_slots, lock_types):
    T = num_slots
    print("\nTimetable by Time Slot:")
    for t in range(num_slots):
        ships = [i+1 for i in range(num_ships) if sample[i * T + t] == 1]
        print(f"  Time slot {t} (Lock type: {lock_types[t]}): Ships {ships}")

# -----------------------------
# Function to run a single instance.
# -----------------------------
def run_instance(num_ships, num_slots):
    B, L = generate_ship_parameters(num_ships)
    lock_types = generate_lock_types(num_slots)
    Q = build_qubo(B, L, lock_types, lambda_ship, lambda_conflict, lambda_length)
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
                sample, B, L, lock_types, lambda_water, lambda_length)
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
