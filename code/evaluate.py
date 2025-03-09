from utils import get_lock_length, water_cost_for_slot
from parameters import *


def evaluate_solution(sample, B, L, lock_types):
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
    penalty = 0
    tandem_count = 0
    cross_fill_count = 0
    infeasible_reason = []

    # 1. Constraint: Each ship is assigned one time.
    for i in range(N):
        assignments = sum(sample[i * T + t] for t in range(T))
        if assignments != 1:
            penalty += penalty_infeasible
            infeasible_reason.append(
                f"Ship {i+1} assigned {assignments} times (Expected 1)."
            )

    # Check if there are more than 2 ships in a given slot
    for t in range(T):
        ships_in_slot = sum(sample[i * T + t] for i in range(N))
        if ships_in_slot > 2:
            penalty += penalty_infeasible
            infeasible_reason.append(f"Time slot {t+1} has {ships_in_slot} ships")

    total_benefit = 0
    for i in range(N):
        for t in range(T):
            if sample[i * T + t] == 1:
                total_benefit += lambda_once_reward

    total_water_cost = 0
    # Evaluate each time slot.
    for t in range(T):
        scheduled_ships = [i for i in range(N) if sample[i * T + t] == 1]
        count = len(scheduled_ships)
        current_lock = lock_types[t]
        cost_t = water_cost_for_slot(current_lock, count)

        ## Length
        if count == 1:
            total_length = sum(L[i] for i in scheduled_ships)
            available_length = get_lock_length(current_lock)
            if total_length > available_length:
                excess = total_length - available_length
                penalty += penalty_infeasible
                infeasible_reason.append(
                    f"Time slot {t} exceeds lock length by {excess} meters"
                )
        ## Check Tandem Lockage and Measure
        if count == 2:
            tandem_count += 1
            total_length = sum(L[i] for i in scheduled_ships)
            available_length = get_lock_length(current_lock)
            if total_length > available_length:
                excess = total_length - available_length
                penalty += penalty_infeasible
                infeasible_reason.append(
                    f"Time slot {t} exceeds lock length by {excess} meters"
                )
        ## Measure water reduction of cross fill
        if t > 0:
            prev_count = sum(sample[i * T + (t - 1)] for i in range(N))
            prev_lock = lock_types[t - 1]
            if (
                prev_lock.startswith("Panamax")
                and lock_types[t].startswith("Panamax")
                and (prev_count == 2 or prev_count == 1)
            ):
                cost_t *= crossfill_factor
                cross_fill_count += 1
        total_water_cost += cost_t

    computed_energy = -total_benefit + penalty + total_water_cost
    return (
        computed_energy,
        total_water_cost,
        total_benefit,
        penalty,
        tandem_count,
        cross_fill_count,
        infeasible_reason,
    )
