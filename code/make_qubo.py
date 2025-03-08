
import itertools
from utils import get_lock_length, water_cost_for_slot
from parameters import *


def build_qubo(B, L, lock_types, dynamic=False):
    """
    Build the QUBO for ship scheduling.

    Decision variable: x[i,t] = 1 if ship i is scheduled in time slot t.

    The QUBO includes:
      - Linear Terms:
          * Reward: subtract ship benefit.
          * Water cost: add water cost for using the slot.
          * Individual infeasibility: if ship i alone exceeds the slot’s capacity.
      - Quadratic Terms:
          * Tandem Reward: gives a bonus when two ships share a time slot.
          * Combined Length Feasibility:
              - If the combined length of two ships exceeds the lock capacity (infeasible),
                a fixed penalty (penalty_infeasible_pair) is applied.
              - Otherwise, if the margin (lock capacity minus combined length) is larger than
                a predefined threshold (margin_bonus_threshold), an extra bonus is applied.
      - Panamax Cross‑fill Reward:
          * For consecutive time slots t and t+1 that form the set {"Panamax_A", "Panamax_B"},
            add a bonus for pairing scheduled ships across these slots.
    """
    N = len(B)
    T = len(lock_types)
    Q = {}

    # Capacity thresholds (in meters)
    panamax_capacity = get_lock_length('Panamax')  # ship must be <= 294 for Panamax
    neo_capacity     = get_lock_length('NeoPanamax')  # for two short ships in NeoPanamax, their sum must be < 366

    # QUBO is stored as a dictionary, with keys as tuples (p, q) corresponding to variables.
    Q = {}

    def add_term(p, q, coeff):
        key = (min(p, q), max(p, q))
        Q[key] = Q.get(key, 0) + coeff

    def idx(i, j):
        """Flatten the two-dimensional indices (ship, slot) into a single index."""
        return i * T + j

    # 1. Enforce that each ship is assigned exactly once.
    for i in range(N):
        indices = [idx(i, j) for j in range(T)]
        # Linear terms: -lam for every assignment variable.
        for j in range(T):
            add_term(idx(i, j), idx(i, j), -lambda_ship)
        # Quadratic terms: 2*lam for every pair of assignments for ship i.
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                add_term(indices[a], indices[b], 2 * lambda_ship)

    # 2. Enforce time-slot capacity constraints and lock-type compatibility.
    for j in range(T):
        # (a) Build the list of flattened indices for slot j.
        slot_indices = [idx(i, j) for i in range(N)]
        lt = lock_types[j]

        if lt != "NeoPanamax":
            # Panamax slot: allow at most one assignment.
            for a in range(len(slot_indices)):
                for b in range(a + 1, len(slot_indices)):
                    add_term(slot_indices[a], slot_indices[b], lambda_time_p)
            # Also, if a ship is “long” (L > panamax_capacity) and is assigned to a Panamax,
            # add a penalty.
            for i in range(N):
                if L[i] > panamax_capacity:
                    add_term(idx(i, j), idx(i, j), lambda_length_p)
        else:
            # NeoPanamax slot: allow up to two assignments.
            # First, add a small quadratic penalty for every pair.
            for a in range(len(slot_indices)):
                for b in range(a + 1, len(slot_indices)):
                    add_term(slot_indices[a], slot_indices[b], lambda_time_np)
            # Now, for every possible pair in this slot, depending on the ships’ types and lengths:
            #   - If one (or both) of the ships is "long", penalize that pairing (they should not share a lock).
            #   - Otherwise (both short), if their combined length exceeds neo_capacity, penalize the pair.
            for i in range(N):
                for k in range(i + 1, N):
                    # These terms only come into play if both ship i and ship k are assigned to slot j.
                    if (L[i] > panamax_capacity) or (L[k] > panamax_capacity):
                        add_term(idx(i, j), idx(k, j), lambda_conflict)
                    else:
                        if L[i] + L[k] > neo_capacity:
                            add_term(idx(i, j), idx(k, j), lambda_length_np)

    if dynamic == True:
        # 1. Linear Terms: per ship and per slot.
        for i in range(N):
            for t in range(T):
                idx = i * T + t
                # If individual ship exceeds the slot's capacity, add infeasibility penalty.
                incomp = dynamic_penalty_infeasible if L[i] > get_lock_length(lock_types[t]) else 0
                # Linear contribution: subtract benefit, add water cost, plus incompatibility penalty.
                Q[(idx, idx)] = Q.get((idx, idx), 0) - B[i] * dynamic_lambda_benefit \
                                + water_cost_for_slot(lock_types[t], 1) * dynamic_lambda_water + incomp

        # 2. Constraint: Each ship is scheduled exactly once or 0.
        # Enforce (sum_t x[i,t] - 1)^2 = sum_t x[i,t]^2 - 2 * sum_t x[i,t] + 1,
        for i in range(N):
            # List of variables for ship i.
            ship_vars = [i * T + t for t in range(T)]
            # Diagonal contributions: each variable gets -lambda_ship.
            for var in ship_vars:
                Q[(var, var)] = Q.get((var, var), 0) - dynamic_lambda_ship
            # Off-diagonal contributions: for each unique pair add +2*lambda_ship.
            if dynamic == True:
                for a in range(len(ship_vars)):
                    for b in range(a + 1, len(ship_vars)):
                        u = ship_vars[a]
                        v = ship_vars[b]
                        key = (u, v) if u <= v else (v, u)
                        Q[key] = Q.get(key, 0) + 2 * dynamic_lambda_ship
            else:
                indices = [i * T + t for t in range(T)]
                for idx1, idx2 in itertools.combinations(indices, 2):
                    key = (min(idx1, idx2), max(idx1, idx2))
                    Q[key] = Q.get(key, 0) + 2 * dynamic_lambda_ship

        # 3. Quadratic Terms: Tandem reward and combined length feasibility.
        for t in range(T):
            # Get the lock capacity for the current time slot.
            capacity = get_lock_length(lock_types[t])
            for i in range(N):
                for j in range(i + 1, N):
                    idx_i = i * T + t
                    idx_j = j * T + t
                    key = (idx_i, idx_j) if idx_i <= idx_j else (idx_j, idx_i)
                    
                    q_val = - dynamic_lambda_tandem
                    
                    combined_length = L[i] + L[j]
                    if combined_length > capacity:
                        # Infeasible: the combined ship lengths exceed the lock capacity.
                        q_val += dynamic_penalty_infeasible

                    Q[key] = Q.get(key, 0) + q_val

        # 4. Quadratic Terms: Panamax Cross‑fill Reward.
        # For consecutive slots t and t+1 with lock types {"Panamax_A", "Panamax_B"},
        # add a bonus for pairing any ships across these slots.
        for t in range(T - 1):
            if {lock_types[t], lock_types[t+1]} == {"Panamax_A", "Panamax_B"}:
                for i in range(N):
                    for j in range(N):
                        idx_i = i * T + t
                        idx_j = j * T + (t + 1)
                        key = (idx_i, idx_j) if idx_i <= idx_j else (idx_j, idx_i)
                        Q[key] = Q.get(key, 0) - dynamic_lambda_crossfill

    return Q
