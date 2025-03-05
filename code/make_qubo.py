from utils import get_lock_length, water_cost_for_slot
from parameters import *
import itertools

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

    # 1. Linear Terms: per ship and per slot.
    for i in range(N):
        for t in range(T):
            idx = i * T + t
            # If individual ship exceeds the slot's capacity, add infeasibility penalty.
            incomp = penalty_infeasible if L[i] > get_lock_length(lock_types[t]) else 0
            # Linear contribution: subtract benefit, add water cost, plus incompatibility penalty.
            Q[(idx, idx)] = Q.get((idx, idx), 0) - B[i] * lambda_benefit \
                             + water_cost_for_slot(lock_types[t], 1) * lambda_water + incomp

    # 2. Constraint: Each ship is scheduled exactly once or 0.
    # Enforce (sum_t x[i,t] - 1)^2 = sum_t x[i,t]^2 - 2 * sum_t x[i,t] + 1,
    for i in range(N):
        # List of variables for ship i.
        ship_vars = [i * T + t for t in range(T)]
        # Diagonal contributions: each variable gets -lambda_ship.
        for var in ship_vars:
            Q[(var, var)] = Q.get((var, var), 0) - lambda_ship
        # Off-diagonal contributions: for each unique pair add +2*lambda_ship.
        if dynamic == True:
            for a in range(len(ship_vars)):
                for b in range(a + 1, len(ship_vars)):
                    u = ship_vars[a]
                    v = ship_vars[b]
                    key = (u, v) if u <= v else (v, u)
                    Q[key] = Q.get(key, 0) + 2 * lambda_ship
        else:
            indices = [i * T + t for t in range(T)]
            for idx1, idx2 in itertools.combinations(indices, 2):
                key = (min(idx1, idx2), max(idx1, idx2))
                Q[key] = Q.get(key, 0) + 2 * lambda_ship

    # 3. Quadratic Terms: Tandem reward and combined length feasibility.
    for t in range(T):
        # Get the lock capacity for the current time slot.
        capacity = get_lock_length(lock_types[t])
        for i in range(N):
            for j in range(i + 1, N):
                idx_i = i * T + t
                idx_j = j * T + t
                key = (idx_i, idx_j) if idx_i <= idx_j else (idx_j, idx_i)
                
                q_val = - lambda_tandem
                
                combined_length = L[i] + L[j]
                if combined_length > capacity:
                    # Infeasible: the combined ship lengths exceed the lock capacity.
                    q_val += penalty_infeasible

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
                    Q[key] = Q.get(key, 0) - lambda_crossfill

    return Q
