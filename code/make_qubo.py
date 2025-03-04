import itertools
from utils import get_lock_length, water_cost_for_slot
from parameters import *


def build_qubo(B, L, lock_types):
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

    # 1. Linear Terms: iterate by ship and time slot.
    for i in range(N):
        for t in range(T):
            idx = i * T + t
            # Apply infeasibility penalty if ship length exceeds slot's allowed length.
            incomp = penalty_infeasible if L[i] > get_lock_length(lock_types[t]) else 0
            # Update linear term: subtract ship benefit, add water cost and incompatibility cost.
            Q[(idx, idx)] = Q.get((idx, idx), 0) - B[i] + water_cost_for_slot(lock_types[t], 1) + incomp

    # 2. Quadratic Terms: Tandem reward for ships scheduled together in the same slot.
    for t in range(T):
        for i in range(N):
            for j in range(i + 1, N):
                idx_i = i * T + t
                idx_j = j * T + t
                key = (idx_i, idx_j) if idx_i <= idx_j else (idx_j, idx_i)
                Q[key] = Q.get(key, 0) - lambda_tandem

    # 3. Constraint: Each ship is scheduled exactly once.
    # Enforce (sum_t x[i,t] - 1)^2 = sum_t x[i,t]^2 - 2 * sum_t x[i,t] + 1,
    # dropping the constant term.
    for i in range(N):
        # List of variables for ship i.
        ship_vars = [i * T + t for t in range(T)]
        # Diagonal contributions: each variable gets -lambda_ship.
        for var in ship_vars:
            Q[(var, var)] = Q.get((var, var), 0) - lambda_ship
        # Off-diagonal contributions: for each unique pair add +2*lambda_ship.
        for a in range(len(ship_vars)):
            for b in range(a + 1, len(ship_vars)):
                u = ship_vars[a]
                v = ship_vars[b]
                key = (u, v) if u <= v else (v, u)
                Q[key] = Q.get(key, 0) + 2 * lambda_ship

    # 4. Constraint: Each time slot must host exactly 2 ships.
    # Enforce (sum_i x[i,t] - 2)^2 = sum_i x[i,t]^2 - 4 * sum_i x[i,t] + 4,
    # dropping the constant term.
    for t in range(T):
        # List of variables for slot t.
        slot_vars = [i * T + t for i in range(N)]
        # Diagonal contributions: each gets -3*lambda_lock.
        for var in slot_vars:
            Q[(var, var)] = Q.get((var, var), 0) - 3 * lambda_length
        # Off-diagonal contributions: each unique pair gets +2*lambda_lock.
        for a in range(len(slot_vars)):
            for b in range(a + 1, len(slot_vars)):
                u = slot_vars[a]
                v = slot_vars[b]
                key = (u, v) if u <= v else (v, u)
                Q[key] = Q.get(key, 0) + 2 * lambda_length

    # 5. Quadratic Terms: Panamax Cross‑fill Reward.
    # For consecutive slots t and t+1, check if the lock types form the pair {"Panamax_A", "Panamax_B"}.
    for t in range(T - 1):
        if {lock_types[t], lock_types[t+1]} == {"Panamax_A", "Panamax_B"}:
            for i in range(N):
                for j in range(N):
                    idx_i = i * T + t
                    idx_j = j * T + (t + 1)
                    key = (idx_i, idx_j) if idx_i <= idx_j else (idx_j, idx_i)
                    Q[key] = Q.get(key, 0) - lambda_crossfill

    return Q