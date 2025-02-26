import itertools
from utils import get_lock_length
from parameters import (
    lambda_ship,
    lambda_conflict,
    lambda_water,
    lambda_length,
    lambda_tandem,
)


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
            Q[(min(idx_i, idx_j), max(idx_i, idx_j))] = (
                Q.get((min(idx_i, idx_j), max(idx_i, idx_j)), 0) + 2 * lambda_conflict
            )

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
