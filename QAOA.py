import itertools
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.visualization import circuit_drawer

# --- Updated get_lock_length using provided lock types ---
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

# --- QUBO Builder for Ship Scheduling ---

def build_qubo(B, L, lock_types, lambda_ship, lambda_conflict, lambda_length):
    """
    Build the QUBO for ship scheduling.
    Decision variable: x[i,t] = 1 if ship i is scheduled in time slot t.
    
    The QUBO includes:
      - Reward term: subtract ship benefit.
      - Ship-once constraint: each ship must be scheduled exactly once.
      - Time-slot capacity constraint: at most two ships per time slot.
      - Tandem lockage length constraint: for each time slot and each pair of ships,
        if their combined length exceeds the available lock length, add a penalty term.
    """
    N = len(B)           # number of ships
    T = len(lock_types)  # number of time slots
    Q = {}
    
    # --- Linear reward term: subtract benefit for each assignment ---
    for i in range(N):
        for t in range(T):
            idx = i * T + t
            # Subtract the benefit B[i] for assigning ship i at time slot t.
            Q[(idx, idx)] = Q.get((idx, idx), 0) - B[i]
    
    # --- Ship-once constraint: each ship must be scheduled exactly once ---
    for i in range(N):
        indices = [i * T + t for t in range(T)]
        for idx in indices:
            Q[(idx, idx)] = Q.get((idx, idx), 0) + lambda_ship - 2 * lambda_ship
        for idx1, idx2 in itertools.combinations(indices, 2):
            key = (min(idx1, idx2), max(idx1, idx2))
            Q[key] = Q.get(key, 0) + 2 * lambda_ship

    # --- Time-slot capacity constraint: at most two ships per time slot ---
    for t in range(T):
        for i, j in itertools.combinations(range(N), 2):
            idx_i = i * T + t
            idx_j = j * T + t
            key = (min(idx_i, idx_j), max(idx_i, idx_j))
            Q[key] = Q.get(key, 0) + 2 * lambda_conflict

    # --- Tandem lockage length constraint ---
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
    return Q

# --- QUBO-to-Ising conversion and QAOA circuit builder ---

def qubo_to_ising(Q, num_qubits):
    """
    Converts a QUBO defined in dictionary Q into Ising parameters.
    Returns:
      constant      : the constant offset.
      linear_coeffs : numpy array of linear coefficients for each qubit.
      two_qubit_terms : dictionary with keys (i,j) and coefficients for Z_i Z_j.
    """
    constant = 0
    linear_coeffs = np.zeros(num_qubits)
    two_qubit_terms = {}
    for (i, j), coeff in Q.items():
        if i == j:
            constant += coeff * (1/2)
            linear_coeffs[i] += -coeff * (1/2)
        else:
            constant += coeff / 4
            linear_coeffs[i] += -coeff / 4
            linear_coeffs[j] += -coeff / 4
            key = (min(i, j), max(i, j))
            two_qubit_terms[key] = two_qubit_terms.get(key, 0) + coeff / 4
    return constant, linear_coeffs, two_qubit_terms

def build_qaoa_circuit(Q, num_qubits, gamma, beta):
    """
    Build a QAOA circuit for the given QUBO dictionary Q.
    
    Parameters:
      Q         : QUBO dictionary.
      num_qubits: total number of qubits (should equal number of QUBO variables).
      gamma, beta: QAOA parameters (can be Parameter objects for parametrization).
      
    Returns:
      qc        : the QAOA QuantumCircuit.
    """
    qc = QuantumCircuit(num_qubits)
    
    # Initial state: uniform superposition via Hadamards
    for i in range(num_qubits):
        qc.h(i)
    
    # Convert the QUBO to an Ising Hamiltonian
    constant, linear, two_qubit = qubo_to_ising(Q, num_qubits)
    
    # Apply the cost unitary U_C(gamma)
    for i in range(num_qubits):
        if abs(linear[i]) > 1e-8:
            qc.rz(2 * gamma * linear[i], i)
    
    for (i, j), coeff in two_qubit.items():
        qc.cx(i, j)
        qc.rz(2 * gamma * coeff, j)
        qc.cx(i, j)
    
    # Apply the mixer unitary U_M(beta)
    for i in range(num_qubits):
        qc.rx(2 * beta, i)
    
    return qc

# Paper QAOA Implemntation Example
B = [10, 15, 20]               # Benefits per ship
L = [30, 40, 35]               # Ship lengths
lock_types = ["Panamax A", "NeoPanamax B"]  # Two time slots; names determine available lock lengths

# Penalty weights:
lambda_ship = 20
lambda_conflict = 100
lambda_length = 1.0

# Build the QUBO dictionary using the ship scheduling builder:
Q = build_qubo(B, L, lock_types, lambda_ship, lambda_conflict, lambda_length)

# Number of ships and time slots determine the total qubits:
N = len(B)
T = len(lock_types)
num_qubits = N * T

# Define symbolic QAOA parameters:
gamma = Parameter('γ')
beta = Parameter('β')

# Build the QAOA circuit based on the QUBO:
qaoa_qc = build_qaoa_circuit(Q, num_qubits, gamma, beta)

# Draw the circuit as an image in a separate Matplotlib window:
circuit_drawer(qaoa_qc, output='mpl')
plt.show()
