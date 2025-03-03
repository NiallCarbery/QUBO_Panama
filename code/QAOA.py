import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.visualization import circuit_drawer

from utils import *
from make_qubo import build_qubo

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


def circuit_plot(num_ships, num_time_slots):
    B, L = generate_ship_parameters(num_ships)
    lock_types = generate_lock_types(num_time_slots)
    Q = build_qubo(B, L, lock_types)

    # Number of ships and time slots determine the total qubits:
    N = len(B)
    T = len(L)
    num_qubits = N * T

    # Define symbolic QAOA parameters:
    gamma = Parameter('γ')
    beta = Parameter('β')

    # Build the QAOA circuit based on the QUBO:
    qaoa_qc = build_qaoa_circuit(Q, num_qubits, gamma, beta)

    # Draw the circuit as an image in a separate Matplotlib window:
    return circuit_drawer(qaoa_qc, output='mpl')
    
