import numpy as np

from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize

from utils import qubo_dict_to_matrix
from make_qubo import build_qubo
from evaluate import evaluate_solution
from embedding import evaluate_mapping_constrained
from utils import bitstring_to_assignment, generate_lock_types
from data_processing import generate_ship_data


def run_simulation(num_ships, num_time_slots):
    """
    Runs the simulation for a given number of ships.

    Parameters:
        num_ships (int): The number of ships to simulate.

    Returns:
        assignments (dict): Mapping from bitstring to its assignment dictionary.
        evaluated_solutions (dict): Mapping from bitstring to its evaluated solution.
        count_dict (dict): The raw counts of each bitstring from the final state sample.
    """
    # Generate ship data based on number of ships.
    ship_data = generate_ship_data(num_ships)
    L = ship_data["Length (m)"].to_numpy()
    B = ship_data["Benefit"].to_numpy()
    lock_types = generate_lock_types(num_time_slots)

    # Build the QUBO from the parameters.
    qubo = build_qubo(B, L, lock_types)
    Q = qubo_dict_to_matrix(qubo)

    # Symmetrize Q by adding its transpose, but keep the diagonal untouched.
    Q = Q + Q.T - np.diag(np.diag(Q))
    for i in range(len(B)):
        Q[i, i] = Q[-1, -1]  # ensure self-interaction gets the same value

    # Scale the QUBO matrix.
    Q = 0.05 * Q

    # Use a random initial guess for the optimizer.
    np.random.seed(0)
    x0 = np.random.random(len(Q) * 2)

    # Optimize the mapping using the Nelder-Mead method.
    res = minimize(
        evaluate_mapping_constrained,
        x0,
        args=(Q,),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None},
    )

    # Reshape the optimized solution into 2D coordinates.
    coords = np.reshape(res.x, (len(Q), 2))
    qubits = {f"q{i}": coord for (i, coord) in enumerate(coords)}

    # Create and draw the register.
    reg = Register(qubits)
    reg.draw(
        blockade_radius=DigitalAnalogDevice.rydberg_blockade_radius(1.0),
        draw_graph=True,
        draw_half_radius=True,
    )

    # Choose a median value from the positive entries of Q for Omega.
    Omega = np.median(Q[Q > 0].flatten())
    delta_0 = -5  # must be negative
    delta_f = -delta_0  # must be positive
    T_sim = 5000  # total simulation time in ns (long enough for propagation)

    # Create an adiabatic pulse with the interpolated waveforms.
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T_sim, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T_sim, [delta_0, 0, delta_f]),
        0,
    )

    # Declare the sequence for the simulation.
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")
    seq.draw()

    print("Running Simulation")
    # Run the simulation using the Qutip emulator.
    simul = QutipEmulator.from_sequence(seq)
    results = simul.run()

    final = results.get_final_state()
    count_dict = results.sample_final_state()

    print("Converting Bitstrings")
    # Convert bitstrings to assignments.
    # we pass T_sim directly. Adjust as needed if your bitstring_to_assignment expects something else.
    assignments = {
        bit: bitstring_to_assignment(bit, len(B), len(lock_types)) for bit in count_dict
    }

    # First, sort the bitstring keys from count_dict in descending order of count.
    sorted_bits = sorted(
        count_dict.keys(), key=lambda bit: count_dict[bit], reverse=True
    )

    # Determine the number of top bits to process (at least one).
    num_top = max(1, int(len(sorted_bits) * 0.1))

    # Slice the top 10% bitstrings.
    top_bits = sorted_bits[:num_top]

    # Evaluate only the top 20% assignments.
    evaluated_solutions = {
        bit: evaluate_solution(assignments[bit], B, L, lock_types) for bit in top_bits
    }

    # Print out the sorted top 10% bitstrings with their counts, assignments, and evaluations.
    for bit in top_bits:
        print(f"Bitstring: {bit}")
        print(f"  Count: {count_dict[bit]}")
        print(f"  Assignment: {assignments[bit]}")
        print(f"  Evaluation: {evaluated_solutions[bit]}")
        print("-" * 40)

    return assignments, evaluated_solutions, count_dict
