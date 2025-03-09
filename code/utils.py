import numpy as np

def generate_lock_types(num_slots):
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
    if lock_type.startswith("Panamax"):
        return 294
    elif lock_type.startswith("NeoPanamax"):
        return 366
    else:
        raise ValueError("Unknown lock type")


def water_cost_for_slot(lock_type, count):
    if count == 0:
        return 0
    elif count in [1, 2]:
        if lock_type.startswith("Panamax"):
            return 30
        elif lock_type.startswith("NeoPanamax"):
            return 45
    else:
        return 300


def print_timetable(sample, num_ships, num_slots, lock_types):
    T = num_slots
    print("\nTimetable by Time Slot:")
    for t in range(num_slots):
        ships = [i + 1 for i in range(num_ships) if sample.get(i * T + t, 0) == 1]
        print(f"  Time slot {t+1} (Lock type: {lock_types[t]}): Ships {ships}")


def baseline_water_usage(lock_types, num_time_slots):
    cost = 0
    # Now, the number of time slots equals num_ships.
    for t in range(num_time_slots):
        cost += water_cost_for_slot(lock_types[t], 1)
    return cost


def print_results(n, T, instance_results):
    """
    Print the results of the runs.
    """
    print(f"Instance with {n} ships (and {T} time slots):")
    print(f"  Optimized total water cost = {instance_results[0]}")
    print(f"  Baseline total water cost   = {instance_results[1]}")
    print(f"  Number of feasible solutions: {instance_results[6]}")
    print(f"  Number of infeasible solutions: {instance_results[7]}")
    print(f"  Tandem lockages used: {instance_results[8]}")
    print(f"  Cross fills applied: {instance_results[9]}")
    print(f"  Length of ships: {instance_results[3]}")
    print(f"  Benefit of ships: {instance_results[2]}")
    if instance_results[10]:
        print(f"  Infeasibility reasons: {instance_results[10]}")

    if instance_results[5] is not None:
        print("  Timetable for best solution:")
        print_timetable(instance_results[5], n, T, instance_results[4])
    print("\n" + "-" * 50 + "\n")

def qubo_dict_to_matrix(qubo):
    """
    Convert a QUBO dictionary to a matrix.
    
    Parameters:
      qubo (dict): A dictionary where keys are tuples (i, j) and values are coefficients.
    
    Returns:
      np.ndarray: A 2D numpy array representing the QUBO matrix.
    """
    # Find the maximum index in the keys to determine the matrix size.
    max_index = 0
    for i, j in qubo.keys():
        max_index = max(max_index, i, j)
    n = max_index + 1
    
    # Initialize an n x n matrix of zeros.
    Q = np.zeros((n, n))
    
    # Fill in the matrix using dictionary items.
    for (i, j), value in qubo.items():
        Q[i, j] = value

    return Q

def bitstring_to_assignment(bitstring, measurement_results, num_ships=2, num_time_slots=2):
    """
    Converts a bitstring into an assignment dictionary.
    For example, with 2 ships and 2 time slots (4 bits), '1010' becomes:
    {0: 1, 1: 0, 2: 1, 3: 0}
    """
    expected_length = num_ships * num_time_slots
    if len(bitstring) != expected_length:
        raise ValueError(f"Expected bitstring of length {expected_length}, got {len(bitstring)}")
    # Create the dictionary by enumerating the bitstring
    assignment = {i: int(bit) for i, bit in enumerate(bitstring)}
    return assignment
