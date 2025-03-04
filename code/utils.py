import random
from parameters import penalty_infeasible


def generate_ship_parameters(num_ships):
    B = [random.randint(8, 12) for _ in range(num_ships)]
    L = [random.randint(25, 60) for _ in range(num_ships)]
    return B, L


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
            return 80
    else:
        return penalty_infeasible * count


def print_timetable(sample, num_ships, num_slots, lock_types):
    T = num_slots
    print("\nTimetable by Time Slot:")
    for t in range(num_slots):
        ships = [i + 1 for i in range(num_ships) if sample.get(i * T + t, 0) == 1]
        print(f"  Time slot {t} (Lock type: {lock_types[t]}): Ships {ships}")


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
    if instance_results[10]:
        print(f"  Infeasibility reasons: {instance_results[10]}")

    if instance_results[5] is not None:
        print("  Timetable for best solution:")
        print_timetable(instance_results[5], n, T, instance_results[4])
    print("\n" + "-" * 50 + "\n")
