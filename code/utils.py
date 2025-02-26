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
        return 55
    elif lock_type.startswith("NeoPanamax"):
        return 70
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
        ships = [i + 1 for i in range(num_ships) if sample[i * T + t] == 1]
        print(f"  Time slot {t} (Lock type: {lock_types[t]}): Ships {ships}")


def baseline_water_usage(lock_types, num_ships):
    cost = 0
    # Now, the number of time slots equals num_ships.
    for t in range(num_ships):
        cost += water_cost_for_slot(lock_types[t], 1)
    return cost
