# Global tuning parameters

lambda_ship = 20  # Penalty for not scheduling a ship exactly once
lambda_conflict = 10  # Penalty for exceeding two ships per time slot
lambda_water = 2.0  # Weight for water cost in overall objective
lambda_length = 20  # Penalty coefficient for tandem lockage length violations
lambda_tandem = 10.0  # Reward for valid tandem lockage
lambda_crossfill = 3.0  # Reward for cross filling
penalty_infeasible = 1000  # Heavy penalty for infeasible assignments
crossfill_factor = 0.3  # Factor for cross filling
instance_sizes = list(range(3, 10, 2))  # Instance sizes (number of ships)
NUM_READS = 10  # Number of reads for the SimulatedAnnealingSampler
