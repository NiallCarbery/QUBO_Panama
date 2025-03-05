# Global tuning parameters

lambda_ship = 15  # Penalty for not scheduling a ship exactly once
lambda_water = 2  # Weight for water cost in overall objective
lambda_tandem = 10  # Reward for valid tandem lockage
lambda_crossfill = 3.0 # Reward for cross filling
lambda_benefit = 15 # Reward for ship assignment scalar
penalty_infeasible = 1000  # Heavy penalty for infeasible assignments

crossfill_factor = 0.3  # Factor for cross filling