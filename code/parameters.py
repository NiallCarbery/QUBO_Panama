# Global tuning parameters

lambda_benefit = 2.75
lambda_once_reward = 65 
lambda_once_penalty = 19         # Penalty for not scheduling a ship exactly once
lambda_conflict = 11.5    # Penalty for exceeding two ships per time slot
lambda_water = 1.0        # Weight for water cost in overall objective
lambda_length = 45    # Penalty coefficient for tandem lockage length violations
lambda_tandem = 10      # Reward for valid tandem lockage
lambda_crossfill = 4.0  # Reward for cross filling
penalty_infeasible = 200 # Heavy penalty for infeasible assignments
crossfill_factor = 0.3    # Factor for cross filling

dynamic_lambda_ship = 20  # Penalty for not scheduling a ship exactly once
dynamic_lambda_conflict = 10  # Penalty for exceeding two ships per time slot
dynamic_lambda_water = 0.1  # Weight for water cost in overall objective
dynamic_lambda_length = 20  # Penalty coefficient for tandem lockage length violations
dynamic_lambda_tandem = 10.0  # Reward for valid tandem lockage
dynamic_lambda_crossfill = 3.0  # Reward for cross filling
dynamic_penalty_infeasible = 1000  # Heavy penalty for infeasible assignments
dynamic_lambda_benefit = 2.0

crossfill_factor = 0.3  # Factor for cross filling