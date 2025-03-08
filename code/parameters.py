# Global tuning parameters

lambda_ship =  55  # ship-assignment constraint weight.
lambda_time_p =  100       # time slot pair penalty for Panamax (max one ship).
lambda_time_np  = 5        # time slot pair penalty for NeoPanamax (allow pair if “good”).
lambda_length_p = 150          # penalty for a long ship (L > 294) in a Panamax slot.
lambda_conflict  = 50          # penalty for mixing long ships (or long with any) in a NeoPanamax slot.
lambda_length_np  =100           # penalty if two short ships in NeoPanamax violate combined capacity.

lambda_once_reward = 65 # Used in evaluate solution, reward fro assigning ship
penalty_infeasible = 200 # Used in evalueate solution, Heavy penalty for infeasible assignments

dynamic_lambda_ship = 20  # Penalty for not scheduling a ship exactly once
dynamic_lambda_conflict = 10  # Penalty for exceeding two ships per time slot
dynamic_lambda_water = 0.1  # Weight for water cost in overall objective
dynamic_lambda_length = 20  # Penalty coefficient for tandem lockage length violations
dynamic_lambda_tandem = 10.0  # Reward for valid tandem lockage
dynamic_lambda_crossfill = 3.0  # Reward for cross filling
dynamic_penalty_infeasible = 1000  # Heavy penalty for infeasible assignments
dynamic_lambda_benefit = 2.0

crossfill_factor = 0.3  # Factor for cross filling