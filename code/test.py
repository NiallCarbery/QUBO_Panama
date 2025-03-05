from make_qubo import build_qubo

from qadence import QuantumModel, QuantumCircuit, Register
from qadence import RydbergDevice, AnalogRX, AnalogRZ, chain

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from qadence import RydbergDevice



def qubo_register_coords(Q: np.ndarray, device: RydbergDevice) -> list:
    """Compute coordinates for register."""

    def evaluate_mapping(new_coords, *args):
        """Cost function to minimize. Ideally, the pairwise
        distances are conserved"""
        Q, shape = args
        new_coords = np.reshape(new_coords, shape)
        interaction_coeff = device.rydberg_level
        new_Q = squareform(interaction_coeff / pdist(new_coords) ** 6)
        return np.linalg.norm(new_Q - Q)

    shape = (len(Q), 2)
    np.random.seed(0)
    x0 = np.random.random(shape).flatten()
    res = minimize(
        evaluate_mapping,
        x0,
        args=(Q, shape),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None},
    )
    return [(x, y) for (x, y) in np.reshape(res.x, (len(Q), 2))]

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


Q = qubo_dict_to_matrix(build_qubo([200,250], [8,6], ['Panamax_A', 'NeoPanamax']))
print("QUBO Matrix Representation:")
print(Q)


def loss(model: QuantumModel, *args) -> tuple[float, dict]:
    to_arr_fn = lambda bitstring: np.array(list(bitstring), dtype=int)
    cost_fn = lambda arr: arr.T @ Q @ arr
    samples = model.sample({}, n_shots=1000)[0]  # extract samples
    cost_fn = sum(samples[key] * cost_fn(to_arr_fn(key)) for key in samples)
    return cost_fn / sum(samples.values()), {}  # We return an optional metrics dict

# Device specification and atomic register
device = RydbergDevice(rydberg_level=70)

reg = Register.from_coordinates(
    qubo_register_coords(Q, device), device_specs=device
)

# Analog variational quantum circuit
layers = 2
block = chain(*[AnalogRX(f"t{i}") * AnalogRZ(f"s{i}") for i in range(layers)])
circuit = QuantumCircuit(reg, block)

model = QuantumModel(circuit)
initial_counts = model.sample({}, n_shots=1000)[0]
print(initial_counts)



# Assume layers is already defined (layers = 2 in your example)
# And your model (QuantumModel) and loss function have been defined.

def objective_fn(params):
    """
    Converts the list of parameters into a dict required by the model
    and computes the loss. The parameters are arranged in pairs:
    (t0, s0, t1, s1, ...).
    """
    # Convert the flat list of parameters into a dictionary:
    # t_i for the mixing rotation, s_i for the embedding rotation.
    param_dict = {}
    for i in range(layers):
        param_dict[f"t{i}"] = params[2 * i]
        param_dict[f"s{i}"] = params[2 * i + 1]
        
    # Evaluate the loss function on the current model parameters.
    # The loss function is assumed to return a tuple (cost, metrics), 
    # but we use only the cost for optimization.
    cost, _ = loss(model, param_dict)
    return cost

# Provide an initial guess for the parameter vector.
# Here we choose random numbers in [0, 2π), but you might want to adjust this.
initial_params = np.random.uniform(0, 2 * np.pi, size=layers * 2)

# Use the Nelder-Mead algorithm, a gradient-free optimization method.
result = minimize(
    objective_fn,
    initial_params,
    method="Nelder-Mead",
    options={"maxiter": 200, "disp": True}
)

print("Optimal parameters found:", result.x)
print("Minimum cost found:", result.fun)
