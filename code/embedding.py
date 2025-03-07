import numpy as np
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path


from utils import generate_lock_types, qubo_dict_to_matrix
from data_processing import generate_ship_data
from make_qubo import build_qubo


from pulser import Register
from pulser.devices import DigitalAnalogDevice

from scipy.optimize import minimize, curve_fit
from scipy.spatial.distance import pdist, squareform

def evaluate_mapping_constrained(new_coords, Q,
                                 d_min_allowed=4e-6,   # minimum separation: 4 μm
                                 r_max=50e-6,          # maximum distance from origin: 50 μm
                                 lambda_neighbors=1e3, # penalty factor for neighbours too close
                                 lambda_origin=1e3):    # penalty factor for being outside allowed radius
    """
    Evaluates the cost of a given embedding configuration.
    
    It consists of three parts:
      1. The base cost comparing the computed interaction matrix
         (assuming a 1/r^6 dependence) with the target QUBO, Q.
      2. A neighbor penalty that adds a cost when any pair of atoms have a distance
         less than the allowed minimum (4 μm).
      3. A radial penalty that adds a cost when any atom is farther than r_max (50 μm) from the origin.
    
    These extra penalty terms force the optimization toward qubit placements that
    meet the device’s spatial constraints.
    """
    n = len(Q)
    coords = np.reshape(new_coords, (n, 2))
    
    # Compute pairwise distances.
    distances = pdist(coords)
    distances = np.clip(distances, 1e-8, None)
    
    # Base cost: how well do the computed interactions (using 1/r^6 scaling) match Q.
    computed_Q = squareform(DigitalAnalogDevice.interaction_coeff / distances**6)
    base_cost = np.linalg.norm(computed_Q - Q)
    
    # Penalty for violations of the minimum inter-atom distance.
    penalty_neighbors = lambda_neighbors * np.sum(np.maximum(0, d_min_allowed - distances)**2)
    
    # Penalty for atoms outside the allowed circle (distance > r_max).
    radii = np.linalg.norm(coords, axis=1)
    penalty_origin = lambda_origin * np.sum(np.maximum(0, radii - r_max)**2)
    
    total_cost = base_cost + penalty_neighbors + penalty_origin
    return total_cost


def run_embedding_for_qubits(n, rep, embeddings_dir, save_embedding=False):
    """
    Runs one embedding instance for n qubits (with num_ships = num_time_slots = n).
    Uses the constrained cost function to encourage a configuration:
       - With every atom within 50 μm from the origin.
       - With a minimum separation of 4 μm between any two atoms.
    
    If save_embedding is True (e.g. for rep==0), the embedding image and JSON data are stored.
    Returns:
         elapsed: The run time for this embedding.
         qubit_positions: The dictionary mapping of qubit positions (if saved).
    """
    start_time = time.time()

    # Create data and QUBO.
    ship_data = generate_ship_data(np.sqrt(n))
    L = ship_data["Length (m)"].to_numpy()
    B = ship_data["Benefit"].to_numpy()
    lock_types = generate_lock_types(int(np.sqrt(n)))
    qubo = build_qubo(B, L, lock_types)
    Q = qubo_dict_to_matrix(qubo)
    
    # Optimization: use a random initial guess (2 coordinates per qubit)
    x0 = np.random.randn(n * 2)
    res = minimize(
        evaluate_mapping_constrained,
        x0,
        args=(Q,),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None}
    )
    coords = np.reshape(res.x, (n, 2))
    qubit_positions = {f"q{i}": coords[i] for i in range(n)}
    
    # Create the register (for drawing) using the Pulsar API.
    reg = Register(qubit_positions)
    
    if save_embedding:
        # Draw and save the embedding.
        blockade_radius = DigitalAnalogDevice.rydberg_blockade_radius(1.0)
        img_filename = str(embeddings_dir / f"embedding_{n}.png")
        reg.draw(
            blockade_radius=blockade_radius,
            draw_graph=True,
            draw_half_radius=True,
            fig_name=img_filename,
            kwargs_savefig={},
            show=False
        )
        print(f"[INFO] Saved embedding image: {img_filename}")
        
        # Save the qubit location mapping as JSON.
        reg_filename = str(embeddings_dir / f"register_{n}.json")
        with open(reg_filename, "w") as f:
            json.dump({q: coord.tolist() for q, coord in qubit_positions.items()}, f, indent=4)
        print(f"[INFO] Saved register data: {reg_filename}")
    
    elapsed = time.time() - start_time
    print(f"[INFO] Time for {n} qubits, rep {rep}: {elapsed:.4f} seconds")
    return elapsed, qubit_positions if save_embedding else None

# --------------------------------------------------
# Experiment and Graphing Framework (As Before)
# --------------------------------------------------
def run_experiment():
    """
    For each qubit count in a defined list, run several repetitions
    using the constrained embedding process.
    
    Only the first repetition’s embedding (the qubit positions) is saved.
    Aggregate the runtime data for later analysis.
    """
    num_qubits_list = [4, 9, 16, 25, 36]
    reps = 5  # number of repetitions per qubit count

    current_dir = Path(__file__).resolve().parent  # "code" folder
    results_dir = current_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = results_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    runtime_results = {}
    for n in num_qubits_list:
        times = []
        print(f"\n[EXPERIMENT] Running {reps} embeddings for {n} qubits...")
        for rep in range(reps):
            save_flag = (rep == 0)
            t, _ = run_embedding_for_qubits(n, rep, embeddings_dir, save_embedding=save_flag)
            times.append(t)
        runtime_results[n] = times
        median_t = np.median(times)
        print(f"[RESULT] {n} qubits: median time = {median_t:.4f} seconds over {reps} runs")
    return runtime_results, embeddings_dir


def power_law(x, A, B):
    """Model: y = A * x^B"""
    return A * x**B

def plot_runtime_results(runtime_results, embeddings_dir):
    """
    Plots a scatter plot of the average runtime versus number of qubits with error bars 
    (standard error of the mean) and overlays a power-law fit.
    
    Parameters:
      runtime_results: Dictionary mapping qubit count to a list of runtime measurements.
      embeddings_dir: A Path object pointing to the results/embeddings directory.
    
    This function:
      1. Computes the mean runtime and standard error for each qubit count.
      2. Generates a scatter plot with these error bars.
      3. Uses scipy.curve_fit to fit the power-law model: y = A * x^B, weighting by the SEM.
      4. Overlays the fitted curve along with a legend showing the fit parameters and errors.
      5. Saves the plot as a PNG file and the runtime/fit data in a JSON file.
    """
    # Get qubit counts and compute average runtime and SEM (standard error of the mean)
    num_qubits = np.array(sorted(runtime_results.keys()))
    mean_times = np.array([np.mean(runtime_results[n]) for n in num_qubits])
    errors = np.array([
        np.std(runtime_results[n], ddof=1) / np.sqrt(len(runtime_results[n]))
        for n in num_qubits
    ])
    
    # Create scatter plot with error bars.
    fig, ax = plt.subplots()
    ax.errorbar(num_qubits, mean_times, yerr=errors, fmt='o', color='blue',
                capsize=5, label="Average runtime", ecolor="black", markersize=8)
    
    # Use curve_fit for the power-law model; provide an initial guess and increase maxfev as needed.
    # Here p0=(mean_times[0], 1) is an example guess with A roughly the first mean runtime and B = 1.
    popt, pcov = curve_fit(
        power_law,
        num_qubits,
        mean_times,
        sigma=errors,
        absolute_sigma=True,
        p0=(mean_times[0], 1),
        maxfev=10000
    )
    
    # Extract parameters and their uncertainties.
    A, B = popt
    perr = np.sqrt(np.diag(pcov))
    A_err, B_err = perr[0], perr[1]
    
    # Generate a smooth fit curve.
    x_fit = np.linspace(num_qubits.min(), num_qubits.max(), 100)
    y_fit = power_law(x_fit, A, B)
    
    # Prepare a label including the fit parameters with their uncertainties.
    label_fit = (f"Power Law Fit: y = {A:.2e} * x^{B:.2e}\n" +
                 f"(A_err = {A_err:.2e}, B_err = {B_err:.2e})")
    ax.plot(x_fit, y_fit, 'r--', label=label_fit)
    
    # Label plot and add a legend.
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Embedding Time (seconds)")
    ax.set_title("Embedding Time vs. Number of Qubits (Power Law Fit)")
    ax.grid(True)
    ax.legend()
    
    # Save the plot as a PNG image.
    graph_filename = str(embeddings_dir / "runtime_vs_qubits_scatter_powerlaw_fit.png")
    plt.savefig(graph_filename)
    plt.close()
    print(f"[INFO] Saved power-law scatter plot: {graph_filename}")
    
    # Save data and fit parameters to a JSON file.
    graph_data = {
        "runtime_results": runtime_results,
        "mean_times": mean_times.tolist(),
        "errors": errors.tolist(),
        "fit_params": {
            "A": A,
            "B": B,
            "A_error": A_err,
            "B_error": B_err
        }
    }
    graph_data_filename = str(embeddings_dir / "graph_data_powerlaw_fit.json")
    with open(graph_data_filename, "w") as f:
        json.dump(graph_data, f, indent=4)
    print(f"[INFO] Saved power-law fit graph data to: {graph_data_filename}")


def main():
    runtime_results, embeddings_dir = run_experiment()
    plot_runtime_results(runtime_results, embeddings_dir)

if __name__ == '__main__':
    main()
