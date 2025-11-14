# /Project/optimization/pso.py

"""
Implements Particle Swarm Optimization (PSO) to find the combination
of TLR subcomponent scores that maximizes the predicted TLR score.

This file is now refactored to be importable and parallel-safe.
- run_pso_optimization() contains the core logic.
- The main() function runs a single instance.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Import the GlobalBestPSO optimizer
try:
    import pyswarms.single.global_best as gbpso
except ImportError:
    print("Error: 'pyswarms' library not found.")
    print("Please install it first: pip install pyswarms")
    exit()

# --- Constants ---
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "..", "models", "tlr_rf_model.joblib"
)
# UPDATE: The features list must match the one used for training the model.
FEATURES = [
    'ss_score', 'fsr_score', 'fqe_score', 
    'fru_score', 'oe_score', 'mir_score',
    'phd_per_faculty_ratio', 'students_per_faculty', 'total_expense_per_student'
]

# Global variable for the model
PREDICTIVE_MODEL = None


def load_model(path):
    """Loads the saved scikit-learn model into the global variable."""
    global PREDICTIVE_MODEL
    # Check if it's already loaded in this process
    if PREDICTIVE_MODEL is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at {path}\n"
                "Please run 'python src/modeling.py' first."
            )
        
        print(f"[Process {os.getpid()}] Loading predictive TLR model...")
        PREDICTIVE_MODEL = joblib.load(path)


def get_bounds():
    """Returns the lower and upper bounds for the search space."""
    # Order must match the FEATURES list.
    # Original scores have their own scales.
    # New engineered features were normalized (0-1), so their bounds are [0, 1].
    lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    upper_bounds = [20, 30, 20, 30, 10, 5, 1, 1, 1]
    return (np.array(lower_bounds), np.array(upper_bounds))


def tlr_objective_function(particles):
    """The objective function to be minimized by PSO."""
    
    # If this is a new process, the model will be None. Load it.
    if PREDICTIVE_MODEL is None:
        load_model(MODEL_PATH)
        
    predictions = PREDICTIVE_MODEL.predict(particles)
    # Return negative score for minimization
    return -predictions


def run_pso_optimization(n_particles=50, iters=100):
    """
    Runs a single, complete PSO optimization instance.
    This function is importable by other scripts.
    """
    print(f"[Process {os.getpid()}] Running PSO with {n_particles} particles for {iters} iterations...")
    
    bounds = get_bounds()
    dimensions = len(FEATURES)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    optimizer = gbpso.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options,
        bounds=bounds
    )

    # Run the optimization
    best_cost, best_pos = optimizer.optimize(tlr_objective_function, iters=iters)
    
    print(f"[Process {os.getpid()}] Finished. Best cost: {-best_cost:.4f}")
    return (best_cost, best_pos)


def main():
    """Main function to run a *single* instance of the PSO."""
    
    print("--- Starting Single PSO for TLR Score Maximization ---")
    
    try:
        # Pre-load the model in the main process
        load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Run the optimization
    best_cost, best_pos = run_pso_optimization()
    
    # 6. Display the results
    max_tlr_score = -best_cost

    print("\n--- PSO Optimization Complete ---")
    print(f"\nBest Possible TLR Score (Predicted): {max_tlr_score:.4f}")
    
    results = pd.DataFrame({
        'Component': FEATURES,
        'Optimal_Score': best_pos,
    })
    results['Optimal_Score'] = results['Optimal_Score'].round(2)
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()