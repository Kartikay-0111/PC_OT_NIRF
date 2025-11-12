# /Project/optimization/run_parallel_pso.py

"""
Uses the 'src/parallel.py' module to run multiple independent
PSO optimizations in parallel (Multi-Start PSO).

This helps find a better global optimum by exploring different
random starting points simultaneously.

Run from project root:
    python optimization/run_parallel_pso.py
"""

import os
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# Ensure the 'src' dir is on sys.path so 'optimization' package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimization.parallel import run_parallel
from optimization.pso import run_pso_optimization, FEATURES

# --- Wrapper Function for Parallelism ---

def run_pso_task(run_id):
    """
    A simple wrapper for run_parallel.
    The 'run_id' is used to seed numpy for different random results.
    """
    # Seed numpy's random generator differently for each process
    # This is crucial for getting different PSO runs
    np.random.seed()
    
    print(f"[Main] Kicking off PSO task {run_id}...")
    
    # Call the imported function
    # We pass 20 particles / 50 iters to make it faster
    best_cost, best_pos = run_pso_optimization(n_particles=20, iters=50)
    
    return (run_id, best_cost, best_pos)


def save_results(score, df_results):
    """Saves the final optimization results to a text file."""
    results_dir = Path(__file__).resolve().parents[2] / "results"
    os.makedirs(results_dir, exist_ok=True)
    output_path = results_dir / "pso_optimization_results.txt"
    
    print(f"\nSaving final results to: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("--- 🏆 PSO Optimization Final Result ---\n\n")
        f.write(f"Best Possible TLR Score (Predicted): {score:.4f}\n\n")
        f.write("Optimal Subcomponent Scores:\n")
        f.write(df_results.to_string(index=False))
        
    print("✅ Results saved.")


def main():
    """Runs the parallel multi-start optimization."""
    
    # --- 1. Set up Tasks ---
    # We will run one PSO optimization for each CPU core
    num_parallel_runs = os.cpu_count() or 4
    print(f"--- Starting Parallel Multi-Start PSO ---")
    print(f"Will run {num_parallel_runs} independent PSO optimizations in parallel.\n")
    
    # Create the list of arguments for run_parallel
    # Each task just needs a unique ID
    tasks = [(i,) for i in range(num_parallel_runs)]
    
    start_time = time.time()

    # --- 2. Run in Parallel ---
    # This is where we use your 'parallel.py' module
    all_results = run_parallel(
        func=run_pso_task,
        args_list=tasks,
        backend="multiprocessing"
    )
    
    end_time = time.time()
    print(f"\n--- Parallel execution finished in {end_time - start_time:.2f} seconds ---")

    # --- 3. Find the Best Result ---
    if not all_results:
        print("Error: No results returned from parallel run.")
        return

    # Find the best result (lowest cost) from all runs
    best_run = min(all_results, key=lambda x: x[1])
    
    best_run_id = best_run[0]
    best_cost = best_run[1]
    best_pos = best_run[2]
    max_tlr_score = -best_cost

    # --- 4. Display Final Result ---
    print("\n--- 🏆 Overall Best Result (from all runs) ---")
    print(f"Found by Run ID: {best_run_id}")
    print(f"Best Possible TLR Score (Predicted): {max_tlr_score:.4f}")
    
    results = pd.DataFrame({
        'Component': FEATURES,
        'Optimal_Score': best_pos,
    })
    results['Optimal_Score'] = results['Optimal_Score'].round(2)
    print(results.to_string(index=False))

    # --- 5. Save Final Result ---
    save_results(max_tlr_score, results)


if __name__ == "__main__":
    main()