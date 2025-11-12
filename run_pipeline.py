"""
Main pipeline script to run the entire NIRF TLR optimization project.

This script executes the following steps in order:
1.  🍼 Generate Data: Creates the initial synthetic dataset.
2.  ✨ Preprocess & Visualize: Cleans the data, engineers new features,
    and generates EDA plots.
3.  🧠 Train Model: Trains a Random Forest model on the processed data.
4.  🚀 Optimize Scores: Runs a parallel PSO to find the optimal subcomponent
    scores that maximize the predicted TLR score.

To run the entire pipeline from the project root:
    python run_pipeline.py
"""

import subprocess
import sys
import os

# --- Configuration ---
PYTHON_EXE = sys.executable  # Use the same python interpreter that runs this script
SRC_DIR = "src"
OPTIMIZATION_DIR = os.path.join(SRC_DIR, "optimization")

# List of scripts to run in order
PIPELINE_STEPS = [
    {
        "name": "🍼 Generate Data",
        "script": os.path.join(SRC_DIR, "utils.py"),
        "description": "Creating the raw tlr_model_input.csv file."
    },
    {
        "name": "✨ Preprocess & Visualize Data",
        "script": os.path.join(SRC_DIR, "preprocess.py"),
        "description": "Cleaning data, adding features, and generating plots."
    },
    {
        "name": "🧠 Train Predictive Model",
        "script": os.path.join(SRC_DIR, "modeling.py"),
        "description": "Training Random Forest and saving tlr_rf_model.joblib."
    },
    {
        "name": "🚀 Find Optimal Scores via PSO",
        "script": os.path.join(OPTIMIZATION_DIR, "run_parallel_pso.py"),
        "description": "Running multi-start PSO to find the best possible TLR score."
    }
]

def run_step(step):
    """Executes a single pipeline step using subprocess."""
    name = step["name"]
    script_path = step["script"]
    description = step["description"]

    print(f"\n{'='*60}")
    print(f"▶️  Running Step: {name}")
    print(f"   - {description}")
    print(f"{'='*60}\n")

    if not os.path.exists(script_path):
        print(f"❌ ERROR: Script not found at '{script_path}'")
        return False

    try:
        # We use subprocess.run to wait for each script to complete.
        # check=True will raise an exception if the script returns a non-zero exit code.
        process = subprocess.run(
            [PYTHON_EXE, script_path],
            check=True,
            text=True,
            capture_output=False  # Set to False to see script output in real-time
        )
        print(f"\n✅ Step '{name}' completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR during step: {name}")
        print(f"   - Script '{script_path}' failed with exit code {e.returncode}.")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not execute '{PYTHON_EXE}'. Is Python installed and in your PATH?")
        return False


def main():
    """Runs the full pipeline."""
    print("🚀🚀🚀 Starting the NIRF TLR Optimization Pipeline 🚀🚀🚀")
    start_time = __import__("time").time()

    for step in PIPELINE_STEPS:
        success = run_step(step)
        if not success:
            print("\nPipeline halted due to an error.")
            break
    
    end_time = __import__("time").time()
    print(f"\n🎉🎉🎉 Pipeline Finished in {end_time - start_time:.2f} seconds 🎉🎉🎉")


if __name__ == "__main__":
    # Ensure the working directory is the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    main()
