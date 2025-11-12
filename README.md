# OT_Lab Project

This project analyzes and optimizes the Teaching-Learning and Resources (TLR) score from the National Institutional Ranking Framework (NIRF) using machine learning and optimization techniques. It uses a synthetic dataset to model the relationship between TLR subcomponents and the final score, then employs Particle Swarm Optimization (PSO) to identify the ideal combination of subcomponent scores to achieve the maximum possible TLR score.

## Project Layout

project/
├─ data/
│   ├─ raw/           # Raw NIRF forms, downloaded datasets (CSV/JSON)
│   └─ processed/     # Cleaned/merged data for analysis
├─ src/
│   ├─ preprocess.py
│   ├─ visualize.py
│   ├─ modeling.py
│   ├─ optimization/
│   │   ├─ pso.py
│   ├─ parallel.py
│   └─ utils.py
├─ parallel/          # (Optional) C/C++ code examples for OpenMP/MPI
├─ results/           # Generated figures, tables, model outputs
└─ requirements.txt

## File Descriptions

- **data/raw**: Raw NIRF forms and datasets downloaded from the official sources.
- **data/processed**: Cleaned and merged datasets for analysis.
- **src/preprocess.py**: Preprocessing code to clean and transform raw data.
- **src/visualize.py**: Visualization code to plot the data and model results.
- **src/modeling.py**: Modeling code to build and train the machine learning models.
- **src/optimization/pso.py**: Particle Swarm Optimization code to find the ideal combination of subcomponent scores.
- **src/optimization/parallel.py**: Parallel processing code to speed up the optimization.
- **src/utils.py**: Utility code for common functions.
- **parallel/**: (Optional) C/C++ code examples for OpenMP/MPI.
- **results/**: Generated figures, tables, and model outputs.

## Usage

- Put raw data in data/raw
- Place cleaned/merged datasets in data/processed
- Use src/ for code (see module stubs)

Run virtual environment
source venv/bin/activate
