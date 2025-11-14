# VJTI NIRF Rank Optimization (OT\_Lab Project)

[](https://www.python.org/)
[](https://pandas.pydata.org/)
[](https://scikit-learn.org/)
[](https://pyswarms.readthedocs.io/)

This project analyzes the **Teaching-Learning and Resources (TLR)** parameter of the NIRF (National Institutional Ranking Framework), which carries 30% of the total weight for engineering institutions.

It uses a synthetic dataset to model the relationship between TLR subcomponents (like *Faculty-Student Ratio*, *Faculty Qualification*, etc.) and the final TLR score. It trains a Random Forest model to act as a "predictor" and then uses a parallelized **Particle Swarm Optimization (PSO)** to find the ideal combination of subcomponent scores that would yield the maximum possible TLR score for an institution.

The entire analysis pipeline can be executed by running a single command.

## 📂 Project Structure

```
.
├── .gitignore
├── data/
│   ├── processed/   (Generated: Cleaned, feature-engineered data)
│   └── raw/         (Empty: Place for initial raw data, if any)
├── Docs/            (Project documentation, reports)
├── models/          (Generated: The trained tlr_rf_model.joblib)
├── reports/
│   └── figures/     (Generated: Plots from visualize.py)
├── README.md
├── requirements.txt
├── run_pipeline.py  (Main executable script)
└── src/
    ├── __init__.py
    ├── modeling.py
    ├── optimization/
    │   ├── __init__.py
    │   ├── parallel.py
    │   ├── pso.py
    │   └── run_parallel_pso.py
    ├── preprocess.py
    ├── utils.py
    └── visualize.py
```

## 📄 File Descriptions

### Core Files (Root)

  * `run_pipeline.py`: **Main executable script.** Runs the entire project workflow from start to finish (data generation, preprocessing, training, and optimization).
  * `requirements.txt`: A list of all Python libraries (e.g., `pandas`, `scikit-learn`, `pyswarms`) needed for this project.
  * `.gitignore`: Tells Git to ignore files like `venv/`, `__pycache__/`, `data/processed/`, `models/`, and `reports/`.

### `src/` (Source Code)

  * `utils.py`: **Data Generator.** Creates the initial synthetic dataset (`tlr_model_input.csv`) that mimics NIRF data.
  * `preprocess.py`: **Data Processor.** Loads the raw data, cleans it, engineers new features (like `phd_per_faculty_ratio`), and saves the final processed file (`tlr_model_features.csv`).
  * `visualize.py`: **Data Analyst.** Generates exploratory data analysis (EDA) plots (like heatmaps and boxplots) and saves them to `reports/figures/`.
  * `modeling.py`: **The "Brain" Trainer.** Loads the *processed* data, trains a Random Forest Regressor to predict `tlr_score`, and saves the final trained model to `models/tlr_rf_model.joblib`.

### `src/optimization/` (Optimization Engine)

  * `pso.py`: **PSO Core.** Contains the main objective function and logic for running a *single* instance of the Particle Swarm Optimization.
  * `parallel.py`: **Parallel Utility.** A general-purpose helper module to run any function in parallel using Python's `multiprocessing` or `MPI`.
  * `run_parallel_pso.py`: **Parallel PSO Script.** Implements the "multi-start" optimization by using `parallel.py` to run multiple PSO instances simultaneously. This is the final step of the pipeline.

### Generated Folders (Not in Git)

  * `data/processed/`: Stores the intermediate CSV datasets created by `utils.py` and `preprocess.py`.
  * `models/`: Stores the final trained `.joblib` model file created by `modeling.py`.
  * `reports/figures/`: Stores all `.png` plots created by `visualize.py`.

-----

## 🚀 Setup and Installation

1.  **Clone the Repository**

    ```bash
    git clone https://your-github-repo-url.com/Project.git
    cd Project
    ```

2.  **Create and Activate a Virtual Environment**

    ```bash
    # Create the virtual environment
    python3 -m venv venv

    # Activate it (Linux/macOS)
    source venv/bin/activate

    # (Alternate) Activate it (Windows PowerShell)
    # .\venv\Scripts\Activate.ps1
    ```

3.  **Install Dependencies**
    With your virtual environment active, install all the required libraries:

    ```bash
    (venv) ➜ pip install -r requirements.txt
    ```

-----

## ⚡ How to Run the Project

With your virtual environment activated, just run the main pipeline script. This single command will execute the entire workflow.

```bash
(venv) ➜ python run_pipeline.py
```

### The Pipeline Workflow

The `run_pipeline.py` script executes the following steps in order:

1.  **Step 1: Data Generation (`utils.py`)**

      * Creates `data/processed/tlr_model_input.csv`.

2.  **Step 2: Data Preprocessing (`preprocess.py`)**

      * Reads the raw data, cleans it, and creates `data/processed/tlr_model_features.csv`.

3.  **Step 3: Model Training (`modeling.py`)**

      * Reads the cleaned data and saves the trained model to `models/tlr_rf_model.joblib`.

4.  **Step 4: Parallel Optimization (`run_parallel_pso.py`)**

      * Loads the trained model and runs multiple PSO instances in parallel.
      * Prints the **final, optimized set of TLR scores** to your terminal.
