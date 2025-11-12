# src/visualize.py

"""
Generates exploratory data analysis (EDA) plots for the NIRF TLR project.
This supports "Phase 1: Baseline Data and Diagnostics"[cite: 26, 27].

Plots generated:
1. Boxplots of all TLR subcomponent scores.
2. A correlation heatmap between scores.
3. A radar chart comparing VJTI vs. a top-tier institute.

Run from project root:
    python src/visualize.py
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Constants ---

# Assumes this script is in 'src/' and data is in 'data/processed/'
# Use the FEATURED data, not the raw input data
DATA_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "data", "processed", "tlr_model_features.csv"
)

# Directory to save generated plots
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), 
    "..", "results", "figures"
)

# Columns to be used for plotting
SCORE_COLS = [
    'ss_score', 'fsr_score', 'fqe_score', 'fru_score', 
    'oe_score', 'mir_score', 'tlr_score'
]

# Columns for the subcomponents only
SUB_COLS = SCORE_COLS[:-1] # All except tlr_score


def load_data(path):
    """Loads the processed TLR dataset."""
    if not os.path.exists(path):
        print(f"Error: Data file not found at {path}")
        print("Please run 'python src/preprocess.py' first to generate the data.")
        return None
    
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

def plot_score_distributions(df, columns, output_dir):
    """
    Generates and saves boxplots for each score column to show distributions.
    """
    print("Generating score distribution boxplots...")
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df[columns], orient='h', palette='viridis')
    plt.title('Distribution of NIRF TLR Subcomponent Scores (All Institutes, All Years)', fontsize=16)
    plt.xlabel('Score')
    
    output_path = os.path.join(output_dir, 'score_distributions_boxplot.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_correlation_heatmap(df, columns, output_dir):
    """
    Generates and saves a heatmap of the correlation matrix for all scores.
    """
    print("Generating correlation heatmap...")
    corr = df[columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        linewidths=0.5
    )
    plt.title('Correlation Heatmap of TLR Scores', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_radar_chart(df, institutes, year, output_dir):
    """
    Generates a radar chart to compare multiple institutions across
    TLR subcomponents for a specific year.
    
    Args:
        df (pd.DataFrame): The main dataframe.
        institutes (list): A list of institute names (e.g., ['VJTI', 'IIT Bombay']).
        year (int): The NIRF year to compare.
        output_dir (str): Path to save the plot.
    """
    print(f"Generating radar chart for {year}...")
    
    # Get the data for the specified institutes and year
    data = df[
        (df['institute_name'].isin(institutes)) & 
        (df['nirf_year'] == year)
    ]
    
    if data.empty:
        print(f"No data found for {institutes} in {year}. Skipping radar chart.")
        return
        
    data = data.set_index('institute_name')
    data = data[SUB_COLS] # Select only the subcomponent scores
    
    # Create the radar chart
    labels = data.columns
    num_vars = len(labels)
    
    # Set up the angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Helper function to plot one institution
    def add_to_radar(institute_name, color):
        values = data.loc[institute_name].tolist()
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, color=color, linewidth=2, label=institute_name)
        ax.fill(angles, values, color=color, alpha=0.25)

    # Plot each institution
    colors = plt.cm.get_cmap('Set1', len(institutes))
    for i, institute in enumerate(institutes):
        if institute in data.index:
            add_to_radar(institute, colors(i))
        else:
            print(f"Warning: No data for '{institute}' in {year}.")

    # Format the chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Set y-axis labels
    ax.set_rlabel_position(180 / num_vars)
    ax.set_yticks([10, 20, 30]) # Set gridlines based on score weights
    ax.set_yticklabels(["10", "20", "30"])
    ax.set_ylim(0, 30) # Max weight for FSR/FRU is 30 [cite: 22]
    
    plt.title(f'TLR Component Comparison ({year})', size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    output_path = os.path.join(output_dir, f'radar_chart_{year}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to run the visualization pipeline."""
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = load_data(DATA_PATH)
    if df is None:
        return
        
    # --- Generate Plots ---
    
    # 1. Score Distributions
    plot_score_distributions(df, SCORE_COLS, OUTPUT_DIR)
    
    # 2. Correlation Heatmap
    plot_correlation_heatmap(df, SCORE_COLS, OUTPUT_DIR)
    
    # 3. Radar Chart Comparison
    # Compare VJTI and IIT Bombay for the latest year
    plot_radar_chart(
        df, 
        institutes=['VJTI', 'IIT Bombay'], 
        year=2025, 
        output_dir=OUTPUT_DIR
    )
    
    print("\n✅ Visualization script finished.")


if __name__ == "__main__":
    main()