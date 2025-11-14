"""
Generates the initial synthetic dataset for the NIRF TLR project.

This script can be run directly to create the raw data file.

Run from project root:
    python src/utils.py
"""
import os
import pandas as pd
import numpy as np
import argparse

def generate_tlr_dataset(num_institutes=40, start_year=2020, end_year=2025, output_path=None):
    """
    Generates a realistic large synthetic dataset for NIRF TLR modeling.
    Each year × institute = 1 row.
    Example: 40 institutes * 6 years = 240 rows.

    If output_path is None the file will be written to:
        <project_root>/data/processed/tlr_model_input.csv
    """
    np.random.seed(42)

    # Sample institute base names
    base_institutes = [
        "VJTI", "IIT Bombay", "IIT Delhi", "IIT Madras", "NIT Trichy",
        "NIT Surathkal", "BITS Pilani", "COEP", "VIT", "Manipal Institute",
        "MIT-WPU", "KJ Somaiya", "SPIT", "PCCOE", "PICT",
        "IIIT Hyderabad", "IIT Kanpur", "IIT Roorkee", "NIT Warangal",
        "IIT Guwahati", "IIT Indore", "NIT Calicut", "IIIT Delhi", "SRM IST"
    ]

    # Extend base list if fewer than required
    while len(base_institutes) < num_institutes:
        base_institutes.append(f"College_{len(base_institutes)+1}")

    data = []

    for year in range(start_year, end_year + 1):
        for inst in base_institutes:
            # --- Generate subcomponent scores ---
            ss = np.random.uniform(10, 20)             # Student Strength (out of 20)
            fsr = np.random.uniform(15, 30)            # Faculty-Student Ratio (out of 30)
            fqe = np.random.uniform(10, 20)            # Faculty Qualification (out of 20)
            fru = np.random.uniform(10, 30)            # Financial Resources (out of 30)
            oe = np.random.uniform(3, 10)              # Online Education (out of 10)
            mir = np.random.uniform(2, 5)              # IKS/Regional Languages (out of 5)

            # --- Weighted TLR calculation ---
            tlr = (ss * 0.2) + (fsr * 0.3) + (fqe * 0.2) + (fru * 0.3)
            tlr += 0.5 * oe + 0.3 * mir                # minor influence of OE & MIR
            tlr = np.clip(tlr + np.random.normal(0, 2), 0, 100)  # add small noise

            # --- Generate related numeric features ---
            student_count = np.random.randint(2000, 12000)
            faculty_count = int(student_count / np.random.uniform(14, 25))
            phd_faculty_count = int(faculty_count * np.random.uniform(0.4, 0.9))
            budget_per_student = np.random.uniform(1.5, 6.5) * 1e5  # ₹
            library_expense = np.random.uniform(10, 80) * 1e5
            lab_expense = np.random.uniform(20, 150) * 1e5
            online_courses = np.random.randint(5, 100)
            iks_courses = np.random.randint(0, 20)
            nirf_rank = np.random.randint(1, 250)

            data.append([
                inst, year, nirf_rank, tlr, ss, fsr, fqe, fru,
                oe, mir, faculty_count, phd_faculty_count, student_count,
                budget_per_student, library_expense, lab_expense,
                online_courses, iks_courses
            ])

    columns = [
        "institute_name", "nirf_year", "nirf_rank", "tlr_score",
        "ss_score", "fsr_score", "fqe_score", "fru_score",
        "oe_score", "mir_score", "faculty_count", "phd_faculty_count",
        "student_count", "budget_per_student", "library_expense",
        "lab_expense", "online_courses", "iks_courses"
    ]

    df = pd.DataFrame(data, columns=columns)

    # Determine default output location relative to project root
    if output_path is None:
        # project root is parent of src/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "tlr_model_input.csv")
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Generated dataset with {len(df)} rows at: {os.path.abspath(output_path)}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic TLR dataset")
    parser.add_argument("--num-institutes", type=int, default=40, help="Number of institutes to simulate")
    parser.add_argument("--start-year", type=int, default=2020, help="Start year (inclusive)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (inclusive)")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    df = generate_tlr_dataset(
        num_institutes=args.num_institutes,
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=args.output
    )
    print(df.head())