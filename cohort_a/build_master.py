#!/usr/bin/env python3
"""
Cohort A - master CSV builder

Combines the per-class weekly score CSVs in <data-dir>/csv/, attaches
Class / Level / Probation from <data-dir>/student_list.csv, and writes:

    CohortA_Master.csv    one row per student per week
    CohortA_Cutlines.csv  level-boundary cutlines (percentiles of overall averages)

Usage:
    python3 build_master.py                      # uses ./sample_data
    python3 build_master.py --data-dir path/to/raw --output-dir path/to/out
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / 'sample_data'

MASTER_FILE = 'CohortA_Master.csv'
CUTLINES_FILE = 'CohortA_Cutlines.csv'

NON_SCORE_COLS = {"Date", "Name", "Class", "Level", "Probation"}

# Level boundary -> percentile of the cohort's overall-average distribution.
# Lower level number = higher proficiency.
CUTLINE_PERCENTILES = {2: 85, 3: 45, 4: 5}


def load_student_list(data_dir):
    """Load student list and build name->level, name->probation, and name->class lookups.

    A Level of 'P' means the student is on probation; they are treated as
    Level 4 for statistics and flagged in the Probation column.
    """
    sl = pd.read_csv(Path(data_dir) / 'student_list.csv')
    level_lookup = {}
    probation_lookup = {}
    class_lookup = {}
    for _, row in sl.iterrows():
        key = f"{row['English Name']} {row['Korean Name']}"
        level = row['Level']
        class_lookup[key] = row['Class']
        if str(level).upper() == 'P':
            level_lookup[key] = 4
            probation_lookup[key] = True
        else:
            level_lookup[key] = int(level)
            probation_lookup[key] = False
    return level_lookup, probation_lookup, class_lookup


def generate_master_csv(data_dir, output_path):
    """Combine all class CSVs using the student list as the source of truth for names, levels, and classes."""
    level_lookup, probation_lookup, class_lookup = load_student_list(data_dir)
    student_names = set(level_lookup.keys())
    print(f"Loaded student list with {len(student_names)} students")

    csv_dir = Path(data_dir) / 'csv'
    csv_files = sorted(glob.glob(str(csv_dir / '*.csv')))
    all_data = []

    for file_path in csv_files:
        file_path = Path(file_path)
        df = pd.read_csv(file_path)
        all_data.append(df)
        print(f"Loaded {file_path.name}: {len(df)} records")

    if not all_data:
        print(f"No CSV files found in {csv_dir}")
        return None

    combined = pd.concat(all_data, ignore_index=True)

    # Keep only students on the roster (this also drops '*'-prefixed reference rows)
    in_list = combined['Name'].isin(student_names)
    skipped = combined[~in_list]['Name'].unique()
    if len(skipped) > 0:
        print(f"\nSkipped {len(skipped)} name(s) not in student list:")
        for name in skipped:
            print(f"  - {name}")
    combined = combined[in_list].copy()

    combined['Date'] = pd.to_datetime(combined['Date'], format='%m/%d/%y')

    combined['Class'] = combined['Name'].map(class_lookup)
    combined['Level'] = combined['Name'].map(level_lookup)
    combined['Probation'] = combined['Name'].map(probation_lookup).infer_objects(copy=False).fillna(False)

    cols = combined.columns.tolist()
    for c in ('Class', 'Level', 'Probation'):
        cols.remove(c)
    new_order = ['Date', 'Name', 'Class', 'Level', 'Probation'] + [c for c in cols if c not in ['Date', 'Name']]
    combined = combined[new_order]

    output_path = Path(output_path)
    combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")
    print(f"Total records: {len(combined)}")

    return combined


def _student_overall_avg(df, name):
    """Compute overall average across all score columns for a single student."""
    rows = df[df["Name"] == name]
    score_cols = [c for c in rows.columns if c not in NON_SCORE_COLS]
    vals = rows[score_cols].apply(pd.to_numeric, errors='coerce').values.flatten()
    vals = vals[~np.isnan(vals.astype(float))]
    if len(vals) > 0:
        return np.mean(vals)
    return None


def generate_cutlines(df, output_path):
    """Compute overall cutlines as plain percentiles of the per-student overall averages.

    Top 15% -> Lv2, middle 40% -> Lv3, bottom 40% -> Lv4, bottom 5% -> probation.
    Cutlines: Lv2 = 85th pctl, Lv3 = 45th pctl, Lv4 = 5th pctl.
    """
    overall_scores = []
    for name in df["Name"].unique():
        avg = _student_overall_avg(df, name)
        if avg is not None:
            overall_scores.append(avg)

    if not overall_scores:
        print("No overall scores found")
        return None

    arr = np.array(overall_scores)
    cutlines_df = pd.DataFrame([
        {"Level_Boundary": level, "Cutline": round(np.percentile(arr, pctl), 2)}
        for level, pctl in CUTLINE_PERCENTILES.items()
    ])
    output_path = Path(output_path)
    cutlines_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved cutlines: {output_path}")
    print(cutlines_df.to_string(index=False))
    return cutlines_df


def main():
    parser = argparse.ArgumentParser(description="Build the Cohort A master CSV and cutlines.")
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR,
                        help='folder containing student_list.csv and csv/<class>.csv (default: ./sample_data)')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR,
                        help='where to write the master and cutline CSVs (default: this folder)')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = generate_master_csv(args.data_dir, args.output_dir / MASTER_FILE)
    if combined is not None:
        generate_cutlines(combined, args.output_dir / CUTLINES_FILE)


if __name__ == '__main__':
    main()
