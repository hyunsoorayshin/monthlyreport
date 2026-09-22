#!/usr/bin/env python3
"""
Cohort B - master CSV builder

Combines the per-class weekly score CSVs in <data-dir>/csv/, attaches
Class / Level / Probation from <data-dir>/student_list.csv, splits the rows
at PERIOD_CUTOFF, and writes:

    CohortB_Master.csv        current period (rows after the cutoff)
    CohortB_Master_Prior.csv  prior period (rows on/before the cutoff);
                              the report uses it as a baseline overlay
    CohortB_Cutlines.csv      bootstrap cutline zones for the current period

Usage:
    python3 build_master.py                      # uses ./sample_data
    python3 build_master.py --data-dir path/to/raw --output-dir path/to/out --cutoff 2026-02-14
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / 'sample_data'

MASTER_FILE = 'CohortB_Master.csv'
PRIOR_MASTER_FILE = 'CohortB_Master_Prior.csv'
CUTLINES_FILE = 'CohortB_Cutlines.csv'

# Boundary between the prior assessment period and the current one.
PERIOD_CUTOFF = pd.Timestamp('2026-02-14')

NON_SCORE_COLS = {"Date", "Name", "Class", "Level", "Probation"}

# Level boundary -> target percentile of the overall-average distribution (center of the zone).
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


def load_combined(data_dir):
    """Read every class CSV, keep rostered students only, attach Class/Level/Probation."""
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
    return combined[new_order]


def generate_master_csvs(data_dir, output_dir, cutoff):
    """Write the current-period master (after cutoff) and the prior-period master (on/before cutoff).

    Returns the current-period DataFrame (used for cutlines), or None.
    """
    combined = load_combined(data_dir)
    if combined is None:
        return None

    current = combined[combined['Date'] > cutoff].copy()
    prior = combined[combined['Date'] <= cutoff].copy()
    print(f"\nSplit at {cutoff.date()}: {len(current)} current rows, {len(prior)} prior rows")

    master_path = Path(output_dir) / MASTER_FILE
    current.to_csv(master_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {master_path}")

    prior_path = Path(output_dir) / PRIOR_MASTER_FILE
    prior.to_csv(prior_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {prior_path}")

    return current


def _student_overall_avg(df, name):
    """Compute overall average across all score columns for a single student."""
    rows = df[df["Name"] == name]
    score_cols = [c for c in rows.columns if c not in NON_SCORE_COLS]
    vals = rows[score_cols].apply(pd.to_numeric, errors='coerce').values.flatten()
    vals = vals[~np.isnan(vals.astype(float))]
    if len(vals) > 0:
        return np.mean(vals)
    return None


def _bootstrap_percentile_zone(scores, pctl, n_bootstrap=2000, band=(40, 60),
                                max_half_width=2.0, seed=42):
    """Estimate a cutline zone for a target percentile via bootstrap resampling.

    Returns (low, center, high):
    - center: median of bootstrap percentile estimates (point estimate)
    - low/high: `band` percentiles of the bootstrap distribution, clipped so the
      zone is at most +/- `max_half_width` around the center

    The 40-60 band (rather than the 25-75 IQR) keeps the number of students who
    land inside the "borderline" zone small; `max_half_width` is a hard cap for
    sparse regions of the distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        estimates[i] = np.percentile(sample, pctl)
    low = float(np.percentile(estimates, band[0]))
    center = float(np.percentile(estimates, 50))
    high = float(np.percentile(estimates, band[1]))
    low = max(low, center - max_half_width)
    high = min(high, center + max_half_width)
    return low, center, high


def generate_cutlines(df, output_path):
    """Compute cutline zones (Low/Center/High) via bootstrap.

    Cutline is the point estimate the monthly report uses; a student whose
    average falls between Cutline_Low and Cutline_High is drawn as borderline.
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

    rows = []
    for level, pctl in CUTLINE_PERCENTILES.items():
        low, center, high = _bootstrap_percentile_zone(arr, pctl)
        rows.append({
            "Level_Boundary": level,
            "Cutline_Low": round(low, 2),
            "Cutline": round(center, 2),
            "Cutline_High": round(high, 2),
            "Zone_Width": round(high - low, 2),
        })

    cutlines_df = pd.DataFrame(rows)
    output_path = Path(output_path)
    cutlines_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved cutlines: {output_path}")
    print(cutlines_df.to_string(index=False))
    return cutlines_df


def main():
    parser = argparse.ArgumentParser(description="Build the Cohort B master CSVs and cutlines.")
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR,
                        help='folder containing student_list.csv and csv/<class>.csv (default: ./sample_data)')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR,
                        help='where to write the master and cutline CSVs (default: this folder)')
    parser.add_argument('--cutoff', type=pd.Timestamp, default=PERIOD_CUTOFF,
                        help=f'last date of the prior period, YYYY-MM-DD (default: {PERIOD_CUTOFF.date()})')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    current = generate_master_csvs(args.data_dir, args.output_dir, args.cutoff)
    if current is not None and len(current) > 0:
        generate_cutlines(current, args.output_dir / CUTLINES_FILE)
    else:
        print("No current-period rows; cutlines not generated")


if __name__ == '__main__':
    main()
