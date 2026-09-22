"""
Level-Up Advice Report Generator
---------------------------------
Radar chart comparing a student to the target level on a 0-5 scale, plus a
per-subject trend readout (least-squares slope and coefficient of variation).

Subjects use 100-point scale: 100=5, 95-99=4, 80-94=3, 60-79=2, 40-59=1, <40=0.
Speaking and Writing are excluded from the radar (different scoring system).

Reads the Cohort B master and cutline CSVs produced by cohort_b/build_master.py.

Usage:
    python3 levelup_report.py "StudentA 학생A"
    python3 levelup_report.py "StudentA 학생A" --target-level 2
    python3 levelup_report.py "StudentA 학생A" --data-dir path/to/folder/with/master
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# Korean font support
for _font in ["Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if _font in {f.name for f in mpl.font_manager.fontManager.ttflist}:
        mpl.rcParams["font.family"] = _font
        break
mpl.rcParams["axes.unicode_minus"] = False

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "cohort_b"
MASTER_FILE = "CohortB_Master.csv"
CUTLINES_FILE = "CohortB_Cutlines.csv"
META_COLS = ["Date", "Name", "Class", "Level", "Probation"]

SCIENCE_COLS = ["Science HW", "Science Test"]
SPEAKING_COLS = ["Debate", "Speaking", "Speaking Assessment"]
WRITING_COLS = [
    "Persuasive Writing", "Persuasive Rewriting",
    "Compare-Contrast Writing", "Compare-Contrast Rewriting",
    "Descriptive Writing", "Descriptive Rewriting",
]
STANDALONE_COLS = ["Vocabulary", "Listening", "Novel", "Article Comprehension", "Reading"]

CATEGORY_ORDER = [
    "Vocabulary", "Science", "Listening", "Novel",
    "Article Comprehension", "Reading",
]


# ── Scale conversion ────────────────────────────────────────────────────────
def score_to_5(val: float) -> float:
    """Convert a 0–100 subject score to 0–5 scale."""
    if val >= 100:
        return 5.0
    if val >= 95:
        return 4.0 + (val - 95) / 5        # 95→4.0, 99→4.8
    if val >= 80:
        return 3.0 + (val - 80) / 15       # 80→3.0, 94→3.93
    if val >= 60:
        return 2.0 + (val - 60) / 20       # 60→2.0, 79→2.95
    if val >= 40:
        return 1.0 + (val - 40) / 20       # 40→1.0, 59→1.95
    return max(0.0, val / 40)               # 0→0, 39→0.975


# ── Data helpers ────────────────────────────────────────────────────────────
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    score_cols = [c for c in df.columns if c not in META_COLS]
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")
    return df


def build_scores(df: pd.DataFrame, name: str | None = None,
                 level: int | None = None) -> tuple[dict, int]:
    """Build combined raw scores. Returns (scores_dict, source_level)."""
    if name:
        rows = df[df["Name"].str.contains(name, case=False, na=False)].copy()
        src_level = int(rows["Level"].dropna().iloc[0])
    elif level is not None:
        rows = df[df["Level"] == level].copy()
        src_level = level
    else:
        return {}, 0

    result = {}

    for col in STANDALONE_COLS:
        if col in rows.columns:
            vals = rows[col].dropna()
            if not vals.empty:
                result[col] = np.mean(vals.values)

    sci_cols = [c for c in SCIENCE_COLS if c in rows.columns]
    if sci_cols:
        sci_vals = rows[sci_cols].values.flatten()
        sci_vals = sci_vals[~np.isnan(sci_vals)]
        if len(sci_vals) > 0:
            result["Science"] = np.mean(sci_vals)

    spk_cols = [c for c in SPEAKING_COLS if c in rows.columns]
    if spk_cols:
        spk_vals = rows[spk_cols].values.flatten()
        spk_vals = spk_vals[~np.isnan(spk_vals)]
        if len(spk_vals) > 0:
            result["Speaking"] = np.mean(spk_vals)

    wrt_cols = [c for c in WRITING_COLS if c in rows.columns]
    if wrt_cols:
        wrt_vals = rows[wrt_cols].values.flatten()
        wrt_vals = wrt_vals[~np.isnan(wrt_vals)]
        if len(wrt_vals) > 0:
            result["Writing"] = np.mean(wrt_vals)

    return result, src_level


def build_subject_analysis(df: pd.DataFrame, name: str) -> dict:
    """
    Compute per-category trend (slope) and stdev from student's time-series.
    Returns {category: {"trend": slope_per_week, "stdev": stdev, "signal": str}}.
    """
    rows = df[df["Name"].str.contains(name, case=False, na=False)].copy()
    rows = rows.sort_values("Date")
    if len(rows) < 2:
        return {}

    # Convert dates to numeric (weeks from first date) for regression
    dates = pd.to_datetime(rows["Date"])
    x = (dates - dates.iloc[0]).dt.days.values.astype(float)

    analysis = {}

    def _analyze(cat, vals_series):
        vals = vals_series.values.astype(float)
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            return None
        xv, yv = x[mask], vals[mask]
        # Linear regression slope (per week)
        slope = np.polyfit(xv, yv, 1)[0] * 7
        stdev = np.std(yv)
        avg = np.mean(yv)
        # Coefficient of variation (normalized stdev)
        cv = stdev / avg if avg > 0 else 0

        # Classify signal
        if slope > 0.5:
            if cv < 0.08:
                signal = "Rising"       # steady improvement
            else:
                signal = "Potential"    # improving but volatile
        elif slope < -0.5:
            signal = "Declining"        # getting worse
        else:
            if cv < 0.08:
                signal = "Plateau"      # flat and stable
            else:
                signal = "Volatile"     # flat but inconsistent
        return {"trend": slope, "stdev": stdev, "signal": signal}

    for col in STANDALONE_COLS:
        if col in rows.columns:
            result = _analyze(col, rows[col])
            if result:
                analysis[col] = result

    sci_cols = [c for c in SCIENCE_COLS if c in rows.columns]
    if sci_cols:
        sci_mean = rows[sci_cols].mean(axis=1)
        result = _analyze("Science", sci_mean)
        if result:
            analysis["Science"] = result

    return analysis


def to_scale(raw_scores: dict) -> dict:
    """Convert raw scores dict to 0–5 scale."""
    return {cat: score_to_5(val) for cat, val in raw_scores.items()}


def level_scaled_stats(df: pd.DataFrame, level: int):
    """Per-student scaled averages at target level, then group stats."""
    rows = df[df["Level"] == level]
    names = rows["Name"].unique()

    per_student = {}
    for n in names:
        raw, src_lv = build_scores(df, name=n)
        if raw:
            per_student[n] = to_scale(raw)

    combined = pd.DataFrame(per_student).T
    return pd.DataFrame({
        "avg": combined.mean(),
        "min": combined.min(),
        "max": combined.max(),
        "q1": combined.quantile(0.25),
        "q3": combined.quantile(0.75),
    })


def load_cutlines_csv(cutlines_csv: Path) -> pd.DataFrame | None:
    """Read the cutline CSV if present. Returns None when missing."""
    if cutlines_csv.exists():
        return pd.read_csv(cutlines_csv, encoding="utf-8-sig")
    return None


def compute_cutlines(df: pd.DataFrame, target_level: int,
                     cdf: pd.DataFrame | None = None) -> dict:
    """
    Compute per-category cutlines between target_level and target_level+1,
    converted to 0-5 scale.

    If the cutline CSV carries a per-category breakdown (a `Category` column),
    those values are used. The cutline CSV written by build_master.py has only
    an overall cutline per level, so per-category cutlines are computed from
    the master data instead.
    """
    CUTLINE_POSITION = 0.5

    if cdf is not None and "Category" in cdf.columns:
        rows = cdf[cdf["Level_Boundary"] == target_level]
        if not rows.empty:
            scaled = {}
            for _, r in rows.iterrows():
                cat = r["Category"]
                if cat == "Overall" or cat not in CATEGORY_ORDER:
                    continue
                scaled[cat] = score_to_5(r["Cutline"])
            return scaled

    # Compute from master data using 1.5-sigma cleaned averages
    upper_lv = target_level
    lower_lv = target_level + 1

    upper_names = df[df["Level"] == upper_lv]["Name"].unique()
    lower_names = df[df["Level"] == lower_lv]["Name"].unique()

    upper_per = {}
    for n in upper_names:
        raw, _ = build_scores(df, name=n)
        if raw:
            upper_per[n] = raw
    lower_per = {}
    for n in lower_names:
        raw, _ = build_scores(df, name=n)
        if raw:
            lower_per[n] = raw

    def _clean_avg(values):
        arr = np.array(values)
        if len(arr) < 3:
            return arr.mean()
        m, s = arr.mean(), arr.std()
        c = arr[(arr >= m - 1.5 * s) & (arr <= m + 1.5 * s)]
        return c.mean() if len(c) > 0 else arr.mean()

    all_cats = set()
    for d in list(upper_per.values()) + list(lower_per.values()):
        all_cats.update(d.keys())

    scaled = {}
    for cat in all_cats:
        if cat not in CATEGORY_ORDER:
            continue
        u_vals = [d[cat] for d in upper_per.values() if cat in d]
        l_vals = [d[cat] for d in lower_per.values() if cat in d]
        if u_vals and l_vals:
            u_avg = _clean_avg(u_vals)
            l_avg = _clean_avg(l_vals)
            cutline_raw = l_avg + (u_avg - l_avg) * CUTLINE_POSITION
            scaled[cat] = score_to_5(cutline_raw)
    return scaled


# ── Signal descriptions ─────────────────────────────────────────────────────
SIGNAL_DESC = {
    "Rising":    "Scores steadily improving.\nGood momentum — keep pushing.",
    "Potential": "Scores improving but fluctuate.\nNeeds more consistency.",
    "Volatile":  "Scores vary a lot.\nStabilize fundamentals first.",
    "Plateau":   "Scores consistent but flat.\nMay need a new approach.",
    "Declining": "Scores dropping recently.\nNeeds immediate attention.",
}
SIGNAL_COLOR = {
    "Rising": "#2E8B57", "Potential": "#D4A017", "Volatile": "#CC6600",
    "Plateau": "#888888", "Declining": "#CC3333",
}


# ── Radar chart ─────────────────────────────────────────────────────────────
def make_radar(
    student_name: str,
    student_scaled: dict,
    target: pd.DataFrame,
    target_level: int,
    current_level: int,
    output_path: Path,
    cutline_scaled: dict | None = None,
    overall_info: dict | None = None,
    subject_analysis: dict | None = None,
):
    cats = [c for c in CATEGORY_ORDER
            if c in student_scaled and c in target.index]
    if not cats:
        sys.exit("No overlapping score categories found.")

    n = len(cats)
    short = student_name.split()[0] if " " in student_name else student_name

    s_vals = [student_scaled[c] for c in cats]
    t_vals = [target.loc[c, "avg"] for c in cats]
    q1_vals = [target.loc[c, "q1"] for c in cats]
    q3_vals = [target.loc[c, "q3"] for c in cats]
    c_vals = [cutline_scaled.get(c, 0) for c in cats] if cutline_scaled else None

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    s_vals += s_vals[:1]
    t_vals += t_vals[:1]
    q1_vals += q1_vals[:1]
    q3_vals += q3_vals[:1]
    if c_vals is not None:
        c_vals += c_vals[:1]

    # Two-panel layout: radar (left) + analysis (right)
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(121, polar=True)

    # Grid at each integer level
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="#888")

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=9)

    # Target level IQR band
    ax.fill_between(angles, q1_vals, q3_vals, alpha=0.10, color="#E87C5D",
                    label=f"Lv{target_level} IQR")

    # Target level avg line
    ax.plot(angles, t_vals, color="#E87C5D", linewidth=1.5,
            label=f"Lv{target_level} Avg", linestyle="--")

    # Cutline polygon
    if c_vals is not None:
        ax.plot(angles, c_vals, color="#888888", linewidth=1.5,
                label="Cutline", linestyle=":", dash_capstyle="round")

    # Student fill
    ax.fill(angles, s_vals, alpha=0.2, color="#4A90D9")
    ax.plot(angles, s_vals, color="#4A90D9", linewidth=2.2, label=short)

    # Score labels on each vertex
    for i in range(n):
        angle = angles[i]
        s = s_vals[i]
        ax.text(angle, s + 0.35, f"{s:.1f}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="#4A90D9")

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=8)
    ax.set_title(
        f"Level-Up Report:  {student_name}\n(Lv{current_level} → Lv{target_level})",
        fontsize=11, fontweight="bold", pad=20,
    )

    # ── Right panel: analysis text ──────────────────────────────────────────
    ax2 = fig.add_subplot(122)
    ax2.axis("off")

    y = 0.95  # start from top

    # Overall averages section
    if overall_info:
        ax2.text(0.0, y, "Overall Average", fontsize=11, fontweight="bold",
                 transform=ax2.transAxes, va="top")
        y -= 0.06
        ax2.text(0.02, y, f"{short}:  {overall_info['student']:.1f}    "
                 f"Lv{target_level} Avg:  {overall_info['target']:.1f}    "
                 f"Cutline:  {overall_info.get('cutline') or 0:.1f}",
                 fontsize=9, fontfamily="monospace",
                 transform=ax2.transAxes, va="top")
        y -= 0.08

    # Subject-by-subject analysis
    ax2.text(0.0, y, "Subject Analysis", fontsize=11, fontweight="bold",
             transform=ax2.transAxes, va="top")
    y -= 0.05

    # Sort subjects: biggest gap first (most improvement needed)
    ranked = []
    for c in cats[:n]:
        gap = target.loc[c, "avg"] - student_scaled[c]
        ranked.append((c, gap))
    ranked.sort(key=lambda x: -x[1])

    for rank, (c, gap) in enumerate(ranked, 1):
        s = student_scaled[c]
        t = target.loc[c, "avg"]
        cut_val = cutline_scaled.get(c, 0) if cutline_scaled else 0

        # Subject header with gap
        above_cut = s >= cut_val
        gap_label = f"+{-gap:.1f} above" if gap < 0 else f"-{gap:.1f} below"
        status = "Above cutline" if above_cut else "Below cutline"

        ax2.text(0.02, y, f"{rank}. {c}", fontsize=9.5, fontweight="bold",
                 transform=ax2.transAxes, va="top")
        ax2.text(0.55, y, f"{gap_label} target  |  {status}",
                 fontsize=8, color="#555555",
                 transform=ax2.transAxes, va="top")
        y -= 0.04

        # Signal analysis
        if subject_analysis and c in subject_analysis:
            a = subject_analysis[c]
            sig = a["signal"]
            color = SIGNAL_COLOR.get(sig, "#333")
            desc = SIGNAL_DESC.get(sig, "")
            trend_dir = "+" if a["trend"] > 0 else ""
            ax2.text(0.04, y,
                     f"{sig}  (trend: {trend_dir}{a['trend']:.1f}/wk,  stdev: {a['stdev']:.1f})",
                     fontsize=8.5, color=color, fontweight="bold",
                     transform=ax2.transAxes, va="top")
            y -= 0.035
            for desc_line in desc.split("\n"):
                ax2.text(0.04, y, desc_line,
                         fontsize=8, color="#555555",
                         transform=ax2.transAxes, va="top")
                y -= 0.03
        else:
            ax2.text(0.04, y, "Not enough data for trend analysis",
                     fontsize=8, color="#999999", fontstyle="italic",
                     transform=ax2.transAxes, va="top")
            y -= 0.03

        y -= 0.02  # spacing between subjects

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Chart saved → {output_path}")
    plt.close(fig)


# ── Summary text ────────────────────────────────────────────────────────────
def print_summary(student_name: str, student_scaled: dict,
                  target: pd.DataFrame, target_level: int,
                  cutline_scaled: dict | None = None,
                  subject_analysis: dict | None = None):
    short = student_name.split()[0] if " " in student_name else student_name
    cats = [c for c in CATEGORY_ORDER
            if c in student_scaled and c in target.index]
    if not cats:
        return

    print(f"\n{'='*78}")
    print(f"  LEVEL-UP REPORT: {student_name} → Level {target_level}")
    print(f"  Scale: 0–5  (5 = best)")
    print(f"{'='*78}")

    has_cut = cutline_scaled is not None
    has_analysis = subject_analysis is not None
    header_cut = "   Cut" if has_cut else ""
    header_sig = "  Signal" if has_analysis else ""
    print(f"  {'Category':22s}  {short:>5s}  {'Lv'+str(target_level):>5s}{header_cut}   Diff  {header_sig}")
    print(f"  {'-'*70}")

    for c in cats:
        s = student_scaled[c]
        t = target.loc[c, "avg"]
        diff = s - t
        sign = "+" if diff >= 0 else ""
        marker = "  " if abs(diff) < 0.3 else (" ^" if diff > 0 else " !")
        cut_str = f"  {cutline_scaled.get(c, 0):4.1f}" if has_cut else ""
        if has_analysis and c in subject_analysis:
            a = subject_analysis[c]
            sig_str = f"  {a['signal']}"
        else:
            sig_str = ""
        print(f"  {c:22s}  {s:5.1f}  {t:5.1f}{cut_str}   ({sign}{diff:.1f}){marker}{sig_str}")

    print(f"\n  ^ = above target   ! = needs work")
    if has_analysis:
        print(f"  Signals: Rising=improving, Potential=improving but volatile,")
        print(f"           Volatile=inconsistent, Plateau=stable/maxed, Declining=dropping")
    print(f"{'='*78}\n")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Level-Up Advice Report")
    parser.add_argument("student", help='Student name as it appears in the master CSV (e.g. "StudentA 학생A")')
    parser.add_argument("--target-level", type=int, default=None,
                        help="Target level (default: current level - 1)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"folder containing {MASTER_FILE} and {CUTLINES_FILE} (default: ./cohort_b)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: <data-dir>/reports/levelup_<name>_to_Lv<n>.png)")
    args = parser.parse_args()

    master_csv = args.data_dir / MASTER_FILE
    if not master_csv.exists():
        sys.exit(f"{master_csv} not found. Run cohort_b/build_master.py first.")
    df = load_data(master_csv)
    cdf = load_cutlines_csv(args.data_dir / CUTLINES_FILE)

    mask = df["Name"].str.contains(args.student, case=False, na=False)
    if not mask.any():
        sys.exit(f"Student '{args.student}' not found.")
    full_name = df.loc[mask, "Name"].iloc[0]
    current_level = int(df.loc[mask, "Level"].dropna().iloc[0])

    target_level = args.target_level if args.target_level else current_level - 1
    if target_level < 1:
        sys.exit("Student is already at the highest level.")

    # Build and scale
    raw, src_lv = build_scores(df, name=args.student)
    s_scaled = to_scale(raw)
    t_stats = level_scaled_stats(df, target_level)

    # Compute cutlines
    cutline_scaled = compute_cutlines(df, target_level, cdf)

    # Compute raw overall averages for the summary box
    student_overall = np.mean(list(raw.values())) if raw else 0
    # Target level: per-student raw averages, then mean
    target_names = df[df["Level"] == target_level]["Name"].unique()
    target_overalls = []
    for n in target_names:
        r, _ = build_scores(df, name=n)
        if r:
            target_overalls.append(np.mean(list(r.values())))
    target_overall = np.mean(target_overalls) if target_overalls else 0
    # Overall cutline for the target level from the cutline CSV, if present
    cutline_overall = None
    if cdf is not None:
        row = cdf[cdf["Level_Boundary"] == target_level]
        if "Category" in cdf.columns:
            row = row[row["Category"] == "Overall"]
        if not row.empty:
            cutline_overall = row.iloc[0]["Cutline"]

    overall_info = {
        "student": student_overall,
        "target": target_overall,
        "cutline": cutline_overall,
    }

    # Output path
    if args.output:
        out = Path(args.output)
    else:
        safe_name = full_name.replace(" ", "_")
        out = args.data_dir / "reports" / f"levelup_{safe_name}_to_Lv{target_level}.png"
        out.parent.mkdir(parents=True, exist_ok=True)

    # Subject analysis (trend + stdev)
    subject_analysis = build_subject_analysis(df, args.student)

    make_radar(full_name, s_scaled, t_stats, target_level, current_level, out,
               cutline_scaled=cutline_scaled, overall_info=overall_info,
               subject_analysis=subject_analysis)
    print_summary(full_name, s_scaled, t_stats, target_level,
                  cutline_scaled=cutline_scaled, subject_analysis=subject_analysis)


if __name__ == "__main__":
    main()
