#!/usr/bin/env python3
"""
Cohort B - monthly report generator

Reads CohortB_Master.csv (current period), CohortB_Master_Prior.csv (prior
period, optional baseline) and CohortB_Cutlines.csv (from build_master.py),
then generates one PNG report per student with:
- Position within level (box plot per level, cutline zones)
- Radar of five skill categories vs the level average, with the prior period overlaid
- Weekly total-average trend with a fitted trend line and dip annotations

Also writes CohortB_Monthly_Summary.csv and CohortB_Risk_List.csv.

Usage:
    python3 monthly_report.py                 # reads/writes in this folder
    python3 monthly_report.py --data-dir path/to/folder/with/master
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent

MASTER_FILE = 'CohortB_Master.csv'
PRIOR_MASTER_FILE = 'CohortB_Master_Prior.csv'
CUTLINES_FILE = 'CohortB_Cutlines.csv'
SUMMARY_FILE = 'CohortB_Monthly_Summary.csv'
RISK_LIST_FILE = 'CohortB_Risk_List.csv'
REPORTS_SUBDIR = 'reports'

# Apple-inspired typography + neutral chart frames
plt.rcParams['font.family'] = ['Apple SD Gothic Neo', 'AppleGothic', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#D2D2D7'
plt.rcParams['axes.labelcolor'] = '#6E6E73'
plt.rcParams['axes.titlecolor'] = '#1D1D1F'
plt.rcParams['xtick.color'] = '#6E6E73'
plt.rcParams['ytick.color'] = '#6E6E73'
plt.rcParams['grid.color'] = '#E5E5EA'
plt.rcParams['grid.linewidth'] = 0.8

# Apple-style palette (used throughout the report)
PALETTE = {
    'bg': '#F5F5F7',
    'surface': '#FFFFFF',
    'border': '#D2D2D7',
    'text_primary': '#1D1D1F',
    'text_secondary': '#6E6E73',
    'text_tertiary': '#86868B',
    'blue': '#007AFF',
    'green': '#34C759',
    'red': '#FF3B30',
    'orange': '#FF9500',
    'gray': '#8E8E93',
    'red_soft': '#FFF1F0',
}

# Score columns to include in calculations
SCORE_COLUMNS = [
    'Vocabulary', 'Article HW', 'Debate', 'Persuasive Writing', 'Persuasive Rewriting',
    'Compare-Contrast Writing', 'Compare-Contrast Rewriting', 'Descriptive Writing', 'Descriptive Rewriting',
    'Science HW', 'Science Test', 'Science Presentation', 'Reading', 'Listening', 'Speaking',
    'Novel', 'Speaking Assessment', 'Article Comprehension', 'Reading HW'
]

# Writing columns to combine into single "Writing" score
WRITING_COLUMNS = [
    'Persuasive Writing', 'Persuasive Rewriting', 'Compare-Contrast Writing', 'Compare-Contrast Rewriting',
    'Descriptive Writing', 'Descriptive Rewriting'
]

# Science columns to combine into single "Science" score
SCIENCE_COLUMNS = ['Science HW', 'Science Test']

# Columns to track individually (excluding combined categories)
INDIVIDUAL_COLUMNS = [col for col in SCORE_COLUMNS if col not in WRITING_COLUMNS and col not in SCIENCE_COLUMNS]

# Radar chart categories (grouping used by the report card)
RADAR_CATEGORIES = {
    'Vocabulary': ['Vocabulary'],
    'Reading': ['Science HW', 'Science Test', 'Article HW', 'Article Comprehension', 'Reading HW', 'Novel'],
    'Listening': ['Listening'],
    'Writing': ['Persuasive Writing', 'Persuasive Rewriting', 'Compare-Contrast Writing', 'Compare-Contrast Rewriting',
                'Descriptive Writing', 'Descriptive Rewriting'],
    'Speaking': ['Science Presentation', 'Speaking', 'Debate', 'Speaking Assessment'],
}

RADAR_LABELS_KR = {
    'Vocabulary': '어휘',
    'Reading': '읽기',
    'Listening': '듣기',
    'Writing': '쓰기',
    'Speaking': '말하기',
}

# Probation students are read from the Probation column in the master CSV (set via student_list)
PROBATION_STUDENTS = set()  # populated at runtime by load_master_data()


def compute_radar_scores(data_df):
    """Average score per radar category from a slice of master data (any rows)."""
    cat_scores = {}
    for category, cols in RADAR_CATEGORIES.items():
        vals = []
        for col in cols:
            if col in data_df.columns:
                vals.extend(data_df[col].dropna().tolist())
        cat_scores[category] = float(np.mean(vals)) if vals else 0.0
    return cat_scores


def load_prior_period_radar(data_dir):
    """Load per-student radar scores from the prior-period master (multiple rows per student, averaged).

    Keys are master `Name` format: "{English Name} {Korean Name}".
    Students with no usable data are omitted (so caller can detect via .get()).
    Returns {} when the prior master does not exist.
    """
    prior = {}
    prior_path = Path(data_dir) / PRIOR_MASTER_FILE
    if prior_path.exists():
        df = pd.read_csv(prior_path)
        for name, group in df.groupby('Name'):
            cat_scores = compute_radar_scores(group)
            if any(v > 0 for v in cat_scores.values()):
                prior[name] = cat_scores
    return prior


def draw_radar(ax, student_radar, level_radar, student_name, level, prior_radar=None):
    """Draw a polar radar: current period, optional prior period, and the level average."""
    categories = list(RADAR_CATEGORIES.keys())
    labels = [RADAR_LABELS_KR[c] for c in categories]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    student_vals = [student_radar.get(c, 0) for c in categories]
    student_vals += student_vals[:1]
    level_vals = [level_radar.get(c, 0) for c in categories]
    level_vals += level_vals[:1]

    # Level average — soft gray dashed (background reference)
    ax.plot(angles, level_vals, color=PALETTE['gray'], linewidth=1.5, linestyle='--',
            label=f'Lv{level} 평균')
    ax.fill(angles, level_vals, color=PALETTE['gray'], alpha=0.06)

    # Prior period — orange dotted (growth reference)
    if prior_radar:
        prior_vals = [prior_radar.get(c, 0) for c in categories]
        prior_vals += prior_vals[:1]
        ax.plot(angles, prior_vals, color=PALETTE['orange'], linewidth=1.8, linestyle=':',
                label='1차 레벨링')

    # Current period — solid blue in front
    ax.plot(angles, student_vals, color=PALETTE['blue'], linewidth=2.4, label='2차 (현재)')
    ax.fill(angles, student_vals, color=PALETTE['blue'], alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11.5, color=PALETTE['text_primary'], fontweight='semibold')
    ax.tick_params(axis='x', pad=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', ''], fontsize=8, color=PALETTE['text_tertiary'])
    ax.grid(True, color=PALETTE['border'], linewidth=0.6, alpha=0.7)
    ax.spines['polar'].set_color(PALETTE['border'])
    ax.spines['polar'].set_linewidth(0.6)
    ax.set_facecolor(PALETTE['surface'])
    ax.set_title('영역별 점수', fontsize=14, fontweight='semibold',
                 color=PALETTE['text_primary'], pad=22)
    # Legend below the polar plot
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3,
                       fontsize=9, frameon=True, handlelength=1.6, columnspacing=1.2,
                       borderpad=0.6)
    legend.get_frame().set_facecolor(PALETTE['surface'])
    legend.get_frame().set_edgecolor(PALETTE['border'])
    legend.get_frame().set_linewidth(0.8)

    if prior_radar is None:
        ax.text(0.5, -0.20, '1차 레벨링 데이터 없음', transform=ax.transAxes,
                ha='center', va='top', fontsize=10, color=PALETTE['text_tertiary'], style='italic')


def load_prior_master(data_dir):
    """Load the prior-period master CSV, or an empty DataFrame if it does not exist."""
    path = Path(data_dir) / PRIOR_MASTER_FILE
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_prior_total_average(student_name, df_prior):
    """Return the student's prior-period total average (mean of all numeric score cells)
    as a single baseline number, or None if the prior master has no rows for them.
    """
    if df_prior.empty:
        return None
    sdf = df_prior[df_prior['Name'] == student_name]
    if len(sdf) == 0:
        return None
    score_cols = [c for c in df_prior.columns if c in SCORE_COLUMNS]
    vals = sdf[score_cols].apply(pd.to_numeric, errors='coerce').values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) > 0:
        return float(vals.mean())
    return None


def compute_weekly_trend(student_name, df_current, prior_total=None,
                         dip_threshold=3.0, subj_drop_threshold=3.0, max_culprits=2):
    """Build the weekly trend series for one student.

    First point (if available) = prior-period total avg (single baseline value).
    Remaining points = per-week averages computed from `df_current` only.

    For weeks where the average drops by >= `dip_threshold` from the previous point,
    annotate the top `max_culprits` subjects scored >= `subj_drop_threshold` below
    the student's typical level across the current period.

    Returns (labels, values, slope, direction, dip_annotations):
    - labels: x-axis labels: '1차' (prior baseline) followed by 'mm/dd' for each week
    - values: y-values aligned with labels
    - slope: least-squares slope over the full series (numpy polyfit, degree 1)
    - direction: 'up' (>0.5), 'down' (<-0.5), or 'neutral'
    - dip_annotations: {index: 'Subject1, Subject2'} for dipping points
    """
    points = []  # list of (label, value, subjects_dict_or_None)
    if prior_total is not None:
        points.append(('1차', float(prior_total), None))

    if not df_current.empty:
        sdf = df_current[df_current['Name'] == student_name].copy()
        if len(sdf) > 0:
            score_cols = [c for c in sdf.columns if c in SCORE_COLUMNS]
            sdf['_week'] = sdf['Date'].dt.to_period('W-SUN').dt.start_time

            weekly_records = []
            for week_start, week_df in sdf.groupby('_week'):
                numeric = week_df[score_cols].apply(pd.to_numeric, errors='coerce')
                row_avgs = numeric.mean(axis=1).dropna()
                if len(row_avgs) == 0:
                    continue
                week_avg = float(row_avgs.mean())

                subj_scores = {}
                for col in score_cols:
                    vals = pd.to_numeric(week_df[col], errors='coerce').dropna()
                    if len(vals) > 0:
                        subj_scores[col] = float(vals.mean())

                weekly_records.append((week_start, week_avg, subj_scores))

            weekly_records.sort(key=lambda r: r[0])
            for week_start, week_avg, subj_scores in weekly_records:
                points.append((week_start.strftime('%m/%d'), week_avg, subj_scores))

    if not points:
        return [], [], 0.0, 'neutral', {}

    # Typical per-subject score across the current-period weeks (for dip-culprit attribution)
    typicals = {}
    for _, _, subj_scores in points:
        if not subj_scores:
            continue
        for subj, score in subj_scores.items():
            typicals.setdefault(subj, []).append(score)
    typicals = {s: float(np.mean(v)) for s, v in typicals.items() if v}

    # Detect dips and identify culprit subjects
    dip_annotations = {}
    for i in range(1, len(points)):
        drop = points[i - 1][1] - points[i][1]
        if drop < dip_threshold:
            continue
        subj_scores = points[i][2]
        if not subj_scores:
            continue
        culprits = []
        for subj, score in subj_scores.items():
            typical = typicals.get(subj)
            if typical is None:
                continue
            diff = score - typical
            if diff < -subj_drop_threshold:
                culprits.append((subj, score, typical, diff))
        culprits.sort(key=lambda c: c[3])  # most-negative diff first
        culprits = culprits[:max_culprits]
        if culprits:
            dip_annotations[i] = ', '.join(c[0] for c in culprits)

    labels = [p[0] for p in points]
    values = [p[1] for p in points]
    if len(values) >= 2:
        slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
    else:
        slope = 0.0
    if slope > 0.5:
        direction = 'up'
    elif slope < -0.5:
        direction = 'down'
    else:
        direction = 'neutral'
    return labels, values, slope, direction, dip_annotations


def draw_weekly_trend(ax, labels, values, slope, direction, dip_annotations=None):
    """Plot the trend series with categorical x-axis and dip-culprit annotations.

    labels[0] is expected to be '1차' (prior-period baseline) when available; the
    rest are weekly date strings. `dip_annotations` is {idx: 'Subject1, Subject2'}.
    """
    if not labels:
        ax.text(0.5, 0.5, '주별 데이터 없음', ha='center', va='center',
                fontsize=12, color='#999', style='italic', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    color_map = {'up': PALETTE['green'], 'down': PALETTE['red'], 'neutral': PALETTE['blue']}
    badge_text = {
        'up': f'↑ 상승 추세 ({slope:+.2f}점/주)',
        'down': f'↓ 하락 추세 ({slope:+.2f}점/주)',
        'neutral': f'→ 유지 ({slope:+.2f}점/주)',
    }
    color = color_map[direction]
    dip_annotations = dip_annotations or {}

    x_idx = list(range(len(labels)))

    # Data line + markers (baseline point highlighted differently)
    ax.plot(x_idx, values, linewidth=2.2, color=color, label='주별 평균', zorder=3)
    for i, (x, y) in enumerate(zip(x_idx, values)):
        if i == 0 and labels[0] == '1차':
            ax.scatter([x], [y], s=80, color=PALETTE['orange'], edgecolors=PALETTE['surface'],
                       linewidths=1.5, zorder=4)
        elif i in dip_annotations:
            ax.scatter([x], [y], s=95, color=PALETTE['red'], edgecolors=PALETTE['surface'],
                       linewidths=1.5, zorder=5, marker='v')
        else:
            ax.scatter([x], [y], s=55, color=color, edgecolors=PALETTE['surface'],
                       linewidths=1.0, zorder=4)

    # Trend line
    if len(values) >= 2:
        m, b = np.polyfit(x_idx, values, 1)
        ax.plot(x_idx, [m * i + b for i in x_idx], linestyle='--', linewidth=1.4,
                color=color, alpha=0.5, label='추세선')

    # Prior/current divider (between baseline and first week, if both present)
    if labels[0] == '1차' and len(labels) > 1:
        ax.axvline(0.5, color=PALETTE['border'], linestyle=':', linewidth=1.2, alpha=0.9)

    ax.set_title('주별 총 평균 추이', fontsize=14, fontweight='semibold',
                 color=PALETTE['text_primary'], pad=14)
    ax.set_ylabel('주별 평균 점수', fontsize=10, color=PALETTE['text_secondary'], labelpad=8)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9, color=PALETTE['text_secondary'])
    ax.tick_params(axis='x', pad=4)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    # Y-axis headroom/footroom for badge + dip labels
    y_data_min = min(values)
    y_data_max = max(values)
    span = max(y_data_max - y_data_min, 5)
    ax.set_ylim(y_data_min - span * 0.18, y_data_max + span * 0.18)
    ax.grid(True, axis='y', color=PALETTE['border'], linewidth=0.6, alpha=0.6)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(PALETTE['border'])
        ax.spines[spine].set_linewidth(0.8)
    ax.set_facecolor(PALETTE['surface'])

    # Dip culprit labels — placed below the dipping point
    if dip_annotations:
        y_min, y_max = ax.get_ylim()
        offset = (y_max - y_min) * 0.07
        for idx, text in dip_annotations.items():
            ax.annotate(
                f'↓ {text}',
                xy=(idx, values[idx]),
                xytext=(idx, values[idx] - offset),
                ha='center', va='top', fontsize=8.5, color=PALETTE['red'], fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE['red_soft'],
                          edgecolor=PALETTE['red'], linewidth=0.8, alpha=0.95),
            )

    # Direction badge — top-left
    ax.text(0.01, 0.97, badge_text[direction], transform=ax.transAxes,
            ha='left', va='top', fontsize=11, fontweight='semibold', color=color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=PALETTE['surface'],
                      edgecolor=color, linewidth=1.2, alpha=0.98))

    legend = ax.legend(loc='upper right', fontsize=9, frameon=True, borderpad=0.6)
    legend.get_frame().set_facecolor(PALETTE['surface'])
    legend.get_frame().set_edgecolor(PALETTE['border'])
    legend.get_frame().set_linewidth(0.8)


def draw_category_summary(ax, student_radar, level_radar, level):
    """Text panel: per-category score, level avg, and diff with arrow."""
    box = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                   boxstyle="round,pad=0.04",
                                   facecolor=PALETTE['surface'],
                                   edgecolor=PALETTE['border'],
                                   linewidth=1)
    ax.add_patch(box)

    # Title — pulled down a touch so it doesn't kiss the box border
    ax.text(0.5, 0.88, '영역별 비교', ha='center', va='center',
            fontsize=14, fontweight='semibold', transform=ax.transAxes,
            color=PALETTE['text_primary'])

    # Header row
    ax.text(0.10, 0.74, '영역', ha='left', va='center',
            fontsize=10.5, fontweight='semibold', transform=ax.transAxes,
            color=PALETTE['text_tertiary'])
    ax.text(0.42, 0.74, '학생', ha='center', va='center',
            fontsize=10.5, fontweight='semibold', transform=ax.transAxes,
            color=PALETTE['text_tertiary'])
    ax.text(0.64, 0.74, f'Lv{level} 평균', ha='center', va='center',
            fontsize=10.5, fontweight='semibold', transform=ax.transAxes,
            color=PALETTE['text_tertiary'])
    ax.text(0.90, 0.74, '차이', ha='right', va='center',
            fontsize=10.5, fontweight='semibold', transform=ax.transAxes,
            color=PALETTE['text_tertiary'])
    ax.hlines(0.685, 0.07, 0.93, transform=ax.transAxes,
              color=PALETTE['border'], linewidth=0.8)

    y = 0.60
    for cat in RADAR_CATEGORIES.keys():
        s = student_radar.get(cat, 0)
        l = level_radar.get(cat, 0)
        diff = s - l
        if diff > 2:
            diff_color, arrow = PALETTE['green'], '↑'
        elif diff < -2:
            diff_color, arrow = PALETTE['red'], '↓'
        else:
            diff_color, arrow = PALETTE['text_secondary'], '→'

        ax.text(0.10, y, RADAR_LABELS_KR[cat], ha='left', va='center',
                fontsize=11, transform=ax.transAxes, fontweight='semibold',
                color=PALETTE['text_primary'])
        ax.text(0.42, y, f'{s:.1f}', ha='center', va='center',
                fontsize=11, transform=ax.transAxes, color=PALETTE['text_primary'])
        ax.text(0.64, y, f'{l:.1f}', ha='center', va='center',
                fontsize=11, transform=ax.transAxes, color=PALETTE['text_secondary'])
        ax.text(0.90, y, f'{diff:+.1f} {arrow}', ha='right', va='center',
                fontsize=11, transform=ax.transAxes, color=diff_color, fontweight='semibold')
        y -= 0.12


def load_master_data(data_dir):
    """Load the master CSV file and populate PROBATION_STUDENTS from Probation column."""
    global PROBATION_STUDENTS
    master_path = Path(data_dir) / MASTER_FILE
    df = pd.read_csv(master_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Populate probation students from the Probation column
    if 'Probation' in df.columns:
        PROBATION_STUDENTS = set(df.loc[df['Probation'] == True, 'Name'].unique())

    return df


def calculate_student_stats(df):
    """Calculate statistics for each student."""
    # Filter out teacher/reference rows (names starting with *)
    students = df[~df['Name'].str.startswith('*', na=False)].groupby('Name')

    results = []
    for name, student_data in students:
        # Get level (most common level for this student)
        level = student_data['Level'].mode().iloc[0] if not student_data['Level'].mode().empty else None

        # Get class
        student_class = student_data['Class'].iloc[0]

        # Calculate total average across all subjects and dates
        all_scores = []
        subject_scores = {}
        subject_trends = {}

        # Process individual columns (not writing or science)
        for col in INDIVIDUAL_COLUMNS:
            if col in student_data.columns:
                scores = student_data[['Date', col]].dropna()
                if len(scores) > 0:
                    subject_scores[col] = scores[col].mean()
                    all_scores.extend(scores[col].tolist())

                    # Calculate trend (comparing first half to second half)
                    if len(scores) >= 2:
                        sorted_scores = scores.sort_values('Date')
                        mid = len(sorted_scores) // 2
                        first_half = sorted_scores.iloc[:mid][col].mean()
                        second_half = sorted_scores.iloc[mid:][col].mean()
                        subject_trends[col] = second_half - first_half

        # Combine all writing columns into single "Writing" score
        writing_scores_all = []
        writing_dated_scores = []
        for col in WRITING_COLUMNS:
            if col in student_data.columns:
                scores = student_data[['Date', col]].dropna()
                if len(scores) > 0:
                    writing_scores_all.extend(scores[col].tolist())
                    all_scores.extend(scores[col].tolist())
                    for _, row in scores.iterrows():
                        writing_dated_scores.append((row['Date'], row[col]))

        if writing_scores_all:
            subject_scores['Writing'] = np.mean(writing_scores_all)
            # Calculate writing trend
            if len(writing_dated_scores) >= 2:
                writing_dated_scores.sort(key=lambda x: x[0])
                mid = len(writing_dated_scores) // 2
                first_half = np.mean([s[1] for s in writing_dated_scores[:mid]])
                second_half = np.mean([s[1] for s in writing_dated_scores[mid:]])
                subject_trends['Writing'] = second_half - first_half

        # Combine all science columns into single "Science" score
        science_scores_all = []
        science_dated_scores = []
        for col in SCIENCE_COLUMNS:
            if col in student_data.columns:
                scores = student_data[['Date', col]].dropna()
                if len(scores) > 0:
                    science_scores_all.extend(scores[col].tolist())
                    all_scores.extend(scores[col].tolist())
                    for _, row in scores.iterrows():
                        science_dated_scores.append((row['Date'], row[col]))

        if science_scores_all:
            subject_scores['Science'] = np.mean(science_scores_all)
            # Calculate science trend
            if len(science_dated_scores) >= 2:
                science_dated_scores.sort(key=lambda x: x[0])
                mid = len(science_dated_scores) // 2
                first_half = np.mean([s[1] for s in science_dated_scores[:mid]])
                second_half = np.mean([s[1] for s in science_dated_scores[mid:]])
                subject_trends['Science'] = second_half - first_half

        if not all_scores:
            continue

        total_avg = np.mean(all_scores)

        # Calculate overall trend
        dated_scores = []
        for col in SCORE_COLUMNS:
            if col in student_data.columns:
                for _, row in student_data[['Date', col]].dropna().iterrows():
                    dated_scores.append((row['Date'], row[col]))

        overall_trend = 0
        if dated_scores:
            dated_scores.sort(key=lambda x: x[0])
            mid = len(dated_scores) // 2
            if mid > 0:
                first_half_avg = np.mean([s[1] for s in dated_scores[:mid]])
                second_half_avg = np.mean([s[1] for s in dated_scores[mid:]])
                overall_trend = second_half_avg - first_half_avg

        # Calculate monthly averages
        monthly_avgs = {}
        for col in SCORE_COLUMNS:
            if col in student_data.columns:
                for _, row in student_data[['Date', col]].dropna().iterrows():
                    month_key = row['Date'].strftime('%Y-%m')
                    monthly_avgs.setdefault(month_key, []).append(row[col])
        monthly_avgs = {k: round(np.mean(v), 1) for k, v in sorted(monthly_avgs.items())}

        # Count weeks of data (unique dates)
        weeks_count = student_data['Date'].nunique()

        results.append({
            'Name': name,
            'Class': student_class,
            'Level': int(level) if pd.notna(level) else None,
            'Total_Average': round(total_avg, 1),
            'Overall_Trend': round(overall_trend, 1),
            'Subject_Scores': subject_scores,
            'Subject_Trends': subject_trends,
            'Monthly_Averages': monthly_avgs,
            'Weeks_Count': weeks_count
        })

    return pd.DataFrame(results)


def calculate_level_subject_averages(student_stats):
    """Calculate average score for each subject at each level."""
    level_subject_avgs = {2: {}, 3: {}, 4: {}}

    for level in [2, 3, 4]:
        level_students = student_stats[student_stats['Level'] == level]

        # Collect all subject scores for this level
        subject_totals = {}
        subject_counts = {}

        for _, student in level_students.iterrows():
            for subject, score in student['Subject_Scores'].items():
                if subject not in subject_totals:
                    subject_totals[subject] = 0
                    subject_counts[subject] = 0
                subject_totals[subject] += score
                subject_counts[subject] += 1

        # Calculate averages
        for subject in subject_totals:
            if subject_counts[subject] > 0:
                level_subject_avgs[level][subject] = subject_totals[subject] / subject_counts[subject]

    return level_subject_avgs


def find_weakest_subject_by_level(student_row, level_subject_avgs):
    """
    Find the subject where student is furthest below their level's average.
    Returns (subject_name, student_score, diff_from_level_avg, trend)
    """
    level = student_row['Level']
    subject_scores = student_row['Subject_Scores']
    subject_trends = student_row.get('Subject_Trends', {})

    if level not in level_subject_avgs or not subject_scores:
        return None, None, None, 0

    level_avgs = level_subject_avgs[level]

    # Find subject with largest negative difference from level average
    worst_diff = float('inf')
    weakest_subject = None
    weakest_score = None

    for subject, score in subject_scores.items():
        if subject in level_avgs:
            diff = score - level_avgs[subject]
            if diff < worst_diff:
                worst_diff = diff
                weakest_subject = subject
                weakest_score = score

    weakest_trend = subject_trends.get(weakest_subject, 0) if weakest_subject else 0

    return weakest_subject, weakest_score, worst_diff, weakest_trend


def calculate_level_stats(student_stats):
    """Calculate statistics per level."""
    level_stats = {}

    for level in [2, 3, 4]:
        level_data = student_stats[student_stats['Level'] == level]
        if len(level_data) > 0:
            level_stats[level] = {
                'average': level_data['Total_Average'].mean(),
                'max': level_data['Total_Average'].max(),
                'min': level_data['Total_Average'].min(),
                'count': len(level_data),
                'students': level_data['Total_Average'].tolist()
            }

    return level_stats


def load_cutlines(data_dir):
    """Load cutline point estimates written by build_master.py."""
    cutlines_path = Path(data_dir) / CUTLINES_FILE
    df = pd.read_csv(cutlines_path)
    cutlines = {}
    for _, row in df.iterrows():
        cutlines[int(row['Level_Boundary'])] = row['Cutline']
    return cutlines


def load_cutline_zones(data_dir):
    """Load cutline zones (Low/Center/High) written by build_master.py.

    Returns: {level_boundary: (low, center, high)}. Falls back to (center, center, center)
    if Low/High columns are missing (a plain single-cutline file).
    """
    cutlines_path = Path(data_dir) / CUTLINES_FILE
    df = pd.read_csv(cutlines_path)
    zones = {}
    has_zone = {'Cutline_Low', 'Cutline_High'}.issubset(df.columns)
    for _, row in df.iterrows():
        lv = int(row['Level_Boundary'])
        center = float(row['Cutline'])
        if has_zone:
            zones[lv] = (float(row['Cutline_Low']), center, float(row['Cutline_High']))
        else:
            zones[lv] = (center, center, center)
    return zones


def calculate_level_probability(student_avg, student_level, level_stats, cutlines, student_name=''):
    """
    Calculate probability of level change using exponential scale.
    Probability increases exponentially with distance from cutline.

    Returns: (direction, probability, status)
    - direction: 'up', 'down', 'probation', 'expulsion', or 'stable'
    - probability: 0-100%
    - status: description string
    """
    if student_level is None:
        return ('stable', 0, 'N/A')

    current_level_stats = level_stats.get(student_level)
    if not current_level_stats:
        return ('stable', 0, 'N/A')

    # Check if student is already on probation
    is_on_probation = student_name in PROBATION_STUDENTS

    def exp_probability(margin, max_prob, base=5):
        """Calculate probability using exponential scale. Higher margin = much higher probability."""
        if margin <= 0:
            return 0
        # Exponential growth: prob = base + (max - base) * (1 - e^(-k*margin))
        # k=0.5 means rapid growth: margin 2 -> ~50%, margin 5 -> ~80%, margin 8 -> ~95%
        k = 0.5
        prob = base + (max_prob - base) * (1 - np.exp(-k * margin))
        return min(max_prob, max(base, prob))

    # Check for level up (to lower number = higher proficiency)
    # Only show if actually above the cutline
    if student_level > 2:  # Can level up
        higher_level = student_level - 1
        higher_stats = level_stats.get(higher_level)

        if higher_stats:
            cutline = cutlines.get(higher_level, higher_stats['min'])
            margin = student_avg - cutline  # positive = above cutline

            if margin > 0:  # Above cutline
                prob = exp_probability(margin, max_prob=70, base=5)
                if margin >= 5:
                    return ('up', prob, f'Strong candidate for Lv{higher_level}')
                elif margin >= 2:
                    return ('up', prob, f'Possible level up to Lv{higher_level}')
                else:
                    return ('up', prob, f'Near Lv{higher_level} cutline')

    # Check for level down (to higher number = lower proficiency)
    # Only show risk if actually below cutline
    if student_level < 4:  # Can level down
        lower_level = student_level + 1
        cutline = cutlines.get(student_level, current_level_stats['min'])
        margin = cutline - student_avg  # positive = below cutline

        if margin > 0:  # Below cutline
            prob = exp_probability(margin, max_prob=70, base=10)
            if margin >= 5:
                return ('down', prob, f'Risk of dropping to Lv{lower_level}')
            elif margin >= 2:
                return ('down', prob, f'Risk of dropping to Lv{lower_level}')
            else:
                return ('down', prob, f'Near Lv{lower_level} cutline')

    # Check for probation/expulsion (Lv4 students)
    if student_level == 4:
        probation_cutline = cutlines.get(4, 50)
        margin_above = student_avg - probation_cutline  # positive = above cutline
        margin_below = probation_cutline - student_avg  # positive = below cutline

        # For probation students: check if they can level up to Lv4
        # Only show level up if above cutline, expulsion risk if below
        if is_on_probation:
            if margin_above > 0:  # Above cutline
                prob = exp_probability(margin_above, max_prob=70, base=5)
                if margin_above >= 5:
                    return ('up', prob, 'Strong candidate for Lv4')
                elif margin_above >= 2:
                    return ('up', prob, 'Possible level up to Lv4')
                else:
                    return ('up', prob, 'Near Lv4 cutline')
            elif margin_below > 0:  # Below cutline
                prob = exp_probability(margin_below, max_prob=70, base=10)
                return ('expulsion', prob, 'Risk of expulsion')
            else:  # Exactly at cutline
                return ('stable', 0, 'Stable in current level')

        # For regular Lv4 students - only show risk if below cutline
        if margin_below > 0:
            prob = exp_probability(margin_below, max_prob=70, base=10)
            return ('probation', prob, 'Risk of probation status')

    # Stable
    return ('stable', 0, 'Stable in current level')


def generate_student_report(student_row, level_stats, cutlines, level_subject_avgs, output_dir,
                            student_radar=None, level_radar_avgs=None,
                            cutline_zones=None, prior_radar=None,
                            df_current=None, prior_total=None):
    """Generate a PNG report for a single student, organized by class folder."""
    name = student_row['Name']
    student_class = student_row['Class']
    level = student_row['Level']

    # Check if student is on probation
    is_on_probation = name in PROBATION_STUDENTS
    level_display = '유예' if is_on_probation else level

    # Create class-specific subfolder
    class_dir = Path(output_dir) / student_class
    class_dir.mkdir(exist_ok=True)
    total_avg = student_row['Total_Average']

    # Create figure
    fig = plt.figure(figsize=(14, 18.5))
    fig.patch.set_facecolor(PALETTE['bg'])

    # Top header: report title + student identity
    fig.suptitle("Monthly Progress Report", fontsize=24, fontweight='bold',
                 y=0.975, color=PALETTE['text_primary'])
    plt.figtext(0.5, 0.916,
                f'{name}   ·   반 {student_class}   ·   레벨 {level_display}',
                ha='center', fontsize=14, color=PALETTE['text_secondary'], fontweight='medium')

    # Warning for insufficient data (top-right, aligned with title baseline)
    weeks_count = student_row.get('Weeks_Count', 0)
    if weeks_count < 4:
        plt.figtext(0.96, 0.975, f'⚠ 데이터 부족 ({weeks_count}주) · 참고용',
                    ha='right', va='center', fontsize=10, color=PALETTE['red'],
                    fontweight='semibold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=PALETTE['red_soft'],
                              edgecolor=PALETTE['red'], linewidth=0.8, alpha=0.95))

    # Grid layout - 3 rows: box plot, radar+category, weekly trend.
    # Box plot explanation is placed as a fig.text just below the box plot (not a row)
    # so it hugs the chart it describes instead of floating in row-sized whitespace.
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.45,
                          left=0.06, right=0.94, top=0.89, bottom=0.045,
                          height_ratios=[1.55, 1.4, 1.0])

    # 1. Level Position Chart (top, spans full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('레벨 내 위치', fontsize=16, fontweight='semibold',
                  color=PALETTE['text_primary'], pad=14)
    ax1.set_facecolor(PALETTE['surface'])
    for spine in ('top', 'right'):
        ax1.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax1.spines[spine].set_color(PALETTE['border'])
        ax1.spines[spine].set_linewidth(0.8)

    # Apple system colors per level
    colors = {2: PALETTE['green'], 3: PALETTE['orange'], 4: PALETTE['red']}
    level_names = {2: '상급 (Lv2)', 3: '중급 (Lv3)', 4: '초급 (Lv4)'}
    y_positions = {2: 3, 3: 1.5, 4: 0}

    for lv, stats in level_stats.items():
        y = y_positions[lv]
        scores = stats['students']

        ax1.boxplot([scores], positions=[y], vert=False, widths=0.55,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors[lv], alpha=0.18,
                                  edgecolor=colors[lv], linewidth=1.4),
                    medianprops=dict(color=colors[lv], linewidth=1.8),
                    whiskerprops=dict(color=colors[lv], linewidth=1.2),
                    capprops=dict(color=colors[lv], linewidth=1.2),
                    flierprops=dict(marker='o', markerfacecolor=colors[lv],
                                    markeredgecolor=colors[lv], markersize=4, alpha=0.45))

        ax1.text(stats['max'] + 1.5, y, level_names[lv], va='center',
                 fontsize=12, fontweight='semibold', color=PALETTE['text_primary'])
        ax1.text(stats['average'], y + 0.55, f'평균 {stats["average"]:.1f}',
                 ha='center', fontsize=10, color=colors[lv], fontweight='semibold')
        ax1.vlines(stats['average'], y - 0.3, y + 0.3, color=colors[lv],
                   linestyle='--', linewidth=1.4, alpha=0.85)

    # Mark student position
    if level in y_positions:
        y = y_positions[level]
        ax1.scatter([total_avg], [y], s=140, c=PALETTE['blue'], zorder=5,
                    edgecolors=PALETTE['surface'], linewidths=2.5, marker='o')
        ax1.annotate(f'{name} · {total_avg:.1f}', (total_avg, y),
                     textcoords="offset points", xytext=(0, 22), ha='center',
                     fontsize=12, fontweight='semibold', color=PALETTE['blue'])

    # Dynamic x-axis bounds
    all_scores = []
    for stats in level_stats.values():
        all_scores.extend(stats['students'])
    x_min = max(0, min(all_scores) - 5)
    x_max = max(all_scores) + 5

    ax1.set_yticks([])
    ax1.set_ylim(-0.8, 4.5)
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel('평균 점수', fontsize=11, color=PALETTE['text_secondary'], labelpad=10)
    ax1.grid(axis='x', color=PALETTE['border'], linewidth=0.6, alpha=0.6)

    # Monthly averages in bottom right of chart (color-coded by change)
    monthly_avgs = student_row.get('Monthly_Averages', {})
    if monthly_avgs:
        months = list(monthly_avgs.items())
        text_parts = []
        for i, (k, v) in enumerate(months):
            if i == 0:
                color = PALETTE['text_secondary']
            else:
                prev_v = months[i - 1][1]
                color = (PALETTE['green'] if v > prev_v
                         else PALETTE['red'] if v < prev_v
                         else PALETTE['text_secondary'])
            arrow = '' if i == 0 else ' → '
            label = f'{arrow}{k[-2:]}월 {v}'
            text_parts.append(TextArea(label, textprops=dict(
                fontsize=9, color=color, fontweight='semibold')))

        box = HPacker(children=text_parts, pad=0, sep=0, align='baseline')
        anchored = AnchoredOffsetbox(loc='lower right', child=box, pad=0.4,
                                      bbox_to_anchor=(0.99, 0.02), bbox_transform=ax1.transAxes,
                                      borderpad=0.3, frameon=True)
        anchored.patch.set_boxstyle('round,pad=0.35')
        anchored.patch.set_facecolor(PALETTE['surface'])
        anchored.patch.set_edgecolor(PALETTE['border'])
        anchored.patch.set_linewidth(0.8)
        ax1.add_artist(anchored)

    # Draw cutline zones (shaded bands) instead of single vertical lines
    for lv, cutline in cutlines.items():
        if cutline_zones and lv in cutline_zones:
            low, center, high = cutline_zones[lv]
        else:
            low = center = high = cutline

        # Shaded zone band — soft neutral so it doesn't fight the level boxes
        if high > low:
            ax1.axvspan(low, high, color=PALETTE['gray'], alpha=0.10, zorder=1)
            ax1.axvline(low, color=PALETTE['border'], linestyle=':', linewidth=0.8, alpha=0.8)
            ax1.axvline(high, color=PALETTE['border'], linestyle=':', linewidth=0.8, alpha=0.8)
        # Center line (point estimate)
        ax1.axvline(center, color=PALETTE['text_tertiary'],
                    linestyle='--', linewidth=1.2, alpha=0.75)

        if lv == 2:
            label = 'Lv2 커트라인 예상 구간'
        elif lv == 3:
            label = 'Lv3 커트라인 예상 구간'
        else:
            label = '유예 커트라인 예상 구간'
        ax1.text(center, 4.0, label, ha='center', fontsize=10,
                color=PALETTE['text_secondary'], fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE['surface'],
                          edgecolor=PALETTE['border'], linewidth=0.6, alpha=0.95))

    # 2. Radar chart + category summary (row 1)
    if student_radar is not None and level_radar_avgs is not None and level in level_radar_avgs:
        ax_radar = fig.add_subplot(gs[1, 0], projection='polar')
        draw_radar(ax_radar, student_radar, level_radar_avgs[level], name, level, prior_radar=prior_radar)

        ax_cat = fig.add_subplot(gs[1, 1])
        ax_cat.axis('off')
        draw_category_summary(ax_cat, student_radar, level_radar_avgs[level], level)

    # 3. Weekly trend (row 2, spans full width)
    if df_current is not None:
        labels_w, values_w, slope_w, direction_w, dips_w = compute_weekly_trend(
            name, df_current, prior_total=prior_total
        )
    else:
        labels_w, values_w, slope_w, direction_w, dips_w = [], [], 0.0, 'neutral', {}
    ax_trend = fig.add_subplot(gs[2, :])
    draw_weekly_trend(ax_trend, labels_w, values_w, slope_w, direction_w, dip_annotations=dips_w)

    # Box plot explanation — sits directly under the box plot, hugging its border
    legend_text = (
        "차트 설명   ·   박스 = 25–75% 구간 (중간 50% 분포)   ·   "
        "박스 내 실선 = 중앙값   ·   점선 = 평균   ·   "
        "수염 = 전체 범위   ·   파란 점 = 학생 위치   ·   "
        "회색 음영 = 커트라인 예상 구간 (구간 안 = 경계선)\n"
        "본 리포트는 반 내 데이터만을 기준으로 한 예측이며, 실제 레벨 조정 결과와 다를 수 있습니다. 참고용으로만 활용해 주세요."
    )
    box_pos = ax1.get_position()
    fig.text(0.5, box_pos.y0 - 0.018, legend_text,
             ha='center', va='top', fontsize=10, color=PALETTE['text_secondary'],
             bbox=dict(boxstyle='round,pad=0.45', facecolor=PALETTE['surface'],
                       edgecolor=PALETTE['border'], linewidth=0.8),
             linespacing=1.4)

    # Save to class-specific folder
    output_path = class_dir / f'{name.replace(" ", "_")}_report.png'
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close()

    return output_path


def generate_all_reports(data_dir, output_subdir=REPORTS_SUBDIR):
    """Generate reports for all students."""
    print("=" * 60)
    print("Cohort B Monthly Report Generator")
    print("=" * 60)

    # Create output directory
    output_dir = Path(data_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1] Loading master data...")
    df = load_master_data(data_dir)
    print(f"Loaded {len(df)} records")

    # Calculate student statistics
    print("\n[2] Calculating student statistics...")
    student_stats = calculate_student_stats(df)
    print(f"Processed {len(student_stats)} students")

    # Calculate level statistics
    print("\n[3] Calculating level statistics...")
    level_stats = calculate_level_stats(student_stats)

    for lv, stats in level_stats.items():
        print(f"  Lv{lv}: avg={stats['average']:.1f}, min={stats['min']:.1f}, max={stats['max']:.1f}, n={stats['count']}")

    # Load cutlines written by build_master.py (center + zone bands)
    cutlines = load_cutlines(data_dir)
    cutline_zones = load_cutline_zones(data_dir)
    print("\n  Cutlines (Low–Center–High):")
    for lv in sorted(cutline_zones):
        low, center, high = cutline_zones[lv]
        print(f"    Lv{lv}: {low:.1f}–{center:.1f}–{high:.1f}")

    # Calculate level subject averages for weakest subject comparison
    level_subject_avgs = calculate_level_subject_averages(student_stats)

    # Precompute per-level radar averages (used for the radar overlay)
    level_radar_avgs = {lv: compute_radar_scores(df[df['Level'] == lv]) for lv in [2, 3, 4]}

    # Prior-period data (optional) for the radar overlay and the trend baseline
    df_prior = load_prior_master(data_dir)
    prior_radar_by_name = load_prior_period_radar(data_dir)
    print(f"\n  Prior-period radar data: {len(prior_radar_by_name)} students")

    # Generate reports
    print(f"\n[4] Generating reports to {output_dir}...")
    generated = []

    for _, student in student_stats.iterrows():
        try:
            student_radar = compute_radar_scores(df[df['Name'] == student['Name']])
            prior_radar = prior_radar_by_name.get(student['Name'])
            prior_total = get_prior_total_average(student['Name'], df_prior)
            path = generate_student_report(
                student, level_stats, cutlines, level_subject_avgs, output_dir,
                student_radar=student_radar, level_radar_avgs=level_radar_avgs,
                cutline_zones=cutline_zones, prior_radar=prior_radar,
                df_current=df, prior_total=prior_total,
            )
            generated.append(path)
            print(f"  ✓ {student['Name']}")
        except Exception as e:
            print(f"  ✗ {student['Name']}: {e}")

    print(f"\n[5] Complete! Generated {len(generated)} reports")

    # Save summary CSV
    summary_path = Path(data_dir) / SUMMARY_FILE

    # Add level change info to summary
    summary_data = []
    for _, student in student_stats.iterrows():
        direction, probability, status = calculate_level_probability(
            student['Total_Average'], student['Level'], level_stats, cutlines, student['Name']
        )
        level_avg = level_stats[student['Level']]['average'] if student['Level'] in level_stats else None

        # Find weakest subject by level comparison
        weakest, weakest_score, weakest_diff, _ = find_weakest_subject_by_level(student, level_subject_avgs)

        # Show 유예 for probation students
        level_display = '유예' if student['Name'] in PROBATION_STUDENTS else student['Level']

        summary_data.append({
            'Name': student['Name'],
            'Class': student['Class'],
            'Level': level_display,
            'Total_Average': student['Total_Average'],
            'Level_Average': round(level_avg, 1) if level_avg else None,
            'Diff_From_Level': round(student['Total_Average'] - level_avg, 1) if level_avg else None,
            'Status': status,
            'Change_Probability': f'{probability:.0f}%' if direction != 'stable' else 'Stable',
            'Weakest_Subject': weakest,
            'Weakest_Score': round(weakest_score, 1) if weakest_score else None,
            'Weakest_Diff_From_Level': round(weakest_diff, 1) if weakest_diff else None,
            'Overall_Trend': student['Overall_Trend']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['Level', 'Total_Average'], ascending=[True, False])
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\nSummary saved to: {summary_path}")

    # Generate at-risk student list (level down, probation, expulsion)
    risk_list = []
    for _, student in student_stats.iterrows():
        direction, probability, status = calculate_level_probability(
            student['Total_Average'], student['Level'], level_stats, cutlines, student['Name']
        )
        if direction in ('down', 'probation', 'expulsion'):
            level_display = '유예' if student['Name'] in PROBATION_STUDENTS else student['Level']
            risk_list.append({
                'Name': student['Name'],
                'Class': student['Class'],
                'Level': level_display,
                'Total_Average': student['Total_Average'],
                'Risk': direction,
                'Probability': round(probability, 1),
                'Status': status,
            })

    if risk_list:
        risk_df = pd.DataFrame(risk_list)
        risk_df = risk_df.sort_values(['Risk', 'Probability'], ascending=[True, False])
        risk_path = Path(data_dir) / RISK_LIST_FILE
        risk_df.to_csv(risk_path, index=False, encoding='utf-8-sig')

        print(f"\n{'=' * 60}")
        print("⚠️  AT-RISK STUDENTS")
        print(f"{'=' * 60}")
        for risk_type in ['expulsion', 'probation', 'down']:
            subset = risk_df[risk_df['Risk'] == risk_type]
            if len(subset) == 0:
                continue
            label = {'expulsion': '🚨 EXPULSION RISK', 'probation': '⚠️  PROBATION RISK', 'down': '⬇️  LEVEL DOWN RISK'}[risk_type]
            print(f"\n{label}")
            print(f"{'Name':<30} {'Class':<8} {'Level':<8} {'Avg':>6} {'Prob':>6}")
            print("-" * 60)
            for _, r in subset.iterrows():
                print(f"{r['Name']:<30} {r['Class']:<8} {r['Level']:<8} {r['Total_Average']:>6.1f} {r['Probability']:>5.1f}%")
        print(f"\nRisk list saved to: {risk_path}")
    else:
        print("\n✓ No at-risk students detected")

    return student_stats, level_stats


def main():
    parser = argparse.ArgumentParser(description="Generate Cohort B monthly PNG reports, summary, and risk list.")
    parser.add_argument('--data-dir', type=Path, default=SCRIPT_DIR,
                        help=f'folder containing {MASTER_FILE}, {CUTLINES_FILE} and optionally {PRIOR_MASTER_FILE} '
                             f'from build_master.py; reports/ and the summary CSVs are written here (default: this folder)')
    args = parser.parse_args()
    generate_all_reports(args.data_dir)


if __name__ == '__main__':
    main()
