#!/usr/bin/env python3
"""
Cohort A - monthly report generator

Reads CohortA_Master.csv and CohortA_Cutlines.csv (from build_master.py) and
generates one PNG report per student showing:
- Total average vs level average
- Position within level (risk of level up/down)
- Weakest subject relative to the level average
- Performance trend
- A comment picked from comment_templates.csv

Also writes CohortA_Monthly_Summary.csv and CohortA_Risk_List.csv.

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

# Configure matplotlib for Korean text
plt.rcParams['font.family'] = ['Arial Unicode MS', 'AppleGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = Path(__file__).resolve().parent

MASTER_FILE = 'CohortA_Master.csv'
CUTLINES_FILE = 'CohortA_Cutlines.csv'
SUMMARY_FILE = 'CohortA_Monthly_Summary.csv'
RISK_LIST_FILE = 'CohortA_Risk_List.csv'
REPORTS_SUBDIR = 'reports'

# Score columns to include in calculations
SCORE_COLUMNS = [
    'Vocabulary', 'Grammar HW', 'Grammar Speaking', 'Science HW', 'Science Test', 'Science Textbook',
    'Reading', 'Grammar Test', 'Speaking', 'Novel'
]

# Grammar columns to combine into single "Grammar" score
GRAMMAR_COLUMNS = ['Grammar HW', 'Grammar Test', 'Grammar Speaking']

# Science columns to combine into single "Science" score
SCIENCE_COLUMNS = ['Science HW', 'Science Test', 'Science Textbook']

# Columns to track individually (excluding combined categories)
INDIVIDUAL_COLUMNS = [col for col in SCORE_COLUMNS if col not in GRAMMAR_COLUMNS and col not in SCIENCE_COLUMNS]

# Probation students are read from the Probation column in the master CSV (set via student_list)
PROBATION_STUDENTS = set()  # populated at runtime by load_master_data()


def load_comment_templates(path):
    """Load comment templates from a CSV with columns
    Subject, Score_Tier, stable, level_up, level_down, probation, expulsion.

    Returns:
        templates: {(subject, tier, level_status): comment_text}
    """
    path = Path(path)
    if not path.exists():
        return {}

    df = pd.read_csv(path, encoding='utf-8-sig')
    templates = {}
    status_columns = ['stable', 'level_up', 'level_down', 'probation', 'expulsion']

    for _, row in df.iterrows():
        subject = row['Subject']
        tier = row['Score_Tier']
        for status in status_columns:
            if pd.notna(row[status]):
                templates[(subject, tier, status)] = row[status]

    return templates


def get_score_tier(diff):
    """Determine score tier based on difference from level average."""
    if diff >= 3.0:
        return 'top'
    elif diff >= 1.0:
        return 'above_avg'
    elif diff >= -0.9:
        return 'average'
    elif diff >= -3.0:
        return 'below_avg'
    else:
        return 'bottom'


def get_student_comment(subject, diff, direction, templates):
    """Select the appropriate comment for a student based on subject, score tier, and level change."""
    tier = get_score_tier(diff)

    # Map direction to CSV column name
    status_map = {
        'up': 'level_up',
        'down': 'level_down',
        'probation': 'probation',
        'expulsion': 'expulsion',
        'stable': 'stable',
    }
    level_status = status_map.get(direction, 'stable')

    return templates.get((subject, tier, level_status), '')


def load_master_data(data_dir):
    """Load the master CSV file and populate PROBATION_STUDENTS from Probation column."""
    global PROBATION_STUDENTS
    master_path = Path(data_dir) / MASTER_FILE
    df = pd.read_csv(master_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Populate probation students from the Probation column
    if 'Probation' in df.columns:
        PROBATION_STUDENTS = set(df.loc[df['Probation'] == True, 'Name'].unique())

    # Convert all score columns to numeric (handles mixed string/numeric values)
    for col in SCORE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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

        # Process individual columns (not grammar or science)
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

        # Combine all grammar columns into single "Grammar" score
        grammar_scores_all = []
        grammar_dated_scores = []
        for col in GRAMMAR_COLUMNS:
            if col in student_data.columns:
                scores = student_data[['Date', col]].dropna()
                if len(scores) > 0:
                    grammar_scores_all.extend(scores[col].tolist())
                    all_scores.extend(scores[col].tolist())
                    for _, row in scores.iterrows():
                        grammar_dated_scores.append((row['Date'], row[col]))

        if grammar_scores_all:
            subject_scores['Grammar'] = np.mean(grammar_scores_all)
            # Calculate grammar trend
            if len(grammar_dated_scores) >= 2:
                grammar_dated_scores.sort(key=lambda x: x[0])
                mid = len(grammar_dated_scores) // 2
                first_half = np.mean([s[1] for s in grammar_dated_scores[:mid]])
                second_half = np.mean([s[1] for s in grammar_dated_scores[mid:]])
                subject_trends['Grammar'] = second_half - first_half

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
    """Load cutlines written by build_master.py."""
    cutlines_path = Path(data_dir) / CUTLINES_FILE
    df = pd.read_csv(cutlines_path)
    cutlines = {}
    for _, row in df.iterrows():
        cutlines[int(row['Level_Boundary'])] = row['Cutline']
    return cutlines


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


def generate_student_report(student_row, level_stats, cutlines, level_subject_avgs, output_dir, templates=None):
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
    overall_trend = student_row['Overall_Trend']

    # Find weakest subject compared to level average
    weakest, weakest_score, weakest_diff, weakest_trend = find_weakest_subject_by_level(
        student_row, level_subject_avgs
    )

    # Calculate level probability
    direction, probability, status = calculate_level_probability(total_avg, level, level_stats, cutlines, name)

    # Translate status to Korean (predictive language - based on class data only)
    status_korean = {
        'Strong candidate for Lv2': 'Lv2 승급 예상',
        'Strong candidate for Lv3': 'Lv3 승급 예상',
        'Strong candidate for Lv4': 'Lv4 승급 예상',
        'Possible level up to Lv2': 'Lv2 승급 가능성',
        'Possible level up to Lv3': 'Lv3 승급 가능성',
        'Possible level up to Lv4': 'Lv4 승급 가능성',
        'Near Lv2 cutline': 'Lv2 커트라인 근접',
        'Near Lv3 cutline': 'Lv3 커트라인 근접',
        'Near Lv4 cutline': 'Lv4 커트라인 근접',
        'Risk of dropping to Lv3': 'Lv3 강등 가능성',
        'Risk of dropping to Lv4': 'Lv4 강등 가능성',
        'Risk of probation status': '유예 가능성',
        'Risk of expulsion': '퇴원 가능성',
        'Stable in current level': '현재 레벨 유지 예상',
        'N/A': '해당 없음'
    }
    status_kr = status_korean.get(status, status)

    # Create figure with more height for action plan section
    fig = plt.figure(figsize=(14, 16))
    fig.patch.set_facecolor('#f5f5f5')

    # Title: more space between main title and class info, less between class info and chart
    fig.suptitle(f'월간 리포트: {name}', fontsize=24, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.94, f'반: {student_class}  |  레벨: {level_display}',
                ha='center', fontsize=15, color='#555')

    # Warning for insufficient data
    weeks_count = student_row.get('Weeks_Count', 0)
    if weeks_count < 4:
        plt.figtext(0.94, 0.97, f'⚠ 데이터 부족 ({weeks_count}주) - 참고용',
                    ha='right', fontsize=10, color='#D32F2F', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#D32F2F', alpha=0.9))

    # Grid layout - 4 rows (chart, legend, status/weakest, action plan)
    gs = fig.add_gridspec(4, 2, hspace=0.25, wspace=0.35,
                          left=0.06, right=0.94, top=0.91, bottom=0.03,
                          height_ratios=[1.6, 0.15, 0.8, 0.6])

    # 1. Level Position Chart (top, spans full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('레벨 내 위치', fontsize=16, fontweight='bold', pad=8)

    # Draw level distributions as box plots
    colors = {2: '#4CAF50', 3: '#FFC107', 4: '#FF5722'}
    level_names = {2: '상급 (Lv2)', 3: '중급 (Lv3)', 4: '초급 (Lv4)'}
    y_positions = {2: 3, 3: 1.5, 4: 0}  # More vertical spacing

    for lv, stats in level_stats.items():
        y = y_positions[lv]
        scores = stats['students']

        # Draw horizontal box plot
        bp = ax1.boxplot([scores], positions=[y], vert=False, widths=0.6,
                         patch_artist=True,
                         boxprops=dict(facecolor=colors[lv], alpha=0.3, edgecolor=colors[lv], linewidth=2),
                         medianprops=dict(color=colors[lv], linewidth=2),
                         whiskerprops=dict(color=colors[lv], linewidth=1.5),
                         capprops=dict(color=colors[lv], linewidth=1.5),
                         flierprops=dict(marker='o', markerfacecolor=colors[lv], markersize=5, alpha=0.5))

        # Level name on right
        ax1.text(stats['max'] + 1.5, y, level_names[lv], va='center', fontsize=12, fontweight='bold')
        # Show average value above box
        ax1.text(stats['average'], y + 0.55, f'평균: {stats["average"]:.1f}',
                ha='center', fontsize=10, color=colors[lv], fontweight='bold')
        # Draw average line (dashed) - different from median
        ax1.vlines(stats['average'], y - 0.3, y + 0.3, color=colors[lv],
                   linestyle='--', linewidth=2, alpha=0.8)

    # Mark student position
    if level in y_positions:
        y = y_positions[level]
        ax1.scatter([total_avg], [y], s=120, c='red', zorder=5, edgecolors='darkred', linewidths=1.5, marker='o')
        ax1.annotate(f'● {name}: {total_avg:.1f}점', (total_avg, y), textcoords="offset points",
                    xytext=(0, 22), ha='center', fontsize=13, fontweight='bold', color='red')

    # Dynamic x-axis minimum based on lowest score across all levels
    all_scores = []
    for stats in level_stats.values():
        all_scores.extend(stats['students'])
    x_min = max(0, min(all_scores) - 5)
    x_max = max(all_scores) + 5

    ax1.set_yticks([])
    ax1.set_ylim(-0.8, 4.5)  # More room at top and bottom
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel('평균 점수', fontsize=12, labelpad=10)
    ax1.grid(axis='x', alpha=0.3)

    # Monthly averages in bottom right of chart (color-coded by change)
    monthly_avgs = student_row.get('Monthly_Averages', {})
    if monthly_avgs:
        months = list(monthly_avgs.items())
        text_parts = []
        for i, (k, v) in enumerate(months):
            if i == 0:
                color = '#555'
            else:
                prev_v = months[i - 1][1]
                color = '#4CAF50' if v > prev_v else '#FF5722' if v < prev_v else '#555'
            arrow = '' if i == 0 else ' → '
            label = f'{arrow}{k[-2:]}월: {v}'
            text_parts.append(TextArea(label, textprops=dict(fontsize=9, color=color, fontweight='bold')))

        box = HPacker(children=text_parts, pad=0, sep=0, align='baseline')
        anchored = AnchoredOffsetbox(loc='lower right', child=box, pad=0.4,
                                      bbox_to_anchor=(0.99, 0.02), bbox_transform=ax1.transAxes,
                                      borderpad=0.3, frameon=True)
        anchored.patch.set_boxstyle('round,pad=0.3')
        anchored.patch.set_facecolor('white')
        anchored.patch.set_edgecolor('#ccc')
        anchored.patch.set_alpha(0.8)
        ax1.add_artist(anchored)

    # Draw cutlines with labels at top (staggered to avoid overlap)
    for lv, cutline in cutlines.items():
        ax1.axvline(cutline, color='#888', linestyle=':', linewidth=2, alpha=0.6)
        if lv == 2:
            label = f'Lv2 예상 커트라인: {cutline:.1f}'
            y_pos = 4.0
        elif lv == 3:
            label = f'Lv3 예상 커트라인: {cutline:.1f}'
            y_pos = 4.0
        else:
            label = f'유예 예상 기준: {cutline:.1f}'
            y_pos = 4.0
        ax1.text(cutline, y_pos, label, ha='center', fontsize=10,
                color='#555', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    # 2. Legend/Explanation (middle row)
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    legend_text = (
        "◆ 차트 설명:   박스 = 25%~75% 구간 (중간 50% 학생 분포)   |   "
        "박스 내 실선 = 중앙값   |   점선 = 평균   |   "
        "수염 = 전체 범위   |   ● 빨간 점 = 학생 위치\n"
        "⚠️ 주의: 본 리포트는 반 내 데이터만을 기준으로 한 예측입니다. "
        "실제 레벨 조정 결과와 다를 수 있으며, 참고용으로만 활용해 주세요."
    )
    ax_legend.text(0.5, 0.5, legend_text, ha='center', va='center', fontsize=10,
                   transform=ax_legend.transAxes, color='#444',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#ccc', linewidth=1.5),
                   linespacing=1.5)

    # 3. Status Box (bottom left)
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axis('off')

    # Status colors
    status_colors = {
        'up': '#4CAF50',
        'down': '#FF5722',
        'probation': '#D32F2F',
        'expulsion': '#880000',  # Dark red for expulsion risk
        'stable': '#2196F3'
    }

    status_box = mpatches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                          boxstyle="round,pad=0.05",
                                          facecolor=status_colors.get(direction, '#2196F3'),
                                          alpha=0.12,
                                          edgecolor=status_colors.get(direction, '#2196F3'),
                                          linewidth=3)
    ax2.add_patch(status_box)

    ax2.text(0.5, 0.85, '레벨 예측', ha='center', va='center',
            fontsize=16, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.62, status_kr, ha='center', va='center',
            fontsize=14, transform=ax2.transAxes, fontweight='bold')

    if direction != 'stable':
        ax2.text(0.5, 0.42, f'예상 확률: {probability:.0f}%', ha='center', va='center',
                fontsize=16, fontweight='bold', color=status_colors.get(direction, '#2196F3'),
                transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.42, '✓ 유지 예상', ha='center', va='center',
                fontsize=16, fontweight='bold', color='#2196F3',
                transform=ax2.transAxes)

    # Summary info below status
    level_avg = level_stats[level]['average'] if level in level_stats else 0
    diff = total_avg - level_avg
    trend_arrow = '↑' if overall_trend > 2 else '↓' if overall_trend < -2 else '→'
    trend_text = f'상승 중 {trend_arrow}' if overall_trend > 2 else f'하락 중 {trend_arrow}' if overall_trend < -2 else f'유지 중 {trend_arrow}'

    ax2.text(0.5, 0.22, f"총 평균: {total_avg:.1f}   |   레벨 평균: {level_avg:.1f}", ha='center', va='center',
            fontsize=11, transform=ax2.transAxes, color='#444')
    ax2.text(0.5, 0.08, f"차이: {diff:+.1f}   |   추세: {trend_text}", ha='center', va='center',
            fontsize=11, transform=ax2.transAxes, color='#444')

    # 4. Weakest Subject (bottom right)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    weak_box = mpatches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#FFF3E0',
                                        alpha=0.5,
                                        edgecolor='#FF9800',
                                        linewidth=2)
    ax4.add_patch(weak_box)

    ax4.text(0.5, 0.85, '보완이 필요한 영역', ha='center', va='center',
            fontsize=16, fontweight='bold', transform=ax4.transAxes)

    if weakest and weakest_score is not None:
        # Get level average for this subject
        level_subj_avg = level_subject_avgs.get(level, {}).get(weakest, weakest_score)
        ax4.text(0.5, 0.65, f'과목: {weakest}', ha='center', va='center',
                fontsize=14, transform=ax4.transAxes, fontweight='bold')
        ax4.text(0.5, 0.45, f'점수: {weakest_score:.1f}점  (레벨 평균: {level_subj_avg:.1f})', ha='center', va='center',
                fontsize=12, transform=ax4.transAxes)
        ax4.text(0.5, 0.25, f'레벨 대비: {weakest_diff:+.1f}점', ha='center', va='center',
                fontsize=14, transform=ax4.transAxes, color='#D84315', fontweight='bold')

    # 5. Action Plan (bottom row, spans full width)
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')

    action_box = mpatches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                          boxstyle="round,pad=0.05",
                                          facecolor='#E8F5E9',
                                          alpha=0.5,
                                          edgecolor='#4CAF50',
                                          linewidth=2)
    ax5.add_patch(action_box)
    ax5.text(0.5, 0.85, '선생님 조언', ha='center', va='center',
            fontsize=16, fontweight='bold', transform=ax5.transAxes)

    if templates and weakest and weakest_diff is not None:
        comment = get_student_comment(weakest, weakest_diff, direction, templates)
        if comment:
            ax5.text(0.5, 0.45, comment, ha='center', va='center',
                    fontsize=11, transform=ax5.transAxes, color='#333',
                    linespacing=1.6, wrap=True)
        else:
            ax5.text(0.5, 0.45, '해당 없음', ha='center', va='center',
                    fontsize=12, transform=ax5.transAxes, color='#999')
    else:
        ax5.text(0.5, 0.45, '해당 없음', ha='center', va='center',
                fontsize=12, transform=ax5.transAxes, color='#999')

    # Save to class-specific folder
    output_path = class_dir / f'{name.replace(" ", "_")}_report.png'
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close()

    return output_path


def generate_all_reports(data_dir, templates_path, output_subdir=REPORTS_SUBDIR):
    """Generate reports for all students."""
    print("=" * 60)
    print("Cohort A Monthly Report Generator")
    print("=" * 60)

    # Create output directory
    output_dir = Path(data_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1] Loading master data...")
    df = load_master_data(data_dir)
    print(f"Loaded {len(df)} records")

    # Load comment templates
    templates = load_comment_templates(templates_path)
    print(f"Loaded {len(templates)} comment templates")

    # Calculate student statistics
    print("\n[2] Calculating student statistics...")
    student_stats = calculate_student_stats(df)
    print(f"Processed {len(student_stats)} students")

    # Calculate level statistics
    print("\n[3] Calculating level statistics...")
    level_stats = calculate_level_stats(student_stats)

    for lv, stats in level_stats.items():
        print(f"  Lv{lv}: avg={stats['average']:.1f}, min={stats['min']:.1f}, max={stats['max']:.1f}, n={stats['count']}")

    # Load cutlines written by build_master.py
    cutlines = load_cutlines(data_dir)
    print("\n  Cutlines:")
    for lv, cut in cutlines.items():
        print(f"    Lv{lv}: {cut:.1f}")

    # Calculate level subject averages for weakest subject comparison
    level_subject_avgs = calculate_level_subject_averages(student_stats)

    # Generate reports
    print(f"\n[4] Generating reports to {output_dir}...")
    generated = []

    for _, student in student_stats.iterrows():
        try:
            path = generate_student_report(student, level_stats, cutlines, level_subject_avgs, output_dir, templates)
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
    parser = argparse.ArgumentParser(description="Generate Cohort A monthly PNG reports, summary, and risk list.")
    parser.add_argument('--data-dir', type=Path, default=SCRIPT_DIR,
                        help=f'folder containing {MASTER_FILE} and {CUTLINES_FILE} from build_master.py; '
                             f'reports/ and the summary CSVs are written here (default: this folder)')
    parser.add_argument('--templates', type=Path, default=SCRIPT_DIR / 'comment_templates.csv',
                        help='comment template CSV (default: ./comment_templates.csv)')
    args = parser.parse_args()
    generate_all_reports(args.data_dir, args.templates)


if __name__ == '__main__':
    main()
