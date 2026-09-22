#!/usr/bin/env python3
"""
Synthetic sample-data generator.

Writes deterministic, obviously invented data in exactly the shape the two
pipelines expect, so the repository can be run end to end without any real
student records:

    cohort_a/sample_data/student_list.csv
    cohort_a/sample_data/csv/A1.csv ... A6.csv
    cohort_a/comment_templates.csv           (placeholder feedback library)
    cohort_b/sample_data/student_list.csv
    cohort_b/sample_data/csv/B1.csv ... B4.csv

Student names are "Student<letters> 학생<letters>" (e.g. "StudentA 학생A"),
unique within each cohort. Every class file also carries one '*'-prefixed
reference row per week, which the pipeline ignores when computing statistics.

Usage:
    python3 make_sample_data.py            # regenerates everything in place
    python3 make_sample_data.py --seed 7   # different but still deterministic data
"""

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 2026

# Weekly score CSV layout per cohort. `sparse` lists subjects that are only
# assessed some weeks (probability a given cell is blank); every other subject
# is blank with probability BASE_BLANK_PROB.
COHORTS = {
    'cohort_a': {
        'classes': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
        'students_per_class': 7,
        'subjects': [
            'Vocabulary', 'Grammar HW', 'Grammar Speaking', 'Science HW', 'Science Test',
            'Science Textbook', 'Reading', 'Grammar Test', 'Speaking', 'Novel',
        ],
        'sparse': {'Science Test': 0.55, 'Science Textbook': 0.40, 'Speaking': 0.45},
        'reference_values': {'Grammar Speaking': 75, 'Speaking': 75},
        'start': date(2026, 1, 2),   # a Friday
        'weeks': 10,
        # level mix: (level 2, level 3, level 4, probation)
        'level_counts': (6, 17, 17, 2),
    },
    'cohort_b': {
        'classes': ['B1', 'B2', 'B3', 'B4'],
        'students_per_class': 7,
        'subjects': [
            'Vocabulary', 'Article HW', 'Debate', 'Persuasive Writing', 'Persuasive Rewriting',
            'Compare-Contrast Writing', 'Compare-Contrast Rewriting', 'Descriptive Writing',
            'Descriptive Rewriting', 'Science HW', 'Science Test', 'Science Presentation',
            'Reading', 'Listening', 'Speaking', 'Novel', 'Speaking Assessment',
            'Article Comprehension', 'Reading HW',
        ],
        'sparse': {
            'Debate': 0.50, 'Persuasive Writing': 0.60, 'Persuasive Rewriting': 0.65,
            'Compare-Contrast Writing': 0.65, 'Compare-Contrast Rewriting': 0.70,
            'Descriptive Writing': 0.65, 'Descriptive Rewriting': 0.70,
            'Science Test': 0.50, 'Science Presentation': 0.60, 'Speaking Assessment': 0.55,
            'Reading': 0.40,
        },
        'reference_values': {'Debate': 75, 'Speaking': 75, 'Speaking Assessment': 75},
        'start': date(2026, 1, 5),   # a Monday; cohort_b/build_master.py splits at 2026-02-14
        'weeks': 12,
        'level_counts': (4, 11, 11, 2),
    },
}

BASE_BLANK_PROB = 0.08
ABSENT_PROB = 0.05          # whole-week absence
LEVEL_MEANS = {2: 90.0, 3: 80.0, 4: 70.0, 'P': 61.0}
REFERENCE_NAME = '*Reference'


def letters(i):
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA', ..."""
    s = ''
    i += 1
    while i > 0:
        i, rem = divmod(i - 1, 26)
        s = chr(ord('A') + rem) + s
    return s


def fmt_date(d):
    """1/3/26 style, matching the pipeline's %m/%d/%y parser."""
    return f"{d.month}/{d.day}/{d.year % 100:02d}"


def build_roster(cfg, rng):
    """Assign invented names and a shuffled level mix across the cohort's classes."""
    n2, n3, n4, npb = cfg['level_counts']
    total = len(cfg['classes']) * cfg['students_per_class']
    assert n2 + n3 + n4 + npb == total, 'level_counts must sum to classes * students_per_class'
    pool = [2] * n2 + [3] * n3 + [4] * n4 + ['P'] * npb
    rng.shuffle(pool)

    roster = []
    idx = 0
    for cls in cfg['classes']:
        for _ in range(cfg['students_per_class']):
            tag = letters(idx)
            roster.append({
                'Class': cls,
                'Korean Name': f'학생{tag}',
                'English Name': f'Student{tag}',
                'Level': pool[idx],
            })
            idx += 1
    return roster


def simulate_scores(cfg, roster, rng):
    """Return {class: {date: [(name, {subject: score|None}), ...]}} including the reference row."""
    subjects = cfg['subjects']
    dates = [cfg['start'] + timedelta(weeks=w) for w in range(cfg['weeks'])]
    mid = (cfg['weeks'] - 1) / 2

    # Per-student latent ability, per-subject offsets, and a slow trend
    params = {}
    for s in roster:
        name = f"{s['English Name']} {s['Korean Name']}"
        base = LEVEL_MEANS[s['Level']] + rng.normal(0, 3.0)
        params[name] = {
            'class': s['Class'],
            'base': base,
            'offsets': {subj: rng.normal(0, 4.0) for subj in subjects},
            'slope': rng.normal(0, 0.35),
        }

    ref_row = {subj: cfg['reference_values'].get(subj, 100) for subj in subjects}

    data = {cls: {} for cls in cfg['classes']}
    for w, d in enumerate(dates):
        for cls in cfg['classes']:
            data[cls][d] = [(REFERENCE_NAME, ref_row)]
        for name, p in params.items():
            if rng.random() < ABSENT_PROB:
                continue
            row = {}
            for subj in subjects:
                blank_prob = cfg['sparse'].get(subj, BASE_BLANK_PROB)
                if rng.random() < blank_prob:
                    row[subj] = None
                    continue
                val = p['base'] + p['offsets'][subj] + p['slope'] * (w - mid) + rng.normal(0, 5.0)
                row[subj] = int(round(min(100.0, max(0.0, val))))
            data[p['class']][d].append((name, row))
    return data, dates


def write_cohort(cohort, cfg, rng):
    out_dir = ROOT / cohort / 'sample_data'
    csv_dir = out_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    roster = build_roster(cfg, rng)
    with open(out_dir / 'student_list.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Class', 'Korean Name', 'English Name', 'Level'])
        writer.writeheader()
        writer.writerows(roster)

    data, dates = simulate_scores(cfg, roster, rng)
    subjects = cfg['subjects']
    for cls in cfg['classes']:
        with open(csv_dir / f'{cls}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Name'] + subjects)
            for d in dates:
                for name, row in data[cls][d]:
                    writer.writerow([fmt_date(d), name] + ['' if row[s] is None else row[s] for s in subjects])
    print(f"{cohort}: {len(roster)} students, {len(cfg['classes'])} classes, {len(dates)} weeks -> {out_dir}")


# ── Placeholder comment templates (cohort_a) ────────────────────────────────
TEMPLATE_SUBJECTS = {
    'Vocabulary': ('어휘', '매주 배운 단어를 소리 내어 복습하는 습관을 권합니다.'),
    'Reading': ('독해', '짧은 지문을 매일 한 편씩 읽고 요약해 보세요.'),
    'Speaking': ('말하기', '배운 표현을 문장으로 말해 보는 연습이 도움이 됩니다.'),
    'Novel': ('소설 읽기', '읽은 내용을 간단히 정리하는 연습을 권합니다.'),
    'Grammar': ('문법', '틀린 문제를 다시 풀어 보며 규칙을 확인해 보세요.'),
    'Science': ('과학 영어', '핵심 용어를 영어로 정리해 두면 도움이 됩니다.'),
}
TEMPLATE_TIERS = {
    'top': '{s} 영역에서 레벨 평균보다 크게 앞서 있습니다.',
    'above_avg': '{s} 영역이 레벨 평균을 조금 웃돌고 있습니다.',
    'average': '{s} 영역이 레벨 평균과 비슷한 수준입니다.',
    'below_avg': '{s} 영역이 레벨 평균보다 조금 낮습니다.',
    'bottom': '{s} 영역이 레벨 평균보다 많이 낮습니다.',
}
TEMPLATE_STATUSES = {
    'stable': '현재 레벨을 안정적으로 유지하고 있습니다.',
    'level_up': '이 영역을 조금 더 보완하면 승급에 도움이 됩니다.',
    'level_down': '레벨 유지를 위해 이 영역의 보완이 필요합니다.',
    'probation': '유예 가능성이 있어 이 영역의 보완이 시급합니다.',
    'expulsion': '전반적인 보완이 시급합니다. 이 영역부터 챙겨 주세요.',
}


def write_comment_templates(path):
    """Write a small generic feedback library: one row per subject x score tier."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject', 'Score_Tier'] + list(TEMPLATE_STATUSES))
        for subject, (label, advice) in TEMPLATE_SUBJECTS.items():
            for tier, tier_text in TEMPLATE_TIERS.items():
                lead = tier_text.format(s=label)
                writer.writerow([subject, tier] + [f'{lead} {advice} {status_text}'
                                                   for status_text in TEMPLATE_STATUSES.values()])
    print(f"comment templates: {len(TEMPLATE_SUBJECTS) * len(TEMPLATE_TIERS)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sample data for both cohorts.")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    for cohort, cfg in COHORTS.items():
        # One RNG stream per cohort so editing one config does not reshuffle the other
        rng = np.random.default_rng([args.seed, sum(map(ord, cohort))])
        write_cohort(cohort, cfg, rng)
    write_comment_templates(ROOT / 'cohort_a' / 'comment_templates.csv')


if __name__ == '__main__':
    main()
