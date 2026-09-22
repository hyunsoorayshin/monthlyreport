# Architecture

Two independent cohorts share one shape of pipeline. Each cohort folder is
self-contained: raw inputs in `sample_data/`, two scripts, generated outputs
written next to the scripts (and git-ignored).

```
raw weekly CSVs  +  student_list.csv
        │
        ▼
  build_master.py        merge classes, attach Class/Level/Probation,
        │                compute cutlines from percentiles of overall averages
        ├── Cohort?_Master.csv
        ├── Cohort?_Cutlines.csv
        │   (cohort_b also: CohortB_Master_Prior.csv, rows on/before the period cutoff)
        ▼
  monthly_report.py      per-student PNG, Monthly_Summary.csv, Risk_List.csv
```

## Inputs

### `sample_data/student_list.csv`
Roster and single source of truth for names, classes and levels.

- Columns: `Class, Korean Name, English Name, Level`
- `Level` is `2`, `3`, `4` or `P` (probation). Lower number = higher level.
  `P` is treated as level 4 for statistics and flagged in a `Probation` column.
- A student's key everywhere else is `"<English Name> <Korean Name>"`.

### `sample_data/csv/<class>.csv`
One file per class, one row per student per assessment date.

- Columns: `Date, Name, <subject columns...>`; dates are `m/d/yy`.
- Blank cells mean "not assessed that week" and are skipped, not zeroed.
- Rows whose `Name` starts with `*` are reference rows and are ignored.
- Names not on the roster are skipped with a warning.

Cohort A has ten subject columns; three grammar columns are averaged into a
single "Grammar" score and three science columns into "Science". Cohort B has
nineteen; six writing columns collapse into "Writing" and two science columns
into "Science". Cohort B additionally groups subjects into five radar
categories (Vocabulary, Reading, Listening, Writing, Speaking).

## Cutlines

`build_master.py` computes each student's overall average (mean of every
numeric score cell), then takes percentiles of that distribution:

| boundary | percentile |
|---|---|
| Lv2 | 85th |
| Lv3 | 45th |
| Lv4 / probation | 5th |

Cohort A writes the plain percentiles. Cohort B bootstraps the percentile
(2000 resamples, seed 42) and writes a `Cutline_Low / Cutline / Cutline_High`
zone (40th-60th percentile of the bootstrap estimates, capped at +/-2 points)
so the report can shade a "borderline" band.

## Level-change heuristic

`calculate_level_probability` in each `monthly_report.py` compares the
student's overall average to the relevant cutline and turns the point margin
into a display percentage with a hand-set saturating curve; see the README for
the exact formula and constants. It is not a fitted model.

## Reports

- **Cohort A** (`cohort_a/monthly_report.py`): box plot of each level's
  distribution with the student marked, cutline lines, a status box with the
  heuristic percentage, the weakest subject relative to the level average,
  and a comment looked up from `comment_templates.csv` by
  `(subject, score tier, status)`.
- **Cohort B** (`cohort_b/monthly_report.py`): the same box plot with shaded
  cutline zones, a five-axis radar (student vs level average, prior period
  overlaid when `CohortB_Master_Prior.csv` exists), a category table, and a
  weekly total-average trend with a least-squares line and dip annotations.
- **`levelup_report.py`** (repo root): for one Cohort B student, a radar on a
  0-5 scale against the next level up, with a per-subject signal
  (Rising / Potential / Volatile / Plateau / Declining) derived from the
  least-squares slope and coefficient of variation of their weekly scores.

## Synthetic data

`make_sample_data.py` regenerates everything under `sample_data/` and the
placeholder `comment_templates.csv` deterministically from a seed. No file in
this repository is derived from real student records.
