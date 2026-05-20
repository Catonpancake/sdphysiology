"""
Stage 2 - compute_trait_scores.py
===================================
processed_survey CSVs  →  trait_scores.csv  (one row per PID)

Scores computed (with reverse-scoring):
  STAI_Trait   – STAI-Y2 trait anxiety        (main_survey.csv, cols STAI_Y_1..20)
  SADS         – Social Anxiety & Distress     (main_survey.csv, cols SADS_1..28)
  BAI          – Beck Anxiety Inventory        (main_survey.csv, cols BAI_1..21)
  Pre_STAI     – STAI-Y1 state, pre-experiment (pre_survey.csv, cols STAI_Y_1..20)
  IPQ          – Igroup Presence Questionnaire TOTAL, reverse-scored (items 3,4,9,11)
                 + subscales IPQ_General / IPQ_Spatial / IPQ_Involve / IPQ_Realism
  SSQ_Total    – Simulator Sickness (covariate)(experiment_survey.csv, SSQ_*)

Output: {OUT_DIR}/trait_scores.csv
  columns: PID, STAI_Trait, SADS, BAI, Pre_STAI, Post_STAI, Delta_STAI,
           IPQ, IPQ_General, IPQ_Spatial, IPQ_Involve, IPQ_Realism, SSQ_Total

Run:
  python compute_trait_scores.py
  python compute_trait_scores.py --survey_dir /path --out_dir /path
"""

import os
import argparse
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
SURVEY_DIR = r"D:\Labroom\SDPhysiology\Data\processed_survey"
OUT_DIR    = r"D:\Labroom\SDPhysiology\Data\MOMENT_ready_v2"

# ── reverse-scoring specs ─────────────────────────────────────────────────────
# Standard STAI-Y2 (trait): 1-indexed item numbers that are REVERSED
# Non-anxiety (positive affect) items:  1,3,6,7,10,13,14,16,19  →  score = 5 - raw
# Item 9 "I worry too much" is anxiety-PRESENT → NOT reversed (removed from set)
STAI_TRAIT_REVERSE  = {1,3,6,7,10,13,14,16,19}      # 1-based item index
STAI_TRAIT_N_ITEMS  = 20
STAI_TRAIT_MAX      = 4

# Standard STAI-Y1 (state, pre-experiment):
# Non-anxiety items: 1,2,5,8,10,11,15,16,19,20  →  score = 5 - raw
STAI_STATE_REVERSE  = {1,2,5,8,10,11,15,16,19,20}
STAI_STATE_N_ITEMS  = 20
STAI_STATE_MAX      = 4

# SADS (Social Avoidance and Distress Scale, 28 items, binary 0/1)
# Items scored as True=1 normally; these items are reversed (True=0):
# Standard SADS reverse (absence-of-distress items): 1,5,9,11,13,15,17,19,21,22,23,25,26,28
# Data coding: {0=False, 1=True}  →  reverse formula: (0+1)-raw = 1-raw  ✓
# IMPORTANT: SADS_MAX = 0 (not 1), so (max+1)-raw = 1-raw gives correct 0/1 flip
SADS_REVERSE   = {1,5,9,11,13,15,17,19,21,22,23,25,26,28}
SADS_N_ITEMS   = 28
SADS_MAX       = 0    # binary 0/1 scale: reverse = (0+1)-raw = 1-raw

# BAI (Beck Anxiety Inventory, 21 items, 0-3 scale): NO reverse items
BAI_N_ITEMS    = 21

# IPQ (Igroup Presence Questionnaire, 14 items, 1-7 scale in processed CSV)
# Reverse items per Surveys.xlsx IPQ sheet (= standard IPQ SP2/SP3/INV3/REAL1):
#   3 (only pictures), 4 (not sense of being), 9 (no attention to real env),
#   11 (VE real/not real).  Reversed score = (min+max) - raw = 8 - raw.
IPQ_REVERSE      = {3, 4, 9, 11}
IPQ_SCALE_MIN    = 1
IPQ_SCALE_MAX    = 7
# Standard IPQ subscales (1-based item indices):
IPQ_SUBSCALES    = {
    "IPQ_General": [1],                 # general "being there"
    "IPQ_Spatial": [2, 3, 4, 5, 6],     # Spatial Presence
    "IPQ_Involve": [7, 8, 9, 10],       # Involvement
    "IPQ_Realism": [11, 12, 13, 14],    # Experienced Realism
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    """Load processed survey CSV. Row 0 = question text header → skip."""
    df = pd.read_csv(path, dtype=str)
    # if row 0 is question text, drop it
    if df.iloc[0]["ID"] in ["ID", "응답자ID", "nan", "NaN"] or not str(df.iloc[0]["ID"]).isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df.dropna(subset=["ID"])
    df["ID"] = df["ID"].astype(int)
    return df


def _get_scale_cols(df: pd.DataFrame, prefix: str, n_items: int) -> list:
    """Return column names matching prefix_1 .. prefix_N present in df."""
    cols = [f"{prefix}_{i}" for i in range(1, n_items + 1)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing columns for {prefix}: {missing}")
    return [c for c in cols if c in df.columns]


def _score_scale(df: pd.DataFrame, prefix: str, n_items: int,
                 reverse_set: set, max_val: int) -> pd.Series:
    """
    Compute total score with reverse scoring.
    reverse_set: 1-based item indices that should be reversed.
    Reversed score = (max_val + 1) - raw.
    """
    cols = _get_scale_cols(df, prefix, n_items)
    vals = df[cols].apply(pd.to_numeric, errors="coerce").copy()

    for i, col in enumerate(cols, start=1):
        if i in reverse_set:
            vals[col] = (max_val + 1) - vals[col]

    return vals.sum(axis=1)


def _score_ipq(df: pd.DataFrame) -> dict:
    """
    Score IPQ with proper reverse-scoring.
    Reverse items (IPQ_REVERSE) flipped via (min+max) - raw = 8 - raw on 1-7 scale.
    Returns dict: {"IPQ": total, "IPQ_General", "IPQ_Spatial",
                   "IPQ_Involve", "IPQ_Realism"} as Series.
    """
    items = {}
    for i in range(1, 15):
        col = f"IPQ_{i}"
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        if i in IPQ_REVERSE:
            v = (IPQ_SCALE_MIN + IPQ_SCALE_MAX) - v
        items[i] = v

    if not items:
        nan = pd.Series(np.nan, index=df.index)
        out = {"IPQ": nan}
        out.update({name: nan for name in IPQ_SUBSCALES})
        return out

    out = {}
    for name, idxs in IPQ_SUBSCALES.items():
        cols = [items[i] for i in idxs if i in items]
        out[name] = sum(cols) if cols else pd.Series(np.nan, index=df.index)
    out["IPQ"] = sum(items.values())  # total (name kept for backward compat)
    return out


def _score_ssq_total(df: pd.DataFrame) -> pd.Series:
    """SSQ Total = sum of all SSQ subscale items * 3.74."""
    ssq_cols = [c for c in df.columns if c.startswith("SSQ_")]
    if not ssq_cols:
        return pd.Series(np.nan, index=df.index)
    vals = df[ssq_cols].apply(pd.to_numeric, errors="coerce")
    return vals.sum(axis=1) * 3.74


# ── main logic ────────────────────────────────────────────────────────────────

def compute_trait_scores(survey_dir: str, out_dir: str) -> pd.DataFrame:
    main_path = os.path.join(survey_dir, "main_survey.csv")
    pre_path  = os.path.join(survey_dir, "pre_survey.csv")
    exp_path  = os.path.join(survey_dir, "experiment_survey.csv")

    main_df = _load_csv(main_path)
    pre_df  = _load_csv(pre_path)
    exp_df  = _load_csv(exp_path)

    print(f"Loaded: main={len(main_df)}, pre={len(pre_df)}, exp={len(exp_df)} rows")

    # ── trait scores ─────────────────────────────────────────────────────────
    scores = pd.DataFrame({"PID": main_df["ID"].values})

    scores["STAI_Trait"] = _score_scale(
        main_df, "STAI_Y", STAI_TRAIT_N_ITEMS, STAI_TRAIT_REVERSE, STAI_TRAIT_MAX
    ).values

    scores["SADS"] = _score_scale(
        main_df, "SADS", SADS_N_ITEMS, SADS_REVERSE, SADS_MAX
    ).values

    scores["BAI"] = _score_scale(
        main_df, "BAI", BAI_N_ITEMS, set(), 4   # no reverse
    ).values

    # ── pre-experiment state anxiety ─────────────────────────────────────────
    pre_scores = pd.DataFrame({
        "PID": pre_df["ID"].values,
        "Pre_STAI": _score_scale(
            pre_df, "STAI_Y", STAI_STATE_N_ITEMS, STAI_STATE_REVERSE, STAI_STATE_MAX
        ).values,
    })
    scores = scores.merge(pre_scores, on="PID", how="left")

    # ── presence & sickness & post-STAI (from experiment survey) ─────────────
    post_stai = _score_scale(
        exp_df, "STAI_Y", STAI_STATE_N_ITEMS, STAI_STATE_REVERSE, STAI_STATE_MAX
    )
    ipq = _score_ipq(exp_df)  # dict: total + 4 subscales (reverse-scored)
    exp_scores = pd.DataFrame({
        "PID":         exp_df["ID"].values,
        "Post_STAI":   post_stai.values,
        "IPQ":         ipq["IPQ"].values,
        "IPQ_General": ipq["IPQ_General"].values,
        "IPQ_Spatial": ipq["IPQ_Spatial"].values,
        "IPQ_Involve": ipq["IPQ_Involve"].values,
        "IPQ_Realism": ipq["IPQ_Realism"].values,
        "SSQ_Total":   _score_ssq_total(exp_df).values,
    })
    scores = scores.merge(exp_scores, on="PID", how="left")

    # ── delta STAI (reactivity) ───────────────────────────────────────────────
    scores["Delta_STAI"] = scores["Post_STAI"] - scores["Pre_STAI"]

    # ── PID formatting: match anonymized folder naming (001, 002, ...) ────────
    scores["PID_str"] = scores["PID"].apply(lambda x: f"{int(x):03d}")

    print("\nTrait scores summary:")
    print(scores[["STAI_Trait","SADS","BAI","Pre_STAI","Post_STAI","Delta_STAI","IPQ","SSQ_Total"]].describe().round(2))

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "trait_scores.csv")
    scores.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(scores)} rows)")

    return scores


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey_dir", default=SURVEY_DIR)
    parser.add_argument("--out_dir",    default=OUT_DIR)
    args = parser.parse_args()

    print("NOTE: Reverse-scoring follows standard STAI/SADS protocols.")
    print("      Verify against your q_map.xlsx files if results look off.\n")
    compute_trait_scores(args.survey_dir, args.out_dir)


if __name__ == "__main__":
    main()
