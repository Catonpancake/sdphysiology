# build_paper_assets.py
# -*- coding: utf-8 -*-
"""
논문용 결과 패키지(표/텍스트) + 그림 세트 생성 스크립트
 - 메인 표(HV=none, seed=20), 민감도 표(y_train/x_variance)
 - 테스트 코호트(10명) 명시 텍스트 템플릿
 - 네거티브 컨트롤(Log 파일에서 자동 파싱)
 - 그림:
     (1) HV 모드별 Test R² (mean ± SE) 막대그래프
     (2) 네거티브 컨트롤 막대그래프 (옵션)
     (3) Ablation Top-K (기본 15개) 막대그래프 — HV별 1장씩
"""

import os, json, math, re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------- 설정 ---------------
IN_DIR = Path(".")         # 입력 파일 위치 (필요시 수정)
OUT_DIR = Path("./paper_assets")  # 출력 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = IN_DIR / "split_fixed_test.json"
SUMMARY_FILES = {
    "none": IN_DIR / "summary_none.json",
    "x_variance": IN_DIR / "summary_x_variance.json",
    "y_train": IN_DIR / "summary_y_train.json",
}
ABLATION_FILES = {
    "none": IN_DIR / "ablation_none.csv",
    "x_variance": IN_DIR / "ablation_x_variance.csv",
    "y_train": IN_DIR / "ablation_y_train.csv",
}
NEG_LOG_FILE = IN_DIR / "negative_controls.log"  # 옵션(없어도 됨)

TOPK_ABLATION = 15  # Ablation 상위 k개
FIG_DPI = 300

# --------------- 유틸 ---------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def se_from_std(std, n):
    if n is None or n <= 1:
        return None
    return std / math.sqrt(n)

def parse_negative_controls(log_path: Path):
    """
    로그 파일에서 네거티브 컨트롤 수치 추출.
    기대 패턴:
      [QC] Label-shift R² ≈ -0.0123
      [QC] Time-order destroyed R² ≈ 0.0345
    """
    if not log_path.exists():
        return None

    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    label_shift = None
    time_destroy = None

    m1 = re.search(r"Label-shift R²\s*≈\s*([\-+]?\d*\.?\d+)", txt)
    if m1:
        label_shift = float(m1.group(1))
    m2 = re.search(r"Time-order destroyed R²\s*≈\s*([\-+]?\d*\.?\d+)", txt)
    if m2:
        time_destroy = float(m2.group(1))

    return {"label_shift_r2": label_shift, "time_destroy_r2": time_destroy}

def to_md_table(df: pd.DataFrame) -> str:
    """DataFrame → Markdown table 문자열"""
    return df.to_markdown(index=False)

def to_latex_table(df: pd.DataFrame) -> str:
    """DataFrame → LaTeX tabular 문자열"""
    return df.to_latex(index=False, float_format="%.4f", escape=False)

# --------------- 1) 결과 패키지 (표/텍스트) ---------------
def build_results_package():
    # a) 테스트 코호트 목록
    split = load_json(SPLIT_FILE)
    test_pids = split.get("test_pids", [])
    train_pids = split.get("train_pids", [])
    val_pids = split.get("val_pids", [])
    seed_used = split.get("seed", None)

    # b) 3개 요약 불러오기
    rows = []
    for mode, fpath in SUMMARY_FILES.items():
        js = load_json(fpath)
        mean_r2 = js.get("seed_mean_r2", None)
        std_r2  = js.get("seed_std_r2", None)
        n_seeds = js.get("n_seeds", None)
        se_r2   = se_from_std(std_r2, n_seeds)
        feat_count = js.get("feat_count", None)
        best_params = js.get("best_params", {})
        num_filters = best_params.get("num_filters", None)
        kernel_size = best_params.get("kernel_size", None)
        dropout = best_params.get("dropout", None)
        lr = best_params.get("learning_rate", None)
        rows.append({
            "HV_MODE": mode,
            "Test R² (mean)": mean_r2,
            "Test R² (SE)": se_r2,
            "Seeds": n_seeds,
            "Features Used": feat_count,
            "num_filters": num_filters,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "learning_rate": lr
        })

    df_summary = pd.DataFrame(rows).sort_values(by="HV_MODE")
    df_summary_path_csv = OUT_DIR / "summary_hv_modes.csv"
    df_summary.to_csv(df_summary_path_csv, index=False, encoding="utf-8-sig")

    # c) 메인 표(논문): HV=none만 추려서 깔끔 표
    df_main = df_summary[df_summary["HV_MODE"] == "none"].copy()
    df_main.rename(columns={
        "Test R² (mean)": "Test R² (mean)",
        "Test R² (SE)": "SE",
        "Features Used": "Feat",
        "num_filters": "Filters",
        "kernel_size": "Kernel",
        "learning_rate": "LR"
    }, inplace=True)
    df_main = df_main[["HV_MODE", "Test R² (mean)", "SE", "Seeds", "Feat", "Filters", "Kernel", "dropout", "LR"]]
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    (OUT_DIR / "texts").mkdir(exist_ok=True)

    # 저장: CSV/Markdown/LaTeX
    df_main.to_csv(OUT_DIR / "tables" / "main_table_hv_none.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "tables" / "main_table_hv_none.md").write_text(to_md_table(df_main), encoding="utf-8")
    (OUT_DIR / "tables" / "main_table_hv_none.tex").write_text(to_latex_table(df_main), encoding="utf-8")

    # d) 민감도 표(3모드)
    (OUT_DIR / "tables" / "sensitivity_hv_all.csv").write_text(df_summary.to_csv(index=False), encoding="utf-8")
    (OUT_DIR / "tables" / "sensitivity_hv_all.md").write_text(to_md_table(df_summary), encoding="utf-8")
    (OUT_DIR / "tables" / "sensitivity_hv_all.tex").write_text(to_latex_table(df_summary), encoding="utf-8")

    # e) 네거티브 컨트롤(있으면)
    neg = parse_negative_controls(NEG_LOG_FILE)
    if neg:
        df_neg = pd.DataFrame([
            {"Control": "Label-shift", "Test R²": neg.get("label_shift_r2")},
            {"Control": "Time-order destroyed", "Test R²": neg.get("time_destroy_r2")},
        ])
        df_neg.dropna().to_csv(OUT_DIR / "tables" / "negative_controls.csv", index=False, encoding="utf-8-sig")
        (OUT_DIR / "tables" / "negative_controls.md").write_text(to_md_table(df_neg.dropna()), encoding="utf-8")
        (OUT_DIR / "tables" / "negative_controls.tex").write_text(to_latex_table(df_neg.dropna()), encoding="utf-8")

    # f) 텍스트 템플릿(Methods/Results 문구 뼈대)
    methods_txt = f"""\
[Methods – Evaluation Setup (Template)]
• Participant-disjoint split (fixed cohort): test={len(test_pids)} PIDs (seed={seed_used}), val={len(val_pids)} PIDs, train={len(train_pids)} PIDs.
• Hyperparameter tuning and feature selection used only the validation set (external validation; no internal split).
• Final reporting used a seed ensemble of {int(df_summary['Seeds'].iloc[0])} with deterministic settings; TF32 disabled explicitly.
• Lag = OFF; window format = (N,T,C); GAP between windows = 10; target centering via train-only hierarchical means (pid, scene -> pid -> global).

[Test cohort PIDs]
{", ".join(map(str, test_pids))}
"""
    (OUT_DIR / "texts" / "methods_template.txt").write_text(methods_txt, encoding="utf-8")

    # 결과 텍스트(모드별 성능 요약)
    lines = ["[Results – HV sensitivity (seed mean ± SE)]"]
    for _, r in df_summary.iterrows():
        m = r["Test R² (mean)"]
        se = r["Test R² (SE)"]
        s = r["Seeds"]
        lines.append(f"- HV={r['HV_MODE']}: R² = {m:.4f} ± {se:.4f} (seeds={int(s)})")
    if neg:
        ls = neg.get("label_shift_r2")
        td = neg.get("time_destroy_r2")
        lines.append(f"- Negative controls: Label-shift R² ≈ {ls}, Time-order destroyed R² ≈ {td}")
    results_txt = "\n".join(lines)
    (OUT_DIR / "texts" / "results_summary.txt").write_text(results_txt, encoding="utf-8")

    print("[OK] Results package saved under:", OUT_DIR)

# --------------- 2) 그림 세트 ---------------
def plot_hv_bar(df_summary: pd.DataFrame):
    # 막대: mean, 오차막대: SE
    modes = df_summary["HV_MODE"].tolist()
    means = df_summary["Test R² (mean)"].tolist()
    ses   = df_summary["Test R² (SE)"].tolist()

    plt.figure()
    x = np.arange(len(modes))
    plt.bar(x, means, yerr=ses, capsize=4)
    plt.xticks(x, modes)
    plt.ylabel("Test R² (mean ± SE)")
    plt.title("HV Mode Sensitivity (Test)")
    plt.tight_layout()
    (OUT_DIR / "figs").mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / "figs" / "hv_modes_test_r2.png", dpi=FIG_DPI)
    plt.close()

def plot_negative_controls(neg_dict):
    if not neg_dict:
        return
    labels = []
    values = []
    if neg_dict.get("label_shift_r2") is not None:
        labels.append("Label-shift")
        values.append(neg_dict["label_shift_r2"])
    if neg_dict.get("time_destroy_r2") is not None:
        labels.append("Time-order destroyed")
        values.append(neg_dict["time_destroy_r2"])
    if not labels:
        return

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Test R²")
    plt.title("Negative Controls")
    plt.tight_layout()
    (OUT_DIR / "figs").mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / "figs" / "negative_controls.png", dpi=FIG_DPI)
    plt.close()

def plot_ablation_topk(ablation_path: Path, mode_name: str, topk: int = 15):
    df = safe_read_csv(ablation_path)
    if df is None:
        return
    # baseline 탐색
    # 기대 컬럼: ["feature_removed", "val_r2"], baseline row: "None (baseline)"
    assert "feature_removed" in df.columns and "val_r2" in df.columns, "Ablation CSV에 필요한 컬럼이 없습니다."
    base_row = df[df["feature_removed"].str.contains("None", case=False, na=False)]
    assert len(base_row) == 1, "baseline 행을 찾지 못했습니다."
    r2_base = float(base_row["val_r2"].iloc[0])

    df2 = df.copy()
    df2 = df2[~df2["feature_removed"].str.contains("None", case=False, na=False)]
    df2["drop_in_r2"] = r2_base - df2["val_r2"]
    df2 = df2.sort_values("drop_in_r2", ascending=False).head(topk)

    plt.figure()
    y_labels = df2["feature_removed"].tolist()[::-1]
    x_vals   = df2["drop_in_r2"].tolist()[::-1]
    y = np.arange(len(y_labels))
    plt.barh(y, x_vals)
    plt.yticks(y, y_labels)
    plt.xlabel("Δ R² on VAL (baseline - removed)")
    plt.title(f"Ablation Top-{topk} (HV={mode_name})")
    plt.tight_layout()
    (OUT_DIR / "figs").mkdir(exist_ok=True)
    out = OUT_DIR / "figs" / f"ablation_top{topk}_{mode_name}.png"
    plt.savefig(out, dpi=FIG_DPI)
    plt.close()

def build_figures():
    # 요약 불러와서 막대+오차막대
    rows = []
    for mode, fpath in SUMMARY_FILES.items():
        js = load_json(fpath)
        rows.append({
            "HV_MODE": mode,
            "Test R² (mean)": js.get("seed_mean_r2", None),
            "Test R² (SE)": se_from_std(js.get("seed_std_r2", None), js.get("n_seeds", None)),
        })
    df_summary = pd.DataFrame(rows).sort_values(by="HV_MODE")
    plot_hv_bar(df_summary)

    # 네거티브 컨트롤
    neg = parse_negative_controls(NEG_LOG_FILE)
    plot_negative_controls(neg)

    # Ablation Top-K (모드별)
    for mode, path in ABLATION_FILES.items():
        if Path(path).exists():
            plot_ablation_topk(Path(path), mode, topk=TOPK_ABLATION)

    print("[OK] Figures saved under:", OUT_DIR / "figs")

# --------------- main ---------------
if __name__ == "__main__":
    build_results_package()
    build_figures()
