#!/usr/bin/env python3
"""
Likert Judge-Agreement Analysis: Per Evaluated Model vs. Pseudo-Groundtruth

For each property (conciseness, divergence, …) and each evaluated model
(claude, chatgpt, grok, …), computes pairwise weighted Cohen's Kappa between
judge models on the Likert scores they assign when judging that model's responses
against the pseudo-groundtruth (gemini_analysis).

Compared to the original likert_kappa_analysis.py the sole difference is that
items are NOT pooled across all evaluated models — kappa is computed separately
per evaluated model so you can see which model's ratings generate the most /
least inter-judge agreement.

Directory layout expected:
    base_dir/
      {judge_name}_{property}/
        Eval_Judge_{judge_name}_{evaluated_model}_vs_gt_*.json

Outputs per (property, evaluated_model):
  - kappa_{property}_{evaluated_model}.png  — lower-triangle judge kappa heatmap
  - distributions_{property}_{evaluated_model}.png — score distributions by judge
  - vs_gt_kappa_results.csv  — all kappa numbers in one table
"""

import argparse
import os
import glob
import json
import re
import math
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from sklearn.metrics import cohen_kappa_score

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino", "Georgia", "STIXGeneral"],
})


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def extract_score(text):
    """Extract leading Likert score (1–5) from a judgment result string."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    json_match = re.match(r'^\s*\{[^}]*"rating"\s*:\s*([1-5])[^}]*\}', text)
    if json_match:
        return int(json_match.group(1))
    digit_match = re.match(r"^\s*([1-5])", text)
    return int(digit_match.group(1)) if digit_match else None


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_vs_gt_results(base_dir):
    """
    Load Likert scores from the xVerify CoT_eval_results directory.

    Returns:
        data[property][evaluated_model][judge_name][item_key] = score
        where item_key = (question, llm_output[:500])
    """
    # property → evaluated_model → judge_name → {item_key: score}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    if not os.path.isdir(base_dir):
        print(f"  WARNING: base_dir does not exist: {base_dir}")
        return data

    for entry in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, entry)
        if not os.path.isdir(subdir):
            continue
        if "_" not in entry:
            print(f"  Skipping unrecognised subdir: {entry}")
            continue

        # "{judge_name}_{property}" — property is last underscore-delimited token
        last_us   = entry.rfind("_")
        judge_name = entry[:last_us]
        prop_name  = entry[last_us + 1:]

        for json_path in sorted(glob.glob(os.path.join(subdir, "*.json"))):
            basename = os.path.basename(json_path)

            # Eval_Judge_{judge_name}_{evaluated_model}_vs_gt_*.json
            # evaluated_model may contain underscores (nova_lite, qwen3_vl)
            prefix = f"Eval_Judge_{judge_name}_"
            if not basename.startswith(prefix):
                continue
            rest     = basename[len(prefix):]
            vs_match = re.search(r"_vs_gt_", rest)
            if not vs_match:
                continue
            evaluated_model = rest[: vs_match.start()]

            try:
                with open(json_path) as f:
                    content = json.load(f)
            except Exception as e:
                print(f"  ERROR reading {json_path}: {e}")
                continue

            dest   = data[prop_name][evaluated_model][judge_name]
            loaded = 0
            for item in content.get("results", []):
                question   = item.get("question", "")
                llm_output = item.get("llm_output", "")
                item_key   = (question, llm_output[:500])

                score = None
                for key, value in item.items():
                    if key.endswith("_judgment_result"):
                        score = extract_score(value)
                        break

                if score is not None and item_key not in dest:
                    dest[item_key] = score
                    loaded += 1

            print(f"    [{prop_name}] judge={judge_name}, model={evaluated_model}: "
                  f"{loaded} items loaded")

    return data


def load_vs_gt_results_multi(base_dirs):
    """
    Load and merge Likert scores from multiple xVerify CoT_eval_results
    directories.  Scores are combined per (property, eval_model, judge, item_key)
    — later entries from the same item_key are ignored (first-seen wins).

    Returns the same nested structure as load_vs_gt_results.
    """
    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for base_dir in base_dirs:
        print(f"\n[loading] {base_dir}")
        partial = load_vs_gt_results(base_dir)
        for prop, models in partial.items():
            for ev_model, judges in models.items():
                for judge, items in judges.items():
                    dest = merged[prop][ev_model][judge]
                    for item_key, score in items.items():
                        if item_key not in dest:
                            dest[item_key] = score
    return merged


# ---------------------------------------------------------------------------
# Alignment: find common items across all judges for a given model
# ---------------------------------------------------------------------------

def align_judges_for_model(judge_item_scores):
    """
    Given {judge_name: {item_key: score}}, return aligned lists
    {judge_name: [score, ...]} for the intersection of item keys.
    """
    if not judge_item_scores:
        return {}

    judge_names = list(judge_item_scores.keys())
    common_keys = set(judge_item_scores[judge_names[0]].keys())
    for jname in judge_names[1:]:
        common_keys &= set(judge_item_scores[jname].keys())

    if not common_keys:
        return {}

    sorted_keys = sorted(common_keys)
    return {
        jname: [judge_item_scores[jname][k] for k in sorted_keys]
        for jname in judge_names
    }


# ---------------------------------------------------------------------------
# Kappa computation (pairwise between judges)
# ---------------------------------------------------------------------------

def compute_pairwise_kappa(aligned_scores):
    """
    Pairwise weighted Cohen's Kappa (quadratic) from {judge: [score, ...]}.
    Returns {(judge1, judge2): kappa_float}.
    """
    judge_names = sorted(aligned_scores.keys())
    kappa_results = {}
    for j1, j2 in itertools.combinations(judge_names, 2):
        y1, y2 = aligned_scores[j1], aligned_scores[j2]
        n = min(len(y1), len(y2))
        if n < 2:
            continue
        try:
            k = cohen_kappa_score(y1[:n], y2[:n], weights="quadratic")
            kappa_results[(j1, j2)] = k
        except Exception as e:
            print(f"    WARNING kappa({j1},{j2}): {e}")
    return kappa_results


def compute_fleiss_kappa(aligned_scores):
    """Fleiss' Kappa from {judge: [score, ...]} (already aligned)."""
    judge_names = sorted(aligned_scores.keys())
    if len(judge_names) < 2:
        return None

    min_len = min(len(aligned_scores[j]) for j in judge_names)
    if min_len == 0:
        return None

    ratings_matrix = np.array(
        [aligned_scores[j][:min_len] for j in judge_names]
    ).T  # (n_items, n_raters)

    n_items, n_raters = ratings_matrix.shape
    n_categories = 5

    cat_counts = np.zeros((n_items, n_categories))
    for i in range(n_items):
        for rating in ratings_matrix[i]:
            cat_counts[i, int(rating) - 1] += 1

    P_i     = (np.sum(cat_counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar   = np.mean(P_i)
    P_j     = np.sum(cat_counts, axis=0) / (n_items * n_raters)
    P_e_bar = np.sum(P_j ** 2)

    if P_e_bar == 1:
        return None
    return (P_bar - P_e_bar) / (1 - P_e_bar)


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def interpret_kappa(kappa):
    if kappa < 0:       return "Poor"
    elif kappa < 0.20:  return "Slight"
    elif kappa < 0.40:  return "Fair"
    elif kappa < 0.60:  return "Moderate"
    elif kappa < 0.80:  return "Substantial"
    else:               return "Almost Perfect"


def clean_name(name):
    """Strip trailing date/version suffixes (e.g. -20251219)."""
    return re.sub(r"-\d{8,}$", "", name)


# ---------------------------------------------------------------------------
# Plotting: kappa heatmap (lower triangle, judge vs judge)
# ---------------------------------------------------------------------------

def plot_kappa_heatmap(kappa_results, judge_names, title, save_path=None):
    """Lower-triangle pairwise kappa heatmap between judges."""
    n = len(judge_names)
    if n < 2:
        print(f"  Not enough judges for kappa plot: {title}")
        return

    kappa_matrix = np.full((n, n), np.nan)
    for i, j in itertools.combinations(range(n), 2):
        m1, m2 = judge_names[i], judge_names[j]
        val = kappa_results.get((m1, m2), kappa_results.get((m2, m1), np.nan))
        kappa_matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(max(14, n * 3.5), max(10, n * 3.2)))

    muted_colors = ["#FFFDE7", "#FFF5CC", "#FFE0B2", "#FFCC80", "#FFAB91"]
    cmap_obj = mcolors.LinearSegmentedColormap.from_list("muted_cols", muted_colors)

    plotted_vals = [
        kappa_matrix[i, j]
        for i in range(1, n) for j in range(i)
        if not np.isnan(kappa_matrix[i, j])
    ]
    data_vmin = min(plotted_vals) if plotted_vals else -1.0
    data_vmax = max(plotted_vals) if plotted_vals else  1.0
    vmin_log  = max(data_vmin, 1e-3)
    vmax_log  = max(data_vmax, 1e-3)
    if vmin_log >= vmax_log:
        vmin_log, vmax_log = 1e-3, 1.0

    norm = mcolors.LogNorm(vmin=vmin_log, vmax=vmax_log)

    for i in range(1, n):
        for j in range(i):
            value = kappa_matrix[i, j]
            if not np.isnan(value):
                color = cmap_obj(norm(max(value, 1e-3)))
                rect  = plt.Rectangle((j, i - 1), 1, 1, facecolor=color, edgecolor="gray")
                ax.add_patch(rect)
                ax.text(j + 0.5, i - 0.5, f"{value:.3f}",
                        ha="center", va="center",
                        color="black", fontsize=36, weight="bold")

    display = [clean_name(jn) for jn in judge_names]
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, n - 1)
    ax.set_xticks(np.arange(n - 1) + 0.5)
    ax.set_yticks(np.arange(n - 1) + 0.5)
    ax.set_xticklabels(display[:-1], rotation=45, ha="right", fontsize=28)
    ax.set_yticklabels(display[1:], fontsize=28)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=32, weight="bold", pad=20)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    if vmin_log < vmax_log:
        log_ticks = np.logspace(math.log10(vmin_log), math.log10(vmax_log), 5)
        cbar.set_ticks(log_ticks)
    cbar.set_label("Weighted Cohen's Kappa", fontsize=26)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    cbar.ax.minorticks_off()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved kappa heatmap → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plotting: score distributions by judge
# ---------------------------------------------------------------------------

def plot_score_distributions(aligned_scores, title,
                              save_dir=None, save_name=None):
    """Histogram + box plot of Likert scores per judge."""
    judge_names = sorted(aligned_scores.keys())
    if not judge_names:
        return

    fig, axes = plt.subplots(1, 2, figsize=(24, 9))

    ax1 = axes[0]
    for judge in judge_names:
        ax1.hist(aligned_scores[judge], alpha=0.55,
                 label=clean_name(judge), bins=5, range=(0.5, 5.5))
    ax1.set_xlabel("Likert Score", fontsize=18)
    ax1.set_ylabel("Frequency", fontsize=18)
    ax1.set_title(title, fontsize=18)
    ax1.legend(fontsize=14)
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.tick_params(axis="both", labelsize=15)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    data_for_box = [aligned_scores[j] for j in judge_names]
    bp = ax2.boxplot(data_for_box,
                     tick_labels=[clean_name(j) for j in judge_names],
                     patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax2.set_xlabel("Judge Model", fontsize=18)
    ax2.set_ylabel("Likert Score", fontsize=18)
    ax2.set_title(title, fontsize=18)
    ax2.set_ylim(0.5, 5.5)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=14)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, save_name or "distributions.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved distributions → {path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(prop, evaluated_model, aligned, kappa_results, fleiss_kappa):
    w = 72
    print(f"\n{'='*w}")
    print(f"  Property: {prop.upper()}  |  Evaluated model: {evaluated_model}")
    print(f"{'='*w}")

    print("\n  Data (aligned items only):")
    for judge in sorted(aligned.keys()):
        scores = aligned[judge]
        print(f"    {clean_name(judge):35s}  N={len(scores):5d}  "
              f"mean={np.mean(scores):.3f}  sd={np.std(scores):.3f}")

    if fleiss_kappa is not None:
        interp = interpret_kappa(fleiss_kappa)
        print(f"\n  Fleiss' κ (all judges) = {fleiss_kappa:.4f}  ({interp})")

    print("\n  Pairwise Weighted Cohen's Kappa (judges):")
    for (j1, j2), k in sorted(kappa_results.items(), key=lambda x: -x[1]):
        print(f"    {clean_name(j1):30s} vs {clean_name(j2):30s}  "
              f"κ = {k:7.4f}  ({interpret_kappa(k)})")

    if kappa_results:
        avg = np.mean(list(kappa_results.values()))
        print(f"\n  Avg pairwise κ = {avg:.4f}")


# ---------------------------------------------------------------------------
# Combined (pooled across eval models) kappa plot
# ---------------------------------------------------------------------------

def plot_combined_kappa(raw, prop, output_dir):
    """
    Pool scores from ALL evaluated models for a given facet and produce
    a single kappa heatmap.  Scores for the same (judge, item_key) pair
    coming from different eval models are concatenated — item_key already
    encodes the llm_output prefix so there is no cross-model collision.
    """
    # {judge_name: {item_key: score}}  — merging across all eval models
    pooled_judge_items = defaultdict(dict)
    n_models = 0
    for ev_model, judge_dict in raw[prop].items():
        n_models += 1
        for judge, item_scores in judge_dict.items():
            for item_key, score in item_scores.items():
                # prefix item_key with eval_model to keep items distinct
                combined_key = (ev_model,) + item_key
                if combined_key not in pooled_judge_items[judge]:
                    pooled_judge_items[judge][combined_key] = score

    aligned = align_judges_for_model(pooled_judge_items)
    if len(aligned) < 2:
        print(f"  [combined {prop}] Not enough judges. Skipping combined plot.")
        return

    n_common = len(next(iter(aligned.values())))
    if n_common < 2:
        print(f"  [combined {prop}] Too few common items ({n_common}). Skipping.")
        return

    kappa_results = compute_pairwise_kappa(aligned)
    fleiss_kappa  = compute_fleiss_kappa(aligned)

    judge_names = sorted(aligned.keys())
    models_str  = ", ".join(sorted(raw[prop].keys()))
    plot_title  = (
        f"Judge Agreement (Weighted Kappa)\n"
        f"Property: {prop.capitalize()}"
        # f"Combined across {n_models} models: {models_str}\n"
        # f"N={n_common:,} items"
    )
    save_path = os.path.join(output_dir, f"kappa_combined_{prop}.png")
    plot_kappa_heatmap(kappa_results, judge_names, plot_title, save_path)

    if fleiss_kappa is not None:
        print(f"  [combined {prop}] Fleiss' κ = {fleiss_kappa:.4f}  "
              f"({interpret_kappa(fleiss_kappa)})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Likert inter-judge kappa analysis (vs pseudo-groundtruth)."
    )
    parser.add_argument(
        "--base_dirs", "-b",
        nargs="+",
        default=None,
        metavar="DIR",
        help="One or more xVerify CoT_eval_results directories to load and merge. "
             "If omitted, falls back to the hard-coded XVERIFY_BASE_DIR.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        metavar="DIR",
        help="Directory to write plots and CSV into. "
             "Defaults to the hard-coded OUTPUT_DIR.",
    )
    parser.add_argument(
        "--combined_kappa",
        action="store_true",
        help="Additionally produce one pooled kappa heatmap per facet, "
             "combining scores across all evaluated models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION (used when no CLI args are supplied)
    # ========================================================================
    # XVERIFY_BASE_DIR = "./CoT_eval_results_no_gemini"
    # OUTPUT_DIR       = "likert_vs_gt_results"

    XVERIFY_BASE_DIR = "./CoT_eval_results/"
    OUTPUT_DIR       = "likert_vs_gt_results_qwen_intern_llava"
    # ========================================================================

    args = _parse_args()

    base_dirs  = args.base_dirs  if args.base_dirs   else [XVERIFY_BASE_DIR]
    output_dir = args.output_dir if args.output_dir  else OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    print(f"[vs-GT mode] Loading from: {base_dirs}\n")
    if len(base_dirs) == 1:
        raw = load_vs_gt_results(base_dirs[0])
    else:
        raw = load_vs_gt_results_multi(base_dirs)

    if not raw:
        print("ERROR: No data found. Check base_dirs and directory naming.")
        exit(1)

    properties = sorted(raw.keys())
    print(f"\nProperties found: {properties}\n")

    all_rows = []

    for prop in properties:
        print(f"\n{'#'*70}")
        print(f"#  PROPERTY: {prop.upper()}")
        print(f"{'#'*70}")

        evaluated_models = sorted(raw[prop].keys())
        print(f"  Evaluated models: {evaluated_models}\n")

        for ev_model in evaluated_models:
            judge_item_scores = raw[prop][ev_model]  # {judge: {item_key: score}}

            # Align by common items across all judges
            aligned = align_judges_for_model(judge_item_scores)

            if len(aligned) < 2:
                print(f"  [{ev_model}] Not enough judges with overlapping items. Skipping.")
                continue

            n_common = len(next(iter(aligned.values())))
            if n_common < 2:
                print(f"  [{ev_model}] Too few common items ({n_common}). Skipping.")
                continue

            # Kappa between judges
            kappa_results = compute_pairwise_kappa(aligned)
            fleiss_kappa  = compute_fleiss_kappa(aligned)

            print_summary(prop, ev_model, aligned, kappa_results, fleiss_kappa)

            # --- kappa heatmap ---
            judge_names = sorted(aligned.keys())
            kappa_save  = os.path.join(output_dir, f"kappa_{prop}_{ev_model}.png")
            plot_title  = (
                f"Judge Agreement (Weighted Kappa) — {prop.capitalize()}\n"
                f"Evaluated model: {ev_model}"
            )
            plot_kappa_heatmap(kappa_results, judge_names, plot_title, kappa_save)

            # --- score distributions ---
            dist_title = (
                f"Score Distributions by Judge — {prop.capitalize()}\n"
                f"Evaluated model: {ev_model}"
            )
            plot_score_distributions(aligned, dist_title, output_dir,
                                     save_name=f"distributions_{prop}_{ev_model}.png")

            # --- accumulate CSV rows ---
            for (j1, j2), k in kappa_results.items():
                all_rows.append({
                    "Property":        prop,
                    "Evaluated_Model": ev_model,
                    "Judge_1":         j1,
                    "Judge_2":         j2,
                    "Weighted_Cohen_Kappa": round(k, 6),
                    "Interpretation":  interpret_kappa(k),
                    "N_items":         n_common,
                })
            if fleiss_kappa is not None:
                all_rows.append({
                    "Property":        prop,
                    "Evaluated_Model": ev_model,
                    "Judge_1":         "ALL JUDGES",
                    "Judge_2":         "OVERALL (Fleiss)",
                    "Weighted_Cohen_Kappa": round(fleiss_kappa, 6),
                    "Interpretation":  f"Fleiss' Kappa: {interpret_kappa(fleiss_kappa)}",
                    "N_items":         n_common,
                })

        # --- combined kappa plot for this facet (opt-in) ---
        if args.combined_kappa:
            print(f"\n  [combined] Generating pooled kappa plot for {prop} …")
            plot_combined_kappa(raw, prop, output_dir)

    # Save CSV
    if all_rows:
        csv_path = os.path.join(output_dir, "vs_gt_kappa_results.csv")
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"\nSaved all kappa results → {csv_path}")

    print(f"\n✓ Analysis complete!  Results in: {output_dir}/")
    print("  Per (property, evaluated_model):")
    for prop in properties:
        for ev_model in sorted(raw[prop].keys()):
            print(f"    kappa_{prop}_{ev_model}.png")
            print(f"    distributions_{prop}_{ev_model}.png")
    if args.combined_kappa:
        for prop in properties:
            print(f"    kappa_combined_{prop}.png")
    print("    vs_gt_kappa_results.csv")
