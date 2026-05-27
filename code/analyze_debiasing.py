"""Analyze debiasing experiment results and generate comparison table."""

import os
import json
import glob
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
from metrics import bsi_categorical, bootstrap_bsi
from biases import DECOY, FRAMING, PRIMACY_RECENCY

BIASES = {
    "decoy_effect": DECOY,
    "framing_effect": FRAMING,
    "primacy_recency": PRIMACY_RECENCY,
}


def load_result(bias_name, model_name, strategy):
    """Load a single result file."""
    filename = f"{bias_name}_{model_name}_{strategy}.json"
    filepath = os.path.join(config.RAW_RESPONSES_DIR, filename)

    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_bsi_for_result(result, bias_def):
    """Compute BSI for a single result."""
    if result is None:
        return None, None

    bsi_values = []

    for variant in result["variants"]:
        ctrl = np.array(variant["control_parsed"])
        treat = np.array(variant["treatment_parsed"])

        if len(ctrl) < 10 or len(treat) < 10:
            continue

        # Find bias_option for this variant
        bias_option = None
        for v in bias_def["variants"]:
            if v["id"] == variant["variant_id"]:
                bias_option = v.get("bias_option")
                break

        if bias_option is None:
            continue

        bsi = bsi_categorical(ctrl, treat, bias_option)
        bsi_values.append(bsi)

    if not bsi_values:
        return None, None

    mean_bsi = np.mean(bsi_values)
    std_bsi = np.std(bsi_values) if len(bsi_values) > 1 else 0

    return mean_bsi, std_bsi


def main():
    model = "llama-3.1-8b"
    strategies = ["baseline", "D1_cot", "D3_counter", "D4_role"]

    print("=" * 70)
    print("DEBIASING ANALYSIS: Llama 3.1 8B")
    print("=" * 70)

    results = []

    for bias_name, bias_def in BIASES.items():
        row = {"bias": bias_name}

        for strategy in strategies:
            result = load_result(bias_name, model, strategy)
            bsi, std = compute_bsi_for_result(result, bias_def)

            if bsi is not None:
                row[strategy] = f"{bsi:.3f}"
                row[f"{strategy}_val"] = bsi
            else:
                row[strategy] = "--"
                row[f"{strategy}_val"] = None

        # Compute best reduction
        if row.get("baseline_val") is not None:
            baseline = row["baseline_val"]
            best_reduction = 0
            best_strategy = None

            for s in ["D1_cot", "D3_counter", "D4_role"]:
                if row.get(f"{s}_val") is not None:
                    reduction = baseline - row[f"{s}_val"]
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_strategy = s

            if best_strategy:
                row["best_delta"] = f"-{best_reduction:.2f} ({best_strategy})"
            else:
                row["best_delta"] = "--"
        else:
            row["best_delta"] = "--"

        results.append(row)

    # Print table
    print(f"\n{'Bias':<20} {'Baseline':<10} {'CoT':<10} {'Counter':<10} {'Role':<10} {'Best Δ':<20}")
    print("-" * 80)

    for row in results:
        print(f"{row['bias']:<20} {row['baseline']:<10} {row['D1_cot']:<10} "
              f"{row['D3_counter']:<10} {row['D4_role']:<10} {row['best_delta']:<20}")

    # Generate LaTeX
    print("\n" + "=" * 70)
    print("LATEX TABLE:")
    print("=" * 70)

    print(r"\begin{tabular}{@{}lccccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Bias} & \textbf{Baseline} & \textbf{D1: CoT} & \textbf{D3: Counter} & \textbf{D4: Role} & \textbf{Best $\Delta$} \\")
    print(r"\midrule")

    for row in results:
        bias_display = row['bias'].replace('_', ' ').title()
        print(f"{bias_display} & ${row['baseline']}$ & ${row['D1_cot']}$ & "
              f"${row['D3_counter']}$ & ${row['D4_role']}$ & {row['best_delta']} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)

    for row in results:
        if row.get("baseline_val") is not None:
            baseline = row["baseline_val"]
            print(f"\n{row['bias']}:")
            print(f"  Baseline: {baseline:.3f}")

            for s in ["D1_cot", "D3_counter", "D4_role"]:
                if row.get(f"{s}_val") is not None:
                    val = row[f"{s}_val"]
                    change = ((val - baseline) / baseline) * 100 if baseline > 0 else 0
                    direction = "↓" if val < baseline else "↑"
                    print(f"  {s}: {val:.3f} ({direction} {abs(change):.1f}%)")


if __name__ == "__main__":
    main()
