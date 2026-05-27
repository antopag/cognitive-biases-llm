"""Preliminary analysis of Llama 3.3 70B results."""

import os, sys, json
code_dir = r"G:\My Drive\Papers\PAPERS (MY)\Cognitive biases LLM\code"
os.chdir(code_dir)
sys.path.insert(0, code_dir)

import numpy as np
import config
from biases import ALL_BIASES
from metrics import (bsi_numerical, bsi_categorical, bootstrap_bsi,
                     compute_bsi_with_uncertainty)

MODEL = "llama-3.3-70b"
results_dir = config.RAW_RESPONSES_DIR

print(f"{'='*70}")
print(f"  PRELIMINARY ANALYSIS: {MODEL}")
print(f"{'='*70}\n")

all_bsi = []

for bias in ALL_BIASES:
    bias_name = bias["name"]
    filepath = os.path.join(results_dir, f"{bias_name}_{MODEL}_baseline.json")

    if not os.path.exists(filepath):
        print(f"  {bias_name}: FILE NOT FOUND")
        continue

    with open(filepath, "r") as f:
        data = json.load(f)

    # Determine bias type
    is_numerical = bias_name in ("anchoring", "overconfidence")
    is_overconfidence = bias_name == "overconfidence"

    variant_bsis = []
    total_ctrl = 0
    total_treat = 0
    total_fail_ctrl = 0
    total_fail_treat = 0

    for i, variant in enumerate(data["variants"]):
        ctrl = variant["control_parsed"]
        treat = variant["treatment_parsed"]
        total_ctrl += len(ctrl)
        total_treat += len(treat)
        total_fail_ctrl += variant["parse_failures_control"]
        total_fail_treat += variant["parse_failures_treatment"]

        if len(ctrl) < 5 or len(treat) < 5:
            variant_bsis.append(np.nan)
            continue

        bias_def = bias["variants"][i]

        if is_overconfidence:
            # Special handling for overconfidence (interval parsing)
            # For now, skip detailed overconfidence analysis
            variant_bsis.append(np.nan)
            continue

        if is_numerical:
            anchor = bias_def.get("anchor")
            if anchor is None:
                variant_bsis.append(np.nan)
                continue
            ctrl_arr = np.array(ctrl, dtype=float)
            treat_arr = np.array(treat, dtype=float)
            bsi = bsi_numerical(ctrl_arr, treat_arr, anchor)
        else:
            bias_option = bias_def.get("bias_option")
            if bias_option is None:
                variant_bsis.append(np.nan)
                continue
            ctrl_arr = np.array(ctrl)
            treat_arr = np.array(treat)
            bsi = bsi_categorical(ctrl_arr, treat_arr, bias_option)

        variant_bsis.append(bsi)

    valid_bsis = [b for b in variant_bsis if not np.isnan(b)]

    if valid_bsis:
        mean_bsi = np.mean(valid_bsis)
        std_bsi = np.std(valid_bsis, ddof=1) if len(valid_bsis) > 1 else 0
    else:
        mean_bsi = np.nan
        std_bsi = np.nan

    fail_rate = (total_fail_ctrl + total_fail_treat) / max(1,
        total_ctrl + total_treat + total_fail_ctrl + total_fail_treat) * 100

    # Significance indicator
    if not np.isnan(mean_bsi):
        sig = "***" if mean_bsi > 0.1 else "**" if mean_bsi > 0.05 else "*" if mean_bsi > 0.01 else ""
    else:
        sig = "N/A"

    print(f"  {bias_name:.<30s} BSI = {mean_bsi:.4f} ± {std_bsi:.4f}  "
          f"fail={fail_rate:.1f}%  variants={len(valid_bsis)}/{len(variant_bsis)}  {sig}")

    if not np.isnan(mean_bsi):
        all_bsi.append({"name": bias_name, "bsi": mean_bsi, "std": std_bsi})

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")

if all_bsi:
    sorted_bsi = sorted(all_bsi, key=lambda x: x["bsi"], reverse=True)
    print(f"\n  Biases ranked by BSI (strongest → weakest):\n")
    for i, b in enumerate(sorted_bsi, 1):
        bar = "█" * int(b["bsi"] * 50)
        print(f"    {i:2d}. {b['name']:.<28s} {b['bsi']:.4f}  {bar}")

    mean_all = np.mean([b["bsi"] for b in all_bsi])
    print(f"\n  Mean BSI across all biases: {mean_all:.4f}")
    print(f"  Biases with BSI > 0.05: {sum(1 for b in all_bsi if b['bsi'] > 0.05)}/{len(all_bsi)}")
    print(f"  Biases with BSI > 0.10: {sum(1 for b in all_bsi if b['bsi'] > 0.10)}/{len(all_bsi)}")

# Detail view for most interesting biases
print(f"\n{'='*70}")
print(f"  DETAILED VIEW: Per-variant BSI")
print(f"{'='*70}\n")

for bias in ALL_BIASES:
    bias_name = bias["name"]
    filepath = os.path.join(results_dir, f"{bias_name}_{MODEL}_baseline.json")
    if not os.path.exists(filepath):
        continue

    with open(filepath, "r") as f:
        data = json.load(f)

    is_numerical = bias_name in ("anchoring",)
    is_interval = bias_name == "overconfidence"

    print(f"  {bias_name}:")
    for i, variant in enumerate(data["variants"]):
        ctrl = variant["control_parsed"]
        treat = variant["treatment_parsed"]
        vid = variant["variant_id"]

        if len(ctrl) < 5 or len(treat) < 5:
            print(f"    {vid}: insufficient data (ctrl={len(ctrl)}, treat={len(treat)})")
            continue

        if is_interval:
            # Intervals are stored as [low, high] lists
            ctrl_widths = [c[1] - c[0] for c in ctrl if isinstance(c, list) and len(c) == 2]
            treat_widths = [t[1] - t[0] for t in treat if isinstance(t, list) and len(t) == 2]
            if ctrl_widths and treat_widths:
                print(f"    {vid}: ctrl_mean_width={np.mean(ctrl_widths):.1f}  "
                      f"treat_mean_width={np.mean(treat_widths):.1f}  n_ctrl={len(ctrl_widths)} n_treat={len(treat_widths)}")
            else:
                print(f"    {vid}: no valid intervals")
        elif is_numerical:
            ctrl_mean = np.mean(ctrl)
            treat_mean = np.mean(treat)
            print(f"    {vid}: ctrl_mean={ctrl_mean:.1f}  treat_mean={treat_mean:.1f}  "
                  f"shift={treat_mean - ctrl_mean:+.1f}")
        else:
            from collections import Counter
            ctrl_counts = Counter(ctrl)
            treat_counts = Counter(treat)
            print(f"    {vid}: ctrl={dict(ctrl_counts)}  treat={dict(treat_counts)}")

    print()
