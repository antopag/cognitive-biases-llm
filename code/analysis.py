"""Analysis and visualization of experiment results."""

import json
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats

import config
from metrics import (bsi_numerical, bsi_categorical, bsi_overconfidence,
                     compute_bsi_with_uncertainty, benjamini_hochberg)

matplotlib.rcParams.update({
    "font.size": 12,
    "figure.figsize": (12, 8),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_results(results_dir: str = None) -> list[dict]:
    """Load all JSON result files from the results directory."""
    if results_dir is None:
        results_dir = config.RAW_RESPONSES_DIR

    files = glob.glob(os.path.join(results_dir, "*.json"))
    results = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            results.append(json.load(fp))
    print(f"Loaded {len(results)} result files.")
    return results


def compute_all_bsi(results: list[dict]) -> pd.DataFrame:
    """Compute BSI for all bias-model-strategy combinations.

    Returns a DataFrame with columns:
        bias_name, model_name, strategy, bsi_mean, sigma_stat, sigma_sys,
        sigma_tot, p_value, significant, n_variants, parse_failure_rate
    """
    rows = []

    for res in results:
        bias_name = res["bias_name"]
        model_name = res["model_name"]
        strategy = res.get("strategy", "baseline")

        if strategy == "D5_ensemble":
            # Handle ensemble results differently
            continue  # TODO: implement ensemble BSI

        # Determine bias type
        bias_type = _infer_bias_type(bias_name)

        variant_data = []
        total_failures = 0
        total_responses = 0

        for v in res["variants"]:
            ctrl = np.array(v["control_parsed"])
            treat = np.array(v["treatment_parsed"])
            failures = v["parse_failures_control"] + v["parse_failures_treatment"]
            total_failures += failures
            total_responses += (len(v["control_responses"]) +
                                len(v["treatment_responses"]))

            if len(ctrl) < 10 or len(treat) < 10:
                continue  # skip variants with too few valid responses

            vd = {
                "control_values": ctrl,
                "treatment_values": treat,
            }

            # Add bias-specific parameters
            if bias_type == "numerical":
                anchor = _get_anchor(bias_name, v["variant_id"])
                if anchor is None:
                    continue
                vd["anchor"] = anchor
            elif bias_type == "categorical":
                bias_option = _get_bias_option(bias_name, v["variant_id"])
                if bias_option is None:
                    continue
                vd["bias_option"] = bias_option

            variant_data.append(vd)

        if not variant_data:
            continue

        bsi_result = compute_bsi_with_uncertainty(
            variant_data, bias_type, n_bootstrap=config.BOOTSTRAP_RESAMPLES)

        parse_failure_rate = (total_failures / total_responses
                              if total_responses > 0 else 0)

        rows.append({
            "bias_name": bias_name,
            "model_name": model_name,
            "strategy": strategy,
            "bsi_mean": bsi_result["bsi_mean"],
            "sigma_stat": bsi_result["sigma_stat"],
            "sigma_sys": bsi_result["sigma_sys"],
            "sigma_tot": bsi_result["sigma_tot"],
            "p_value": bsi_result["p_value"],
            "significant": bsi_result["significant"],
            "n_variants": len(variant_data),
            "parse_failure_rate": parse_failure_rate,
        })

    df = pd.DataFrame(rows)

    # Apply Benjamini-Hochberg correction
    if len(df) > 0 and "p_value" in df.columns:
        p_vals = df["p_value"].fillna(1.0).tolist()
        bh_significant = benjamini_hochberg(p_vals, alpha=config.ALPHA)
        df["significant_bh"] = bh_significant

    return df


def _infer_bias_type(bias_name: str) -> str:
    """Infer bias type from bias name."""
    numerical = {"anchoring", "overconfidence"}
    if bias_name in numerical:
        return "numerical"
    return "categorical"


def _get_anchor(bias_name: str, variant_id: str) -> float | None:
    """Look up anchor value for a given bias variant."""
    from biases import ALL_BIASES
    for b in ALL_BIASES:
        if b["name"] == bias_name:
            for v in b["variants"]:
                if v["id"] == variant_id:
                    return v.get("anchor")
    return None


def _get_bias_option(bias_name: str, variant_id: str) -> str | None:
    """Look up bias option for a given bias variant."""
    from biases import ALL_BIASES
    for b in ALL_BIASES:
        if b["name"] == bias_name:
            for v in b["variants"]:
                if v["id"] == variant_id:
                    return v.get("bias_option")
    return None


# ============================================================
# Visualization
# ============================================================

def plot_heatmap(df: pd.DataFrame, strategy: str = "baseline",
                 save: bool = True):
    """Plot BSI heatmap: biases x models."""
    subset = df[df["strategy"] == strategy]

    pivot = subset.pivot_table(
        index="bias_name", columns="model_name",
        values="bsi_mean", aggfunc="mean")

    # Order biases by category (11 biases - overconfidence excluded)
    bias_order = [
        "anchoring", "availability_heuristic", "representativeness",
        "framing_effect", "sunk_cost_fallacy",
        "status_quo_bias", "decoy_effect", "confirmation_bias",
        "bandwagon_effect", "authority_bias", "primacy_recency",
    ]

    # Exclude MiniMax due to >70% parse failures
    minimax_cols = [c for c in pivot.columns if "minimax" in c.lower()]
    if minimax_cols:
        pivot = pivot.drop(columns=minimax_cols)
    present = [b for b in bias_order if b in pivot.index]
    pivot = pivot.reindex(present)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Bias Strength Index (BSI)"})
    ax.set_title(f"Cognitive Bias Strength Index — {strategy}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Cognitive Bias")
    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR, f"heatmap_{strategy}.pdf")
        fig.savefig(path)
        print(f"Saved: {path}")

    return fig


def plot_bsi_with_errorbars(df: pd.DataFrame, model_name: str = "gpt-4o",
                            save: bool = True):
    """Plot BSI with error bars (stat + sys) for a single model."""
    subset = df[(df["model_name"] == model_name) &
                (df["strategy"] == "baseline")]
    subset = subset.sort_values("bsi_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(subset))
    colors = []
    for _, row in subset.iterrows():
        if row.get("significant_bh", False):
            colors.append("#d32f2f")
        else:
            colors.append("#9e9e9e")

    ax.barh(list(y_pos), subset["bsi_mean"], xerr=subset["sigma_tot"],
            color=colors, edgecolor="black", linewidth=0.5,
            capsize=3, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(subset["bias_name"])
    ax.set_xlabel("Bias Strength Index (BSI)")
    ax.set_title(f"BSI with Total Uncertainty — {model_name}",
                 fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d32f2f", label="Significant (BH-corrected)"),
        Patch(facecolor="#9e9e9e", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR,
                            f"bsi_errorbars_{model_name}.pdf")
        fig.savefig(path)
        print(f"Saved: {path}")

    return fig


def plot_debiasing_comparison(df: pd.DataFrame, model_name: str = "gpt-4o",
                              save: bool = True):
    """Compare debiasing strategies for a single model."""
    subset = df[df["model_name"] == model_name]

    # Check if model has debiasing data (more than just baseline)
    strategies = subset["strategy"].unique()
    if len(strategies) <= 1:
        print(f"Skipping debiasing plot for {model_name}: no debiasing data")
        return None

    pivot = subset.pivot_table(
        index="bias_name", columns="strategy",
        values="bsi_mean", aggfunc="mean")

    # Compute reduction relative to baseline
    if "baseline" in pivot.columns:
        for col in pivot.columns:
            if col != "baseline":
                pivot[col] = 1 - pivot[col] / pivot["baseline"]
        pivot = pivot.drop(columns=["baseline"])

    # Restrict to biases that were actually evaluated under debiasing
    # (drop rows where all strategy values are NaN). This keeps the figure
    # aligned with Table 4 and prevents empty x-axis positions that could
    # be misread as zero-reduction outcomes.
    pivot = pivot.dropna(how="all")

    # Check if there's data to plot after processing
    if pivot.empty or pivot.select_dtypes(include='number').empty:
        print(f"Skipping debiasing plot for {model_name}: no numeric data after processing")
        return None

    # Rename strategy labels for paper consistency
    label_map = {"D1_cot": "D1: CoT", "D3_counter": "D2: Counter", "D4_role": "D3: Role"}
    pivot = pivot.rename(columns=label_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, width=0.7)
    ax.set_ylabel("BSI Reduction (fraction)")
    ax.set_xlabel("Cognitive Bias")
    ax.set_title(f"Debiasing Effectiveness — {model_name}",
                 fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    # Extend y-axis below zero so any strategy that worsens bias would be
    # visible as a downward bar (none observed in current results).
    y_min = min(-0.05, pivot.min().min() - 0.05)
    y_max = max(0.85, pivot.max().max() + 0.05)
    ax.set_ylim(y_min, y_max)
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR,
                            f"debiasing_{model_name}.pdf")
        fig.savefig(path)
        print(f"Saved: {path}")

    return fig


def plot_uncertainty_decomposition(df: pd.DataFrame,
                                   model_name: str = "gpt-4o",
                                   save: bool = True):
    """Plot stacked bar of statistical vs systematic uncertainty."""
    subset = df[(df["model_name"] == model_name) &
                (df["strategy"] == "baseline")]
    subset = subset.sort_values("sigma_tot", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(subset))
    ax.barh(list(y_pos), subset["sigma_stat"], label="Statistical",
            color="#1976d2", alpha=0.8)
    ax.barh(list(y_pos), subset["sigma_sys"], left=subset["sigma_stat"],
            label="Systematic (prompt)", color="#ff9800", alpha=0.8)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(subset["bias_name"])
    ax.set_xlabel("Uncertainty")
    ax.set_title(f"Uncertainty Decomposition — {model_name}",
                 fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR,
                            f"uncertainty_{model_name}.pdf")
        fig.savefig(path)
        print(f"Saved: {path}")

    return fig


def generate_latex_table(df: pd.DataFrame, strategy: str = "baseline") -> str:
    """Generate LaTeX table of BSI results for the paper."""
    subset = df[df["strategy"] == strategy].copy()

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Bias Strength Index (BSI) with total uncertainty "
                 r"for all bias--model combinations. "
                 r"Bold values indicate statistical significance after "
                 r"Benjamini--Hochberg correction ($\alpha = 0.05$).}")
    lines.append(r"\label{tab:bsi-results}")
    lines.append(r"\small")

    models = sorted(subset["model_name"].unique())
    ncols = len(models) + 1
    col_spec = "@{}l" + "c" * len(models) + "@{}"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = r"\textbf{Bias} & " + " & ".join(
        [r"\textbf{" + m.replace("_", r"\_") + "}" for m in models])
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    bias_order = [
        "anchoring", "availability_heuristic", "representativeness",
        "overconfidence", "framing_effect", "sunk_cost_fallacy",
        "status_quo_bias", "decoy_effect", "confirmation_bias",
        "bandwagon_effect", "authority_bias", "primacy_recency",
    ]

    for bias_name in bias_order:
        row_data = subset[subset["bias_name"] == bias_name]
        if row_data.empty:
            continue

        display_name = bias_name.replace("_", " ").title()
        cells = [display_name]

        for model in models:
            cell = row_data[row_data["model_name"] == model]
            if cell.empty:
                cells.append("--")
            else:
                r = cell.iloc[0]
                val = f"${r['bsi_mean']:.3f} \\pm {r['sigma_tot']:.3f}$"
                if r.get("significant_bh", False):
                    val = r"\textbf{" + val + "}"
                cells.append(val)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# --- Entry point ---

if __name__ == "__main__":
    results = load_results()

    if not results:
        print("No results found. Run experiment.py first.")
        exit(1)

    df = compute_all_bsi(results)
    print(f"\nComputed BSI for {len(df)} bias-model combinations.\n")
    print(df.to_string(index=False))

    # Save summary
    summary_path = os.path.join(config.RESULTS_DIR, "bsi_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # Generate plots
    models_present = df["model_name"].unique()
    strategies_present = df["strategy"].unique()

    for strategy in strategies_present:
        plot_heatmap(df, strategy=strategy)
        plt.close('all')

    for model in models_present:
        plot_bsi_with_errorbars(df, model_name=model)
        plot_uncertainty_decomposition(df, model_name=model)
        plt.close('all')

    if len(strategies_present) > 1:
        for model in models_present:
            plot_debiasing_comparison(df, model_name=model)
            plt.close('all')

    # Generate LaTeX table
    latex = generate_latex_table(df)
    latex_path = os.path.join(config.RESULTS_DIR, "bsi_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nSaved LaTeX table: {latex_path}")
