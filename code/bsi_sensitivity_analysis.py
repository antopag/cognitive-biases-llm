"""
BSI Sensitivity Analysis for Cognitive Biases Paper
Addresses reviewer concern: "BSI_num depends on |a − x̄control| in denominator, which can be unstable"

Analyses:
1. BSI stability across variants with different anchor distances
2. Winsorization of denominator (floor at various thresholds)
3. Comparison of BSI vs Cohen's d vs simple difference
4. Bootstrap confidence intervals for robustness
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Path to raw responses
RAW_RESPONSES_DIR = Path(r"G:\My Drive\Papers\PAPERS (MY)\Cognitive biases LLM\results\raw_responses")

# Known anchor values from experimental design (approximate)
# Format: variant_id -> (anchor_value, true_value)
ANCHOR_INFO = {
    'anchoring_v1': {'anchor': 9000000, 'true': 2700000, 'description': 'Population of Chicago'},
    'anchoring_v2': {'anchor': 150, 'true': 54, 'description': 'African countries in UN'},
    'anchoring_v3': {'anchor': 90, 'true': 54, 'description': 'African countries (alt)'},
    'anchoring_v4': {'anchor': 500, 'true': 330, 'description': 'Height of redwood'},
    'anchoring_v5': {'anchor': 120, 'true': 71, 'description': 'Gandhi age'},
}


def load_anchoring_data():
    """Load all anchoring baseline data"""
    results = []
    for json_file in RAW_RESPONSES_DIR.glob("anchoring_*_baseline.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)
    return results


def calculate_bsi_variants(results):
    """Calculate BSI for each variant with different metrics"""
    records = []

    for result in results:
        model = result['model_name']

        for variant in result.get('variants', []):
            variant_id = variant.get('variant_id', 'unknown')

            control_parsed = [v for v in variant.get('control_parsed', [])
                            if v is not None and isinstance(v, (int, float)) and not np.isnan(v)]
            treatment_parsed = [v for v in variant.get('treatment_parsed', [])
                              if v is not None and isinstance(v, (int, float)) and not np.isnan(v)]

            if len(control_parsed) < 5 or len(treatment_parsed) < 5:
                continue

            ctrl_mean = np.mean(control_parsed)
            treat_mean = np.mean(treatment_parsed)
            ctrl_std = np.std(control_parsed, ddof=1)
            treat_std = np.std(treatment_parsed, ddof=1)
            pooled_std = np.sqrt((ctrl_std**2 + treat_std**2) / 2)

            # Get anchor info if available
            anchor_info = ANCHOR_INFO.get(variant_id, {})
            anchor = anchor_info.get('anchor', None)
            true_val = anchor_info.get('true', None)

            # Raw difference
            raw_diff = treat_mean - ctrl_mean

            # Cohen's d
            cohens_d = raw_diff / pooled_std if pooled_std > 0 else np.nan

            # BSI standard (if anchor known)
            if anchor is not None:
                denominator = abs(anchor - ctrl_mean)
                bsi_standard = abs(raw_diff) / denominator if denominator > 0 else np.nan

                # Anchor distance (relative)
                if true_val:
                    anchor_distance_pct = abs(anchor - true_val) / true_val * 100
                else:
                    anchor_distance_pct = np.nan
            else:
                bsi_standard = np.nan
                anchor_distance_pct = np.nan
                denominator = np.nan

            # BSI with winsorized denominator (floor at 10% of control mean)
            if anchor is not None:
                floor_10pct = 0.1 * abs(ctrl_mean) if ctrl_mean != 0 else 1
                denom_winsor_10 = max(abs(anchor - ctrl_mean), floor_10pct)
                bsi_winsor_10 = abs(raw_diff) / denom_winsor_10

                # Floor at 5% of control mean
                floor_5pct = 0.05 * abs(ctrl_mean) if ctrl_mean != 0 else 1
                denom_winsor_5 = max(abs(anchor - ctrl_mean), floor_5pct)
                bsi_winsor_5 = abs(raw_diff) / denom_winsor_5
            else:
                bsi_winsor_10 = np.nan
                bsi_winsor_5 = np.nan

            records.append({
                'model': model,
                'variant': variant_id,
                'n_control': len(control_parsed),
                'n_treatment': len(treatment_parsed),
                'ctrl_mean': ctrl_mean,
                'treat_mean': treat_mean,
                'ctrl_std': ctrl_std,
                'raw_diff': raw_diff,
                'cohens_d': cohens_d,
                'anchor': anchor,
                'anchor_distance_pct': anchor_distance_pct,
                'denominator': denominator,
                'bsi_standard': bsi_standard,
                'bsi_winsor_10': bsi_winsor_10,
                'bsi_winsor_5': bsi_winsor_5,
            })

    return pd.DataFrame(records)


def analyze_denominator_stability(df):
    """Analyze how small denominators affect BSI"""
    print("\n" + "="*70)
    print("ANALYSIS 1: Denominator Stability")
    print("="*70)

    # Filter to rows with valid BSI
    valid = df[df['bsi_standard'].notna()].copy()

    if len(valid) == 0:
        print("No valid BSI values found.")
        return

    # Categorize by denominator size relative to control mean
    valid['denom_ratio'] = valid['denominator'] / valid['ctrl_mean'].abs()

    print("\nDenominator statistics:")
    print(f"  Min denominator ratio: {valid['denom_ratio'].min():.4f}")
    print(f"  Max denominator ratio: {valid['denom_ratio'].max():.4f}")
    print(f"  Mean denominator ratio: {valid['denom_ratio'].mean():.4f}")

    # Cases where denominator is small (< 10% of control mean)
    small_denom = valid[valid['denom_ratio'] < 0.1]
    if len(small_denom) > 0:
        print(f"\nCases with small denominator (<10% of ctrl_mean): {len(small_denom)}")
        print(small_denom[['model', 'variant', 'denom_ratio', 'bsi_standard', 'bsi_winsor_10']].to_string())
    else:
        print("\nNo cases with problematically small denominators.")

    # Compare BSI standard vs winsorized
    print("\n" + "-"*50)
    print("Comparison: BSI Standard vs Winsorized (10% floor)")
    print("-"*50)

    comparison = valid[['model', 'variant', 'bsi_standard', 'bsi_winsor_10', 'cohens_d']].copy()
    comparison['bsi_diff'] = comparison['bsi_standard'] - comparison['bsi_winsor_10']
    comparison = comparison.sort_values('bsi_diff', key=abs, ascending=False)

    print("\nTop 10 cases with largest BSI change after winsorization:")
    print(comparison.head(10).to_string(index=False))


def compare_effect_size_metrics(df):
    """Compare BSI with Cohen's d and raw difference"""
    print("\n" + "="*70)
    print("ANALYSIS 2: Effect Size Metric Comparison")
    print("="*70)

    valid = df[df['bsi_standard'].notna() & df['cohens_d'].notna()].copy()

    if len(valid) == 0:
        print("No valid data for comparison.")
        return

    # Correlation between metrics
    corr_bsi_d = valid['bsi_standard'].corr(valid['cohens_d'])

    print(f"\nCorrelation between BSI and Cohen's d: {corr_bsi_d:.3f}")

    # Summary by model
    print("\n" + "-"*50)
    print("Effect sizes by model (mean across variants)")
    print("-"*50)

    model_summary = valid.groupby('model').agg({
        'bsi_standard': ['mean', 'std'],
        'bsi_winsor_10': ['mean', 'std'],
        'cohens_d': ['mean', 'std'],
        'raw_diff': ['mean', 'std']
    }).round(4)

    print(model_summary.to_string())

    # Rank comparison
    print("\n" + "-"*50)
    print("Model rankings by different metrics")
    print("-"*50)

    rankings = valid.groupby('model').agg({
        'bsi_standard': 'mean',
        'bsi_winsor_10': 'mean',
        'cohens_d': lambda x: abs(x).mean()  # Use absolute Cohen's d
    })

    rankings['rank_bsi'] = rankings['bsi_standard'].rank(ascending=False)
    rankings['rank_bsi_w'] = rankings['bsi_winsor_10'].rank(ascending=False)
    rankings['rank_cohens_d'] = rankings['cohens_d'].rank(ascending=False)

    print(rankings[['rank_bsi', 'rank_bsi_w', 'rank_cohens_d']].to_string())

    # Rank correlation
    rank_corr = rankings[['rank_bsi', 'rank_cohens_d']].corr().iloc[0, 1]
    print(f"\nSpearman correlation between BSI and Cohen's d rankings: {rank_corr:.3f}")


def bootstrap_bsi_ci(control, treatment, anchor, n_boot=1000, alpha=0.05):
    """Bootstrap confidence interval for BSI"""
    bsi_boots = []

    for _ in range(n_boot):
        ctrl_boot = np.random.choice(control, size=len(control), replace=True)
        treat_boot = np.random.choice(treatment, size=len(treatment), replace=True)

        ctrl_mean = np.mean(ctrl_boot)
        treat_mean = np.mean(treat_boot)

        denom = abs(anchor - ctrl_mean)
        if denom > 0:
            bsi = abs(treat_mean - ctrl_mean) / denom
            bsi_boots.append(bsi)

    if len(bsi_boots) < 100:
        return np.nan, np.nan, np.nan

    ci_low = np.percentile(bsi_boots, 100 * alpha / 2)
    ci_high = np.percentile(bsi_boots, 100 * (1 - alpha / 2))

    return np.mean(bsi_boots), ci_low, ci_high


def analyze_bootstrap_stability(results):
    """Analyze BSI stability via bootstrap"""
    print("\n" + "="*70)
    print("ANALYSIS 3: Bootstrap Confidence Intervals")
    print("="*70)

    bootstrap_results = []

    for result in results:
        model = result['model_name']

        for variant in result.get('variants', []):
            variant_id = variant.get('variant_id', 'unknown')

            control_parsed = [v for v in variant.get('control_parsed', [])
                            if v is not None and isinstance(v, (int, float)) and not np.isnan(v)]
            treatment_parsed = [v for v in variant.get('treatment_parsed', [])
                              if v is not None and isinstance(v, (int, float)) and not np.isnan(v)]

            if len(control_parsed) < 10 or len(treatment_parsed) < 10:
                continue

            anchor_info = ANCHOR_INFO.get(variant_id, {})
            anchor = anchor_info.get('anchor', None)

            if anchor is None:
                continue

            bsi_mean, ci_low, ci_high = bootstrap_bsi_ci(
                control_parsed, treatment_parsed, anchor, n_boot=1000
            )

            if not np.isnan(bsi_mean):
                ci_width = ci_high - ci_low
                bootstrap_results.append({
                    'model': model,
                    'variant': variant_id,
                    'bsi_mean': bsi_mean,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'ci_width': ci_width,
                    'cv': ci_width / bsi_mean if bsi_mean > 0 else np.nan
                })

    if not bootstrap_results:
        print("No valid bootstrap results.")
        return

    boot_df = pd.DataFrame(bootstrap_results)

    print("\nBootstrap results (95% CI):")
    print(boot_df.round(4).to_string(index=False))

    print("\n" + "-"*50)
    print("Cases with wide confidence intervals (CV > 1):")
    print("-"*50)

    wide_ci = boot_df[boot_df['cv'] > 1]
    if len(wide_ci) > 0:
        print(wide_ci.to_string(index=False))
    else:
        print("No cases with CV > 1 (all estimates are reasonably stable).")

    return boot_df


def create_summary_table(df):
    """Create summary table for appendix"""
    print("\n" + "="*70)
    print("SUMMARY TABLE FOR APPENDIX")
    print("="*70)

    valid = df[df['bsi_standard'].notna()].copy()

    if len(valid) == 0:
        print("No valid data.")
        return

    # Aggregate by model
    summary = valid.groupby('model').agg({
        'bsi_standard': ['mean', 'std', 'min', 'max'],
        'bsi_winsor_10': ['mean', 'std'],
        'cohens_d': ['mean', 'std'],
        'n_control': 'sum',
        'n_treatment': 'sum'
    }).round(4)

    print("\nAnchoring Bias: BSI Sensitivity Analysis Summary")
    print(summary.to_string())

    # Recommendations
    print("\n" + "-"*50)
    print("RECOMMENDATIONS")
    print("-"*50)

    # Check if winsorization changes conclusions
    valid['rank_std'] = valid.groupby('variant')['bsi_standard'].rank()
    valid['rank_win'] = valid.groupby('variant')['bsi_winsor_10'].rank()
    rank_changes = (valid['rank_std'] != valid['rank_win']).sum()

    print(f"\n1. Winsorization impact: {rank_changes} rank changes across {len(valid)} observations")

    corr = valid['bsi_standard'].corr(valid['bsi_winsor_10'])
    print(f"2. BSI vs BSI-winsorized correlation: {corr:.4f}")

    corr_d = valid['bsi_standard'].corr(valid['cohens_d'].abs())
    print(f"3. BSI vs |Cohen's d| correlation: {corr_d:.4f}")

    if corr > 0.95:
        print("\nConclusion: Winsorization has minimal impact; BSI estimates are stable.")
    else:
        print("\nConclusion: Some instability detected; winsorized BSI recommended for robustness.")


def save_results(df, output_dir):
    """Save results to CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    df.to_csv(output_path / 'bsi_sensitivity_analysis.csv', index=False)
    print(f"\nResults saved to {output_path}")


def main():
    print("Loading anchoring data...")
    results = load_anchoring_data()
    print(f"Loaded {len(results)} anchoring experiment files")

    print("\nCalculating BSI variants...")
    df = calculate_bsi_variants(results)
    print(f"Computed metrics for {len(df)} model-variant combinations")

    # Run analyses
    analyze_denominator_stability(df)
    compare_effect_size_metrics(df)
    boot_df = analyze_bootstrap_stability(results)
    create_summary_table(df)

    # Save results
    output_dir = RAW_RESPONSES_DIR.parent / 'bsi_sensitivity_analysis'
    save_results(df, output_dir)

    return df, boot_df


if __name__ == "__main__":
    df, boot_df = main()
