"""
GLMM Analysis for Cognitive Biases Paper
Replaces t-tests with Generalized Linear Mixed Models

Model specification:
- Categorical biases: logistic GLMM with bias_choice ~ treatment + (1|variant)
- Numerical biases: linear mixed model with estimate ~ treatment + (1|variant)
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# Path to raw responses
RAW_RESPONSES_DIR = Path(r"G:\My Drive\Papers\PAPERS (MY)\Cognitive biases LLM\results\raw_responses")

# Bias type classification
CATEGORICAL_BIASES = [
    'framing_effect', 'sunk_cost_fallacy', 'status_quo_bias',
    'decoy_effect', 'confirmation_bias', 'bandwagon_effect',
    'authority_bias', 'primacy_recency'
]

NUMERICAL_BIASES = [
    'anchoring', 'availability_heuristic', 'representativeness', 'overconfidence'
]

# For categorical biases, define which response indicates "biased" behavior
# This mapping depends on the specific prompt design
BIAS_RESPONSE_MAPPING = {
    'framing_effect': {'control_biased': 'A', 'treatment_biased': 'B'},  # Risk-seeking in loss frame
    'sunk_cost_fallacy': {'biased': 'A'},  # Continuing despite sunk cost
    'status_quo_bias': {'biased': 'A'},  # Choosing status quo
    'decoy_effect': {'biased': 'B'},  # Choosing target over competitor
    'confirmation_bias': {'biased': 'A'},  # Confirming prior belief
    'bandwagon_effect': {'biased': 'A'},  # Following majority
    'authority_bias': {'biased': 'A'},  # Following authority
    'primacy_recency': {'control_biased': 'A', 'treatment_biased': 'B'},  # Order-dependent
}


def load_all_results():
    """Load all baseline JSON files"""
    results = []
    for json_file in RAW_RESPONSES_DIR.glob("*_baseline.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)
    return results


def build_trial_level_data_categorical(results):
    """
    Build trial-level dataset for categorical biases.
    Each row = one trial with columns: model, bias, variant, treatment, response, biased
    """
    records = []

    for result in results:
        model = result['model_name']
        bias = result['bias_name']

        if bias not in CATEGORICAL_BIASES:
            continue

        for variant in result.get('variants', []):
            variant_id = variant.get('variant_id', 'unknown')

            # Control trials
            for trial_idx, response in enumerate(variant.get('control_responses', [])):
                if response in ['A', 'B', 'C']:  # Valid response
                    # Determine if response is "biased"
                    mapping = BIAS_RESPONSE_MAPPING.get(bias, {})
                    if 'control_biased' in mapping:
                        is_biased = 1 if response == mapping['control_biased'] else 0
                    elif 'biased' in mapping:
                        is_biased = 1 if response == mapping['biased'] else 0
                    else:
                        is_biased = None

                    records.append({
                        'model': model,
                        'bias': bias,
                        'variant': variant_id,
                        'treatment': 0,
                        'trial': trial_idx,
                        'response': response,
                        'biased': is_biased
                    })

            # Treatment trials
            for trial_idx, response in enumerate(variant.get('treatment_responses', [])):
                if response in ['A', 'B', 'C']:  # Valid response
                    mapping = BIAS_RESPONSE_MAPPING.get(bias, {})
                    if 'treatment_biased' in mapping:
                        is_biased = 1 if response == mapping['treatment_biased'] else 0
                    elif 'biased' in mapping:
                        is_biased = 1 if response == mapping['biased'] else 0
                    else:
                        is_biased = None

                    records.append({
                        'model': model,
                        'bias': bias,
                        'variant': variant_id,
                        'treatment': 1,
                        'trial': trial_idx,
                        'response': response,
                        'biased': is_biased
                    })

    return pd.DataFrame(records)


def is_numeric(value):
    """Check if value can be converted to float"""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return not np.isnan(value) if isinstance(value, float) else True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def build_trial_level_data_numerical(results):
    """
    Build trial-level dataset for numerical biases.
    Each row = one trial with columns: model, bias, variant, treatment, estimate
    """
    records = []

    for result in results:
        model = result['model_name']
        bias = result['bias_name']

        if bias not in NUMERICAL_BIASES:
            continue

        for variant in result.get('variants', []):
            variant_id = variant.get('variant_id', 'unknown')

            # Control trials - use parsed values
            for trial_idx, value in enumerate(variant.get('control_parsed', [])):
                if is_numeric(value):
                    records.append({
                        'model': model,
                        'bias': bias,
                        'variant': variant_id,
                        'treatment': 0,
                        'trial': trial_idx,
                        'estimate': float(value)
                    })

            # Treatment trials
            for trial_idx, value in enumerate(variant.get('treatment_parsed', [])):
                if is_numeric(value):
                    records.append({
                        'model': model,
                        'bias': bias,
                        'variant': variant_id,
                        'treatment': 1,
                        'trial': trial_idx,
                        'estimate': float(value)
                    })

    return pd.DataFrame(records)


def fit_glmm_categorical(df, model_name, bias_name):
    """
    Fit logistic GLMM for a categorical bias:
    biased ~ treatment + (1|variant)

    Returns coefficient, SE, z-value, p-value, odds ratio
    """
    subset = df[(df['model'] == model_name) & (df['bias'] == bias_name)].copy()
    subset = subset.dropna(subset=['biased'])

    if len(subset) < 20:
        return None

    # Check variance in outcome
    if subset['biased'].nunique() < 2:
        return {'note': 'No variance in outcome'}

    # Check if we have multiple variants
    n_variants = subset['variant'].nunique()

    try:
        if n_variants > 1:
            # Fit GLMM with random intercept for variant
            model = smf.mixedlm("biased ~ treatment", subset, groups=subset["variant"])
            result = model.fit(method='powell', maxiter=500)
        else:
            # Fall back to simple logistic regression
            model = smf.logit("biased ~ treatment", subset)
            result = model.fit(disp=False)

        coef = result.params.get('treatment', np.nan)
        se = result.bse.get('treatment', np.nan)

        # For mixed models, use Wald z-test
        if hasattr(result, 'tvalues'):
            z_val = result.tvalues.get('treatment', np.nan)
        else:
            z_val = coef / se if se > 0 else np.nan

        p_val = result.pvalues.get('treatment', np.nan)

        # Odds ratio
        odds_ratio = np.exp(coef) if not np.isnan(coef) else np.nan

        return {
            'model': model_name,
            'bias': bias_name,
            'n_trials': len(subset),
            'n_variants': n_variants,
            'coefficient': coef,
            'std_error': se,
            'z_value': z_val,
            'p_value': p_val,
            'odds_ratio': odds_ratio,
            'control_mean': subset[subset['treatment'] == 0]['biased'].mean(),
            'treatment_mean': subset[subset['treatment'] == 1]['biased'].mean()
        }

    except Exception as e:
        return {'model': model_name, 'bias': bias_name, 'error': str(e)}


def fit_lmm_numerical(df, model_name, bias_name):
    """
    Fit linear mixed model for numerical bias:
    estimate ~ treatment + (1|variant)

    Returns coefficient, SE, t-value, p-value
    """
    subset = df[(df['model'] == model_name) & (df['bias'] == bias_name)].copy()

    if len(subset) < 20:
        return None

    n_variants = subset['variant'].nunique()

    try:
        if n_variants > 1:
            # Fit LMM with random intercept for variant
            model = smf.mixedlm("estimate ~ treatment", subset, groups=subset["variant"])
            result = model.fit(method='powell', maxiter=500)
        else:
            # Fall back to OLS
            model = smf.ols("estimate ~ treatment", subset)
            result = model.fit()

        coef = result.params.get('treatment', np.nan)
        se = result.bse.get('treatment', np.nan)

        if hasattr(result, 'tvalues'):
            t_val = result.tvalues.get('treatment', np.nan)
        else:
            t_val = coef / se if se > 0 else np.nan

        p_val = result.pvalues.get('treatment', np.nan)

        # Effect size (Cohen's d approximation)
        ctrl_std = subset[subset['treatment'] == 0]['estimate'].std()
        cohens_d = coef / ctrl_std if ctrl_std > 0 else np.nan

        return {
            'model': model_name,
            'bias': bias_name,
            'n_trials': len(subset),
            'n_variants': n_variants,
            'coefficient': coef,
            'std_error': se,
            't_value': t_val,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'control_mean': subset[subset['treatment'] == 0]['estimate'].mean(),
            'treatment_mean': subset[subset['treatment'] == 1]['estimate'].mean()
        }

    except Exception as e:
        return {'model': model_name, 'bias': bias_name, 'error': str(e)}


def run_all_glmm_analyses(cat_df, num_df):
    """Run GLMM for all model-bias combinations"""

    categorical_results = []
    numerical_results = []

    models = cat_df['model'].unique().tolist() if len(cat_df) > 0 else []
    models_num = num_df['model'].unique().tolist() if len(num_df) > 0 else []
    all_models = list(set(models + models_num))

    print("\n" + "="*70)
    print("GLMM ANALYSIS: CATEGORICAL BIASES")
    print("="*70)

    for model in sorted(all_models):
        for bias in CATEGORICAL_BIASES:
            result = fit_glmm_categorical(cat_df, model, bias)
            if result and 'coefficient' in result:
                categorical_results.append(result)
                print(f"{model:25s} | {bias:20s} | coef={result['coefficient']:+.3f} | "
                      f"OR={result['odds_ratio']:.2f} | p={result['p_value']:.4f}")

    print("\n" + "="*70)
    print("GLMM ANALYSIS: NUMERICAL BIASES")
    print("="*70)

    for model in sorted(all_models):
        for bias in NUMERICAL_BIASES:
            result = fit_lmm_numerical(num_df, model, bias)
            if result and 'coefficient' in result:
                numerical_results.append(result)
                print(f"{model:25s} | {bias:20s} | coef={result['coefficient']:+.2f} | "
                      f"d={result['cohens_d']:.2f} | p={result['p_value']:.4f}")

    return categorical_results, numerical_results


def apply_fdr_correction(results_list, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction"""
    p_values = [r.get('p_value', 1.0) for r in results_list if 'p_value' in r]
    n = len(p_values)

    if n == 0:
        return results_list

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # BH correction
    bh_critical = [(i + 1) / n * alpha for i in range(n)]

    # Find largest p-value that is <= its critical value
    significant = sorted_p <= bh_critical

    # Create mapping
    corrected = {}
    for i, idx in enumerate(sorted_indices):
        corrected[idx] = significant[i]

    # Add significance to results
    for i, r in enumerate(results_list):
        if 'p_value' in r:
            r['significant_fdr'] = corrected.get(i, False)

    return results_list


def create_summary_tables(cat_results, num_results):
    """Create summary tables for paper"""

    print("\n" + "="*70)
    print("TABLE: GLMM Results Summary (Categorical Biases)")
    print("="*70)

    if cat_results:
        cat_df = pd.DataFrame(cat_results)
        # Pivot to model x bias
        if 'odds_ratio' in cat_df.columns:
            pivot = cat_df.pivot(index='bias', columns='model', values='odds_ratio')
            print("\nOdds Ratios (treatment effect):")
            print(pivot.round(2).to_string())

            # P-values
            pivot_p = cat_df.pivot(index='bias', columns='model', values='p_value')
            print("\nP-values:")
            print(pivot_p.round(4).to_string())

    print("\n" + "="*70)
    print("TABLE: GLMM Results Summary (Numerical Biases)")
    print("="*70)

    if num_results:
        num_df = pd.DataFrame(num_results)
        if 'cohens_d' in num_df.columns:
            pivot = num_df.pivot(index='bias', columns='model', values='cohens_d')
            print("\nCohen's d (effect size):")
            print(pivot.round(2).to_string())

            pivot_p = num_df.pivot(index='bias', columns='model', values='p_value')
            print("\nP-values:")
            print(pivot_p.round(4).to_string())

    # Significant results after FDR correction
    print("\n" + "="*70)
    print("SIGNIFICANT RESULTS (FDR-corrected, α=0.05)")
    print("="*70)

    all_results = cat_results + num_results
    all_results = apply_fdr_correction(all_results)

    sig_results = [r for r in all_results if r.get('significant_fdr', False)]
    if sig_results:
        for r in sig_results:
            effect = r.get('odds_ratio', r.get('cohens_d', 'N/A'))
            print(f"{r['model']:25s} | {r['bias']:20s} | effect={effect:.2f} | p={r['p_value']:.4f}")
    else:
        print("No significant results after FDR correction.")

    return all_results


def save_results(cat_results, num_results, output_dir):
    """Save results to CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if cat_results:
        pd.DataFrame(cat_results).to_csv(output_path / 'glmm_categorical_results.csv', index=False)
    if num_results:
        pd.DataFrame(num_results).to_csv(output_path / 'glmm_numerical_results.csv', index=False)

    print(f"\nResults saved to {output_path}")


def main():
    print("Loading data...")
    results = load_all_results()
    print(f"Loaded {len(results)} experiment files")

    print("\nBuilding trial-level datasets...")
    cat_df = build_trial_level_data_categorical(results)
    num_df = build_trial_level_data_numerical(results)

    print(f"Categorical data: {len(cat_df)} trials")
    print(f"Numerical data: {len(num_df)} trials")

    # Run GLMM analyses
    cat_results, num_results = run_all_glmm_analyses(cat_df, num_df)

    # Create summary tables
    all_results = create_summary_tables(cat_results, num_results)

    # Save results
    output_dir = RAW_RESPONSES_DIR.parent / 'glmm_analysis'
    save_results(cat_results, num_results, output_dir)

    return cat_df, num_df, cat_results, num_results


if __name__ == "__main__":
    cat_df, num_df, cat_results, num_results = main()
