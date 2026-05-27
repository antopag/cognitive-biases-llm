"""
GLMM Overdispersion and Sensitivity Analysis
Addresses reviewer concern: "GLMMs treat repeated stochastic samples as independent trials,
potentially inflating effective sample size and p-values"

Analyses:
1. Overdispersion diagnostics (residual deviance / df)
2. Intraclass Correlation Coefficients (ICC)
3. Sensitivity analysis with reduced trial counts
4. Comparison: standard GLMM vs quasi-binomial vs aggregated binomial
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
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

# Bias response mapping (which response indicates "biased" behavior)
BIAS_RESPONSE_MAPPING = {
    'framing_effect': {'control_biased': 'A', 'treatment_biased': 'B'},
    'sunk_cost_fallacy': {'biased': 'A'},
    'status_quo_bias': {'biased': 'A'},
    'decoy_effect': {'biased': 'B'},
    'confirmation_bias': {'biased': 'A'},
    'bandwagon_effect': {'biased': 'A'},
    'authority_bias': {'biased': 'A'},
    'primacy_recency': {'control_biased': 'A', 'treatment_biased': 'B'},
}


def load_all_results():
    """Load all baseline JSON files"""
    results = []
    for json_file in RAW_RESPONSES_DIR.glob("*_baseline.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)
    return results


def build_trial_level_data(results):
    """Build trial-level dataset for categorical biases"""
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
                if response in ['A', 'B', 'C']:
                    mapping = BIAS_RESPONSE_MAPPING.get(bias, {})
                    if 'control_biased' in mapping:
                        is_biased = 1 if response == mapping['control_biased'] else 0
                    elif 'biased' in mapping:
                        is_biased = 1 if response == mapping['biased'] else 0
                    else:
                        is_biased = None

                    if is_biased is not None:
                        records.append({
                            'model': model,
                            'bias': bias,
                            'variant': variant_id,
                            'treatment': 0,
                            'trial': trial_idx,
                            'biased': is_biased,
                            'obs_id': f"{model}_{bias}_{variant_id}_ctrl_{trial_idx}"
                        })

            # Treatment trials
            for trial_idx, response in enumerate(variant.get('treatment_responses', [])):
                if response in ['A', 'B', 'C']:
                    mapping = BIAS_RESPONSE_MAPPING.get(bias, {})
                    if 'treatment_biased' in mapping:
                        is_biased = 1 if response == mapping['treatment_biased'] else 0
                    elif 'biased' in mapping:
                        is_biased = 1 if response == mapping['biased'] else 0
                    else:
                        is_biased = None

                    if is_biased is not None:
                        records.append({
                            'model': model,
                            'bias': bias,
                            'variant': variant_id,
                            'treatment': 1,
                            'trial': trial_idx,
                            'biased': is_biased,
                            'obs_id': f"{model}_{bias}_{variant_id}_treat_{trial_idx}"
                        })

    return pd.DataFrame(records)


def calculate_icc(df, model_name, bias_name):
    """
    Calculate Intraclass Correlation Coefficient (ICC)
    ICC = var(between variants) / (var(between) + var(within))

    High ICC indicates substantial clustering by variant, justifying random effects.
    """
    subset = df[(df['model'] == model_name) & (df['bias'] == bias_name)].copy()

    if len(subset) < 20:
        return None

    # Calculate variant-level means
    variant_means = subset.groupby('variant')['biased'].mean()

    if len(variant_means) < 2:
        return {'icc': np.nan, 'note': 'Only 1 variant'}

    # Overall mean
    grand_mean = subset['biased'].mean()

    # Between-variant variance
    n_per_variant = subset.groupby('variant').size().mean()
    var_between = np.var(variant_means, ddof=1)

    # Within-variant variance (pooled)
    var_within_list = []
    for variant in subset['variant'].unique():
        v_data = subset[subset['variant'] == variant]['biased']
        if len(v_data) > 1:
            var_within_list.append(np.var(v_data, ddof=1))

    var_within = np.mean(var_within_list) if var_within_list else 0

    # ICC calculation
    if var_between + var_within > 0:
        icc = var_between / (var_between + var_within)
    else:
        icc = 0

    return {
        'model': model_name,
        'bias': bias_name,
        'icc': icc,
        'var_between': var_between,
        'var_within': var_within,
        'n_variants': len(variant_means),
        'n_trials': len(subset)
    }


def fit_glmm_with_diagnostics(df, model_name, bias_name):
    """
    Fit GLMM and compute overdispersion diagnostics.

    Note: statsmodels mixedlm uses linear link, not logistic.
    For proper binary GLMM, we also fit aggregated binomial with quasi-likelihood.
    """
    subset = df[(df['model'] == model_name) & (df['bias'] == bias_name)].copy()
    subset = subset.dropna(subset=['biased'])

    if len(subset) < 20:
        return None

    if subset['biased'].nunique() < 2:
        return {'note': 'No variance in outcome'}

    n_variants = subset['variant'].nunique()

    results = {
        'model': model_name,
        'bias': bias_name,
        'n_trials': len(subset),
        'n_variants': n_variants
    }

    try:
        # Method 1: Standard mixed model (linear approximation)
        if n_variants > 1:
            lmm = smf.mixedlm("biased ~ treatment", subset, groups=subset["variant"])
            lmm_result = lmm.fit(method='powell', maxiter=500)

            results['lmm_coef'] = lmm_result.params.get('treatment', np.nan)
            results['lmm_se'] = lmm_result.bse.get('treatment', np.nan)
            results['lmm_pvalue'] = lmm_result.pvalues.get('treatment', np.nan)

            # Random effects variance
            if hasattr(lmm_result, 'cov_re'):
                results['re_variance'] = float(lmm_result.cov_re.iloc[0, 0]) if hasattr(lmm_result.cov_re, 'iloc') else float(lmm_result.cov_re)

        # Method 2: Aggregated binomial GLM (variant-level)
        agg = subset.groupby(['variant', 'treatment']).agg(
            successes=('biased', 'sum'),
            trials=('biased', 'count')
        ).reset_index()
        agg['failures'] = agg['trials'] - agg['successes']

        if len(agg) >= 4:  # Need enough data points
            # Standard binomial GLM
            glm_binom = sm.GLM(
                agg[['successes', 'failures']],
                sm.add_constant(agg['treatment']),
                family=sm.families.Binomial()
            )
            glm_result = glm_binom.fit()

            results['glm_coef'] = glm_result.params[1]  # treatment coefficient
            results['glm_se'] = glm_result.bse[1]
            results['glm_pvalue'] = glm_result.pvalues[1]
            results['glm_or'] = np.exp(results['glm_coef'])

            # Overdispersion: Pearson chi-square / df
            results['pearson_chi2'] = glm_result.pearson_chi2
            results['df_resid'] = glm_result.df_resid
            results['overdispersion'] = glm_result.pearson_chi2 / glm_result.df_resid if glm_result.df_resid > 0 else np.nan

            # Deviance / df
            results['deviance'] = glm_result.deviance
            results['deviance_ratio'] = glm_result.deviance / glm_result.df_resid if glm_result.df_resid > 0 else np.nan

            # Method 3: Quasi-binomial (accounts for overdispersion)
            # Scale parameter estimated from Pearson chi-square
            scale = results['overdispersion'] if results['overdispersion'] > 0 else 1.0

            # Adjusted standard errors
            results['quasi_se'] = results['glm_se'] * np.sqrt(scale)
            results['quasi_pvalue'] = 2 * (1 - stats.norm.cdf(abs(results['glm_coef'] / results['quasi_se'])))

    except Exception as e:
        results['error'] = str(e)

    return results


def sensitivity_analysis_reduced_trials(df, model_name, bias_name, trial_fractions=[1.0, 0.5, 0.25, 0.1]):
    """
    Sensitivity analysis: how do p-values change with fewer trials?
    If p-values are stable, the effect is robust. If they inflate dramatically
    with more trials, this suggests dependence/overdispersion issues.
    """
    subset = df[(df['model'] == model_name) & (df['bias'] == bias_name)].copy()
    subset = subset.dropna(subset=['biased'])

    if len(subset) < 100:
        return None

    results = []

    for frac in trial_fractions:
        n_sample = int(len(subset) * frac)
        if n_sample < 20:
            continue

        # Sample trials (stratified by variant and treatment)
        sampled_list = []
        for (var, treat), group in subset.groupby(['variant', 'treatment']):
            n_sample_group = max(1, int(len(group) * frac))
            sampled_list.append(group.sample(n=min(len(group), n_sample_group), random_state=42))

        if not sampled_list:
            continue

        sampled = pd.concat(sampled_list, ignore_index=True)

        if len(sampled) < 20:
            continue

        n_variants = sampled['variant'].nunique()

        try:
            if n_variants > 1:
                lmm = smf.mixedlm("biased ~ treatment", sampled, groups=sampled["variant"])
                lmm_result = lmm.fit(method='powell', maxiter=500)

                results.append({
                    'fraction': frac,
                    'n_trials': len(sampled),
                    'coef': lmm_result.params.get('treatment', np.nan),
                    'se': lmm_result.bse.get('treatment', np.nan),
                    'pvalue': lmm_result.pvalues.get('treatment', np.nan)
                })
        except:
            pass

    return results


def run_full_analysis(df):
    """Run complete overdispersion analysis for all model-bias combinations"""

    all_models = df['model'].unique()

    icc_results = []
    glmm_results = []
    sensitivity_results = []

    print("\n" + "="*70)
    print("OVERDISPERSION AND ICC ANALYSIS")
    print("="*70)

    for model in sorted(all_models):
        for bias in CATEGORICAL_BIASES:
            # ICC
            icc = calculate_icc(df, model, bias)
            if icc and 'icc' in icc:
                icc_results.append(icc)

            # GLMM with diagnostics
            glmm = fit_glmm_with_diagnostics(df, model, bias)
            if glmm and 'glm_coef' in glmm:
                glmm_results.append(glmm)

            # Sensitivity (only for cases with enough data)
            sens = sensitivity_analysis_reduced_trials(df, model, bias)
            if sens:
                for s in sens:
                    s['model'] = model
                    s['bias'] = bias
                    sensitivity_results.append(s)

    return icc_results, glmm_results, sensitivity_results


def print_summary(icc_results, glmm_results, sensitivity_results):
    """Print summary tables"""

    print("\n" + "="*70)
    print("1. INTRACLASS CORRELATION COEFFICIENTS (ICC)")
    print("="*70)
    print("ICC measures clustering by variant. High ICC (>0.1) justifies random effects.")
    print("-"*70)

    if icc_results:
        icc_df = pd.DataFrame(icc_results)
        # Summary by bias
        icc_summary = icc_df.groupby('bias')['icc'].agg(['mean', 'std', 'min', 'max']).round(3)
        print("\nICC by bias type:")
        print(icc_summary.to_string())

        # Overall
        print(f"\nOverall mean ICC: {icc_df['icc'].mean():.3f} (SD: {icc_df['icc'].std():.3f})")
        print(f"ICC > 0.1 in {(icc_df['icc'] > 0.1).sum()} of {len(icc_df)} cases ({100*(icc_df['icc'] > 0.1).mean():.1f}%)")

    print("\n" + "="*70)
    print("2. OVERDISPERSION DIAGNOSTICS")
    print("="*70)
    print("Overdispersion ratio = Pearson chi-square / df_resid")
    print("Ratio > 1 indicates overdispersion; ratio >> 1 suggests model misspecification.")
    print("-"*70)

    if glmm_results:
        glmm_df = pd.DataFrame(glmm_results)

        if 'overdispersion' in glmm_df.columns:
            # Filter valid overdispersion values
            valid_od = glmm_df[glmm_df['overdispersion'].notna()].copy()

            if len(valid_od) > 0:
                print("\nOverdispersion ratio by bias:")
                od_summary = valid_od.groupby('bias')['overdispersion'].agg(['mean', 'std', 'min', 'max']).round(3)
                print(od_summary.to_string())

                print(f"\nOverall mean overdispersion: {valid_od['overdispersion'].mean():.3f}")
                print(f"Cases with overdispersion > 1.5: {(valid_od['overdispersion'] > 1.5).sum()} of {len(valid_od)}")
                print(f"Cases with overdispersion > 2.0: {(valid_od['overdispersion'] > 2.0).sum()} of {len(valid_od)}")

        # Compare standard vs quasi-binomial p-values
        if 'glm_pvalue' in glmm_df.columns and 'quasi_pvalue' in glmm_df.columns:
            print("\n" + "-"*70)
            print("Comparison: Standard GLM vs Quasi-binomial p-values")
            print("-"*70)

            comparison = glmm_df[['model', 'bias', 'glm_pvalue', 'quasi_pvalue', 'overdispersion']].dropna()
            comparison['pvalue_ratio'] = comparison['quasi_pvalue'] / comparison['glm_pvalue']

            # Cases where significance changes
            sig_standard = comparison['glm_pvalue'] < 0.05
            sig_quasi = comparison['quasi_pvalue'] < 0.05

            print(f"\nSignificant (p<0.05) with standard GLM: {sig_standard.sum()}")
            print(f"Significant (p<0.05) with quasi-binomial: {sig_quasi.sum()}")
            print(f"Lost significance after quasi correction: {(sig_standard & ~sig_quasi).sum()}")

    print("\n" + "="*70)
    print("3. SENSITIVITY ANALYSIS: P-VALUE STABILITY")
    print("="*70)
    print("Testing if p-values inflate with more trials (suggesting dependence).")
    print("-"*70)

    if sensitivity_results:
        sens_df = pd.DataFrame(sensitivity_results)

        # For each model-bias, check if p-value decreases dramatically with more trials
        print("\nP-value stability by trial fraction (selected examples):")

        # Show a few examples
        examples = sens_df.groupby(['model', 'bias']).apply(
            lambda x: x.sort_values('fraction')[['fraction', 'n_trials', 'pvalue']].head(4)
        ).reset_index(drop=True)

        print(examples.head(20).to_string(index=False))


def save_results(icc_results, glmm_results, sensitivity_results, output_dir):
    """Save results to CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if icc_results:
        pd.DataFrame(icc_results).to_csv(output_path / 'icc_results.csv', index=False)
    if glmm_results:
        pd.DataFrame(glmm_results).to_csv(output_path / 'glmm_overdispersion_results.csv', index=False)
    if sensitivity_results:
        pd.DataFrame(sensitivity_results).to_csv(output_path / 'sensitivity_analysis.csv', index=False)

    print(f"\nResults saved to {output_path}")


def generate_latex_table(icc_results, glmm_results):
    """Generate LaTeX table for appendix"""

    print("\n" + "="*70)
    print("LATEX TABLE FOR APPENDIX")
    print("="*70)

    if not glmm_results:
        return

    glmm_df = pd.DataFrame(glmm_results)

    print("""
\\begin{table}[h]
\\centering
\\caption{Overdispersion diagnostics and ICC for GLMM analysis. Overdispersion ratio $>1$ indicates extra-binomial variation; quasi-binomial p-values account for this inflation.}
\\label{tab:overdispersion}
\\small
\\begin{tabular}{llcccc}
\\toprule
Bias & Mean ICC & Mean Overdisp. & Sig. (GLM) & Sig. (Quasi) & Lost Sig. \\\\
\\midrule""")

    if icc_results:
        icc_df = pd.DataFrame(icc_results)

        for bias in CATEGORICAL_BIASES:
            bias_icc = icc_df[icc_df['bias'] == bias]['icc'].mean()
            bias_glmm = glmm_df[glmm_df['bias'] == bias]

            if len(bias_glmm) > 0 and 'overdispersion' in bias_glmm.columns:
                od = bias_glmm['overdispersion'].mean()
                n_sig_glm = (bias_glmm['glm_pvalue'] < 0.05).sum() if 'glm_pvalue' in bias_glmm.columns else 0
                n_sig_quasi = (bias_glmm['quasi_pvalue'] < 0.05).sum() if 'quasi_pvalue' in bias_glmm.columns else 0
                lost = n_sig_glm - n_sig_quasi

                bias_display = bias.replace('_', ' ').title()
                print(f"{bias_display} & {bias_icc:.3f} & {od:.2f} & {n_sig_glm} & {n_sig_quasi} & {lost} \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}""")


def main():
    print("Loading data...")
    results = load_all_results()
    print(f"Loaded {len(results)} experiment files")

    print("\nBuilding trial-level dataset...")
    df = build_trial_level_data(results)
    print(f"Built dataset with {len(df)} trials")

    print("\nRunning analyses...")
    icc_results, glmm_results, sensitivity_results = run_full_analysis(df)

    print_summary(icc_results, glmm_results, sensitivity_results)
    generate_latex_table(icc_results, glmm_results)

    # Save results
    output_dir = RAW_RESPONSES_DIR.parent / 'overdispersion_analysis'
    save_results(icc_results, glmm_results, sensitivity_results, output_dir)

    return df, icc_results, glmm_results, sensitivity_results


if __name__ == "__main__":
    df, icc_results, glmm_results, sensitivity_results = main()
