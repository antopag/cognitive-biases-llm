"""
Parse Failure Analysis for Cognitive Biases Paper
Analyzes parse failure rates by model, bias, and condition (control vs treatment)
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
from scipy import stats

# Path to raw responses
RAW_RESPONSES_DIR = Path(r"G:\My Drive\Papers\PAPERS (MY)\Cognitive biases LLM\results\raw_responses")

def load_all_results():
    """Load all baseline JSON files"""
    results = []

    for json_file in RAW_RESPONSES_DIR.glob("*_baseline.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)

    return results

def analyze_parse_failures(results):
    """Analyze parse failures by model, bias, and condition"""

    records = []

    for result in results:
        model = result['model_name']
        bias = result['bias_name']

        for variant in result.get('variants', []):
            variant_id = variant.get('variant_id', 'unknown')

            # Get explicit failure counts from JSON
            ctrl_failures = variant.get('parse_failures_control', 0)
            treat_failures = variant.get('parse_failures_treatment', 0)

            # Calculate totals from responses arrays
            ctrl_responses = variant.get('control_responses', [])
            treat_responses = variant.get('treatment_responses', [])
            ctrl_total = len(ctrl_responses) if ctrl_responses else 100  # default 100 trials
            treat_total = len(treat_responses) if treat_responses else 100

            records.append({
                'model': model,
                'bias': bias,
                'variant': variant_id,
                'condition': 'control',
                'failures': ctrl_failures,
                'total': ctrl_total,
                'failure_rate': ctrl_failures / ctrl_total if ctrl_total > 0 else 0
            })

            records.append({
                'model': model,
                'bias': bias,
                'variant': variant_id,
                'condition': 'treatment',
                'failures': treat_failures,
                'total': treat_total,
                'failure_rate': treat_failures / treat_total if treat_total > 0 else 0
            })

    return pd.DataFrame(records)

def create_summary_tables(df):
    """Create summary tables for the paper"""

    # 1. Overall failure rate by model
    model_summary = df.groupby('model').agg({
        'failures': 'sum',
        'total': 'sum'
    }).reset_index()
    model_summary['failure_rate'] = model_summary['failures'] / model_summary['total']
    model_summary = model_summary.sort_values('failure_rate', ascending=False)

    print("\n" + "="*60)
    print("TABLE 1: Parse Failure Rate by Model")
    print("="*60)
    print(model_summary.to_string(index=False))

    # 2. Failure rate by model and condition
    model_cond = df.groupby(['model', 'condition']).agg({
        'failures': 'sum',
        'total': 'sum'
    }).reset_index()
    model_cond['failure_rate'] = model_cond['failures'] / model_cond['total']

    # Pivot for easier reading
    pivot_cond = model_cond.pivot(index='model', columns='condition', values='failure_rate')
    pivot_cond['diff'] = pivot_cond['treatment'] - pivot_cond['control']

    print("\n" + "="*60)
    print("TABLE 2: Parse Failure Rate by Model and Condition")
    print("="*60)
    print(pivot_cond.round(3).to_string())

    # 3. Chi-square test: failures differ between control/treatment?
    print("\n" + "="*60)
    print("TABLE 3: Chi-Square Test (Control vs Treatment Failures)")
    print("="*60)

    chi_results = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]

        ctrl = model_data[model_data['condition'] == 'control']
        treat = model_data[model_data['condition'] == 'treatment']

        ctrl_fail = ctrl['failures'].sum()
        ctrl_success = ctrl['total'].sum() - ctrl_fail
        treat_fail = treat['failures'].sum()
        treat_success = treat['total'].sum() - treat_fail

        # Contingency table
        contingency = [[ctrl_fail, ctrl_success], [treat_fail, treat_success]]

        if ctrl_fail + treat_fail > 0:  # Only test if there are failures
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            chi_results.append({
                'model': model,
                'ctrl_fail_rate': ctrl_fail / (ctrl_fail + ctrl_success),
                'treat_fail_rate': treat_fail / (treat_fail + treat_success),
                'chi2': chi2,
                'p_value': p,
                'significant': 'Yes' if p < 0.05 else 'No'
            })
        else:
            chi_results.append({
                'model': model,
                'ctrl_fail_rate': 0,
                'treat_fail_rate': 0,
                'chi2': 0,
                'p_value': 1.0,
                'significant': 'No'
            })

    chi_df = pd.DataFrame(chi_results)
    print(chi_df.to_string(index=False))

    # 4. Failure rate by bias
    bias_summary = df.groupby('bias').agg({
        'failures': 'sum',
        'total': 'sum'
    }).reset_index()
    bias_summary['failure_rate'] = bias_summary['failures'] / bias_summary['total']
    bias_summary = bias_summary.sort_values('failure_rate', ascending=False)

    print("\n" + "="*60)
    print("TABLE 4: Parse Failure Rate by Bias Type")
    print("="*60)
    print(bias_summary.to_string(index=False))

    # 5. Detailed: Model x Bias matrix
    model_bias = df.groupby(['model', 'bias']).agg({
        'failures': 'sum',
        'total': 'sum'
    }).reset_index()
    model_bias['failure_rate'] = model_bias['failures'] / model_bias['total']

    pivot_mb = model_bias.pivot(index='bias', columns='model', values='failure_rate')

    print("\n" + "="*60)
    print("TABLE 5: Parse Failure Rate Matrix (Bias x Model)")
    print("="*60)
    print(pivot_mb.round(2).to_string())

    # 6. Detailed breakdown by model, bias, condition
    detailed = df.groupby(['model', 'bias', 'condition']).agg({
        'failures': 'sum',
        'total': 'sum'
    }).reset_index()
    detailed['failure_rate'] = detailed['failures'] / detailed['total']

    print("\n" + "="*60)
    print("TABLE 6: Detailed Breakdown (Model x Bias x Condition)")
    print("="*60)

    # Find cases where control/treatment differ significantly
    significant_diffs = []
    for model in detailed['model'].unique():
        for bias in detailed['bias'].unique():
            subset = detailed[(detailed['model'] == model) & (detailed['bias'] == bias)]
            if len(subset) == 2:
                ctrl = subset[subset['condition'] == 'control']['failure_rate'].values
                treat = subset[subset['condition'] == 'treatment']['failure_rate'].values
                if len(ctrl) > 0 and len(treat) > 0:
                    diff = treat[0] - ctrl[0]
                    if abs(diff) > 0.05:  # >5% difference
                        significant_diffs.append({
                            'model': model,
                            'bias': bias,
                            'ctrl_fail': ctrl[0],
                            'treat_fail': treat[0],
                            'diff': diff
                        })

    if significant_diffs:
        sig_df = pd.DataFrame(significant_diffs).sort_values('diff', key=abs, ascending=False)
        print("\nCases with >5% difference between control and treatment:")
        print(sig_df.round(3).to_string(index=False))
    else:
        print("\nNo cases with >5% difference between control and treatment.")

    return {
        'model_summary': model_summary,
        'model_condition': pivot_cond,
        'chi_tests': chi_df,
        'bias_summary': bias_summary,
        'model_bias_matrix': pivot_mb,
        'significant_diffs': significant_diffs
    }

def save_results(summaries, output_dir):
    """Save results to CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    summaries['model_summary'].to_csv(output_path / 'parse_failures_by_model.csv', index=False)
    summaries['model_condition'].to_csv(output_path / 'parse_failures_by_model_condition.csv')
    summaries['chi_tests'].to_csv(output_path / 'parse_failures_chi_tests.csv', index=False)
    summaries['bias_summary'].to_csv(output_path / 'parse_failures_by_bias.csv', index=False)
    summaries['model_bias_matrix'].to_csv(output_path / 'parse_failures_matrix.csv')

    if summaries['significant_diffs']:
        pd.DataFrame(summaries['significant_diffs']).to_csv(
            output_path / 'parse_failures_significant_diffs.csv', index=False
        )

    print(f"\nResults saved to {output_path}")

def main():
    print("Loading data...")
    results = load_all_results()
    print(f"Loaded {len(results)} experiment files")

    print("\nAnalyzing parse failures...")
    df = analyze_parse_failures(results)
    print(f"Total records: {len(df)}")

    summaries = create_summary_tables(df)

    # Save results
    output_dir = RAW_RESPONSES_DIR.parent / 'parse_failure_analysis'
    save_results(summaries, output_dir)

    return df, summaries

if __name__ == "__main__":
    df, summaries = main()
