"""
Compute bootstrap confidence intervals for debiasing results.
Run in Spyder to get CIs for the debiasing table.
"""

import json
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw_responses"
N_BOOTSTRAP = 10000
ALPHA = 0.05  # 95% CI

def load_debiasing_data(bias_name, strategy):
    """Load debiasing results for a specific bias and strategy."""
    filename = f"{bias_name}_llama-3.1-8b_{strategy}.json"
    filepath = RESULTS_DIR / filename

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None

    with open(filepath, 'r') as f:
        return json.load(f)

def compute_bsi_categorical(control_responses, treatment_responses, bias_option='B'):
    """Compute BSI for categorical bias (shift toward bias option)."""
    # Clean responses
    control = [r.strip().upper().replace('.', '') for r in control_responses if r]
    treatment = [r.strip().upper().replace('.', '') for r in treatment_responses if r]

    # Filter valid responses (A or B)
    control = [r for r in control if r in ['A', 'B']]
    treatment = [r for r in treatment if r in ['A', 'B']]

    if not control or not treatment:
        return np.nan

    p_control = sum(1 for r in control if r == bias_option) / len(control)
    p_treatment = sum(1 for r in treatment if r == bias_option) / len(treatment)

    return abs(p_treatment - p_control)

def bootstrap_bsi(control_responses, treatment_responses, bias_option='B', n_bootstrap=10000):
    """Compute bootstrap distribution of BSI."""
    # Clean responses
    control = [r.strip().upper().replace('.', '') for r in control_responses if r]
    treatment = [r.strip().upper().replace('.', '') for r in treatment_responses if r]

    # Filter valid responses
    control = [r for r in control if r in ['A', 'B']]
    treatment = [r for r in treatment if r in ['A', 'B']]

    if len(control) < 10 or len(treatment) < 10:
        return None

    bsi_samples = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        ctrl_sample = np.random.choice(control, size=len(control), replace=True)
        treat_sample = np.random.choice(treatment, size=len(treatment), replace=True)

        p_ctrl = sum(1 for r in ctrl_sample if r == bias_option) / len(ctrl_sample)
        p_treat = sum(1 for r in treat_sample if r == bias_option) / len(treat_sample)

        bsi_samples.append(abs(p_treat - p_ctrl))

    return np.array(bsi_samples)

def analyze_debiasing_with_cis():
    """Analyze all debiasing results with confidence intervals."""

    biases = ['decoy_effect', 'framing_effect', 'primacy_recency']
    strategies = ['D1_cot', 'D3_counter', 'D4_role']

    # Bias option mapping (which option indicates bias)
    bias_options = {
        'decoy_effect': 'B',      # B is the target option with decoy
        'framing_effect': 'B',    # B is risk-seeking (loss frame bias)
        'primacy_recency': 'B',   # B is negative judgment (primacy bias)
    }

    results = []

    for bias in biases:
        print(f"\n{'='*60}")
        print(f"BIAS: {bias}")
        print('='*60)

        for strategy in strategies:
            data = load_debiasing_data(bias, strategy)
            if data is None:
                continue

            # Aggregate across variants
            all_control = []
            all_treatment = []

            for variant in data.get('variants', []):
                all_control.extend(variant.get('control_responses', []))
                all_treatment.extend(variant.get('treatment_responses', []))

            if not all_control or not all_treatment:
                print(f"  {strategy}: No data")
                continue

            # Compute BSI
            bias_opt = bias_options.get(bias, 'B')
            bsi = compute_bsi_categorical(all_control, all_treatment, bias_opt)

            # Bootstrap CI
            bootstrap_dist = bootstrap_bsi(all_control, all_treatment, bias_opt, N_BOOTSTRAP)

            if bootstrap_dist is not None:
                ci_lower = np.percentile(bootstrap_dist, 100 * ALPHA / 2)
                ci_upper = np.percentile(bootstrap_dist, 100 * (1 - ALPHA / 2))
                se = np.std(bootstrap_dist)

                print(f"  {strategy}: BSI = {bsi:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] (SE = {se:.3f})")

                results.append({
                    'bias': bias,
                    'strategy': strategy,
                    'bsi': bsi,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'se': se,
                    'n_control': len([r for r in all_control if r.strip()]),
                    'n_treatment': len([r for r in all_treatment if r.strip()])
                })
            else:
                print(f"  {strategy}: Insufficient data for bootstrap")

    return results

def generate_latex_table(results):
    """Generate LaTeX table with CIs."""
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)

    # Organize by bias
    biases = ['decoy_effect', 'framing_effect', 'primacy_recency']
    strategies = ['D1_cot', 'D3_counter', 'D4_role']

    # Baseline BSI values (from previous analysis)
    baselines = {
        'decoy_effect': 0.892,
        'framing_effect': 0.700,
        'primacy_recency': 0.966
    }

    bias_names = {
        'decoy_effect': 'Decoy',
        'framing_effect': 'Framing',
        'primacy_recency': 'Primacy/Rec.'
    }

    strategy_names = {
        'D1_cot': 'CoT',
        'D3_counter': 'Counter',
        'D4_role': 'Role'
    }

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Effect of debiasing strategies on Llama 3.1 8B with 95\% bootstrap confidence intervals. Values are BSI (lower = less biased). Best result per bias shown in bold.}")
    print(r"\label{tab:debiasing}")
    print(r"\small")
    print(r"\begin{tabular}{@{}lccccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Bias} & \textbf{Baseline} & \textbf{CoT} & \textbf{Counter} & \textbf{Role} & \textbf{Best} \\")
    print(r"\midrule")

    for bias in biases:
        baseline = baselines[bias]
        row_results = {s: None for s in strategies}

        for r in results:
            if r['bias'] == bias:
                row_results[r['strategy']] = r

        # Find best (lowest BSI)
        valid_results = [(s, r) for s, r in row_results.items() if r is not None]
        if valid_results:
            best_strategy = min(valid_results, key=lambda x: x[1]['bsi'])[0]
            best_reduction = (1 - min(r['bsi'] for _, r in valid_results) / baseline) * 100
        else:
            best_strategy = None
            best_reduction = 0

        # Build row
        row = f"{bias_names[bias]} & ${baseline:.2f}$"

        for strategy in strategies:
            r = row_results[strategy]
            if r is not None:
                bsi_str = f"{r['bsi']:.2f}"
                ci_str = f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"

                if strategy == best_strategy:
                    row += f" & $\\mathbf{{{bsi_str}}}$ {ci_str}"
                else:
                    row += f" & ${bsi_str}$ {ci_str}"
            else:
                row += " & --"

        row += f" & $-{best_reduction:.0f}\\%$ \\\\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    print("Computing bootstrap CIs for debiasing results...")
    print(f"Using {N_BOOTSTRAP} bootstrap samples, {100*(1-ALPHA):.0f}% CI")

    results = analyze_debiasing_with_cis()

    if results:
        generate_latex_table(results)
    else:
        print("No results to display.")
