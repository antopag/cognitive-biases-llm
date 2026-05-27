"""Quick test: 1 numerical + 1 categorical bias on all free models (3 trials)."""

import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biases import ANCHORING, FRAMING
from experiment import run_single_bias_experiment, save_results

FREE_MODELS = [
    "gemini-2.5-flash",
    "llama-3.3-70b",
    "llama-3.1-8b",
    "gemma-2-9b",
    "mistral-large",
]

for model_name in FREE_MODELS:
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")

    # Test anchoring (numerical)
    result = run_single_bias_experiment(ANCHORING, model_name, n_trials=3)
    save_results(result)
    v = result["variants"][0]  # just show first variant
    print(f"  Anchoring v1: ctrl={v['control_parsed'][:3]}  treat={v['treatment_parsed'][:3]}  fail={v['parse_failures_control']+v['parse_failures_treatment']}")

    # Test framing (categorical)
    result = run_single_bias_experiment(FRAMING, model_name, n_trials=3)
    save_results(result)
    v = result["variants"][0]
    print(f"  Framing v1:   ctrl={v['control_parsed'][:3]}  treat={v['treatment_parsed'][:3]}  fail={v['parse_failures_control']+v['parse_failures_treatment']}")

print("\n\nDONE!")
