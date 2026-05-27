"""Quick test: anchoring bias on all working models (3 trials each)."""

import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biases import ANCHORING
from experiment import run_single_bias_experiment, save_results

WORKING_MODELS = ["gemini-2.5-flash", "llama-3.3-70b", "mistral-large"]

for model_name in WORKING_MODELS:
    print(f"\n{'='*50}")
    print(f"  Testing: {model_name}")
    print(f"{'='*50}")

    result = run_single_bias_experiment(ANCHORING, model_name, n_trials=3)
    save_results(result)

    for v in result["variants"]:
        print(f"\n  {v['variant_id']}:")
        print(f"    Control:   {v['control_parsed']}")
        print(f"    Treatment: {v['treatment_parsed']}")
        print(f"    Failures:  ctrl={v['parse_failures_control']}, treat={v['parse_failures_treatment']}")

print("\n\nDONE!")
