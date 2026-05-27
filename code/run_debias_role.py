"""Run D4_role debiasing strategy on Llama-3.1-8b."""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiment import run_full_experiment
from biases import DECOY, FRAMING, PRIMACY_RECENCY

print("=" * 60)
print("DEBIASING: D4_role (Role-Based Prompting)")
print("Model: llama-3.1-8b")
print("Biases: decoy_effect, framing_effect, primacy_recency")
print("=" * 60)

run_full_experiment(
    models=["llama-3.1-8b"],
    biases=[DECOY, FRAMING, PRIMACY_RECENCY],
    strategies=["D4_role"],
    n_trials=100
)

print("\n*** D4_role COMPLETE ***")
