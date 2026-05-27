"""Run D1_cot debiasing strategy on Llama-3.1-8b."""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiment import run_full_experiment
from biases import DECOY, FRAMING, PRIMACY_RECENCY

print("=" * 60)
print("DEBIASING: D1_cot (Chain-of-Thought)")
print("Model: llama-3.1-8b")
print("Biases: decoy_effect, framing_effect, primacy_recency")
print("=" * 60)

run_full_experiment(
    models=["llama-3.1-8b"],
    biases=[DECOY, FRAMING, PRIMACY_RECENCY],
    strategies=["D1_cot"],
    n_trials=100
)

print("\n*** D1_cot COMPLETE ***")
