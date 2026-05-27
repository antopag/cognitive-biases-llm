"""Run full baseline experiment on all free models."""

import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment import run_full_experiment

# Models that work without paid credits
FREE_MODELS = [
    "gemini-2.5-flash",
    "llama-3.3-70b",
    "llama-3.1-8b",
    "gemma-2-9b",
    "mistral-large",
]

run_full_experiment(
    models=FREE_MODELS,
    strategies=["baseline"],
    n_trials=100,
)
