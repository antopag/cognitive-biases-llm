"""Production run: full baseline experiment on all working models."""

import os, sys
code_dir = r"G:\My Drive\Papers\PAPERS (MY)\Cognitive biases LLM\code"
os.chdir(code_dir)
sys.path.insert(0, code_dir)

from experiment import run_full_experiment

# --- Phase 1: Fast models (1s delay) ---
FAST_MODELS = [
    "mistral-large",       # free, fast
    "gemini-2.5-flash",    # paid, fast (quota issues)
    "gpt-4.1-mini",        # paid, fast (replaced gpt-4o due to rate limits)
    "claude-3.5-sonnet",   # paid, fast
]

print("=" * 60)
print("  PHASE 1: Fast models (N=100)")
print("=" * 60)

run_full_experiment(
    models=FAST_MODELS,
    strategies=["baseline"],
    n_trials=100,
)

# --- Phase 2: Groq models (3s delay, fewer trials) ---
GROQ_MODELS = [
    "llama-3.3-70b",
    "llama-3.1-8b",
]

print("\n" + "=" * 60)
print("  PHASE 2: Groq models (N=30)")
print("=" * 60)

run_full_experiment(
    models=GROQ_MODELS,
    strategies=["baseline"],
    n_trials=30,
)
