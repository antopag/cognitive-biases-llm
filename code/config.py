"""Configuration for the cognitive biases experiment."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# --- Models ---
MODELS = {
    "gpt-4.1-mini": {
        "provider": "openai",
        "model_id": "gpt-4.1-mini",
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "model_id": "gemini-2.5-flash",
    },
    "llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
    },
    "llama-3.1-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
    },
    "deepseek-v3": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
    },
    "mistral-large": {
        "provider": "mistral",
        "model_id": "mistral-large-latest",
    },
    "deepseek-v3": {
        "provider": "together",
        "model_id": "deepseek-ai/DeepSeek-V3",
    },
    "minimax-m2.5": {
        "provider": "together",
        "model_id": "MiniMaxAI/MiniMax-M2.5",
    },
}

# --- Experiment Parameters ---
N_TRIALS = 100          # trials per prompt variant per condition
TEMPERATURE = 0.7       # default sampling temperature
MAX_TOKENS = 512        # max output tokens
BOOTSTRAP_RESAMPLES = 10_000
ALPHA = 0.05            # significance level

# Temperature settings for ensemble debiasing
ENSEMBLE_TEMPERATURES = [0.3, 0.7, 1.0]

# --- Output Paths ---
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RAW_RESPONSES_DIR = os.path.join(RESULTS_DIR, "raw_responses")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

for d in [RESULTS_DIR, RAW_RESPONSES_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
