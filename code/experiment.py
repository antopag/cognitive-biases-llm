"""Main experiment runner for cognitive bias evaluation."""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from biases import ALL_BIASES
from models import query_model
from parsing import parse_response
from debiasing import (apply_cot, apply_self_reflection, apply_counter_prompt,
                       apply_role_prompt, ensemble_debias)


def run_single_bias_experiment(bias: dict, model_name: str,
                               strategy: str = "baseline",
                               n_trials: int = None,
                               temperature: float = None) -> dict:
    """Run experiment for a single bias on a single model.

    Args:
        bias: Bias definition dict from biases.py.
        model_name: Key from config.MODELS.
        strategy: One of "baseline", "D1_cot", "D2_reflection",
                  "D3_counter", "D4_role".
        n_trials: Trials per variant per condition (default from config).
        temperature: Sampling temperature (default from config).

    Returns:
        Dict with results for each variant, including raw responses and parsed values.
    """
    if n_trials is None:
        n_trials = config.N_TRIALS
    if temperature is None:
        temperature = config.TEMPERATURE

    bias_name = bias["name"]
    results = {
        "bias_name": bias_name,
        "model_name": model_name,
        "strategy": strategy,
        "timestamp": datetime.now().isoformat(),
        "variants": [],
    }

    for variant in tqdm(bias["variants"], desc=f"{bias_name}", leave=False):
        variant_id = variant["id"]
        parse_type = variant["parse_type"]
        options = variant.get("options")

        variant_result = {
            "variant_id": variant_id,
            "control_responses": [],
            "treatment_responses": [],
            "control_parsed": [],
            "treatment_parsed": [],
            "parse_failures_control": 0,
            "parse_failures_treatment": 0,
        }

        # Run control and treatment conditions
        for condition in ["control", "treatment"]:
            prompt_key = condition
            base_prompt = variant[prompt_key]

            for trial in range(n_trials):
                # Apply debiasing strategy to prompt
                prompt = base_prompt
                if strategy == "D1_cot":
                    prompt = apply_cot(prompt)
                elif strategy == "D3_counter":
                    prompt = apply_counter_prompt(prompt, bias_name)
                elif strategy == "D4_role":
                    prompt = apply_role_prompt(prompt)

                # Use shorter max_tokens for simple answers
                # Note: Gemini 2.5 uses extra tokens for "thinking", so need higher limit
                mt = 150 if parse_type in ("number", "choice") else config.MAX_TOKENS

                # Groq free tier needs longer delays
                provider = config.MODELS[model_name]["provider"]
                delay = 3 if provider == "groq" else 1

                try:
                    response = query_model(model_name, prompt,
                                           temperature=temperature,
                                           max_tokens=mt)
                    time.sleep(delay)  # rate limit protection
                except Exception as e:
                    response = f"ERROR: {e}"
                    time.sleep(delay * 5)  # longer pause after errors

                # D2: self-reflection requires a second call
                if strategy == "D2_reflection" and "ERROR" not in response:
                    try:
                        response = apply_self_reflection(
                            model_name, prompt, response,
                            temperature=temperature)
                    except Exception as e:
                        pass  # keep original response

                # Parse response
                parsed = parse_response(response, parse_type, options)

                variant_result[f"{condition}_responses"].append(response)
                if parsed is not None:
                    variant_result[f"{condition}_parsed"].append(parsed)
                else:
                    variant_result[f"parse_failures_{condition}"] += 1

            # Brief pause between conditions to avoid rate limits
            time.sleep(1)

        results["variants"].append(variant_result)

    return results


def run_ensemble_experiment(bias: dict, model_name: str,
                            n_trials: int = None) -> dict:
    """Run ensemble debiasing experiment for a single bias.

    For each trial, aggregate across all prompt variants and temperatures.
    """
    if n_trials is None:
        n_trials = config.N_TRIALS

    bias_name = bias["name"]
    results = {
        "bias_name": bias_name,
        "model_name": model_name,
        "strategy": "D5_ensemble",
        "timestamp": datetime.now().isoformat(),
        "ensemble_results": [],
    }

    # Collect treatment prompts from all variants
    treatment_prompts = [v["treatment"] for v in bias["variants"]]
    control_prompts = [v["control"] for v in bias["variants"]]

    parse_type = bias["variants"][0]["parse_type"]
    options = bias["variants"][0].get("options")

    for trial in tqdm(range(n_trials), desc=f"{bias_name} ensemble",
                      leave=False):
        # Ensemble for treatment
        treat_result = ensemble_debias(
            model_name, treatment_prompts,
            config.ENSEMBLE_TEMPERATURES,
            parse_type, options)

        # Ensemble for control
        ctrl_result = ensemble_debias(
            model_name, control_prompts,
            config.ENSEMBLE_TEMPERATURES,
            parse_type, options)

        results["ensemble_results"].append({
            "trial": trial,
            "control_answer": ctrl_result["answer"],
            "treatment_answer": treat_result["answer"],
            "control_n_members": ctrl_result["n_members"],
            "treatment_n_members": treat_result["n_members"],
        })

    return results


def save_results(results: dict, output_dir: str = None):
    """Save experiment results to JSON file."""
    if output_dir is None:
        output_dir = config.RAW_RESPONSES_DIR

    filename = (f"{results['bias_name']}_{results['model_name']}_"
                f"{results['strategy']}.json")
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"Saved: {filepath}")
    return filepath


def run_full_experiment(models: list[str] = None,
                       biases: list[dict] = None,
                       strategies: list[str] = None,
                       n_trials: int = None):
    """Run the full experiment across all models, biases, and strategies.

    Args:
        models: List of model names (default: all from config).
        biases: List of bias dicts (default: all 12).
        strategies: List of strategy names (default: baseline only).
        n_trials: Trials per configuration (default from config).
    """
    if models is None:
        models = list(config.MODELS.keys())
    if biases is None:
        biases = ALL_BIASES
    if strategies is None:
        strategies = ["baseline"]
    if n_trials is None:
        n_trials = config.N_TRIALS

    total = len(models) * len(biases) * len(strategies)
    print(f"Running {total} experiments ({len(models)} models x "
          f"{len(biases)} biases x {len(strategies)} strategies)")
    print(f"Trials per configuration: {n_trials}")
    print(f"Estimated total API calls: ~{total * n_trials * 2 * 4}")
    print()

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for strategy in strategies:
            print(f"\n  Strategy: {strategy}")

            for bias in tqdm(biases, desc=f"  {model_name}/{strategy}"):
                # Skip if results already exist
                filename = (f"{bias['name']}_{model_name}_"
                            f"{strategy}.json")
                filepath = os.path.join(config.RAW_RESPONSES_DIR, filename)
                if os.path.exists(filepath):
                    tqdm.write(f"    Skipping {bias['name']} (already done)")
                    continue

                if strategy == "D5_ensemble":
                    results = run_ensemble_experiment(
                        bias, model_name, n_trials=n_trials)
                else:
                    results = run_single_bias_experiment(
                        bias, model_name, strategy=strategy,
                        n_trials=n_trials)

                save_results(results)

    print("\nExperiment complete!")


# --- Entry point ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run cognitive bias experiments on LLMs")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to test (default: all)")
    parser.add_argument("--biases", nargs="+", default=None,
                        help="Bias names to test (default: all)")
    parser.add_argument("--strategies", nargs="+",
                        default=["baseline"],
                        help="Debiasing strategies (default: baseline)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of trials per configuration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment plan without running")

    args = parser.parse_args()

    # Filter biases by name if specified
    if args.biases:
        selected = [b for b in ALL_BIASES if b["name"] in args.biases]
    else:
        selected = ALL_BIASES

    if args.dry_run:
        models = args.models or list(config.MODELS.keys())
        strategies = args.strategies
        n = args.n_trials or config.N_TRIALS
        total_calls = 0
        for m in models:
            for s in strategies:
                for b in selected:
                    n_variants = len(b["variants"])
                    calls = n_variants * 2 * n  # variants * conditions * trials
                    if s == "D2_reflection":
                        calls *= 2  # double for reflection
                    if s == "D5_ensemble":
                        calls = n * n_variants * len(config.ENSEMBLE_TEMPERATURES) * 2
                    total_calls += calls
        print(f"Dry run summary:")
        print(f"  Models: {models}")
        print(f"  Biases: {[b['name'] for b in selected]}")
        print(f"  Strategies: {strategies}")
        print(f"  Trials per config: {n}")
        print(f"  Estimated total API calls: {total_calls}")
    else:
        run_full_experiment(
            models=args.models,
            biases=selected,
            strategies=args.strategies,
            n_trials=args.n_trials,
        )
