"""Implementation of five debiasing strategies."""

import numpy as np
from models import query_model
from parsing import parse_response


# --- D1: Zero-shot Chain-of-Thought ---

def apply_cot(prompt: str) -> str:
    """Append chain-of-thought instruction to prompt."""
    return prompt.rstrip() + "\n\nLet's think step by step."


# --- D2: Structured Self-Reflection ---

REFLECTION_PROMPT = (
    "Review your previous answer below. Could any cognitive bias "
    "(such as anchoring, framing, confirmation bias, availability heuristic, "
    "sunk cost fallacy, bandwagon effect, or others) have influenced your response? "
    "If so, identify the bias and provide a corrected answer.\n\n"
    "Your previous answer: {previous_answer}\n\n"
    "Provide your corrected answer in the same format as the original question."
)


def apply_self_reflection(model_name: str, prompt: str, initial_response: str,
                          temperature: float = 0.7) -> str:
    """Query model with self-reflection on its initial response."""
    reflection = REFLECTION_PROMPT.format(previous_answer=initial_response)
    full_prompt = prompt + "\n\n" + reflection
    return query_model(model_name, full_prompt, temperature=temperature)


# --- D3: Adversarial Counter-Prompting ---

COUNTER_PROMPTS = {
    "anchoring": "Be aware that this question contains a reference number that may act as an anchor. Ignore any irrelevant numerical suggestions and base your answer solely on your knowledge.",
    "availability_heuristic": "Be aware that vivid or recent examples mentioned in this question may trigger the availability heuristic. Base your answer on actual statistical data, not on how easily examples come to mind.",
    "representativeness": "Be aware that descriptive details may trigger the representativeness heuristic. Consider base rates and statistical probability rather than stereotypical similarity.",
    "overconfidence": "Be aware of the tendency toward overconfidence. Ensure your confidence interval is wide enough to genuinely reflect your uncertainty.",
    "framing_effect": "Be aware that this question may frame information in a way designed to influence your judgment. Consider the objective facts regardless of how they are presented.",
    "sunk_cost_fallacy": "Be aware that past investments (time, money, effort) mentioned in this question are sunk costs and should not influence forward-looking decisions. Focus only on future costs and benefits.",
    "status_quo_bias": "Be aware that the current state described may trigger status quo bias. Evaluate all options on their merits, regardless of which is the default or current choice.",
    "decoy_effect": "Be aware that one of the options may serve as a decoy designed to make another option look more attractive. Evaluate each option independently on its own merits.",
    "confirmation_bias": "Be aware that a prior belief or perspective is suggested in this question. Evaluate all evidence objectively, giving appropriate weight to study quality and sample size, regardless of whether findings confirm or disconfirm the suggested belief.",
    "bandwagon_effect": "Be aware that information about what others think or do is included in this question. Form your own independent judgment based on the merits of the issue, not on popularity or majority opinion.",
    "authority_bias": "Be aware that an authority figure or prestigious institution is mentioned. Evaluate the claim based on the quality of the evidence, not on who made the claim.",
    "primacy_recency": "Be aware that the order in which information is presented can influence judgment. Give equal weight to all items regardless of their position in the list.",
}


def apply_counter_prompt(prompt: str, bias_name: str) -> str:
    """Prepend bias-specific warning to prompt."""
    warning = COUNTER_PROMPTS.get(bias_name, "")
    if warning:
        return warning + "\n\n" + prompt
    return prompt


# --- D4: Role-Based Prompting ---

ROLE_PROMPT = (
    "You are a rational decision analyst trained in behavioral economics "
    "and cognitive psychology. You are an expert at identifying and avoiding "
    "cognitive biases in reasoning. Provide your most objective, unbiased assessment."
)


def apply_role_prompt(prompt: str) -> str:
    """Prepend rational analyst role to prompt."""
    return ROLE_PROMPT + "\n\n" + prompt


# --- D5: Ensemble Debiasing ---

def ensemble_debias(model_name: str, prompts: list[str],
                    temperatures: list[float],
                    parse_type: str,
                    options: list[str] = None) -> dict:
    """Apply ensemble debiasing by aggregating across prompt variants and temperatures.

    Args:
        model_name: The model to query.
        prompts: List of semantically equivalent prompt variants.
        temperatures: List of temperature values to sample at.
        parse_type: "number", "choice", or "interval".
        options: For choice type, the valid options.

    Returns:
        Dict with 'answer' (aggregated), 'raw_answers' (all individual), 'n_members'.
    """
    raw_answers = []

    for prompt in prompts:
        for temp in temperatures:
            response = query_model(model_name, prompt, temperature=temp)
            parsed = parse_response(response, parse_type, options)
            if parsed is not None:
                raw_answers.append(parsed)

    if not raw_answers:
        return {"answer": None, "raw_answers": [], "n_members": 0}

    if parse_type == "number":
        answer = float(np.median(raw_answers))
    elif parse_type == "choice":
        # Majority voting
        from collections import Counter
        counts = Counter(raw_answers)
        answer = counts.most_common(1)[0][0]
    elif parse_type == "interval":
        lows = [a[0] for a in raw_answers]
        highs = [a[1] for a in raw_answers]
        answer = (float(np.median(lows)), float(np.median(highs)))
    else:
        answer = raw_answers[0]

    return {
        "answer": answer,
        "raw_answers": raw_answers,
        "n_members": len(raw_answers),
    }


# --- Strategy registry ---

STRATEGIES = {
    "baseline": None,
    "D1_cot": apply_cot,
    "D2_reflection": "reflection",  # special handling needed
    "D3_counter": "counter",        # needs bias_name
    "D4_role": apply_role_prompt,
    "D5_ensemble": "ensemble",      # special handling needed
}
