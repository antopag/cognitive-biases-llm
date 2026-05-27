"""Unified interface for querying multiple LLM providers."""

import time
from tenacity import retry, stop_after_attempt, wait_exponential
import config


def _get_openai_client(provider="openai"):
    from openai import OpenAI
    if provider == "groq":
        return OpenAI(api_key=config.GROQ_API_KEY,
                      base_url="https://api.groq.com/openai/v1")
    if provider == "deepseek":
        return OpenAI(api_key=config.DEEPSEEK_API_KEY,
                      base_url="https://api.deepseek.com")
    if provider == "together":
        return OpenAI(api_key=config.TOGETHER_API_KEY,
                      base_url="https://api.together.xyz/v1")
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _get_anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _get_google_client():
    from google import genai
    return genai.Client(api_key=config.GOOGLE_API_KEY)


def _get_mistral_client():
    from mistralai import Mistral
    return Mistral(api_key=config.MISTRAL_API_KEY)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=5, max=30))
def query_model(model_name: str, prompt: str, temperature: float = None,
                max_tokens: int = None) -> str:
    """Query a model and return the text response.

    Args:
        model_name: Key from config.MODELS.
        prompt: The user prompt to send.
        temperature: Sampling temperature (default from config).
        max_tokens: Max output tokens (default from config).

    Returns:
        The model's text response.
    """
    if temperature is None:
        temperature = config.TEMPERATURE
    if max_tokens is None:
        max_tokens = config.MAX_TOKENS

    model_cfg = config.MODELS[model_name]
    provider = model_cfg["provider"]
    model_id = model_cfg["model_id"]

    if provider in ("openai", "groq", "deepseek", "together"):
        client = _get_openai_client(provider)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    elif provider == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    elif provider == "google":
        from google.genai import types
        client = _get_google_client()
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=gen_config,
        )
        # Handle both response.text and nested structure
        if response.text:
            return response.text.strip()
        elif response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            raise ValueError("Empty response from Gemini")

    elif provider == "mistral":
        client = _get_mistral_client()
        response = client.chat.complete(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")
