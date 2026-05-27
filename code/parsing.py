"""Response parsing utilities for extracting structured answers from LLM outputs."""

import re


def parse_number(response: str) -> float | None:
    """Extract a numerical value from a free-text response."""
    text = response.strip().replace(",", "").replace(" ", "").replace("%", "")

    # Try to find patterns like "15000000", "15 million", "2.7 million", etc.
    million_match = re.search(r"([\d.]+)\s*million", response, re.IGNORECASE)
    if million_match:
        return float(million_match.group(1)) * 1_000_000

    billion_match = re.search(r"([\d.]+)\s*billion", response, re.IGNORECASE)
    if billion_match:
        return float(billion_match.group(1)) * 1_000_000_000

    # Try to find a plain number (possibly with decimals)
    numbers = re.findall(r"-?[\d]+\.?[\d]*", text)
    if numbers:
        # Return the first number found
        return float(numbers[0])

    return None


def parse_choice(response: str, options: list[str]) -> str | None:
    """Extract a choice (A, B, C, ...) from a free-text response."""
    text = response.strip().upper()

    # Direct single-letter answer
    if text in options:
        return text

    # Look for patterns like "(A)", "A.", "A)", "Answer: A", "I choose A"
    for opt in options:
        patterns = [
            rf"\b{opt}\b",       # standalone letter
            rf"\({opt}\)",       # (A)
            rf"{opt}\)",         # A)
            rf"{opt}\.",         # A.
        ]
        for pat in patterns:
            if re.search(pat, text):
                # Make sure it's the primary answer, not just mentioned
                # Check if it appears early in the response
                match = re.search(pat, text)
                if match and match.start() < 50:
                    return opt

    # Fallback: find the last mentioned option
    last_pos = -1
    last_opt = None
    for opt in options:
        positions = [m.start() for m in re.finditer(rf"\b{opt}\b", text)]
        if positions:
            # Use first occurrence
            if last_pos == -1 or positions[0] < last_pos:
                last_pos = positions[0]
                last_opt = opt

    return last_opt


def _clean_number(s: str) -> float | None:
    """Convert a string to float, stripping trailing dots/extra dots."""
    s = s.strip().rstrip(".")
    # Handle multiple decimal points (keep only first)
    parts = s.split(".")
    if len(parts) > 2:
        s = parts[0] + "." + "".join(parts[1:])
    try:
        return float(s)
    except ValueError:
        return None


def parse_interval(response: str) -> tuple[float, float] | None:
    """Extract a LOW-HIGH interval from a response."""
    text = response.strip().replace(",", "")

    # Pattern: LOW-HIGH, LOW to HIGH, LOW – HIGH
    match = re.search(r"([\d.]+)\s*[-–—to]+\s*([\d.]+)", text)
    if match:
        low = _clean_number(match.group(1))
        high = _clean_number(match.group(2))
        if low is None or high is None:
            pass  # fall through to fallback
        elif low <= high:
            return (low, high)
        else:
            return (high, low)

    # Fallback: find two numbers
    numbers = re.findall(r"[\d]+\.?[\d]*", text)
    if len(numbers) >= 2:
        v1 = _clean_number(numbers[0])
        v2 = _clean_number(numbers[1])
        if v1 is not None and v2 is not None:
            vals = sorted([v1, v2])
            return (vals[0], vals[1])

    return None


def parse_response(response: str, parse_type: str, options: list[str] = None):
    """Dispatch to the appropriate parser."""
    # Skip error responses
    if response.startswith("ERROR:"):
        return None
    if parse_type == "number":
        return parse_number(response)
    elif parse_type == "choice":
        return parse_choice(response, options or ["A", "B"])
    elif parse_type == "interval":
        return parse_interval(response)
    else:
        raise ValueError(f"Unknown parse_type: {parse_type}")
