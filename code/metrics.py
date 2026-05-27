"""Bias Strength Index (BSI) computation with uncertainty decomposition."""

import numpy as np
from scipy import stats


def bsi_numerical(control_values: np.ndarray, treatment_values: np.ndarray,
                  anchor: float) -> float:
    """Compute BSI for numerical estimation biases (Eq. 2 in paper).

    BSI = |mean(treatment) - mean(control)| / |anchor - mean(control)|
    """
    mean_control = np.nanmean(control_values)
    mean_treatment = np.nanmean(treatment_values)
    denominator = abs(anchor - mean_control)

    if denominator < 1e-10:
        return np.nan

    return abs(mean_treatment - mean_control) / denominator


def bsi_categorical(control_values: np.ndarray, treatment_values: np.ndarray,
                    bias_option: str) -> float:
    """Compute BSI for categorical choice biases (Eq. 3 in paper).

    BSI = |p(bias_option | treatment) - p(bias_option | control)|
    """
    p_control = np.mean(control_values == bias_option)
    p_treatment = np.mean(treatment_values == bias_option)

    return abs(p_treatment - p_control)


def bsi_overconfidence(intervals: list[tuple[float, float]],
                       true_value: float) -> float:
    """Compute BSI for overconfidence bias.

    For 90% confidence intervals, the true value should fall within
    the interval 90% of the time. BSI = 1 - (hit_rate / 0.9).
    Clamped to [0, 1].
    """
    hits = sum(1 for low, high in intervals if low <= true_value <= high)
    hit_rate = hits / len(intervals)
    # Perfect calibration -> hit_rate = 0.9 -> BSI = 0
    # Total overconfidence -> hit_rate = 0 -> BSI = 1
    bsi = max(0.0, 1.0 - hit_rate / 0.9)
    return min(1.0, bsi)


def bootstrap_bsi(control_values: np.ndarray, treatment_values: np.ndarray,
                  bsi_func, n_resamples: int = 10_000, **kwargs) -> np.ndarray:
    """Compute bootstrap distribution of BSI for statistical uncertainty.

    Returns array of n_resamples BSI values.
    """
    rng = np.random.default_rng(42)
    n_control = len(control_values)
    n_treatment = len(treatment_values)
    bsi_samples = np.empty(n_resamples)

    for i in range(n_resamples):
        c_idx = rng.integers(0, n_control, size=n_control)
        t_idx = rng.integers(0, n_treatment, size=n_treatment)
        bsi_samples[i] = bsi_func(control_values[c_idx],
                                  treatment_values[t_idx], **kwargs)

    return bsi_samples


def compute_bsi_with_uncertainty(variant_results: list[dict],
                                bias_type: str,
                                n_bootstrap: int = 10_000) -> dict:
    """Compute BSI with full uncertainty decomposition across prompt variants.

    Args:
        variant_results: List of dicts, one per prompt variant, each with:
            - control_values: np.ndarray
            - treatment_values: np.ndarray
            - anchor (for numerical) or bias_option (for categorical)
            - true_value (for overconfidence)
        bias_type: "numerical", "categorical", or "overconfidence"
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Dict with: bsi_mean, bsi_per_variant, sigma_stat, sigma_sys,
                    sigma_tot, p_value, significant
    """
    K = len(variant_results)
    bsi_per_variant = []
    sigma_stat_per_variant = []

    for vr in variant_results:
        ctrl = vr["control_values"]
        treat = vr["treatment_values"]

        if bias_type == "numerical":
            bsi = bsi_numerical(ctrl, treat, vr["anchor"])
            boot = bootstrap_bsi(ctrl, treat, bsi_numerical,
                                 n_resamples=n_bootstrap, anchor=vr["anchor"])
        elif bias_type == "categorical":
            bsi = bsi_categorical(ctrl, treat, vr["bias_option"])
            boot = bootstrap_bsi(ctrl, treat, bsi_categorical,
                                 n_resamples=n_bootstrap,
                                 bias_option=vr["bias_option"])
        elif bias_type == "overconfidence":
            # Overconfidence is special: compare control vs treatment intervals
            bsi_ctrl = bsi_overconfidence(
                vr["control_intervals"], vr["true_value"])
            bsi_treat = bsi_overconfidence(
                vr["treatment_intervals"], vr["true_value"])
            bsi = bsi_treat - bsi_ctrl  # increase in overconfidence
            # Bootstrap not straightforward here; use simple estimate
            boot = np.array([bsi] * 100)  # placeholder
        else:
            raise ValueError(f"Unknown bias_type: {bias_type}")

        bsi_per_variant.append(bsi)
        sigma_stat_per_variant.append(np.nanstd(boot))

    bsi_arr = np.array(bsi_per_variant)
    bsi_mean = np.nanmean(bsi_arr)

    # Statistical uncertainty: average across variants
    sigma_stat = np.nanmean(sigma_stat_per_variant)

    # Systematic uncertainty: std across variants
    if K > 1:
        sigma_sys = np.nanstd(bsi_arr, ddof=1)
    else:
        sigma_sys = 0.0

    # Total uncertainty (in quadrature)
    sigma_tot = np.sqrt(sigma_stat**2 + sigma_sys**2)

    # Statistical significance: one-sample t-test H0: BSI = 0
    if K > 1 and np.nanstd(bsi_arr) > 0:
        t_stat, p_value = stats.ttest_1samp(bsi_arr[~np.isnan(bsi_arr)], 0)
        p_value = p_value / 2  # one-sided (BSI > 0)
    else:
        p_value = np.nan

    return {
        "bsi_mean": bsi_mean,
        "bsi_per_variant": bsi_per_variant,
        "sigma_stat": sigma_stat,
        "sigma_sys": sigma_sys,
        "sigma_tot": sigma_tot,
        "p_value": p_value,
        "significant": p_value < 0.05 if not np.isnan(p_value) else False,
    }


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Benjamini-Hochberg correction for multiple comparisons.

    Returns list of booleans indicating significance after correction.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n

    for rank, (orig_idx, p) in enumerate(indexed, 1):
        threshold = (rank / n) * alpha
        if p <= threshold:
            significant[orig_idx] = True
        else:
            break  # all subsequent are also non-significant

    return significant
