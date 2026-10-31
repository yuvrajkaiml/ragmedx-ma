"""
Statistical utilities for RAGMedX-MA evaluation.

Functions
---------
wilson_ci         -- Wilson score confidence interval for a proportion
mcnemar_test      -- McNemar test (exact when discordant pairs are few)
paired_bootstrap  -- Paired bootstrap CI for difference in a continuous metric
holm_correction   -- Holm-Bonferroni step-down multiple-comparison correction
format_result     -- Formatted result string with n, numerator/denominator, diff, CI, p

All p-values are two-sided unless otherwise noted.
"""

import math
import random
from typing import Any, Callable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Wilson Score Confidence Interval
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Definition:
        centre = (k + z^2/2) / (n + z^2)
        half   = (z / (n + z^2)) * sqrt(k*(n-k)/n + z^2/4)
        lower  = centre - half
        upper  = centre + half

    where z = scipy.stats.norm.ppf(1 - alpha/2) is the critical value for
    a two-sided interval at level (1 - alpha).

    Unlike the normal approximation (Wald) interval, the Wilson CI maintains
    coverage near the boundary (k=0 or k=n) and for small n.

    Parameters
    ----------
    k     : int — number of successes (0 <= k <= n)
    n     : int — number of trials (n >= 1)
    alpha : float — significance level; default 0.05 → 95% CI

    Returns
    -------
    (lower, upper) : tuple of float, each in [0, 1]

    Raises
    ------
    ValueError if n < 1 or k < 0 or k > n.
    """
    if n < 1:
        raise ValueError(f"wilson_ci: n must be >= 1, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"wilson_ci: k must be in [0, n], got k={k}, n={n}")

    # Use math-only implementation to avoid scipy import at module level.
    # z for alpha=0.05 → 1.959964; use erfinv-based formula.
    # scipy is available in both venvs; use it for correctness.
    try:
        from scipy.stats import norm as _norm
        z = _norm.ppf(1.0 - alpha / 2.0)
    except ImportError:
        # Fallback: hardcode z for common alpha values
        _Z_TABLE = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758, 0.001: 3.2905}
        if alpha in _Z_TABLE:
            z = _Z_TABLE[alpha]
        else:
            raise ImportError("scipy not available; only alpha in {0.10, 0.05, 0.01, 0.001} supported without scipy")

    z2 = z * z
    denom = n + z2
    centre = (k + z2 / 2.0) / denom
    half = (z / denom) * math.sqrt(k * (n - k) / n + z2 / 4.0)

    lower = max(0.0, centre - half)
    upper = min(1.0, centre + half)
    return (round(lower, 6), round(upper, 6))


# ---------------------------------------------------------------------------
# McNemar Test
# ---------------------------------------------------------------------------

def mcnemar_test(
    table: Tuple[Tuple[int, int], Tuple[int, int]],
    exact_threshold: int = 25,
) -> Tuple[float, float, str]:
    """McNemar's test for paired binary outcomes.

    The 2x2 contingency table is structured as:
        table[0][0] = n00  (both wrong / both abstain)
        table[0][1] = n01  (A wrong, B correct)  ← discordant
        table[1][0] = n10  (A correct, B wrong)  ← discordant
        table[1][1] = n11  (both correct)

    Discordant pairs: b = n01, c = n10.

    When b + c < exact_threshold: use the exact binomial test
        H0: P(discordant = n01 out of b+c) = 0.5
        Two-sided p-value computed from Binomial(n=b+c, p=0.5).

    When b + c >= exact_threshold: use the chi-squared approximation
        chi2 = (|b - c| - 1)^2 / (b + c)   (with continuity correction)
        p-value from chi2 distribution with df=1.

    Parameters
    ----------
    table           : 2x2 nested tuple of ints ((n00, n01), (n10, n11))
    exact_threshold : int — use exact test when b+c < this value

    Returns
    -------
    (statistic, p_value, method_used) where method_used is
    'exact_binomial' or 'chi_squared_continuity_corrected'.

    Raises
    ------
    ValueError for malformed table.
    """
    if len(table) != 2 or len(table[0]) != 2 or len(table[1]) != 2:
        raise ValueError("mcnemar_test: table must be a 2x2 structure")

    b = table[0][1]  # n01
    c = table[1][0]  # n10
    discordant = b + c

    if discordant < 0:
        raise ValueError("mcnemar_test: cell counts must be non-negative")

    if discordant < exact_threshold:
        # Exact two-sided binomial test: P(X <= min(b,c)) * 2
        from scipy.stats import binom as _binom
        # Two-sided: sum both tails
        smaller = min(b, c)
        p_value = float(2.0 * _binom.cdf(smaller, discordant, 0.5))
        p_value = min(p_value, 1.0)
        statistic = float(smaller)
        return (round(statistic, 4), round(p_value, 6), "exact_binomial")
    else:
        # Chi-squared with continuity correction (Edwards)
        chi2 = (abs(b - c) - 1.0) ** 2 / discordant if discordant > 0 else 0.0
        from scipy.stats import chi2 as _chi2
        p_value = float(1.0 - _chi2.cdf(chi2, df=1))
        return (round(chi2, 4), round(p_value, 6), "chi_squared_continuity_corrected")


# ---------------------------------------------------------------------------
# Paired Bootstrap CI
# ---------------------------------------------------------------------------

def paired_bootstrap(
    metric_fn: Callable[[List[Any], List[Any]], float],
    y_a: List[Any],
    y_b: List[Any],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired bootstrap confidence interval for the difference of a metric.

    Computes (metric(y_a) - metric(y_b)) and a 95% CI via the percentile
    bootstrap method.

    Procedure:
      1. Compute observed_diff = metric_fn(y_a, y_b_baseline) is NOT what we
         want; instead, observed_diff = metric_fn(y_a, labels) -
         metric_fn(y_b, labels).
         We assume metric_fn(predictions, labels) → scalar.

      For the paired bootstrap on two prediction sets y_a and y_b over the
      same set of labels:
        observed_diff = metric_fn(y_a) - metric_fn(y_b)
      where metric_fn takes a single sequence and returns a scalar.

      In practice this is used for continuous metrics where metric_fn maps a
      list of values to a single float (e.g. mean ROUGE-L).

      2. For each bootstrap iteration b:
           - Sample n indices with replacement from range(n).
           - Compute diff_b = metric_fn(y_a[indices]) - metric_fn(y_b[indices]).
      3. 95% CI from the 2.5th and 97.5th percentiles of the bootstrap
         distribution.

    Parameters
    ----------
    metric_fn   : Callable[[List], float]
                  Takes a single list and returns a scalar metric value.
                  Called as metric_fn(sample) where sample is a subsample of
                  either y_a or y_b.
    y_a         : list — predictions/values for system A
    y_b         : list — predictions/values for system B (same length as y_a)
    n_bootstrap : int — number of bootstrap replicates; default 10000
    seed        : int — random seed for reproducibility; default 42

    Returns
    -------
    (observed_diff, ci_lower, ci_upper) : tuple of float
        observed_diff = metric_fn(y_a) - metric_fn(y_b)
        ci_lower, ci_upper = 2.5th and 97.5th percentiles of the bootstrap
        distribution of differences.

    Raises
    ------
    ValueError if y_a and y_b have different lengths or are empty.
    """
    if not y_a or not y_b or len(y_a) != len(y_b):
        raise ValueError("paired_bootstrap: y_a and y_b must be non-empty and same length")

    n = len(y_a)
    observed_diff = metric_fn(y_a) - metric_fn(y_b)

    rng = random.Random(seed)
    boot_diffs: List[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample_a = [y_a[i] for i in indices]
        sample_b = [y_b[i] for i in indices]
        diff_b = metric_fn(sample_a) - metric_fn(sample_b)
        boot_diffs.append(diff_b)

    boot_diffs.sort()
    lo_idx = int(math.floor(0.025 * n_bootstrap))
    hi_idx = int(math.ceil(0.975 * n_bootstrap)) - 1
    hi_idx = min(hi_idx, n_bootstrap - 1)

    ci_lower = boot_diffs[lo_idx]
    ci_upper = boot_diffs[hi_idx]

    return (round(observed_diff, 6), round(ci_lower, 6), round(ci_upper, 6))


# ---------------------------------------------------------------------------
# Holm-Bonferroni Correction
# ---------------------------------------------------------------------------

def holm_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """Holm-Bonferroni step-down multiple comparison correction.

    Procedure:
      1. Sort p-values in ascending order.
      2. For the i-th smallest p-value (1-indexed), compute the adjusted
         threshold: alpha / (m - i + 1) where m = total number of tests.
      3. Reject H_i if p_i <= alpha / (m - i + 1) AND all H_j for j < i
         were rejected (step-down).
      4. Adjusted p-value for hypothesis i: p_adj_i = max(p_j * (m - j + 1))
         for all j <= i (in sorted order), capped at 1.0.

    Returns the rejected flags and adjusted p-values in the ORIGINAL order
    of the input p_values list.

    Parameters
    ----------
    p_values : list of float — raw p-values from m hypothesis tests
    alpha    : float — family-wise error rate; default 0.05

    Returns
    -------
    (rejected_flags, adjusted_p_values) where both are lists of length m,
    in the same order as the input p_values.

    Raises
    ------
    ValueError if p_values is empty.
    """
    if not p_values:
        raise ValueError("holm_correction: p_values must be non-empty")

    m = len(p_values)
    # Create sorted index
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    rejected = [False] * m

    running_max_adj = 0.0
    stop_rejecting = False

    for rank_0, (orig_idx, p) in enumerate(indexed):
        # rank is 1-based
        rank = rank_0 + 1
        adj = p * (m - rank + 1)
        adj = max(adj, running_max_adj)
        adj = min(adj, 1.0)
        running_max_adj = adj
        adjusted[orig_idx] = round(adj, 8)

        if not stop_rejecting and p <= alpha / (m - rank + 1):
            rejected[orig_idx] = True
        else:
            stop_rejecting = True
            rejected[orig_idx] = False

    return (rejected, adjusted)


# ---------------------------------------------------------------------------
# Result Formatter
# ---------------------------------------------------------------------------

def format_result(
    n: int,
    numerator: int,
    denominator: int,
    diff: float,
    ci: Tuple[float, float],
    p_value: float,
) -> str:
    """Format a statistical result as a human-readable string.

    Template:
        n={n}, {numerator}/{denominator} ({pct:.1f}%),
        diff={diff:+.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], p={p_value:.4f}

    Parameters
    ----------
    n           : int   — sample size
    numerator   : int   — count of successes
    denominator : int   — denominator for the proportion
    diff        : float — absolute difference (use +/- format)
    ci          : tuple (lower, upper) — 95% confidence interval
    p_value     : float — p-value

    Returns
    -------
    Formatted string.

    Raises
    ------
    ValueError if denominator is 0.
    """
    if denominator == 0:
        raise ValueError("format_result: denominator must be > 0")
    pct = 100.0 * numerator / denominator
    return (
        f"n={n}, {numerator}/{denominator} ({pct:.1f}%), "
        f"diff={diff:+.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], p={p_value:.4f}"
    )
