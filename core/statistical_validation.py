#!/usr/bin/env python3
"""
Statistical Validation Module

PHASE 4 FIX: Addresses the multiple comparisons problem in hyperparameter optimization.

When testing 53 million parameter combinations, you WILL find profitable-looking
configurations by pure chance. This module provides statistical tools to separate
genuine alpha from noise.

Key concepts:
1. Bootstrap Sharpe Ratio with Confidence Intervals
2. Bonferroni Correction for multiple comparisons
3. Out-of-Sample validation
4. T-test for strategy significance
"""

import math
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# =============================================================================
# Configuration Constants
# =============================================================================

# Minimum thresholds for a strategy to be considered valid
MIN_SHARPE_RATIO = 1.5           # In-sample Sharpe must be >= 1.5
MIN_SHARPE_CI_LOWER = 1.0        # 95% CI lower bound must be >= 1.0
MAX_SHARPE_DEGRADATION_PCT = 30  # OOS Sharpe can drop at most 30% from in-sample
MIN_ROUNDS_FOR_SIGNIFICANCE = 30 # Need at least 30 rounds for statistical validity
MIN_WIN_RATE_PCT = 40            # Win rate must be at least 40%


# =============================================================================
# Core Statistical Functions
# =============================================================================

def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    avg_round_duration_days: Optional[float] = None,
    total_test_duration_days: Optional[float] = None
) -> float:
    """
    Calculate Sharpe Ratio from a list of returns.

    PHASE 2.3 FIX: Proper annualization based on actual trading frequency.

    Args:
        returns: List of percentage returns per trade/round
        risk_free_rate: Risk-free rate (default 0 for simplicity)
        avg_round_duration_days: Average duration of each round in days
                                 If None and total_test_duration_days provided,
                                 calculates from total duration / num returns
        total_test_duration_days: Total duration of the backtest in days
                                  Used to calculate rounds_per_year if available

    Returns:
        Sharpe ratio (properly annualized)
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    excess_return = mean_return - risk_free_rate

    # Calculate standard deviation
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0001

    if std_dev == 0:
        return 0.0

    # IMPROVED: Calculate rounds_per_year from actual data whenever possible
    # Priority:
    # 1. Use avg_round_duration_days if provided directly
    # 2. Calculate from total_test_duration_days and num_returns
    # 3. Conservative fallback (10 rounds/year - very conservative)

    if avg_round_duration_days is not None and avg_round_duration_days > 0:
        # Use actual round duration to calculate trades per year
        rounds_per_year = 365 / avg_round_duration_days
    elif total_test_duration_days is not None and total_test_duration_days > 0:
        # Calculate from total test duration and number of rounds
        avg_duration = total_test_duration_days / len(returns)
        rounds_per_year = 365 / avg_duration
    else:
        # CONSERVATIVE fallback: assume 10 rounds per year
        # This is lower than before (50) to avoid overstating Sharpe
        # Better to underestimate than overestimate significance
        rounds_per_year = 10

    annualization_factor = math.sqrt(rounds_per_year)

    return (excess_return / std_dev) * annualization_factor


def calculate_bootstrap_sharpe_ci(
    returns: List[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Calculate Sharpe Ratio with bootstrap confidence interval.

    This is crucial for understanding the uncertainty in our Sharpe estimate.
    A strategy with Sharpe 2.0 but CI [0.5, 3.5] is much riskier than
    a strategy with Sharpe 1.5 but CI [1.2, 1.8].

    Args:
        returns: List of percentage returns
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 95%)
        random_seed: Optional seed for reproducibility

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    if len(returns) < 10:
        sharpe = calculate_sharpe_ratio(returns)
        return (sharpe, sharpe * 0.5, sharpe * 1.5)

    if random_seed is not None:
        random.seed(random_seed)

    # Point estimate
    point_estimate = calculate_sharpe_ratio(returns)

    # Bootstrap resampling
    n = len(returns)
    bootstrap_sharpes = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = [returns[random.randint(0, n - 1)] for _ in range(n)]
        bootstrap_sharpes.append(calculate_sharpe_ratio(sample))

    # Sort for percentile calculation
    bootstrap_sharpes.sort()

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    ci_lower = bootstrap_sharpes[lower_idx]
    ci_upper = bootstrap_sharpes[upper_idx]

    return (point_estimate, ci_lower, ci_upper)


def calculate_t_statistic(returns: List[float], null_mean: float = 0.0) -> Tuple[float, float]:
    """
    Calculate t-statistic and p-value for returns being significantly different from null.

    Args:
        returns: List of returns
        null_mean: Hypothesized mean (default 0 = strategy has no edge)

    Returns:
        Tuple of (t_statistic, p_value)
    """
    if len(returns) < 3:
        return (0.0, 1.0)

    n = len(returns)
    sample_mean = sum(returns) / n
    variance = sum((r - sample_mean) ** 2 for r in returns) / (n - 1)
    std_error = math.sqrt(variance / n)

    if std_error == 0:
        return (0.0, 1.0)

    t_stat = (sample_mean - null_mean) / std_error

    # Approximate p-value using normal approximation (valid for n > 30)
    # For exact p-value, would need scipy.stats.t.sf
    # This is a conservative approximation
    z = abs(t_stat)
    if z > 3.5:
        p_value = 0.0005
    elif z > 3.0:
        p_value = 0.003
    elif z > 2.5:
        p_value = 0.012
    elif z > 2.0:
        p_value = 0.046
    elif z > 1.96:
        p_value = 0.05
    elif z > 1.5:
        p_value = 0.13
    elif z > 1.0:
        p_value = 0.32
    else:
        p_value = 1.0 - (z * 0.4)  # Rough approximation

    # Two-tailed p-value
    p_value = min(p_value * 2, 1.0)

    return (t_stat, p_value)


def apply_bonferroni_correction(
    p_value: float,
    num_tests: int
) -> Tuple[float, bool]:
    """
    Apply Bonferroni correction for multiple comparisons.

    When running 53 million backtests, the corrected significance threshold
    becomes extremely strict: 0.05 / 53,000,000 ≈ 9.4e-10

    NOTE: This is overly conservative for optimization. Use FDR instead.

    Args:
        p_value: Original p-value from t-test
        num_tests: Number of tests performed (e.g., number of parameter combinations)

    Returns:
        Tuple of (corrected_threshold, is_significant)
    """
    corrected_alpha = 0.05 / num_tests
    is_significant = p_value <= corrected_alpha

    return (corrected_alpha, is_significant)


def apply_benjamini_hochberg_fdr(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[float, bool, float]]:
    """
    Apply Benjamini-Hochberg procedure to control False Discovery Rate.

    PHASE 2.3 FIX: This is much less strict than Bonferroni and is
    appropriate for exploratory optimization where we expect some
    true positives among many tests.

    The procedure:
    1. Sort p-values in ascending order
    2. For each p-value at rank i, calculate threshold: (i/m) * alpha
    3. Find largest i where p_i <= threshold
    4. All p-values at rank <= i are significant

    Args:
        p_values: List of p-values from multiple tests
        alpha: Desired FDR level (default 0.05 = 5% false discoveries)

    Returns:
        List of tuples: (original_p_value, is_significant, adjusted_p_value)
    """
    if not p_values:
        return []

    m = len(p_values)

    # Create indexed list for tracking original positions
    indexed_pvals = [(p, i) for i, p in enumerate(p_values)]

    # Sort by p-value ascending
    indexed_pvals.sort(key=lambda x: x[0])

    # Calculate BH threshold for each rank
    # Threshold at rank i is: (i/m) * alpha
    results = [None] * m
    max_significant_rank = -1

    for rank, (p_val, orig_idx) in enumerate(indexed_pvals, start=1):
        bh_threshold = (rank / m) * alpha

        if p_val <= bh_threshold:
            max_significant_rank = rank

    # Mark all p-values at rank <= max_significant_rank as significant
    for rank, (p_val, orig_idx) in enumerate(indexed_pvals, start=1):
        is_significant = rank <= max_significant_rank

        # Calculate adjusted p-value (BH-adjusted)
        # Adjusted p-value = min(p_val * m / rank, 1.0)
        adjusted_p = min(p_val * m / rank, 1.0)

        results[orig_idx] = (p_val, is_significant, adjusted_p)

    return results


def apply_fdr_single_test(
    p_value: float,
    num_tests: int,
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Apply FDR correction for a single p-value (simplified version).

    When we only have one p-value but know we ran many tests, we can
    approximate the FDR threshold. This is less accurate than the full
    BH procedure but works when we don't have all p-values.

    The intuition: If we ran N tests and got one p-value p, the expected
    number of false positives at threshold p is N*p. For FDR control,
    we want this to be at most alpha * (number of discoveries).

    Approximation: significant if p <= alpha * rank / N
    For a single best result (rank 1): p <= alpha / N (same as Bonferroni!)

    However, if we assume the top result is truly the best among k
    genuine effects, the threshold relaxes significantly.

    Args:
        p_value: The p-value to test
        num_tests: Total number of tests performed
        alpha: Desired FDR level

    Returns:
        Tuple of (effective_threshold, is_significant)
    """
    # TIGHTENED: Use Bonferroni-equivalent strictness.
    # The previous heuristic (num_tests * 0.01) assumed 1% are real effects,
    # but with 70M+ tests, this is far too lenient (would allow 700,000 "effects").
    #
    # Bonferroni: threshold = alpha / num_tests
    # This is very strict but appropriate given extreme multiple testing.

    effective_tests = num_tests  # Bonferroni-equivalent: all tests count
    fdr_threshold = alpha / effective_tests

    # Cap at reasonable bounds (don't overflow on very large num_tests)
    fdr_threshold = max(fdr_threshold, 1e-12)  # Minimum threshold (numerical stability)

    is_significant = p_value <= fdr_threshold

    return (fdr_threshold, is_significant)


# =============================================================================
# Validation Data Class
# =============================================================================

@dataclass
class StrategyValidationResult:
    """Results from statistical validation of a strategy."""

    # Basic metrics
    total_rounds: int
    total_pnl: float
    win_rate: float

    # Sharpe analysis
    sharpe_ratio: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float

    # Statistical significance
    t_statistic: float
    p_value: float
    corrected_alpha: float
    is_statistically_significant: bool

    # FDR-based significance (PHASE 2.3 FIX)
    fdr_threshold: float = 0.05
    is_fdr_significant: bool = False

    # Out-of-sample performance
    oos_sharpe_ratio: float = 0.0
    sharpe_degradation_pct: float = 0.0

    # Final verdict
    passes_validation: bool = False
    rejection_reasons: List[str] = None

    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = []


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_strategy_statistically(
    in_sample_returns: List[float],
    out_of_sample_returns: Optional[List[float]] = None,
    num_tests_performed: int = 1,
    min_sharpe: float = MIN_SHARPE_RATIO,
    min_sharpe_ci_lower: float = MIN_SHARPE_CI_LOWER,
    max_degradation_pct: float = MAX_SHARPE_DEGRADATION_PCT,
    min_rounds: int = MIN_ROUNDS_FOR_SIGNIFICANCE,
    min_win_rate: float = MIN_WIN_RATE_PCT,
    verbose: bool = False
) -> StrategyValidationResult:
    """
    Comprehensive statistical validation of a trading strategy.

    This function applies multiple validation checks to determine if a
    strategy has genuine predictive power or is just a result of data mining.

    Args:
        in_sample_returns: List of returns from training/optimization period
        out_of_sample_returns: List of returns from validation period (optional)
        num_tests_performed: Number of parameter combinations tested
        min_sharpe: Minimum required Sharpe ratio
        min_sharpe_ci_lower: Minimum Sharpe CI lower bound
        max_degradation_pct: Maximum allowed OOS Sharpe degradation
        min_rounds: Minimum number of trades for validity
        min_win_rate: Minimum win rate percentage
        verbose: Print validation details

    Returns:
        StrategyValidationResult with all metrics and pass/fail status
    """
    rejection_reasons = []

    # Basic metrics
    total_rounds = len(in_sample_returns)
    total_pnl = sum(in_sample_returns)
    winning = sum(1 for r in in_sample_returns if r > 0)
    win_rate = (winning / total_rounds * 100) if total_rounds > 0 else 0

    # Sharpe ratio with bootstrap CI
    sharpe, ci_lower, ci_upper = calculate_bootstrap_sharpe_ci(
        in_sample_returns,
        n_bootstrap=10000,
        random_seed=42  # For reproducibility
    )

    # T-test for significance
    t_stat, p_value = calculate_t_statistic(in_sample_returns, null_mean=0.0)

    # Bonferroni correction (kept for reference, but too strict)
    corrected_alpha, is_bonferroni_significant = apply_bonferroni_correction(p_value, num_tests_performed)

    # PHASE 2.3 FIX: Use FDR instead of Bonferroni for significance testing
    # FDR is much more appropriate for exploratory optimization
    fdr_threshold, is_fdr_significant = apply_fdr_single_test(p_value, num_tests_performed)

    # Use FDR for the actual significance determination
    is_significant = is_fdr_significant

    # Out-of-sample validation
    oos_sharpe = 0.0
    degradation = 0.0
    if out_of_sample_returns and len(out_of_sample_returns) >= 10:
        oos_sharpe = calculate_sharpe_ratio(out_of_sample_returns)
        if sharpe > 0:
            degradation = ((sharpe - oos_sharpe) / sharpe) * 100

    # Validation checks
    passes = True

    # Check 1: Minimum rounds
    if total_rounds < min_rounds:
        passes = False
        rejection_reasons.append(f"Insufficient rounds: {total_rounds} < {min_rounds}")

    # Check 2: Minimum win rate
    if win_rate < min_win_rate:
        passes = False
        rejection_reasons.append(f"Low win rate: {win_rate:.1f}% < {min_win_rate}%")

    # Check 3: Minimum Sharpe ratio
    if sharpe < min_sharpe:
        passes = False
        rejection_reasons.append(f"Low Sharpe ratio: {sharpe:.2f} < {min_sharpe}")

    # Check 4: Sharpe CI lower bound
    if ci_lower < min_sharpe_ci_lower:
        passes = False
        rejection_reasons.append(f"Low Sharpe CI: [{ci_lower:.2f}, {ci_upper:.2f}], lower < {min_sharpe_ci_lower}")

    # Check 5: Statistical significance (with FDR correction)
    # PHASE 2.3 FIX: Use FDR instead of Bonferroni
    if not is_significant:
        passes = False
        rejection_reasons.append(f"Not significant after FDR correction: p={p_value:.2e} > {fdr_threshold:.2e}")

    # Check 6: OOS degradation
    if out_of_sample_returns and degradation > max_degradation_pct:
        passes = False
        rejection_reasons.append(f"High OOS degradation: {degradation:.1f}% > {max_degradation_pct}%")

    # Check 7: OOS profitability
    if out_of_sample_returns and oos_sharpe < 0:
        passes = False
        rejection_reasons.append(f"Negative OOS Sharpe: {oos_sharpe:.2f}")

    result = StrategyValidationResult(
        total_rounds=total_rounds,
        total_pnl=total_pnl,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        sharpe_ci_lower=ci_lower,
        sharpe_ci_upper=ci_upper,
        t_statistic=t_stat,
        p_value=p_value,
        corrected_alpha=corrected_alpha,
        is_statistically_significant=is_significant,
        fdr_threshold=fdr_threshold,
        is_fdr_significant=is_fdr_significant,
        oos_sharpe_ratio=oos_sharpe,
        sharpe_degradation_pct=degradation,
        passes_validation=passes,
        rejection_reasons=rejection_reasons
    )

    if verbose:
        print("\n" + "=" * 60)
        print("STATISTICAL VALIDATION RESULTS")
        print("=" * 60)
        print(f"Total Rounds: {total_rounds}")
        print(f"Total P&L: {total_pnl:+.2f}%")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"\nSharpe Ratio: {sharpe:.3f}")
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"\nT-Statistic: {t_stat:.3f}")
        print(f"P-Value: {p_value:.2e}")
        print(f"Bonferroni Alpha: {corrected_alpha:.2e} (n={num_tests_performed:,}) - TOO STRICT")
        print(f"FDR Threshold: {fdr_threshold:.2e} (PHASE 2.3 FIX)")
        print(f"Statistically Significant (FDR): {is_significant}")

        if out_of_sample_returns:
            print(f"\nOOS Sharpe: {oos_sharpe:.3f}")
            print(f"Degradation: {degradation:.1f}%")

        print(f"\n{'PASSED' if passes else 'FAILED'} Validation")
        if rejection_reasons:
            print("Rejection reasons:")
            for reason in rejection_reasons:
                print(f"  - {reason}")
        print("=" * 60)

    return result


def quick_validation_check(
    returns: List[float],
    num_tests: int = 1000000
) -> Tuple[bool, str]:
    """
    Quick validation check for filtering during optimization.

    Returns a simple pass/fail with reason, without full bootstrap.
    Use this during grid search to quickly filter out bad candidates.

    Args:
        returns: List of returns
        num_tests: Number of tests (for Bonferroni)

    Returns:
        Tuple of (passes, reason)
    """
    if len(returns) < 20:
        return (False, f"Too few rounds: {len(returns)}")

    mean_ret = sum(returns) / len(returns)
    if mean_ret <= 0:
        return (False, f"Negative mean return: {mean_ret:.2f}%")

    # Quick Sharpe check (not annualized, just ratio)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0001
    quick_sharpe = mean_ret / std

    if quick_sharpe < 0.3:  # Very loose threshold for quick filter
        return (False, f"Low return/risk ratio: {quick_sharpe:.2f}")

    winning = sum(1 for r in returns if r > 0)
    win_rate = winning / len(returns) * 100
    if win_rate < 35:
        return (False, f"Low win rate: {win_rate:.1f}%")

    return (True, "Passed quick checks")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Statistical Validation Module...")

    # Create synthetic returns (slightly profitable strategy)
    random.seed(42)
    in_sample = [random.gauss(0.5, 2.0) for _ in range(100)]
    out_of_sample = [random.gauss(0.3, 2.0) for _ in range(50)]

    # Test with 1 test (no multiple comparison problem)
    print("\n--- Single Test (no correction) ---")
    result1 = validate_strategy_statistically(
        in_sample,
        out_of_sample,
        num_tests_performed=1,
        verbose=True
    )

    # Test with 53M tests (realistic optimization scenario)
    print("\n--- 53 Million Tests (Bonferroni correction) ---")
    result2 = validate_strategy_statistically(
        in_sample,
        out_of_sample,
        num_tests_performed=53_000_000,
        verbose=True
    )

    # Test quick validation
    print("\n--- Quick Validation Check ---")
    passes, reason = quick_validation_check(in_sample)
    print(f"Quick check: {'PASS' if passes else 'FAIL'} - {reason}")
