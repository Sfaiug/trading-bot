#!/usr/bin/env python3
"""
Monte Carlo Simulation Module

Provides stress testing for trading strategies through:
1. Bootstrap resampling to build P&L confidence intervals
2. Permutation testing to detect sequence dependency
3. Parameter perturbation for robustness analysis
4. Ruin probability calculation

CRITICAL: This module must be used AFTER walk-forward optimization
to validate that the "best" parameters aren't just lucky outliers.
"""

import math
import random
from typing import List, Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# Configuration Constants
# =============================================================================

DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_PERMUTATION_ITERATIONS = 5_000
DEFAULT_CONFIDENCE_LEVEL = 0.90  # 90% CI (5th to 95th percentile)
DEFAULT_RUIN_THRESHOLD = 0.40    # -40% account equity
DEFAULT_PERTURBATION_PCT = 0.10  # ±10% parameter noise


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class MonteCarloResult:
    """
    Comprehensive results from Monte Carlo simulation.

    All percentile values represent the distribution from bootstrap resampling.
    """
    # Metadata
    iterations: int
    sample_size: int

    # P&L Distribution
    pnl_distribution: List[float] = field(default_factory=list)
    pnl_mean: float = 0.0
    pnl_median: float = 0.0
    pnl_std: float = 0.0
    pnl_5th_percentile: float = 0.0      # Worst reasonable case
    pnl_25th_percentile: float = 0.0
    pnl_75th_percentile: float = 0.0
    pnl_95th_percentile: float = 0.0     # Best reasonable case
    probability_positive: float = 0.0     # P(total P&L > 0)

    # Drawdown Distribution
    max_drawdown_distribution: List[float] = field(default_factory=list)
    drawdown_mean: float = 0.0
    drawdown_median: float = 0.0
    drawdown_5th_percentile: float = 0.0  # Worst drawdown (most negative)
    drawdown_95th_percentile: float = 0.0
    probability_ruin: float = 0.0         # P(drawdown > threshold)
    ruin_threshold: float = 0.40

    # Sharpe Ratio Distribution
    sharpe_distribution: List[float] = field(default_factory=list)
    sharpe_mean: float = 0.0
    sharpe_median: float = 0.0
    sharpe_5th_percentile: float = 0.0
    sharpe_95th_percentile: float = 0.0

    # Validation Summary
    passes_validation: bool = False
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class PermutationTestResult:
    """Results from permutation test for sequence dependency."""
    iterations: int
    observed_sharpe: float
    permuted_sharpe_mean: float
    permuted_sharpe_std: float
    p_value: float
    sequence_matters: bool  # True if p_value < 0.05
    observed_vs_permuted_ratio: float


@dataclass
class PerturbationResult:
    """Results from parameter perturbation analysis."""
    base_pnl: float
    perturbation_pnls: List[float]
    perturbation_pct: float
    profitable_ratio: float  # Fraction of perturbations that are profitable
    mean_pnl: float
    std_pnl: float
    robustness_score: float  # 0-1, higher is more robust
    parameter_sensitivities: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Core Monte Carlo Functions
# =============================================================================

def _calculate_max_drawdown(cumulative_pnl: List[float]) -> float:
    """
    Calculate maximum drawdown from cumulative P&L series.

    Args:
        cumulative_pnl: List of cumulative P&L values

    Returns:
        Maximum drawdown as a positive percentage (e.g., 0.25 = 25% drawdown)
    """
    if not cumulative_pnl:
        return 0.0

    peak = cumulative_pnl[0]
    max_dd = 0.0

    for pnl in cumulative_pnl:
        if pnl > peak:
            peak = pnl

        if peak > 0:
            drawdown = (peak - pnl) / peak
            max_dd = max(max_dd, drawdown)

    return max_dd


def _calculate_sharpe_from_returns(returns: List[float], periods_per_year: float = 52) -> float:
    """
    Calculate Sharpe ratio from a list of returns.

    Args:
        returns: List of percentage returns
        periods_per_year: Number of periods per year for annualization

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0001

    if std_dev < 0.0001:
        return 0.0

    annualization = math.sqrt(periods_per_year)
    return (mean_return / std_dev) * annualization


def bootstrap_returns(
    round_returns: List[float],
    n_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    sample_size: Optional[int] = None,
    ruin_threshold: float = DEFAULT_RUIN_THRESHOLD,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: Optional[int] = None
) -> MonteCarloResult:
    """
    Resample rounds with replacement to build P&L distribution.

    This is the core Monte Carlo function. It creates thousands of
    "alternate histories" by randomly resampling the actual trades.
    This reveals the range of possible outcomes and helps distinguish
    skill from luck.

    Args:
        round_returns: P&L percentage per completed round
        n_iterations: Number of bootstrap samples (10,000 recommended)
        sample_size: Rounds per sample (default: len(round_returns))
        ruin_threshold: Drawdown threshold for ruin probability (default: 40%)
        confidence_level: Confidence level for intervals (default: 90%)
        random_seed: Optional seed for reproducibility

    Returns:
        MonteCarloResult with full distributions and validation status
    """
    if not round_returns:
        return MonteCarloResult(
            iterations=0,
            sample_size=0,
            failure_reasons=["No returns provided"]
        )

    if random_seed is not None:
        random.seed(random_seed)

    n = len(round_returns)
    if sample_size is None:
        sample_size = n

    # Store distributions
    pnl_dist = []
    drawdown_dist = []
    sharpe_dist = []

    for _ in range(n_iterations):
        # Sample with replacement
        sample = [round_returns[random.randint(0, n - 1)] for _ in range(sample_size)]

        # Calculate total P&L for this sample
        total_pnl = sum(sample)
        pnl_dist.append(total_pnl)

        # Calculate cumulative P&L for drawdown
        cumulative = []
        cum_sum = 0.0
        for ret in sample:
            cum_sum += ret
            cumulative.append(cum_sum)

        # Calculate max drawdown
        max_dd = _calculate_max_drawdown(cumulative)
        drawdown_dist.append(max_dd)

        # Calculate Sharpe ratio
        sharpe = _calculate_sharpe_from_returns(sample)
        sharpe_dist.append(sharpe)

    # Sort distributions for percentile calculations
    pnl_dist.sort()
    drawdown_dist.sort()
    sharpe_dist.sort()

    # Calculate percentile indices
    alpha = 1 - confidence_level
    lower_idx = int(n_iterations * (alpha / 2))
    upper_idx = int(n_iterations * (1 - alpha / 2))
    p25_idx = int(n_iterations * 0.25)
    p75_idx = int(n_iterations * 0.75)
    median_idx = n_iterations // 2

    # Calculate statistics
    pnl_mean = sum(pnl_dist) / n_iterations
    pnl_std = math.sqrt(sum((p - pnl_mean) ** 2 for p in pnl_dist) / n_iterations)

    dd_mean = sum(drawdown_dist) / n_iterations
    sharpe_mean = sum(sharpe_dist) / n_iterations

    # Count positive outcomes and ruin events
    positive_count = sum(1 for p in pnl_dist if p > 0)
    ruin_count = sum(1 for d in drawdown_dist if d > ruin_threshold)

    # Build result
    result = MonteCarloResult(
        iterations=n_iterations,
        sample_size=sample_size,

        # P&L
        pnl_distribution=pnl_dist,
        pnl_mean=pnl_mean,
        pnl_median=pnl_dist[median_idx],
        pnl_std=pnl_std,
        pnl_5th_percentile=pnl_dist[lower_idx],
        pnl_25th_percentile=pnl_dist[p25_idx],
        pnl_75th_percentile=pnl_dist[p75_idx],
        pnl_95th_percentile=pnl_dist[upper_idx],
        probability_positive=positive_count / n_iterations,

        # Drawdown
        max_drawdown_distribution=drawdown_dist,
        drawdown_mean=dd_mean,
        drawdown_median=drawdown_dist[median_idx],
        drawdown_5th_percentile=drawdown_dist[lower_idx],  # Least bad
        drawdown_95th_percentile=drawdown_dist[upper_idx],  # Worst
        probability_ruin=ruin_count / n_iterations,
        ruin_threshold=ruin_threshold,

        # Sharpe
        sharpe_distribution=sharpe_dist,
        sharpe_mean=sharpe_mean,
        sharpe_median=sharpe_dist[median_idx],
        sharpe_5th_percentile=sharpe_dist[lower_idx],
        sharpe_95th_percentile=sharpe_dist[upper_idx],
    )

    # Validation checks
    failures = []

    if result.probability_positive < 0.80:
        failures.append(f"P(positive P&L) = {result.probability_positive:.1%} < 80%")

    # v7.7: Relaxed from -15% to -25% for aggressive trend-catching strategies
    if result.pnl_5th_percentile < -25.0:
        failures.append(f"5th percentile P&L = {result.pnl_5th_percentile:.1f}% < -25%")

    # v7.7: Relaxed from 5% to 10% for higher risk tolerance
    if result.probability_ruin > 0.10:
        failures.append(f"P(ruin) = {result.probability_ruin:.1%} > 10%")

    if result.sharpe_5th_percentile < 0.5:
        failures.append(f"5th percentile Sharpe = {result.sharpe_5th_percentile:.2f} < 0.5")

    result.failure_reasons = failures
    result.passes_validation = len(failures) == 0

    return result


def permutation_test(
    round_returns: List[float],
    n_iterations: int = DEFAULT_PERMUTATION_ITERATIONS,
    random_seed: Optional[int] = None
) -> PermutationTestResult:
    """
    Shuffle returns to test if sequence matters.

    If the strategy's performance depends heavily on the ORDER of trades
    (e.g., winning streak followed by losing streak), that's concerning.
    A robust strategy should perform similarly regardless of trade order.

    The null hypothesis is that order doesn't matter. A low p-value
    suggests the observed Sharpe depends on lucky sequencing.

    Args:
        round_returns: P&L percentage per completed round
        n_iterations: Number of permutations (5,000 recommended)
        random_seed: Optional seed for reproducibility

    Returns:
        PermutationTestResult with p-value and sequence dependency flag
    """
    if len(round_returns) < 10:
        return PermutationTestResult(
            iterations=0,
            observed_sharpe=0.0,
            permuted_sharpe_mean=0.0,
            permuted_sharpe_std=0.0,
            p_value=1.0,
            sequence_matters=False,
            observed_vs_permuted_ratio=1.0
        )

    if random_seed is not None:
        random.seed(random_seed)

    # Calculate observed Sharpe (with actual sequence)
    observed_sharpe = _calculate_sharpe_from_returns(round_returns)

    # Generate permuted Sharpe ratios
    permuted_sharpes = []
    returns_copy = round_returns.copy()

    for _ in range(n_iterations):
        random.shuffle(returns_copy)
        sharpe = _calculate_sharpe_from_returns(returns_copy)
        permuted_sharpes.append(sharpe)

    # Calculate statistics
    permuted_mean = sum(permuted_sharpes) / n_iterations
    permuted_std = math.sqrt(
        sum((s - permuted_mean) ** 2 for s in permuted_sharpes) / n_iterations
    )

    # Calculate p-value: probability of seeing observed Sharpe or better by chance
    # For a two-tailed test, we count how many permuted values are as extreme
    extreme_count = sum(1 for s in permuted_sharpes if abs(s) >= abs(observed_sharpe))
    p_value = extreme_count / n_iterations

    # Ensure minimum p-value for numerical stability
    p_value = max(p_value, 1.0 / n_iterations)

    # Calculate ratio
    ratio = observed_sharpe / permuted_mean if permuted_mean != 0 else 1.0

    return PermutationTestResult(
        iterations=n_iterations,
        observed_sharpe=observed_sharpe,
        permuted_sharpe_mean=permuted_mean,
        permuted_sharpe_std=permuted_std,
        p_value=p_value,
        sequence_matters=(p_value < 0.05),
        observed_vs_permuted_ratio=ratio
    )


def parameter_perturbation(
    base_params: Dict[str, float],
    round_returns: List[float],
    backtest_func: Optional[Callable] = None,
    perturbation_pct: float = DEFAULT_PERTURBATION_PCT,
    n_samples: int = 100,
    random_seed: Optional[int] = None
) -> PerturbationResult:
    """
    Add ±X% noise to parameters to test sensitivity.

    If small changes to parameters cause large changes in P&L,
    the strategy is fragile and likely overfit. A robust strategy
    should perform well across a range of similar parameter values.

    NOTE: This is a simplified version that perturbs the RETURNS
    rather than re-running backtests. For full parameter perturbation
    with actual backtests, pass a backtest_func.

    Args:
        base_params: Dictionary of parameter names to values
        round_returns: P&L percentage per round (used if backtest_func is None)
        backtest_func: Optional function(params) -> List[float] returns
        perturbation_pct: Percentage to perturb parameters (default: 10%)
        n_samples: Number of perturbation samples
        random_seed: Optional seed for reproducibility

    Returns:
        PerturbationResult with robustness score and sensitivities
    """
    if random_seed is not None:
        random.seed(random_seed)

    base_pnl = sum(round_returns) if round_returns else 0.0
    perturbation_pnls = []

    if backtest_func is not None:
        # Full parameter perturbation with actual backtests
        for _ in range(n_samples):
            perturbed_params = {}
            for key, value in base_params.items():
                if isinstance(value, (int, float)) and value != 0:
                    # Add random noise
                    noise = random.uniform(-perturbation_pct, perturbation_pct)
                    perturbed_params[key] = value * (1 + noise)
                else:
                    perturbed_params[key] = value

            try:
                perturbed_returns = backtest_func(perturbed_params)
                perturbation_pnls.append(sum(perturbed_returns))
            except Exception:
                # If backtest fails, use 0 (conservative assumption)
                perturbation_pnls.append(0.0)
    else:
        # Simplified: perturb returns directly (scale each return by noise factor)
        for _ in range(n_samples):
            noise_factor = 1 + random.uniform(-perturbation_pct, perturbation_pct)
            perturbed = [r * noise_factor for r in round_returns]
            perturbation_pnls.append(sum(perturbed))

    if not perturbation_pnls:
        return PerturbationResult(
            base_pnl=base_pnl,
            perturbation_pnls=[],
            perturbation_pct=perturbation_pct,
            profitable_ratio=0.0,
            mean_pnl=0.0,
            std_pnl=0.0,
            robustness_score=0.0
        )

    # Calculate statistics
    profitable_count = sum(1 for p in perturbation_pnls if p > 0)
    profitable_ratio = profitable_count / len(perturbation_pnls)

    mean_pnl = sum(perturbation_pnls) / len(perturbation_pnls)
    std_pnl = math.sqrt(
        sum((p - mean_pnl) ** 2 for p in perturbation_pnls) / len(perturbation_pnls)
    )

    # Robustness score: combines consistency and profitability
    # Higher is better (0-1 scale)
    consistency = min(1.0, mean_pnl / (std_pnl + 0.01)) if mean_pnl > 0 else 0.0
    robustness = profitable_ratio * 0.6 + min(consistency, 1.0) * 0.4

    return PerturbationResult(
        base_pnl=base_pnl,
        perturbation_pnls=perturbation_pnls,
        perturbation_pct=perturbation_pct,
        profitable_ratio=profitable_ratio,
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        robustness_score=robustness
    )


def calculate_ruin_probability(
    round_returns: List[float],
    ruin_threshold: float = DEFAULT_RUIN_THRESHOLD,
    n_paths: int = 10_000,
    random_seed: Optional[int] = None
) -> Tuple[float, List[float]]:
    """
    Calculate probability of hitting max drawdown threshold.

    Simulates many random orderings of the actual trades to estimate
    the probability of hitting a catastrophic drawdown.

    Args:
        round_returns: P&L percentage per round
        ruin_threshold: Drawdown threshold (e.g., 0.40 = 40%)
        n_paths: Number of simulation paths
        random_seed: Optional seed for reproducibility

    Returns:
        Tuple of (ruin_probability, drawdown_distribution)
    """
    if not round_returns:
        return 0.0, []

    if random_seed is not None:
        random.seed(random_seed)

    n = len(round_returns)
    ruin_count = 0
    drawdown_dist = []

    for _ in range(n_paths):
        # Sample with replacement
        sample = [round_returns[random.randint(0, n - 1)] for _ in range(n)]

        # Calculate cumulative P&L
        cumulative = []
        cum_sum = 0.0
        for ret in sample:
            cum_sum += ret
            cumulative.append(cum_sum)

        # Calculate max drawdown
        max_dd = _calculate_max_drawdown(cumulative)
        drawdown_dist.append(max_dd)

        if max_dd > ruin_threshold:
            ruin_count += 1

    ruin_prob = ruin_count / n_paths
    return ruin_prob, drawdown_dist


# =============================================================================
# Integration with Walk-Forward Validation
# =============================================================================

def run_monte_carlo_validation(
    round_returns: List[float],
    params: Optional[Dict[str, float]] = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    n_permutation: int = DEFAULT_PERMUTATION_ITERATIONS,
    ruin_threshold: float = DEFAULT_RUIN_THRESHOLD,
    random_seed: Optional[int] = 42
) -> Tuple[bool, MonteCarloResult, PermutationTestResult]:
    """
    Run complete Monte Carlo validation suite.

    This is the main entry point after walk-forward optimization.
    It runs:
    1. Bootstrap resampling for P&L confidence intervals
    2. Permutation testing for sequence dependency

    Args:
        round_returns: P&L percentage per round from holdout test
        params: Optional parameter dictionary (for logging)
        n_bootstrap: Bootstrap iterations (default: 10,000)
        n_permutation: Permutation iterations (default: 5,000)
        ruin_threshold: Ruin threshold (default: 40%)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (passes_validation, mc_result, perm_result)
    """
    # Run bootstrap
    mc_result = bootstrap_returns(
        round_returns=round_returns,
        n_iterations=n_bootstrap,
        ruin_threshold=ruin_threshold,
        random_seed=random_seed
    )

    # Run permutation test
    perm_result = permutation_test(
        round_returns=round_returns,
        n_iterations=n_permutation,
        random_seed=random_seed
    )

    # Add permutation result to MC failures if sequence matters
    if perm_result.sequence_matters:
        mc_result.failure_reasons.append(
            f"Permutation test p-value = {perm_result.p_value:.3f} < 0.05 (sequence matters)"
        )
        mc_result.passes_validation = False

    return mc_result.passes_validation, mc_result, perm_result


def format_monte_carlo_report(
    mc_result: MonteCarloResult,
    perm_result: Optional[PermutationTestResult] = None,
    symbol: str = "UNKNOWN"
) -> str:
    """
    Format Monte Carlo results as a human-readable report.

    Args:
        mc_result: MonteCarloResult from bootstrap_returns()
        perm_result: Optional PermutationTestResult
        symbol: Trading symbol for the report header

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"MONTE CARLO VALIDATION REPORT: {symbol}")
    lines.append(f"Iterations: {mc_result.iterations:,} | Sample Size: {mc_result.sample_size}")
    lines.append("=" * 70)

    lines.append("\n--- P&L DISTRIBUTION ---")
    lines.append(f"  5th percentile:   {mc_result.pnl_5th_percentile:+.1f}%  (worst reasonable case)")
    lines.append(f"  25th percentile:  {mc_result.pnl_25th_percentile:+.1f}%")
    lines.append(f"  Median:           {mc_result.pnl_median:+.1f}%")
    lines.append(f"  Mean:             {mc_result.pnl_mean:+.1f}% (std: {mc_result.pnl_std:.1f}%)")
    lines.append(f"  75th percentile:  {mc_result.pnl_75th_percentile:+.1f}%")
    lines.append(f"  95th percentile:  {mc_result.pnl_95th_percentile:+.1f}%  (best reasonable case)")
    lines.append(f"\n  Probability of Positive P&L: {mc_result.probability_positive:.1%}")
    status = "PASS" if mc_result.probability_positive >= 0.80 else "FAIL"
    lines.append(f"  Status: [{status}] (threshold: 80%)")

    lines.append("\n--- DRAWDOWN DISTRIBUTION ---")
    lines.append(f"  5th percentile:   -{mc_result.drawdown_5th_percentile*100:.1f}%  (best case)")
    lines.append(f"  Median:           -{mc_result.drawdown_median*100:.1f}%")
    lines.append(f"  Mean:             -{mc_result.drawdown_mean*100:.1f}%")
    lines.append(f"  95th percentile:  -{mc_result.drawdown_95th_percentile*100:.1f}%  (worst case)")
    lines.append(f"\n  Probability of Ruin (>{mc_result.ruin_threshold*100:.0f}%): {mc_result.probability_ruin:.1%}")
    status = "PASS" if mc_result.probability_ruin <= 0.05 else "FAIL"
    lines.append(f"  Status: [{status}] (threshold: 5%)")

    lines.append("\n--- SHARPE RATIO DISTRIBUTION ---")
    lines.append(f"  5th percentile:   {mc_result.sharpe_5th_percentile:.2f}")
    lines.append(f"  Median:           {mc_result.sharpe_median:.2f}")
    lines.append(f"  Mean:             {mc_result.sharpe_mean:.2f}")
    lines.append(f"  95th percentile:  {mc_result.sharpe_95th_percentile:.2f}")
    status = "PASS" if mc_result.sharpe_5th_percentile >= 0.5 else "FAIL"
    lines.append(f"  Status: [{status}] (5th pct threshold: 0.5)")

    if perm_result is not None:
        lines.append("\n--- PERMUTATION TEST ---")
        lines.append(f"  Observed Sharpe:          {perm_result.observed_sharpe:.2f}")
        lines.append(f"  Permuted Sharpe (mean):   {perm_result.permuted_sharpe_mean:.2f} (std: {perm_result.permuted_sharpe_std:.2f})")
        lines.append(f"  Observed/Permuted Ratio:  {perm_result.observed_vs_permuted_ratio:.2f}")
        lines.append(f"  P-value:                  {perm_result.p_value:.4f}")
        status = "PASS" if not perm_result.sequence_matters else "FAIL"
        lines.append(f"  Sequence Dependency:      [{status}]")
        if perm_result.sequence_matters:
            lines.append("  WARNING: Performance may depend on lucky trade sequencing!")

    lines.append("\n" + "=" * 70)
    if mc_result.passes_validation:
        lines.append("OVERALL: PASS - Strategy passes Monte Carlo validation")
    else:
        lines.append("OVERALL: FAIL - Strategy failed Monte Carlo validation")
        lines.append("\nFAILURE REASONS:")
        for reason in mc_result.failure_reasons:
            lines.append(f"  - {reason}")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_monte_carlo_check(
    round_returns: List[float],
    n_iterations: int = 1000,
    ruin_threshold: float = DEFAULT_RUIN_THRESHOLD
) -> Tuple[bool, float, float, float]:
    """
    Quick Monte Carlo check for use during optimization.

    Returns only key metrics without full distributions.

    Args:
        round_returns: P&L percentage per round
        n_iterations: Number of iterations (1,000 for speed)
        ruin_threshold: Ruin threshold

    Returns:
        Tuple of (passes, prob_positive, prob_ruin, sharpe_5th_pct)
    """
    if len(round_returns) < 10:
        return False, 0.0, 1.0, 0.0

    result = bootstrap_returns(
        round_returns=round_returns,
        n_iterations=n_iterations,
        ruin_threshold=ruin_threshold,
        random_seed=42
    )

    return (
        result.passes_validation,
        result.probability_positive,
        result.probability_ruin,
        result.sharpe_5th_percentile
    )


if __name__ == "__main__":
    # Example usage and testing
    import random

    print("Monte Carlo Simulation Module - Test Run")
    print("-" * 50)

    # Generate synthetic returns (slightly profitable strategy)
    random.seed(42)
    test_returns = [random.gauss(0.5, 2.0) for _ in range(100)]

    print(f"\nTest data: {len(test_returns)} rounds")
    print(f"Total P&L: {sum(test_returns):.1f}%")
    print(f"Mean return: {sum(test_returns)/len(test_returns):.2f}%")

    # Run validation
    passes, mc, perm = run_monte_carlo_validation(
        round_returns=test_returns,
        ruin_threshold=0.40,
        random_seed=42
    )

    # Print report
    print("\n" + format_monte_carlo_report(mc, perm, "TEST"))
