#!/usr/bin/env python3
"""
Market Regime Detection Module

PHASE 5 FIX: Addresses market regime blindness in optimization.

Parameters optimized on mixed 2020-2024 data may work well on average
but fail in specific regimes. This module:
1. Detects market regimes (volatility, trend)
2. Labels historical data by regime
3. Validates strategies across ALL regimes

A robust strategy should be profitable across most regime types,
not just on average.
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Generator
from dataclasses import dataclass
import math


# =============================================================================
# Regime Types
# =============================================================================

class VolatilityRegime(Enum):
    """Volatility classification."""
    LOW = "low_vol"       # Below 25th percentile
    NORMAL = "normal_vol" # 25th-75th percentile
    HIGH = "high_vol"     # Above 75th percentile


class TrendRegime(Enum):
    """Trend direction classification."""
    BULL = "bull"         # Strong uptrend (>20% in 30 days)
    BEAR = "bear"         # Strong downtrend (<-20% in 30 days)
    SIDEWAYS = "sideways" # Neither strong up nor down


class MarketRegime(Enum):
    """Combined market regime classification."""
    HIGH_VOL_BULL = "high_vol_bull"
    HIGH_VOL_BEAR = "high_vol_bear"
    HIGH_VOL_SIDEWAYS = "high_vol_sideways"
    LOW_VOL_BULL = "low_vol_bull"
    LOW_VOL_BEAR = "low_vol_bear"
    LOW_VOL_SIDEWAYS = "low_vol_sideways"
    NORMAL_VOL_BULL = "normal_vol_bull"
    NORMAL_VOL_BEAR = "normal_vol_bear"
    NORMAL_VOL_SIDEWAYS = "normal_vol_sideways"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RegimeLabel:
    """Labels for a time period."""
    start_time: datetime
    end_time: datetime
    volatility: VolatilityRegime
    trend: TrendRegime
    combined: MarketRegime
    volatility_pct: float  # Actual volatility value
    trend_pct: float       # Actual trend value (30-day return)


@dataclass
class RegimeValidationResult:
    """Results of validating a strategy across regimes."""
    regime: MarketRegime
    num_rounds: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    is_profitable: bool


@dataclass
class CrossRegimeValidation:
    """Complete cross-regime validation results."""
    regime_results: Dict[MarketRegime, RegimeValidationResult]
    regimes_tested: int
    regimes_profitable: int
    overall_pass: bool
    failure_regimes: List[MarketRegime]


# =============================================================================
# Regime Detection Functions
# =============================================================================

def calculate_volatility_percentile(
    prices: List[Tuple[datetime, float]],
    window_hours: int = 24
) -> List[Tuple[datetime, float]]:
    """
    Calculate rolling volatility for each time point.

    Returns list of (timestamp, volatility_pct) tuples.
    Volatility is calculated as the standard deviation of returns
    within the rolling window, expressed as a percentage.
    """
    if len(prices) < 2:
        return []

    volatilities = []
    window_size = max(10, window_hours * 60)  # Approximate ticks per hour

    for i in range(window_size, len(prices)):
        window_prices = [p for _, p in prices[i-window_size:i]]

        # Calculate returns
        returns = [
            (window_prices[j] - window_prices[j-1]) / window_prices[j-1] * 100
            for j in range(1, len(window_prices))
        ]

        if returns:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            vol = math.sqrt(variance)
            volatilities.append((prices[i][0], vol))

    return volatilities


def calculate_trend(
    prices: List[Tuple[datetime, float]],
    lookback_hours: int = 24 * 30  # 30 days
) -> List[Tuple[datetime, float]]:
    """
    Calculate rolling trend (percentage return) for each time point.

    Returns list of (timestamp, trend_pct) tuples.
    """
    if len(prices) < 2:
        return []

    trends = []
    window_size = max(10, lookback_hours * 60)

    for i in range(window_size, len(prices)):
        start_price = prices[i - window_size][1]
        end_price = prices[i][1]

        trend_pct = ((end_price - start_price) / start_price) * 100
        trends.append((prices[i][0], trend_pct))

    return trends


def calculate_data_driven_thresholds(
    volatilities: List[Tuple[datetime, float]],
    trends: List[Tuple[datetime, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    PHASE 5 FIX: Calculate regime thresholds from actual data percentiles.

    Instead of hard-coded thresholds, use the 25th and 75th percentiles
    of the actual volatility and trend distributions.

    Returns:
        Tuple of (vol_thresholds, trend_thresholds)
        where vol_thresholds = (25th_percentile, 75th_percentile)
        and trend_thresholds = (25th_percentile, 75th_percentile)
    """
    # Extract values
    vol_values = sorted([v for _, v in volatilities])
    trend_values = sorted([t for _, t in trends])

    if not vol_values or not trend_values:
        # Fallback to defaults
        return ((0.5, 1.5), (-20.0, 20.0))

    # Calculate percentiles
    def percentile(data: List[float], p: float) -> float:
        idx = int(len(data) * p)
        return data[min(idx, len(data) - 1)]

    vol_25 = percentile(vol_values, 0.25)
    vol_75 = percentile(vol_values, 0.75)
    trend_25 = percentile(trend_values, 0.25)
    trend_75 = percentile(trend_values, 0.75)

    return ((vol_25, vol_75), (trend_25, trend_75))


def classify_volatility(
    vol_pct: float,
    vol_percentiles: Tuple[float, float] = (0.5, 1.5)
) -> VolatilityRegime:
    """
    Classify volatility level.

    Args:
        vol_pct: Volatility percentage
        vol_percentiles: (low_threshold, high_threshold)
    """
    low_thresh, high_thresh = vol_percentiles

    if vol_pct < low_thresh:
        return VolatilityRegime.LOW
    elif vol_pct > high_thresh:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.NORMAL


def classify_trend(
    trend_pct: float,
    trend_thresholds: Tuple[float, float] = (-20.0, 20.0)
) -> TrendRegime:
    """
    Classify trend direction.

    Args:
        trend_pct: 30-day return percentage
        trend_thresholds: (bear_threshold, bull_threshold)
    """
    bear_thresh, bull_thresh = trend_thresholds

    if trend_pct < bear_thresh:
        return TrendRegime.BEAR
    elif trend_pct > bull_thresh:
        return TrendRegime.BULL
    else:
        return TrendRegime.SIDEWAYS


def combine_regimes(vol: VolatilityRegime, trend: TrendRegime) -> MarketRegime:
    """Combine volatility and trend into a single regime."""
    regime_map = {
        (VolatilityRegime.HIGH, TrendRegime.BULL): MarketRegime.HIGH_VOL_BULL,
        (VolatilityRegime.HIGH, TrendRegime.BEAR): MarketRegime.HIGH_VOL_BEAR,
        (VolatilityRegime.HIGH, TrendRegime.SIDEWAYS): MarketRegime.HIGH_VOL_SIDEWAYS,
        (VolatilityRegime.LOW, TrendRegime.BULL): MarketRegime.LOW_VOL_BULL,
        (VolatilityRegime.LOW, TrendRegime.BEAR): MarketRegime.LOW_VOL_BEAR,
        (VolatilityRegime.LOW, TrendRegime.SIDEWAYS): MarketRegime.LOW_VOL_SIDEWAYS,
        (VolatilityRegime.NORMAL, TrendRegime.BULL): MarketRegime.NORMAL_VOL_BULL,
        (VolatilityRegime.NORMAL, TrendRegime.BEAR): MarketRegime.NORMAL_VOL_BEAR,
        (VolatilityRegime.NORMAL, TrendRegime.SIDEWAYS): MarketRegime.NORMAL_VOL_SIDEWAYS,
    }
    return regime_map[(vol, trend)]


def detect_regimes(
    prices: List[Tuple[datetime, float]],
    vol_window_hours: int = 24,
    trend_lookback_hours: int = 24 * 30,
    vol_thresholds: Tuple[float, float] = (0.5, 1.5),
    trend_thresholds: Tuple[float, float] = (-20.0, 20.0),
    verbose: bool = False
) -> List[RegimeLabel]:
    """
    Detect market regimes for each time point in price data.

    Args:
        prices: List of (timestamp, price) tuples
        vol_window_hours: Hours for volatility calculation
        trend_lookback_hours: Hours for trend calculation
        vol_thresholds: (low_vol, high_vol) thresholds
        trend_thresholds: (bear, bull) thresholds
        verbose: Print progress

    Returns:
        List of RegimeLabel for each valid time point
    """
    if verbose:
        print(f"Detecting regimes for {len(prices):,} price points...")

    # Calculate volatility and trend
    vol_data = calculate_volatility_percentile(prices, vol_window_hours)
    trend_data = calculate_trend(prices, trend_lookback_hours)

    if verbose:
        print(f"  Volatility points: {len(vol_data):,}")
        print(f"  Trend points: {len(trend_data):,}")

    # Create timestamp -> value maps
    vol_map = {ts: vol for ts, vol in vol_data}
    trend_map = {ts: trend for ts, trend in trend_data}

    # Find overlapping timestamps
    vol_times = set(vol_map.keys())
    trend_times = set(trend_map.keys())
    common_times = sorted(vol_times & trend_times)

    if verbose:
        print(f"  Common time points: {len(common_times):,}")

    # Generate regime labels
    labels = []
    for ts in common_times:
        vol_pct = vol_map[ts]
        trend_pct = trend_map[ts]

        vol_regime = classify_volatility(vol_pct, vol_thresholds)
        trend_regime = classify_trend(trend_pct, trend_thresholds)
        combined = combine_regimes(vol_regime, trend_regime)

        labels.append(RegimeLabel(
            start_time=ts,
            end_time=ts,  # Single point
            volatility=vol_regime,
            trend=trend_regime,
            combined=combined,
            volatility_pct=vol_pct,
            trend_pct=trend_pct
        ))

    if verbose:
        # Count regimes
        regime_counts = {}
        for label in labels:
            r = label.combined
            regime_counts[r] = regime_counts.get(r, 0) + 1

        print("\n  Regime Distribution:")
        for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
            pct = count / len(labels) * 100
            print(f"    {regime.value}: {count:,} ({pct:.1f}%)")

    return labels


def segment_prices_by_regime(
    prices: List[Tuple[datetime, float]],
    regime_labels: List[RegimeLabel]
) -> Dict[MarketRegime, List[Tuple[datetime, float]]]:
    """
    Segment price data by market regime.

    Returns a dict mapping each regime to its price data.
    """
    # Create timestamp -> regime map
    time_to_regime = {label.start_time: label.combined for label in regime_labels}

    # Segment prices
    segments = {regime: [] for regime in MarketRegime}

    for ts, price in prices:
        if ts in time_to_regime:
            regime = time_to_regime[ts]
            segments[regime].append((ts, price))

    # Remove empty segments
    return {r: p for r, p in segments.items() if p}


# =============================================================================
# Regime Validation Functions
# =============================================================================

def validate_strategy_across_regimes(
    regime_returns: Dict[MarketRegime, List[float]],
    min_rounds_per_regime: int = 10,
    min_profitable_regimes_pct: float = 70.0,
    verbose: bool = False
) -> CrossRegimeValidation:
    """
    Validate that a strategy is profitable across different market regimes.

    A truly robust strategy should be profitable in most regime types,
    not just on average across mixed data.

    Args:
        regime_returns: Dict mapping regime to list of per-round returns
        min_rounds_per_regime: Minimum rounds to consider a regime tested
        min_profitable_regimes_pct: Required % of regimes to be profitable
        verbose: Print details

    Returns:
        CrossRegimeValidation with pass/fail and per-regime results
    """
    results = {}
    regimes_tested = 0
    regimes_profitable = 0
    failure_regimes = []

    for regime, returns in regime_returns.items():
        if len(returns) < min_rounds_per_regime:
            continue  # Skip regimes with insufficient data

        regimes_tested += 1
        total_pnl = sum(returns)
        avg_pnl = total_pnl / len(returns)
        winning = sum(1 for r in returns if r > 0)
        win_rate = winning / len(returns) * 100
        is_profitable = total_pnl > 0

        if is_profitable:
            regimes_profitable += 1
        else:
            failure_regimes.append(regime)

        results[regime] = RegimeValidationResult(
            regime=regime,
            num_rounds=len(returns),
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            win_rate=win_rate,
            is_profitable=is_profitable
        )

    # Check if we pass
    if regimes_tested == 0:
        overall_pass = False
    else:
        profitable_pct = (regimes_profitable / regimes_tested) * 100
        overall_pass = profitable_pct >= min_profitable_regimes_pct

    validation = CrossRegimeValidation(
        regime_results=results,
        regimes_tested=regimes_tested,
        regimes_profitable=regimes_profitable,
        overall_pass=overall_pass,
        failure_regimes=failure_regimes
    )

    if verbose:
        print("\n" + "=" * 60)
        print("CROSS-REGIME VALIDATION RESULTS")
        print("=" * 60)
        print(f"Regimes Tested: {regimes_tested}")
        print(f"Regimes Profitable: {regimes_profitable}")
        if regimes_tested > 0:
            print(f"Success Rate: {regimes_profitable/regimes_tested*100:.1f}%")

        print(f"\nPer-Regime Breakdown:")
        for regime, result in sorted(results.items(), key=lambda x: -x[1].total_pnl):
            status = "PASS" if result.is_profitable else "FAIL"
            print(f"  {regime.value:25} | {result.num_rounds:4} rounds | "
                  f"{result.total_pnl:+7.2f}% | WR: {result.win_rate:5.1f}% | {status}")

        print(f"\nOverall: {'PASSED' if overall_pass else 'FAILED'}")
        if failure_regimes:
            print(f"Failed in: {[r.value for r in failure_regimes]}")
        print("=" * 60)

    return validation


def quick_regime_check(
    prices: List[Tuple[datetime, float]],
    sample_size: int = 1000
) -> Dict[str, float]:
    """
    Quick regime analysis for a subset of prices.

    Useful for getting a quick sense of the data's regime composition
    without processing all data.

    Returns:
        Dict with regime distribution percentages
    """
    if len(prices) < sample_size:
        sample = prices
    else:
        step = len(prices) // sample_size
        sample = prices[::step]

    labels = detect_regimes(sample, verbose=False)

    # Count regimes
    counts = {}
    for label in labels:
        r = label.combined.value
        counts[r] = counts.get(r, 0) + 1

    total = len(labels) if labels else 1
    return {r: c / total * 100 for r, c in counts.items()}


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import random

    print("Testing Regime Detection Module...")

    # Generate synthetic price data with different regimes
    random.seed(42)
    prices = []
    base_time = datetime(2023, 1, 1)
    price = 100.0

    # Simulate 1 year of hourly data with varying regimes
    for hour in range(24 * 365):
        ts = base_time + timedelta(hours=hour)

        # Add regime-dependent noise
        if hour < 24 * 90:  # Q1: Bull market, high vol
            drift = 0.02
            vol = 1.5
        elif hour < 24 * 180:  # Q2: Bear market, high vol
            drift = -0.03
            vol = 2.0
        elif hour < 24 * 270:  # Q3: Sideways, low vol
            drift = 0.001
            vol = 0.3
        else:  # Q4: Bull market, normal vol
            drift = 0.015
            vol = 0.8

        # Random walk with drift
        change = drift + random.gauss(0, vol)
        price *= (1 + change / 100)
        prices.append((ts, price))

    print(f"\nGenerated {len(prices):,} synthetic prices")
    print(f"Price range: ${min(p for _, p in prices):.2f} - ${max(p for _, p in prices):.2f}")

    # Detect regimes
    labels = detect_regimes(prices, verbose=True)

    # Test quick regime check
    print("\n--- Quick Regime Check ---")
    dist = quick_regime_check(prices)
    for regime, pct in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {regime}: {pct:.1f}%")

    # Test cross-regime validation with synthetic returns
    print("\n--- Cross-Regime Validation ---")
    random.seed(123)
    regime_returns = {
        MarketRegime.HIGH_VOL_BULL: [random.gauss(1.5, 3) for _ in range(50)],
        MarketRegime.HIGH_VOL_BEAR: [random.gauss(0.8, 4) for _ in range(40)],
        MarketRegime.HIGH_VOL_SIDEWAYS: [random.gauss(-0.5, 2) for _ in range(30)],
        MarketRegime.LOW_VOL_BULL: [random.gauss(0.3, 1) for _ in range(60)],
        MarketRegime.LOW_VOL_SIDEWAYS: [random.gauss(0.1, 0.5) for _ in range(45)],
        MarketRegime.NORMAL_VOL_BULL: [random.gauss(0.5, 1.5) for _ in range(70)],
    }

    validation = validate_strategy_across_regimes(regime_returns, verbose=True)
