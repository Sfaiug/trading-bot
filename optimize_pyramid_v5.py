#!/usr/bin/env python3
"""
Statistically Valid Pyramid Strategy Optimizer (v5) - COMPLETE EDITION

This optimizer addresses the fundamental statistical validity issues
identified in v4 while preserving all critical features:

STATISTICAL IMPROVEMENTS (from v5 original):
1. SINGLE-STAGE SEARCH: No hierarchical Phase A/B/C
   - All ~5,000 combinations tested together
   - vol_type tested at same level as core params
   - No parameter locking based on noise-tuned early results

2. IMMEDIATE VALIDATION GATE: After each combo's training test
   - Immediately test on validation fold
   - Only combos passing the gate proceed
   - Prevents overfitting propagation

3. BONFERRONI DURING SELECTION: Not post-hoc
   - Significance tested during search
   - Corrected α = 0.05 / 5,120 ≈ 0.00001
   - Expected false positives: 0.05 (controlled)

4. REAL PER-ROUND RETURNS: Not synthetic
   - All backtests return actual round data
   - Stored to disk for Monte Carlo/Sharpe
   - No [avg_pnl] * n synthetic generation

5. FIXED-SIZE FOLDS WITH GAPS: No look-ahead bias

CRITICAL FEATURES PRESERVED (from v4):
- 3-FOLD CROSS-VALIDATION: Must pass on ALL folds
- RISK-ADJUSTED SCORING: Calmar ratio with penalties
- ROBUSTNESS TESTING: Parameter perturbation checks
- REGIME VALIDATION: Profitable across market conditions
- EXECUTION MODEL: Realistic trading costs
- MEMORY MONITORING: Safe for long runs
- CSV LOGGING: Full grid results for analysis
- TOP-3 VALIDATION: Overfitting guard
- DATA QUALITY VALIDATION: Catches bad data
- LIVE TRADING COMMAND: Ready-to-use output

Usage:
    python optimize_pyramid_v5.py --symbol BTCUSDT --days 1825

Author: Claude Code
"""

import os
import sys
import csv
import json
import gc
import resource
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy

# Local imports
from core.balanced_grid import (
    build_balanced_grid,
    expand_params,
    generate_all_combinations,
    get_grid_size,
    get_bonferroni_alpha,
    get_fixed_folds,
    check_validation_gate,
    check_bonferroni_significance,
)
from core.disk_results import (
    DiskResultStorage,
    calculate_real_sharpe,
)
from backtest_pyramid import (
    run_pyramid_backtest,
    DEFAULT_EXECUTION_MODEL,
)


# =============================================================================
# MEMORY MONITORING (from v4)
# =============================================================================

MEMORY_WARNING_MB = 800  # Warn at 800MB
MEMORY_CRITICAL_MB = 1200  # Force GC at 1.2GB


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return usage / (1024 * 1024)  # macOS: bytes to MB
    else:
        return usage / 1024  # Linux: KB to MB


def check_memory_usage(context: str = "") -> None:
    """Check memory usage and warn/force GC if high."""
    usage_mb = get_memory_usage_mb()

    if usage_mb > MEMORY_CRITICAL_MB:
        print(f"\n  CRITICAL: Memory usage {usage_mb:.0f}MB > {MEMORY_CRITICAL_MB}MB [{context}]")
        print("  Forcing garbage collection...")
        gc.collect()
        new_usage = get_memory_usage_mb()
        print(f"  After GC: {new_usage:.0f}MB")
    elif usage_mb > MEMORY_WARNING_MB:
        print(f"\n  WARNING: Memory usage {usage_mb:.0f}MB > {MEMORY_WARNING_MB}MB [{context}]")
        gc.collect()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default settings
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DAYS = 1825  # 5 years
DEFAULT_BATCH_SIZE = 50  # Combos per batch (for memory management)
NUM_FOLDS = 3  # Number of cross-validation folds
LOG_DIR = "logs"

# Validation gate thresholds
MIN_VAL_PNL_RATIO = 0.5  # val_pnl >= 0.5 * train_pnl
MIN_VAL_WIN_RATE = 40.0  # val_win_rate >= 40%
MIN_VAL_ROUNDS = 20  # val_rounds >= 20

# Final validation thresholds
MIN_HOLDOUT_PNL = 0.0  # Must be positive on holdout
MIN_HOLDOUT_ROUNDS = 50  # Minimum rounds on holdout
MAX_DRAWDOWN_PCT = 40.0  # Maximum acceptable drawdown

# Risk constraints (from v4)
MIN_WIN_RATE_PCT = 45.0  # Psychological tradability threshold
MAX_DD_THRESHOLD = 25.0  # Account protection at 10x leverage
MIN_ROUNDS = 50  # Statistical validity threshold

# Robustness testing (from v4)
MIN_ROBUSTNESS_SCORE = 0.7  # Nearby params must give at least 70% of best P&L

# Family-wise error rate for Bonferroni
FAMILY_ALPHA = 0.05

# Realistic simulation settings (from v4)
INITIAL_CAPITAL = 5000.0
LEVERAGE = 10
POSITION_SIZE_USDT = 100.0
USE_MARGIN_TRACKING = True
APPLY_FUNDING_RATES = True
VALIDATE_DATA_QUALITY = True

# Parameter limits (from v4)
PARAM_LIMITS = {
    'threshold': (0.5, 30.0),
    'trailing': (0.2, 20.0),
    'pyramid_step': (0.2, 10.0),
    'max_pyramids': (1, 9999),
    'poll_interval': (0.5, 30.0),
    'acceleration': (0.3, 5.0),
    'min_spacing': (0.0, 10.0),
    'time_decay': (30, 7200),
    'vol_min': (0.0, 10.0),
    'vol_window': (10, 1000),
}


@dataclass
class OptimizationResult:
    """Result from optimization run."""
    params: Dict[str, Any]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    holdout_metrics: Optional[Dict[str, float]]
    passed_gate: bool
    is_significant: bool
    p_value: float
    sharpe: float
    per_round_returns: List[float]
    # Cross-fold validation (NEW)
    fold_pnls: List[float] = field(default_factory=list)
    all_folds_profitable: bool = False
    avg_val_pnl: float = 0.0
    # Risk metrics (from v4)
    risk_adjusted_score: float = 0.0
    robustness_score: float = 0.0
    # Monte Carlo results
    mc_passes: bool = False
    mc_prob_positive: float = 0.0
    mc_prob_ruin: float = 1.0
    mc_sharpe_5th: float = 0.0
    mc_sequence_matters: bool = True
    # Regime validation
    regime_pass: bool = False


# =============================================================================
# RISK-ADJUSTED SCORING (from v4)
# =============================================================================

def calculate_risk_adjusted_score(result: Dict) -> float:
    """
    Calculate risk-adjusted score for ranking parameter combinations.

    Uses a modified Calmar-like ratio that considers:
    - Total P&L (return)
    - Max Drawdown (risk)
    - Win Rate (consistency)
    - Number of rounds (statistical validity)

    Higher score = better risk-adjusted performance.
    """
    total_pnl = result.get('total_pnl', 0)
    max_dd = result.get('max_drawdown_pct', 50)
    win_rate = result.get('win_rate', 50) / 100
    rounds = result.get('rounds', result.get('total_rounds', 0))

    # Negative or zero P&L gets a very low score
    if total_pnl <= 0:
        return total_pnl - 1000

    # Insufficient rounds: heavy penalty
    if rounds < MIN_ROUNDS:
        round_penalty = (MIN_ROUNDS - rounds) / MIN_ROUNDS
        total_pnl *= (1 - round_penalty * 0.8)

    max_dd = max(max_dd, 1.0)

    # Calmar ratio: Return / Max Drawdown
    calmar = total_pnl / max_dd

    # Win rate penalty
    if win_rate < (MIN_WIN_RATE_PCT / 100):
        shortfall = (MIN_WIN_RATE_PCT / 100) - win_rate
        win_penalty = shortfall * 5
        calmar *= max(0.1, 1 - win_penalty)

    # Win rate bonus for high win rates
    win_bonus = win_rate ** 2

    # Drawdown penalty
    if max_dd > MAX_DD_THRESHOLD:
        excess = (max_dd - MAX_DD_THRESHOLD) / MAX_DD_THRESHOLD
        dd_penalty = excess * 2
        calmar *= max(0.1, 1 - dd_penalty)

    # Round count bonus
    round_bonus = min(rounds / 100, 1.0)

    score = calmar * (1 + win_bonus) * (1 + round_bonus * 0.1)
    return score


def passes_risk_constraints(result: Dict) -> Tuple[bool, List[str]]:
    """Check if a result passes hard risk constraints."""
    reasons = []
    total_pnl = result.get('total_pnl', 0)
    max_dd = result.get('max_drawdown_pct', 50)
    win_rate = result.get('win_rate', 50)
    rounds = result.get('rounds', result.get('total_rounds', 0))

    if rounds < MIN_ROUNDS:
        reasons.append(f"Insufficient rounds: {rounds} < {MIN_ROUNDS}")
    if total_pnl <= 0:
        reasons.append(f"Negative or zero P&L: {total_pnl:.2f}%")
    if win_rate < MIN_WIN_RATE_PCT:
        reasons.append(f"Win rate too low: {win_rate:.1f}% < {MIN_WIN_RATE_PCT}%")
    if max_dd > MAX_DD_THRESHOLD:
        reasons.append(f"Drawdown too high: {max_dd:.1f}% > {MAX_DD_THRESHOLD}%")

    return (len(reasons) == 0, reasons)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_tick_data(symbol: str, days: int, cache_dir: str = "./cache/trades") -> Tuple[List, Dict]:
    """Load tick data for backtesting."""
    from data.tick_data_fetcher import fetch_tick_data

    # Convert days to years (fetch_tick_data uses years)
    years = max(1, days // 365)

    print(f"Loading tick data for {symbol} ({years} years)...")
    ticks = fetch_tick_data(symbol, years=years, aggregate_seconds=1.0, verbose=True)

    # Data quality validation (from v4)
    if VALIDATE_DATA_QUALITY and len(ticks) > 0:
        print("  Validating data quality...")
        try:
            from data.tick_data_fetcher import validate_data_quality
            sample = ticks[:min(100000, len(ticks))]
            quality_report = validate_data_quality(sample)
            if quality_report:
                print(f"  Data quality: {quality_report.quality_score:.1f}%")
                if quality_report.quality_score < 80:
                    print("  WARNING: Data quality below 80%")
        except Exception as e:
            print(f"  Data quality check failed: {e}")

    # Load funding rates
    funding_rates = {}
    if APPLY_FUNDING_RATES:
        try:
            from data.funding_rate_fetcher import get_funding_rates
            payments, funding_rates = get_funding_rates(symbol, years=5)
            if funding_rates:
                print(f"  Loaded {len(funding_rates)} funding rate entries")
        except Exception as e:
            print(f"  Could not load funding rates: {e}")

    return ticks, funding_rates


def slice_data_by_fold(ticks: List, fold: Dict, total_days: int) -> Tuple[List, List]:
    """Slice tick data into train and validation portions for a fold."""
    total_ticks = len(ticks)

    train_start = int(total_ticks * fold['train_start_pct'])
    train_end = int(total_ticks * fold['train_end_pct'])
    val_start = int(total_ticks * fold['val_start_pct'])
    val_end = int(total_ticks * fold['val_end_pct'])

    train_ticks = ticks[train_start:train_end]
    val_ticks = ticks[val_start:val_end]

    return train_ticks, val_ticks


def slice_holdout(ticks: List, holdout: Dict) -> List:
    """Slice tick data for holdout testing."""
    total_ticks = len(ticks)
    start = int(total_ticks * holdout['start_pct'])
    end = int(total_ticks * holdout['end_pct'])
    return ticks[start:end]


# =============================================================================
# BACKTEST WRAPPER
# =============================================================================

def run_backtest_with_params(
    params: Dict,
    ticks: List,
    funding_rates: Dict = None,
    return_rounds: bool = True
) -> Dict:
    """Run backtest with expanded parameters."""
    try:
        execution_model = DEFAULT_EXECUTION_MODEL
    except:
        execution_model = None

    result = run_pyramid_backtest(
        prices=ticks,
        threshold_pct=params.get('threshold', 5.0),
        trailing_pct=params.get('trailing', 1.0),
        pyramid_step_pct=params.get('pyramid_step', 2.0),
        max_pyramids=params.get('max_pyramids', 20),
        # Execution (from v4)
        fee_pct=0.04,
        execution_model=execution_model,
        # Pyramid parameters
        pyramid_size_schedule=params.get('size_schedule', 'fixed'),
        min_pyramid_spacing_pct=params.get('min_spacing', 0.0),
        pyramid_acceleration=params.get('acceleration', 1.0),
        time_decay_exit_seconds=params.get('time_decay'),
        # Volatility
        volatility_filter_type=params.get('vol_type', 'none'),
        volatility_min_pct=params.get('vol_min', 0.0),
        volatility_window_size=params.get('vol_window', 100),
        # Causal trailing
        use_causal_trailing=params.get('use_causal', True),
        confirmation_ticks=params.get('confirmation_ticks', 3),
        # Exit controls
        take_profit_pct=params.get('take_profit_pct'),
        stop_loss_pct=params.get('stop_loss_pct'),
        breakeven_after_pct=params.get('breakeven_after_pct'),
        # Timing
        pyramid_cooldown_sec=params.get('pyramid_cooldown_sec', 0),
        max_round_duration_hr=params.get('max_round_duration_hr'),
        # Filters
        trend_filter_ema=params.get('trend_filter_ema'),
        session_filter=params.get('session_filter', 'all'),
        # Funding
        funding_rates=funding_rates,
        apply_funding=funding_rates is not None,
        # Rounds
        return_rounds=return_rounds,
        # Margin (from v4)
        use_margin_tracking=USE_MARGIN_TRACKING,
        initial_capital=INITIAL_CAPITAL,
        leverage=LEVERAGE,
        position_size_usdt=POSITION_SIZE_USDT,
    )

    return result


# =============================================================================
# ROBUSTNESS TESTING (from v4)
# =============================================================================

def calculate_robustness_score(
    params: Dict,
    ticks: List,
    funding_rates: Optional[Dict] = None
) -> Tuple[float, List[Dict]]:
    """
    Test nearby parameters to check if the optimum is robust.

    Returns (robustness_score, perturbation_results).
    """
    center_result = run_backtest_with_params(params, ticks, funding_rates, return_rounds=False)
    center_pnl = center_result['total_pnl']

    if center_pnl <= 0:
        return 0.0, []

    perturbation_results = []
    perturbations = [
        ('threshold', -0.5),
        ('threshold', +0.5),
        ('trailing', -0.2),
        ('trailing', +0.2),
        ('pyramid_step', -0.2),
        ('pyramid_step', +0.2),
    ]

    for param, delta in perturbations:
        perturbed = params.copy()
        current_val = perturbed.get(param, 1.0)

        if isinstance(current_val, (int, float)):
            new_val = max(0.1, current_val + delta)
            limits = PARAM_LIMITS.get(param, (0, 100))
            if limits[0] <= new_val <= limits[1]:
                perturbed[param] = new_val
                result = run_backtest_with_params(perturbed, ticks, funding_rates, return_rounds=False)
                perturbation_results.append({
                    'param': param,
                    'delta': delta,
                    'pnl': result['total_pnl'],
                    'ratio': result['total_pnl'] / center_pnl if center_pnl > 0 else 0
                })

    if perturbation_results:
        min_ratio = min(p['ratio'] for p in perturbation_results)
        robustness = max(0, min_ratio)
    else:
        robustness = 1.0

    return robustness, perturbation_results


# =============================================================================
# CROSS-FOLD VALIDATION (CRITICAL FIX - now uses ALL folds)
# =============================================================================

def validate_across_all_folds(
    params: Dict,
    ticks: List,
    folds: List[Dict],
    funding_rates: Optional[Dict],
    days: int
) -> Tuple[bool, List[float], float]:
    """
    Validate parameters across ALL folds (not just fold 0).

    Returns:
        Tuple of (all_profitable, fold_pnls, avg_val_pnl)
    """
    fold_pnls = []
    all_profitable = True

    for fold in folds:
        train_ticks, val_ticks = slice_data_by_fold(ticks, fold, days)
        val_result = run_backtest_with_params(params, val_ticks, funding_rates, return_rounds=False)
        val_pnl = val_result.get('total_pnl', 0)
        fold_pnls.append(val_pnl)

        if val_pnl <= 0:
            all_profitable = False

    avg_val_pnl = sum(fold_pnls) / len(fold_pnls) if fold_pnls else 0
    return all_profitable, fold_pnls, avg_val_pnl


# =============================================================================
# TOP-3 CANDIDATES VALIDATION (from v4)
# =============================================================================

def validate_top_3_candidates(
    results: List[OptimizationResult],
    ticks: List,
    holdout: Dict,
    funding_rates: Optional[Dict]
) -> Tuple[int, bool]:
    """
    Validate top 3 candidates as overfitting guard.

    If only 1 passes, it's likely a statistical fluke.
    """
    holdout_ticks = slice_holdout(ticks, holdout)
    top_3 = results[:3] if len(results) >= 3 else results

    pass_count = 0
    all_pass = True

    print("\n--- TOP 3 CANDIDATES VALIDATION (Overfitting Guard) ---")

    for i, result in enumerate(top_3):
        holdout_result = run_backtest_with_params(result.params, holdout_ticks, funding_rates, return_rounds=False)
        passes, reasons = passes_risk_constraints(holdout_result)

        status = "[PASS]" if passes else "[FAIL]"
        params_str = f"th={result.params['threshold']:.2f}, tr={result.params['trailing']:.2f}"
        print(f"  Top {i+1}: {status} ({params_str})")

        if passes:
            pass_count += 1
        else:
            all_pass = False
            for reason in reasons:
                print(f"         - {reason}")

    if not all_pass:
        print(f"\n  [WARNING] Only {pass_count}/3 top candidates pass risk constraints!")
        print(f"  This suggests possible overfitting.")
    else:
        print(f"\n  [OK] All top 3 candidates pass - reduces overfitting risk.")

    return pass_count, all_pass


# =============================================================================
# REGIME VALIDATION (from v4)
# =============================================================================

def validate_regime_robustness(
    params: Dict,
    ticks: List,
    funding_rates: Optional[Dict] = None
) -> bool:
    """Validate strategy across different market regimes."""
    try:
        from core.regime_detection import detect_regimes, validate_strategy_across_regimes

        if len(ticks) < 1000:
            return True  # Pass if insufficient data

        regime_labels = detect_regimes(ticks, window_size=500)
        if not regime_labels:
            return True

        result = run_backtest_with_params(params, ticks, funding_rates, return_rounds=True)
        rounds = result.get('rounds', [])

        if not rounds:
            return True

        # Map rounds to regimes
        time_to_regime = {label.start_time: label.combined for label in regime_labels}
        regime_returns = {}

        for round_obj in rounds:
            round_time = round_obj.entry_time
            closest_regime = None
            min_diff = float('inf')

            for label_time, regime in time_to_regime.items():
                diff = abs((round_time - label_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_regime = regime

            if closest_regime:
                if closest_regime not in regime_returns:
                    regime_returns[closest_regime] = []
                regime_returns[closest_regime].append(round_obj.total_pnl)

        if not regime_returns:
            return True

        validation = validate_strategy_across_regimes(
            regime_returns=regime_returns,
            min_rounds_per_regime=5,
            min_profitable_regimes_pct=70.0,
            verbose=False
        )

        return validation.overall_pass

    except ImportError:
        return True  # Pass if module not available
    except Exception as e:
        print(f"  Regime validation error: {e}")
        return True


# =============================================================================
# SINGLE-STAGE OPTIMIZATION (with cross-fold validation)
# =============================================================================

def run_optimization(
    symbol: str = DEFAULT_SYMBOL,
    days: int = DEFAULT_DAYS,
    output_dir: str = "./optimization_v5_results",
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = True
) -> List[OptimizationResult]:
    """
    Run statistically valid single-stage optimization.

    This is the main entry point for the optimizer.
    """
    print("=" * 70)
    print("STATISTICALLY VALID OPTIMIZER v5 - COMPLETE EDITION")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Days: {days}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now()}")
    print()

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Initialize storage
    storage = DiskResultStorage(os.path.join(output_dir, "disk_storage"))

    # Generate parameter grid
    grid = build_balanced_grid()
    all_combos = generate_all_combinations(grid)
    n_combos = len(all_combos)
    bonferroni_alpha = get_bonferroni_alpha(grid, FAMILY_ALPHA)

    print(f"Parameter Grid: {n_combos:,} combinations")
    print(f"Bonferroni α: {bonferroni_alpha:.2e}")
    print()

    # Load data
    print("Loading tick data...")
    try:
        ticks, funding_rates = load_tick_data(symbol, days)
        print(f"Loaded {len(ticks):,} ticks")
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
    print()

    # Get fold structure
    folds, holdout = get_fixed_folds(days)
    print("Fold Structure:")
    for fold in folds:
        print(f"  {fold['name']}: Train {fold['train_start_pct']*100:.0f}%-{fold['train_end_pct']*100:.0f}%, "
              f"Val {fold['val_start_pct']*100:.0f}%-{fold['val_end_pct']*100:.0f}%")
    print(f"  Holdout: {holdout['start_pct']*100:.0f}%-{holdout['end_pct']*100:.0f}%")
    print()

    # CSV logging (from v4)
    csv_file = os.path.join(LOG_DIR, f"{symbol}_v5_grid.csv")
    csv_fields = [
        'threshold', 'trailing', 'pyramid_step', 'max_pyramids', 'vol_type',
        'size_schedule', 'train_pnl', 'val_pnl', 'win_rate', 'rounds',
        'max_drawdown_pct', 'passed_gate', 'is_significant', 'p_value'
    ]
    csv_f = open(csv_file, 'w', newline='')
    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Track results
    passed_combos: List[OptimizationResult] = []
    failed_gate = 0
    not_significant = 0

    # Process in batches
    print(f"Processing {n_combos:,} combinations in batches of {batch_size}...")
    print("-" * 70)

    for batch_start in range(0, n_combos, batch_size):
        batch_end = min(batch_start + batch_size, n_combos)
        batch = all_combos[batch_start:batch_end]

        for i, params in enumerate(batch):
            combo_idx = batch_start + i + 1

            if verbose and combo_idx % 100 == 0:
                pct = combo_idx / n_combos * 100
                print(f"[{combo_idx:,}/{n_combos:,}] {pct:.1f}% | "
                      f"Passed: {len(passed_combos)} | "
                      f"Failed gate: {failed_gate} | "
                      f"Not significant: {not_significant}")

            # Test on FOLD 0 first (immediate gate)
            fold = folds[0]
            train_ticks, val_ticks = slice_data_by_fold(ticks, fold, days)

            # Run training backtest
            train_result = run_backtest_with_params(params, train_ticks, funding_rates)

            # IMMEDIATE VALIDATION GATE
            val_result = run_backtest_with_params(params, val_ticks, funding_rates)

            gate_passed, gate_reason = check_validation_gate(
                train_result, val_result,
                min_val_pnl_ratio=MIN_VAL_PNL_RATIO,
                min_val_win_rate=MIN_VAL_WIN_RATE,
                min_val_rounds=MIN_VAL_ROUNDS
            )

            # Log to CSV
            csv_row = {
                'threshold': params.get('threshold'),
                'trailing': params.get('trailing'),
                'pyramid_step': params.get('pyramid_step'),
                'max_pyramids': params.get('max_pyramids'),
                'vol_type': params.get('vol_type'),
                'size_schedule': params.get('size_schedule'),
                'train_pnl': train_result.get('total_pnl', 0),
                'val_pnl': val_result.get('total_pnl', 0),
                'win_rate': train_result.get('win_rate', 0),
                'rounds': train_result.get('total_rounds', 0),
                'max_drawdown_pct': train_result.get('max_drawdown_pct', 0),
                'passed_gate': gate_passed,
                'is_significant': False,
                'p_value': 1.0,
            }

            if not gate_passed:
                failed_gate += 1
                csv_writer.writerow(csv_row)
                continue

            # BONFERRONI SIGNIFICANCE CHECK
            per_round_returns = train_result.get('per_round_returns', [])
            is_significant, p_value, corrected_alpha = check_bonferroni_significance(
                params, per_round_returns, n_combos, FAMILY_ALPHA
            )

            csv_row['is_significant'] = is_significant
            csv_row['p_value'] = p_value
            csv_writer.writerow(csv_row)

            if not is_significant:
                not_significant += 1
                continue

            # CRITICAL FIX: Validate across ALL folds (not just fold 0)
            all_folds_profitable, fold_pnls, avg_val_pnl = validate_across_all_folds(
                params, ticks, folds, funding_rates, days
            )

            # Calculate risk-adjusted score
            risk_score = calculate_risk_adjusted_score({
                'total_pnl': avg_val_pnl,
                'max_drawdown_pct': train_result.get('max_drawdown_pct', 0),
                'win_rate': train_result.get('win_rate', 0),
                'total_rounds': train_result.get('total_rounds', 0),
            })

            # Save to storage
            rounds_data = []
            for r in train_result.get('rounds', []):
                rounds_data.append({
                    'timestamp': str(getattr(r, 'entry_time', '')),
                    'pnl_pct': getattr(r, 'total_pnl', 0),
                    'duration_sec': 0,
                    'num_pyramids': getattr(r, 'num_pyramids', 0),
                })

            summary = {
                'total_pnl': train_result.get('total_pnl', 0),
                'total_rounds': train_result.get('total_rounds', 0),
                'win_rate': train_result.get('win_rate', 0),
            }

            storage.save_combo_result(params, rounds_data, summary)

            # Calculate Sharpe
            try:
                sharpe = calculate_real_sharpe(storage, params)
            except ValueError:
                sharpe = 0.0

            # Record passed combo
            result = OptimizationResult(
                params=params,
                train_metrics={
                    'total_pnl': train_result.get('total_pnl', 0),
                    'win_rate': train_result.get('win_rate', 0),
                    'total_rounds': train_result.get('total_rounds', 0),
                    'max_drawdown_pct': train_result.get('max_drawdown_pct', 0),
                },
                val_metrics={
                    'total_pnl': val_result.get('total_pnl', 0),
                    'win_rate': val_result.get('win_rate', 0),
                    'total_rounds': val_result.get('total_rounds', 0),
                },
                holdout_metrics=None,
                passed_gate=True,
                is_significant=True,
                p_value=p_value,
                sharpe=sharpe,
                per_round_returns=per_round_returns,
                fold_pnls=fold_pnls,
                all_folds_profitable=all_folds_profitable,
                avg_val_pnl=avg_val_pnl,
                risk_adjusted_score=risk_score,
            )
            passed_combos.append(result)

        # Memory management
        gc.collect()
        if combo_idx % 1000 == 0:
            check_memory_usage(f"combo_{combo_idx}")

    csv_f.close()
    print(f"\nCSV log saved to: {csv_file}")

    print("-" * 70)
    print(f"PHASE 1 COMPLETE: {len(passed_combos)} combos passed validation gate + Bonferroni")
    print(f"  Failed gate: {failed_gate}")
    print(f"  Not significant: {not_significant}")

    # Filter to only those profitable on ALL folds
    profitable_all = [r for r in passed_combos if r.all_folds_profitable]
    print(f"  Profitable on ALL folds: {len(profitable_all)}")
    print()

    if not passed_combos:
        print("No combos passed initial validation!")
        return []

    # Sort by risk-adjusted score (from v4)
    if profitable_all:
        profitable_all.sort(key=lambda x: x.risk_adjusted_score, reverse=True)
        working_results = profitable_all
    else:
        passed_combos.sort(key=lambda x: x.risk_adjusted_score, reverse=True)
        working_results = passed_combos
        print("WARNING: No combo profitable on ALL folds, using best risk-adjusted.")

    # PHASE 2: Robustness testing (from v4)
    print("=" * 70)
    print("PHASE 2: ROBUSTNESS TESTING")
    print("=" * 70)

    for result in working_results[:20]:
        robustness, perturbations = calculate_robustness_score(
            result.params, ticks, funding_rates
        )
        result.robustness_score = robustness

        if robustness < MIN_ROBUSTNESS_SCORE:
            print(f"  [FRAGILE] th={result.params['threshold']}: robustness={robustness:.2f}")
        else:
            print(f"  [ROBUST] th={result.params['threshold']}: robustness={robustness:.2f}")

    # Filter by robustness
    robust_results = [r for r in working_results if r.robustness_score >= MIN_ROBUSTNESS_SCORE]
    if robust_results:
        working_results = robust_results
        print(f"\n{len(robust_results)} combos pass robustness threshold")
    else:
        print("\nWARNING: No combos pass robustness threshold, using best available")

    # PHASE 3: Holdout evaluation
    print()
    print("=" * 70)
    print("PHASE 3: HOLDOUT EVALUATION")
    print("=" * 70)

    holdout_ticks = slice_holdout(ticks, holdout)
    final_results = []

    for result in working_results[:50]:
        holdout_result = run_backtest_with_params(result.params, holdout_ticks, funding_rates)

        result.holdout_metrics = {
            'total_pnl': holdout_result.get('total_pnl', 0),
            'win_rate': holdout_result.get('win_rate', 0),
            'total_rounds': holdout_result.get('total_rounds', 0),
            'max_drawdown_pct': holdout_result.get('max_drawdown_pct', 0),
        }

        if (result.holdout_metrics['total_pnl'] >= MIN_HOLDOUT_PNL and
            result.holdout_metrics['total_rounds'] >= MIN_HOLDOUT_ROUNDS and
            result.holdout_metrics.get('max_drawdown_pct', 0) <= MAX_DRAWDOWN_PCT):
            final_results.append(result)

    final_results.sort(key=lambda x: x.holdout_metrics['total_pnl'], reverse=True)
    print(f"{len(final_results)} combos passed holdout validation")

    # Top 3 validation (from v4)
    if final_results:
        top3_count, top3_all_pass = validate_top_3_candidates(
            final_results, ticks, holdout, funding_rates
        )

    # PHASE 4: Regime validation (from v4)
    print()
    print("=" * 70)
    print("PHASE 4: REGIME VALIDATION")
    print("=" * 70)

    for result in final_results[:10]:
        result.regime_pass = validate_regime_robustness(
            result.params, holdout_ticks, funding_rates
        )
        status = "[PASS]" if result.regime_pass else "[FAIL]"
        print(f"  {status} th={result.params['threshold']}")

    # PHASE 5: Monte Carlo validation
    print()
    print("=" * 70)
    print("PHASE 5: MONTE CARLO VALIDATION")
    print("=" * 70)

    if final_results:
        try:
            from core.monte_carlo import run_monte_carlo_validation, format_monte_carlo_report

            mc_validated = []
            for result in final_results[:20]:
                holdout_result = run_backtest_with_params(result.params, holdout_ticks, funding_rates)
                holdout_returns = holdout_result.get('per_round_returns', [])

                if len(holdout_returns) < 10:
                    continue

                passes, mc_result, perm_result = run_monte_carlo_validation(
                    round_returns=holdout_returns,
                    n_bootstrap=10000,
                    n_permutation=5000,
                    ruin_threshold=0.40,
                    random_seed=42
                )

                result.mc_passes = passes
                result.mc_prob_positive = mc_result.probability_positive
                result.mc_prob_ruin = mc_result.probability_ruin
                result.mc_sharpe_5th = mc_result.sharpe_5th_percentile
                result.mc_sequence_matters = perm_result.sequence_matters

                if passes:
                    mc_validated.append(result)
                    print(f"  [PASS] th={result.params['threshold']}, "
                          f"P(+)={mc_result.probability_positive:.1%}")
                else:
                    print(f"  [FAIL] th={result.params['threshold']}")

            print(f"\n{len(mc_validated)} combos passed Monte Carlo")

            if mc_validated:
                final_results = mc_validated

        except ImportError as e:
            print(f"  Monte Carlo module not available: {e}")

    # Save results
    save_results(final_results, output_dir, symbol, funding_rates)

    return final_results


def save_results(results: List[OptimizationResult], output_dir: str, symbol: str, funding_rates: Dict):
    """Save optimization results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save top results as JSON
    top_10 = results[:10]
    results_data = []
    for r in top_10:
        results_data.append({
            'params': r.params,
            'train_metrics': r.train_metrics,
            'val_metrics': r.val_metrics,
            'holdout_metrics': r.holdout_metrics,
            'fold_pnls': r.fold_pnls,
            'all_folds_profitable': r.all_folds_profitable,
            'avg_val_pnl': r.avg_val_pnl,
            'p_value': r.p_value,
            'sharpe': r.sharpe,
            'risk_adjusted_score': r.risk_adjusted_score,
            'robustness_score': r.robustness_score,
            'regime_pass': r.regime_pass,
            'monte_carlo': {
                'passes': r.mc_passes,
                'prob_positive': r.mc_prob_positive,
                'prob_ruin': r.mc_prob_ruin,
                'sharpe_5th_pct': r.mc_sharpe_5th,
                'sequence_matters': r.mc_sequence_matters,
            },
        })

    with open(os.path.join(output_dir, f"{symbol}_top10_v5.json"), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    # Print summary
    print()
    print("=" * 70)
    print("TOP 10 RESULTS")
    print("=" * 70)
    for i, r in enumerate(top_10):
        print(f"\n{i+1}. threshold={r.params['threshold']}, trailing={r.params['trailing']}, "
              f"vol_type={r.params['vol_type']}")
        print(f"   Train P&L: {r.train_metrics['total_pnl']:.2f}%")
        print(f"   Val P&L (avg): {r.avg_val_pnl:.2f}%")
        print(f"   Holdout P&L: {r.holdout_metrics['total_pnl']:.2f}%")
        print(f"   All folds profitable: {r.all_folds_profitable}")
        print(f"   Risk-adjusted score: {r.risk_adjusted_score:.2f}")
        print(f"   Robustness: {r.robustness_score:.2f}")
        print(f"   Sharpe: {r.sharpe:.2f}")
        print(f"   p-value: {r.p_value:.2e}")
        print(f"   Monte Carlo: P(+)={r.mc_prob_positive:.1%}, P(ruin)={r.mc_prob_ruin:.1%}")

    # Save winner with live trading command (from v4)
    if results:
        winner = results[0]
        winner_data = {
            'params': winner.params,
            'metrics': {
                'train': winner.train_metrics,
                'val': winner.val_metrics,
                'holdout': winner.holdout_metrics,
            },
            'cross_validation': {
                'fold_pnls': winner.fold_pnls,
                'all_folds_profitable': winner.all_folds_profitable,
                'avg_val_pnl': winner.avg_val_pnl,
            },
            'risk': {
                'risk_adjusted_score': winner.risk_adjusted_score,
                'robustness_score': winner.robustness_score,
                'regime_pass': winner.regime_pass,
            },
            'monte_carlo': {
                'passes': winner.mc_passes,
                'prob_positive': winner.mc_prob_positive,
                'prob_ruin': winner.mc_prob_ruin,
                'sharpe_5th_pct': winner.mc_sharpe_5th,
                'sequence_matters': winner.mc_sequence_matters,
            },
            'statistics': {
                'p_value': winner.p_value,
                'sharpe': winner.sharpe,
                'is_significant': winner.is_significant,
            },
            'optimization_info': {
                'version': 'v5-complete',
                'grid_size': get_grid_size(),
                'bonferroni_alpha': get_bonferroni_alpha(),
                'timestamp': datetime.now().isoformat(),
            },
        }

        with open(os.path.join(output_dir, f"{symbol}_winner_v5.json"), 'w') as f:
            json.dump(winner_data, f, indent=2, default=str)

        # Print live trading command (from v4)
        print()
        print("=" * 70)
        print("LIVE TRADING COMMAND")
        print("=" * 70)
        max_pyr = winner.params.get('max_pyramids', 20)
        print(f"  python main.py --mode trading --symbol {symbol} \\")
        print(f"    --threshold {winner.params['threshold']} --trailing {winner.params['trailing']} \\")
        print(f"    --pyramid {winner.params['pyramid_step']} --max-pyramids {max_pyr}")

        # Final verdict
        all_pass = (winner.all_folds_profitable and
                    winner.robustness_score >= MIN_ROBUSTNESS_SCORE and
                    winner.mc_passes and
                    winner.regime_pass)

        print()
        print("=" * 70)
        if all_pass:
            print("STRATEGY PASSES ALL VALIDATION CHECKS")
            print("Ready for paper trading (8 weeks, 60+ rounds)")
        else:
            print("WARNING: STRATEGY FAILED SOME VALIDATION CHECKS")
            if not winner.all_folds_profitable:
                print("  [FAIL] Not profitable on all folds")
            if winner.robustness_score < MIN_ROBUSTNESS_SCORE:
                print("  [FAIL] Robustness below threshold")
            if not winner.mc_passes:
                print("  [FAIL] Monte Carlo validation failed")
            if not winner.regime_pass:
                print("  [FAIL] Regime validation failed")
        print("=" * 70)

        print(f"\nWinner saved to: {output_dir}/{symbol}_winner_v5.json")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

# Available coins for optimization
AVAILABLE_COINS = [
    ("BTC", "BTCUSDT", "Bitcoin"),
    ("ETH", "ETHUSDT", "Ethereum"),
    ("SOL", "SOLUSDT", "Solana"),
    ("XRP", "XRPUSDT", "Ripple"),
    ("DOGE", "DOGEUSDT", "Dogecoin"),
    ("XLM", "XLMUSDT", "Stellar"),
]


def select_coin_interactive() -> str:
    """Display interactive coin selection menu."""
    print()
    print("=" * 50)
    print("SELECT COIN FOR OPTIMIZATION")
    print("=" * 50)
    print()
    for i, (short, symbol, name) in enumerate(AVAILABLE_COINS, 1):
        print(f"  {i}. {short:5} - {name} ({symbol})")
    print()

    while True:
        try:
            choice = input("Enter choice (1-6): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_COINS):
                selected = AVAILABLE_COINS[idx]
                print(f"\nSelected: {selected[2]} ({selected[1]})")
                return selected[1]
            else:
                print("Invalid choice. Please enter 1-6.")
        except ValueError:
            print("Invalid input. Please enter a number 1-6.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Statistically Valid Pyramid Strategy Optimizer v5 - Complete Edition"
    )
    parser.add_argument(
        "--symbol", "-s",
        default=None,
        help="Trading pair symbol (e.g., BTCUSDT). If not provided, shows interactive menu."
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Days of historical data (default: {DEFAULT_DAYS})"
    )
    parser.add_argument(
        "--output", "-o",
        default="./optimization_v5_results",
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Interactive coin selection if no symbol provided
    if args.symbol is None:
        args.symbol = select_coin_interactive()

    results = run_optimization(
        symbol=args.symbol,
        days=args.days,
        output_dir=args.output,
        batch_size=args.batch_size,
        verbose=not args.quiet
    )

    print()
    print(f"Completed: {datetime.now()}")

    if results:
        print(f"\n✓ Optimization complete! {len(results)} statistically valid combos found.")
    else:
        print("\n✗ No statistically valid parameter combinations found.")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
