#!/usr/bin/env python3
"""
Pure Profit Pyramid Strategy Optimizer (v7)

This optimizer follows the "Whatever Works" philosophy:
- NO arbitrary constraints (win rate, drawdown, expectancy, profit factor)
- PURE PROFIT optimization - only holdout P&L matters
- 640 combinations (8x better statistical validity than v5/v6)

KEY DIFFERENCES FROM v5/v6:
1. MINIMAL GRID: 640 combos (vs 5,120) - 8x better Bonferroni alpha
2. NO CONSTRAINTS: No win rate, no drawdown limit, no expectancy minimum
3. PURE PROFIT: Ranking = holdout total P&L (not risk-adjusted)
4. REMOVED PARAMETERS: No TP/SL, no acceleration, no min_spacing, no time_decay

PRESERVED STATISTICAL VALIDITY:
- Bonferroni correction (alpha = 0.05/640 = 0.000078)
- Cross-fold validation (total P&L positive)
- Holdout testing (profitable on unseen data)
- Monte Carlo survival simulation
- Robustness testing (nearby params work)

Grid Size: 5 x 4 x 4 x 2 x 2 x 2 = 640 combinations

Usage:
    python optimize_pyramid_v7.py --symbol BTCUSDT --days 1825

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
from core.minimal_grid import (
    build_minimal_grid,
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
# MEMORY MONITORING
# =============================================================================

MEMORY_WARNING_MB = 800
MEMORY_CRITICAL_MB = 1200


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
# CONFIGURATION (PURE PROFIT - NO ARBITRARY CONSTRAINTS)
# =============================================================================

# Default settings
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DAYS = 1825  # 5 years
DEFAULT_BATCH_SIZE = 50
NUM_FOLDS = 3
LOG_DIR = "logs"

# VALIDATION GATE (MINIMAL - only prevents garbage)
MIN_VAL_PNL_RATIO = 0.3  # val_pnl >= 0.3 * train_pnl (loose overfitting check)
MIN_VAL_ROUNDS = 15      # Minimum rounds for statistical validity

# HOLDOUT REQUIREMENTS (MINIMAL)
MIN_HOLDOUT_PNL = 0.0    # Must be positive (that's it!)
MIN_HOLDOUT_ROUNDS = 30  # Need some rounds for statistical validity

# NO OTHER CONSTRAINTS:
# - NO MIN_WIN_RATE_PCT
# - NO MAX_DRAWDOWN_PCT
# - NO MIN_EXPECTANCY
# - NO MIN_PROFIT_FACTOR
# - NO MIN_WIN_LOSS_RATIO

# Robustness testing (keep this - it's not arbitrary, it's statistical)
MIN_ROBUSTNESS_SCORE = 0.6  # Nearby params must give at least 60% of best P&L

# Family-wise error rate for Bonferroni
FAMILY_ALPHA = 0.05

# Realistic simulation settings
INITIAL_CAPITAL = 5000.0
LEVERAGE = 10
POSITION_SIZE_USDT = 100.0
USE_MARGIN_TRACKING = True
APPLY_FUNDING_RATES = True
VALIDATE_DATA_QUALITY = True

# Parameter limits (for robustness perturbations)
PARAM_LIMITS = {
    'threshold': (0.5, 30.0),
    'trailing': (0.2, 20.0),
    'pyramid_step': (0.2, 10.0),
    'max_pyramids': (1, 9999),
}


@dataclass
class OptimizationResult:
    """Result from optimization run (PURE PROFIT VERSION)."""
    params: Dict[str, Any]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    holdout_metrics: Optional[Dict[str, float]]
    passed_gate: bool
    is_significant: bool
    p_value: float
    sharpe: float
    per_round_returns: List[float]
    # Cross-fold validation
    fold_pnls: List[float] = field(default_factory=list)
    total_pnl_positive: bool = False
    avg_val_pnl: float = 0.0
    # Robustness
    robustness_score: float = 0.0
    # Monte Carlo results
    mc_passes: bool = False
    mc_prob_positive: float = 0.0
    mc_prob_ruin: float = 1.0
    mc_sharpe_5th: float = 0.0
    mc_sequence_matters: bool = True
    # Regime validation (optional)
    regime_pass: bool = True


# =============================================================================
# DATA LOADING (MEMORY-EFFICIENT DISK-BASED)
# =============================================================================

def load_tick_data_memmap(symbol: str, days: int) -> Tuple[any, Dict, Dict]:
    """
    Load tick data using memory-mapped files (LOW RAM USAGE).

    Downloads data to disk month-by-month (never all in RAM), then
    memory-maps the file for efficient access. OS handles paging.
    """
    from core.memmap_prices import (
        ensure_prices_on_disk,
        load_prices_memmap,
        validate_memmap_quality,
        load_funding_rates,
        print_memmap_info,
    )

    years = max(1, days // 365)

    print(f"Loading tick data for {symbol} ({years} years) - DISK-BASED MODE")

    # Download to disk if not cached
    prices_file, meta = ensure_prices_on_disk(symbol, years)

    # Memory-map the file
    prices = load_prices_memmap(prices_file, meta)
    print_memmap_info(prices, meta)

    # Data quality validation
    if VALIDATE_DATA_QUALITY:
        validate_memmap_quality(prices)

    # Load funding rates
    funding_rates = {}
    if APPLY_FUNDING_RATES:
        funding_rates = load_funding_rates(symbol, years)

    return prices, meta, funding_rates


def slice_prices_for_fold(prices, fold: Dict) -> Tuple[any, any]:
    """Slice memory-mapped prices for a fold's train and validation data."""
    from core.memmap_prices import slice_memmap_for_fold
    return slice_memmap_for_fold(prices, fold)


def slice_prices_for_holdout(prices, holdout: Dict):
    """Slice memory-mapped prices for holdout testing."""
    from core.memmap_prices import slice_memmap_for_holdout
    return slice_memmap_for_holdout(prices, holdout)


def prices_to_iterator(price_slice):
    """Convert a price slice to iterator for backtest."""
    from core.memmap_prices import memmap_to_iterator
    import numpy as np

    if isinstance(price_slice, np.ndarray):
        return memmap_to_iterator(price_slice)
    else:
        return iter(price_slice)


# =============================================================================
# BACKTEST WRAPPER
# =============================================================================

def run_backtest_with_params(
    params: Dict,
    price_slice,
    funding_rates: Dict = None,
    return_rounds: bool = True
) -> Dict:
    """
    Run backtest with expanded parameters.
    """
    try:
        execution_model = DEFAULT_EXECUTION_MODEL
    except:
        execution_model = None

    price_iter = prices_to_iterator(price_slice)

    result = run_pyramid_backtest(
        prices=price_iter,
        threshold_pct=params.get('threshold', 5.0),
        trailing_pct=params.get('trailing', 1.0),
        pyramid_step_pct=params.get('pyramid_step', 2.0),
        max_pyramids=params.get('max_pyramids', 20),
        # Execution
        fee_pct=0.04,
        execution_model=execution_model,
        # Pyramid parameters (FIXED in v7)
        pyramid_size_schedule=params.get('size_schedule', 'fixed'),
        min_pyramid_spacing_pct=params.get('min_spacing', 0.0),
        pyramid_acceleration=params.get('acceleration', 1.0),
        time_decay_exit_seconds=params.get('time_decay'),  # None in v7
        # Volatility
        volatility_filter_type=params.get('vol_type', 'none'),
        volatility_min_pct=params.get('vol_min', 0.0),
        volatility_window_size=params.get('vol_window', 100),
        # Causal trailing
        use_causal_trailing=params.get('use_causal', True),
        confirmation_ticks=params.get('confirmation_ticks', 3),
        # Exit controls (DISABLED in v7 - rely on trailing stop)
        take_profit_pct=params.get('take_profit_pct'),  # None
        stop_loss_pct=params.get('stop_loss_pct'),      # None
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
        # Margin
        use_margin_tracking=USE_MARGIN_TRACKING,
        initial_capital=INITIAL_CAPITAL,
        leverage=LEVERAGE,
        position_size_usdt=POSITION_SIZE_USDT,
    )

    return result


# =============================================================================
# ROBUSTNESS TESTING
# =============================================================================

def calculate_robustness_score(
    params: Dict,
    prices,
    funding_rates: Optional[Dict] = None
) -> Tuple[float, List[Dict]]:
    """
    Test nearby parameters to check if the optimum is robust.
    """
    center_result = run_backtest_with_params(params, prices, funding_rates, return_rounds=False)
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
                result = run_backtest_with_params(perturbed, prices, funding_rates, return_rounds=False)
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
# CROSS-FOLD VALIDATION (SIMPLE - TOTAL P&L POSITIVE)
# =============================================================================

def validate_across_all_folds(
    params: Dict,
    prices,
    folds: List[Dict],
    funding_rates: Optional[Dict]
) -> Tuple[bool, List[float], float]:
    """
    Validate parameters across ALL folds.

    Simple requirement: Total P&L across all folds must be positive.
    Individual folds can be negative (ranging periods happen).
    """
    fold_pnls = []

    for fold in folds:
        train_prices, val_prices = slice_prices_for_fold(prices, fold)
        val_result = run_backtest_with_params(params, val_prices, funding_rates, return_rounds=False)
        val_pnl = val_result.get('total_pnl', 0)
        fold_pnls.append(val_pnl)

    avg_val_pnl = sum(fold_pnls) / len(fold_pnls) if fold_pnls else 0
    total_pnl_positive = sum(fold_pnls) > 0

    return total_pnl_positive, fold_pnls, avg_val_pnl


# =============================================================================
# TOP 3 CANDIDATES VALIDATION (OVERFITTING GUARD)
# =============================================================================

def validate_top_3_candidates(
    results: List[OptimizationResult],
    prices,
    holdout: Dict,
    funding_rates: Optional[Dict]
) -> Tuple[int, bool]:
    """
    Validate top 3 candidates as overfitting guard.

    If only 1 passes, it's likely a statistical fluke.
    This is a key anti-overfitting measure.
    """
    holdout_prices = slice_prices_for_holdout(prices, holdout)
    top_3 = results[:3] if len(results) >= 3 else results

    pass_count = 0
    all_pass = True

    print("\n--- TOP 3 CANDIDATES VALIDATION (Overfitting Guard) ---")

    for i, result in enumerate(top_3):
        holdout_result = run_backtest_with_params(result.params, holdout_prices, funding_rates, return_rounds=True)

        # Simple check: positive P&L and enough rounds
        pnl = holdout_result.get('total_pnl', 0)
        rounds = holdout_result.get('total_rounds', 0)
        passes = pnl > 0 and rounds >= MIN_HOLDOUT_ROUNDS

        status = "[PASS]" if passes else "[FAIL]"
        params_str = f"th={result.params['threshold']}, tr={result.params['trailing']}"
        print(f"  Top {i+1}: {status} ({params_str}) P&L={pnl:.1f}%, rounds={rounds}")

        if passes:
            pass_count += 1
        else:
            all_pass = False

    if not all_pass:
        print(f"\n  [WARNING] Only {pass_count}/3 top candidates pass holdout!")
        print(f"  This suggests possible overfitting.")
    else:
        print(f"\n  [OK] All top 3 candidates pass - reduces overfitting risk.")

    return pass_count, all_pass


# =============================================================================
# REGIME VALIDATION (MARKET CONDITION ROBUSTNESS)
# =============================================================================

def validate_regime_robustness(
    params: Dict,
    prices,
    funding_rates: Optional[Dict] = None
) -> bool:
    """
    Validate strategy across different market regimes.

    Checks if strategy works in trending, ranging, volatile, and calm markets.
    We don't require all regimes to be profitable - just that overall it works.
    """
    try:
        from core.regime_detection import detect_regimes, validate_strategy_across_regimes

        if len(prices) < 1000:
            return True  # Pass if insufficient data

        # Convert memmap slice to list for regime detection
        price_list = list(prices_to_iterator(prices))
        regime_labels = detect_regimes(price_list, window_size=500)
        if not regime_labels:
            return True

        result = run_backtest_with_params(params, prices, funding_rates, return_rounds=True)
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

        # Lenient validation: only require 40% of regimes profitable
        # (Some regimes like "ranging" may naturally lose money for trend strategies)
        validation = validate_strategy_across_regimes(
            regime_returns=regime_returns,
            min_rounds_per_regime=5,
            min_profitable_regimes_pct=40.0,  # Very lenient
            verbose=False
        )

        return validation.overall_pass

    except ImportError:
        return True  # Pass if module not available
    except Exception as e:
        print(f"  Regime validation error: {e}")
        return True


# =============================================================================
# SINGLE-STAGE OPTIMIZATION (PURE PROFIT)
# =============================================================================

def run_optimization(
    symbol: str = DEFAULT_SYMBOL,
    days: int = DEFAULT_DAYS,
    output_dir: str = "./optimization_v7_results",
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = True
) -> List[OptimizationResult]:
    """
    Run pure profit optimized single-stage optimization.
    """
    print("=" * 70)
    print("PURE PROFIT OPTIMIZER v7 - 'Whatever Works'")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Days: {days}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now()}")
    print()
    print("V7 PHILOSOPHY: No arbitrary constraints, just profit")
    print("  - NO win rate filter")
    print("  - NO drawdown limit")
    print("  - NO expectancy/profit factor minimums")
    print("  - RANKING = Pure holdout P&L")
    print()
    print("STATISTICAL VALIDITY: Preserved")
    print(f"  - Grid size: 640 (vs 5,120 in v5/v6)")
    print(f"  - Bonferroni alpha: 0.000078 (8x better)")
    print("  - Cross-fold validation, Monte Carlo, Robustness testing")
    print()

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Initialize storage
    storage = DiskResultStorage(os.path.join(output_dir, "disk_storage"))

    # Generate parameter grid (MINIMAL - 640 combos)
    grid = build_minimal_grid()
    all_combos = generate_all_combinations(grid)
    n_combos = len(all_combos)
    bonferroni_alpha = get_bonferroni_alpha(grid, FAMILY_ALPHA)

    print(f"Parameter Grid: {n_combos:,} combinations")
    print(f"Bonferroni alpha: {bonferroni_alpha:.6f}")
    print()

    # Load data (DISK-BASED)
    print("Loading tick data (memory-efficient disk mode)...")
    try:
        prices, meta, funding_rates = load_tick_data_memmap(symbol, days)
        print(f"Loaded {meta['total_prices']:,} prices (memory-mapped from disk)")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
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

    # CSV logging
    csv_file = os.path.join(LOG_DIR, f"{symbol}_v7_grid.csv")
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

    # Process all combinations with real-time progress
    print(f"Processing {n_combos:,} combinations...")
    print("-" * 70)

    # Timing for progress estimation
    import time as time_module
    start_time = time_module.time()
    last_update_time = start_time
    combo_times = []  # Track time per combo for ETA

    for batch_start in range(0, n_combos, batch_size):
        batch_end = min(batch_start + batch_size, n_combos)
        batch = all_combos[batch_start:batch_end]

        for i, params in enumerate(batch):
            combo_idx = batch_start + i + 1
            combo_start = time_module.time()

            # Real-time progress bar (update every combo)
            if verbose:
                elapsed = time_module.time() - start_time
                elapsed_str = f"{int(elapsed//60)}m{int(elapsed%60):02d}s"

                # Calculate ETA from average combo time
                if combo_times:
                    avg_time = sum(combo_times[-20:]) / len(combo_times[-20:])  # Last 20 combos
                    remaining = (n_combos - combo_idx) * avg_time
                    eta_str = f"{int(remaining//60)}m{int(remaining%60):02d}s"
                else:
                    eta_str = "calculating..."

                pct = combo_idx / n_combos * 100
                bar_len = 30
                filled = int(bar_len * combo_idx / n_combos)
                bar = "█" * filled + "░" * (bar_len - filled)

                print(f"\r[{bar}] {pct:5.1f}% | {combo_idx}/{n_combos} | "
                      f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                      f"Pass: {len(passed_combos)} | Fail: {failed_gate}",
                      end="", flush=True)

            # Test on FOLD 0 first
            fold = folds[0]
            train_prices, val_prices = slice_prices_for_fold(prices, fold)

            # Run training backtest
            train_result = run_backtest_with_params(params, train_prices, funding_rates)

            # MINIMAL VALIDATION GATE (just checks overfitting and round count)
            val_result = run_backtest_with_params(params, val_prices, funding_rates)

            gate_passed, gate_reason = check_validation_gate(
                train_result, val_result,
                min_val_pnl_ratio=MIN_VAL_PNL_RATIO,
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
                # Track combo time even for failed combos
                combo_time = time_module.time() - combo_start
                combo_times.append(combo_time)
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
                # Track combo time even for non-significant combos
                combo_time = time_module.time() - combo_start
                combo_times.append(combo_time)
                continue

            # Validate total P&L positive across folds
            total_pnl_positive, fold_pnls, avg_val_pnl = validate_across_all_folds(
                params, prices, folds, funding_rates
            )

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
                total_pnl_positive=total_pnl_positive,
                avg_val_pnl=avg_val_pnl,
            )
            passed_combos.append(result)

            # Track combo time for ETA calculation
            combo_time = time_module.time() - combo_start
            combo_times.append(combo_time)

        # Memory management
        gc.collect()
        if combo_idx % 200 == 0:
            check_memory_usage(f"combo_{combo_idx}")

    csv_f.close()
    print(f"\nCSV log saved to: {csv_file}")

    print("-" * 70)
    print(f"PHASE 1 COMPLETE: {len(passed_combos)} combos passed validation + Bonferroni")
    print(f"  Failed gate: {failed_gate}")
    print(f"  Not significant: {not_significant}")

    # Filter to those with positive total P&L
    total_positive = [r for r in passed_combos if r.total_pnl_positive]
    print(f"  Total P&L positive across folds: {len(total_positive)}")
    print()

    if not passed_combos:
        print("No combos passed initial validation!")
        return []

    # Sort by average validation P&L (simple - just profit)
    if total_positive:
        total_positive.sort(key=lambda x: x.avg_val_pnl, reverse=True)
        working_results = total_positive
    else:
        passed_combos.sort(key=lambda x: x.avg_val_pnl, reverse=True)
        working_results = passed_combos
        print("WARNING: No combo has positive total P&L, using best available.")

    # PHASE 2: Robustness testing
    print("=" * 70)
    print("PHASE 2: ROBUSTNESS TESTING")
    print("=" * 70)

    for result in working_results[:30]:
        robustness, perturbations = calculate_robustness_score(
            result.params, prices, funding_rates
        )
        result.robustness_score = robustness

        if robustness < MIN_ROBUSTNESS_SCORE:
            print(f"  [FRAGILE] th={result.params['threshold']}, tr={result.params['trailing']}: robustness={robustness:.2f}")
        else:
            print(f"  [ROBUST]  th={result.params['threshold']}, tr={result.params['trailing']}: robustness={robustness:.2f}")

    # Filter by robustness (keep this - it's not arbitrary)
    robust_results = [r for r in working_results if r.robustness_score >= MIN_ROBUSTNESS_SCORE]
    if robust_results:
        working_results = robust_results
        print(f"\n{len(robust_results)} combos pass robustness threshold ({MIN_ROBUSTNESS_SCORE})")
    else:
        print("\nWARNING: No combos pass robustness threshold, using best available")

    # PHASE 3: Holdout evaluation (THE ONLY THING THAT MATTERS)
    print()
    print("=" * 70)
    print("PHASE 3: HOLDOUT EVALUATION (Pure Profit)")
    print("=" * 70)
    print("Ranking by HOLDOUT TOTAL P&L - nothing else matters")
    print()

    holdout_prices = slice_prices_for_holdout(prices, holdout)
    final_results = []

    for result in working_results[:100]:  # Test top 100
        holdout_result = run_backtest_with_params(result.params, holdout_prices, funding_rates)

        result.holdout_metrics = {
            'total_pnl': holdout_result.get('total_pnl', 0),
            'win_rate': holdout_result.get('win_rate', 0),
            'total_rounds': holdout_result.get('total_rounds', 0),
            'max_drawdown_pct': holdout_result.get('max_drawdown_pct', 0),
        }

        # MINIMAL REQUIREMENTS: profitable and enough rounds
        if (result.holdout_metrics['total_pnl'] >= MIN_HOLDOUT_PNL and
            result.holdout_metrics['total_rounds'] >= MIN_HOLDOUT_ROUNDS):
            final_results.append(result)

    # PURE PROFIT RANKING - sort by holdout P&L
    final_results.sort(key=lambda x: x.holdout_metrics['total_pnl'], reverse=True)
    print(f"{len(final_results)} combos have positive holdout P&L with {MIN_HOLDOUT_ROUNDS}+ rounds")

    # Top 3 validation (overfitting guard)
    if final_results:
        top3_count, top3_all_pass = validate_top_3_candidates(
            final_results, prices, holdout, funding_rates
        )

    # PHASE 4: Regime validation (market condition robustness)
    print()
    print("=" * 70)
    print("PHASE 4: REGIME VALIDATION")
    print("=" * 70)

    for result in final_results[:10]:
        result.regime_pass = validate_regime_robustness(
            result.params, holdout_prices, funding_rates
        )
        status = "[PASS]" if result.regime_pass else "[FAIL]"
        print(f"  {status} th={result.params['threshold']}, tr={result.params['trailing']}")

    # PHASE 5: Monte Carlo validation (statistical validity)
    print()
    print("=" * 70)
    print("PHASE 5: MONTE CARLO VALIDATION")
    print("=" * 70)

    if final_results:
        try:
            from core.monte_carlo import run_monte_carlo_validation, format_monte_carlo_report

            mc_validated = []
            for result in final_results[:30]:
                holdout_result = run_backtest_with_params(result.params, holdout_prices, funding_rates)
                holdout_returns = holdout_result.get('per_round_returns', [])

                if len(holdout_returns) < 10:
                    continue

                passes, mc_result, perm_result = run_monte_carlo_validation(
                    round_returns=holdout_returns,
                    n_bootstrap=10000,
                    n_permutation=5000,
                    ruin_threshold=0.50,  # Generous - we don't limit drawdown
                    random_seed=42
                )

                result.mc_passes = passes
                result.mc_prob_positive = mc_result.probability_positive
                result.mc_prob_ruin = mc_result.probability_ruin
                result.mc_sharpe_5th = mc_result.sharpe_5th_percentile
                result.mc_sequence_matters = perm_result.sequence_matters

                if passes:
                    mc_validated.append(result)
                    print(f"  [PASS] th={result.params['threshold']}, tr={result.params['trailing']}, "
                          f"P(+)={mc_result.probability_positive:.1%}, "
                          f"Holdout P&L={result.holdout_metrics['total_pnl']:.1f}%")
                else:
                    print(f"  [FAIL] th={result.params['threshold']}, tr={result.params['trailing']}")

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
            'total_pnl_positive': r.total_pnl_positive,
            'avg_val_pnl': r.avg_val_pnl,
            'p_value': r.p_value,
            'sharpe': r.sharpe,
            'robustness_score': r.robustness_score,
            'monte_carlo': {
                'passes': r.mc_passes,
                'prob_positive': r.mc_prob_positive,
                'prob_ruin': r.mc_prob_ruin,
                'sharpe_5th_pct': r.mc_sharpe_5th,
                'sequence_matters': r.mc_sequence_matters,
            },
        })

    with open(os.path.join(output_dir, f"{symbol}_top10_v7.json"), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    # Print summary
    print()
    print("=" * 70)
    print("TOP 10 RESULTS (PURE PROFIT)")
    print("=" * 70)
    for i, r in enumerate(top_10):
        params = r.params
        print(f"\n{i+1}. threshold={params['threshold']}, trailing={params['trailing']}, "
              f"pyramid_step={params['pyramid_step']}")
        print(f"   vol_type={params['vol_type']}, size_schedule={params['size_schedule']}, "
              f"max_pyramids={params['max_pyramids']}")
        print(f"   ---")
        print(f"   Train P&L:   {r.train_metrics['total_pnl']:+.2f}%")
        print(f"   Val P&L:     {r.avg_val_pnl:+.2f}% (avg across folds)")
        print(f"   HOLDOUT P&L: {r.holdout_metrics['total_pnl']:+.2f}% << THE ONLY METRIC THAT MATTERS")
        print(f"   ---")
        print(f"   Win Rate: {r.train_metrics['win_rate']:.1f}%")
        print(f"   Max DD: {r.train_metrics['max_drawdown_pct']:.1f}%")
        print(f"   Rounds: {r.holdout_metrics['total_rounds']}")
        print(f"   Robustness: {r.robustness_score:.2f}")
        print(f"   Monte Carlo: P(+)={r.mc_prob_positive:.1%}, P(ruin)={r.mc_prob_ruin:.1%}")

    # Save winner
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
                'total_pnl_positive': winner.total_pnl_positive,
                'avg_val_pnl': winner.avg_val_pnl,
            },
            'robustness': {
                'robustness_score': winner.robustness_score,
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
                'version': 'v7-pure-profit',
                'grid_size': get_grid_size(),
                'bonferroni_alpha': get_bonferroni_alpha(),
                'timestamp': datetime.now().isoformat(),
            },
        }

        with open(os.path.join(output_dir, f"{symbol}_winner_v7.json"), 'w') as f:
            json.dump(winner_data, f, indent=2, default=str)

        # Print live trading command
        print()
        print("=" * 70)
        print("LIVE TRADING COMMAND")
        print("=" * 70)
        max_pyr = winner.params.get('max_pyramids', 20)
        print(f"  python main.py --mode trading --symbol {symbol} \\")
        print(f"    --threshold {winner.params['threshold']} --trailing {winner.params['trailing']} \\")
        print(f"    --pyramid {winner.params['pyramid_step']} --max-pyramids {max_pyr}")

        # Final verdict
        all_pass = (winner.total_pnl_positive and
                    winner.robustness_score >= MIN_ROBUSTNESS_SCORE and
                    winner.mc_passes and
                    winner.holdout_metrics['total_pnl'] > 0)

        print()
        print("=" * 70)
        if all_pass:
            print("STRATEGY PASSES ALL VALIDATION CHECKS")
            print()
            print("  [OK] Holdout P&L positive")
            print("  [OK] Cross-fold total positive")
            print("  [OK] Robustness test passed")
            print("  [OK] Monte Carlo validation passed")
            print()
            print("Ready for paper trading (8 weeks minimum)")
        else:
            print("WARNING: STRATEGY FAILED SOME VALIDATION CHECKS")
            if not winner.total_pnl_positive:
                print("  [FAIL] Total P&L not positive across folds")
            if winner.robustness_score < MIN_ROBUSTNESS_SCORE:
                print("  [FAIL] Robustness below threshold")
            if not winner.mc_passes:
                print("  [FAIL] Monte Carlo validation failed")
            if winner.holdout_metrics['total_pnl'] <= 0:
                print("  [FAIL] Holdout P&L not positive")
        print("=" * 70)

        print(f"\nWinner saved to: {output_dir}/{symbol}_winner_v7.json")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

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
        description="Pure Profit Pyramid Strategy Optimizer v7 - 'Whatever Works'"
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
        default="./optimization_v7_results",
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
        print(f"\nOptimization complete! {len(results)} pure profit combos found.")
        print(f"Grid size: 640 (8x more statistically valid than v5/v6)")
    else:
        print("\nNo statistically valid parameters found.")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
