#!/usr/bin/env python3
"""
Autonomous Single-Coin Pyramid Strategy Optimizer v4 - ULTIMATE EDITION

Key Improvements over v3:
1. DISK STREAMING: Never loads all ticks to memory (8GB RAM safe)
2. HIERARCHICAL OPTIMIZATION: Dense core params, then refine new params
3. 3-FOLD WALK-FORWARD: Must be profitable across ALL validation periods
4. FAILURE HANDLING: Expands search if all validation fails
5. ROBUSTNESS CHECK: Rejects fragile parameter sets

Usage:
    python optimize_pyramid_v4.py

    # When prompted, select coin and years of data
    # Optimizer runs for weeks until finding absolute best
"""

import os
import sys
import csv
import json
import time
import gc
import struct
import resource  # For memory monitoring
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Callable, Generator, Any
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.tick_data_fetcher import (
    get_years_from_user,
    create_filtered_tick_streamer,
    create_configurable_tick_streamer,
    DataGranularity,
    count_filtered_ticks,
    aggregate_ticks_to_interval
)
from backtest_pyramid import (
    run_pyramid_backtest,
    PyramidRound,
    ExecutionModel,
    DEFAULT_EXECUTION_MODEL,
    CONSERVATIVE_EXECUTION_MODEL,
    # PHASE 5: Realistic simulation imports
    RiskLimits,
    DEFAULT_RISK_LIMITS
)
# PHASE 5: Funding rate and data quality imports
from data.funding_rate_fetcher import get_funding_rates, create_funding_rate_lookup
from data.tick_data_fetcher import validate_data_quality, DataQualityReport
from core.statistical_validation import (
    validate_strategy_statistically,
    quick_validation_check,
    StrategyValidationResult
)
from core.regime_detection import (
    MarketRegime,
    detect_regimes,
    validate_strategy_across_regimes,
    CrossRegimeValidation
)
# PHASE 5: Monte Carlo stress testing
from core.monte_carlo import (
    run_monte_carlo_validation,
    format_monte_carlo_report,
    MonteCarloResult,
    PermutationTestResult
)


# =============================================================================
# MEMORY MONITORING
# =============================================================================

# Memory limit per instance (for 6 concurrent on 8GB system)
MEMORY_WARNING_MB = 800  # Warn at 800MB
MEMORY_CRITICAL_MB = 1200  # Force GC at 1.2GB


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    # ru_maxrss is in bytes on macOS, KB on Linux
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

DEFAULT_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "XLMUSDT"]

# Optimization phases
MAX_ROUNDS_PER_PHASE = 8
CONVERGENCE_THRESHOLD = 0.05  # 0.05% improvement threshold (tighter for dense grid)
TOP_N_CORE = 20  # Top core param combos to refine
TOP_N_FINAL = 10  # Top final combos for fine-tuning
NUM_FOLDS = 3  # Walk-forward validation folds

# Failure handling
MAX_EXPANSION_ATTEMPTS = 3
MIN_ACCEPTABLE_VAL_PNL = 10.0  # Minimum 10% validation P&L to accept

# Robustness thresholds
MIN_ROBUSTNESS_SCORE = 0.7  # Nearby params must give at least 70% of best P&L

LOG_DIR = "logs"
CACHE_DIR = "cache/folds"

# Parameter limits (absolute bounds)
PARAM_LIMITS = {
    'threshold': (0.5, 30.0),
    'trailing': (0.2, 20.0),
    'pyramid_step': (0.2, 10.0),
    'max_pyramids': (1, 9999),
    'poll_interval': (0.5, 30.0),  # Minimum 0.5s (realistic for live trading)
    'acceleration': (0.3, 5.0),
    'min_spacing': (0.0, 10.0),
    'time_decay': (30, 7200),
    'vol_min': (0.0, 10.0),
    'vol_window': (10, 1000),
}

# =============================================================================
# PHASE 5: REALISTIC SIMULATION SETTINGS
# =============================================================================
# These settings ensure the optimizer uses realistic trading conditions
# matching what would happen in live trading on Binance Futures

INITIAL_CAPITAL = 5000.0  # Starting capital in USDT
LEVERAGE = 10  # 10x leverage (liquidation at ~10% adverse move)
POSITION_SIZE_USDT = 100.0  # Position size per unit in USDT
MAX_REALISTIC_PYRAMIDS = 50  # With $5K at 10x, max ~50 pyramids feasible

# Enable/disable realistic features (all should be True for production)
USE_MARGIN_TRACKING = True  # Track margin, enforce liquidation
APPLY_FUNDING_RATES = True  # Apply 8-hour funding (5-20% annual drag)
VALIDATE_DATA_QUALITY = True  # Check for gaps, anomalies before optimization


# =============================================================================
# VERY DENSE PARAMETER GRIDS
# =============================================================================

def build_dense_core_grid() -> Dict[str, List]:
    """
    Dense grid for the 5 core parameters.
    Other parameters use neutral defaults during Phase A.
    """
    return {
        # Core 5 parameters - VERY DENSE
        'threshold': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],  # 15
        'trailing': [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0],  # 11
        'pyramid_step': [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0],  # 10
        'max_pyramids': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],  # 10 (capped at MAX_REALISTIC_PYRAMIDS)
        # FIXED: Set poll_interval to exactly match live trading's PRICE_CHECK_INTERVAL default.
        # Live trading uses config/settings.py:PRICE_CHECK_INTERVAL which defaults to 1.0s.
        # Optimizing for different intervals would produce params that don't transfer to live.
        'poll_interval': [1.0],  # Must match live PRICE_CHECK_INTERVAL (default: 1.0s)
        # Neutral defaults for other params
        'size_schedule': ['fixed'],
        'acceleration': [1.0],
        'min_spacing': [0.0],
        'time_decay': [None],
        'vol_type': ['none'],
        'vol_min': [0.0],
        'vol_window': [100],
        # PHASE 1 FIX: Causal trailing stop params (always use causal mode)
        'use_causal': [True],
        'confirmation_ticks': [3],  # Default: 3 ticks to confirm peak
    }
    # Core combos: 15 × 11 × 10 × 12 × 11 = 217,800


def build_dense_new_params_grid() -> Dict[str, List]:
    """
    Phase 6: Enable ALL parameters for comprehensive optimization.
    User confirmed unlimited runtime is acceptable.

    Existing params (now enabled):
    - size_schedule, acceleration, min_spacing, time_decay
    - vol_type, vol_min, vol_window, confirmation_ticks

    New params (Phase 6):
    - take_profit_pct, stop_loss_pct, breakeven_after_pct
    - pyramid_cooldown_sec, max_round_duration_hr
    - trend_filter_ema, session_filter

    Grid size: 3×5×5×4×3×4×3×4 = 21,600 combos (existing)
              × 6×5×4×4×5×5×4 = 48,000 (new) = ~1B total

    Too many! Use multi-stage search (see build_stage_X_grids).
    """
    return {
        # EXISTING PARAMS (now enabled with full grids)
        'size_schedule': ['fixed', 'linear_decay', 'exp_decay'],  # 3
        'acceleration': [0.8, 0.9, 1.0, 1.1, 1.2],  # 5
        'min_spacing': [0.0, 0.1, 0.2, 0.3, 0.5],  # 5
        'time_decay': [None, 300, 900, 3600],  # 4 (seconds: None, 5min, 15min, 1hr)
        'vol_type': ['none', 'stddev', 'range'],  # 3
        'vol_min': [0.0, 0.5, 1.0, 2.0],  # 4
        'vol_window': [50, 100, 200],  # 3
        'confirmation_ticks': [1, 2, 3, 5],  # 4
        # NEW PARAMS (Phase 6)
        'take_profit_pct': [None, 5, 10, 15, 20, 30],  # 6
        'stop_loss_pct': [None, 10, 15, 20, 30],  # 5
        'breakeven_after_pct': [None, 3, 5, 7],  # 4
        'pyramid_cooldown_sec': [0, 60, 300, 600],  # 4
        'max_round_duration_hr': [None, 1, 4, 8, 24],  # 5
        'trend_filter_ema': [None, 20, 50, 100, 200],  # 5
        'session_filter': ['all', 'asia', 'europe', 'us'],  # 4
    }
    # Total: 3×5×5×4×3×4×3×4 × 6×5×4×4×5×5×4 = ~1 billion
    # Will use multi-stage search to make this tractable


def build_fine_tuning_perturbations(best_params: Dict) -> List[Dict]:
    """
    Generate a list of small perturbations around the best parameters.
    Tests ONE parameter at a time, not all combinations.

    Phase 6: Now includes perturbations for ALL params including:
    - Core 4: threshold, trailing, pyramid_step, max_pyramids
    - Extended: acceleration, min_spacing, confirmation_ticks
    - New: take_profit_pct, stop_loss_pct, breakeven_after_pct,
           pyramid_cooldown_sec, max_round_duration_hr, trend_filter_ema

    Returns list of param dicts to test (~100 total).
    """
    perturbations = [best_params.copy()]  # Start with the best

    # Numeric params with delta values
    numeric_deltas = {
        # Core params
        'threshold': [0.25, 0.5, 1.0],
        'trailing': [0.05, 0.1, 0.2],
        'pyramid_step': [0.05, 0.1, 0.2],
        # Extended params (now enabled)
        'acceleration': [0.05, 0.1],
        'min_spacing': [0.05, 0.1],
        # New params (Phase 6)
        'take_profit_pct': [1, 2, 5],
        'stop_loss_pct': [2, 5],
        'breakeven_after_pct': [1, 2],
        'pyramid_cooldown_sec': [30, 60, 120],
        'max_round_duration_hr': [0.5, 1, 2],
    }

    for param, deltas in numeric_deltas.items():
        best_val = best_params.get(param)
        if best_val is None or not isinstance(best_val, (int, float)):
            continue
        limits = PARAM_LIMITS.get(param, (0, 100))

        for delta in deltas:
            # Test +delta
            new_val = round(best_val + delta, 3)
            if limits[0] <= new_val <= limits[1]:
                new_params = best_params.copy()
                new_params[param] = new_val
                perturbations.append(new_params)

            # Test -delta
            new_val = round(best_val - delta, 3)
            if limits[0] <= new_val <= limits[1]:
                new_params = best_params.copy()
                new_params[param] = new_val
                perturbations.append(new_params)

    # For max_pyramids
    max_pyr = best_params.get('max_pyramids', 80)
    if max_pyr != 'unlimited' and isinstance(max_pyr, int) and max_pyr < 9999:
        for delta_pct in [0.05, 0.1, 0.2]:
            delta = max(1, int(max_pyr * delta_pct))
            for sign in [1, -1]:
                new_val = max(1, max_pyr + sign * delta)
                new_params = best_params.copy()
                new_params['max_pyramids'] = new_val
                perturbations.append(new_params)

    # For confirmation_ticks (integer param)
    conf_ticks = best_params.get('confirmation_ticks', 3)
    if isinstance(conf_ticks, int):
        for delta in [-1, 1, 2]:
            new_val = max(1, conf_ticks + delta)
            if new_val != conf_ticks:
                new_params = best_params.copy()
                new_params['confirmation_ticks'] = new_val
                perturbations.append(new_params)

    # For trend_filter_ema (None or integer EMA period)
    ema = best_params.get('trend_filter_ema')
    if ema is not None and isinstance(ema, int):
        for delta in [-10, 10, 25]:
            new_val = max(5, ema + delta)
            new_params = best_params.copy()
            new_params['trend_filter_ema'] = new_val
            perturbations.append(new_params)

    return perturbations


def build_fine_tuning_grid(best_params: Dict) -> Dict[str, List]:
    """
    Legacy function - returns a minimal grid for compatibility.
    For actual fine-tuning, use build_fine_tuning_perturbations().
    """
    # Return just the best params to avoid cartesian explosion
    return {k: [v] for k, v in best_params.items()}


# =============================================================================
# DISK STREAMING - RAM FIX
# =============================================================================

def create_fold_caches(
    coin: str,
    tick_streamer: Callable,
    num_folds: int = NUM_FOLDS
) -> List[Dict[str, str]]:
    """
    Create binary cache files for each fold's train/val data.
    Streams directly from source to disk - never loads all to memory.

    Returns list of {train_cache: path, val_cache: path} for each fold.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check if caches already exist (skip re-creation to save time and memory)
    existing_caches = []
    for fold_num in range(num_folds):
        train_cache = os.path.join(CACHE_DIR, f"{coin}_fold{fold_num}_train.bin")
        val_cache = os.path.join(CACHE_DIR, f"{coin}_fold{fold_num}_val.bin")

        if os.path.exists(train_cache) and os.path.exists(val_cache):
            train_size = os.path.getsize(train_cache) // 16
            val_size = os.path.getsize(val_cache) // 16
            if train_size > 0 and val_size > 0:
                existing_caches.append({
                    'fold_num': fold_num,
                    'train_cache': train_cache,
                    'val_cache': val_cache,
                    'train_size': train_size,
                    'val_size': val_size,
                })

    if len(existing_caches) == num_folds:
        total_train = sum(c['train_size'] for c in existing_caches)
        total_val = sum(c['val_size'] for c in existing_caches)
        print(f"  Using existing fold caches ({total_train:,} train, {total_val:,} val ticks)")
        return existing_caches

    print("  Creating new fold caches...")

    # First, count total ticks (streaming through)
    print("  Counting total ticks...")
    total_ticks = 0
    for _ in tick_streamer():
        total_ticks += 1
    print(f"  Total ticks: {total_ticks:,}")

    if total_ticks < 10000:
        raise ValueError(f"Insufficient data: {total_ticks} ticks")

    # Calculate fold boundaries
    # CRITICAL FIX: Each fold must have UNIQUE validation data!
    # Previous bug: Folds 1 & 2 both validated on 80-100%, same as holdout.
    #
    # NEW: Expanding window on 0-80%, holdout (80-100%) is NEVER touched during CV.
    #   Fold 0: Train 0-40%, Val 40-53% (13% val)
    #   Fold 1: Train 0-53%, Val 53-67% (14% val)
    #   Fold 2: Train 0-67%, Val 67-80% (13% val)
    #   Holdout: 80-100% (tested only AFTER all CV completes)
    #
    # This ensures:
    # 1. Each fold validates on different, non-overlapping data
    # 2. Holdout is truly unseen during optimization
    # 3. Training always uses past data (no look-ahead)
    folds = []

    # Validation boundaries (each fold gets ~13-14% of data)
    val_boundaries = [
        (0.40, 0.53),  # Fold 0: validate on 40-53%
        (0.53, 0.67),  # Fold 1: validate on 53-67%
        (0.67, 0.80),  # Fold 2: validate on 67-80%
    ]

    for fold_num in range(num_folds):
        val_start_pct, val_end_pct = val_boundaries[fold_num]
        train_end_pct = val_start_pct  # Training ends where validation starts

        folds.append({
            'fold_num': fold_num,
            'train_start': 0,
            'train_end': int(total_ticks * train_end_pct),
            'val_start': int(total_ticks * val_start_pct),
            'val_end': int(total_ticks * val_end_pct),
        })

    # Stream ticks to fold cache files
    cache_files = []

    for fold in folds:
        fold_num = fold['fold_num']
        train_cache = os.path.join(CACHE_DIR, f"{coin}_fold{fold_num}_train.bin")
        val_cache = os.path.join(CACHE_DIR, f"{coin}_fold{fold_num}_val.bin")

        print(f"  Creating fold {fold_num} caches...")

        with open(train_cache, 'wb') as train_f, open(val_cache, 'wb') as val_f:
            for i, (ts, price) in enumerate(tick_streamer()):
                # Pack as: timestamp (double) + price (double) = 16 bytes
                packed = struct.pack('dd', ts.timestamp(), price)

                if fold['train_start'] <= i < fold['train_end']:
                    train_f.write(packed)
                if fold['val_start'] <= i < fold['val_end']:
                    val_f.write(packed)

        train_size = os.path.getsize(train_cache) // 16
        val_size = os.path.getsize(val_cache) // 16
        print(f"    Train: {train_size:,} ticks, Val: {val_size:,} ticks")

        cache_files.append({
            'fold_num': fold_num,
            'train_cache': train_cache,
            'val_cache': val_cache,
            'train_size': train_size,
            'val_size': val_size,
        })

    return cache_files


def create_disk_streamer(cache_file: str) -> Generator[Tuple[datetime, float], None, None]:
    """
    Yield ticks from binary cache file without loading all to memory.
    Each tick is 16 bytes: 8-byte timestamp (double) + 8-byte price (double).
    """
    with open(cache_file, 'rb') as f:
        while True:
            data = f.read(16)
            if not data or len(data) < 16:
                break
            ts_float, price = struct.unpack('dd', data)
            yield (datetime.fromtimestamp(ts_float), price)


# =============================================================================
# PHASE 2.2: TRUE HOLDOUT TEST SET
# =============================================================================

def create_holdout_cache(
    coin: str,
    tick_streamer: Callable,
    holdout_pct: float = 0.20
) -> Tuple[str, int, int]:
    """
    Create a TRUE holdout cache from the most recent data.

    CRITICAL: This data should NEVER be seen during optimization.
    Only evaluate on holdout ONCE at the very end.

    Args:
        coin: Coin symbol
        tick_streamer: Function that yields (timestamp, price) tuples
        holdout_pct: Percentage of data to reserve (default 20%)

    Returns:
        Tuple of (holdout_cache_path, holdout_ticks, non_holdout_ticks)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    holdout_cache = os.path.join(CACHE_DIR, f"{coin}_holdout.bin")
    non_holdout_cache = os.path.join(CACHE_DIR, f"{coin}_non_holdout.bin")

    # Check if already exists
    if os.path.exists(holdout_cache) and os.path.exists(non_holdout_cache):
        holdout_ticks = os.path.getsize(holdout_cache) // 16
        non_holdout_ticks = os.path.getsize(non_holdout_cache) // 16
        if holdout_ticks > 0 and non_holdout_ticks > 0:
            print(f"  Using existing holdout cache ({holdout_ticks:,} holdout, {non_holdout_ticks:,} train/val)")
            return holdout_cache, holdout_ticks, non_holdout_ticks

    print("  Creating holdout cache...")

    # First count total ticks
    total_ticks = 0
    for _ in tick_streamer():
        total_ticks += 1
    print(f"  Total ticks: {total_ticks:,}")

    # Calculate holdout boundary
    holdout_start = int(total_ticks * (1 - holdout_pct))
    print(f"  Holdout: last {holdout_pct*100:.0f}% = ticks {holdout_start:,} to {total_ticks:,}")

    # Stream to cache files
    with open(holdout_cache, 'wb') as holdout_f, open(non_holdout_cache, 'wb') as non_holdout_f:
        for i, (ts, price) in enumerate(tick_streamer()):
            packed = struct.pack('dd', ts.timestamp(), price)
            if i >= holdout_start:
                holdout_f.write(packed)
            else:
                non_holdout_f.write(packed)

    holdout_ticks = os.path.getsize(holdout_cache) // 16
    non_holdout_ticks = os.path.getsize(non_holdout_cache) // 16
    print(f"  Created: {holdout_ticks:,} holdout, {non_holdout_ticks:,} train/val")

    return holdout_cache, holdout_ticks, non_holdout_ticks


def evaluate_on_holdout(
    holdout_cache: str,
    best_params: Dict,
    verbose: bool = True,
    funding_rates: Optional[Dict] = None  # PHASE 5: Funding rate lookup
) -> Dict:
    """
    Evaluate best parameters on TRUE holdout data.

    IMPORTANT: Call this ONLY ONCE at the very end of optimization.
    Running multiple times defeats the purpose of a holdout set.

    Args:
        holdout_cache: Path to holdout cache file
        best_params: Best parameters from optimization
        verbose: Print detailed results
        funding_rates: PHASE 5 - Dict mapping timestamps to funding rates

    Returns:
        Backtest result dictionary
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HOLDOUT EVALUATION (TRUE OUT-OF-SAMPLE)")
        print("=" * 70)
        print("WARNING: This should only be run ONCE at the very end!")
        print("-" * 70)

    # Create streamer for holdout data
    streamer = create_disk_streamer(holdout_cache)

    # Run backtest with best params
    result = run_pyramid_backtest(
        prices=streamer,
        threshold_pct=best_params.get('threshold', 2.0),
        trailing_pct=best_params.get('trailing', 1.0),
        pyramid_step_pct=best_params.get('pyramid_step', 1.0),
        max_pyramids=best_params.get('max_pyramids', 20),
        pyramid_size_schedule=best_params.get('size_schedule', 'fixed'),
        pyramid_acceleration=best_params.get('acceleration', 1.0),
        min_pyramid_spacing_pct=best_params.get('min_spacing', 0.0),
        time_decay_exit_seconds=best_params.get('time_decay'),
        volatility_filter_type=best_params.get('vol_type', 'none'),
        volatility_min_pct=best_params.get('vol_min', 0.0),
        volatility_window_size=best_params.get('vol_window', 100),
        use_causal_trailing=best_params.get('use_causal', True),
        confirmation_ticks=best_params.get('confirmation_ticks', 3),
        execution_model=DEFAULT_EXECUTION_MODEL,
        # PHASE 5: Realistic simulation (margin/leverage/funding)
        initial_capital=INITIAL_CAPITAL,
        leverage=LEVERAGE,
        position_size_usdt=POSITION_SIZE_USDT,
        use_margin_tracking=USE_MARGIN_TRACKING,
        funding_rates=funding_rates,
        apply_funding=APPLY_FUNDING_RATES if funding_rates else False,
        return_rounds=False,
        # PHASE 6: New exit controls
        take_profit_pct=best_params.get('take_profit_pct'),
        stop_loss_pct=best_params.get('stop_loss_pct'),
        breakeven_after_pct=best_params.get('breakeven_after_pct'),
        # PHASE 6: New timing controls
        pyramid_cooldown_sec=best_params.get('pyramid_cooldown_sec', 0),
        max_round_duration_hr=best_params.get('max_round_duration_hr'),
        # PHASE 6: New filters
        trend_filter_ema=best_params.get('trend_filter_ema'),
        session_filter=best_params.get('session_filter', 'all'),
    )

    if verbose:
        print(f"\nHOLDOUT RESULTS:")
        print(f"  Total P&L: {result['total_pnl']:+.2f}%")
        print(f"  Rounds: {result['total_rounds']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Avg P&L/Round: {result['avg_pnl']:+.2f}%")
        print(f"  Avg Pyramids: {result['avg_pyramids']:.1f}")

        # Compare to optimization results
        print("\n" + "-" * 70)
        print("INTERPRETATION:")
        if result['total_pnl'] > 0:
            print("  [POSITIVE] Strategy shows profit on unseen data")
        else:
            print("  [NEGATIVE] Strategy shows loss on unseen data - CAUTION!")

        if result['win_rate'] >= 50:
            print(f"  [GOOD] Win rate {result['win_rate']:.1f}% >= 50%")
        else:
            print(f"  [CAUTION] Win rate {result['win_rate']:.1f}% < 50%")

        print("=" * 70)

    return result


def count_cache_ticks(cache_file: str) -> int:
    """Count ticks in a cache file without loading to memory."""
    return os.path.getsize(cache_file) // 16


def validate_regime_robustness(
    holdout_cache: str,
    best_params: Dict,
    funding_rates: Optional[Dict] = None,
    min_profitable_regimes_pct: float = 70.0,
    verbose: bool = True
) -> Tuple[bool, Optional[CrossRegimeValidation]]:
    """
    Validate that the strategy is profitable across different market regimes.

    This ensures the strategy isn't just optimized for one specific market
    condition (e.g., only works in bull markets).

    Args:
        holdout_cache: Path to holdout cache file
        best_params: Best parameters from optimization
        funding_rates: Optional funding rate lookup
        min_profitable_regimes_pct: Require profit in this % of regimes
        verbose: Print detailed results

    Returns:
        Tuple of (passes: bool, validation_result: CrossRegimeValidation or None)
    """
    if verbose:
        print("\n" + "-" * 70)
        print("REGIME VALIDATION: Checking profitability across market conditions...")

    # 1. Load holdout prices for regime detection
    prices = []
    streamer = create_disk_streamer(holdout_cache)
    for ts, price in streamer:
        prices.append((ts, price))

    if len(prices) < 1000:
        if verbose:
            print("  [SKIP] Insufficient data for regime analysis (<1000 ticks)")
        return (True, None)  # Pass by default if insufficient data

    # 2. Detect regimes in the holdout data
    regime_labels = detect_regimes(prices, window_size=500)

    if not regime_labels:
        if verbose:
            print("  [SKIP] Could not detect regimes in holdout data")
        return (True, None)

    # 3. Re-run backtest WITH rounds to get per-round timing
    streamer2 = create_disk_streamer(holdout_cache)
    result_with_rounds = run_pyramid_backtest(
        prices=streamer2,
        threshold_pct=best_params.get('threshold', 2.0),
        trailing_pct=best_params.get('trailing', 1.0),
        pyramid_step_pct=best_params.get('pyramid_step', 1.0),
        max_pyramids=best_params.get('max_pyramids', 20),
        pyramid_size_schedule=best_params.get('size_schedule', 'fixed'),
        pyramid_acceleration=best_params.get('acceleration', 1.0),
        min_pyramid_spacing_pct=best_params.get('min_spacing', 0.0),
        time_decay_exit_seconds=best_params.get('time_decay'),
        volatility_filter_type=best_params.get('vol_type', 'none'),
        volatility_min_pct=best_params.get('vol_min', 0.0),
        volatility_window_size=best_params.get('vol_window', 100),
        use_causal_trailing=best_params.get('use_causal', True),
        confirmation_ticks=best_params.get('confirmation_ticks', 3),
        execution_model=DEFAULT_EXECUTION_MODEL,
        initial_capital=INITIAL_CAPITAL,
        leverage=LEVERAGE,
        position_size_usdt=POSITION_SIZE_USDT,
        use_margin_tracking=USE_MARGIN_TRACKING,
        funding_rates=funding_rates,
        apply_funding=APPLY_FUNDING_RATES if funding_rates else False,
        return_rounds=True,  # Need rounds for regime mapping
        # PHASE 6: New exit controls
        take_profit_pct=best_params.get('take_profit_pct'),
        stop_loss_pct=best_params.get('stop_loss_pct'),
        breakeven_after_pct=best_params.get('breakeven_after_pct'),
        # PHASE 6: New timing controls
        pyramid_cooldown_sec=best_params.get('pyramid_cooldown_sec', 0),
        max_round_duration_hr=best_params.get('max_round_duration_hr'),
        # PHASE 6: New filters
        trend_filter_ema=best_params.get('trend_filter_ema'),
        session_filter=best_params.get('session_filter', 'all'),
    )

    rounds = result_with_rounds.get('rounds', [])
    if not rounds:
        if verbose:
            print("  [SKIP] No rounds returned from backtest")
        return (True, None)

    # 4. Create a time -> regime lookup
    time_to_regime = {}
    for label in regime_labels:
        time_to_regime[label.start_time] = label.combined

    # 5. Map rounds to regimes and group by regime
    regime_returns: Dict[MarketRegime, List[float]] = {}

    for round_obj in rounds:
        # Find the regime at round entry time (closest match)
        round_time = round_obj.entry_time
        closest_regime = None
        min_diff = float('inf')

        for label_time, regime in time_to_regime.items():
            diff = abs((round_time - label_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_regime = regime

        if closest_regime is not None:
            if closest_regime not in regime_returns:
                regime_returns[closest_regime] = []
            regime_returns[closest_regime].append(round_obj.total_pnl)

    if not regime_returns:
        if verbose:
            print("  [SKIP] Could not map rounds to regimes")
        return (True, None)

    # 6. Run cross-regime validation
    validation = validate_strategy_across_regimes(
        regime_returns=regime_returns,
        min_rounds_per_regime=5,  # Lower threshold for holdout (less data)
        min_profitable_regimes_pct=min_profitable_regimes_pct,
        verbose=verbose
    )

    return (validation.overall_pass, validation)


# =============================================================================
# GRID SEARCH WITH STREAMING
# =============================================================================

def calculate_grid_size(param_grid: Dict[str, List]) -> int:
    """Calculate total combinations in a parameter grid."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


# =============================================================================
# PHASE 6: RISK-ADJUSTED SCORING
# =============================================================================
# These constraints ensure optimizer finds strategies suitable for live trading

# RISK THRESHOLDS for post-hoc filtering (applied AFTER optimization):
# During optimization, we use soft penalties to allow full exploration.
# Hard thresholds are only enforced when selecting final candidates.
MIN_WIN_RATE_PCT = 45.0       # Psychological tradability threshold
MAX_DRAWDOWN_PCT = 25.0       # Account protection at 10x leverage
MIN_ROUNDS = 50               # Statistical validity threshold


def passes_risk_constraints(result: Dict) -> Tuple[bool, List[str]]:
    """
    POST-HOC filter: Check if a result passes hard risk constraints.

    This should be called AFTER optimization to filter final candidates.
    During optimization, we use soft penalties (see calculate_risk_adjusted_score).

    Returns:
        Tuple of (passes: bool, rejection_reasons: List[str])
    """
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
    if max_dd > MAX_DRAWDOWN_PCT:
        reasons.append(f"Drawdown too high: {max_dd:.1f}% > {MAX_DRAWDOWN_PCT}%")

    return (len(reasons) == 0, reasons)


def calculate_risk_adjusted_score(result: Dict) -> float:
    """
    Calculate risk-adjusted score for ranking parameter combinations.

    IMPORTANT: Uses SOFT PENALTIES, not hard disqualification.
    This allows the optimizer to explore all parameter regions without bias.
    Hard thresholds are applied post-hoc via passes_risk_constraints().

    Uses a modified Calmar-like ratio that considers:
    - Total P&L (return)
    - Max Drawdown (risk)
    - Win Rate (consistency)
    - Number of rounds (statistical validity)

    Higher score = better risk-adjusted performance.
    """
    total_pnl = result.get('total_pnl', 0)
    max_dd = result.get('max_drawdown_pct', 50)  # Default high if missing
    win_rate = result.get('win_rate', 50) / 100  # Convert to decimal
    rounds = result.get('rounds', result.get('total_rounds', 0))

    # SOFT PENALTIES instead of hard disqualification
    # This allows ranking even "bad" results to avoid search bias

    # Negative or zero P&L gets a very low (but not disqualifying) score
    if total_pnl <= 0:
        # Return a small negative score based on how bad it is
        return total_pnl - 1000  # e.g., -20% P&L -> -1020 score

    # Insufficient rounds: heavy penalty but still rankable
    if rounds < MIN_ROUNDS:
        round_penalty = (MIN_ROUNDS - rounds) / MIN_ROUNDS  # 0-1 scale
        total_pnl *= (1 - round_penalty * 0.8)  # Up to 80% penalty

    # Ensure drawdown is positive for calculation (avoid division issues)
    max_dd = max(max_dd, 1.0)

    # Risk-adjusted score components:
    # 1. Calmar ratio: Return / Max Drawdown (core risk-adjusted metric)
    calmar = total_pnl / max_dd

    # 2. Win rate component - continuous penalty for low win rates
    # Below threshold: apply penalty proportional to shortfall
    if win_rate < (MIN_WIN_RATE_PCT / 100):
        shortfall = (MIN_WIN_RATE_PCT / 100) - win_rate
        win_penalty = shortfall * 5  # 5x penalty for each % below threshold
        calmar *= max(0.1, 1 - win_penalty)  # At most 90% reduction

    # Win rate bonus for high win rates
    win_bonus = win_rate ** 2

    # 3. Drawdown penalty - continuous for high drawdowns
    if max_dd > MAX_DRAWDOWN_PCT:
        excess = (max_dd - MAX_DRAWDOWN_PCT) / MAX_DRAWDOWN_PCT
        dd_penalty = excess * 2  # 2x penalty for excess drawdown
        calmar *= max(0.1, 1 - dd_penalty)  # At most 90% reduction

    # 4. Round count bonus (prefer strategies that trade more, up to a point)
    # Max bonus at 100 rounds - more trades = more statistical confidence
    round_bonus = min(rounds / 100, 1.0)

    # Combined score: Calmar * (1 + win_bonus) * (1 + round_bonus * 0.1)
    # The 0.1 factor keeps round bonus from dominating
    score = calmar * (1 + win_bonus) * (1 + round_bonus * 0.1)

    return score


def run_streaming_grid_search(
    coin: str,
    cache_file: str,
    param_grid: Dict,
    phase_name: str,
    base_params: Optional[Dict] = None,
    funding_rates: Optional[Dict] = None  # PHASE 5: Funding rate lookup
) -> List[Dict]:
    """
    Run grid search using disk streaming - never loads all data to memory.

    Args:
        coin: Trading pair symbol
        cache_file: Path to binary tick cache
        param_grid: Parameters to grid search
        phase_name: Name for logging (e.g., "Phase_A", "Phase_B_combo1")
        base_params: Fixed params to merge with grid params (for Phase B)
        funding_rates: PHASE 5 - Dict mapping timestamps to funding rates

    Returns:
        Top results sorted by P&L descending
    """
    total = calculate_grid_size(param_grid)
    print(f"\n  {phase_name}: Testing {total:,} combinations...")

    top_results = []
    max_top = 100  # Keep top 100 for diversity

    log_file = os.path.join(LOG_DIR, f"{coin}_{phase_name}_grid.csv")

    fieldnames = [
        'threshold', 'trailing', 'pyramid_step', 'max_pyramids', 'poll_interval',
        'size_schedule', 'acceleration', 'min_spacing', 'time_decay',
        'vol_type', 'vol_min', 'vol_window',
        'total_pnl', 'rounds', 'avg_pnl', 'win_rate', 'avg_pyramids',
        'max_drawdown_pct'  # PHASE 6: Track risk metric for sorting
    ]

    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        completed = 0
        start = time.time()

        # Get all param names and their values
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        # Generate all combinations using nested iteration
        # This avoids creating huge list of combos in memory
        def iterate_combos(values_list, current_combo=[]):
            if not values_list:
                yield dict(zip(param_names, current_combo))
            else:
                for val in values_list[0]:
                    yield from iterate_combos(values_list[1:], current_combo + [val])

        for params in iterate_combos(param_values):
            completed += 1

            # Merge with base params if provided
            if base_params:
                full_params = {**base_params, **params}
            else:
                full_params = params

            # Progress update
            if completed % 500 == 0 or completed == 1:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 1
                remaining = (total - completed) / rate
                pct = (completed / total) * 100
                print(f"\r    [{completed:,}/{total:,}] {pct:.1f}% | "
                      f"{remaining/3600:.1f}h remaining    ",
                      end="", flush=True)

            try:
                # Get poll_interval value
                poll = full_params.get('poll_interval', 0)
                if isinstance(poll, str):
                    poll = 0 if poll == 'tick' else float(poll.replace('s', ''))

                # Stream from disk and aggregate
                # NOTE: aggregate_ticks_to_interval expects a CALLABLE, not a generator
                tick_streamer = lambda cf=cache_file: create_disk_streamer(cf)
                aggregated = aggregate_ticks_to_interval(tick_streamer, poll)

                # Get time_decay value
                time_decay = full_params.get('time_decay')
                if isinstance(time_decay, str):
                    time_decay = None if time_decay == 'None' else int(time_decay.replace('s', ''))

                # Get max_pyramids value
                max_pyr = full_params.get('max_pyramids', 80)
                if max_pyr == 'unlimited':
                    max_pyr = 9999

                # Run backtest with streaming data (return_rounds=False saves memory in grid search)
                # Uses Phase 1 fix (causal trailing) and Phase 2 fix (execution model)
                result = run_pyramid_backtest(
                    aggregated,
                    threshold_pct=full_params['threshold'],
                    trailing_pct=full_params['trailing'],
                    pyramid_step_pct=full_params['pyramid_step'],
                    max_pyramids=max_pyr,
                    verbose=False,
                    pyramid_size_schedule=full_params.get('size_schedule', 'fixed'),
                    min_pyramid_spacing_pct=full_params.get('min_spacing', 0.0),
                    pyramid_acceleration=full_params.get('acceleration', 1.0),
                    time_decay_exit_seconds=time_decay,
                    volatility_filter_type=full_params.get('vol_type', 'none'),
                    volatility_min_pct=full_params.get('vol_min', 0.0),
                    volatility_window_size=full_params.get('vol_window', 100),
                    # PHASE 1 FIX: Causal trailing stop (no look-ahead bias)
                    use_causal_trailing=full_params.get('use_causal', True),
                    confirmation_ticks=full_params.get('confirmation_ticks', 3),
                    # PHASE 2 FIX: Realistic execution costs
                    execution_model=DEFAULT_EXECUTION_MODEL,
                    # PHASE 5: Realistic simulation (margin/leverage/funding)
                    initial_capital=INITIAL_CAPITAL,
                    leverage=LEVERAGE,
                    position_size_usdt=POSITION_SIZE_USDT,
                    use_margin_tracking=USE_MARGIN_TRACKING,
                    funding_rates=funding_rates,
                    apply_funding=APPLY_FUNDING_RATES if funding_rates else False,
                    return_rounds=False,  # MEMORY OPTIMIZATION: Skip round accumulation
                    # PHASE 6: New exit controls
                    take_profit_pct=full_params.get('take_profit_pct'),
                    stop_loss_pct=full_params.get('stop_loss_pct'),
                    breakeven_after_pct=full_params.get('breakeven_after_pct'),
                    # PHASE 6: New timing controls
                    pyramid_cooldown_sec=full_params.get('pyramid_cooldown_sec', 0),
                    max_round_duration_hr=full_params.get('max_round_duration_hr'),
                    # PHASE 6: New filters
                    trend_filter_ema=full_params.get('trend_filter_ema'),
                    session_filter=full_params.get('session_filter', 'all'),
                )

                # Format for storage
                poll_label = "tick" if poll == 0 else f"{poll}s"
                time_decay_label = 'None' if time_decay is None else f'{time_decay}s'

                entry = {
                    'threshold': full_params['threshold'],
                    'trailing': full_params['trailing'],
                    'pyramid_step': full_params['pyramid_step'],
                    'max_pyramids': max_pyr if max_pyr < 9999 else 'unlimited',
                    'poll_interval': poll_label,
                    'size_schedule': full_params.get('size_schedule', 'fixed'),
                    'acceleration': full_params.get('acceleration', 1.0),
                    'min_spacing': full_params.get('min_spacing', 0.0),
                    'time_decay': time_decay_label,
                    'vol_type': full_params.get('vol_type', 'none'),
                    'vol_min': full_params.get('vol_min', 0.0),
                    'vol_window': full_params.get('vol_window', 100),
                    'confirmation_ticks': full_params.get('confirmation_ticks', 3),
                    # PHASE 6: New parameters
                    'take_profit_pct': full_params.get('take_profit_pct'),
                    'stop_loss_pct': full_params.get('stop_loss_pct'),
                    'breakeven_after_pct': full_params.get('breakeven_after_pct'),
                    'pyramid_cooldown_sec': full_params.get('pyramid_cooldown_sec', 0),
                    'max_round_duration_hr': full_params.get('max_round_duration_hr'),
                    'trend_filter_ema': full_params.get('trend_filter_ema'),
                    'session_filter': full_params.get('session_filter', 'all'),
                    # Results
                    'total_pnl': result['total_pnl'],
                    'rounds': result['total_rounds'],
                    'avg_pnl': result['avg_pnl'],
                    'win_rate': result['win_rate'],
                    'avg_pyramids': result['avg_pyramids'],
                    'max_drawdown_pct': result.get('max_drawdown_pct', 0)  # Risk metric
                }

                # Write to disk immediately
                writer.writerow(entry)
                f.flush()

                # Track top N (using risk-adjusted score, not just total_pnl)
                entry_score = calculate_risk_adjusted_score(entry)
                if len(top_results) < max_top:
                    top_results.append(entry)
                    top_results.sort(key=calculate_risk_adjusted_score, reverse=True)
                elif entry_score > calculate_risk_adjusted_score(top_results[-1]):
                    top_results[-1] = entry
                    top_results.sort(key=calculate_risk_adjusted_score, reverse=True)

            except Exception as e:
                if completed % 5000 == 0:
                    print(f"\n    Error at combo {completed}: {e}")

            # Periodic garbage collection (every 1000 for memory safety with 6 concurrent instances)
            if completed % 1000 == 0:
                gc.collect()
            # Check memory usage periodically
            if completed % 5000 == 0:
                check_memory_usage(f"grid_search:{phase_name}:{completed}")

    print()  # Newline after progress
    return top_results


# =============================================================================
# SINGLE BACKTEST FOR VALIDATION
# =============================================================================

def run_single_backtest_streaming(
    cache_file: str,
    params: Dict,
    funding_rates: Optional[Dict] = None  # PHASE 5: Funding rate lookup
) -> Dict:
    """
    Run a single backtest with streaming from disk cache.

    Returns full result dict including total_pnl.
    """
    # Parse poll_interval
    poll = params.get('poll_interval', 0)
    if isinstance(poll, str):
        poll = 0 if poll == 'tick' else float(poll.replace('s', ''))

    # Parse time_decay
    time_decay = params.get('time_decay')
    if isinstance(time_decay, str):
        time_decay = None if time_decay == 'None' else int(time_decay.replace('s', ''))

    # Parse max_pyramids
    max_pyr = params.get('max_pyramids', 80)
    if max_pyr == 'unlimited':
        max_pyr = 9999

    # Stream from disk
    # NOTE: aggregate_ticks_to_interval expects a CALLABLE, not a generator
    tick_streamer = lambda cf=cache_file: create_disk_streamer(cf)
    aggregated = aggregate_ticks_to_interval(tick_streamer, poll)

    result = run_pyramid_backtest(
        aggregated,
        threshold_pct=params['threshold'],
        trailing_pct=params['trailing'],
        pyramid_step_pct=params['pyramid_step'],
        max_pyramids=max_pyr,
        verbose=False,
        pyramid_size_schedule=params.get('size_schedule', 'fixed'),
        min_pyramid_spacing_pct=params.get('min_spacing', 0.0),
        pyramid_acceleration=params.get('acceleration', 1.0),
        time_decay_exit_seconds=time_decay,
        volatility_filter_type=params.get('vol_type', 'none'),
        volatility_min_pct=params.get('vol_min', 0.0),
        volatility_window_size=params.get('vol_window', 100),
        # PHASE 1 FIX: Causal trailing stop
        use_causal_trailing=params.get('use_causal', True),
        confirmation_ticks=params.get('confirmation_ticks', 3),
        # PHASE 2 FIX: Realistic execution costs
        execution_model=DEFAULT_EXECUTION_MODEL,
        # PHASE 5: Realistic simulation (margin/leverage/funding)
        initial_capital=INITIAL_CAPITAL,
        leverage=LEVERAGE,
        position_size_usdt=POSITION_SIZE_USDT,
        use_margin_tracking=USE_MARGIN_TRACKING,
        funding_rates=funding_rates,
        apply_funding=APPLY_FUNDING_RATES if funding_rates else False,
        return_rounds=False,  # MEMORY OPTIMIZATION: Skip round accumulation
        # PHASE 6: New exit controls
        take_profit_pct=params.get('take_profit_pct'),
        stop_loss_pct=params.get('stop_loss_pct'),
        breakeven_after_pct=params.get('breakeven_after_pct'),
        # PHASE 6: New timing controls
        pyramid_cooldown_sec=params.get('pyramid_cooldown_sec', 0),
        max_round_duration_hr=params.get('max_round_duration_hr'),
        # PHASE 6: New filters
        trend_filter_ema=params.get('trend_filter_ema'),
        session_filter=params.get('session_filter', 'all'),
    )

    return result


# =============================================================================
# ROBUSTNESS TESTING
# =============================================================================

def calculate_robustness_score(
    cache_file: str,
    best_params: Dict,
    funding_rates: Optional[Dict] = None  # PHASE 5: Funding rate lookup
) -> Tuple[float, List[Dict]]:
    """
    Test nearby parameters to check if the optimum is robust.

    Returns (robustness_score, perturbation_results).
    Score of 1.0 = all perturbations give similar P&L
    Score of 0.5 = perturbations give only 50% of best P&L
    """
    center_result = run_single_backtest_streaming(cache_file, best_params, funding_rates)
    center_pnl = center_result['total_pnl']

    if center_pnl <= 0:
        return 0.0, []

    perturbation_results = []

    # Test small changes to key parameters
    perturbations = [
        ('threshold', -0.5),
        ('threshold', +0.5),
        ('trailing', -0.2),
        ('trailing', +0.2),
        ('pyramid_step', -0.2),
        ('pyramid_step', +0.2),
    ]

    for param, delta in perturbations:
        perturbed = best_params.copy()
        current_val = perturbed.get(param, 1.0)

        if isinstance(current_val, (int, float)):
            perturbed[param] = max(0.1, current_val + delta)

            result = run_single_backtest_streaming(cache_file, perturbed, funding_rates)
            perturbation_results.append({
                'param': param,
                'delta': delta,
                'pnl': result['total_pnl'],
                'ratio': result['total_pnl'] / center_pnl if center_pnl > 0 else 0
            })

    # Robustness = minimum ratio among perturbations
    if perturbation_results:
        min_ratio = min(p['ratio'] for p in perturbation_results)
        robustness = max(0, min_ratio)
    else:
        robustness = 1.0

    return robustness, perturbation_results


# =============================================================================
# HIERARCHICAL OPTIMIZATION
# =============================================================================

def run_hierarchical_optimization(
    coin: str,
    train_cache: str,
    fold_num: int,
    funding_rates: Optional[Dict] = None  # PHASE 5: Funding rate lookup
) -> List[Dict]:
    """
    Run 3-phase hierarchical optimization on one fold.

    Phase A: Dense core param search (5 params, ~218K combos)
    Phase B: For top 20, search new params (7 params, ~882K each)
    Phase C: Fine-tune top 10 overall

    Returns top 10 parameter sets with their P&L.
    """
    print(f"\n{'=' * 70}")
    print(f"FOLD {fold_num} - HIERARCHICAL OPTIMIZATION")
    print(f"{'=' * 70}")

    # =========================================================================
    # PHASE A: Core Parameter Optimization
    # =========================================================================
    print(f"\n--- PHASE A: Core Parameters (Dense Grid) ---")

    core_grid = build_dense_core_grid()
    core_size = calculate_grid_size(core_grid)
    print(f"  Grid size: {core_size:,} combinations")

    phase_a_results = run_streaming_grid_search(
        coin=coin,
        cache_file=train_cache,
        param_grid=core_grid,
        phase_name=f"fold{fold_num}_phaseA",
        funding_rates=funding_rates  # PHASE 5
    )

    if not phase_a_results:
        print("  ERROR: No results from Phase A!")
        return []

    # Take top N for Phase B
    top_core = phase_a_results[:TOP_N_CORE]
    print(f"\n  Phase A complete. Top {len(top_core)} core combos selected.")
    for i, r in enumerate(top_core[:5]):
        print(f"    #{i+1}: T={r['threshold']}% Tr={r['trailing']}% -> {r['total_pnl']:+.2f}%")

    # =========================================================================
    # PHASE B: New Parameter Optimization
    # =========================================================================
    print(f"\n--- PHASE B: New Parameters for Top {TOP_N_CORE} Core Combos ---")

    new_params_grid = build_dense_new_params_grid()
    new_size = calculate_grid_size(new_params_grid)
    print(f"  New params grid: {new_size:,} combos per core combo")
    print(f"  Total Phase B: {new_size * len(top_core):,} combinations")

    all_phase_b_results = []

    for i, core_params in enumerate(top_core):
        # Extract just the core param values
        base_params = {
            'threshold': core_params['threshold'],
            'trailing': core_params['trailing'],
            'pyramid_step': core_params['pyramid_step'],
            'max_pyramids': core_params['max_pyramids'],
            'poll_interval': core_params['poll_interval'],
        }

        phase_b_results = run_streaming_grid_search(
            coin=coin,
            cache_file=train_cache,
            param_grid=new_params_grid,
            phase_name=f"fold{fold_num}_phaseB_core{i}",
            base_params=base_params,
            funding_rates=funding_rates  # PHASE 5
        )

        if phase_b_results:
            all_phase_b_results.extend(phase_b_results[:10])  # Keep top 10 from each

        gc.collect()

    # Sort all Phase B results and take top N
    all_phase_b_results.sort(key=calculate_risk_adjusted_score, reverse=True)
    top_overall = all_phase_b_results[:TOP_N_FINAL]

    print(f"\n  Phase B complete. Top {len(top_overall)} overall combos:")
    for i, r in enumerate(top_overall[:5]):
        print(f"    #{i+1}: T={r['threshold']}% Tr={r['trailing']}% "
              f"sched={r['size_schedule']} -> {r['total_pnl']:+.2f}%")

    # =========================================================================
    # PHASE C: Fine-Tuning (Perturbation-based, ~100 tests per combo)
    # =========================================================================
    print(f"\n--- PHASE C: Fine-Tuning Top {TOP_N_FINAL} Combos ---")

    final_results = []

    for i, base_params in enumerate(top_overall):
        print(f"\n  Fine-tuning combo {i+1}/{len(top_overall)}...")

        # Generate perturbations (one param at a time)
        perturbations = build_fine_tuning_perturbations(base_params)
        print(f"    Testing {len(perturbations)} perturbations...")

        best_pnl = base_params['total_pnl']
        best_result = base_params

        for j, perturbed_params in enumerate(perturbations):
            if (j + 1) % 20 == 0:
                print(f"\r    [{j+1}/{len(perturbations)}]", end="", flush=True)

            result = run_single_backtest_streaming(train_cache, perturbed_params, funding_rates)

            if result['total_pnl'] > best_pnl:
                best_pnl = result['total_pnl']
                # Copy perturbed params and add result fields
                best_result = perturbed_params.copy()
                best_result['total_pnl'] = result['total_pnl']
                best_result['rounds'] = result['total_rounds']
                best_result['avg_pnl'] = result['avg_pnl']
                best_result['win_rate'] = result['win_rate']
                best_result['avg_pyramids'] = result['avg_pyramids']

        print(f"\r    Best P&L: {best_pnl:+.2f}% (original: {base_params['total_pnl']:+.2f}%)")
        final_results.append(best_result)

        gc.collect()

    # Sort final results
    final_results.sort(key=calculate_risk_adjusted_score, reverse=True)

    print(f"\n  Phase C complete. Final top 5:")
    for i, r in enumerate(final_results[:5]):
        print(f"    #{i+1}: T={r['threshold']}% Tr={r['trailing']}% -> {r['total_pnl']:+.2f}%")

    return final_results[:TOP_N_FINAL]


# =============================================================================
# 3-FOLD WALK-FORWARD VALIDATION
# =============================================================================

def run_multi_fold_optimization(
    coin: str,
    cache_files: List[Dict],
    funding_rates: Optional[Dict] = None  # PHASE 5: Funding rate lookup
) -> Dict:
    """
    Run optimization on each fold and find parameters that work across ALL folds.

    Returns the parameter set with best average validation P&L across folds,
    but only if it's profitable on ALL folds.
    """
    fold_results = []

    for fold_info in cache_files:
        fold_num = fold_info['fold_num']
        train_cache = fold_info['train_cache']
        val_cache = fold_info['val_cache']

        # Run hierarchical optimization on this fold's training data
        train_results = run_hierarchical_optimization(
            coin=coin,
            train_cache=train_cache,
            fold_num=fold_num,
            funding_rates=funding_rates  # PHASE 5
        )

        if not train_results:
            print(f"  WARNING: Fold {fold_num} produced no results!")
            continue

        # Validate top 10 on this fold's validation data
        print(f"\n  Validating top {len(train_results)} on Fold {fold_num} validation data...")

        validated = []
        for params in train_results:
            val_result = run_single_backtest_streaming(val_cache, params, funding_rates)
            # Include risk metrics for risk-adjusted scoring
            validated.append({
                'params': params,
                'train_pnl': params['total_pnl'],
                'val_pnl': val_result['total_pnl'],
                'total_pnl': val_result['total_pnl'],  # Alias for risk_adjusted_score
                'max_drawdown_pct': val_result.get('max_drawdown_pct', 0),
                'win_rate': val_result.get('win_rate', 50),
                'rounds': val_result.get('total_rounds', 0),
            })

        # Sort by risk-adjusted validation score, not just val_pnl
        validated.sort(key=calculate_risk_adjusted_score, reverse=True)

        print(f"  Fold {fold_num} validation results:")
        for i, v in enumerate(validated[:5]):
            print(f"    #{i+1}: Train={v['train_pnl']:+.1f}%, Val={v['val_pnl']:+.1f}%")

        fold_results.append({
            'fold_num': fold_num,
            'validated': validated
        })

        gc.collect()

    # Find parameters that are profitable on ALL folds
    print(f"\n{'=' * 70}")
    print("CROSS-FOLD ANALYSIS")
    print(f"{'=' * 70}")

    if len(fold_results) < NUM_FOLDS:
        print("  ERROR: Not all folds completed!")
        return None

    # For each parameter set from fold 0, check if similar exists in other folds
    # and calculate average validation P&L
    cross_fold_results = []

    for result in fold_results[0]['validated']:
        params = result['params']
        total_val_pnl = result['val_pnl']
        all_profitable = result['val_pnl'] > 0
        fold_pnls = [result['val_pnl']]

        # Test this exact param set on other folds' validation data
        # Track risk metrics from all folds
        fold_max_dd = [result.get('max_drawdown_pct', 0)]
        fold_win_rates = [result.get('win_rate', 50)]
        fold_rounds = [result.get('rounds', 0)]

        for fold_info in cache_files[1:]:
            val_result = run_single_backtest_streaming(fold_info['val_cache'], params, funding_rates)
            fold_pnls.append(val_result['total_pnl'])
            fold_max_dd.append(val_result.get('max_drawdown_pct', 0))
            fold_win_rates.append(val_result.get('win_rate', 50))
            fold_rounds.append(val_result.get('total_rounds', 0))
            total_val_pnl += val_result['total_pnl']
            if val_result['total_pnl'] <= 0:
                all_profitable = False

        avg_val_pnl = total_val_pnl / NUM_FOLDS

        cross_fold_results.append({
            'params': params,
            'fold_pnls': fold_pnls,
            'avg_val_pnl': avg_val_pnl,
            'all_profitable': all_profitable,
            'min_fold_pnl': min(fold_pnls),
            # Risk metrics for risk-adjusted sorting
            'total_pnl': avg_val_pnl,  # Alias for calculate_risk_adjusted_score
            'max_drawdown_pct': max(fold_max_dd),  # Worst drawdown across folds
            'win_rate': sum(fold_win_rates) / len(fold_win_rates),  # Average win rate
            'rounds': sum(fold_rounds),  # Total rounds across folds
        })

    # Filter to only those profitable on ALL folds
    profitable_all = [r for r in cross_fold_results if r['all_profitable']]

    if profitable_all:
        # Sort by risk-adjusted score (considers P&L, drawdown, win rate)
        profitable_all.sort(key=calculate_risk_adjusted_score, reverse=True)
        winner = profitable_all[0]
        print(f"\n  Found {len(profitable_all)} param sets profitable on ALL folds!")
    else:
        # Fallback: sort by risk-adjusted score
        cross_fold_results.sort(key=calculate_risk_adjusted_score, reverse=True)
        winner = cross_fold_results[0]
        print(f"\n  WARNING: No param set profitable on ALL folds!")
        print(f"  Using best risk-adjusted: min_fold_pnl = {winner['min_fold_pnl']:+.1f}%")

    # =========================================================================
    # TOP 3 CANDIDATES VALIDATION (Phase 2 Enhancement)
    # =========================================================================
    # Require top 3 parameter sets to pass risk constraints as overfitting guard.
    # If only 1 param set passes, it's likely a statistical fluke (selection bias).
    # =========================================================================
    print("\n--- TOP 3 CANDIDATES VALIDATION (Overfitting Guard) ---")
    source_list = profitable_all if profitable_all else cross_fold_results
    top_3_candidates = source_list[:3] if len(source_list) >= 3 else source_list

    top_3_pass_count = 0
    top_3_all_pass = True
    for i, candidate in enumerate(top_3_candidates):
        passes, reasons = passes_risk_constraints(candidate)
        status = "[PASS]" if passes else "[FAIL]"
        # FIX: Access nested 'params' dict, not top-level candidate dict
        params_str = f"th={candidate['params']['threshold']:.2f}, tr={candidate['params']['trailing']:.2f}, ps={candidate['params']['pyramid_step']:.2f}"
        print(f"  Top {i+1}: {status} ({params_str})")
        if passes:
            top_3_pass_count += 1
        else:
            top_3_all_pass = False
            for reason in reasons:
                print(f"         - {reason}")

    if not top_3_all_pass:
        print(f"\n  [WARNING] Only {top_3_pass_count}/3 top candidates pass risk constraints!")
        print(f"  This suggests possible overfitting - the 'best' result may be a statistical fluke.")
        print(f"  Consider: using more data, reducing parameter grid, or requiring stricter validation.")
    else:
        print(f"\n  [OK] All top 3 candidates pass risk constraints - reduces overfitting risk.")

    return {
        'winner': winner,
        'all_results': cross_fold_results[:20],
        'num_profitable_all': len(profitable_all),
        'top_3_all_pass': top_3_all_pass,
        'top_3_pass_count': top_3_pass_count,
    }


# =============================================================================
# FAILURE HANDLING
# =============================================================================

def expand_parameter_grid(current_grid: Dict, expansion_num: int) -> Dict:
    """
    Expand parameter ranges for retry after failure.
    """
    expanded = deepcopy(current_grid)

    # Expansion factors
    factor = 1.3 ** expansion_num

    for param in ['threshold', 'trailing', 'pyramid_step']:
        if param in expanded:
            current_max = max(expanded[param])
            current_min = min(expanded[param])
            new_max = min(PARAM_LIMITS[param][1], current_max * factor)
            new_min = max(PARAM_LIMITS[param][0], current_min / factor)

            # Add new values at edges
            expanded[param] = sorted(set(
                [new_min] + expanded[param] + [new_max]
            ))

    return expanded


# =============================================================================
# INTERACTIVE COIN SELECTION
# =============================================================================

def select_coin_interactive() -> str:
    """Prompt user to select a single coin at startup."""
    print("\n" + "=" * 60)
    print("AVAILABLE COINS:")
    print("=" * 60)
    for i, coin in enumerate(DEFAULT_COINS, 1):
        print(f"  {i}. {coin}")
    print()

    while True:
        user_input = input("Enter coin symbol (e.g., BTCUSDT) or number: ").strip().upper()

        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(DEFAULT_COINS):
                return DEFAULT_COINS[idx]
            print(f"Invalid number. Enter 1-{len(DEFAULT_COINS)}")
            continue

        if user_input in DEFAULT_COINS:
            return user_input

        if user_input.endswith("USDT") and len(user_input) >= 5:
            confirm = input(f"'{user_input}' not in default list. Use anyway? [y/N]: ").strip().lower()
            if confirm == 'y':
                return user_input

        print(f"Invalid coin. Enter a symbol like BTCUSDT or a number 1-{len(DEFAULT_COINS)}")


# =============================================================================
# FINAL OUTPUT
# =============================================================================

def run_statistical_validation(
    cache_files: List[Dict],
    winner_params: Dict,
    num_tests_performed: int = 1_000_000,
    funding_rates: Optional[Dict] = None  # PHASE 5: Funding rate lookup
) -> StrategyValidationResult:
    """
    Run statistical validation on the winner parameters.

    Phase 4 fix: Apply Bonferroni correction for multiple comparisons
    and check if the strategy passes statistical significance tests.
    """
    # Collect returns from all validation folds
    in_sample_returns = []
    out_of_sample_returns = []

    for fold_info in cache_files:
        # Run backtest on training data
        train_result = run_single_backtest_streaming(fold_info['train_cache'], winner_params, funding_rates)
        if train_result['total_rounds'] > 0:
            # Estimate per-round returns
            avg_return = train_result['avg_pnl']
            for _ in range(train_result['total_rounds']):
                in_sample_returns.append(avg_return)

        # Run backtest on validation data
        val_result = run_single_backtest_streaming(fold_info['val_cache'], winner_params, funding_rates)
        if val_result['total_rounds'] > 0:
            avg_return = val_result['avg_pnl']
            for _ in range(val_result['total_rounds']):
                out_of_sample_returns.append(avg_return)

    # Run statistical validation
    validation = validate_strategy_statistically(
        in_sample_returns=in_sample_returns,
        out_of_sample_returns=out_of_sample_returns,
        num_tests_performed=num_tests_performed,
        verbose=True
    )

    return validation


def print_final_recommendation(coin: str, result: Dict, stat_validation: Optional[StrategyValidationResult] = None):
    """Print final recommendation with cross-fold validation and statistical results."""
    winner = result['winner']
    params = winner['params']

    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION COMPLETE: {coin}")
    print(f"{'=' * 70}")

    print(f"\nCROSS-FOLD VALIDATION RESULTS:")
    print(f"  Param sets profitable on ALL folds: {result['num_profitable_all']}")

    print(f"\n  Winner validation P&L by fold:")
    for i, pnl in enumerate(winner['fold_pnls']):
        print(f"    Fold {i}: {pnl:+.1f}%")
    print(f"    Average: {winner['avg_val_pnl']:+.1f}%")

    # Show statistical validation results if available
    if stat_validation:
        print(f"\nSTATISTICAL VALIDATION (Phase 4 Fix):")
        print(f"  Sharpe Ratio: {stat_validation.sharpe_ratio:.3f}")
        print(f"  95% CI: [{stat_validation.sharpe_ci_lower:.3f}, {stat_validation.sharpe_ci_upper:.3f}]")
        print(f"  P-Value: {stat_validation.p_value:.2e}")
        print(f"  Bonferroni Alpha: {stat_validation.corrected_alpha:.2e}")
        print(f"  Statistically Significant: {stat_validation.is_statistically_significant}")
        print(f"  OOS Sharpe: {stat_validation.oos_sharpe_ratio:.3f}")
        print(f"  Degradation: {stat_validation.sharpe_degradation_pct:.1f}%")
        print(f"  PASSES VALIDATION: {stat_validation.passes_validation}")
        if stat_validation.rejection_reasons:
            print(f"  Rejection Reasons:")
            for reason in stat_validation.rejection_reasons:
                print(f"    - {reason}")

    print(f"\nABSOLUTE BEST PARAMETERS:")
    print(f"  threshold_pct:      {params['threshold']}%")
    print(f"  trailing_pct:       {params['trailing']}%")
    print(f"  pyramid_step_pct:   {params['pyramid_step']}%")
    print(f"  max_pyramids:       {params['max_pyramids']}")
    print(f"  poll_interval:      {params['poll_interval']}")
    print(f"  size_schedule:      {params['size_schedule']}")
    print(f"  acceleration:       {params['acceleration']}")
    print(f"  min_spacing:        {params['min_spacing']}%")
    print(f"  time_decay:         {params['time_decay']}")
    print(f"  vol_type:           {params['vol_type']}")
    print(f"  vol_min:            {params['vol_min']}%")
    print(f"  vol_window:         {params['vol_window']}")

    # Live trading command
    poll = params['poll_interval']
    if poll == 'tick':
        poll_val = 0
    elif isinstance(poll, str):
        poll_val = float(poll.replace('s', ''))
    else:
        poll_val = poll

    max_pyr = params['max_pyramids']
    if max_pyr == 'unlimited':
        max_pyr = 9999

    print(f"\nLIVE TRADING COMMAND:")
    print(f"  python main.py --mode trading --symbol {coin} \\")
    print(f"    --threshold {params['threshold']} --trailing {params['trailing']} \\")
    print(f"    --pyramid {params['pyramid_step']} --max-pyramids {max_pyr}")

    print(f"{'=' * 70}")


def save_final_result(coin: str, result: Dict):
    """Save final result to JSON."""
    winner = result['winner']

    output = {
        'coin': coin,
        'winner_params': winner['params'],
        'fold_validation_pnls': winner['fold_pnls'],
        'avg_validation_pnl': winner['avg_val_pnl'],
        'all_folds_profitable': winner['all_profitable'],
        'num_profitable_all_folds': result['num_profitable_all'],
        'completed': datetime.now().isoformat()
    }

    json_file = os.path.join(LOG_DIR, f"{coin}_v4_final_result.json")
    with open(json_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {json_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("AUTONOMOUS PYRAMID STRATEGY OPTIMIZER v4 - ULTIMATE EDITION")
    print("(Hierarchical + 3-Fold Walk-Forward + Disk Streaming)")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # 1. Interactive coin selection
    coin = select_coin_interactive()
    print(f"\nSelected: {coin}")

    # 2. Get years of data
    years = get_years_from_user()

    # 3. Ensure directories exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 4. PHASE 5: Fetch funding rates for realistic simulation
    print(f"\n{'=' * 70}")
    print("PHASE 0: FETCHING FUNDING RATES (Realistic Simulation)")
    print(f"{'=' * 70}")

    funding_rates = None
    if APPLY_FUNDING_RATES:
        try:
            print(f"  Fetching historical funding rates for {coin}...")
            # get_funding_rates returns (payments_list, lookup_dict)
            payments, funding_rates = get_funding_rates(coin, years=5)
            if funding_rates:
                print(f"  Loaded {len(funding_rates)} funding rate entries")
                # Calculate average annual funding
                avg_rate = sum(funding_rates.values()) / len(funding_rates) if funding_rates else 0
                annual_drag = avg_rate * 3 * 365 * 100  # 3 per day, 365 days, as %
                print(f"  Average funding rate: {avg_rate*100:.4f}% ({annual_drag:.1f}% annual)")
            else:
                print("  WARNING: Could not fetch funding rates, continuing without")
        except Exception as e:
            print(f"  WARNING: Failed to fetch funding rates: {e}")
            print("  Continuing without funding rate simulation")

    # 5. Load tick data and create fold caches
    print(f"\n{'=' * 70}")
    print("PHASE 1: CREATING FOLD CACHES (Disk Streaming)")
    print(f"{'=' * 70}")

    try:
        tick_streamer = create_filtered_tick_streamer(
            coin,
            years=years,
            verbose=True,
            preload_to_memory=False  # CRITICAL: Stream from disk to avoid 500MB-1GB memory per instance
        )

        # PHASE 5: Validate data quality before optimization
        if VALIDATE_DATA_QUALITY:
            print("\n  Validating data quality...")
            try:
                # FIX: validate_data_quality expects List, not callable
                # Sample first 100k ticks for quality check (avoids loading entire dataset)
                sample_ticks = []
                for i, tick in enumerate(tick_streamer()):
                    sample_ticks.append(tick)
                    if i >= 100_000:
                        break

                quality_report = validate_data_quality(sample_ticks)
                if quality_report:
                    # Calculate quality score from report attributes
                    quality_score = quality_report.quality_score
                    print(f"  Data quality: {quality_score:.1f}%")
                    if quality_score < 80:
                        print("  WARNING: Data quality below 80%, results may be unreliable")
                    if quality_report.gap_count > 0:
                        print(f"  Detected {quality_report.gap_count} data gaps")
                # Recreate streamer after validation consumed it
                tick_streamer = create_filtered_tick_streamer(
                    coin, years=years, verbose=False, preload_to_memory=False
                )
            except Exception as e:
                print(f"  Data quality check failed: {e}")
                # Recreate streamer
                tick_streamer = create_filtered_tick_streamer(
                    coin, years=years, verbose=False, preload_to_memory=False
                )

        cache_files = create_fold_caches(coin, tick_streamer, NUM_FOLDS)

        total_train = sum(c['train_size'] for c in cache_files)
        total_val = sum(c['val_size'] for c in cache_files)
        print(f"\n  Total across folds: {total_train:,} train ticks, {total_val:,} val ticks")

        # PHASE 5: Create holdout cache for final evaluation
        print("\n  Creating holdout cache (20% of data)...")
        tick_streamer = create_filtered_tick_streamer(
            coin, years=years, verbose=False, preload_to_memory=False
        )
        holdout_cache, holdout_ticks, train_ticks = create_holdout_cache(coin, tick_streamer)
        print(f"  Holdout: {holdout_ticks:,} ticks, Training: {train_ticks:,} ticks")

        # Memory check after fold cache creation
        check_memory_usage("after_fold_cache_creation")

    except Exception as e:
        print(f"ERROR: Failed to create fold caches: {e}")
        return

    # 6. Run multi-fold optimization
    print(f"\n{'=' * 70}")
    print("PHASE 2: MULTI-FOLD HIERARCHICAL OPTIMIZATION")
    print(f"{'=' * 70}")

    total_start = time.time()

    result = run_multi_fold_optimization(coin, cache_files, funding_rates)

    if result is None:
        print("\nERROR: Optimization failed!")
        return

    total_time = time.time() - total_start

    # 6. Check for validation failure
    if result['num_profitable_all'] == 0:
        print("\n" + "=" * 70)
        print("WARNING: NO PARAMETERS PROFITABLE ON ALL FOLDS!")
        print("=" * 70)
        print("This strategy may not work reliably for this coin.")
        print("Consider:")
        print("  1. Using different parameters")
        print("  2. Trying a different coin")
        print("  3. The market conditions may not suit this strategy")

    # 7. Robustness check on winner
    print(f"\n{'=' * 70}")
    print("PHASE 3: ROBUSTNESS CHECK")
    print(f"{'=' * 70}")

    winner_params = result['winner']['params']
    robustness, perturbations = calculate_robustness_score(
        cache_files[0]['train_cache'],
        winner_params,
        funding_rates  # PHASE 5
    )

    print(f"  Robustness score: {robustness:.2f}")
    if robustness < MIN_ROBUSTNESS_SCORE:
        print(f"  WARNING: Low robustness! Small parameter changes cause large P&L drops.")
        print(f"  Perturbation results:")
        for p in perturbations:
            print(f"    {p['param']} {p['delta']:+.1f}: P&L ratio = {p['ratio']:.2f}")

    # 8. Statistical validation (Phase 4 fix)
    print(f"\n{'=' * 70}")
    print("PHASE 4: STATISTICAL VALIDATION")
    print(f"{'=' * 70}")

    # Estimate number of tests performed (core grid × new params × fine-tuning)
    core_grid = build_dense_core_grid()
    new_grid = build_dense_new_params_grid()
    core_size = calculate_grid_size(core_grid)
    new_size = calculate_grid_size(new_grid)
    estimated_tests = core_size + (TOP_N_CORE * new_size) + (TOP_N_FINAL * 100)
    print(f"  Estimated tests performed: {estimated_tests:,}")

    stat_validation = run_statistical_validation(
        cache_files=cache_files,
        winner_params=winner_params,
        num_tests_performed=estimated_tests,
        funding_rates=funding_rates  # PHASE 5
    )

    # 9. PHASE 5: Holdout evaluation (TRUE out-of-sample test)
    print(f"\n{'=' * 70}")
    print("PHASE 5: HOLDOUT EVALUATION (TRUE OUT-OF-SAMPLE)")
    print(f"{'=' * 70}")

    holdout_result = evaluate_on_holdout(
        holdout_cache=holdout_cache,
        best_params=winner_params,
        verbose=True,
        funding_rates=funding_rates
    )

    # Check if holdout result is consistent with validation
    holdout_pass = holdout_result['total_pnl'] > 0
    if holdout_pass:
        print("\n  [PASS] Holdout evaluation positive!")
    else:
        print("\n  [FAIL] Holdout evaluation negative - strategy may be overfit!")

    # VERIFY: Funding rates were actually applied (if enabled)
    if APPLY_FUNDING_RATES and funding_rates:
        funding_events = holdout_result.get('funding_events', 0)
        total_funding_paid = holdout_result.get('total_funding_paid', 0)
        if funding_events > 0:
            print(f"  [OK] Funding rates applied: {funding_events} events, ${total_funding_paid:.2f} paid")
        else:
            print("  [WARN] Funding rates enabled but no funding events occurred!")
            print("         This may indicate a bug or insufficient test duration.")

    # POST-HOC RISK CONSTRAINT CHECK (applied after optimization completes)
    # This ensures final candidates meet risk thresholds without biasing the search
    risk_pass, risk_reasons = passes_risk_constraints(holdout_result)
    if risk_pass:
        print("  [PASS] Risk constraints passed on holdout data")
    else:
        print("  [FAIL] Risk constraints failed on holdout data:")
        for reason in risk_reasons:
            print(f"         - {reason}")

    # REGIME VALIDATION (must profit in 70%+ of market regimes)
    # This ensures the strategy works across different market conditions
    regime_pass, regime_validation = validate_regime_robustness(
        holdout_cache=holdout_cache,
        best_params=winner_params,
        funding_rates=funding_rates,
        min_profitable_regimes_pct=70.0,
        verbose=True
    )
    if regime_pass:
        print("  [PASS] Regime validation passed")
    else:
        print("  [FAIL] Regime validation failed - strategy may only work in specific conditions")

    # PHASE 5: MONTE CARLO STRESS TESTING
    # This validates that results aren't due to lucky sequencing
    print(f"\n{'=' * 70}")
    print("PHASE 5.1: MONTE CARLO STRESS TESTING")
    print(f"{'=' * 70}")

    # Extract round returns from holdout result
    # We need to get per-round P&L from the holdout backtest
    holdout_rounds = holdout_result.get('rounds', [])
    if holdout_rounds:
        round_returns = [r.total_pnl if hasattr(r, 'total_pnl') else r.get('total_pnl', 0)
                        for r in holdout_rounds]
    else:
        # Fallback: estimate from total P&L and rounds
        total_pnl = holdout_result.get('total_pnl', 0)
        num_rounds = holdout_result.get('total_rounds', 1)
        avg_pnl = total_pnl / num_rounds if num_rounds > 0 else 0
        # Generate synthetic returns (less accurate but usable)
        round_returns = [avg_pnl] * num_rounds
        print("  [WARN] Using estimated round returns (no detailed round data available)")

    if len(round_returns) >= 10:
        # Run Monte Carlo validation with -40% ruin threshold (user specified)
        mc_pass, mc_result, perm_result = run_monte_carlo_validation(
            round_returns=round_returns,
            params=winner_params,
            n_bootstrap=10000,
            n_permutation=5000,
            ruin_threshold=0.40,  # User specified -40% safety halt
            random_seed=42
        )

        # Print Monte Carlo report
        print(format_monte_carlo_report(mc_result, perm_result, coin))

        if mc_pass:
            print("\n  [PASS] Monte Carlo validation passed!")
        else:
            print("\n  [FAIL] Monte Carlo validation failed:")
            for reason in mc_result.failure_reasons:
                print(f"         - {reason}")
    else:
        print(f"  [SKIP] Insufficient rounds for Monte Carlo ({len(round_returns)} < 10)")
        mc_pass = False
        mc_result = None
        perm_result = None

    # 10. Save and display final results
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")

    save_final_result(coin, result)
    print_final_recommendation(coin, result, stat_validation)

    # Final verdict - must pass ALL validation checks (including Monte Carlo)
    all_pass = (stat_validation.passes_validation and holdout_pass and
                risk_pass and regime_pass and mc_pass)
    if all_pass:
        print("\n" + "=" * 70)
        print("STRATEGY PASSES ALL VALIDATION CHECKS (INCLUDING MONTE CARLO)")
        print("Ready for paper trading validation (4 weeks, 60+ rounds)")
        print("=" * 70)
        print("\nNEXT STEPS:")
        print("  1. Run paper trading on Binance Futures Testnet")
        print("  2. Complete 60+ rounds (approximately 4 weeks)")
        print("  3. Verify P&L is within Monte Carlo confidence interval")
        print("  4. Only then proceed to live trading")
    else:
        print("\n" + "=" * 70)
        print("WARNING: STRATEGY FAILED VALIDATION CHECKS")
        print("=" * 70)
        if not stat_validation.passes_validation:
            print("  [FAIL] Statistical validation failed")
        if not holdout_pass:
            print("  [FAIL] Holdout evaluation failed (likely overfit)")
        if not risk_pass:
            print("  [FAIL] Risk constraints failed (see reasons above)")
        if not regime_pass:
            print("  [FAIL] Regime validation failed (may only work in specific market conditions)")
        if not mc_pass:
            print("  [FAIL] Monte Carlo validation failed (results may be luck, not skill)")
        print("\nConsider:")
        print("  1. The results may be due to overfitting / data mining")
        print("  2. Try a different coin or time period")
        print("  3. Use smaller position sizes if proceeding")
        print("  4. DO NOT proceed to live trading without passing all checks")
        print("=" * 70)

    print(f"\nTotal optimization time: {total_time/3600:.1f} hours")
    print(f"Completed: {datetime.now()}")


if __name__ == "__main__":
    main()
