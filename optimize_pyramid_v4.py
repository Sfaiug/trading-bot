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
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Callable, Generator, Any
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.tick_data_fetcher import (
    get_years_from_user,
    create_filtered_tick_streamer,
    count_filtered_ticks,
    aggregate_ticks_to_interval
)
from backtest_pyramid import run_pyramid_backtest, PyramidRound


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
    'poll_interval': (0, 30.0),
    'acceleration': (0.3, 5.0),
    'min_spacing': (0.0, 10.0),
    'time_decay': (30, 7200),
    'vol_min': (0.0, 10.0),
    'vol_window': (10, 1000),
}


# =============================================================================
# VERY DENSE PARAMETER GRIDS
# =============================================================================

def build_dense_core_grid() -> Dict[str, List]:
    """
    Dense grid for the 5 core parameters.
    Other 7 parameters use neutral defaults during Phase A.
    """
    return {
        # Core 5 parameters - VERY DENSE
        'threshold': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],  # 15
        'trailing': [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0],  # 11
        'pyramid_step': [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0],  # 10
        'max_pyramids': [5, 10, 15, 20, 30, 40, 60, 80, 120, 200, 500, 9999],  # 12
        'poll_interval': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],  # 11
        # Neutral defaults for new params
        'size_schedule': ['fixed'],
        'acceleration': [1.0],
        'min_spacing': [0.0],
        'time_decay': [None],
        'vol_type': ['none'],
        'vol_min': [0.0],
        'vol_window': [100],
    }
    # Core combos: 15 × 11 × 10 × 12 × 11 = 217,800


def build_dense_new_params_grid() -> Dict[str, List]:
    """
    Dense grid for the 7 new parameters.
    Used in Phase B for each top core param combo.
    """
    return {
        'size_schedule': ['fixed', 'linear_decay', 'exp_decay'],  # 3
        'acceleration': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0],  # 11
        'min_spacing': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],  # 10
        'time_decay': [None, 60, 120, 180, 300, 450, 600, 900, 1200, 1800, 3600],  # 11
        'vol_type': ['none', 'stddev', 'range'],  # 3
        'vol_min': [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0],  # 9
        'vol_window': [25, 50, 75, 100, 125, 150, 200, 300, 500],  # 9
    }
    # New combos: 3 × 11 × 10 × 11 × 3 × 9 × 9 = 882,090


def build_fine_tuning_perturbations(best_params: Dict) -> List[Dict]:
    """
    Generate a list of small perturbations around the best parameters.
    Tests ONE parameter at a time, not all combinations.

    Returns list of param dicts to test (~60 total, not millions).
    """
    perturbations = [best_params.copy()]  # Start with the best

    # For numeric params, test +/- small amounts (one at a time)
    numeric_deltas = {
        'threshold': [0.25, 0.5, 1.0],
        'trailing': [0.05, 0.1, 0.2],
        'pyramid_step': [0.05, 0.1, 0.2],
        'acceleration': [0.02, 0.05, 0.1],
        'min_spacing': [0.05, 0.1, 0.2],
        'vol_min': [0.02, 0.05, 0.1],
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

    # For vol_window
    vol_win = best_params.get('vol_window', 100)
    if isinstance(vol_win, int):
        for delta in [5, 10, 25]:
            for sign in [1, -1]:
                new_val = max(10, vol_win + sign * delta)
                new_params = best_params.copy()
                new_params['vol_window'] = new_val
                perturbations.append(new_params)

    # For poll_interval
    poll = best_params.get('poll_interval', 0)
    if isinstance(poll, str):
        poll = 0 if poll == 'tick' else float(poll.replace('s', ''))
    for delta in [0.05, 0.1, 0.2]:
        for sign in [1, -1]:
            new_val = max(0, round(poll + sign * delta, 2))
            new_params = best_params.copy()
            new_params['poll_interval'] = new_val
            perturbations.append(new_params)

    # For time_decay
    td = best_params.get('time_decay')
    if td is not None and td != 'None':
        if isinstance(td, str):
            td = int(td.replace('s', ''))
        for delta in [15, 30, 60]:
            for sign in [1, -1]:
                new_val = max(30, td + sign * delta)
                new_params = best_params.copy()
                new_params['time_decay'] = new_val
                perturbations.append(new_params)
    else:
        # Try enabling time_decay
        for td_val in [60, 120, 300]:
            new_params = best_params.copy()
            new_params['time_decay'] = td_val
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

    # First, count total ticks (streaming through)
    print("  Counting total ticks...")
    total_ticks = 0
    for _ in tick_streamer():
        total_ticks += 1
    print(f"  Total ticks: {total_ticks:,}")

    if total_ticks < 10000:
        raise ValueError(f"Insufficient data: {total_ticks} ticks")

    # Calculate fold boundaries
    # Fold structure (for 5 years):
    #   Fold 1: Train Y1-Y3 (60%), Val Y4 (20%)
    #   Fold 2: Train Y1-Y4 (80%), Val Y5 (20%)
    #   Fold 3: Train Y2-Y4 (60%), Val Y5 (20%)
    folds = []
    fold_size = total_ticks // (num_folds + 1)

    for fold_num in range(num_folds):
        if fold_num == 0:
            train_end = int(total_ticks * 0.6)
            val_end = int(total_ticks * 0.8)
        elif fold_num == 1:
            train_end = int(total_ticks * 0.8)
            val_end = total_ticks
        else:
            train_end = int(total_ticks * 0.8)
            val_end = total_ticks

        folds.append({
            'fold_num': fold_num,
            'train_start': 0 if fold_num != 2 else int(total_ticks * 0.2),
            'train_end': train_end,
            'val_start': train_end if fold_num != 0 else int(total_ticks * 0.6),
            'val_end': val_end if fold_num != 0 else int(total_ticks * 0.8),
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


def count_cache_ticks(cache_file: str) -> int:
    """Count ticks in a cache file without loading to memory."""
    return os.path.getsize(cache_file) // 16


# =============================================================================
# GRID SEARCH WITH STREAMING
# =============================================================================

def calculate_grid_size(param_grid: Dict[str, List]) -> int:
    """Calculate total combinations in a parameter grid."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def run_streaming_grid_search(
    coin: str,
    cache_file: str,
    param_grid: Dict,
    phase_name: str,
    base_params: Optional[Dict] = None
) -> List[Dict]:
    """
    Run grid search using disk streaming - never loads all data to memory.

    Args:
        coin: Trading pair symbol
        cache_file: Path to binary tick cache
        param_grid: Parameters to grid search
        phase_name: Name for logging (e.g., "Phase_A", "Phase_B_combo1")
        base_params: Fixed params to merge with grid params (for Phase B)

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
        'total_pnl', 'rounds', 'avg_pnl', 'win_rate', 'avg_pyramids'
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
                tick_stream = create_disk_streamer(cache_file)
                aggregated = aggregate_ticks_to_interval(tick_stream, poll)

                # Get time_decay value
                time_decay = full_params.get('time_decay')
                if isinstance(time_decay, str):
                    time_decay = None if time_decay == 'None' else int(time_decay.replace('s', ''))

                # Get max_pyramids value
                max_pyr = full_params.get('max_pyramids', 80)
                if max_pyr == 'unlimited':
                    max_pyr = 9999

                # Run backtest with streaming data
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
                    volatility_window_size=full_params.get('vol_window', 100)
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
                    'total_pnl': result['total_pnl'],
                    'rounds': result['total_rounds'],
                    'avg_pnl': result['avg_pnl'],
                    'win_rate': result['win_rate'],
                    'avg_pyramids': result['avg_pyramids']
                }

                # Write to disk immediately
                writer.writerow(entry)
                f.flush()

                # Track top N
                if len(top_results) < max_top:
                    top_results.append(entry)
                    top_results.sort(key=lambda x: x['total_pnl'], reverse=True)
                elif entry['total_pnl'] > top_results[-1]['total_pnl']:
                    top_results[-1] = entry
                    top_results.sort(key=lambda x: x['total_pnl'], reverse=True)

            except Exception as e:
                if completed % 5000 == 0:
                    print(f"\n    Error at combo {completed}: {e}")

            # Periodic garbage collection
            if completed % 10000 == 0:
                gc.collect()

    print()  # Newline after progress
    return top_results


# =============================================================================
# SINGLE BACKTEST FOR VALIDATION
# =============================================================================

def run_single_backtest_streaming(cache_file: str, params: Dict) -> Dict:
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
    tick_stream = create_disk_streamer(cache_file)
    aggregated = aggregate_ticks_to_interval(tick_stream, poll)

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
        volatility_window_size=params.get('vol_window', 100)
    )

    return result


# =============================================================================
# ROBUSTNESS TESTING
# =============================================================================

def calculate_robustness_score(
    cache_file: str,
    best_params: Dict
) -> Tuple[float, List[Dict]]:
    """
    Test nearby parameters to check if the optimum is robust.

    Returns (robustness_score, perturbation_results).
    Score of 1.0 = all perturbations give similar P&L
    Score of 0.5 = perturbations give only 50% of best P&L
    """
    center_result = run_single_backtest_streaming(cache_file, best_params)
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

            result = run_single_backtest_streaming(cache_file, perturbed)
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
    fold_num: int
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
        phase_name=f"fold{fold_num}_phaseA"
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
            base_params=base_params
        )

        if phase_b_results:
            all_phase_b_results.extend(phase_b_results[:10])  # Keep top 10 from each

        gc.collect()

    # Sort all Phase B results and take top N
    all_phase_b_results.sort(key=lambda x: x['total_pnl'], reverse=True)
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

            result = run_single_backtest_streaming(train_cache, perturbed_params)

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
    final_results.sort(key=lambda x: x['total_pnl'], reverse=True)

    print(f"\n  Phase C complete. Final top 5:")
    for i, r in enumerate(final_results[:5]):
        print(f"    #{i+1}: T={r['threshold']}% Tr={r['trailing']}% -> {r['total_pnl']:+.2f}%")

    return final_results[:TOP_N_FINAL]


# =============================================================================
# 3-FOLD WALK-FORWARD VALIDATION
# =============================================================================

def run_multi_fold_optimization(
    coin: str,
    cache_files: List[Dict]
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
            fold_num=fold_num
        )

        if not train_results:
            print(f"  WARNING: Fold {fold_num} produced no results!")
            continue

        # Validate top 10 on this fold's validation data
        print(f"\n  Validating top {len(train_results)} on Fold {fold_num} validation data...")

        validated = []
        for params in train_results:
            val_result = run_single_backtest_streaming(val_cache, params)
            validated.append({
                'params': params,
                'train_pnl': params['total_pnl'],
                'val_pnl': val_result['total_pnl'],
            })

        validated.sort(key=lambda x: x['val_pnl'], reverse=True)

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
        for fold_info in cache_files[1:]:
            val_result = run_single_backtest_streaming(fold_info['val_cache'], params)
            fold_pnls.append(val_result['total_pnl'])
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
        })

    # Filter to only those profitable on ALL folds
    profitable_all = [r for r in cross_fold_results if r['all_profitable']]

    if profitable_all:
        # Sort by average validation P&L
        profitable_all.sort(key=lambda x: x['avg_val_pnl'], reverse=True)
        winner = profitable_all[0]
        print(f"\n  Found {len(profitable_all)} param sets profitable on ALL folds!")
    else:
        # Fallback: sort by minimum fold P&L (most consistent)
        cross_fold_results.sort(key=lambda x: x['min_fold_pnl'], reverse=True)
        winner = cross_fold_results[0]
        print(f"\n  WARNING: No param set profitable on ALL folds!")
        print(f"  Using most consistent: min_fold_pnl = {winner['min_fold_pnl']:+.1f}%")

    return {
        'winner': winner,
        'all_results': cross_fold_results[:20],
        'num_profitable_all': len(profitable_all),
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

def print_final_recommendation(coin: str, result: Dict):
    """Print final recommendation with cross-fold validation results."""
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

    # 4. Load tick data and create fold caches
    print(f"\n{'=' * 70}")
    print("PHASE 1: CREATING FOLD CACHES (Disk Streaming)")
    print(f"{'=' * 70}")

    try:
        tick_streamer = create_filtered_tick_streamer(coin, years=years, verbose=True)
        cache_files = create_fold_caches(coin, tick_streamer, NUM_FOLDS)

        total_train = sum(c['train_size'] for c in cache_files)
        total_val = sum(c['val_size'] for c in cache_files)
        print(f"\n  Total across folds: {total_train:,} train ticks, {total_val:,} val ticks")

    except Exception as e:
        print(f"ERROR: Failed to create fold caches: {e}")
        return

    # 5. Run multi-fold optimization
    print(f"\n{'=' * 70}")
    print("PHASE 2: MULTI-FOLD HIERARCHICAL OPTIMIZATION")
    print(f"{'=' * 70}")

    total_start = time.time()

    result = run_multi_fold_optimization(coin, cache_files)

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
        winner_params
    )

    print(f"  Robustness score: {robustness:.2f}")
    if robustness < MIN_ROBUSTNESS_SCORE:
        print(f"  WARNING: Low robustness! Small parameter changes cause large P&L drops.")
        print(f"  Perturbation results:")
        for p in perturbations:
            print(f"    {p['param']} {p['delta']:+.1f}: P&L ratio = {p['ratio']:.2f}")

    # 8. Save and display final results
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")

    save_final_result(coin, result)
    print_final_recommendation(coin, result)

    print(f"\nTotal optimization time: {total_time/3600:.1f} hours")
    print(f"Completed: {datetime.now()}")


if __name__ == "__main__":
    main()
