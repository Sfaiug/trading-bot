#!/usr/bin/env python3
"""
Enhanced Pyramid Strategy Optimizer v2

Features:
- Tick-level data from Binance Data Portal
- max_pyramids as 4th optimization variable
- Comprehensive analytics and insights
- Monthly P&L breakdown
- Account sizing recommendations

Usage:
    python optimize_pyramid_v2.py
    python optimize_pyramid_v2.py --coins BTCUSDT,ETHUSDT --quick-test
"""

import os
import sys
import csv
import time
import json
import gc
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Callable, Generator
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.tick_data_fetcher import (
    fetch_tick_data,
    get_years_from_user,
    create_price_streamer,
    count_cached_prices,
    create_tick_streamer_raw,
    count_cached_ticks_raw,
    create_filtered_tick_streamer,
    count_filtered_ticks,
    aggregate_ticks_to_interval
)
from backtest_pyramid import run_pyramid_backtest, PyramidRound
from analytics import (
    RoundSummary, 
    generate_analytics_summary, 
    print_analytics_summary,
    calculate_monthly_breakdown
)


# Configuration
DEFAULT_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "XLMUSDT"]

# Grid parameters - FULL (Original 5)
THRESHOLDS = [float(i) for i in range(1, 21)]  # 1-20%
TRAILINGS = [1.0 + 0.2 * i for i in range(21)]  # 1-5% in 0.2 steps
PYRAMID_STEPS = [1.0 + 0.2 * i for i in range(11)]  # 1-3% in 0.2 steps
MAX_PYRAMIDS = [5, 10, 20, 40, 80, 160, 320, 640, 9999]  # 9999 = unlimited
POLL_INTERVALS = [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]  # 0 = tick-by-tick (live)

# Grid parameters - NEW 7 PARAMETERS
SIZE_SCHEDULES = ['fixed', 'linear_decay', 'exp_decay']
ACCELERATIONS = [1.0, 1.2, 1.5]
MIN_SPACINGS = [0.0, 1.0, 2.0]
TIME_DECAYS = [None, 300, 900]  # None = disabled, seconds otherwise
VOL_TYPES = ['none', 'stddev', 'range']
VOL_MINS = [0.0, 0.5, 1.0]
VOL_WINDOWS = [50, 100, 200]

# Grid parameters - QUICK TEST (Original 5)
THRESHOLDS_QUICK = [1, 3, 5, 10, 15]
TRAILINGS_QUICK = [1.0, 2.0, 3.0, 4.0]
PYRAMID_STEPS_QUICK = [1.0, 2.0, 3.0]
MAX_PYRAMIDS_QUICK = [10, 40, 160]
POLL_INTERVALS_QUICK = [0, 0.4, 1.6, 6.4]

# Grid parameters - QUICK TEST (New 7)
SIZE_SCHEDULES_QUICK = ['fixed', 'linear_decay']
ACCELERATIONS_QUICK = [1.0, 1.3]
MIN_SPACINGS_QUICK = [0.0, 1.0]
TIME_DECAYS_QUICK = [None, 600]
VOL_TYPES_QUICK = ['none', 'stddev']
VOL_MINS_QUICK = [0.0, 0.5]
VOL_WINDOWS_QUICK = [100]

TOP_N = 10
REFINED_COMBOS = 500
LOG_DIR = "logs"

# Convergence settings
MAX_CONVERGENCE_ROUNDS = 10
CONVERGENCE_THRESHOLD = 0.1  # Stop when improvement < 0.1%


@dataclass
class CoinResult:
    """Lightweight result container - only stores essential data, not full result lists."""
    coin: str
    best_result: Dict          # Only the best parameters found
    analytics: Dict            # Analytics summary
    grid_log_file: str         # Path to CSV with full grid results
    refined_log_file: str      # Path to CSV with refined results


def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)


def rounds_to_summaries(rounds: List[PyramidRound]) -> List[RoundSummary]:
    """Convert PyramidRound to RoundSummary for analytics."""
    return [
        RoundSummary(
            entry_time=r.entry_time,
            exit_time=r.exit_time,
            pnl_percent=r.total_pnl,
            num_pyramids=r.num_pyramids,
            direction=r.direction
        )
        for r in rounds
    ]


def run_grid_search(
    coin: str,
    tick_streamer: Callable,
    param_grid: Dict[str, List]
) -> Tuple[str, List[Dict]]:
    """
    Run grid search over all parameter combinations.

    Memory-efficient: streams prices from disk for each backtest,
    only keeps top N results in memory.

    Args:
        coin: Trading pair symbol
        tick_streamer: Callable that returns a generator of (timestamp, price) tick data
        param_grid: Dictionary of parameter names to lists of values to test

    Returns:
        Tuple of (log_file_path, top_10_results)
    """
    # Extract parameter lists
    thresholds = param_grid.get('threshold', [5.0])
    trailings = param_grid.get('trailing', [2.0])
    pyramid_steps = param_grid.get('pyramid_step', [2.0])
    max_pyramids_list = param_grid.get('max_pyramids', [20])
    poll_intervals = param_grid.get('poll_interval', [1.0])
    size_schedules = param_grid.get('size_schedule', ['fixed'])
    accelerations = param_grid.get('acceleration', [1.0])
    min_spacings = param_grid.get('min_spacing', [0.0])
    time_decays = param_grid.get('time_decay', [None])
    vol_types = param_grid.get('vol_type', ['none'])
    vol_mins = param_grid.get('vol_min', [0.0])
    vol_windows = param_grid.get('vol_window', [100])

    # Calculate total combinations
    total = (len(thresholds) * len(trailings) * len(pyramid_steps) *
             len(max_pyramids_list) * len(poll_intervals) * len(size_schedules) *
             len(accelerations) * len(min_spacings) * len(time_decays) *
             len(vol_types) * len(vol_mins) * len(vol_windows))

    top_results = []  # Only keep top N in memory

    log_file = os.path.join(LOG_DIR, f"{coin}_pyramid_v3_grid.csv")

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

        for poll_interval in poll_intervals:
            poll_label = "tick" if poll_interval == 0 else f"{poll_interval}s"

            for size_sched in size_schedules:
                for accel in accelerations:
                    for min_space in min_spacings:
                        for time_decay in time_decays:
                            for vol_type in vol_types:
                                for vol_min in vol_mins:
                                    for vol_win in vol_windows:
                                        for threshold in thresholds:
                                            for trailing in trailings:
                                                for pyramid_step in pyramid_steps:
                                                    for max_pyr in max_pyramids_list:
                                                        completed += 1

                                                        elapsed = time.time() - start
                                                        rate = completed / elapsed if elapsed > 0 else 1
                                                        remaining = (total - completed) / rate

                                                        if completed % 500 == 0 or completed == 1:
                                                            pct = (completed / total) * 100
                                                            print(f"\r    [{completed:,}/{total:,}] {pct:.1f}% | "
                                                                  f"{remaining/3600:.1f}h remaining    ",
                                                                  end="", flush=True)

                                                        try:
                                                            # Aggregate tick data to the poll interval
                                                            price_stream = aggregate_ticks_to_interval(
                                                                tick_streamer,
                                                                poll_interval
                                                            )

                                                            # Run backtest with all parameters
                                                            result = run_pyramid_backtest(
                                                                price_stream,
                                                                threshold_pct=threshold,
                                                                trailing_pct=trailing,
                                                                pyramid_step_pct=pyramid_step,
                                                                max_pyramids=max_pyr,
                                                                verbose=False,
                                                                pyramid_size_schedule=size_sched,
                                                                min_pyramid_spacing_pct=min_space,
                                                                pyramid_acceleration=accel,
                                                                time_decay_exit_seconds=time_decay,
                                                                volatility_filter_type=vol_type,
                                                                volatility_min_pct=vol_min,
                                                                volatility_window_size=vol_win
                                                            )

                                                            time_decay_label = 'None' if time_decay is None else f'{time_decay}s'
                                                            entry = {
                                                                'threshold': threshold,
                                                                'trailing': trailing,
                                                                'pyramid_step': pyramid_step,
                                                                'max_pyramids': max_pyr if max_pyr < 9999 else 'unlimited',
                                                                'poll_interval': poll_label,
                                                                'size_schedule': size_sched,
                                                                'acceleration': accel,
                                                                'min_spacing': min_space,
                                                                'time_decay': time_decay_label,
                                                                'vol_type': vol_type,
                                                                'vol_min': vol_min,
                                                                'vol_window': vol_win,
                                                                'total_pnl': result['total_pnl'],
                                                                'rounds': result['total_rounds'],
                                                                'avg_pnl': result['avg_pnl'],
                                                                'win_rate': result['win_rate'],
                                                                'avg_pyramids': result['avg_pyramids']
                                                            }

                                                            # Write immediately to disk
                                                            writer.writerow(entry)
                                                            f.flush()

                                                            # Maintain only top N in memory
                                                            if len(top_results) < TOP_N:
                                                                top_results.append(entry)
                                                                top_results.sort(key=lambda x: x['total_pnl'], reverse=True)
                                                            elif entry['total_pnl'] > top_results[-1]['total_pnl']:
                                                                top_results[-1] = entry
                                                                top_results.sort(key=lambda x: x['total_pnl'], reverse=True)

                                                        except Exception as e:
                                                            if completed % 1000 == 0:
                                                                print(f"\n  Error: {e}")

    print()
    return log_file, top_results


def run_refined_search(
    coin: str,
    tick_streamer: Callable,
    top_results: List[Dict],
    step_divisor: int = 2
) -> Tuple[str, Dict]:
    """
    Refine search around top results with all 12 parameters.

    Memory-efficient: streams prices from disk for each backtest,
    only keeps the single best result in memory.

    Args:
        coin: Trading pair symbol
        tick_streamer: Callable that returns a generator of (timestamp, price) tick data
        top_results: Top N results from grid search to refine around
        step_divisor: How much finer to make the grid (2 = half steps, 4 = quarter steps)

    Returns:
        Tuple of (log_file_path, best_result)
    """
    best_result = None

    log_file = os.path.join(LOG_DIR, f"{coin}_pyramid_v3_refined.csv")

    fieldnames = [
        'base_config',
        'threshold', 'trailing', 'pyramid_step', 'max_pyramids', 'poll_interval',
        'size_schedule', 'acceleration', 'min_spacing', 'time_decay',
        'vol_type', 'vol_min', 'vol_window',
        'total_pnl', 'rounds', 'avg_pnl', 'win_rate', 'avg_pyramids'
    ]

    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, top in enumerate(top_results, 1):
            # Extract base values from top result
            base_t = top['threshold']
            base_tr = top['trailing']
            base_p = top['pyramid_step']
            base_mp = top['max_pyramids']
            base_poll = top.get('poll_interval', '1.0s')
            base_size_sched = top.get('size_schedule', 'fixed')
            base_accel = top.get('acceleration', 1.0)
            base_min_space = top.get('min_spacing', 0.0)
            base_time_decay = top.get('time_decay', 'None')
            base_vol_type = top.get('vol_type', 'none')
            base_vol_min = top.get('vol_min', 0.0)
            base_vol_win = top.get('vol_window', 100)

            # Handle 'unlimited' string
            if base_mp == 'unlimited':
                base_mp = 9999

            # Parse poll interval from string (e.g., "tick" or "0.4s")
            if base_poll == 'tick':
                base_poll_val = 0
            else:
                base_poll_val = float(base_poll.replace('s', ''))

            # Parse time_decay
            if base_time_decay == 'None' or base_time_decay is None:
                base_time_decay_val = None
            else:
                base_time_decay_val = int(str(base_time_decay).replace('s', ''))

            base_config = f"#{rank}: T={base_t}% Tr={base_tr}% P={base_p}%"
            print(f"\n  Refining {base_config}")

            # Generate refined values around each numeric parameter
            step = 0.1 / step_divisor
            thresholds = [round(base_t + step * i, 2) for i in range(-3, 4) if base_t + step * i > 0]
            trailings = [round(base_tr + step * i, 2) for i in range(-3, 4) if base_tr + step * i > 0]
            pyramids = [round(base_p + step * i, 2) for i in range(-3, 4) if base_p + step * i > 0]

            # For max_pyramids, test nearby values
            if base_mp >= 9999:
                max_pyr_variants = [640, 9999]
            else:
                mp_low = max(5, int(base_mp * 0.7))
                mp_high = min(9999, int(base_mp * 1.5))
                max_pyr_variants = sorted(set([mp_low, base_mp, mp_high]))

            # For poll_interval, test nearby values
            if base_poll_val == 0:
                poll_variants = [0, 0.1]
            else:
                poll_variants = [max(0.1, base_poll_val / 2), base_poll_val, base_poll_val * 2]

            # Categorical params - keep the best one only
            size_schedules = [base_size_sched]

            # Refine acceleration around base
            accel_step = 0.1 / step_divisor
            accelerations = [round(base_accel + accel_step * i, 2) for i in range(-2, 3) if base_accel + accel_step * i >= 0.5]

            # Refine min_spacing around base
            space_step = 0.5 / step_divisor
            min_spacings = [round(base_min_space + space_step * i, 2) for i in range(-2, 3) if base_min_space + space_step * i >= 0]

            # Time decay - test nearby values
            if base_time_decay_val is None:
                time_decays = [None]
            else:
                time_decays = [int(base_time_decay_val * 0.5), base_time_decay_val, int(base_time_decay_val * 1.5)]
                time_decays = [t for t in time_decays if t >= 60]

            # Volatility - keep same type, refine threshold
            vol_types = [base_vol_type]
            vol_step = 0.25 / step_divisor
            vol_mins = [round(base_vol_min + vol_step * i, 2) for i in range(-2, 3) if base_vol_min + vol_step * i >= 0]
            vol_windows = [base_vol_win]  # Keep same window

            total = (len(thresholds) * len(trailings) * len(pyramids) *
                     len(max_pyr_variants) * len(poll_variants) * len(size_schedules) *
                     len(accelerations) * len(min_spacings) * len(time_decays) *
                     len(vol_types) * len(vol_mins) * len(vol_windows))
            completed = 0

            for poll_interval in poll_variants:
                poll_label = "tick" if poll_interval == 0 else f"{poll_interval}s"

                for size_sched in size_schedules:
                    for accel in accelerations:
                        for min_space in min_spacings:
                            for time_decay in time_decays:
                                for vol_type in vol_types:
                                    for vol_min in vol_mins:
                                        for vol_win in vol_windows:
                                            for threshold in thresholds:
                                                for trailing in trailings:
                                                    for pyramid_step in pyramids:
                                                        for max_pyr in max_pyr_variants:
                                                            completed += 1

                                                            if completed % 100 == 0:
                                                                print(f"\r    [{completed}/{total}]    ", end="", flush=True)

                                                            try:
                                                                price_stream = aggregate_ticks_to_interval(
                                                                    tick_streamer,
                                                                    poll_interval
                                                                )

                                                                result = run_pyramid_backtest(
                                                                    price_stream,
                                                                    threshold_pct=threshold,
                                                                    trailing_pct=trailing,
                                                                    pyramid_step_pct=pyramid_step,
                                                                    max_pyramids=max_pyr,
                                                                    verbose=False,
                                                                    pyramid_size_schedule=size_sched,
                                                                    min_pyramid_spacing_pct=min_space,
                                                                    pyramid_acceleration=accel,
                                                                    time_decay_exit_seconds=time_decay,
                                                                    volatility_filter_type=vol_type,
                                                                    volatility_min_pct=vol_min,
                                                                    volatility_window_size=vol_win
                                                                )

                                                                time_decay_label = 'None' if time_decay is None else f'{time_decay}s'
                                                                entry = {
                                                                    'base_config': base_config,
                                                                    'threshold': threshold,
                                                                    'trailing': trailing,
                                                                    'pyramid_step': pyramid_step,
                                                                    'max_pyramids': max_pyr if max_pyr < 9999 else 'unlimited',
                                                                    'poll_interval': poll_label,
                                                                    'size_schedule': size_sched,
                                                                    'acceleration': accel,
                                                                    'min_spacing': min_space,
                                                                    'time_decay': time_decay_label,
                                                                    'vol_type': vol_type,
                                                                    'vol_min': vol_min,
                                                                    'vol_window': vol_win,
                                                                    'total_pnl': result['total_pnl'],
                                                                    'rounds': result['total_rounds'],
                                                                    'avg_pnl': result['avg_pnl'],
                                                                    'win_rate': result['win_rate'],
                                                                    'avg_pyramids': result['avg_pyramids']
                                                                }

                                                                writer.writerow(entry)
                                                                f.flush()

                                                                if best_result is None or entry['total_pnl'] > best_result['total_pnl']:
                                                                    best_result = entry

                                                            except:
                                                                pass

            print()

    return log_file, best_result


def process_coin(
    coin: str,
    tick_streamer: Callable,
    num_ticks: int,
    quick_test: bool = False,
    param_overrides: Dict[str, List] = None
) -> CoinResult:
    """
    Process a single coin through grid search and refinement with all 12 parameters.

    Memory-efficient: uses tick_streamer to stream data from disk
    for each backtest, avoiding loading all ticks into memory.

    Args:
        coin: Trading pair symbol
        tick_streamer: Callable that returns a generator of (timestamp, price) tick data
        num_ticks: Number of ticks in the dataset (for display)
        quick_test: Use reduced grid for quick testing
        param_overrides: Optional dict to override default parameter grids

    Returns:
        CoinResult with best parameters and analytics
    """
    print(f"\n{'#' * 70}")
    print(f"# PROCESSING: {coin}")
    print(f"{'#' * 70}")
    print(f"Data: {num_ticks:,} filtered ticks (streaming from memory)")

    # Build parameter grid
    if quick_test:
        param_grid = {
            'threshold': THRESHOLDS_QUICK,
            'trailing': TRAILINGS_QUICK,
            'pyramid_step': PYRAMID_STEPS_QUICK,
            'max_pyramids': MAX_PYRAMIDS_QUICK,
            'poll_interval': POLL_INTERVALS_QUICK,
            'size_schedule': SIZE_SCHEDULES_QUICK,
            'acceleration': ACCELERATIONS_QUICK,
            'min_spacing': MIN_SPACINGS_QUICK,
            'time_decay': TIME_DECAYS_QUICK,
            'vol_type': VOL_TYPES_QUICK,
            'vol_min': VOL_MINS_QUICK,
            'vol_window': VOL_WINDOWS_QUICK,
        }
    else:
        param_grid = {
            'threshold': THRESHOLDS,
            'trailing': TRAILINGS,
            'pyramid_step': PYRAMID_STEPS,
            'max_pyramids': MAX_PYRAMIDS,
            'poll_interval': POLL_INTERVALS,
            'size_schedule': SIZE_SCHEDULES,
            'acceleration': ACCELERATIONS,
            'min_spacing': MIN_SPACINGS,
            'time_decay': TIME_DECAYS,
            'vol_type': VOL_TYPES,
            'vol_min': VOL_MINS,
            'vol_window': VOL_WINDOWS,
        }

    # Apply any overrides
    if param_overrides:
        param_grid.update(param_overrides)

    # Calculate total combinations
    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)

    # Grid search
    print(f"\n--- Grid Search ({total_combos:,} combinations) ---")
    poll_labels = ['tick' if p == 0 else f'{p}s' for p in param_grid['poll_interval']]
    print(f"  Poll intervals: {poll_labels}")
    print(f"  Size schedules: {param_grid['size_schedule']}")
    print(f"  Accelerations: {param_grid['acceleration']}")
    start = time.time()
    grid_log_file, top_10 = run_grid_search(coin, tick_streamer, param_grid)
    print(f"  ✓ Completed in {(time.time()-start)/60:.1f} minutes")

    # Save top 10
    if top_10:
        with open(os.path.join(LOG_DIR, f"{coin}_pyramid_v3_top10.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=top_10[0].keys())
            writer.writeheader()
            writer.writerows(top_10)

    print(f"\n--- Top {TOP_N} ---")
    for i, r in enumerate(top_10, 1):
        mp = r['max_pyramids']
        poll = r.get('poll_interval', '1s')
        size_s = r.get('size_schedule', 'fixed')
        accel = r.get('acceleration', 1.0)
        print(f"  #{i}: T={r['threshold']}% Tr={r['trailing']}% P={r['pyramid_step']}% "
              f"MP={mp} Poll={poll} Sz={size_s} Acc={accel} → {r['total_pnl']:+.2f}%")

    # Refined search
    print(f"\n--- Refined Search ---")
    start = time.time()
    refined_log_file, best = run_refined_search(coin, tick_streamer, top_10)
    print(f"  ✓ Completed in {(time.time()-start)/60:.1f} minutes")

    # Fall back to top grid result if refined search found nothing better
    if best is None:
        best = top_10[0] if top_10 else {
            'threshold': 5, 'trailing': 2, 'pyramid_step': 2,
            'max_pyramids': 20, 'poll_interval': '1.0s',
            'size_schedule': 'fixed', 'acceleration': 1.0,
            'min_spacing': 0.0, 'time_decay': 'None',
            'vol_type': 'none', 'vol_min': 0.0, 'vol_window': 100,
            'total_pnl': 0, 'win_rate': 0
        }

    # Run best config again to get full rounds for analytics
    best_max_pyr = best['max_pyramids'] if best['max_pyramids'] != 'unlimited' else 9999

    # Parse poll interval for final run
    best_poll = best.get('poll_interval', '1.0s')
    if best_poll == 'tick':
        best_poll_val = 0
    else:
        best_poll_val = float(best_poll.replace('s', ''))

    # Parse time_decay
    best_time_decay = best.get('time_decay', 'None')
    if best_time_decay == 'None' or best_time_decay is None:
        best_time_decay_val = None
    else:
        best_time_decay_val = int(str(best_time_decay).replace('s', ''))

    # Aggregate ticks to best poll interval and run
    price_stream = aggregate_ticks_to_interval(tick_streamer, best_poll_val)
    best_result = run_pyramid_backtest(
        price_stream,
        threshold_pct=best['threshold'],
        trailing_pct=best['trailing'],
        pyramid_step_pct=best['pyramid_step'],
        max_pyramids=best_max_pyr,
        verbose=False,
        pyramid_size_schedule=best.get('size_schedule', 'fixed'),
        min_pyramid_spacing_pct=best.get('min_spacing', 0.0),
        pyramid_acceleration=best.get('acceleration', 1.0),
        time_decay_exit_seconds=best_time_decay_val,
        volatility_filter_type=best.get('vol_type', 'none'),
        volatility_min_pct=best.get('vol_min', 0.0),
        volatility_window_size=best.get('vol_window', 100)
    )

    # Generate analytics - pop rounds to avoid keeping them in memory
    rounds = best_result.pop('rounds', [])
    summaries = rounds_to_summaries(rounds)
    analytics = generate_analytics_summary(summaries, best_max_pyr)

    # Explicit cleanup of rounds data
    del rounds
    del summaries

    print(f"\n🏆 BEST FOR {coin}:")
    print(f"   Threshold:     {best['threshold']}%")
    print(f"   Trailing:      {best['trailing']}%")
    print(f"   Pyramid Step:  {best['pyramid_step']}%")
    print(f"   Max Pyramids:  {best['max_pyramids']}")
    print(f"   Poll Interval: {best.get('poll_interval', '1.0s')}")
    print(f"   Size Schedule: {best.get('size_schedule', 'fixed')}")
    print(f"   Acceleration:  {best.get('acceleration', 1.0)}")
    print(f"   Min Spacing:   {best.get('min_spacing', 0.0)}%")
    print(f"   Time Decay:    {best.get('time_decay', 'None')}")
    print(f"   Vol Type:      {best.get('vol_type', 'none')}")
    print(f"   Vol Min:       {best.get('vol_min', 0.0)}%")
    print(f"   Total P&L:     {best['total_pnl']:+.2f}%")
    print(f"   Win Rate:      {best['win_rate']:.1f}%")

    # Print analytics summary
    print_analytics_summary(analytics, coin)

    return CoinResult(coin, best, analytics, grid_log_file, refined_log_file)


def print_final_report(results: Dict[str, CoinResult], years: int):
    """Print and save comprehensive final report with all 12 parameters."""
    print("\n" + "=" * 140)
    print("ENHANCED PYRAMID STRATEGY v3 - FINAL RECOMMENDATIONS")
    print("=" * 140)
    print(f"Data period: {years} year(s) of tick data")
    print()

    # Compact header for terminal (key params only)
    print(f"{'Coin':<10} {'Thresh':<8} {'Trail':<8} {'PyrStp':<8} {'MaxPyr':<8} {'Poll':<8} "
          f"{'SzSched':<12} {'Accel':<6} {'P&L':<10} {'WinRate':<8}")
    print("-" * 140)

    final_data = []

    for coin in results:
        r = results[coin].best_result
        a = results[coin].analytics

        mp = r['max_pyramids'] if r['max_pyramids'] != 'unlimited' else '∞'
        poll = r.get('poll_interval', '1.0s')
        size_sched = r.get('size_schedule', 'fixed')
        accel = r.get('acceleration', 1.0)
        min_acct = a['account_sizing']['recommended_account']

        print(f"{coin:<10} {r['threshold']:<8.1f}% {r['trailing']:<8.1f}% {r['pyramid_step']:<8.1f}% "
              f"{mp:<8} {poll:<8} {size_sched:<12} {accel:<6.1f} {r['total_pnl']:+8.1f}%  {r['win_rate']:<8.1f}%")

        final_data.append({
            'coin': coin,
            'threshold': r['threshold'],
            'trailing': r['trailing'],
            'pyramid_step': r['pyramid_step'],
            'max_pyramids': r['max_pyramids'],
            'poll_interval': poll,
            'size_schedule': r.get('size_schedule', 'fixed'),
            'acceleration': r.get('acceleration', 1.0),
            'min_spacing': r.get('min_spacing', 0.0),
            'time_decay': r.get('time_decay', 'None'),
            'vol_type': r.get('vol_type', 'none'),
            'vol_min': r.get('vol_min', 0.0),
            'vol_window': r.get('vol_window', 100),
            'total_pnl': r['total_pnl'],
            'avg_pnl': r['avg_pnl'],
            'win_rate': r['win_rate'],
            'avg_pyramids': r['avg_pyramids'],
            'max_drawdown': a['max_drawdown'],
            'sharpe_ratio': a['sharpe_ratio'],
            'min_account': a['account_sizing']['min_account'],
            'recommended_account': a['account_sizing']['recommended_account'],
            'profitable_months': a['profitable_months'],
            'total_months': a['total_months']
        })

    print("-" * 140)

    # Save final recommendations
    with open(os.path.join(LOG_DIR, "pyramid_v3_recommendations.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_data[0].keys())
        writer.writeheader()
        writer.writerows(final_data)

    # Save monthly breakdown for each coin
    for coin, cr in results.items():
        monthly = cr.analytics.get('monthly_breakdown', {})
        if monthly:
            with open(os.path.join(LOG_DIR, f"{coin}_monthly_breakdown.csv"), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['month', 'pnl', 'rounds', 'wins', 'win_rate'])
                writer.writeheader()
                for month, data in sorted(monthly.items()):
                    wr = (data['wins'] / data['rounds'] * 100) if data['rounds'] > 0 else 0
                    writer.writerow({
                        'month': month,
                        'pnl': round(data['pnl'], 2),
                        'rounds': data['rounds'],
                        'wins': data['wins'],
                        'win_rate': round(wr, 1)
                    })

    # Summary text file
    with open(os.path.join(LOG_DIR, "pyramid_v3_summary.txt"), 'w') as f:
        f.write("ENHANCED PYRAMID STRATEGY OPTIMIZER v3\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Data: {years} year(s) of tick data\n")
        f.write(f"Parameters: 12 (threshold, trailing, pyramid_step, max_pyramids,\n")
        f.write(f"            poll_interval, size_schedule, acceleration, min_spacing,\n")
        f.write(f"            time_decay, vol_type, vol_min, vol_window)\n\n")

        for item in final_data:
            f.write(f"{item['coin']}:\n")
            f.write(f"  --- Core Parameters ---\n")
            f.write(f"  Threshold:     {item['threshold']:.1f}%\n")
            f.write(f"  Trailing:      {item['trailing']:.1f}%\n")
            f.write(f"  Pyramid Step:  {item['pyramid_step']:.1f}%\n")
            f.write(f"  Max Pyramids:  {item['max_pyramids']}\n")
            f.write(f"  Poll Interval: {item['poll_interval']}\n")
            f.write(f"  --- New Parameters ---\n")
            f.write(f"  Size Schedule: {item['size_schedule']}\n")
            f.write(f"  Acceleration:  {item['acceleration']}\n")
            f.write(f"  Min Spacing:   {item['min_spacing']}%\n")
            f.write(f"  Time Decay:    {item['time_decay']}\n")
            f.write(f"  Vol Type:      {item['vol_type']}\n")
            f.write(f"  Vol Min:       {item['vol_min']}%\n")
            f.write(f"  Vol Window:    {item['vol_window']}\n")
            f.write(f"  --- Results ---\n")
            f.write(f"  Total P&L:     {item['total_pnl']:+.2f}%\n")
            f.write(f"  Win Rate:      {item['win_rate']:.1f}%\n")
            f.write(f"  Sharpe Ratio:  {item['sharpe_ratio']:.2f}\n")
            f.write(f"  Max Drawdown:  {item['max_drawdown']:.2f}%\n")
            f.write(f"  Min Account:   ${item['recommended_account']:.0f} (recommended)\n\n")

    print(f"\n✓ Results saved to {LOG_DIR}/")
    print(f"  - pyramid_v3_recommendations.csv")
    print(f"  - pyramid_v3_summary.txt")
    print(f"  - [COIN]_monthly_breakdown.csv")


def verify_with_raw_ticks(
    coin: str,
    best_params: Dict,
    years: int,
    min_move_pct: float = 0.01,
    verbose: bool = True
) -> Dict:
    """
    Verify best parameters using filtered tick-by-tick data.

    Uses filtered tick streaming that keeps only price moves >= min_move_pct.
    This provides maximum accuracy while being much faster than raw ticks.

    Args:
        coin: Trading pair symbol
        best_params: Best parameters from grid search
        years: Number of years of data
        min_move_pct: Minimum price move % to include (default 0.01% = 1 basis point)
        verbose: Print progress

    Returns:
        Dict with tick-level backtest results for comparison
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"TICK-BY-TICK VERIFICATION: {coin}")
        print(f"{'='*60}")
        print(f"Parameters: T={best_params['threshold']}% Tr={best_params['trailing']}% "
              f"P={best_params['pyramid_step']}% MP={best_params['max_pyramids']}")
        print(f"Filter: {min_move_pct}% minimum move (1 basis point)")

    # Create filtered tick streamer (downloads raw if not cached, then filters)
    tick_streamer = create_filtered_tick_streamer(coin, years, min_move_pct, verbose)

    # Count ticks for reporting
    total_ticks, filtered_ticks = count_filtered_ticks(coin, years, min_move_pct)

    if verbose:
        if total_ticks > 0:
            reduction = (1 - filtered_ticks / total_ticks) * 100
            print(f"\nTick data: {total_ticks:,} raw → {filtered_ticks:,} filtered ({reduction:.1f}% reduction)")
        print(f"Running backtest on {filtered_ticks:,} filtered ticks...")

    start_time = time.time()

    # Get max_pyramids value
    max_pyr = best_params['max_pyramids']
    if max_pyr == 'unlimited':
        max_pyr = 9999

    # Run backtest with filtered ticks
    result = run_pyramid_backtest(
        tick_streamer(),
        threshold_pct=best_params['threshold'],
        trailing_pct=best_params['trailing'],
        pyramid_step_pct=best_params['pyramid_step'],
        max_pyramids=max_pyr,
        verbose=False
    )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n✓ Tick verification completed in {elapsed:.1f} seconds")
        print(f"\n--- TICK-LEVEL RESULTS ---")
        print(f"  Total P&L:     {result['total_pnl']:+.2f}%")
        print(f"  Rounds:        {result['total_rounds']}")
        print(f"  Win Rate:      {result['win_rate']:.1f}%")
        print(f"  Avg P&L:       {result['avg_pnl']:+.2f}%")
        print(f"  Avg Pyramids:  {result['avg_pyramids']:.1f}")

    return {
        'total_pnl': result['total_pnl'],
        'total_rounds': result['total_rounds'],
        'win_rate': result['win_rate'],
        'avg_pnl': result['avg_pnl'],
        'avg_pyramids': result['avg_pyramids'],
        'num_ticks_raw': total_ticks,
        'num_ticks_filtered': filtered_ticks,
        'elapsed_seconds': elapsed
    }


def print_verification_comparison(
    coin: str,
    aggregated_result: Dict,
    tick_result: Dict
):
    """Print comparison between aggregated and tick-level results."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {coin}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'1s Aggregated':>15} {'Tick-by-Tick':>15} {'Difference':>12}")
    print("-" * 62)

    metrics = [
        ('Total P&L', 'total_pnl', '%'),
        ('Rounds', 'total_rounds', ''),
        ('Win Rate', 'win_rate', '%'),
        ('Avg P&L', 'avg_pnl', '%'),
        ('Avg Pyramids', 'avg_pyramids', ''),
    ]

    for label, key, suffix in metrics:
        agg_val = aggregated_result.get(key, 0)
        tick_val = tick_result.get(key, 0)

        if key in ['total_pnl', 'win_rate', 'avg_pnl']:
            diff = tick_val - agg_val
            diff_str = f"{diff:+.2f}{suffix}"
            agg_str = f"{agg_val:.2f}{suffix}"
            tick_str = f"{tick_val:.2f}{suffix}"
        else:
            diff = tick_val - agg_val
            diff_str = f"{diff:+.1f}"
            agg_str = f"{agg_val:.1f}"
            tick_str = f"{tick_val:.1f}"

        print(f"{label:<20} {agg_str:>15} {tick_str:>15} {diff_str:>12}")

    print("-" * 62)

    # Summary assessment
    pnl_diff_pct = abs(tick_result['total_pnl'] - aggregated_result['total_pnl'])
    if pnl_diff_pct < 1:
        print("✓ Results are consistent (< 1% difference in total P&L)")
    elif pnl_diff_pct < 5:
        print("⚠ Minor difference (1-5% in total P&L)")
    else:
        print("⚠ Significant difference (> 5% in total P&L) - tick data may reveal different patterns")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Pyramid Strategy Optimizer v3 (12 Parameters, Memory-Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with defaults
  python optimize_pyramid_v2.py --quick-test --coins BTCUSDT

  # Custom parameter ranges
  python optimize_pyramid_v2.py --acceleration=1.0,1.1,1.2,1.3 --time-decay=None,300,600

  # Single coin, full grid
  python optimize_pyramid_v2.py --coins BTCUSDT
        """
    )
    parser.add_argument("--coins", type=str, default=None, help="Comma-separated coins to test")
    parser.add_argument("--quick-test", action="store_true", help="Use reduced grid for quick testing")
    parser.add_argument("--verify-ticks", action="store_true",
                        help="Verify best results with filtered tick-by-tick data (maximum accuracy)")
    parser.add_argument("--ticks-only", action="store_true",
                        help="Skip grid search, run ONLY with filtered tick data")
    parser.add_argument("--tick-filter", type=float, default=0.01,
                        help="Minimum price move %% to include in tick data (default: 0.01%% = 1 basis point)")

    # New parameter CLI args (allow overriding default grids)
    parser.add_argument("--size-schedule", type=str, default=None,
                        help="Pyramid size schedules (comma-separated). Default: fixed,linear_decay,exp_decay")
    parser.add_argument("--acceleration", type=str, default=None,
                        help="Pyramid acceleration values (comma-separated). Default: 1.0,1.2,1.5")
    parser.add_argument("--min-spacing", type=str, default=None,
                        help="Minimum pyramid spacing %% (comma-separated). Default: 0.0,1.0,2.0")
    parser.add_argument("--time-decay", type=str, default=None,
                        help="Time decay exit seconds (comma-separated, use 'None' for disabled). Default: None,300,900")
    parser.add_argument("--vol-type", type=str, default=None,
                        help="Volatility filter types (comma-separated). Default: none,stddev,range")
    parser.add_argument("--vol-min", type=str, default=None,
                        help="Volatility minimum %% thresholds (comma-separated). Default: 0.0,0.5,1.0")
    parser.add_argument("--vol-window", type=str, default=None,
                        help="Volatility window sizes (comma-separated). Default: 50,100,200")

    args = parser.parse_args()

    # Build parameter overrides from CLI args
    param_overrides = {}

    if args.size_schedule:
        param_overrides['size_schedule'] = args.size_schedule.split(',')

    if args.acceleration:
        param_overrides['acceleration'] = [float(x) for x in args.acceleration.split(',')]

    if args.min_spacing:
        param_overrides['min_spacing'] = [float(x) for x in args.min_spacing.split(',')]

    if args.time_decay:
        param_overrides['time_decay'] = [
            None if x.lower() == 'none' else int(x)
            for x in args.time_decay.split(',')
        ]

    if args.vol_type:
        param_overrides['vol_type'] = args.vol_type.split(',')

    if args.vol_min:
        param_overrides['vol_min'] = [float(x) for x in args.vol_min.split(',')]

    if args.vol_window:
        param_overrides['vol_window'] = [int(x) for x in args.vol_window.split(',')]

    print("=" * 70)
    print("ENHANCED PYRAMID STRATEGY OPTIMIZER v3")
    print("(12 Parameters, Memory-Optimized)")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()

    # Prompt for years of data
    years = get_years_from_user()

    # Parse coins
    coins = args.coins.split(",") if args.coins else DEFAULT_COINS

    # Calculate grid size
    if args.quick_test:
        base_combos = (len(THRESHOLDS_QUICK) * len(TRAILINGS_QUICK) * len(PYRAMID_STEPS_QUICK) *
                       len(MAX_PYRAMIDS_QUICK) * len(POLL_INTERVALS_QUICK))
        new_combos = (len(SIZE_SCHEDULES_QUICK) * len(ACCELERATIONS_QUICK) * len(MIN_SPACINGS_QUICK) *
                      len(TIME_DECAYS_QUICK) * len(VOL_TYPES_QUICK) * len(VOL_MINS_QUICK) * len(VOL_WINDOWS_QUICK))
        poll_intervals = POLL_INTERVALS_QUICK
        print("\n[QUICK TEST MODE]")
    else:
        base_combos = (len(THRESHOLDS) * len(TRAILINGS) * len(PYRAMID_STEPS) *
                       len(MAX_PYRAMIDS) * len(POLL_INTERVALS))
        new_combos = (len(SIZE_SCHEDULES) * len(ACCELERATIONS) * len(MIN_SPACINGS) *
                      len(TIME_DECAYS) * len(VOL_TYPES) * len(VOL_MINS) * len(VOL_WINDOWS))
        poll_intervals = POLL_INTERVALS

    total_combos = base_combos * new_combos

    poll_labels = ['tick' if p == 0 else f'{p}s' for p in poll_intervals]
    print(f"\nConfiguration:")
    print(f"  Coins:          {', '.join(coins)}")
    print(f"  Data:           {years} year(s) of tick data")
    print(f"  Grid:           {total_combos:,} combinations per coin")
    print(f"  Parameters:     12 (5 original + 7 new)")
    print(f"  Poll Intervals: {poll_labels}")
    print(f"  Memory Mode:    Streaming from memory cache")
    if param_overrides:
        print(f"  Overrides:      {list(param_overrides.keys())}")
    print("=" * 70)

    ensure_dirs()

    # Phase 1: Download and cache all tick data first (sequential to avoid OOM)
    print("\n" + "=" * 70)
    print("PHASE 1: DOWNLOADING & CACHING TICK DATA")
    print("=" * 70)
    print(f"Tick filter: {args.tick_filter}% minimum move")

    valid_coins = []
    for i, coin in enumerate(coins, 1):
        print(f"\n[{i}/{len(coins)}] {coin}...")
        try:
            # This downloads raw ticks and creates a filtered streamer
            tick_streamer = create_filtered_tick_streamer(
                coin, years=years, min_move_pct=args.tick_filter, verbose=True
            )
            total_ticks, filtered_ticks = count_filtered_ticks(coin, years, args.tick_filter)

            if filtered_ticks > 1000:
                valid_coins.append((coin, tick_streamer, filtered_ticks))
                reduction = (1 - filtered_ticks / total_ticks) * 100 if total_ticks > 0 else 0
                print(f"  ✓ Ready: {filtered_ticks:,} filtered ticks ({reduction:.1f}% reduction from {total_ticks:,} raw)")
            else:
                print(f"  ⚠ Insufficient data for {coin}, skipping")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if not valid_coins:
        print("\nERROR: No data available!")
        return

    print(f"\n✓ {len(valid_coins)} coins ready for optimization")

    # Phase 2: Process each coin sequentially (memory-efficient)
    print("\n" + "=" * 70)
    print("PHASE 2: GRID SEARCH & OPTIMIZATION")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    for i, (coin, tick_streamer, num_ticks) in enumerate(valid_coins, 1):
        print(f"\n[{i}/{len(valid_coins)}] Processing {coin}...")

        # Process this coin (streams from memory, minimal RAM)
        result = process_coin(
            coin, tick_streamer, num_ticks,
            quick_test=args.quick_test,
            param_overrides=param_overrides if param_overrides else None
        )
        all_results[coin] = result

        # Explicit garbage collection between coins to release any lingering memory
        gc.collect()
        print(f"  ✓ Memory released for {coin}")

    total_time = time.time() - total_start

    # Final report
    print_final_report(all_results, years)

    # Phase 3: Tick-by-tick verification (if requested)
    if args.verify_ticks or args.ticks_only:
        print("\n" + "=" * 70)
        print("PHASE 3: TICK-BY-TICK VERIFICATION")
        print("=" * 70)
        print(f"Verifying best parameters with filtered tick data")
        print(f"Filter: {args.tick_filter}% minimum move (keeps significant price changes)")
        print("This provides maximum simulation accuracy.\n")

        tick_results = {}
        for coin, result in all_results.items():
            try:
                tick_result = verify_with_raw_ticks(
                    coin,
                    result.best_result,
                    years,
                    min_move_pct=args.tick_filter,
                    verbose=True
                )
                tick_results[coin] = tick_result

                # Print comparison
                print_verification_comparison(
                    coin,
                    result.best_result,
                    tick_result
                )

                gc.collect()
            except Exception as e:
                print(f"  ✗ Error verifying {coin}: {e}")

        # Save tick verification results
        if tick_results:
            tick_file = os.path.join(LOG_DIR, "pyramid_v2_tick_verification.csv")
            with open(tick_file, 'w', newline='') as f:
                fieldnames = ['coin', 'total_pnl_1s', 'total_pnl_tick', 'pnl_diff',
                              'rounds_1s', 'rounds_tick', 'win_rate_1s', 'win_rate_tick',
                              'ticks_raw', 'ticks_filtered', 'tick_filter_pct', 'verification_time_sec']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for coin, tick_res in tick_results.items():
                    agg_res = all_results[coin].best_result
                    writer.writerow({
                        'coin': coin,
                        'total_pnl_1s': round(agg_res['total_pnl'], 2),
                        'total_pnl_tick': round(tick_res['total_pnl'], 2),
                        'pnl_diff': round(tick_res['total_pnl'] - agg_res['total_pnl'], 2),
                        'rounds_1s': agg_res.get('rounds', agg_res.get('total_rounds', 0)),
                        'rounds_tick': tick_res['total_rounds'],
                        'win_rate_1s': round(agg_res['win_rate'], 1),
                        'win_rate_tick': round(tick_res['win_rate'], 1),
                        'ticks_raw': tick_res['num_ticks_raw'],
                        'ticks_filtered': tick_res['num_ticks_filtered'],
                        'tick_filter_pct': args.tick_filter,
                        'verification_time_sec': round(tick_res['elapsed_seconds'], 1)
                    })

            print(f"\n✓ Tick verification saved to {tick_file}")

    print()
    print("=" * 70)
    print(f"Total runtime: {total_time/3600:.1f} hours")
    print(f"Completed: {datetime.now()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
