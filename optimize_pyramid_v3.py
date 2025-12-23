#!/usr/bin/env python3
"""
Autonomous Single-Coin Pyramid Strategy Optimizer v3 - REVISED

Features:
- Interactive coin selection at startup
- Walk-forward validation (train on 80%, validate on 20%)
- Multi-basin search: explores top 3 DIVERSE parameter regions
- Iterative convergence with proper boundary extension
- Crash recovery with checkpoints
- Overfitting detection and warnings

Usage:
    python optimize_pyramid_v3.py

    # When prompted, select coin and years of data
    # Optimizer runs autonomously for hours/days until convergence
"""

import os
import sys
import csv
import json
import time
import gc
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.tick_data_fetcher import (
    get_years_from_user,
    create_filtered_tick_streamer,
    count_filtered_ticks,
    aggregate_ticks_to_interval
)
from backtest_pyramid import run_pyramid_backtest, PyramidRound
from analytics import (
    RoundSummary,
    generate_analytics_summary,
    print_analytics_summary
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "XLMUSDT"]

# Convergence settings
MAX_ROUNDS = 10
CONVERGENCE_THRESHOLD = 0.1  # Stop when improvement < 0.1%
NUM_BASINS = 3  # Number of diverse parameter regions to explore
MIN_DIVERSITY_DISTANCE = 0.15  # Minimum normalized distance between basins
TOP_N = 100  # Keep top N results for diversity selection

# Walk-forward settings
TRAIN_RATIO = 0.8  # 80% train, 20% validation

LOG_DIR = "logs"

# Parameter limits (absolute bounds for safety)
PARAM_LIMITS = {
    'threshold': (0.5, 50.0),
    'trailing': (0.2, 30.0),
    'pyramid_step': (0.2, 20.0),
    'max_pyramids': (1, 9999),
    'poll_interval': (0, 60.0),
    'acceleration': (0.3, 5.0),
    'min_spacing': (0.0, 10.0),
    'time_decay': (30, 7200),  # None handled separately
    'vol_min': (0.0, 10.0),
    'vol_window': (10, 1000),
}


# =============================================================================
# ROUND 1: COARSE GRID (All Options - ~28M combinations)
# =============================================================================

def build_coarse_grid() -> Dict[str, List]:
    """
    Generate Round 1 aggressive coarse parameter grid with ALL options.

    Includes exp_decay and range that were previously missing.
    """
    return {
        # Core 5 parameters (8x8x6x5x5 = 9,600)
        'threshold': [1, 3, 5, 7, 10, 13, 16, 20],
        'trailing': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        'pyramid_step': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'max_pyramids': [5, 20, 80, 320, 9999],
        'poll_interval': [0, 0.2, 0.8, 3.2, 6.4],
        # New 7 parameters - ALL OPTIONS (3x4x3x3x3x3x3 = 2,916)
        'size_schedule': ['fixed', 'linear_decay', 'exp_decay'],  # All 3 options
        'acceleration': [0.8, 1.0, 1.2, 1.5],
        'min_spacing': [0.0, 1.0, 2.5],
        'time_decay': [None, 300, 900],
        'vol_type': ['none', 'stddev', 'range'],  # All 3 options
        'vol_min': [0.0, 0.5, 1.0],
        'vol_window': [50, 100, 200],
    }
    # Total: 9,600 x 2,916 = 27,993,600 combinations (~28M)


def calculate_grid_size(param_grid: Dict[str, List]) -> int:
    """Calculate total combinations in a parameter grid."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


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

        # Check if user entered a number
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(DEFAULT_COINS):
                return DEFAULT_COINS[idx]
            print(f"Invalid number. Enter 1-{len(DEFAULT_COINS)}")
            continue

        # Check if user entered a valid coin symbol
        if user_input in DEFAULT_COINS:
            return user_input

        # Check if it's a valid Binance futures symbol
        if user_input.endswith("USDT") and len(user_input) >= 5:
            confirm = input(f"'{user_input}' not in default list. Use anyway? [y/N]: ").strip().lower()
            if confirm == 'y':
                return user_input

        print(f"Invalid coin. Enter a symbol like BTCUSDT or a number 1-{len(DEFAULT_COINS)}")


# =============================================================================
# WALK-FORWARD DATA SPLIT
# =============================================================================

def split_data_walk_forward(
    tick_streamer: Callable,
    train_ratio: float = TRAIN_RATIO
) -> Tuple[List[Tuple[datetime, float]], List[Tuple[datetime, float]]]:
    """
    Split tick data into train and validation sets for walk-forward validation.

    Args:
        tick_streamer: Function that returns an iterator of (timestamp, price) tuples
        train_ratio: Fraction of data for training (default 0.8 = 80%)

    Returns:
        (train_ticks, val_ticks) - Lists of (timestamp, price) tuples
    """
    print("  Loading all tick data for train/validation split...")
    all_ticks = list(tick_streamer())
    total = len(all_ticks)

    if total == 0:
        raise ValueError("No tick data available")

    split_idx = int(total * train_ratio)
    train_ticks = all_ticks[:split_idx]
    val_ticks = all_ticks[split_idx:]

    # Get date ranges
    train_start = train_ticks[0][0] if train_ticks else None
    train_end = train_ticks[-1][0] if train_ticks else None
    val_start = val_ticks[0][0] if val_ticks else None
    val_end = val_ticks[-1][0] if val_ticks else None

    print(f"  Train: {len(train_ticks):,} ticks ({train_start} to {train_end})")
    print(f"  Validation: {len(val_ticks):,} ticks ({val_start} to {val_end})")

    return train_ticks, val_ticks


def create_tick_iterator(ticks: List[Tuple[datetime, float]]) -> Callable:
    """Create a tick streamer function from a list of ticks."""
    def streamer():
        for tick in ticks:
            yield tick
    return streamer


# =============================================================================
# DIVERSITY METRIC FOR MULTI-BASIN SELECTION
# =============================================================================

def param_distance(config1: Dict, config2: Dict) -> float:
    """
    Calculate normalized distance between two configurations.

    Uses key numeric parameters to measure how different two configs are.
    Returns value between 0 (identical) and 1 (maximally different).
    """
    distance = 0
    params_to_compare = ['threshold', 'trailing', 'pyramid_step', 'max_pyramids']

    for param in params_to_compare:
        val1 = config1.get(param, 0)
        val2 = config2.get(param, 0)

        # Handle 'unlimited' max_pyramids
        if val1 == 'unlimited':
            val1 = 9999
        if val2 == 'unlimited':
            val2 = 9999

        # Convert poll_interval labels
        if param == 'poll_interval':
            if isinstance(val1, str):
                val1 = 0 if val1 == 'tick' else float(val1.replace('s', ''))
            if isinstance(val2, str):
                val2 = 0 if val2 == 'tick' else float(val2.replace('s', ''))

        limits = PARAM_LIMITS.get(param, (0, 100))
        range_span = limits[1] - limits[0]
        if range_span > 0:
            diff = abs(float(val1) - float(val2)) / range_span
            distance += diff

    return distance / len(params_to_compare)


def select_diverse_top_n(
    results: List[Dict],
    n: int = NUM_BASINS,
    min_distance: float = MIN_DIVERSITY_DISTANCE
) -> List[Dict]:
    """
    Select top N configurations that are sufficiently different from each other.

    This prevents all basins from being nearly identical parameter sets.

    Args:
        results: List of result dicts sorted by P&L descending
        n: Number of diverse configurations to select
        min_distance: Minimum normalized distance between selected configs

    Returns:
        List of n diverse configurations (or fewer if not enough diversity exists)
    """
    if not results:
        return []

    selected = [results[0]]  # Best always included

    for config in results[1:]:
        if len(selected) >= n:
            break
        # Check distance to all already selected configs
        is_diverse = all(
            param_distance(config, s) >= min_distance
            for s in selected
        )
        if is_diverse:
            selected.append(config)

    # If we couldn't find enough diverse configs, fill with best remaining
    if len(selected) < n:
        for config in results[1:]:
            if config not in selected:
                selected.append(config)
            if len(selected) >= n:
                break

    return selected


# =============================================================================
# GRID NARROWING WITH DELAYED CATEGORICAL LOCK-IN
# =============================================================================

def build_narrowed_grid(
    round_num: int,
    best_params: Dict,
    previous_ranges: Dict
) -> Dict[str, List]:
    """
    Narrow parameter space around best from previous round.

    Key fixes:
    - Categorical parameters (size_schedule, vol_type) stay unlocked until Round 3
    - Numeric parameters narrow progressively each round
    """
    narrowed = {}

    # Narrowing factor decreases each round
    # Round 2: +/-40% of range, Round 3: +/-25%, Round 4+: +/-15%
    narrow_pct = max(0.15, 0.5 / round_num)

    for param, values in previous_ranges.items():
        # Handle categorical parameters with DELAYED lock-in
        if param in ['size_schedule', 'vol_type']:
            if round_num <= 2:
                # Rounds 1-2: Keep ALL categorical options
                narrowed[param] = values
            elif round_num == 3:
                # Round 3: Keep top 2 (best + one alternative)
                best_val = best_params.get(param)
                alternatives = [v for v in values if v != best_val]
                narrowed[param] = [best_val] + alternatives[:1]
            else:
                # Round 4+: Lock to best only
                best_val = best_params.get(param)
                narrowed[param] = [best_val] if best_val else values[:1]
            continue

        # Handle None-containing parameters (time_decay)
        if param == 'time_decay':
            best_val = best_params.get('time_decay')
            if best_val is None or best_val == 'None':
                # Best was no time decay - test nearby values or keep None
                narrowed[param] = [None, 60, 120]
            else:
                # Best was a specific value - narrow around it
                if isinstance(best_val, str):
                    best_val = int(best_val.replace('s', ''))
                low = max(30, int(best_val * 0.6))
                high = min(7200, int(best_val * 1.5))
                step = max(30, (high - low) // 5)
                narrowed[param] = list(range(low, high + 1, step))
            continue

        # Get best value and previous range
        best_val = best_params.get(param)
        if best_val is None or not values:
            narrowed[param] = values
            continue

        # Handle 'unlimited' max_pyramids
        if param == 'max_pyramids' and best_val == 'unlimited':
            best_val = 9999

        # Handle poll_interval labels
        if param == 'poll_interval' and isinstance(best_val, str):
            best_val = 0 if best_val == 'tick' else float(best_val.replace('s', ''))

        # Calculate numeric range
        numeric_vals = sorted([v for v in values if isinstance(v, (int, float))])
        if len(numeric_vals) < 2:
            narrowed[param] = values
            continue

        range_span = numeric_vals[-1] - numeric_vals[0]
        narrow_range = range_span * narrow_pct

        # Create new range centered on best
        if param in ['max_pyramids', 'vol_window']:
            # Integer parameters
            low = max(int(PARAM_LIMITS.get(param, (1, 9999))[0]), int(best_val - narrow_range))
            high = min(int(PARAM_LIMITS.get(param, (1, 9999))[1]), int(best_val + narrow_range))
            step = max(1, (high - low) // 6)
            narrowed[param] = list(range(low, high + 1, step))
        else:
            # Float parameters
            limits = PARAM_LIMITS.get(param, (0, 100))
            low = max(limits[0], best_val - narrow_range)
            high = min(limits[1], best_val + narrow_range)
            step = (high - low) / 6
            if step > 0:
                narrowed[param] = [round(low + i * step, 3) for i in range(7)]
            else:
                narrowed[param] = [best_val]

    return narrowed


# =============================================================================
# BOUNDARY DETECTION & EXTENSION (FIXED ORDER)
# =============================================================================

def check_boundaries(
    best_params: Dict,
    param_ranges: Dict
) -> List[Tuple[str, str, float]]:
    """
    Check if best parameters are at grid boundaries.

    Returns list of (param_name, 'min'|'max', value) for params at boundaries.
    """
    boundaries = []

    for param, values in param_ranges.items():
        # Skip categorical parameters
        if param in ['size_schedule', 'vol_type']:
            continue

        # Skip time_decay None
        best_val = best_params.get(param)
        if param == 'time_decay' and (best_val is None or best_val == 'None'):
            continue

        if best_val is None:
            continue

        # Handle 'unlimited'
        if best_val == 'unlimited':
            best_val = 9999

        # Handle poll_interval labels
        if param == 'poll_interval' and isinstance(best_val, str):
            best_val = 0 if best_val == 'tick' else float(best_val.replace('s', ''))

        # Get numeric values
        numeric_vals = sorted([v for v in values if v is not None and isinstance(v, (int, float))])
        if len(numeric_vals) < 2:
            continue

        # Check if at min or max
        if float(best_val) <= numeric_vals[0]:
            boundaries.append((param, 'min', best_val))
        elif float(best_val) >= numeric_vals[-1]:
            boundaries.append((param, 'max', best_val))

    return boundaries


def extend_range(
    param_ranges: Dict,
    boundaries: List[Tuple[str, str, float]]
) -> Dict[str, List]:
    """
    Extend parameter ranges where best was at boundary.

    This is now called AFTER narrowing, not before.
    """
    extended = deepcopy(param_ranges)

    for param, direction, value in boundaries:
        if param not in extended:
            continue

        values = extended[param]
        numeric_vals = sorted([v for v in values if v is not None and isinstance(v, (int, float))])

        if len(numeric_vals) < 2:
            continue

        # Calculate step size from existing range
        step = (numeric_vals[-1] - numeric_vals[0]) / (len(numeric_vals) - 1)
        limits = PARAM_LIMITS.get(param, (0, 9999))

        if direction == 'max':
            # Extend upward
            new_max = min(limits[1], numeric_vals[-1] + step * 3)
            new_vals = list(numeric_vals) + [round(numeric_vals[-1] + step * i, 3) for i in range(1, 4)]
            new_vals = [v for v in new_vals if v <= new_max]
        else:
            # Extend downward
            new_min = max(limits[0], numeric_vals[0] - step * 3)
            new_vals = [round(numeric_vals[0] - step * i, 3) for i in range(3, 0, -1)] + list(numeric_vals)
            new_vals = [v for v in new_vals if v >= new_min]

        # Round integers
        if param in ['max_pyramids', 'vol_window']:
            new_vals = [int(v) for v in new_vals]

        extended[param] = sorted(set(new_vals))
        print(f"    Extended {param}: {len(values)} -> {len(extended[param])} values")

    return extended


# =============================================================================
# GRID SEARCH
# =============================================================================

def run_grid_search(
    coin: str,
    train_data: List[Tuple[datetime, float]],
    param_grid: Dict,
    round_num: int,
    basin_id: int = 0
) -> Tuple[str, List[Dict]]:
    """
    Run grid search over all parameter combinations using TRAIN data only.

    Returns (log_file_path, top_N_results)
    """
    total = calculate_grid_size(param_grid)
    print(f"\n  Basin {basin_id}: Testing {total:,} combinations...")

    top_results = []

    log_file = os.path.join(LOG_DIR, f"{coin}_round{round_num}_basin{basin_id}_grid.csv")

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

        # Nested loops over all parameters
        for poll_interval in param_grid['poll_interval']:
            poll_label = "tick" if poll_interval == 0 else f"{poll_interval}s"

            # Pre-aggregate ticks for this poll interval
            tick_iterator = create_tick_iterator(train_data)
            aggregated_prices = list(aggregate_ticks_to_interval(tick_iterator, poll_interval))

            for size_sched in param_grid['size_schedule']:
                for accel in param_grid['acceleration']:
                    for min_space in param_grid['min_spacing']:
                        for time_decay in param_grid['time_decay']:
                            for vol_type in param_grid['vol_type']:
                                for vol_min in param_grid['vol_min']:
                                    for vol_win in param_grid['vol_window']:
                                        for threshold in param_grid['threshold']:
                                            for trailing in param_grid['trailing']:
                                                for pyramid_step in param_grid['pyramid_step']:
                                                    for max_pyr in param_grid['max_pyramids']:
                                                        completed += 1

                                                        # Progress update
                                                        if completed % 1000 == 0 or completed == 1:
                                                            elapsed = time.time() - start
                                                            rate = completed / elapsed if elapsed > 0 else 1
                                                            remaining = (total - completed) / rate
                                                            pct = (completed / total) * 100
                                                            print(f"\r    [{completed:,}/{total:,}] {pct:.1f}% | "
                                                                  f"{remaining/3600:.1f}h remaining    ",
                                                                  end="", flush=True)

                                                        try:
                                                            # Run backtest on pre-aggregated data
                                                            result = run_pyramid_backtest(
                                                                iter(aggregated_prices),
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

                                                            # Write to disk
                                                            writer.writerow(entry)
                                                            f.flush()

                                                            # Track top N for diversity selection
                                                            if len(top_results) < TOP_N:
                                                                top_results.append(entry)
                                                                top_results.sort(key=lambda x: x['total_pnl'], reverse=True)
                                                            elif entry['total_pnl'] > top_results[-1]['total_pnl']:
                                                                top_results[-1] = entry
                                                                top_results.sort(key=lambda x: x['total_pnl'], reverse=True)

                                                        except Exception as e:
                                                            if completed % 10000 == 0:
                                                                print(f"\n    Error: {e}")

    print()  # Newline after progress
    return log_file, top_results


# =============================================================================
# SINGLE BACKTEST FOR VALIDATION
# =============================================================================

def run_single_backtest(
    tick_data: List[Tuple[datetime, float]],
    params: Dict
) -> float:
    """
    Run a single backtest with specific parameters.

    Returns total P&L percentage.
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
    max_pyr = params.get('max_pyramids', 20)
    if max_pyr == 'unlimited':
        max_pyr = 9999

    # Aggregate tick data
    tick_iterator = create_tick_iterator(tick_data)
    price_stream = aggregate_ticks_to_interval(tick_iterator, poll)

    result = run_pyramid_backtest(
        price_stream,
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

    return result['total_pnl']


# =============================================================================
# CHECKPOINT SAVE/LOAD
# =============================================================================

@dataclass
class Basin:
    """Represents a parameter basin being explored."""
    id: int
    best_params: Dict
    best_pnl: float
    current_ranges: Dict
    rounds_without_improvement: int = 0
    converged: bool = False


def save_checkpoint(
    coin: str,
    round_num: int,
    basins: List[Basin],
    round_history: List[Dict]
) -> None:
    """Save checkpoint for crash recovery."""
    checkpoint = {
        'coin': coin,
        'current_round': round_num,
        'basins': [
            {
                'id': b.id,
                'best_params': b.best_params,
                'best_pnl': b.best_pnl,
                'current_ranges': b.current_ranges,
                'rounds_without_improvement': b.rounds_without_improvement,
                'converged': b.converged
            }
            for b in basins
        ],
        'round_history': round_history,
        'timestamp': datetime.now().isoformat()
    }

    checkpoint_file = os.path.join(LOG_DIR, f"{coin}_checkpoint.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)

    print(f"  Checkpoint saved: {checkpoint_file}")


def load_checkpoint(coin: str) -> Optional[Dict]:
    """Load checkpoint if exists."""
    checkpoint_file = os.path.join(LOG_DIR, f"{coin}_checkpoint.json")

    if not os.path.exists(checkpoint_file):
        return None

    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


# =============================================================================
# MULTI-BASIN ITERATIVE OPTIMIZATION
# =============================================================================

def run_multi_basin_optimization(
    coin: str,
    train_data: List[Tuple[datetime, float]],
    num_basins: int = NUM_BASINS,
    max_rounds: int = MAX_ROUNDS,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    resume_from: Optional[Dict] = None
) -> List[Basin]:
    """
    Run iterative optimization across multiple diverse parameter basins.

    This prevents getting stuck in local optima by exploring different
    regions of the parameter space in parallel.

    Returns list of converged basins with their best parameters.
    """
    # Initialize or resume
    if resume_from:
        round_history = resume_from.get('round_history', [])
        basins = [
            Basin(
                id=b['id'],
                best_params=b['best_params'],
                best_pnl=b['best_pnl'],
                current_ranges=b['current_ranges'],
                rounds_without_improvement=b.get('rounds_without_improvement', 0),
                converged=b.get('converged', False)
            )
            for b in resume_from['basins']
        ]
        start_round = resume_from['current_round'] + 1
        print(f"\n  Resuming from round {start_round} with {len(basins)} basins")
    else:
        round_history = []
        basins = []
        start_round = 1

    # Round 1: Coarse grid to find diverse starting points
    if start_round == 1:
        print(f"\n{'=' * 70}")
        print(f"ROUND 1 / {max_rounds} - COARSE GRID (Finding Diverse Basins)")
        print(f"{'=' * 70}")

        coarse_grid = build_coarse_grid()
        grid_size = calculate_grid_size(coarse_grid)
        print(f"  Grid size: {grid_size:,} combinations")

        start_time = time.time()
        log_file, top_results = run_grid_search(coin, train_data, coarse_grid, 1)
        elapsed = time.time() - start_time

        if not top_results:
            print("  ERROR: No results from grid search!")
            return []

        # Select diverse basins
        diverse_configs = select_diverse_top_n(top_results, num_basins, MIN_DIVERSITY_DISTANCE)
        print(f"\n  Round 1 completed in {elapsed/3600:.2f} hours")
        print(f"  Selected {len(diverse_configs)} diverse basins:")

        for i, config in enumerate(diverse_configs):
            basin = Basin(
                id=i,
                best_params=config,
                best_pnl=config['total_pnl'],
                current_ranges=coarse_grid
            )
            basins.append(basin)
            print(f"    Basin {i}: T={config['threshold']}% Tr={config['trailing']}% -> {config['total_pnl']:+.2f}%")

        round_history.append({
            'round': 1,
            'basins': [{'id': b.id, 'pnl': b.best_pnl} for b in basins],
            'combos_tested': grid_size,
            'elapsed_hours': elapsed / 3600
        })

        save_checkpoint(coin, 1, basins, round_history)
        start_round = 2

    # Rounds 2+: Refine each basin until convergence
    for round_num in range(start_round, max_rounds + 1):
        print(f"\n{'=' * 70}")
        print(f"ROUND {round_num} / {max_rounds}")
        print(f"{'=' * 70}")

        active_basins = [b for b in basins if not b.converged]
        if not active_basins:
            print("  All basins converged!")
            break

        print(f"  Active basins: {len(active_basins)} / {len(basins)}")

        round_start = time.time()
        round_combos = 0

        for basin in active_basins:
            previous_pnl = basin.best_pnl

            # 1. NARROW first (fixed order)
            basin.current_ranges = build_narrowed_grid(
                round_num, basin.best_params, basin.current_ranges
            )

            # 2. THEN check boundaries and extend if needed
            boundaries = check_boundaries(basin.best_params, basin.current_ranges)
            if boundaries:
                print(f"  Basin {basin.id}: Best at boundary for {[b[0] for b in boundaries]}")
                basin.current_ranges = extend_range(basin.current_ranges, boundaries)

            # Run grid search
            grid_size = calculate_grid_size(basin.current_ranges)
            round_combos += grid_size

            _, top_results = run_grid_search(
                coin, train_data, basin.current_ranges, round_num, basin.id
            )

            if top_results:
                basin.best_params = top_results[0]
                basin.best_pnl = top_results[0]['total_pnl']

                # Calculate improvement (FIXED: check for true convergence)
                improvement = basin.best_pnl - previous_pnl

                print(f"  Basin {basin.id}: {previous_pnl:+.2f}% -> {basin.best_pnl:+.2f}% "
                      f"(improvement: {improvement:+.2f}%)")

                # FIXED convergence check: only converge on small POSITIVE improvement
                if 0 <= improvement < convergence_threshold:
                    basin.converged = True
                    basin.rounds_without_improvement = 0
                    print(f"    -> CONVERGED (improvement < {convergence_threshold}%)")
                elif improvement < 0:
                    # Deterioration - don't converge, but track it
                    basin.rounds_without_improvement += 1
                    print(f"    -> Deterioration detected, continuing search")
                    if basin.rounds_without_improvement >= 3:
                        basin.converged = True
                        print(f"    -> Giving up after 3 rounds without improvement")
                else:
                    basin.rounds_without_improvement = 0

        round_elapsed = time.time() - round_start
        round_history.append({
            'round': round_num,
            'basins': [{'id': b.id, 'pnl': b.best_pnl, 'converged': b.converged} for b in basins],
            'combos_tested': round_combos,
            'elapsed_hours': round_elapsed / 3600
        })

        save_checkpoint(coin, round_num, basins, round_history)

        # Check if all converged
        if all(b.converged for b in basins):
            print(f"\n  All basins converged!")
            break

        gc.collect()

    print(f"\n{'=' * 70}")
    print(f"MULTI-BASIN OPTIMIZATION COMPLETE after {round_num} rounds")
    print(f"{'=' * 70}")
    for b in basins:
        status = "CONVERGED" if b.converged else "MAX ROUNDS"
        print(f"  Basin {b.id}: {b.best_pnl:+.2f}% [{status}]")

    return basins


# =============================================================================
# VALIDATION AND FINAL OUTPUT
# =============================================================================

def validate_basins(
    basins: List[Basin],
    val_data: List[Tuple[datetime, float]]
) -> List[Dict]:
    """
    Validate each basin's best parameters on holdout data.

    Returns list of validated results with train and validation P&L.
    """
    validated = []

    for basin in basins:
        val_pnl = run_single_backtest(val_data, basin.best_params)
        overfit_ratio = basin.best_pnl / val_pnl if val_pnl > 0 else float('inf')

        validated.append({
            'basin_id': basin.id,
            'params': basin.best_params,
            'train_pnl': basin.best_pnl,
            'val_pnl': val_pnl,
            'overfit_ratio': overfit_ratio
        })

    return validated


def print_final_recommendation(coin: str, validated_results: List[Dict]):
    """Print final recommendation with train vs validation comparison."""
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION COMPLETE: {coin}")
    print(f"{'=' * 70}")

    # Sort by validation P&L (not train!)
    sorted_results = sorted(validated_results, key=lambda x: x['val_pnl'], reverse=True)
    winner = sorted_results[0]

    print(f"\nVALIDATION RESULTS (sorted by validation P&L):")
    for r in sorted_results:
        ratio_warn = " <-- WINNER" if r == winner else ""
        if r['overfit_ratio'] > 2.0:
            ratio_warn += " [HIGH OVERFIT]"
        print(f"  Basin {r['basin_id']}: Train={r['train_pnl']:+.1f}%, Val={r['val_pnl']:+.1f}%, "
              f"Ratio={r['overfit_ratio']:.1f}x{ratio_warn}")

    print(f"\nABSOLUTE BEST PARAMETERS (Validated on Year 5):")
    params = winner['params']
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

    print(f"\nPERFORMANCE:")
    print(f"  Train P&L (Years 1-4):   {winner['train_pnl']:+.2f}%")
    print(f"  Validation P&L (Year 5): {winner['val_pnl']:+.2f}%")
    print(f"  Overfit Ratio:           {winner['overfit_ratio']:.2f}x")

    if winner['overfit_ratio'] > 2.0:
        print(f"\n  WARNING: Train P&L is {winner['overfit_ratio']:.1f}x validation P&L")
        print(f"  This suggests overfitting. Consider using a more robust basin:")
        for r in sorted_results[1:]:
            if r['overfit_ratio'] < winner['overfit_ratio']:
                print(f"    Basin {r['basin_id']}: ratio={r['overfit_ratio']:.1f}x, val={r['val_pnl']:+.1f}%")

    # Parse poll interval for command
    poll = params['poll_interval']
    if poll == 'tick':
        poll_val = 0
    else:
        poll_val = float(poll.replace('s', ''))

    max_pyr = params['max_pyramids']
    if max_pyr == 'unlimited':
        max_pyr = 9999

    print(f"\nLIVE TRADING COMMAND:")
    print(f"  python main.py --mode trading --symbol {coin} \\")
    print(f"    --threshold {params['threshold']} --trailing {params['trailing']} \\")
    print(f"    --pyramid {params['pyramid_step']} --max-pyramids {max_pyr}")

    print(f"{'=' * 70}")

    return winner


def save_final_result(coin: str, validated_results: List[Dict], round_history: List[Dict]):
    """Save final result to JSON and CSV."""
    winner = max(validated_results, key=lambda x: x['val_pnl'])

    # JSON with full details
    result = {
        'coin': coin,
        'winner': winner,
        'all_basins': validated_results,
        'round_history': round_history,
        'completed': datetime.now().isoformat()
    }

    json_file = os.path.join(LOG_DIR, f"{coin}_final_result.json")
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Simple text config for easy reference
    txt_file = os.path.join(LOG_DIR, f"{coin}_live_config.txt")
    with open(txt_file, 'w') as f:
        f.write(f"# {coin} Optimized Parameters\n")
        f.write(f"# Generated: {datetime.now()}\n")
        f.write(f"# Train P&L: {winner['train_pnl']:+.2f}%\n")
        f.write(f"# Validation P&L: {winner['val_pnl']:+.2f}%\n")
        f.write(f"# Overfit Ratio: {winner['overfit_ratio']:.2f}x\n\n")
        for key, value in winner['params'].items():
            f.write(f"{key}={value}\n")

    # Round history CSV
    csv_file = os.path.join(LOG_DIR, f"{coin}_round_history.csv")
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['round', 'basins_status', 'combos_tested', 'elapsed_hours']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rh in round_history:
            writer.writerow({
                'round': rh['round'],
                'basins_status': str(rh['basins']),
                'combos_tested': rh['combos_tested'],
                'elapsed_hours': round(rh['elapsed_hours'], 2)
            })

    print(f"\nResults saved:")
    print(f"  - {json_file}")
    print(f"  - {txt_file}")
    print(f"  - {csv_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("AUTONOMOUS PYRAMID STRATEGY OPTIMIZER v3")
    print("(Multi-Basin Walk-Forward Validation)")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # 1. Interactive coin selection
    coin = select_coin_interactive()
    print(f"\nSelected: {coin}")

    # 2. Get years of data
    years = get_years_from_user()

    # 3. Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # 4. Check for existing checkpoint
    checkpoint = load_checkpoint(coin)
    resume_from = None

    if checkpoint:
        print(f"\nFound checkpoint from {checkpoint['timestamp']}")
        print(f"  Round: {checkpoint['current_round']}")
        print(f"  Basins: {len(checkpoint['basins'])}")
        resume = input("Resume from checkpoint? [Y/n]: ").strip().lower()
        if resume != 'n':
            resume_from = checkpoint
        else:
            print("Starting fresh (checkpoint will be overwritten)")

    # 5. Download/cache tick data and split for walk-forward
    print(f"\n{'=' * 70}")
    print("PHASE 1: LOADING AND SPLITTING DATA")
    print(f"{'=' * 70}")

    try:
        tick_streamer = create_filtered_tick_streamer(coin, years=years, verbose=True)
        train_data, val_data = split_data_walk_forward(tick_streamer, TRAIN_RATIO)

        if len(train_data) < 1000:
            print(f"ERROR: Insufficient training data ({len(train_data)} ticks)")
            return
        if len(val_data) < 100:
            print(f"ERROR: Insufficient validation data ({len(val_data)} ticks)")
            return

    except Exception as e:
        print(f"ERROR: Failed to load tick data: {e}")
        return

    # 6. Run multi-basin optimization on TRAIN data
    print(f"\n{'=' * 70}")
    print("PHASE 2: MULTI-BASIN OPTIMIZATION (Train Data)")
    print(f"{'=' * 70}")
    print(f"Max rounds: {MAX_ROUNDS}")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD}%")
    print(f"Number of basins: {NUM_BASINS}")

    total_start = time.time()

    basins = run_multi_basin_optimization(
        coin=coin,
        train_data=train_data,
        num_basins=NUM_BASINS,
        max_rounds=MAX_ROUNDS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        resume_from=resume_from
    )

    if not basins:
        print("ERROR: No basins produced by optimization")
        return

    # 7. Validate on holdout data
    print(f"\n{'=' * 70}")
    print("PHASE 3: VALIDATION (Year 5 Holdout Data)")
    print(f"{'=' * 70}")

    validated_results = validate_basins(basins, val_data)

    total_time = time.time() - total_start

    # 8. Save and display final results
    print(f"\n{'=' * 70}")
    print("PHASE 4: FINAL RESULTS")
    print(f"{'=' * 70}")

    # Load round history from checkpoint
    checkpoint = load_checkpoint(coin)
    round_history = checkpoint.get('round_history', []) if checkpoint else []

    save_final_result(coin, validated_results, round_history)
    winner = print_final_recommendation(coin, validated_results)

    print(f"\nTotal optimization time: {total_time/3600:.1f} hours")
    print(f"Completed: {datetime.now()}")

    # Cleanup checkpoint (optimization complete)
    checkpoint_file = os.path.join(LOG_DIR, f"{coin}_checkpoint.json")
    if os.path.exists(checkpoint_file):
        os.rename(checkpoint_file, checkpoint_file.replace('.json', '_completed.json'))
        print(f"\nCheckpoint archived (optimization complete)")


if __name__ == "__main__":
    main()
