#!/usr/bin/env python3
"""
Autonomous Single-Coin Pyramid Strategy Optimizer v3

Features:
- Interactive coin selection at startup
- Iterative convergence: narrows parameters each round until improvement < 0.1%
- Boundary extension: extends search if best is at parameter edge
- Crash recovery: saves checkpoint after each round
- Final output: absolute best parameters for live trading

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
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
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
TOP_N = 10

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
# ROUND 1: COARSE GRID (Aggressive - ~17M combinations)
# =============================================================================

def build_coarse_grid() -> Dict[str, List]:
    """
    Generate Round 1 aggressive coarse parameter grid.

    Target: ~15-20 million combinations
    - Core 5 params: ~10K combinations
    - New 7 params: ~1.5K combinations
    - Total: ~15M combinations = 12-24 hours
    """
    return {
        # Core 5 parameters (8×8×6×5×5 = 9,600)
        'threshold': [1, 3, 5, 7, 10, 13, 16, 20],
        'trailing': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        'pyramid_step': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'max_pyramids': [5, 20, 80, 320, 9999],
        'poll_interval': [0, 0.2, 0.8, 3.2, 6.4],
        # New 7 parameters (2×4×3×3×2×3×3 = 1,296)
        'size_schedule': ['fixed', 'linear_decay'],  # exp_decay added in refinement if linear_decay wins
        'acceleration': [0.8, 1.0, 1.2, 1.5],
        'min_spacing': [0.0, 1.0, 2.5],
        'time_decay': [None, 300, 900],
        'vol_type': ['none', 'stddev'],  # range added in refinement if stddev wins
        'vol_min': [0.0, 0.5, 1.0],
        'vol_window': [50, 100, 200],
    }
    # Total: 9,600 × 1,296 = 12,441,600 combinations (~12.4M)


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
# GRID NARROWING
# =============================================================================

def build_narrowed_grid(
    round_num: int,
    best_params: Dict,
    previous_ranges: Dict
) -> Dict[str, List]:
    """
    Narrow parameter space around best from previous round.

    Each round narrows by roughly 50% and uses finer step sizes.
    """
    narrowed = {}

    # Narrowing factor decreases each round
    # Round 2: ±40% of range, Round 3: ±25%, Round 4+: ±15%
    narrow_pct = max(0.15, 0.5 / round_num)

    for param, values in previous_ranges.items():
        # Handle categorical parameters - keep all or just the best
        if param in ['size_schedule', 'vol_type']:
            # Keep best + maybe one alternative
            best_val = best_params.get(param)
            if best_val in values:
                narrowed[param] = [best_val]
            else:
                narrowed[param] = values[:1]
            continue

        # Handle None-containing parameters (time_decay)
        if param == 'time_decay':
            best_val = best_params.get('time_decay')
            if best_val is None:
                # Best was no time decay - test nearby values or keep None
                narrowed[param] = [None, 60, 120]
            else:
                # Best was a specific value - narrow around it
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
# BOUNDARY DETECTION & EXTENSION
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
        if param == 'time_decay' and best_params.get(param) is None:
            continue

        best_val = best_params.get(param)
        if best_val is None:
            continue

        # Handle 'unlimited'
        if best_val == 'unlimited':
            best_val = 9999

        # Get numeric values
        numeric_vals = sorted([v for v in values if v is not None and isinstance(v, (int, float))])
        if len(numeric_vals) < 2:
            continue

        # Check if at min or max
        if best_val <= numeric_vals[0]:
            boundaries.append((param, 'min', best_val))
        elif best_val >= numeric_vals[-1]:
            boundaries.append((param, 'max', best_val))

    return boundaries


def extend_range(
    param_ranges: Dict,
    boundaries: List[Tuple[str, str, float]]
) -> Dict[str, List]:
    """
    Extend parameter ranges where best was at boundary.
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
        print(f"    Extended {param}: {values} → {extended[param]}")

    return extended


# =============================================================================
# GRID SEARCH
# =============================================================================

def run_grid_search(
    coin: str,
    tick_streamer: Callable,
    param_grid: Dict,
    round_num: int
) -> Tuple[str, List[Dict]]:
    """
    Run grid search over all parameter combinations.

    Returns (log_file_path, top_N_results)
    """
    # Calculate total combinations
    total = 1
    for values in param_grid.values():
        total *= len(values)

    print(f"\n  Testing {total:,} combinations...")

    top_results = []

    log_file = os.path.join(LOG_DIR, f"{coin}_round{round_num}_grid.csv")

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
                                                            # Aggregate tick data
                                                            price_stream = aggregate_ticks_to_interval(
                                                                tick_streamer,
                                                                poll_interval
                                                            )

                                                            # Run backtest
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

                                                            # Write to disk
                                                            writer.writerow(entry)
                                                            f.flush()

                                                            # Track top N
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
# CHECKPOINT SAVE/LOAD
# =============================================================================

def save_checkpoint(
    coin: str,
    round_num: int,
    best_params: Dict,
    best_pnl: float,
    round_history: List[Dict],
    current_ranges: Dict
) -> None:
    """Save checkpoint for crash recovery."""
    checkpoint = {
        'coin': coin,
        'current_round': round_num,
        'best_pnl': best_pnl,
        'best_params': best_params,
        'round_history': round_history,
        'current_param_ranges': current_ranges,
        'timestamp': datetime.now().isoformat()
    }

    checkpoint_file = os.path.join(LOG_DIR, f"{coin}_checkpoint.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)

    print(f"  ✓ Checkpoint saved: {checkpoint_file}")


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
# ITERATIVE CONVERGENCE
# =============================================================================

def run_iterative_optimization(
    coin: str,
    tick_streamer: Callable,
    max_rounds: int = MAX_ROUNDS,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    resume_from: Optional[Dict] = None
) -> Tuple[Dict, List[Dict]]:
    """
    Run iterative optimization until convergence.

    Returns (best_params, round_history)
    """
    # Initialize or resume
    if resume_from:
        round_history = resume_from['round_history']
        best_params = resume_from['best_params']
        best_pnl = resume_from['best_pnl']
        current_ranges = resume_from['current_param_ranges']
        start_round = resume_from['current_round'] + 1
        print(f"\n  Resuming from round {start_round} (best P&L: {best_pnl:+.2f}%)")
    else:
        round_history = []
        best_params = None
        best_pnl = float('-inf')
        current_ranges = build_coarse_grid()
        start_round = 1

    for round_num in range(start_round, max_rounds + 1):
        print(f"\n{'=' * 70}")
        print(f"ROUND {round_num} / {max_rounds}")
        print(f"{'=' * 70}")

        # Calculate grid size
        grid_size = 1
        for values in current_ranges.values():
            grid_size *= len(values)
        print(f"  Grid size: {grid_size:,} combinations")

        # Run grid search
        start_time = time.time()
        log_file, top_results = run_grid_search(coin, tick_streamer, current_ranges, round_num)
        elapsed = time.time() - start_time

        if not top_results:
            print("  ERROR: No results from grid search!")
            break

        # Get best from this round
        round_best = top_results[0]
        round_pnl = round_best['total_pnl']

        print(f"\n  Round {round_num} completed in {elapsed/3600:.2f} hours")
        print(f"  Best P&L this round: {round_pnl:+.2f}%")
        print(f"  Top 3:")
        for i, r in enumerate(top_results[:3], 1):
            print(f"    #{i}: T={r['threshold']}% Tr={r['trailing']}% → {r['total_pnl']:+.2f}%")

        # Calculate improvement
        if best_pnl > float('-inf'):
            improvement = round_pnl - best_pnl
            print(f"  Improvement: {improvement:+.2f}%")
        else:
            improvement = float('inf')

        # Update best if improved
        if round_pnl > best_pnl:
            best_pnl = round_pnl
            best_params = round_best

        # Record round history
        round_history.append({
            'round': round_num,
            'best_pnl': round_pnl,
            'improvement': improvement if improvement != float('inf') else None,
            'combos_tested': grid_size,
            'elapsed_hours': elapsed / 3600,
            'params': round_best
        })

        # Save checkpoint
        save_checkpoint(coin, round_num, best_params, best_pnl, round_history, current_ranges)

        # Check convergence
        if round_num > 1 and improvement < convergence_threshold:
            print(f"\n  ✓ CONVERGED: Improvement ({improvement:+.2f}%) < threshold ({convergence_threshold}%)")
            break

        # Check for boundaries
        boundaries = check_boundaries(round_best, current_ranges)
        if boundaries:
            print(f"\n  ⚠ Best at boundary for: {[b[0] for b in boundaries]}")
            current_ranges = extend_range(current_ranges, boundaries)

        # Narrow for next round
        if round_num < max_rounds:
            current_ranges = build_narrowed_grid(round_num + 1, best_params, current_ranges)
            next_size = 1
            for values in current_ranges.values():
                next_size *= len(values)
            print(f"\n  Next round grid: {next_size:,} combinations")

        # Force garbage collection
        gc.collect()

    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION COMPLETE after {round_num} rounds")
    print(f"Best P&L: {best_pnl:+.2f}%")
    print(f"{'=' * 70}")

    return best_params, round_history


# =============================================================================
# FINAL OUTPUT
# =============================================================================

def print_final_recommendation(coin: str, best_params: Dict, analytics: Dict = None):
    """Print final recommendation in ready-to-use format."""
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION COMPLETE: {coin}")
    print(f"{'=' * 70}")

    print(f"\nABSOLUTE BEST PARAMETERS:")
    print(f"  threshold_pct:      {best_params['threshold']}%")
    print(f"  trailing_pct:       {best_params['trailing']}%")
    print(f"  pyramid_step_pct:   {best_params['pyramid_step']}%")
    print(f"  max_pyramids:       {best_params['max_pyramids']}")
    print(f"  poll_interval:      {best_params['poll_interval']}")
    print(f"  size_schedule:      {best_params['size_schedule']}")
    print(f"  acceleration:       {best_params['acceleration']}")
    print(f"  min_spacing:        {best_params['min_spacing']}%")
    print(f"  time_decay:         {best_params['time_decay']}")
    print(f"  vol_type:           {best_params['vol_type']}")
    print(f"  vol_min:            {best_params['vol_min']}%")
    print(f"  vol_window:         {best_params['vol_window']}")

    print(f"\nPERFORMANCE:")
    print(f"  Total P&L:          {best_params['total_pnl']:+.2f}%")
    print(f"  Win Rate:           {best_params['win_rate']:.1f}%")
    print(f"  Avg P&L per Round:  {best_params['avg_pnl']:+.2f}%")
    print(f"  Avg Pyramids:       {best_params['avg_pyramids']:.1f}")

    if analytics:
        print(f"  Max Drawdown:       {analytics.get('max_drawdown', 'N/A')}%")
        print(f"  Sharpe Ratio:       {analytics.get('sharpe_ratio', 'N/A')}")

    # Parse poll interval for command
    poll = best_params['poll_interval']
    if poll == 'tick':
        poll_val = 0
    else:
        poll_val = float(poll.replace('s', ''))

    max_pyr = best_params['max_pyramids']
    if max_pyr == 'unlimited':
        max_pyr = 9999

    print(f"\nLIVE TRADING COMMAND:")
    print(f"  python main.py --mode trading --symbol {coin} \\")
    print(f"    --threshold {best_params['threshold']} --trailing {best_params['trailing']} \\")
    print(f"    --pyramid {best_params['pyramid_step']} --max-pyramids {max_pyr}")

    print(f"{'=' * 70}")


def save_final_result(coin: str, best_params: Dict, round_history: List[Dict]):
    """Save final result to JSON and CSV."""
    # JSON with full details
    result = {
        'coin': coin,
        'best_params': best_params,
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
        f.write(f"# Generated: {datetime.now()}\n\n")
        for key, value in best_params.items():
            f.write(f"{key}={value}\n")

    # Round history CSV
    csv_file = os.path.join(LOG_DIR, f"{coin}_round_history.csv")
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['round', 'best_pnl', 'improvement', 'combos_tested', 'elapsed_hours']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rh in round_history:
            writer.writerow({
                'round': rh['round'],
                'best_pnl': rh['best_pnl'],
                'improvement': rh.get('improvement'),
                'combos_tested': rh['combos_tested'],
                'elapsed_hours': round(rh['elapsed_hours'], 2)
            })

    print(f"\n✓ Results saved:")
    print(f"  - {json_file}")
    print(f"  - {txt_file}")
    print(f"  - {csv_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("AUTONOMOUS PYRAMID STRATEGY OPTIMIZER v3")
    print("(Single-Coin Iterative Convergence)")
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
        print(f"\n⚠ Found checkpoint from {checkpoint['timestamp']}")
        print(f"  Round: {checkpoint['current_round']}")
        print(f"  Best P&L: {checkpoint['best_pnl']:+.2f}%")
        resume = input("Resume from checkpoint? [Y/n]: ").strip().lower()
        if resume != 'n':
            resume_from = checkpoint
        else:
            print("Starting fresh (checkpoint will be overwritten)")

    # 5. Download/cache tick data
    print(f"\n{'=' * 70}")
    print("PHASE 1: LOADING TICK DATA")
    print(f"{'=' * 70}")

    try:
        tick_streamer = create_filtered_tick_streamer(coin, years=years, verbose=True)
        total_ticks, filtered_ticks = count_filtered_ticks(coin, years, min_move_pct=0.01)

        if filtered_ticks < 1000:
            print(f"ERROR: Insufficient data for {coin} ({filtered_ticks} ticks)")
            return

        reduction = (1 - filtered_ticks / total_ticks) * 100 if total_ticks > 0 else 0
        print(f"\n✓ Loaded: {filtered_ticks:,} filtered ticks ({reduction:.1f}% reduction)")

    except Exception as e:
        print(f"ERROR: Failed to load tick data: {e}")
        return

    # 6. Run iterative optimization
    print(f"\n{'=' * 70}")
    print("PHASE 2: ITERATIVE OPTIMIZATION")
    print(f"{'=' * 70}")
    print(f"Max rounds: {MAX_ROUNDS}")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD}%")

    total_start = time.time()

    best_params, round_history = run_iterative_optimization(
        coin=coin,
        tick_streamer=tick_streamer,
        max_rounds=MAX_ROUNDS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        resume_from=resume_from
    )

    total_time = time.time() - total_start

    # 7. Save and display final results
    print(f"\n{'=' * 70}")
    print("PHASE 3: FINAL RESULTS")
    print(f"{'=' * 70}")

    save_final_result(coin, best_params, round_history)
    print_final_recommendation(coin, best_params)

    print(f"\nTotal optimization time: {total_time/3600:.1f} hours")
    print(f"Completed: {datetime.now()}")

    # Cleanup checkpoint (optimization complete)
    checkpoint_file = os.path.join(LOG_DIR, f"{coin}_checkpoint.json")
    if os.path.exists(checkpoint_file):
        os.rename(checkpoint_file, checkpoint_file.replace('.json', '_completed.json'))
        print(f"\n✓ Checkpoint archived (optimization complete)")


if __name__ == "__main__":
    main()
