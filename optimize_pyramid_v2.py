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
    count_filtered_ticks
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

# Grid parameters - FULL
THRESHOLDS = [float(i) for i in range(1, 21)]  # 1-20%
TRAILINGS = [1.0 + 0.2 * i for i in range(21)]  # 1-5% in 0.2 steps
PYRAMID_STEPS = [1.0 + 0.2 * i for i in range(11)]  # 1-3% in 0.2 steps
MAX_PYRAMIDS = [5, 10, 20, 40, 80, 160, 320, 640, 9999]  # 9999 = unlimited

# Grid parameters - QUICK TEST
THRESHOLDS_QUICK = [1, 3, 5, 10, 15]
TRAILINGS_QUICK = [1.0, 2.0, 3.0, 4.0]
PYRAMID_STEPS_QUICK = [1.0, 2.0, 3.0]
MAX_PYRAMIDS_QUICK = [10, 40, 160]

TOP_N = 10
REFINED_COMBOS = 500
LOG_DIR = "logs"


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
    price_streamer: Callable,
    thresholds: List[float],
    trailings: List[float],
    pyramid_steps: List[float],
    max_pyramids_list: List[int]
) -> Tuple[str, List[Dict]]:
    """
    Run grid search over all parameter combinations.

    Memory-efficient: streams prices from disk for each backtest,
    only keeps top N results in memory.

    Args:
        coin: Trading pair symbol
        price_streamer: Callable that returns a generator of (timestamp, price)
        thresholds: List of threshold percentages to test
        trailings: List of trailing stop percentages to test
        pyramid_steps: List of pyramid step percentages to test
        max_pyramids_list: List of max pyramid counts to test

    Returns:
        Tuple of (log_file_path, top_10_results)
    """
    total = len(thresholds) * len(trailings) * len(pyramid_steps) * len(max_pyramids_list)
    top_results = []  # Only keep top N in memory

    log_file = os.path.join(LOG_DIR, f"{coin}_pyramid_v2_grid.csv")

    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'threshold', 'trailing', 'pyramid_step', 'max_pyramids',
            'total_pnl', 'rounds', 'avg_pnl', 'win_rate', 'avg_pyramids'
        ])
        writer.writeheader()

        completed = 0
        start = time.time()

        for threshold in thresholds:
            for trailing in trailings:
                for pyramid_step in pyramid_steps:
                    for max_pyr in max_pyramids_list:
                        completed += 1

                        elapsed = time.time() - start
                        rate = completed / elapsed if elapsed > 0 else 1
                        remaining = (total - completed) / rate

                        if completed % 100 == 0 or completed == 1:
                            pct = (completed / total) * 100
                            print(f"\r  [{completed:,}/{total:,}] {pct:.1f}% | {remaining/3600:.1f}h remaining    ",
                                  end="", flush=True)

                        try:
                            # Stream prices from disk for this backtest
                            result = run_pyramid_backtest(
                                price_streamer(),  # Fresh generator for each backtest
                                threshold_pct=threshold,
                                trailing_pct=trailing,
                                pyramid_step_pct=pyramid_step,
                                max_pyramids=max_pyr,
                                verbose=False
                            )

                            entry = {
                                'threshold': threshold,
                                'trailing': trailing,
                                'pyramid_step': pyramid_step,
                                'max_pyramids': max_pyr if max_pyr < 9999 else 'unlimited',
                                'total_pnl': result['total_pnl'],
                                'rounds': result['total_rounds'],
                                'avg_pnl': result['avg_pnl'],
                                'win_rate': result['win_rate'],
                                'avg_pyramids': result['avg_pyramids']
                            }

                            # Write immediately to disk
                            writer.writerow(entry)
                            f.flush()

                            # Maintain only top N in memory (sorted insert)
                            if len(top_results) < TOP_N:
                                top_results.append(entry)
                                top_results.sort(key=lambda x: x['total_pnl'], reverse=True)
                            elif entry['total_pnl'] > top_results[-1]['total_pnl']:
                                top_results[-1] = entry
                                top_results.sort(key=lambda x: x['total_pnl'], reverse=True)

                        except Exception as e:
                            print(f"\n  Error: {e}")

    print()
    return log_file, top_results


def run_refined_search(
    coin: str,
    price_streamer: Callable,
    top_results: List[Dict]
) -> Tuple[str, Dict]:
    """
    Refine search around top results.

    Memory-efficient: streams prices from disk for each backtest,
    only keeps the single best result in memory.

    Args:
        coin: Trading pair symbol
        price_streamer: Callable that returns a generator of (timestamp, price)
        top_results: Top N results from grid search to refine around

    Returns:
        Tuple of (log_file_path, best_result)
    """
    best_result = None

    log_file = os.path.join(LOG_DIR, f"{coin}_pyramid_v2_refined.csv")

    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'base_threshold', 'base_trailing', 'base_pyramid', 'base_max_pyr',
            'threshold', 'trailing', 'pyramid_step', 'max_pyramids',
            'total_pnl', 'rounds', 'avg_pnl', 'win_rate', 'avg_pyramids'
        ])
        writer.writeheader()

        for rank, top in enumerate(top_results, 1):
            base_t = top['threshold']
            base_tr = top['trailing']
            base_p = top['pyramid_step']
            base_mp = top['max_pyramids']

            # Handle 'unlimited' string
            if base_mp == 'unlimited':
                base_mp = 9999

            print(f"\n  Refining #{rank}: {base_t}% / {base_tr}% / {base_p}% / {base_mp} pyramids")

            # Generate refined grid around this point
            thresholds = [round(base_t + 0.1 * i, 1) for i in range(-5, 6) if base_t + 0.1 * i > 0]
            trailings = [round(base_tr + 0.1 * i, 2) for i in range(-5, 6) if base_tr + 0.1 * i > 0]
            pyramids = [round(base_p + 0.1 * i, 2) for i in range(-5, 6) if base_p + 0.1 * i > 0]

            # For max_pyramids, test ±50% around the value
            if base_mp >= 9999:
                max_pyr_variants = [640, 9999]
            else:
                mp_low = max(5, int(base_mp * 0.5))
                mp_high = min(9999, int(base_mp * 2))
                max_pyr_variants = sorted(set([mp_low, base_mp, mp_high]))

            total = len(thresholds) * len(trailings) * len(pyramids) * len(max_pyr_variants)
            completed = 0

            for threshold in thresholds:
                for trailing in trailings:
                    for pyramid_step in pyramids:
                        for max_pyr in max_pyr_variants:
                            completed += 1

                            if completed % 100 == 0:
                                print(f"\r    [{completed}/{total}]    ", end="", flush=True)

                            try:
                                # Stream prices from disk for this backtest
                                result = run_pyramid_backtest(
                                    price_streamer(),  # Fresh generator for each backtest
                                    threshold_pct=threshold,
                                    trailing_pct=trailing,
                                    pyramid_step_pct=pyramid_step,
                                    max_pyramids=max_pyr,
                                    verbose=False
                                )

                                entry = {
                                    'base_threshold': base_t,
                                    'base_trailing': base_tr,
                                    'base_pyramid': base_p,
                                    'base_max_pyr': base_mp if base_mp < 9999 else 'unlimited',
                                    'threshold': threshold,
                                    'trailing': trailing,
                                    'pyramid_step': pyramid_step,
                                    'max_pyramids': max_pyr if max_pyr < 9999 else 'unlimited',
                                    'total_pnl': result['total_pnl'],
                                    'rounds': result['total_rounds'],
                                    'avg_pnl': result['avg_pnl'],
                                    'win_rate': result['win_rate'],
                                    'avg_pyramids': result['avg_pyramids']
                                }

                                # Write immediately to disk
                                writer.writerow(entry)
                                f.flush()

                                # Track only the best result
                                if best_result is None or entry['total_pnl'] > best_result['total_pnl']:
                                    best_result = entry

                            except:
                                pass

            print()

    return log_file, best_result


def process_coin(
    coin: str,
    price_streamer: Callable,
    num_prices: int,
    quick_test: bool = False
) -> CoinResult:
    """
    Process a single coin through grid search and refinement.

    Memory-efficient: uses price_streamer to stream data from disk
    for each backtest, avoiding loading all prices into memory.

    Args:
        coin: Trading pair symbol
        price_streamer: Callable that returns a generator of (timestamp, price)
        num_prices: Number of prices in the dataset (for display)
        quick_test: Use reduced grid for quick testing

    Returns:
        CoinResult with best parameters and analytics
    """
    print(f"\n{'#' * 70}")
    print(f"# PROCESSING: {coin}")
    print(f"{'#' * 70}")
    print(f"Data: {num_prices:,} price points (streaming from disk)")

    # Select grid parameters
    if quick_test:
        thresholds = THRESHOLDS_QUICK
        trailings = TRAILINGS_QUICK
        pyramid_steps = PYRAMID_STEPS_QUICK
        max_pyramids = MAX_PYRAMIDS_QUICK
    else:
        thresholds = THRESHOLDS
        trailings = TRAILINGS
        pyramid_steps = PYRAMID_STEPS
        max_pyramids = MAX_PYRAMIDS

    total_combos = len(thresholds) * len(trailings) * len(pyramid_steps) * len(max_pyramids)

    # Grid search
    print(f"\n--- Grid Search ({total_combos:,} combinations) ---")
    start = time.time()
    grid_log_file, top_10 = run_grid_search(coin, price_streamer, thresholds, trailings, pyramid_steps, max_pyramids)
    print(f"  ✓ Completed in {(time.time()-start)/60:.1f} minutes")

    # Save top 10
    if top_10:
        with open(os.path.join(LOG_DIR, f"{coin}_pyramid_v2_top10.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=top_10[0].keys())
            writer.writeheader()
            writer.writerows(top_10)

    print(f"\n--- Top {TOP_N} ---")
    for i, r in enumerate(top_10, 1):
        mp = r['max_pyramids']
        print(f"  #{i}: T={r['threshold']}% Tr={r['trailing']}% P={r['pyramid_step']}% MP={mp} → {r['total_pnl']:+.2f}%")

    # Refined search
    print(f"\n--- Refined Search ---")
    start = time.time()
    refined_log_file, best = run_refined_search(coin, price_streamer, top_10)
    print(f"  ✓ Completed in {(time.time()-start)/60:.1f} minutes")

    # Fall back to top grid result if refined search found nothing better
    if best is None:
        best = top_10[0] if top_10 else {'threshold': 5, 'trailing': 2, 'pyramid_step': 2, 'max_pyramids': 20, 'total_pnl': 0, 'win_rate': 0}

    # Run best config again to get full rounds for analytics
    best_max_pyr = best['max_pyramids'] if best['max_pyramids'] != 'unlimited' else 9999
    best_result = run_pyramid_backtest(
        price_streamer(),  # Stream from disk one more time
        threshold_pct=best['threshold'],
        trailing_pct=best['trailing'],
        pyramid_step_pct=best['pyramid_step'],
        max_pyramids=best_max_pyr,
        verbose=False
    )

    # Generate analytics - pop rounds to avoid keeping them in memory
    rounds = best_result.pop('rounds', [])
    summaries = rounds_to_summaries(rounds)
    analytics = generate_analytics_summary(summaries, best_max_pyr)

    # Explicit cleanup of rounds data
    del rounds
    del summaries

    print(f"\n🏆 BEST FOR {coin}:")
    print(f"   Threshold:    {best['threshold']}%")
    print(f"   Trailing:     {best['trailing']}%")
    print(f"   Pyramid Step: {best['pyramid_step']}%")
    print(f"   Max Pyramids: {best['max_pyramids']}")
    print(f"   Total P&L:    {best['total_pnl']:+.2f}%")
    print(f"   Win Rate:     {best['win_rate']:.1f}%")

    # Print analytics summary
    print_analytics_summary(analytics, coin)

    return CoinResult(coin, best, analytics, grid_log_file, refined_log_file)


def print_final_report(results: Dict[str, CoinResult], years: int):
    """Print and save comprehensive final report."""
    print("\n" + "=" * 100)
    print("ENHANCED PYRAMID STRATEGY - FINAL RECOMMENDATIONS")
    print("=" * 100)
    print(f"Data period: {years} year(s) of tick data")
    print()
    
    # Header
    print(f"{'Coin':<10} {'Threshold':<10} {'Trailing':<10} {'PyrStep':<10} {'MaxPyr':<10} "
          f"{'P&L':<10} {'WinRate':<10} {'MaxDD':<10} {'MinAcct':<12}")
    print("-" * 100)
    
    final_data = []
    
    for coin in results:
        r = results[coin].best_result
        a = results[coin].analytics
        
        mp = r['max_pyramids'] if r['max_pyramids'] != 'unlimited' else '∞'
        min_acct = a['account_sizing']['recommended_account']
        
        print(f"{coin:<10} {r['threshold']:<10.1f}% {r['trailing']:<10.1f}% {r['pyramid_step']:<10.1f}% "
              f"{mp:<10} {r['total_pnl']:+8.1f}%  {r['win_rate']:<10.1f}% {a['max_drawdown']:<10.1f}% ${min_acct:<10.0f}")
        
        final_data.append({
            'coin': coin,
            'threshold': r['threshold'],
            'trailing': r['trailing'],
            'pyramid_step': r['pyramid_step'],
            'max_pyramids': r['max_pyramids'],
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
    
    print("-" * 100)
    
    # Save final recommendations
    with open(os.path.join(LOG_DIR, "pyramid_v2_recommendations.csv"), 'w', newline='') as f:
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
    with open(os.path.join(LOG_DIR, "pyramid_v2_summary.txt"), 'w') as f:
        f.write("ENHANCED PYRAMID STRATEGY OPTIMIZER v2\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Data: {years} year(s) of tick data\n\n")
        
        for item in final_data:
            f.write(f"{item['coin']}:\n")
            f.write(f"  Threshold:     {item['threshold']:.1f}%\n")
            f.write(f"  Trailing:      {item['trailing']:.1f}%\n")
            f.write(f"  Pyramid Step:  {item['pyramid_step']:.1f}%\n")
            f.write(f"  Max Pyramids:  {item['max_pyramids']}\n")
            f.write(f"  Total P&L:     {item['total_pnl']:+.2f}%\n")
            f.write(f"  Win Rate:      {item['win_rate']:.1f}%\n")
            f.write(f"  Sharpe Ratio:  {item['sharpe_ratio']:.2f}\n")
            f.write(f"  Max Drawdown:  {item['max_drawdown']:.2f}%\n")
            f.write(f"  Min Account:   ${item['recommended_account']:.0f} (recommended)\n\n")
    
    print(f"\n✓ Results saved to {LOG_DIR}/")
    print(f"  - pyramid_v2_recommendations.csv")
    print(f"  - pyramid_v2_summary.txt")
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

    parser = argparse.ArgumentParser(description="Enhanced Pyramid Strategy Optimizer v2 (Memory-Optimized)")
    parser.add_argument("--coins", type=str, default=None, help="Comma-separated coins to test")
    parser.add_argument("--quick-test", action="store_true", help="Use reduced grid for quick testing")
    parser.add_argument("--verify-ticks", action="store_true",
                        help="Verify best results with filtered tick-by-tick data (maximum accuracy)")
    parser.add_argument("--ticks-only", action="store_true",
                        help="Skip grid search, run ONLY with filtered tick data")
    parser.add_argument("--tick-filter", type=float, default=0.01,
                        help="Minimum price move %% to include in tick data (default: 0.01%% = 1 basis point)")
    args = parser.parse_args()

    print("=" * 70)
    print("ENHANCED PYRAMID STRATEGY OPTIMIZER v2")
    print("(Memory-Optimized: Streaming from Disk)")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()

    # Prompt for years of data
    years = get_years_from_user()

    # Parse coins
    coins = args.coins.split(",") if args.coins else DEFAULT_COINS

    # Grid info
    if args.quick_test:
        combos = len(THRESHOLDS_QUICK) * len(TRAILINGS_QUICK) * len(PYRAMID_STEPS_QUICK) * len(MAX_PYRAMIDS_QUICK)
        print("\n[QUICK TEST MODE]")
    else:
        combos = len(THRESHOLDS) * len(TRAILINGS) * len(PYRAMID_STEPS) * len(MAX_PYRAMIDS)

    print(f"\nConfiguration:")
    print(f"  Coins:        {', '.join(coins)}")
    print(f"  Data:         {years} year(s) of tick data")
    print(f"  Grid:         {combos:,} combinations per coin")
    print(f"  Max Pyramids: {MAX_PYRAMIDS if not args.quick_test else MAX_PYRAMIDS_QUICK}")
    print(f"  Memory Mode:  Streaming (low RAM usage)")
    print("=" * 70)

    ensure_dirs()

    # Phase 1: Download and cache all data first (sequential to avoid OOM)
    print("\n" + "=" * 70)
    print("PHASE 1: DOWNLOADING & CACHING DATA")
    print("=" * 70)

    valid_coins = []
    for i, coin in enumerate(coins, 1):
        print(f"\n[{i}/{len(coins)}] {coin}...")
        try:
            # This downloads and caches to disk, returning a streamer function
            price_streamer = create_price_streamer(coin, years=years, verbose=True)
            num_prices = count_cached_prices(coin, years)

            if num_prices > 1000:
                valid_coins.append((coin, price_streamer, num_prices))
                print(f"  ✓ Ready: {num_prices:,} prices cached")
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

    for i, (coin, price_streamer, num_prices) in enumerate(valid_coins, 1):
        print(f"\n[{i}/{len(valid_coins)}] Processing {coin}...")

        # Process this coin (streams from disk, minimal RAM)
        result = process_coin(coin, price_streamer, num_prices, quick_test=args.quick_test)
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
