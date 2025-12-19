#!/usr/bin/env python3
"""
Comprehensive Parameter Optimizer for Hedge Trailing Strategy

Grid search across:
- 15 thresholds (1% to 15%)
- 20 multipliers (0.1 to 2.0)
= 300 total combinations

Uses 365 days of 1-minute candle data.
After finding top performers, does a refined search around the best values.

Usage:
    python optimize.py
    
This will take several hours to run. Recommended to run overnight on a server.
Results are saved to logs/optimization_results.csv
"""

import os
import sys
import csv
import time
from datetime import datetime
from typing import List, Tuple, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import fetch_historical_data, run_backtest


def run_grid_search(
    prices: List[Tuple[datetime, float]],
    thresholds: List[float],
    multipliers: List[float],
    log_file: str = None
) -> List[Dict]:
    """
    Run grid search across all threshold/multiplier combinations.
    
    Args:
        prices: Historical price data
        thresholds: List of threshold percentages to test
        multipliers: List of multipliers to test
        log_file: Optional CSV file to log results as they come in
        
    Returns:
        List of result dictionaries sorted by P&L
    """
    total_combinations = len(thresholds) * len(multipliers)
    results = []
    
    print(f"\n{'='*70}")
    print(f"PHASE 1: GRID SEARCH ({total_combinations} combinations)")
    print(f"{'='*70}")
    print(f"Thresholds: {min(thresholds)}% to {max(thresholds)}% ({len(thresholds)} values)")
    print(f"Multipliers: {min(multipliers)} to {max(multipliers)} ({len(multipliers)} values)")
    print(f"{'='*70}\n")
    
    # Open log file for progressive saving
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        f = open(log_file, 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=[
            'threshold', 'multiplier', 'total_pnl', 'rounds', 'avg_pnl', 
            'win_rate', 'long_pnl', 'short_pnl'
        ])
        writer.writeheader()
    else:
        f = None
        writer = None
    
    start_time = time.time()
    completed = 0
    
    for threshold in thresholds:
        for multiplier in multipliers:
            completed += 1
            
            # Progress indicator
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total_combinations - completed) / rate if rate > 0 else 0
            
            print(f"[{completed}/{total_combinations}] Testing {threshold}% / mult={multiplier:.2f}...", 
                  end=" ", flush=True)
            
            try:
                result = run_backtest(
                    prices, 
                    threshold, 
                    single_pos_multiplier=multiplier,
                    verbose=False
                )
                
                result_entry = {
                    'threshold': threshold,
                    'multiplier': multiplier,
                    'total_pnl': result['total_pnl'],
                    'rounds': result['total_rounds'],
                    'avg_pnl': result['avg_pnl_per_round'],
                    'win_rate': result['win_rate'],
                    'long_pnl': result['total_long_pnl'],
                    'short_pnl': result['total_short_pnl']
                }
                
                results.append(result_entry)
                
                # Progressive save
                if writer:
                    writer.writerow(result_entry)
                    f.flush()
                
                emoji = "✅" if result['total_pnl'] > 0 else "❌"
                print(f"{emoji} P&L: {result['total_pnl']:+.2f}% ({int(remaining/60)}m remaining)")
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'threshold': threshold,
                    'multiplier': multiplier,
                    'total_pnl': float('-inf'),
                    'rounds': 0,
                    'avg_pnl': 0,
                    'win_rate': 0,
                    'long_pnl': 0,
                    'short_pnl': 0
                })
    
    if f:
        f.close()
    
    # Sort by P&L
    results.sort(key=lambda x: x['total_pnl'], reverse=True)
    
    return results


def run_refined_search(
    prices: List[Tuple[datetime, float]],
    best_threshold: float,
    best_multiplier: float,
    log_file: str = None
) -> List[Dict]:
    """
    Run refined search around the best parameters found.
    Tests ±0.5% threshold and ±0.15 multiplier with fine granularity.
    """
    # Refined ranges
    threshold_min = max(0.5, best_threshold - 0.5)
    threshold_max = best_threshold + 0.5
    thresholds = [threshold_min + i * 0.1 for i in range(int((threshold_max - threshold_min) / 0.1) + 1)]
    
    multiplier_min = max(0.1, best_multiplier - 0.15)
    multiplier_max = min(2.0, best_multiplier + 0.15)
    multipliers = [multiplier_min + i * 0.05 for i in range(int((multiplier_max - multiplier_min) / 0.05) + 1)]
    
    total_combinations = len(thresholds) * len(multipliers)
    results = []
    
    print(f"\n{'='*70}")
    print(f"PHASE 2: REFINED SEARCH ({total_combinations} combinations)")
    print(f"{'='*70}")
    print(f"Centered on: threshold={best_threshold}%, multiplier={best_multiplier}")
    print(f"Thresholds: {min(thresholds):.1f}% to {max(thresholds):.1f}% ({len(thresholds)} values)")
    print(f"Multipliers: {min(multipliers):.2f} to {max(multipliers):.2f} ({len(multipliers)} values)")
    print(f"{'='*70}\n")
    
    # Open refined log file
    if log_file:
        f = open(log_file.replace('.csv', '_refined.csv'), 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=[
            'threshold', 'multiplier', 'total_pnl', 'rounds', 'avg_pnl', 
            'win_rate', 'long_pnl', 'short_pnl'
        ])
        writer.writeheader()
    else:
        f = None
        writer = None
    
    completed = 0
    
    for threshold in thresholds:
        for multiplier in multipliers:
            completed += 1
            print(f"[{completed}/{total_combinations}] Refining {threshold:.1f}% / mult={multiplier:.2f}...", 
                  end=" ", flush=True)
            
            try:
                result = run_backtest(
                    prices, 
                    threshold, 
                    single_pos_multiplier=multiplier,
                    verbose=False
                )
                
                result_entry = {
                    'threshold': threshold,
                    'multiplier': multiplier,
                    'total_pnl': result['total_pnl'],
                    'rounds': result['total_rounds'],
                    'avg_pnl': result['avg_pnl_per_round'],
                    'win_rate': result['win_rate'],
                    'long_pnl': result['total_long_pnl'],
                    'short_pnl': result['total_short_pnl']
                }
                
                results.append(result_entry)
                
                if writer:
                    writer.writerow(result_entry)
                    f.flush()
                
                emoji = "✅" if result['total_pnl'] > 0 else "❌"
                print(f"{emoji} P&L: {result['total_pnl']:+.2f}%")
                
            except Exception as e:
                print(f"Error: {e}")
    
    if f:
        f.close()
    
    results.sort(key=lambda x: x['total_pnl'], reverse=True)
    return results


def print_top_results(results: List[Dict], title: str, top_n: int = 20):
    """Print top N results in a formatted table."""
    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")
    print(f"{'Rank':<6} {'Threshold':<10} {'Multiplier':<12} {'Total P&L':<12} {'Rounds':<8} {'Avg P&L':<10} {'Win Rate':<10}")
    print("-" * 90)
    
    for i, r in enumerate(results[:top_n], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji}{i:<4} {r['threshold']:<10.1f}% {r['multiplier']:<12.2f} {r['total_pnl']:+10.2f}%  "
              f"{r['rounds']:<8} {r['avg_pnl']:+8.3f}%  {r['win_rate']:8.1f}%")
    
    print("-" * 90)


def main():
    print("=" * 70)
    print("HEDGE TRAILING STRATEGY - COMPREHENSIVE OPTIMIZER")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    print("  Data: 365 days of SOLUSDT 1-minute candles")
    print("  Grid: 15 thresholds × 20 multipliers = 300 combinations")
    print("  Refinement: ±0.5% threshold, ±0.15 multiplier around best")
    print("=" * 70)
    print()
    
    # Fetch historical data
    print("Fetching 365 days of 1-minute price data...")
    print("(This will take a few minutes)")
    print()
    
    prices = fetch_historical_data("SOLUSDT", days=365, interval="1m")
    
    if len(prices) < 1000:
        print("ERROR: Not enough price data fetched!")
        return
    
    print(f"\n✓ Fetched {len(prices):,} data points")
    print(f"  Price range: ${min(p for _, p in prices):.2f} - ${max(p for _, p in prices):.2f}")
    print(f"  Date range: {prices[0][0].strftime('%Y-%m-%d')} to {prices[-1][0].strftime('%Y-%m-%d')}")
    
    # Define search grid
    thresholds = [float(i) for i in range(1, 16)]  # 1% to 15%
    multipliers = [0.1 * i for i in range(1, 21)]  # 0.1 to 2.0
    
    log_file = "logs/optimization_results.csv"
    
    # Phase 1: Grid search
    grid_start = time.time()
    grid_results = run_grid_search(prices, thresholds, multipliers, log_file)
    grid_time = time.time() - grid_start
    
    print(f"\n✓ Grid search completed in {grid_time/60:.1f} minutes")
    
    # Print top 20 from grid search
    print_top_results(grid_results, "TOP 20 FROM GRID SEARCH", 20)
    
    # Get best result for refinement
    best = grid_results[0]
    
    # Phase 2: Refined search around best
    print(f"\n🎯 Best from grid: threshold={best['threshold']}%, multiplier={best['multiplier']}")
    print(f"   P&L: {best['total_pnl']:+.2f}% over {best['rounds']} rounds")
    
    refined_start = time.time()
    refined_results = run_refined_search(
        prices, 
        best['threshold'], 
        best['multiplier'],
        log_file
    )
    refined_time = time.time() - refined_start
    
    print(f"\n✓ Refined search completed in {refined_time/60:.1f} minutes")
    
    # Print top results from refined search
    print_top_results(refined_results, "TOP 10 FROM REFINED SEARCH", 10)
    
    # Final winner
    final_best = refined_results[0]
    
    print()
    print("=" * 70)
    print("🏆 OPTIMAL PARAMETERS FOUND")
    print("=" * 70)
    print(f"  Threshold:    {final_best['threshold']:.1f}%")
    print(f"  Multiplier:   {final_best['multiplier']:.2f}")
    print(f"  Single Stop:  {final_best['threshold'] * final_best['multiplier']:.2f}%")
    print()
    print(f"  Total P&L:    {final_best['total_pnl']:+.2f}%")
    print(f"  Rounds:       {final_best['rounds']}")
    print(f"  Avg P&L:      {final_best['avg_pnl']:+.3f}% per round")
    print(f"  Win Rate:     {final_best['win_rate']:.1f}%")
    print(f"  Long P&L:     {final_best['long_pnl']:+.2f}%")
    print(f"  Short P&L:    {final_best['short_pnl']:+.2f}%")
    print("=" * 70)
    print()
    print(f"Results saved to: {log_file}")
    print(f"Refined results saved to: {log_file.replace('.csv', '_refined.csv')}")
    print()
    print(f"Total runtime: {(grid_time + refined_time)/60:.1f} minutes")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save final summary
    with open("logs/optimization_summary.txt", "w") as f:
        f.write("HEDGE TRAILING STRATEGY - OPTIMIZATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: 365 days of SOLUSDT 1-minute candles\n\n")
        f.write("OPTIMAL PARAMETERS:\n")
        f.write(f"  Threshold:    {final_best['threshold']:.1f}%\n")
        f.write(f"  Multiplier:   {final_best['multiplier']:.2f}\n")
        f.write(f"  Single Stop:  {final_best['threshold'] * final_best['multiplier']:.2f}%\n\n")
        f.write("PERFORMANCE:\n")
        f.write(f"  Total P&L:    {final_best['total_pnl']:+.2f}%\n")
        f.write(f"  Rounds:       {final_best['rounds']}\n")
        f.write(f"  Avg P&L:      {final_best['avg_pnl']:+.3f}% per round\n")
        f.write(f"  Win Rate:     {final_best['win_rate']:.1f}%\n")
    
    print("Summary saved to: logs/optimization_summary.txt")


if __name__ == "__main__":
    main()
