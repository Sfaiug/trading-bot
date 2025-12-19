#!/usr/bin/env python3
"""
Multi-Coin Parameter Optimizer for Hedge Trailing Strategy

Tests 6 coins (SOL, BTC, ETH, XLM, XRP, DOGE) with:
- 3 years of 1-minute candle data
- 300 parameter combinations (15 thresholds × 20 multipliers)
- Refined search around top 5 performers per coin
- Final recommendations per coin

Usage:
    python optimize_multi.py
    
Estimated runtime: ~21 hours
Results saved to logs/ directory
"""

import os
import sys
import csv
import time
import json
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import fetch_historical_data, run_backtest


# Configuration
COINS = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "XLMUSDT", "XRPUSDT", "DOGEUSDT"]
DAYS = 1095  # 3 years
INTERVAL = "1m"
THRESHOLDS = [float(i) for i in range(1, 16)]  # 1% to 15%
MULTIPLIERS = [0.1 * i for i in range(1, 21)]  # 0.1 to 2.0
TOP_N = 5  # Top performers to refine
CACHE_DIR = "cache"
LOG_DIR = "logs"


@dataclass
class CoinResult:
    """Stores results for a single coin."""
    coin: str
    grid_results: List[Dict]
    top_5: List[Dict]
    refined_results: List[Dict]
    best_result: Dict


def ensure_dirs():
    """Create necessary directories."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def get_cache_path(coin: str) -> str:
    """Get cache file path for a coin's price data."""
    return os.path.join(CACHE_DIR, f"{coin}_{DAYS}d_{INTERVAL}.json")


def save_prices_to_cache(coin: str, prices: List[Tuple[datetime, float]]):
    """Save price data to cache file."""
    cache_data = [(ts.isoformat(), price) for ts, price in prices]
    with open(get_cache_path(coin), 'w') as f:
        json.dump(cache_data, f)
    print(f"  ✓ Cached {len(prices):,} candles for {coin}")


def load_prices_from_cache(coin: str) -> Optional[List[Tuple[datetime, float]]]:
    """Load price data from cache if available."""
    cache_path = get_cache_path(coin)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            prices = [(datetime.fromisoformat(ts), price) for ts, price in cache_data]
            print(f"  ✓ Loaded {len(prices):,} candles from cache for {coin}")
            return prices
        except Exception as e:
            print(f"  ⚠ Cache error for {coin}: {e}")
    return None


def fetch_all_coin_data() -> Dict[str, List[Tuple[datetime, float]]]:
    """Fetch historical data for all coins, using cache when available."""
    print("\n" + "=" * 70)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 70)
    print(f"Fetching {DAYS} days of {INTERVAL} candles for {len(COINS)} coins...")
    print()
    
    all_prices = {}
    
    for i, coin in enumerate(COINS, 1):
        print(f"[{i}/{len(COINS)}] {coin}:")
        
        # Try cache first
        prices = load_prices_from_cache(coin)
        
        if prices is None:
            # Fetch from Binance
            print(f"  Fetching from Binance API...")
            prices = fetch_historical_data(coin, DAYS, INTERVAL)
            
            if len(prices) > 1000:
                save_prices_to_cache(coin, prices)
            else:
                print(f"  ⚠ Only got {len(prices)} candles, skipping cache")
        
        if len(prices) < 1000:
            print(f"  ❌ Insufficient data for {coin}, skipping")
            continue
            
        all_prices[coin] = prices
        print(f"  Price range: ${min(p for _, p in prices):.4f} - ${max(p for _, p in prices):.4f}")
        print()
    
    return all_prices


def run_grid_search(
    coin: str,
    prices: List[Tuple[datetime, float]]
) -> List[Dict]:
    """Run grid search for a single coin."""
    total = len(THRESHOLDS) * len(MULTIPLIERS)
    results = []
    
    log_file = os.path.join(LOG_DIR, f"{coin}_grid_results.csv")
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'threshold', 'multiplier', 'total_pnl', 'rounds', 'avg_pnl',
            'win_rate', 'long_pnl', 'short_pnl'
        ])
        writer.writeheader()
        
        completed = 0
        start_time = time.time()
        
        for threshold in THRESHOLDS:
            for multiplier in MULTIPLIERS:
                completed += 1
                
                # Progress
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 1
                remaining = (total - completed) / rate
                
                print(f"\r  [{completed}/{total}] {threshold}%/{multiplier:.1f} "
                      f"({remaining/60:.0f}m left)    ", end="", flush=True)
                
                try:
                    result = run_backtest(
                        prices,
                        threshold,
                        single_pos_multiplier=multiplier,
                        verbose=False
                    )
                    
                    entry = {
                        'threshold': threshold,
                        'multiplier': multiplier,
                        'total_pnl': result['total_pnl'],
                        'rounds': result['total_rounds'],
                        'avg_pnl': result['avg_pnl_per_round'],
                        'win_rate': result['win_rate'],
                        'long_pnl': result['total_long_pnl'],
                        'short_pnl': result['total_short_pnl']
                    }
                    
                    results.append(entry)
                    writer.writerow(entry)
                    f.flush()
                    
                except Exception as e:
                    print(f"\n  ⚠ Error at {threshold}%/{multiplier}: {e}")
    
    print()  # New line after progress
    
    # Sort by P&L
    results.sort(key=lambda x: x['total_pnl'], reverse=True)
    return results


def run_refined_search(
    coin: str,
    prices: List[Tuple[datetime, float]],
    top_results: List[Dict]
) -> List[Dict]:
    """Run refined search around top performers."""
    all_refined = []
    
    log_file = os.path.join(LOG_DIR, f"{coin}_refined_results.csv")
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'base_threshold', 'base_multiplier', 'threshold', 'multiplier',
            'total_pnl', 'rounds', 'avg_pnl', 'win_rate', 'long_pnl', 'short_pnl'
        ])
        writer.writeheader()
        
        for rank, top in enumerate(top_results, 1):
            base_t = top['threshold']
            base_m = top['multiplier']
            
            print(f"\n  Refining #{rank}: {base_t}% / {base_m:.2f}...")
            
            # Generate refined grid
            t_min = max(0.5, base_t - 0.5)
            t_max = base_t + 0.5
            thresholds = [round(t_min + i * 0.1, 1) for i in range(int((t_max - t_min) / 0.1) + 1)]
            
            m_min = max(0.1, base_m - 0.2)
            m_max = min(2.0, base_m + 0.2)
            multipliers = [round(m_min + i * 0.05, 2) for i in range(int((m_max - m_min) / 0.05) + 1)]
            
            total = len(thresholds) * len(multipliers)
            completed = 0
            
            for threshold in thresholds:
                for multiplier in multipliers:
                    completed += 1
                    print(f"\r    [{completed}/{total}]    ", end="", flush=True)
                    
                    try:
                        result = run_backtest(
                            prices,
                            threshold,
                            single_pos_multiplier=multiplier,
                            verbose=False
                        )
                        
                        entry = {
                            'base_threshold': base_t,
                            'base_multiplier': base_m,
                            'threshold': threshold,
                            'multiplier': multiplier,
                            'total_pnl': result['total_pnl'],
                            'rounds': result['total_rounds'],
                            'avg_pnl': result['avg_pnl_per_round'],
                            'win_rate': result['win_rate'],
                            'long_pnl': result['total_long_pnl'],
                            'short_pnl': result['total_short_pnl']
                        }
                        
                        all_refined.append(entry)
                        writer.writerow(entry)
                        f.flush()
                        
                    except Exception as e:
                        pass
            
            print()
    
    # Sort by P&L
    all_refined.sort(key=lambda x: x['total_pnl'], reverse=True)
    return all_refined


def process_coin(coin: str, prices: List[Tuple[datetime, float]]) -> CoinResult:
    """Process a single coin: grid search + refinement."""
    print(f"\n{'#' * 70}")
    print(f"# PROCESSING: {coin}")
    print(f"{'#' * 70}")
    print(f"Data points: {len(prices):,}")
    print(f"Date range: {prices[0][0].strftime('%Y-%m-%d')} to {prices[-1][0].strftime('%Y-%m-%d')}")
    
    # Phase 2: Grid Search
    print(f"\n--- Grid Search (300 combinations) ---")
    start = time.time()
    grid_results = run_grid_search(coin, prices)
    grid_time = time.time() - start
    print(f"  ✓ Completed in {grid_time/60:.1f} minutes")
    
    # Select top 5
    top_5 = grid_results[:TOP_N]
    
    # Save top 5
    top5_file = os.path.join(LOG_DIR, f"{coin}_top5.csv")
    with open(top5_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=top_5[0].keys())
        writer.writeheader()
        writer.writerows(top_5)
    
    print(f"\n--- Top {TOP_N} from Grid Search ---")
    for i, r in enumerate(top_5, 1):
        print(f"  #{i}: {r['threshold']}% / {r['multiplier']:.2f} → {r['total_pnl']:+.2f}%")
    
    # Phase 3: Refined Search
    print(f"\n--- Refined Search (around top {TOP_N}) ---")
    start = time.time()
    refined_results = run_refined_search(coin, prices, top_5)
    refine_time = time.time() - start
    print(f"  ✓ Completed in {refine_time/60:.1f} minutes")
    
    # Best result
    best = refined_results[0] if refined_results else top_5[0]
    
    print(f"\n🏆 BEST FOR {coin}:")
    print(f"   Threshold:  {best['threshold']}%")
    print(f"   Multiplier: {best['multiplier']:.2f}")
    print(f"   Total P&L:  {best['total_pnl']:+.2f}%")
    print(f"   Rounds:     {best['rounds']}")
    print(f"   Win Rate:   {best['win_rate']:.1f}%")
    
    return CoinResult(
        coin=coin,
        grid_results=grid_results,
        top_5=top_5,
        refined_results=refined_results,
        best_result=best
    )


def print_final_report(results: Dict[str, CoinResult]):
    """Print and save final recommendations."""
    print("\n" + "=" * 90)
    print("FINAL RECOMMENDATIONS - OPTIMAL PARAMETERS PER COIN")
    print("=" * 90)
    print()
    print(f"{'Coin':<10} {'Threshold':<12} {'Multiplier':<12} {'Single Stop':<12} "
          f"{'P&L (3yr)':<12} {'Rounds':<8} {'Win Rate':<10}")
    print("-" * 90)
    
    final_data = []
    
    for coin in COINS:
        if coin not in results:
            continue
        
        r = results[coin].best_result
        single_stop = r['threshold'] * r['multiplier']
        
        print(f"{coin:<10} {r['threshold']:<12.1f}% {r['multiplier']:<12.2f} {single_stop:<12.2f}% "
              f"{r['total_pnl']:+10.2f}%  {r['rounds']:<8} {r['win_rate']:<10.1f}%")
        
        final_data.append({
            'coin': coin,
            'threshold': r['threshold'],
            'multiplier': r['multiplier'],
            'single_stop': single_stop,
            'total_pnl': r['total_pnl'],
            'rounds': r['rounds'],
            'avg_pnl': r['avg_pnl'],
            'win_rate': r['win_rate'],
            'long_pnl': r['long_pnl'],
            'short_pnl': r['short_pnl']
        })
    
    print("-" * 90)
    
    # Save to CSV
    final_file = os.path.join(LOG_DIR, "final_recommendations.csv")
    with open(final_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_data[0].keys())
        writer.writeheader()
        writer.writerows(final_data)
    
    # Save summary
    summary_file = os.path.join(LOG_DIR, "optimization_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("MULTI-COIN HEDGE TRAILING STRATEGY OPTIMIZER\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: {DAYS} days, {INTERVAL} candles\n")
        f.write(f"Coins tested: {', '.join(COINS)}\n\n")
        f.write("OPTIMAL PARAMETERS:\n\n")
        
        for item in final_data:
            f.write(f"{item['coin']}:\n")
            f.write(f"  Threshold:    {item['threshold']:.1f}%\n")
            f.write(f"  Multiplier:   {item['multiplier']:.2f}\n")
            f.write(f"  Single Stop:  {item['single_stop']:.2f}%\n")
            f.write(f"  Total P&L:    {item['total_pnl']:+.2f}%\n")
            f.write(f"  Rounds:       {item['rounds']}\n")
            f.write(f"  Win Rate:     {item['win_rate']:.1f}%\n\n")
    
    print(f"\n✓ Results saved to {LOG_DIR}/")
    print(f"  - final_recommendations.csv")
    print(f"  - optimization_summary.txt")


def main():
    print("=" * 70)
    print("MULTI-COIN HEDGE TRAILING STRATEGY OPTIMIZER")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    print(f"  Coins:       {', '.join(COINS)}")
    print(f"  Data:        {DAYS} days of {INTERVAL} candles")
    print(f"  Grid:        {len(THRESHOLDS)} thresholds × {len(MULTIPLIERS)} multipliers = {len(THRESHOLDS)*len(MULTIPLIERS)} combos")
    print(f"  Refinement:  Top {TOP_N} per coin")
    print("=" * 70)
    
    ensure_dirs()
    
    # Phase 1: Fetch all data
    all_prices = fetch_all_coin_data()
    
    if not all_prices:
        print("ERROR: No price data fetched!")
        return
    
    # Process each coin
    all_results = {}
    total_start = time.time()
    
    for coin, prices in all_prices.items():
        result = process_coin(coin, prices)
        all_results[coin] = result
    
    total_time = time.time() - total_start
    
    # Final report
    print_final_report(all_results)
    
    print()
    print("=" * 70)
    print(f"Total runtime: {total_time/3600:.1f} hours")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
