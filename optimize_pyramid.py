#!/usr/bin/env python3
"""
Pyramid Strategy Multi-Coin Optimizer

Tests 6 coins with 3 variables:
- Threshold: 1-20% (step 1%) = 20 values
- Trailing: 1-5% (step 0.2%) = 21 values  
- Pyramid step: 1-3% (step 0.2%) = 11 values
- Total: 4,620 combinations per coin

Estimated runtime: ~8 days for all 6 coins + refinement
"""

import os
import sys
import csv
import time
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import fetch_historical_data
from backtest_pyramid import run_pyramid_backtest


# Configuration
COINS = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "XLMUSDT", "XRPUSDT", "DOGEUSDT"]
DAYS = 1095  # 3 years
INTERVAL = "1m"

# Grid parameters
THRESHOLDS = [float(i) for i in range(1, 21)]  # 1-20%
TRAILINGS = [1.0 + 0.2 * i for i in range(21)]  # 1-5% in 0.2 steps
PYRAMID_STEPS = [1.0 + 0.2 * i for i in range(11)]  # 1-3% in 0.2 steps

TOP_N = 10
REFINED_COMBOS = 500
CACHE_DIR = "cache"
LOG_DIR = "logs"


@dataclass
class CoinResult:
    coin: str
    grid_results: List[Dict]
    top_10: List[Dict]
    refined_results: List[Dict]
    best_result: Dict


def ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def get_cache_path(coin: str) -> str:
    return os.path.join(CACHE_DIR, f"{coin}_{DAYS}d_{INTERVAL}.json")


def save_to_cache(coin: str, prices: List[Tuple[datetime, float]]):
    cache_data = [(ts.isoformat(), price) for ts, price in prices]
    with open(get_cache_path(coin), 'w') as f:
        json.dump(cache_data, f)


def load_from_cache(coin: str) -> Optional[List[Tuple[datetime, float]]]:
    path = get_cache_path(coin)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return [(datetime.fromisoformat(ts), p) for ts, p in data]
        except:
            pass
    return None


def fetch_all_data() -> Dict[str, List]:
    print("\n" + "=" * 70)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 70)
    
    all_prices = {}
    
    for i, coin in enumerate(COINS, 1):
        print(f"\n[{i}/{len(COINS)}] {coin}:")
        
        prices = load_from_cache(coin)
        if prices:
            print(f"  ✓ Loaded {len(prices):,} from cache")
        else:
            print(f"  Fetching from Binance...")
            prices = fetch_historical_data(coin, DAYS, INTERVAL)
            if len(prices) > 1000:
                save_to_cache(coin, prices)
                print(f"  ✓ Fetched {len(prices):,} and cached")
        
        if len(prices) > 1000:
            all_prices[coin] = prices
            print(f"  Range: ${min(p for _, p in prices):.4f} - ${max(p for _, p in prices):.4f}")
    
    return all_prices


def run_grid_search(coin: str, prices: List) -> List[Dict]:
    total = len(THRESHOLDS) * len(TRAILINGS) * len(PYRAMID_STEPS)
    results = []
    
    log_file = os.path.join(LOG_DIR, f"{coin}_pyramid_grid.csv")
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'threshold', 'trailing', 'pyramid_step', 'total_pnl', 
            'rounds', 'avg_pnl', 'win_rate', 'avg_pyramids'
        ])
        writer.writeheader()
        
        completed = 0
        start = time.time()
        
        for threshold in THRESHOLDS:
            for trailing in TRAILINGS:
                for pyramid_step in PYRAMID_STEPS:
                    completed += 1
                    
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 1
                    remaining = (total - completed) / rate
                    
                    if completed % 100 == 0 or completed == 1:
                        print(f"\r  [{completed}/{total}] {remaining/3600:.1f}h remaining    ", 
                              end="", flush=True)
                    
                    try:
                        result = run_pyramid_backtest(
                            prices,
                            threshold_pct=threshold,
                            trailing_pct=trailing,
                            pyramid_step_pct=pyramid_step,
                            verbose=False
                        )
                        
                        entry = {
                            'threshold': threshold,
                            'trailing': trailing,
                            'pyramid_step': pyramid_step,
                            'total_pnl': result['total_pnl'],
                            'rounds': result['total_rounds'],
                            'avg_pnl': result['avg_pnl'],
                            'win_rate': result['win_rate'],
                            'avg_pyramids': result['avg_pyramids']
                        }
                        
                        results.append(entry)
                        writer.writerow(entry)
                        f.flush()
                        
                    except Exception as e:
                        print(f"\n  Error: {e}")
    
    print()
    results.sort(key=lambda x: x['total_pnl'], reverse=True)
    return results


def run_refined_search(coin: str, prices: List, top_results: List[Dict]) -> List[Dict]:
    all_refined = []
    combos_per_top = REFINED_COMBOS // len(top_results)
    
    log_file = os.path.join(LOG_DIR, f"{coin}_pyramid_refined.csv")
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'base_threshold', 'base_trailing', 'base_pyramid',
            'threshold', 'trailing', 'pyramid_step',
            'total_pnl', 'rounds', 'avg_pnl', 'win_rate', 'avg_pyramids'
        ])
        writer.writeheader()
        
        for rank, top in enumerate(top_results, 1):
            base_t = top['threshold']
            base_tr = top['trailing']
            base_p = top['pyramid_step']
            
            print(f"\n  Refining #{rank}: {base_t}% / {base_tr}% / {base_p}%")
            
            # Generate refined grid around this point
            thresholds = [round(base_t + 0.1 * i, 1) for i in range(-5, 6) if base_t + 0.1 * i > 0]
            trailings = [round(base_tr + 0.1 * i, 2) for i in range(-5, 6) if base_tr + 0.1 * i > 0]
            pyramids = [round(base_p + 0.1 * i, 2) for i in range(-5, 6) if base_p + 0.1 * i > 0]
            
            total = len(thresholds) * len(trailings) * len(pyramids)
            completed = 0
            
            for threshold in thresholds:
                for trailing in trailings:
                    for pyramid_step in pyramids:
                        completed += 1
                        
                        if completed % 50 == 0:
                            print(f"\r    [{completed}/{total}]    ", end="", flush=True)
                        
                        try:
                            result = run_pyramid_backtest(
                                prices,
                                threshold_pct=threshold,
                                trailing_pct=trailing,
                                pyramid_step_pct=pyramid_step,
                                verbose=False
                            )
                            
                            entry = {
                                'base_threshold': base_t,
                                'base_trailing': base_tr,
                                'base_pyramid': base_p,
                                'threshold': threshold,
                                'trailing': trailing,
                                'pyramid_step': pyramid_step,
                                'total_pnl': result['total_pnl'],
                                'rounds': result['total_rounds'],
                                'avg_pnl': result['avg_pnl'],
                                'win_rate': result['win_rate'],
                                'avg_pyramids': result['avg_pyramids']
                            }
                            
                            all_refined.append(entry)
                            writer.writerow(entry)
                            f.flush()
                            
                        except:
                            pass
            
            print()
    
    all_refined.sort(key=lambda x: x['total_pnl'], reverse=True)
    return all_refined


def process_coin(coin: str, prices: List) -> CoinResult:
    print(f"\n{'#' * 70}")
    print(f"# PROCESSING: {coin}")
    print(f"{'#' * 70}")
    print(f"Data: {len(prices):,} candles")
    
    # Grid search
    print(f"\n--- Grid Search ({len(THRESHOLDS)*len(TRAILINGS)*len(PYRAMID_STEPS):,} combinations) ---")
    start = time.time()
    grid_results = run_grid_search(coin, prices)
    print(f"  ✓ Completed in {(time.time()-start)/3600:.1f} hours")
    
    # Top 10
    top_10 = grid_results[:TOP_N]
    
    # Save top 10
    with open(os.path.join(LOG_DIR, f"{coin}_pyramid_top10.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=top_10[0].keys())
        writer.writeheader()
        writer.writerows(top_10)
    
    print(f"\n--- Top {TOP_N} ---")
    for i, r in enumerate(top_10, 1):
        print(f"  #{i}: T={r['threshold']}% Tr={r['trailing']}% P={r['pyramid_step']}% → {r['total_pnl']:+.2f}%")
    
    # Refined search
    print(f"\n--- Refined Search ---")
    start = time.time()
    refined = run_refined_search(coin, prices, top_10)
    print(f"  ✓ Completed in {(time.time()-start)/3600:.1f} hours")
    
    best = refined[0] if refined else top_10[0]
    
    print(f"\n🏆 BEST FOR {coin}:")
    print(f"   Threshold:    {best['threshold']}%")
    print(f"   Trailing:     {best['trailing']}%")
    print(f"   Pyramid Step: {best['pyramid_step']}%")
    print(f"   Total P&L:    {best['total_pnl']:+.2f}%")
    print(f"   Win Rate:     {best['win_rate']:.1f}%")
    
    return CoinResult(coin, grid_results, top_10, refined, best)


def print_final_report(results: Dict[str, CoinResult]):
    print("\n" + "=" * 100)
    print("PYRAMID STRATEGY - FINAL RECOMMENDATIONS")
    print("=" * 100)
    print()
    print(f"{'Coin':<10} {'Threshold':<12} {'Trailing':<12} {'Pyramid':<12} "
          f"{'P&L (3yr)':<12} {'Rounds':<8} {'Win Rate':<10} {'Avg Pyr':<8}")
    print("-" * 100)
    
    final_data = []
    
    for coin in COINS:
        if coin not in results:
            continue
        
        r = results[coin].best_result
        print(f"{coin:<10} {r['threshold']:<12.1f}% {r['trailing']:<12.1f}% {r['pyramid_step']:<12.1f}% "
              f"{r['total_pnl']:+10.2f}%  {r['rounds']:<8} {r['win_rate']:<10.1f}% {r['avg_pyramids']:<8.1f}")
        
        final_data.append({
            'coin': coin,
            'threshold': r['threshold'],
            'trailing': r['trailing'],
            'pyramid_step': r['pyramid_step'],
            'total_pnl': r['total_pnl'],
            'rounds': r['rounds'],
            'avg_pnl': r['avg_pnl'],
            'win_rate': r['win_rate'],
            'avg_pyramids': r['avg_pyramids']
        })
    
    print("-" * 100)
    
    # Save final
    with open(os.path.join(LOG_DIR, "pyramid_recommendations.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_data[0].keys())
        writer.writeheader()
        writer.writerows(final_data)
    
    # Summary
    with open(os.path.join(LOG_DIR, "pyramid_summary.txt"), 'w') as f:
        f.write("PYRAMID STRATEGY OPTIMIZER RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Data: {DAYS} days, {INTERVAL} candles\n\n")
        
        for item in final_data:
            f.write(f"{item['coin']}:\n")
            f.write(f"  Threshold:    {item['threshold']:.1f}%\n")
            f.write(f"  Trailing:     {item['trailing']:.1f}%\n")
            f.write(f"  Pyramid Step: {item['pyramid_step']:.1f}%\n")
            f.write(f"  Total P&L:    {item['total_pnl']:+.2f}%\n")
            f.write(f"  Win Rate:     {item['win_rate']:.1f}%\n\n")
    
    print(f"\n✓ Saved to {LOG_DIR}/pyramid_recommendations.csv")


def main():
    print("=" * 70)
    print("PYRAMID STRATEGY MULTI-COIN OPTIMIZER")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()
    print("Configuration:")
    print(f"  Coins:     {', '.join(COINS)}")
    print(f"  Data:      {DAYS} days of {INTERVAL} candles")
    print(f"  Threshold: {min(THRESHOLDS)}-{max(THRESHOLDS)}% ({len(THRESHOLDS)} values)")
    print(f"  Trailing:  {min(TRAILINGS)}-{max(TRAILINGS)}% ({len(TRAILINGS)} values)")
    print(f"  Pyramid:   {min(PYRAMID_STEPS)}-{max(PYRAMID_STEPS)}% ({len(PYRAMID_STEPS)} values)")
    print(f"  Grid:      {len(THRESHOLDS)*len(TRAILINGS)*len(PYRAMID_STEPS):,} combos/coin")
    print("=" * 70)
    
    ensure_dirs()
    all_prices = fetch_all_data()
    
    if not all_prices:
        print("ERROR: No data!")
        return
    
    all_results = {}
    start = time.time()
    
    for coin, prices in all_prices.items():
        result = process_coin(coin, prices)
        all_results[coin] = result
    
    total_time = time.time() - start
    
    print_final_report(all_results)
    
    print()
    print("=" * 70)
    print(f"Total runtime: {total_time/3600:.1f} hours")
    print(f"Completed: {datetime.now()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
