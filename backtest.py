#!/usr/bin/env python3
"""
Backtesting Simulation for Hedge Trailing Strategy

Runs the strategy against historical SOL/USDT price data to compare
different threshold percentages and find the optimal setting.

Usage:
    python backtest.py --thresholds 0.5,1,2,3,5 --days 1095
"""

import argparse
import requests
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import csv
import os


@dataclass
class BacktestPosition:
    """Tracks a single position during backtesting."""
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    max_profit_pct: float = 0.0  # Highest profit % reached (relative to entry)
    is_open: bool = True
    exit_price: float = 0.0
    exit_time: datetime = None
    pnl_percent: float = 0.0


@dataclass
class BacktestRound:
    """Tracks a complete round (LONG + SHORT closed)."""
    round_num: int
    entry_price: float
    entry_time: datetime
    long_exit_price: float = 0.0
    short_exit_price: float = 0.0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    duration_seconds: float = 0.0
    
    @property
    def total_pnl(self) -> float:
        return self.long_pnl + self.short_pnl


def fetch_historical_data(symbol: str = "SOLUSDT", days: int = 1095, interval: str = "1h") -> List[Tuple[datetime, float]]:
    """
    Fetch historical price data from Binance.
    
    Args:
        symbol: Trading pair
        days: Number of days of history to fetch
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        
    Returns:
        List of (timestamp, close_price) tuples
    """
    print(f"Fetching {days} days of {symbol} historical data ({interval} candles)...")
    
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    
    current_start = start_time
    batch_count = 0
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            for candle in data:
                timestamp = datetime.fromtimestamp(candle[0] / 1000)
                close_price = float(candle[4])
                all_data.append((timestamp, close_price))
            
            # Move to next batch
            current_start = data[-1][0] + 1
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"  Fetched {len(all_data)} data points...")
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    interval_name = {"1m": "minute", "5m": "5-min", "15m": "15-min", "1h": "hourly", "4h": "4-hour", "1d": "daily"}.get(interval, interval)
    print(f"✓ Fetched {len(all_data)} {interval_name} candles")
    return all_data


def run_backtest(
    prices: List[Tuple[datetime, float]],
    threshold_percent: float,
    position_size: float = 1.0,
    fee_percent: float = 0.04,  # Binance Futures taker fee (0.04%)
    single_pos_multiplier: float = 0.5,  # Tighter stop when one position closed
    verbose: bool = False
) -> Dict:
    """
    Run backtest simulation for a single threshold.
    
    Args:
        prices: List of (timestamp, price) tuples
        threshold_percent: Trailing stop threshold
        position_size: Size of each position
        fee_percent: Trading fee per trade (default: 0.04% for Binance Futures)
        single_pos_multiplier: Multiplier for threshold when only one position is open (default: 0.5 = half)
        verbose: Print detailed output
        
    Returns:
        Dictionary with backtest results
    """
    rounds: List[BacktestRound] = []
    current_round = 0
    
    long_pos: BacktestPosition = None
    short_pos: BacktestPosition = None
    
    total_long_pnl = 0.0
    total_short_pnl = 0.0
    total_fees = 0.0
    
    for i, (timestamp, price) in enumerate(prices):
        # Open new positions if none are open
        if long_pos is None and short_pos is None:
            current_round += 1
            long_pos = BacktestPosition(
                side='LONG',
                entry_price=price,
                entry_time=timestamp
            )
            short_pos = BacktestPosition(
                side='SHORT',
                entry_price=price,
                entry_time=timestamp
            )
            # Pay entry fees (2 positions x fee)
            total_fees += 2 * fee_percent
            
            if verbose:
                print(f"Round {current_round}: Opened @ ${price:.2f} at {timestamp}")
        
        # Update max profit for each position
        if long_pos and long_pos.is_open:
            current_profit = ((price - long_pos.entry_price) / long_pos.entry_price) * 100
            long_pos.max_profit_pct = max(long_pos.max_profit_pct, current_profit)
        
        if short_pos and short_pos.is_open:
            current_profit = ((short_pos.entry_price - price) / short_pos.entry_price) * 100
            short_pos.max_profit_pct = max(short_pos.max_profit_pct, current_profit)
        
        # Determine effective thresholds based on which positions are still open
        # Use tighter threshold when only one position remains (to lock in profits)
        long_open = long_pos and long_pos.is_open
        short_open = short_pos and short_pos.is_open
        
        # If only one position is open, use tighter threshold
        if long_open and not short_open:
            effective_long_threshold = threshold_percent * single_pos_multiplier
        else:
            effective_long_threshold = threshold_percent
            
        if short_open and not long_open:
            effective_short_threshold = threshold_percent * single_pos_multiplier
        else:
            effective_short_threshold = threshold_percent
        
        # Check trailing stops (entry-price relative)
        # LONG closes when current profit drops threshold% below max profit
        if long_pos and long_pos.is_open:
            current_profit = ((price - long_pos.entry_price) / long_pos.entry_price) * 100
            trigger_profit = long_pos.max_profit_pct - effective_long_threshold
            if current_profit <= trigger_profit:
                long_pos.is_open = False
                long_pos.exit_price = price
                long_pos.exit_time = timestamp
                # Deduct exit fee from P&L
                long_pos.pnl_percent = current_profit - fee_percent
                total_long_pnl += long_pos.pnl_percent
                total_fees += fee_percent
                
                if verbose:
                    print(f"  LONG closed @ ${price:.2f} | Profit: {long_pos.pnl_percent:+.2f}% (max was {long_pos.max_profit_pct:+.2f}%, threshold: {effective_long_threshold:.2f}%)")
        
        # SHORT closes when current profit drops threshold% below max profit
        if short_pos and short_pos.is_open:
            current_profit = ((short_pos.entry_price - price) / short_pos.entry_price) * 100
            trigger_profit = short_pos.max_profit_pct - effective_short_threshold
            if current_profit <= trigger_profit:
                short_pos.is_open = False
                short_pos.exit_price = price
                short_pos.exit_time = timestamp
                # Deduct exit fee from P&L
                short_pos.pnl_percent = current_profit - fee_percent
                total_short_pnl += short_pos.pnl_percent
                total_fees += fee_percent
                
                if verbose:
                    print(f"  SHORT closed @ ${price:.2f} | Profit: {short_pos.pnl_percent:+.2f}% (max was {short_pos.max_profit_pct:+.2f}%, threshold: {effective_short_threshold:.2f}%)")
        
        # If both positions closed, record round and reset
        if long_pos and short_pos and not long_pos.is_open and not short_pos.is_open:
            round_result = BacktestRound(
                round_num=current_round,
                entry_price=long_pos.entry_price,
                entry_time=long_pos.entry_time,
                long_exit_price=long_pos.exit_price,
                short_exit_price=short_pos.exit_price,
                long_pnl=long_pos.pnl_percent,
                short_pnl=short_pos.pnl_percent,
                duration_seconds=(long_pos.exit_time - long_pos.entry_time).total_seconds()
            )
            rounds.append(round_result)
            
            if verbose:
                print(f"  Round {current_round} complete: {round_result.total_pnl:+.2f}%\n")
            
            # Reset for next round
            long_pos = None
            short_pos = None
    
    # Calculate statistics
    if not rounds:
        return {
            'threshold': threshold_percent,
            'total_rounds': 0,
            'total_pnl': 0,
            'avg_pnl_per_round': 0,
            'win_rate': 0,
            'avg_duration_hours': 0,
            'best_round': 0,
            'worst_round': 0,
            'total_long_pnl': 0,
            'total_short_pnl': 0,
        }
    
    pnls = [r.total_pnl for r in rounds]
    winning_rounds = sum(1 for p in pnls if p > 0)
    
    return {
        'threshold': threshold_percent,
        'total_rounds': len(rounds),
        'total_pnl': sum(pnls),
        'avg_pnl_per_round': sum(pnls) / len(pnls),
        'win_rate': (winning_rounds / len(rounds)) * 100,
        'avg_duration_hours': sum(r.duration_seconds for r in rounds) / len(rounds) / 3600,
        'best_round': max(pnls),
        'worst_round': min(pnls),
        'total_long_pnl': total_long_pnl,
        'total_short_pnl': total_short_pnl,
        'rounds': rounds
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Hedge Trailing Strategy on Historical Data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-t", "--thresholds",
        type=str,
        default="0.5,1,2,3,5",
        help="Comma-separated threshold percentages to test (default: 0.5,1,2,3,5)"
    )
    
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=1095,
        help="Number of days of history to test (default: 1095 = 3 years)"
    )
    
    parser.add_argument(
        "-i", "--interval",
        type=str,
        default="1h",
        help="Candle interval: 1m, 5m, 15m, 1h, 4h, 1d (default: 1h)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="SOLUSDT",
        help="Trading pair (default: SOLUSDT)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed output for each round"
    )
    
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export results to CSV file"
    )
    
    parser.add_argument(
        "-m", "--multiplier",
        type=float,
        default=0.5,
        help="Single position threshold multiplier (default: 0.5)"
    )
    
    args = parser.parse_args()
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    
    print("=" * 70)
    print("HEDGE TRAILING STRATEGY - BACKTEST")
    print("=" * 70)
    print(f"Symbol:     {args.symbol}")
    print(f"Period:     {args.days} days (~{args.days/365:.1f} years)")
    print(f"Interval:   {args.interval} candles")
    print(f"Thresholds: {', '.join(f'{t}%' for t in thresholds)}")
    print("=" * 70)
    print()
    
    # Fetch historical data
    prices = fetch_historical_data(args.symbol, args.days, args.interval)
    
    if len(prices) < 100:
        print("Error: Not enough price data fetched. Check your internet connection.")
        return
    
    print(f"\nPrice range: ${min(p for _, p in prices):.2f} - ${max(p for _, p in prices):.2f}")
    print(f"Date range: {prices[0][0].strftime('%Y-%m-%d')} to {prices[-1][0].strftime('%Y-%m-%d')}")
    print()
    
    # Run backtests for each threshold
    results = []
    
    print("Running backtests...")
    print("-" * 70)
    
    for threshold in thresholds:
        print(f"\n  Testing {threshold}% threshold...", end=" ", flush=True)
        result = run_backtest(prices, threshold, single_pos_multiplier=args.multiplier, verbose=args.verbose)
        results.append(result)
        print(f"Done! {result['total_rounds']} rounds, {result['total_pnl']:+.1f}% total")
    
    # Sort by total P&L
    results.sort(key=lambda x: x['total_pnl'], reverse=True)
    
    # Print results table
    print("\n")
    print("=" * 70)
    print("BACKTEST RESULTS - RANKED BY TOTAL P&L")
    print("=" * 70)
    print()
    print(f"{'Rank':<5} {'Threshold':<10} {'Rounds':<8} {'Total P&L':<12} {'Avg/Round':<10} {'Win Rate':<10} {'Avg Hrs':<8}")
    print("-" * 70)
    
    for i, r in enumerate(results, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji}{i:<3} {r['threshold']:<10.1f}% {r['total_rounds']:<8} {r['total_pnl']:+10.2f}%  {r['avg_pnl_per_round']:+8.2f}%  {r['win_rate']:8.1f}%  {r['avg_duration_hours']:6.1f}h")
    
    print("-" * 70)
    
    # Winner summary
    winner = results[0]
    print()
    print("=" * 70)
    print(f"🏆 BEST THRESHOLD: {winner['threshold']}%")
    print("=" * 70)
    print(f"  Total Rounds:    {winner['total_rounds']}")
    print(f"  Total P&L:       {winner['total_pnl']:+.2f}%")
    print(f"  Average/Round:   {winner['avg_pnl_per_round']:+.2f}%")
    print(f"  Win Rate:        {winner['win_rate']:.1f}%")
    print(f"  Best Round:      {winner['best_round']:+.2f}%")
    print(f"  Worst Round:     {winner['worst_round']:+.2f}%")
    print(f"  Long P&L:        {winner['total_long_pnl']:+.2f}%")
    print(f"  Short P&L:       {winner['total_short_pnl']:+.2f}%")
    print("=" * 70)
    
    # Export if requested
    if args.export:
        export_path = args.export
        with open(export_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'threshold', 'total_rounds', 'total_pnl', 'avg_pnl_per_round',
                'win_rate', 'avg_duration_hours', 'best_round', 'worst_round',
                'total_long_pnl', 'total_short_pnl'
            ])
            writer.writeheader()
            for r in results:
                row = {k: v for k, v in r.items() if k != 'rounds'}
                writer.writerow(row)
        print(f"\n✓ Results exported to {export_path}")


if __name__ == "__main__":
    main()
