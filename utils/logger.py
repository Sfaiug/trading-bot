"""
Logging utilities for trade tracking and analysis.
"""

import os
import csv
from datetime import datetime
from typing import List, Dict, Any

from config.settings import LOG_DIR, TRADE_LOG_FILE


def ensure_log_dir():
    """Create logs directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)


def log_trade(
    round_number: int,
    side: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    peak_price: float,
    trough_price: float,
    pnl_percent: float,
    threshold_percent: float,
    entry_time: datetime,
    exit_time: datetime
):
    """
    Log a completed trade to CSV file.
    
    Args:
        round_number: Trading round number
        side: 'LONG' or 'SHORT'
        entry_price: Price at entry
        exit_price: Price at exit
        quantity: Position size
        peak_price: Highest price reached (for longs)
        trough_price: Lowest price reached (for shorts)
        pnl_percent: Profit/loss percentage
        threshold_percent: Trailing stop threshold used
        entry_time: When position was opened
        exit_time: When position was closed
    """
    ensure_log_dir()
    
    file_exists = os.path.exists(TRADE_LOG_FILE)
    
    headers = [
        'timestamp',
        'round',
        'side',
        'entry_price',
        'exit_price',
        'quantity',
        'peak_price',
        'trough_price',
        'pnl_percent',
        'threshold_percent',
        'entry_time',
        'exit_time',
        'duration_seconds'
    ]
    
    duration = (exit_time - entry_time).total_seconds()
    
    row = {
        'timestamp': datetime.now().isoformat(),
        'round': round_number,
        'side': side,
        'entry_price': f"{entry_price:.6f}",
        'exit_price': f"{exit_price:.6f}",
        'quantity': quantity,
        'peak_price': f"{peak_price:.6f}",
        'trough_price': f"{trough_price:.6f}",
        'pnl_percent': f"{pnl_percent:.4f}",
        'threshold_percent': threshold_percent,
        'entry_time': entry_time.isoformat(),
        'exit_time': exit_time.isoformat(),
        'duration_seconds': f"{duration:.1f}"
    }
    
    with open(TRADE_LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_round_summary(
    round_number: int,
    long_pnl: float,
    short_pnl: float,
    total_pnl: float,
    threshold_percent: float
):
    """Print and optionally log round summary."""
    summary_file = os.path.join(LOG_DIR, "rounds.csv")
    ensure_log_dir()
    
    file_exists = os.path.exists(summary_file)
    
    headers = ['timestamp', 'round', 'long_pnl', 'short_pnl', 'total_pnl', 'threshold_percent']
    
    row = {
        'timestamp': datetime.now().isoformat(),
        'round': round_number,
        'long_pnl': f"{long_pnl:.4f}",
        'short_pnl': f"{short_pnl:.4f}",
        'total_pnl': f"{total_pnl:.4f}",
        'threshold_percent': threshold_percent
    }
    
    with open(summary_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_session_stats() -> Dict[str, Any]:
    """
    Calculate statistics from current session's trade log.
    
    Returns:
        Dictionary with session statistics
    """
    if not os.path.exists(TRADE_LOG_FILE):
        return {'total_trades': 0, 'message': 'No trades logged yet'}
    
    with open(TRADE_LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        trades = list(reader)
    
    if not trades:
        return {'total_trades': 0, 'message': 'No trades logged yet'}
    
    pnls = [float(t['pnl_percent']) for t in trades]
    
    return {
        'total_trades': len(trades),
        'total_pnl': sum(pnls),
        'avg_pnl': sum(pnls) / len(pnls),
        'winning_trades': sum(1 for p in pnls if p > 0),
        'losing_trades': sum(1 for p in pnls if p < 0),
        'best_trade': max(pnls),
        'worst_trade': min(pnls)
    }


def print_session_stats():
    """Print formatted session statistics."""
    stats = get_session_stats()
    
    if stats.get('message'):
        print(stats['message'])
        return
    
    print(f"\n{'='*60}")
    print("SESSION STATISTICS")
    print(f"{'='*60}")
    print(f"  Total Trades:   {stats['total_trades']}")
    print(f"  Winning Trades: {stats['winning_trades']}")
    print(f"  Losing Trades:  {stats['losing_trades']}")
    print(f"  Win Rate:       {stats['winning_trades']/stats['total_trades']*100:.1f}%")
    print(f"  Total P&L:      {stats['total_pnl']:+.2f}%")
    print(f"  Average P&L:    {stats['avg_pnl']:+.2f}%")
    print(f"  Best Trade:     {stats['best_trade']:+.2f}%")
    print(f"  Worst Trade:    {stats['worst_trade']:+.2f}%")
    print(f"{'='*60}\n")
