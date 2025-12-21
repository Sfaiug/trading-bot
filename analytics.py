#!/usr/bin/env python3
"""
Analytics Module for Pyramid Strategy Backtesting

Provides deep analysis functions for backtest results:
- Monthly P&L breakdown
- Drawdown analysis
- Minimum account size calculation
- Sharpe ratio
- Win/loss streak analysis
"""

from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class RoundSummary:
    """Simplified round data for analytics."""
    entry_time: datetime
    exit_time: datetime
    pnl_percent: float
    num_pyramids: int
    direction: str


def calculate_monthly_breakdown(rounds: List[RoundSummary]) -> Dict[str, Dict[str, float]]:
    """
    Group P&L by month.
    
    Returns:
        Dict with keys as 'YYYY-MM' and values containing:
        - pnl: total P&L for that month
        - rounds: number of rounds completed
        - wins: number of winning rounds
    """
    monthly = {}
    
    for r in rounds:
        month_key = r.exit_time.strftime('%Y-%m')
        
        if month_key not in monthly:
            monthly[month_key] = {'pnl': 0.0, 'rounds': 0, 'wins': 0}
        
        monthly[month_key]['pnl'] += r.pnl_percent
        monthly[month_key]['rounds'] += 1
        if r.pnl_percent > 0:
            monthly[month_key]['wins'] += 1
    
    return monthly


def calculate_max_drawdown(rounds: List[RoundSummary]) -> Tuple[float, int, str, str]:
    """
    Calculate maximum drawdown from backtest results.
    
    Returns:
        Tuple of (max_drawdown_pct, duration_rounds, start_date, end_date)
    """
    if not rounds:
        return 0.0, 0, "", ""
    
    # Track cumulative equity curve
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0
    
    # Track drawdown duration
    in_drawdown = False
    drawdown_start_idx = 0
    max_dd_start_idx = 0
    max_dd_end_idx = 0
    max_dd_duration = 0
    current_dd_duration = 0
    
    for i, r in enumerate(rounds):
        cumulative_pnl += r.pnl_percent
        
        if cumulative_pnl > peak_pnl:
            # New peak - reset drawdown tracking
            peak_pnl = cumulative_pnl
            in_drawdown = False
            current_dd_duration = 0
        else:
            # In drawdown
            current_drawdown = peak_pnl - cumulative_pnl
            
            if not in_drawdown:
                in_drawdown = True
                drawdown_start_idx = i
            
            current_dd_duration += 1
            
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                max_dd_start_idx = drawdown_start_idx
                max_dd_end_idx = i
                max_dd_duration = current_dd_duration
    
    start_date = rounds[max_dd_start_idx].entry_time.strftime('%Y-%m-%d') if rounds else ""
    end_date = rounds[max_dd_end_idx].exit_time.strftime('%Y-%m-%d') if rounds else ""
    
    return max_drawdown, max_dd_duration, start_date, end_date


def calculate_min_account_size(
    rounds: List[RoundSummary],
    max_pyramids: int,
    position_size_usd: float,
    leverage: int,
    safety_margin: float = 1.5
) -> Dict[str, float]:
    """
    Calculate minimum account size needed to survive worst drawdown.
    
    Args:
        rounds: List of round results
        max_pyramids: Maximum pyramid positions used
        position_size_usd: Dollar amount per position
        leverage: Leverage used
        safety_margin: Multiplier for safety (default 1.5x)
        
    Returns:
        Dict with:
        - min_account: Minimum account size needed
        - initial_margin: Margin needed for max positions
        - max_loss: Worst case loss in dollars (at position_size_usd)
        - recommended_account: Recommended account with safety margin
    """
    # Calculate worst case scenario
    max_drawdown_pct, _, _, _ = calculate_max_drawdown(rounds)
    
    # Maximum positions at any time: 2 (hedge) + max_pyramids
    max_positions = 2 + max_pyramids
    
    # Total notional at max positions
    max_notional = max_positions * position_size_usd
    
    # Margin required = notional / leverage
    initial_margin = max_notional / leverage
    
    # Worst case loss (at the specified position size)
    max_loss_pct = max_drawdown_pct
    max_loss_usd = (max_loss_pct / 100) * max_notional
    
    # Minimum account = margin + max loss buffer
    min_account = initial_margin + max_loss_usd
    
    # Recommended with safety margin
    recommended = min_account * safety_margin
    
    return {
        'min_account': round(min_account, 2),
        'initial_margin': round(initial_margin, 2),
        'max_loss_usd': round(max_loss_usd, 2),
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'recommended_account': round(recommended, 2)
    }


def calculate_sharpe_ratio(
    rounds: List[RoundSummary],
    risk_free_rate: float = 0.0,
    periods_per_year: float = 365
) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted returns).
    
    Args:
        rounds: List of round results
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: How many rounds per year on average
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(rounds) < 2:
        return 0.0
    
    returns = [r.pnl_percent for r in rounds]
    
    # Mean return
    mean_return = sum(returns) / len(returns)
    
    # Standard deviation
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0001
    
    if std_dev == 0:
        return 0.0
    
    # Estimate periods per year based on actual data
    if rounds:
        total_days = (rounds[-1].exit_time - rounds[0].entry_time).days
        if total_days > 0:
            periods_per_year = (len(rounds) / total_days) * 365
    
    # Annualized Sharpe: (mean - rf) / std * sqrt(periods)
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    sharpe = (excess_return / std_dev) * math.sqrt(periods_per_year)
    
    return round(sharpe, 2)


def analyze_win_loss_streaks(rounds: List[RoundSummary]) -> Dict[str, int]:
    """
    Analyze consecutive wins and losses.
    
    Returns:
        Dict with max_win_streak, max_loss_streak, current_streak
    """
    if not rounds:
        return {'max_win_streak': 0, 'max_loss_streak': 0, 'current_streak': 0}
    
    max_wins = 0
    max_losses = 0
    current_streak = 0
    current_type = None  # 'win' or 'loss'
    
    for r in rounds:
        is_win = r.pnl_percent > 0
        
        if current_type is None:
            current_type = 'win' if is_win else 'loss'
            current_streak = 1
        elif (is_win and current_type == 'win') or (not is_win and current_type == 'loss'):
            current_streak += 1
        else:
            # Streak broken
            if current_type == 'win':
                max_wins = max(max_wins, current_streak)
            else:
                max_losses = max(max_losses, current_streak)
            
            current_type = 'win' if is_win else 'loss'
            current_streak = 1
    
    # Check final streak
    if current_type == 'win':
        max_wins = max(max_wins, current_streak)
    else:
        max_losses = max(max_losses, current_streak)
    
    return {
        'max_win_streak': max_wins,
        'max_loss_streak': max_losses,
        'current_streak': current_streak,
        'current_streak_type': current_type or 'none'
    }


def generate_analytics_summary(
    rounds: List[RoundSummary],
    max_pyramids: int,
    position_size_usd: float = 10.0,
    leverage: int = 10
) -> Dict:
    """
    Generate complete analytics summary.
    
    Returns:
        Dict with all analytics metrics.
    """
    monthly = calculate_monthly_breakdown(rounds)
    max_dd, dd_duration, dd_start, dd_end = calculate_max_drawdown(rounds)
    account_sizing = calculate_min_account_size(rounds, max_pyramids, position_size_usd, leverage)
    sharpe = calculate_sharpe_ratio(rounds)
    streaks = analyze_win_loss_streaks(rounds)
    
    # Calculate additional stats
    total_pnl = sum(r.pnl_percent for r in rounds)
    winning_rounds = sum(1 for r in rounds if r.pnl_percent > 0)
    
    # Best and worst months
    best_month = max(monthly.items(), key=lambda x: x[1]['pnl']) if monthly else (None, {'pnl': 0})
    worst_month = min(monthly.items(), key=lambda x: x[1]['pnl']) if monthly else (None, {'pnl': 0})
    
    # Profitable months count
    profitable_months = sum(1 for m in monthly.values() if m['pnl'] > 0)
    
    return {
        'total_rounds': len(rounds),
        'total_pnl': round(total_pnl, 2),
        'win_rate': round((winning_rounds / len(rounds)) * 100, 1) if rounds else 0,
        'avg_pnl': round(total_pnl / len(rounds), 2) if rounds else 0,
        
        # Monthly analysis
        'monthly_breakdown': monthly,
        'profitable_months': profitable_months,
        'total_months': len(monthly),
        'best_month': {'month': best_month[0], 'pnl': round(best_month[1]['pnl'], 2)} if best_month[0] else None,
        'worst_month': {'month': worst_month[0], 'pnl': round(worst_month[1]['pnl'], 2)} if worst_month[0] else None,
        
        # Drawdown
        'max_drawdown': round(max_dd, 2),
        'drawdown_duration': dd_duration,
        'drawdown_period': f"{dd_start} to {dd_end}",
        
        # Account sizing
        'account_sizing': account_sizing,
        
        # Risk metrics
        'sharpe_ratio': sharpe,
        'streaks': streaks,
    }


def print_analytics_summary(summary: Dict, coin: str = ""):
    """Pretty print analytics summary."""
    print("\n" + "=" * 70)
    print(f"📊 DETAILED ANALYTICS{f' - {coin}' if coin else ''}")
    print("=" * 70)
    
    # Basic stats
    print(f"\n{'─'*35} PERFORMANCE {'─'*35}")
    print(f"  Total Rounds:    {summary['total_rounds']}")
    print(f"  Total P&L:       {summary['total_pnl']:+.2f}%")
    print(f"  Avg P&L/Round:   {summary['avg_pnl']:+.2f}%")
    print(f"  Win Rate:        {summary['win_rate']:.1f}%")
    print(f"  Sharpe Ratio:    {summary['sharpe_ratio']:.2f}")
    
    # Monthly
    print(f"\n{'─'*35} MONTHLY {'─'*35}")
    print(f"  Profitable Months: {summary['profitable_months']}/{summary['total_months']}")
    if summary['best_month']:
        print(f"  Best Month:      {summary['best_month']['month']} ({summary['best_month']['pnl']:+.2f}%)")
    if summary['worst_month']:
        print(f"  Worst Month:     {summary['worst_month']['month']} ({summary['worst_month']['pnl']:+.2f}%)")
    
    # Streaks
    print(f"\n{'─'*35} STREAKS {'─'*35}")
    print(f"  Max Win Streak:  {summary['streaks']['max_win_streak']} rounds")
    print(f"  Max Loss Streak: {summary['streaks']['max_loss_streak']} rounds")
    
    # Risk
    print(f"\n{'─'*35} RISK {'─'*35}")
    print(f"  Max Drawdown:    {summary['max_drawdown']:.2f}%")
    print(f"  DD Duration:     {summary['drawdown_duration']} rounds")
    print(f"  DD Period:       {summary['drawdown_period']}")
    
    # Account sizing
    acct = summary['account_sizing']
    print(f"\n{'─'*35} ACCOUNT SIZING {'─'*35}")
    print(f"  Minimum Account: ${acct['min_account']:.2f}")
    print(f"  Recommended:     ${acct['recommended_account']:.2f} (1.5x safety)")
    print(f"  Initial Margin:  ${acct['initial_margin']:.2f}")
    print(f"  Max Loss:        ${acct['max_loss_usd']:.2f}")
    
    print("=" * 70)
