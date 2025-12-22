#!/usr/bin/env python3
"""
Pyramid Strategy Backtest Module

Strategy Logic:
1. Open hedge at entry price (1 LONG + 1 SHORT)
2. Losing side closes when price moves threshold% against it
3. Pyramid reference = price where losing side closed
4. Add positions every pyramid_step% from pyramid reference
5. All positions close when profit drops trailing% from max
"""

from datetime import datetime
from typing import List, Tuple, Dict, Iterator, Union, Iterable, Optional
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class PyramidPosition:
    """A single position in the pyramid."""
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    size: float = 1.0
    is_open: bool = True
    exit_price: float = 0.0
    pnl_percent: float = 0.0


@dataclass 
class PyramidRound:
    """Tracks a complete pyramid round."""
    entry_price: float
    entry_time: datetime
    direction: str = ''  # 'LONG' or 'SHORT' (winning direction)
    pyramid_reference: float = 0.0  # Price where losing side closed
    positions: List[PyramidPosition] = field(default_factory=list)
    max_profit_pct: float = 0.0
    exit_price: float = 0.0
    exit_time: datetime = None
    total_pnl: float = 0.0
    num_pyramids: int = 0


def calculate_profit_pct(entry_price: float, current_price: float, is_long: bool) -> float:
    """Calculate profit percentage."""
    if is_long:
        return ((current_price - entry_price) / entry_price) * 100
    else:
        return ((entry_price - current_price) / entry_price) * 100


def _calculate_pyramid_size(level: int, schedule: str) -> float:
    """
    Calculate position size based on pyramid level and schedule.

    Args:
        level: Pyramid level (1-based)
        schedule: 'fixed', 'linear_decay', or 'exp_decay'

    Returns:
        Position size multiplier
    """
    if schedule == 'fixed':
        return 1.0
    elif schedule == 'linear_decay':
        # Size decreases linearly: 1.0, 0.8, 0.6, 0.4, 0.2 (min 0.2)
        return max(1.0 - (level - 1) * 0.2, 0.2)
    elif schedule == 'exp_decay':
        # Size decreases exponentially: 1.0, 0.7, 0.49, 0.34...
        return 0.7 ** (level - 1)
    return 1.0


def _calculate_volatility(prices: deque, method: str) -> float:
    """
    Calculate volatility from recent prices.

    Args:
        prices: Recent price history
        method: 'none', 'stddev', or 'range'

    Returns:
        Volatility measure (higher = more volatile)
    """
    if method == 'none' or len(prices) < 10:
        return float('inf')  # Always pass filter

    prices_list = list(prices)

    if method == 'stddev':
        # Standard deviation of returns
        returns = [
            (prices_list[i] - prices_list[i-1]) / prices_list[i-1] * 100
            for i in range(1, len(prices_list))
        ]
        return statistics.stdev(returns) if len(returns) > 1 else 0.0

    elif method == 'range':
        # Range as % of mean
        min_price = min(prices_list)
        max_price = max(prices_list)
        avg_price = sum(prices_list) / len(prices_list)
        return ((max_price - min_price) / avg_price) * 100

    return float('inf')


def run_pyramid_backtest(
    prices: Iterable[Tuple[datetime, float]],
    threshold_pct: float,
    trailing_pct: float,
    pyramid_step_pct: float,
    fee_pct: float = 0.04,
    max_pyramids: int = 20,
    verbose: bool = False,
    # NEW PARAMETERS:
    pyramid_size_schedule: str = 'fixed',
    min_pyramid_spacing_pct: float = 0.0,
    pyramid_acceleration: float = 1.0,
    time_decay_exit_seconds: Optional[float] = None,
    volatility_filter_type: str = 'none',
    volatility_min_pct: float = 0.0,
    volatility_window_size: int = 100
) -> Dict:
    """
    Run pyramid strategy backtest.

    Args:
        prices: Iterable of (timestamp, price) tuples (list or generator for streaming)
        threshold_pct: % move that closes the losing hedge side
        trailing_pct: % drop from max profit that closes all positions
        pyramid_step_pct: % interval to add new positions (from pyramid reference)
        fee_pct: Trading fee per trade
        max_pyramids: Maximum number of pyramid positions
        verbose: Print detailed output

        # NEW PARAMETERS:
        pyramid_size_schedule: Position sizing ('fixed', 'linear_decay', 'exp_decay')
        min_pyramid_spacing_pct: Minimum % between pyramid entries (0 = disabled)
        pyramid_acceleration: Pyramid spacing multiplier (1.0 = linear, >1 = exponential)
        time_decay_exit_seconds: Force exit after N seconds (None = disabled)
        volatility_filter_type: Volatility filter ('none', 'stddev', 'range')
        volatility_min_pct: Minimum volatility to add pyramids (0 = disabled)
        volatility_window_size: Number of prices for volatility calculation

    Returns:
        Dictionary with backtest results
    """
    rounds: List[PyramidRound] = []

    # State tracking
    long_pos: PyramidPosition = None
    short_pos: PyramidPosition = None
    pyramid_positions: List[PyramidPosition] = []
    pyramid_reference: float = 0.0
    direction: str = ''
    max_profit_pct: float = 0.0
    next_pyramid_level: int = 1
    round_entry_price: float = 0.0
    round_entry_time: datetime = None

    # NEW: Additional state for new parameters
    last_pyramid_price: float = 0.0  # For min_pyramid_spacing
    round_start_time: datetime = None  # For time_decay_exit
    recent_prices: deque = deque(maxlen=volatility_window_size)  # For volatility filter

    total_pnl = 0.0
    total_fees = 0.0
    total_rounds = 0
    winning_rounds = 0
    
    for timestamp, price in prices:
        # Track recent prices for volatility calculation
        recent_prices.append(price)

        # === PHASE 1: Open initial hedge if no positions ===
        if long_pos is None and short_pos is None and not pyramid_positions:
            long_pos = PyramidPosition(side='LONG', entry_price=price, entry_time=timestamp)
            short_pos = PyramidPosition(side='SHORT', entry_price=price, entry_time=timestamp)
            round_entry_price = price
            round_entry_time = timestamp
            round_start_time = timestamp  # NEW: Track for time_decay_exit
            direction = ''
            pyramid_reference = 0.0
            max_profit_pct = 0.0
            next_pyramid_level = 1
            last_pyramid_price = 0.0  # NEW: Reset for min_pyramid_spacing
            total_fees += 2 * fee_pct  # Entry fees

            if verbose:
                print(f"\n[{timestamp}] OPENED HEDGE @ ${price:.2f}")
            continue
        
        # === PHASE 2: Check if losing side should close (determines direction) ===
        if direction == '':
            # Calculate profits for both sides
            long_profit = calculate_profit_pct(long_pos.entry_price, price, is_long=True)
            short_profit = calculate_profit_pct(short_pos.entry_price, price, is_long=False)
            
            # Check if LONG loses (price dropped threshold%)
            if long_profit <= -threshold_pct:
                # SHORT wins, LONG loses
                direction = 'SHORT'
                pyramid_reference = price
                long_pos.is_open = False
                long_pos.exit_price = price
                long_pos.pnl_percent = long_profit - fee_pct
                total_fees += fee_pct
                
                # SHORT becomes first pyramid position
                pyramid_positions = [short_pos]
                short_pos = None
                long_pos = None
                max_profit_pct = short_profit
                
                if verbose:
                    print(f"[{timestamp}] LONG CLOSED @ ${price:.2f} ({long_profit:+.2f}%) → Direction: SHORT")
            
            # Check if SHORT loses (price rose threshold%)
            elif short_profit <= -threshold_pct:
                # LONG wins, SHORT loses
                direction = 'LONG'
                pyramid_reference = price
                short_pos.is_open = False
                short_pos.exit_price = price
                short_pos.pnl_percent = short_profit - fee_pct
                total_fees += fee_pct
                
                # LONG becomes first pyramid position
                pyramid_positions = [long_pos]
                long_pos = None
                short_pos = None
                max_profit_pct = long_profit
                
                if verbose:
                    print(f"[{timestamp}] SHORT CLOSED @ ${price:.2f} ({short_profit:+.2f}%) → Direction: LONG")
        
        # === PHASE 3: Direction established - manage pyramid ===
        if direction != '' and pyramid_positions:
            is_long = (direction == 'LONG')
            
            # Calculate profit from ORIGINAL ENTRY PRICE (Option B)
            if is_long:
                profit_from_entry = ((price - round_entry_price) / round_entry_price) * 100
            else:
                profit_from_entry = ((round_entry_price - price) / round_entry_price) * 100
            
            # Update max profit (entry-relative)
            if profit_from_entry > max_profit_pct:
                max_profit_pct = profit_from_entry
            
            # Check for new pyramid level
            if len(pyramid_positions) < max_pyramids:
                # Calculate how far price has moved from pyramid reference
                if is_long:
                    move_from_ref = ((price - pyramid_reference) / pyramid_reference) * 100
                else:
                    move_from_ref = ((pyramid_reference - price) / pyramid_reference) * 100

                # Calculate pyramid threshold with acceleration
                if pyramid_acceleration == 1.0:
                    # Linear spacing (original behavior)
                    pyramid_threshold = next_pyramid_level * pyramid_step_pct
                else:
                    # Exponential spacing: sum of geometric series
                    pyramid_threshold = pyramid_step_pct * sum(
                        pyramid_acceleration ** i for i in range(next_pyramid_level)
                    )

                # Check if we've reached next pyramid level
                if move_from_ref >= pyramid_threshold:
                    # NEW: Check minimum spacing from last pyramid
                    spacing_ok = True
                    if last_pyramid_price > 0 and min_pyramid_spacing_pct > 0:
                        if is_long:
                            spacing = ((price - last_pyramid_price) / last_pyramid_price) * 100
                        else:
                            spacing = ((last_pyramid_price - price) / last_pyramid_price) * 100
                        spacing_ok = spacing >= min_pyramid_spacing_pct

                    # NEW: Check volatility filter
                    current_vol = _calculate_volatility(recent_prices, volatility_filter_type)
                    volatility_ok = current_vol >= volatility_min_pct

                    if spacing_ok and volatility_ok:
                        # Add new pyramid position with calculated size
                        pyramid_level = len(pyramid_positions) + 1
                        pos_size = _calculate_pyramid_size(pyramid_level, pyramid_size_schedule)

                        new_pos = PyramidPosition(
                            side=direction,
                            entry_price=price,
                            entry_time=timestamp,
                            size=pos_size
                        )
                        pyramid_positions.append(new_pos)
                        last_pyramid_price = price  # Track for spacing check
                        total_fees += fee_pct
                        next_pyramid_level += 1

                        if verbose:
                            print(f"[{timestamp}] PYRAMID #{len(pyramid_positions)} @ ${price:.2f} "
                                  f"(+{move_from_ref:.1f}% from ref, size={pos_size:.2f})")
                    elif verbose and not spacing_ok:
                        print(f"[{timestamp}] PYRAMID SKIPPED (spacing {spacing:.2f}% < min {min_pyramid_spacing_pct}%)")
                    elif verbose and not volatility_ok:
                        print(f"[{timestamp}] PYRAMID SKIPPED (vol {current_vol:.2f}% < min {volatility_min_pct}%)")
            
            # Check time-based exit (NEW)
            time_exit = False
            if time_decay_exit_seconds is not None and round_start_time is not None:
                time_elapsed = (timestamp - round_start_time).total_seconds()
                time_exit = time_elapsed >= time_decay_exit_seconds

            # Check trailing stop (entry-relative)
            trigger_profit = max_profit_pct - trailing_pct
            trail_exit = profit_from_entry <= trigger_profit

            if time_exit or trail_exit:
                # Close all pyramid positions
                round_pnl = 0.0
                
                for pos in pyramid_positions:
                    pos.is_open = False
                    pos.exit_price = price
                    pos.pnl_percent = calculate_profit_pct(pos.entry_price, price, is_long) - fee_pct
                    round_pnl += pos.pnl_percent
                    total_fees += fee_pct
                
                # Add the losing hedge P&L (includes its exit fee)
                losing_pnl = -threshold_pct - fee_pct
                round_pnl += losing_pnl
                
                # Deduct entry fees (2 positions opened at start)
                round_pnl -= 2 * fee_pct
                
                total_pnl += round_pnl
                total_rounds += 1
                if round_pnl > 0:
                    winning_rounds += 1
                
                if verbose:
                    exit_reason = "TIME DECAY" if time_exit else "TRAILING STOP"
                    print(f"[{timestamp}] {exit_reason} @ ${price:.2f} | Pyramids: {len(pyramid_positions)} | "
                          f"Max: {max_profit_pct:+.2f}% | Profit: {profit_from_entry:+.2f}% | Round P&L: {round_pnl:+.2f}%")

                # Record round
                rounds.append(PyramidRound(
                    entry_price=round_entry_price,
                    entry_time=round_entry_time,
                    direction=direction,
                    pyramid_reference=pyramid_reference,
                    positions=pyramid_positions.copy(),
                    max_profit_pct=max_profit_pct,
                    exit_price=price,
                    exit_time=timestamp,
                    total_pnl=round_pnl,
                    num_pyramids=len(pyramid_positions)
                ))

                # Reset for next round
                pyramid_positions = []
                direction = ''
                pyramid_reference = 0.0
                max_profit_pct = 0.0
                next_pyramid_level = 1
                last_pyramid_price = 0.0  # NEW: Reset for min_pyramid_spacing
    
    # Calculate statistics
    if total_rounds == 0:
        return {
            'threshold': threshold_pct,
            'trailing': trailing_pct,
            'pyramid_step': pyramid_step_pct,
            'total_rounds': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'win_rate': 0,
            'avg_pyramids': 0,
            'total_fees': total_fees
        }
    
    avg_pyramids = sum(r.num_pyramids for r in rounds) / len(rounds) if rounds else 0
    
    return {
        'threshold': threshold_pct,
        'trailing': trailing_pct,
        'pyramid_step': pyramid_step_pct,
        'total_rounds': total_rounds,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / total_rounds,
        'win_rate': (winning_rounds / total_rounds) * 100,
        'avg_pyramids': avg_pyramids,
        'total_fees': total_fees,
        'rounds': rounds
    }


# Test function
if __name__ == "__main__":
    # Simple test with synthetic data
    print("Testing pyramid backtest...")
    
    # Create test prices: $100 -> $110 -> $105
    from datetime import timedelta
    
    test_prices = []
    base_time = datetime.now()
    
    # Uptrend
    for i in range(100):
        price = 100 + i * 0.1
        test_prices.append((base_time + timedelta(minutes=i), price))
    
    # Reversal
    for i in range(50):
        price = 110 - i * 0.1
        test_prices.append((base_time + timedelta(minutes=100+i), price))
    
    result = run_pyramid_backtest(
        test_prices,
        threshold_pct=1.0,
        trailing_pct=1.0,
        pyramid_step_pct=2.0,
        verbose=True
    )
    
    print(f"\nResult: {result['total_pnl']:+.2f}% over {result['total_rounds']} rounds")
    print(f"Avg pyramids: {result['avg_pyramids']:.1f}")
