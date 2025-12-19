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
from typing import List, Tuple, Dict
from dataclasses import dataclass, field


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


def run_pyramid_backtest(
    prices: List[Tuple[datetime, float]],
    threshold_pct: float,
    trailing_pct: float,
    pyramid_step_pct: float,
    fee_pct: float = 0.04,
    max_pyramids: int = 20,
    verbose: bool = False
) -> Dict:
    """
    Run pyramid strategy backtest.
    
    Args:
        prices: List of (timestamp, price) tuples
        threshold_pct: % move that closes the losing hedge side
        trailing_pct: % drop from max profit that closes all positions
        pyramid_step_pct: % interval to add new positions (from pyramid reference)
        fee_pct: Trading fee per trade
        max_pyramids: Maximum number of pyramid positions
        verbose: Print detailed output
        
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
    
    total_pnl = 0.0
    total_fees = 0.0
    total_rounds = 0
    winning_rounds = 0
    
    for timestamp, price in prices:
        
        # === PHASE 1: Open initial hedge if no positions ===
        if long_pos is None and short_pos is None and not pyramid_positions:
            long_pos = PyramidPosition(side='LONG', entry_price=price, entry_time=timestamp)
            short_pos = PyramidPosition(side='SHORT', entry_price=price, entry_time=timestamp)
            round_entry_price = price
            round_entry_time = timestamp
            direction = ''
            pyramid_reference = 0.0
            max_profit_pct = 0.0
            next_pyramid_level = 1
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
            
            # Calculate current total profit (average of all positions)
            total_profit = 0.0
            for pos in pyramid_positions:
                total_profit += calculate_profit_pct(pos.entry_price, price, is_long)
            avg_profit = total_profit / len(pyramid_positions)
            
            # Update max profit
            if avg_profit > max_profit_pct:
                max_profit_pct = avg_profit
            
            # Check for new pyramid level
            if len(pyramid_positions) < max_pyramids:
                # Calculate how far price has moved from pyramid reference
                if is_long:
                    move_from_ref = ((price - pyramid_reference) / pyramid_reference) * 100
                else:
                    move_from_ref = ((pyramid_reference - price) / pyramid_reference) * 100
                
                # Check if we've reached next pyramid level
                pyramid_threshold = next_pyramid_level * pyramid_step_pct
                if move_from_ref >= pyramid_threshold:
                    # Add new pyramid position
                    new_pos = PyramidPosition(
                        side=direction,
                        entry_price=price,
                        entry_time=timestamp
                    )
                    pyramid_positions.append(new_pos)
                    total_fees += fee_pct
                    next_pyramid_level += 1
                    
                    if verbose:
                        print(f"[{timestamp}] PYRAMID #{len(pyramid_positions)} @ ${price:.2f} (+{move_from_ref:.1f}% from ref)")
            
            # Check trailing stop
            trigger_profit = max_profit_pct - trailing_pct
            if avg_profit <= trigger_profit:
                # Close all pyramid positions
                round_pnl = 0.0
                
                for pos in pyramid_positions:
                    pos.is_open = False
                    pos.exit_price = price
                    pos.pnl_percent = calculate_profit_pct(pos.entry_price, price, is_long) - fee_pct
                    round_pnl += pos.pnl_percent
                    total_fees += fee_pct
                
                # Add the losing hedge P&L
                losing_pnl = -threshold_pct - fee_pct
                round_pnl += losing_pnl
                
                total_pnl += round_pnl
                total_rounds += 1
                if round_pnl > 0:
                    winning_rounds += 1
                
                if verbose:
                    print(f"[{timestamp}] ALL CLOSED @ ${price:.2f} | Pyramids: {len(pyramid_positions)} | "
                          f"Max: {max_profit_pct:+.2f}% | Round P&L: {round_pnl:+.2f}%")
                
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
