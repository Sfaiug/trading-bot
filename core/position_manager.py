"""
Position management for tracking hedged positions.
Uses entry-price-relative profit tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Represents a single trading position."""
    side: PositionSide
    entry_price: float
    quantity: float
    entry_time: datetime = field(default_factory=datetime.now)
    
    # Tracking max profit (entry-price relative)
    max_profit_pct: float = 0.0  # Highest profit % reached since position opened
    
    # Status
    is_open: bool = True
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl_percent: Optional[float] = None
    
    def get_current_profit(self, current_price: float) -> float:
        """Calculate current profit percentage relative to entry price."""
        if self.side == PositionSide.LONG:
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100
    
    def update_max_profit(self, current_price: float) -> bool:
        """Update max profit if current profit exceeds it. Returns True if updated."""
        current_profit = self.get_current_profit(current_price)
        if current_profit > self.max_profit_pct:
            self.max_profit_pct = current_profit
            return True
        return False
    
    def should_close(self, current_price: float, threshold_pct: float) -> bool:
        """Check if position should close based on trailing stop."""
        current_profit = self.get_current_profit(current_price)
        trigger_profit = self.max_profit_pct - threshold_pct
        return current_profit <= trigger_profit
    
    def close(self, exit_price: float):
        """Mark position as closed."""
        self.is_open = False
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.pnl_percent = self.get_current_profit(exit_price)


class PositionManager:
    """Manages hedged positions (one long + one short)."""
    
    def __init__(self):
        self.long_position: Optional[Position] = None
        self.short_position: Optional[Position] = None
        self.round_number: int = 0
        self.history: list = []
    
    def open_hedged_positions(self, entry_price: float, quantity: float):
        """
        Create both long and short positions at the same entry price.
        
        Args:
            entry_price: Current market price at entry
            quantity: Amount for each position
        """
        self.round_number += 1
        
        self.long_position = Position(
            side=PositionSide.LONG,
            entry_price=entry_price,
            quantity=quantity
        )
        
        self.short_position = Position(
            side=PositionSide.SHORT,
            entry_price=entry_price,
            quantity=quantity
        )
        
        print(f"\n{'='*60}")
        print(f"ROUND {self.round_number} STARTED")
        print(f"{'='*60}")
        print(f"Entry Price: ${entry_price:.4f}")
        print(f"Quantity: {quantity}")
    
    def update_max_profits(self, current_price: float):
        """Update max profit for open positions."""
        if self.long_position and self.long_position.is_open:
            if self.long_position.update_max_profit(current_price):
                print(f"  📈 New LONG max profit: {self.long_position.max_profit_pct:+.2f}%")
        
        if self.short_position and self.short_position.is_open:
            if self.short_position.update_max_profit(current_price):
                print(f"  📉 New SHORT max profit: {self.short_position.max_profit_pct:+.2f}%")
    
    def close_long(self, exit_price: float):
        """Close the long position."""
        if self.long_position and self.long_position.is_open:
            self.long_position.close(exit_price)
            self.history.append(self.long_position)
            
            pnl = self.long_position.pnl_percent
            emoji = "✅" if pnl >= 0 else "❌"
            print(f"\n{emoji} LONG CLOSED @ ${exit_price:.4f}")
            print(f"   Entry: ${self.long_position.entry_price:.4f}")
            print(f"   Max Profit: {self.long_position.max_profit_pct:+.2f}%")
            print(f"   P&L: {pnl:+.2f}%")
    
    def close_short(self, exit_price: float):
        """Close the short position."""
        if self.short_position and self.short_position.is_open:
            self.short_position.close(exit_price)
            self.history.append(self.short_position)
            
            pnl = self.short_position.pnl_percent
            emoji = "✅" if pnl >= 0 else "❌"
            print(f"\n{emoji} SHORT CLOSED @ ${exit_price:.4f}")
            print(f"   Entry: ${self.short_position.entry_price:.4f}")
            print(f"   Max Profit: {self.short_position.max_profit_pct:+.2f}%")
            print(f"   P&L: {pnl:+.2f}%")
    
    def has_open_positions(self) -> bool:
        """Check if any positions are still open."""
        long_open = self.long_position and self.long_position.is_open
        short_open = self.short_position and self.short_position.is_open
        return long_open or short_open
    
    def is_long_open(self) -> bool:
        """Check if long position is open."""
        return self.long_position and self.long_position.is_open
    
    def is_short_open(self) -> bool:
        """Check if short position is open."""
        return self.short_position and self.short_position.is_open
    
    def get_round_summary(self) -> dict:
        """Get summary of the current/last round."""
        long_pnl = self.long_position.pnl_percent if self.long_position else 0
        short_pnl = self.short_position.pnl_percent if self.short_position else 0
        
        return {
            'round': self.round_number,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'total_pnl': (long_pnl or 0) + (short_pnl or 0)
        }
    
    def print_round_summary(self):
        """Print summary after both positions closed."""
        summary = self.get_round_summary()
        print(f"\n{'='*60}")
        print(f"ROUND {summary['round']} COMPLETE")
        print(f"{'='*60}")
        print(f"  LONG P&L:  {summary['long_pnl']:+.2f}%")
        print(f"  SHORT P&L: {summary['short_pnl']:+.2f}%")
        print(f"  TOTAL:     {summary['total_pnl']:+.2f}%")
        print(f"{'='*60}\n")
