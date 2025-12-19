#!/usr/bin/env python3
"""
Pyramid Trading Strategy - Live Trading

Strategy Logic:
1. Open hedge at entry price (1 LONG + 1 SHORT)
2. Losing side closes when price moves threshold% against it
3. Pyramid reference = price where losing side closed
4. Add positions every pyramid_step% from pyramid reference
5. All positions close when profit from entry drops trailing% from max
"""

import time
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field

from core.exchange import BinanceExchange
from config.settings import PRICE_CHECK_INTERVAL


@dataclass
class Position:
    """A single position in the pyramid."""
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    size: float = 1.0
    is_open: bool = True


class PyramidTradingStrategy:
    """
    Live pyramid trading strategy.
    Opens hedge, pyramids into winning direction, trails profit.
    """
    
    def __init__(
        self,
        exchange: BinanceExchange,
        symbol: str,
        threshold_pct: float = 1.0,
        trailing_pct: float = 1.0,
        pyramid_step_pct: float = 2.0,
        position_size: float = 1.0,
        max_pyramids: int = 10,
        max_rounds: Optional[int] = None
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.threshold_pct = threshold_pct
        self.trailing_pct = trailing_pct
        self.pyramid_step_pct = pyramid_step_pct
        self.position_size = position_size
        self.max_pyramids = max_pyramids
        self.max_rounds = max_rounds
        
        # State
        self.running = False
        self.round_number = 0
        self.total_pnl = 0.0
        
        # Current round state
        self.long_pos: Position = None
        self.short_pos: Position = None
        self.pyramid_positions: List[Position] = []
        self.pyramid_reference: float = 0.0
        self.direction: str = ''
        self.max_profit_pct: float = 0.0
        self.next_pyramid_level: int = 1
        self.round_entry_price: float = 0.0
    
    def run(self):
        """Main strategy loop."""
        self.running = True
        
        self._print_header()
        
        try:
            while self.running:
                if self.max_rounds and self.round_number >= self.max_rounds:
                    print(f"\n✓ Completed {self.max_rounds} rounds. Stopping.")
                    break
                
                # Start new round if no positions
                if not self._has_positions():
                    self._open_hedge()
                
                # Get current price
                price = self.exchange.get_price(self.symbol)
                
                # Process price update
                self._process_price(price)
                
                time.sleep(PRICE_CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            self._cleanup()
        
        self._print_summary()
    
    def _print_header(self):
        print(f"\n{'#'*60}")
        print(f"PYRAMID TRADING STRATEGY")
        print(f"{'#'*60}")
        print(f"Symbol:       {self.symbol}")
        print(f"Threshold:    {self.threshold_pct}% (losing side closes)")
        print(f"Trailing:     {self.trailing_pct}% (profit lock)")
        print(f"Pyramid Step: {self.pyramid_step_pct}%")
        print(f"Size:         {self.position_size}")
        print(f"Max Pyramids: {self.max_pyramids}")
        print(f"Max Rounds:   {self.max_rounds or 'Unlimited'}")
        print(f"{'#'*60}\n")
    
    def _has_positions(self) -> bool:
        return (self.long_pos is not None or 
                self.short_pos is not None or 
                len(self.pyramid_positions) > 0)
    
    def _open_hedge(self):
        """Open initial LONG + SHORT hedge."""
        self.round_number += 1
        price = self.exchange.get_price(self.symbol)
        
        print(f"\n{'='*60}")
        print(f"ROUND {self.round_number} - Opening hedge @ ${price:.4f}")
        print(f"{'='*60}")
        
        # Open positions on exchange
        self.exchange.open_hedged_positions(self.symbol, self.position_size)
        
        # Track locally
        self.long_pos = Position(side='LONG', entry_price=price, 
                                  entry_time=datetime.now(), size=self.position_size)
        self.short_pos = Position(side='SHORT', entry_price=price,
                                   entry_time=datetime.now(), size=self.position_size)
        
        self.round_entry_price = price
        self.direction = ''
        self.pyramid_reference = 0.0
        self.max_profit_pct = 0.0
        self.next_pyramid_level = 1
        self.pyramid_positions = []
    
    def _process_price(self, price: float):
        """Process a price update."""
        
        # Phase 1: Determine direction (if not set)
        if self.direction == '':
            self._check_direction(price)
        
        # Phase 2: Manage pyramid (if direction set)
        if self.direction != '' and self.pyramid_positions:
            self._manage_pyramid(price)
    
    def _check_direction(self, price: float):
        """Check if losing side should close to determine direction."""
        if not self.long_pos or not self.short_pos:
            return
        
        entry = self.long_pos.entry_price
        long_profit = ((price - entry) / entry) * 100
        short_profit = ((entry - price) / entry) * 100
        
        # LONG loses
        if long_profit <= -self.threshold_pct:
            self.direction = 'SHORT'
            self.pyramid_reference = price
            
            # Close LONG on exchange
            self.exchange.close_long_hedge(self.symbol, self.position_size)
            
            print(f"\n❌ LONG CLOSED @ ${price:.4f} ({long_profit:+.2f}%) → Direction: SHORT")
            
            # SHORT becomes first pyramid position
            self.pyramid_positions = [self.short_pos]
            self.long_pos = None
            self.short_pos = None
            self.max_profit_pct = short_profit
        
        # SHORT loses
        elif short_profit <= -self.threshold_pct:
            self.direction = 'LONG'
            self.pyramid_reference = price
            
            # Close SHORT on exchange
            self.exchange.close_short_hedge(self.symbol, self.position_size)
            
            print(f"\n❌ SHORT CLOSED @ ${price:.4f} ({short_profit:+.2f}%) → Direction: LONG")
            
            # LONG becomes first pyramid position
            self.pyramid_positions = [self.long_pos]
            self.long_pos = None
            self.short_pos = None
            self.max_profit_pct = long_profit
        
        else:
            # Still waiting for direction
            self._print_status(price, long_profit, short_profit)
    
    def _manage_pyramid(self, price: float):
        """Manage pyramid positions - add new ones, check trailing stop."""
        is_long = (self.direction == 'LONG')
        
        # Calculate profit from original entry
        if is_long:
            profit_from_entry = ((price - self.round_entry_price) / self.round_entry_price) * 100
        else:
            profit_from_entry = ((self.round_entry_price - price) / self.round_entry_price) * 100
        
        # Update max profit
        if profit_from_entry > self.max_profit_pct:
            self.max_profit_pct = profit_from_entry
        
        # Check for new pyramid level
        if len(self.pyramid_positions) < self.max_pyramids:
            if is_long:
                move_from_ref = ((price - self.pyramid_reference) / self.pyramid_reference) * 100
            else:
                move_from_ref = ((self.pyramid_reference - price) / self.pyramid_reference) * 100
            
            pyramid_threshold = self.next_pyramid_level * self.pyramid_step_pct
            if move_from_ref >= pyramid_threshold:
                self._add_pyramid(price, move_from_ref)
        
        # Check trailing stop
        trigger_profit = self.max_profit_pct - self.trailing_pct
        if profit_from_entry <= trigger_profit:
            self._close_all(price, profit_from_entry)
        else:
            self._print_pyramid_status(price, profit_from_entry)
    
    def _add_pyramid(self, price: float, move_from_ref: float):
        """Add a new pyramid position."""
        # Open on exchange
        if self.direction == 'LONG':
            self.exchange.open_hedged_positions(self.symbol, self.position_size)
            # Close the short side immediately (we only want long)
            self.exchange.close_short_hedge(self.symbol, self.position_size)
        else:
            self.exchange.open_hedged_positions(self.symbol, self.position_size)
            # Close the long side immediately (we only want short)
            self.exchange.close_long_hedge(self.symbol, self.position_size)
        
        new_pos = Position(
            side=self.direction,
            entry_price=price,
            entry_time=datetime.now(),
            size=self.position_size
        )
        self.pyramid_positions.append(new_pos)
        self.next_pyramid_level += 1
        
        print(f"\n📈 PYRAMID #{len(self.pyramid_positions)} @ ${price:.4f} (+{move_from_ref:.1f}% from ref)")
    
    def _close_all(self, price: float, profit_from_entry: float):
        """Close all pyramid positions."""
        total_size = sum(p.size for p in self.pyramid_positions)
        
        # Close all on exchange
        if self.direction == 'LONG':
            self.exchange.close_long_hedge(self.symbol, total_size)
        else:
            self.exchange.close_short_hedge(self.symbol, total_size)
        
        # Calculate P&L
        round_pnl = 0.0
        is_long = (self.direction == 'LONG')
        
        for pos in self.pyramid_positions:
            if is_long:
                pnl = ((price - pos.entry_price) / pos.entry_price) * 100
            else:
                pnl = ((pos.entry_price - price) / pos.entry_price) * 100
            round_pnl += pnl
        
        # Subtract losing hedge
        round_pnl -= self.threshold_pct
        
        self.total_pnl += round_pnl
        
        emoji = "✅" if round_pnl > 0 else "❌"
        print(f"\n{emoji} ALL CLOSED @ ${price:.4f}")
        print(f"   Pyramids: {len(self.pyramid_positions)}")
        print(f"   Max Profit: {self.max_profit_pct:+.2f}%")
        print(f"   Exit Profit: {profit_from_entry:+.2f}%")
        print(f"   Round P&L: {round_pnl:+.2f}%")
        print(f"   Total P&L: {self.total_pnl:+.2f}%")
        
        # Reset
        self.pyramid_positions = []
        self.direction = ''
    
    def _print_status(self, price: float, long_profit: float, short_profit: float):
        """Print hedge status."""
        print(f"\r  ${price:.4f} | LONG: {long_profit:+.2f}% | SHORT: {short_profit:+.2f}% | Waiting for direction...    ", 
              end="", flush=True)
    
    def _print_pyramid_status(self, price: float, profit_from_entry: float):
        """Print pyramid status."""
        trigger = self.max_profit_pct - self.trailing_pct
        print(f"\r  ${price:.4f} | {self.direction} x{len(self.pyramid_positions)} | "
              f"Profit: {profit_from_entry:+.2f}% | Max: {self.max_profit_pct:+.2f}% | "
              f"Trigger: {trigger:+.2f}%    ", 
              end="", flush=True)
    
    def _cleanup(self):
        """Close all positions on shutdown."""
        print("\n\n⚠️  SHUTTING DOWN - Closing all positions...")
        
        try:
            # Close initial hedge if still open
            if self.long_pos:
                print(f"  Closing LONG hedge...")
                self.exchange.close_long_hedge(self.symbol, self.position_size)
                print(f"  ✓ LONG hedge closed")
            
            if self.short_pos:
                print(f"  Closing SHORT hedge...")
                self.exchange.close_short_hedge(self.symbol, self.position_size)
                print(f"  ✓ SHORT hedge closed")
            
            # Close all pyramid positions
            if self.pyramid_positions:
                total_size = sum(p.size for p in self.pyramid_positions)
                print(f"  Closing {len(self.pyramid_positions)} pyramid positions (total: {total_size})...")
                if self.direction == 'LONG':
                    self.exchange.close_long_hedge(self.symbol, total_size)
                else:
                    self.exchange.close_short_hedge(self.symbol, total_size)
                print(f"  ✓ All pyramid positions closed")
            
            # Verify no positions remain
            remaining = self.exchange.get_positions(self.symbol)
            if remaining:
                print(f"\n  ⚠️  Warning: {len(remaining)} positions still open!")
                for pos in remaining:
                    amt = float(pos['positionAmt'])
                    if amt != 0:
                        side = 'LONG' if amt > 0 else 'SHORT'
                        print(f"     {side}: {abs(amt)} @ ${float(pos['entryPrice']):.4f}")
            else:
                print(f"\n  ✓ All positions successfully closed!")
                
        except Exception as e:
            print(f"\n  ❌ Error during cleanup: {e}")
            print(f"  Please manually check and close any remaining positions!")
    
    def _print_summary(self):
        """Print final summary."""
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Rounds Completed: {self.round_number}")
        print(f"Total P&L: {self.total_pnl:+.2f}%")
        print(f"{'='*60}")
