"""
Hedge Trailing Strategy

Opens simultaneous long and short positions, then watches price movements.
Each position closes when profit drops X% below max profit (entry-price relative).
Uses tighter trailing stop when only one position remains open.
"""

import time
from datetime import datetime
from typing import Optional

from core.exchange import BinanceExchange
from core.position_manager import PositionManager, PositionSide
from core.trailing_stop import calculate_profit_pct
from utils.logger import log_trade, log_round_summary, print_session_stats
from config.settings import SYMBOL, POSITION_SIZE, PRICE_CHECK_INTERVAL


class HedgeTrailingStrategy:
    """
    Main strategy: open hedged positions, trail max profit, close on retracement.
    Uses tighter trailing stop when only one position remains (to lock in profits).
    """
    
    def __init__(
        self,
        exchange: BinanceExchange,
        threshold_percent: float = 1.0,
        position_size: float = POSITION_SIZE,
        symbol: str = SYMBOL,
        max_rounds: Optional[int] = None,
        single_pos_multiplier: float = 0.5  # Tighter stop when one position closed
    ):
        """
        Initialize the strategy.
        
        Args:
            exchange: Connected BinanceExchange instance
            threshold_percent: Trailing stop threshold (e.g., 1.0 for 1%)
            position_size: Amount to trade per position
            symbol: Trading pair
            max_rounds: Maximum rounds to run (None for infinite)
            single_pos_multiplier: Threshold multiplier when only one position is open (default 0.5)
        """
        self.exchange = exchange
        self.threshold_percent = threshold_percent
        self.position_size = position_size
        self.symbol = symbol
        self.max_rounds = max_rounds
        self.single_pos_multiplier = single_pos_multiplier
        
        self.position_manager = PositionManager()
        self.running = False
    
    def run(self):
        """Main strategy loop."""
        self.running = True
        
        print(f"\n{'#'*60}")
        print(f"HEDGE TRAILING STRATEGY")
        print(f"{'#'*60}")
        print(f"Symbol:    {self.symbol}")
        print(f"Size:      {self.position_size}")
        print(f"Threshold: {self.threshold_percent}% (single: {self.threshold_percent * self.single_pos_multiplier}%)")
        print(f"Max Rounds: {self.max_rounds or 'Unlimited'}")
        print(f"{'#'*60}\n")
        
        try:
            while self.running:
                # Check round limit
                if self.max_rounds and self.position_manager.round_number >= self.max_rounds:
                    print(f"\n✓ Completed {self.max_rounds} rounds. Stopping.")
                    break
                
                # Start new round
                self._open_hedged_positions()
                
                # Monitor and trail until both closed
                self._monitor_positions()
                
                # Log round summary
                self._log_round()
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            self._cleanup()
        
        # Final stats
        print_session_stats()
    
    def stop(self):
        """Signal strategy to stop after current round."""
        self.running = False
    
    def _open_hedged_positions(self):
        """Open both long and short positions."""
        entry_price = self.exchange.get_price(self.symbol)
        
        # Open positions on exchange
        self.exchange.open_hedged_positions(self.symbol, self.position_size)
        
        # Track positions locally
        self.position_manager.open_hedged_positions(entry_price, self.position_size)
    
    def _get_effective_threshold(self) -> float:
        """Get current effective threshold based on open positions."""
        long_open = self.position_manager.is_long_open()
        short_open = self.position_manager.is_short_open()
        
        # If only one position is open, use tighter threshold
        if (long_open and not short_open) or (short_open and not long_open):
            return self.threshold_percent * self.single_pos_multiplier
        return self.threshold_percent
    
    def _monitor_positions(self):
        """Monitor price and close positions based on trailing stops."""
        while self.position_manager.has_open_positions():
            if not self.running:
                break
            
            # Get current price
            current_price = self.exchange.get_price(self.symbol)
            
            # Update max profits
            self.position_manager.update_max_profits(current_price)
            
            # Get effective threshold (tighter when only one position open)
            effective_threshold = self._get_effective_threshold()
            
            # Check if long should close (profit dropped threshold% below max)
            if self.position_manager.is_long_open():
                pos = self.position_manager.long_position
                current_profit = pos.get_current_profit(current_price)
                trigger_profit = pos.max_profit_pct - effective_threshold
                if current_profit <= trigger_profit:
                    self._close_long(current_price)
            
            # Check if short should close (profit dropped threshold% below max)
            if self.position_manager.is_short_open():
                pos = self.position_manager.short_position
                current_profit = pos.get_current_profit(current_price)
                trigger_profit = pos.max_profit_pct - effective_threshold
                if current_profit <= trigger_profit:
                    self._close_short(current_price)
            
            # Status update
            self._print_status(current_price, effective_threshold)
            
            # Wait before next check
            time.sleep(PRICE_CHECK_INTERVAL)
    
    def _close_long(self, exit_price: float):
        """Close the long position."""
        pos = self.position_manager.long_position
        
        # Close on exchange
        self.exchange.close_long_hedge(self.symbol, self.position_size)
        
        # Update local tracking
        self.position_manager.close_long(exit_price)
        
        # Log trade
        log_trade(
            round_number=self.position_manager.round_number,
            side="LONG",
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            peak_price=pos.entry_price * (1 + pos.max_profit_pct / 100),
            trough_price=pos.entry_price,
            pnl_percent=pos.pnl_percent,
            threshold_percent=self.threshold_percent,
            entry_time=pos.entry_time,
            exit_time=datetime.now()
        )
    
    def _close_short(self, exit_price: float):
        """Close the short position."""
        pos = self.position_manager.short_position
        
        # Close on exchange
        self.exchange.close_short_hedge(self.symbol, self.position_size)
        
        # Update local tracking
        self.position_manager.close_short(exit_price)
        
        # Log trade
        log_trade(
            round_number=self.position_manager.round_number,
            side="SHORT",
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            peak_price=pos.entry_price,
            trough_price=pos.entry_price * (1 - pos.max_profit_pct / 100),
            pnl_percent=pos.pnl_percent,
            threshold_percent=self.threshold_percent,
            entry_time=pos.entry_time,
            exit_time=datetime.now()
        )
    
    def _log_round(self):
        """Log and display round summary."""
        summary = self.position_manager.get_round_summary()
        self.position_manager.print_round_summary()
        
        log_round_summary(
            round_number=summary['round'],
            long_pnl=summary['long_pnl'] or 0,
            short_pnl=summary['short_pnl'] or 0,
            total_pnl=summary['total_pnl'] or 0,
            threshold_percent=self.threshold_percent
        )
    
    def _print_status(self, current_price: float, effective_threshold: float):
        """Print current status (single line, updated in place)."""
        long_status = ""
        short_status = ""
        
        if self.position_manager.is_long_open():
            pos = self.position_manager.long_position
            pnl = pos.get_current_profit(current_price)
            long_status = f"LONG: {pnl:+.2f}% (max: {pos.max_profit_pct:+.2f}%)"
        else:
            long_status = "LONG: CLOSED"
        
        if self.position_manager.is_short_open():
            pos = self.position_manager.short_position
            pnl = pos.get_current_profit(current_price)
            short_status = f"SHORT: {pnl:+.2f}% (max: {pos.max_profit_pct:+.2f}%)"
        else:
            short_status = "SHORT: CLOSED"
        
        print(f"\r  ${current_price:.4f} | {long_status} | {short_status} [thresh: {effective_threshold:.2f}%]    ", end="", flush=True)
    
    def _cleanup(self):
        """Clean up on shutdown - close any open positions."""
        print("\n\nCleaning up open positions...")
        
        current_price = self.exchange.get_price(self.symbol)
        
        if self.position_manager.is_long_open():
            self._close_long(current_price)
        
        if self.position_manager.is_short_open():
            self._close_short(current_price)
        
        self._log_round()
