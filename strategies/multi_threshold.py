"""
Multi-Threshold Hedge Trailing Strategy

Runs multiple threshold levels simultaneously on the same symbol.
Each threshold has its own pair of LONG/SHORT positions.
Uses entry-price-relative profit tracking.
Uses tighter trailing stop when only one position remains open.
"""

import time
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from core.exchange import BinanceExchange
from core.trailing_stop import calculate_profit_pct
from utils.logger import log_trade, log_round_summary, print_session_stats
from config.settings import SYMBOL, POSITION_SIZE, PRICE_CHECK_INTERVAL


@dataclass
class ThresholdPair:
    """Tracks a single hedged position pair for one threshold."""
    threshold: float
    quantity: float
    single_pos_multiplier: float = 0.5  # Tighter stop when one position closed
    
    # Position tracking
    entry_price: float = 0.0
    entry_time: datetime = None
    
    # Long position (entry-price relative)
    long_open: bool = False
    long_max_profit_pct: float = 0.0
    long_pnl: float = 0.0
    long_exit_price: float = 0.0
    
    # Short position (entry-price relative)
    short_open: bool = False
    short_max_profit_pct: float = 0.0
    short_pnl: float = 0.0
    short_exit_price: float = 0.0
    
    # Stats
    round_number: int = 0
    
    def open_positions(self, price: float):
        """Open both long and short positions."""
        self.round_number += 1
        self.entry_price = price
        self.entry_time = datetime.now()
        
        self.long_open = True
        self.long_max_profit_pct = 0.0
        self.long_pnl = 0.0
        
        self.short_open = True
        self.short_max_profit_pct = 0.0
        self.short_pnl = 0.0
    
    def get_long_profit(self, price: float) -> float:
        """Get current long profit % relative to entry."""
        return ((price - self.entry_price) / self.entry_price) * 100
    
    def get_short_profit(self, price: float) -> float:
        """Get current short profit % relative to entry."""
        return ((self.entry_price - price) / self.entry_price) * 100
    
    def update_max_profits(self, price: float) -> tuple:
        """Update max profits. Returns (long_updated, short_updated) booleans."""
        long_updated = False
        short_updated = False
        
        if self.long_open:
            current_profit = self.get_long_profit(price)
            if current_profit > self.long_max_profit_pct:
                self.long_max_profit_pct = current_profit
                long_updated = True
        
        if self.short_open:
            current_profit = self.get_short_profit(price)
            if current_profit > self.short_max_profit_pct:
                self.short_max_profit_pct = current_profit
                short_updated = True
        
        return long_updated, short_updated
    
    def get_effective_threshold(self, for_long: bool) -> float:
        """Get effective threshold based on which positions are open."""
        # If only one position is open, use tighter threshold
        if for_long and self.long_open and not self.short_open:
            return self.threshold * self.single_pos_multiplier
        if not for_long and self.short_open and not self.long_open:
            return self.threshold * self.single_pos_multiplier
        return self.threshold
    
    def should_close_long(self, price: float) -> bool:
        """Check if long should close."""
        if not self.long_open:
            return False
        current_profit = self.get_long_profit(price)
        effective_threshold = self.get_effective_threshold(for_long=True)
        trigger_profit = self.long_max_profit_pct - effective_threshold
        return current_profit <= trigger_profit
    
    def should_close_short(self, price: float) -> bool:
        """Check if short should close."""
        if not self.short_open:
            return False
        current_profit = self.get_short_profit(price)
        effective_threshold = self.get_effective_threshold(for_long=False)
        trigger_profit = self.short_max_profit_pct - effective_threshold
        return current_profit <= trigger_profit
    
    def has_open_positions(self) -> bool:
        return self.long_open or self.short_open
    
    def close_long(self, exit_price: float):
        """Close long position."""
        self.long_open = False
        self.long_exit_price = exit_price
        self.long_pnl = self.get_long_profit(exit_price)
    
    def close_short(self, exit_price: float):
        """Close short position."""
        self.short_open = False
        self.short_exit_price = exit_price
        self.short_pnl = self.get_short_profit(exit_price)
    
    def get_total_pnl(self) -> float:
        return self.long_pnl + self.short_pnl


class MultiThresholdStrategy:
    """
    Runs multiple threshold levels concurrently.
    Each threshold has its own LONG+SHORT pair.
    Uses tighter trailing stop when only one position remains.
    """
    
    def __init__(
        self,
        exchange: BinanceExchange,
        thresholds: List[float],
        position_size: float = POSITION_SIZE,
        symbol: str = SYMBOL,
        max_rounds: Optional[int] = None,
        single_pos_multiplier: float = 0.5
    ):
        self.exchange = exchange
        self.thresholds = sorted(thresholds)
        self.position_size = position_size
        self.symbol = symbol
        self.max_rounds = max_rounds
        self.single_pos_multiplier = single_pos_multiplier
        
        # Create a pair tracker for each threshold
        self.pairs: Dict[float, ThresholdPair] = {
            t: ThresholdPair(threshold=t, quantity=position_size, single_pos_multiplier=single_pos_multiplier)
            for t in thresholds
        }
        
        self.running = False
        self.total_rounds = 0
        self.cumulative_pnl: Dict[float, float] = {t: 0.0 for t in thresholds}
    
    def run(self):
        """Main strategy loop."""
        self.running = True
        
        self._print_header()
        
        try:
            # Initially open all positions
            self._open_all_positions()
            
            while self.running:
                # Check round limit
                min_rounds = min(p.round_number for p in self.pairs.values())
                if self.max_rounds and min_rounds >= self.max_rounds:
                    print(f"\n✓ Completed {self.max_rounds} rounds. Stopping.")
                    break
                
                # Get current price
                current_price = self.exchange.get_price(self.symbol)
                
                # Update all pairs and check for closes
                self._process_price_update(current_price)
                
                # Print status
                self._print_status(current_price)
                
                time.sleep(PRICE_CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            self._cleanup()
        
        self._print_final_stats()
    
    def _print_header(self):
        print(f"\n{'#'*70}")
        print(f"MULTI-THRESHOLD HEDGE STRATEGY")
        print(f"{'#'*70}")
        print(f"Symbol:     {self.symbol}")
        print(f"Size:       {self.position_size} per threshold")
        print(f"Thresholds: {', '.join(f'{t}%' for t in self.thresholds)}")
        print(f"Single Pos: {', '.join(f'{t * self.single_pos_multiplier}%' for t in self.thresholds)}")
        print(f"Max Rounds: {self.max_rounds or 'Unlimited'}")
        print(f"Total Positions: {len(self.thresholds) * 2} ({len(self.thresholds)} LONG + {len(self.thresholds)} SHORT)")
        print(f"{'#'*70}\n")
    
    def _open_all_positions(self):
        """Open positions for all thresholds."""
        entry_price = self.exchange.get_price(self.symbol)
        
        print(f"\n{'='*70}")
        print(f"OPENING ALL POSITIONS @ ${entry_price:.4f}")
        print(f"{'='*70}")
        
        for threshold, pair in self.pairs.items():
            if not pair.has_open_positions():
                self.exchange.open_hedged_positions(self.symbol, self.position_size)
                pair.open_positions(entry_price)
                print(f"  [{threshold}%] Opened LONG + SHORT (Round {pair.round_number})")
    
    def _process_price_update(self, current_price: float):
        """Process price update for all threshold pairs."""
        
        for threshold, pair in self.pairs.items():
            if not pair.has_open_positions():
                self._log_pair_round(pair)
                self._open_pair(pair, current_price)
                continue
            
            # Update max profits
            pair.update_max_profits(current_price)
            
            # Check long trailing stop
            if pair.long_open and pair.should_close_long(current_price):
                self._close_pair_long(pair, current_price)
            
            # Check short trailing stop
            if pair.short_open and pair.should_close_short(current_price):
                self._close_pair_short(pair, current_price)
    
    def _open_pair(self, pair: ThresholdPair, price: float):
        """Open a single pair's positions."""
        self.exchange.open_hedged_positions(self.symbol, self.position_size)
        pair.open_positions(price)
        print(f"\n  [{pair.threshold}%] NEW ROUND {pair.round_number} @ ${price:.4f}")
    
    def _close_pair_long(self, pair: ThresholdPair, price: float):
        """Close a pair's long position."""
        self.exchange.close_long_hedge(self.symbol, self.position_size)
        effective_threshold = pair.get_effective_threshold(for_long=True)
        pair.close_long(price)
        
        emoji = "✅" if pair.long_pnl >= 0 else "❌"
        print(f"\n{emoji} [{pair.threshold}%] LONG CLOSED: {pair.long_pnl:+.2f}% (max: {pair.long_max_profit_pct:+.2f}%, thresh: {effective_threshold:.2f}%)")
        
        log_trade(
            round_number=pair.round_number,
            side="LONG",
            entry_price=pair.entry_price,
            exit_price=price,
            quantity=pair.quantity,
            peak_price=pair.entry_price * (1 + pair.long_max_profit_pct / 100),
            trough_price=pair.entry_price,
            pnl_percent=pair.long_pnl,
            threshold_percent=pair.threshold,
            entry_time=pair.entry_time,
            exit_time=datetime.now()
        )
    
    def _close_pair_short(self, pair: ThresholdPair, price: float):
        """Close a pair's short position."""
        self.exchange.close_short_hedge(self.symbol, self.position_size)
        effective_threshold = pair.get_effective_threshold(for_long=False)
        pair.close_short(price)
        
        emoji = "✅" if pair.short_pnl >= 0 else "❌"
        print(f"\n{emoji} [{pair.threshold}%] SHORT CLOSED: {pair.short_pnl:+.2f}% (max: {pair.short_max_profit_pct:+.2f}%, thresh: {effective_threshold:.2f}%)")
        
        log_trade(
            round_number=pair.round_number,
            side="SHORT",
            entry_price=pair.entry_price,
            exit_price=price,
            quantity=pair.quantity,
            peak_price=pair.entry_price,
            trough_price=pair.entry_price * (1 - pair.short_max_profit_pct / 100),
            pnl_percent=pair.short_pnl,
            threshold_percent=pair.threshold,
            entry_time=pair.entry_time,
            exit_time=datetime.now()
        )
    
    def _log_pair_round(self, pair: ThresholdPair):
        """Log completed round for a pair."""
        total_pnl = pair.get_total_pnl()
        self.cumulative_pnl[pair.threshold] += total_pnl
        
        print(f"\n  [{pair.threshold}%] Round {pair.round_number} Complete: {total_pnl:+.2f}% (Cumulative: {self.cumulative_pnl[pair.threshold]:+.2f}%)")
        
        log_round_summary(
            round_number=pair.round_number,
            long_pnl=pair.long_pnl,
            short_pnl=pair.short_pnl,
            total_pnl=total_pnl,
            threshold_percent=pair.threshold
        )
    
    def _print_status(self, current_price: float):
        """Print current status of all pairs."""
        status_parts = [f"${current_price:.2f}"]
        
        for threshold, pair in self.pairs.items():
            if pair.long_open:
                pnl = pair.get_long_profit(current_price)
                status_parts.append(f"[{threshold}%] L:{pnl:+.1f}%")
            else:
                status_parts.append(f"[{threshold}%] L:--")
            
            if pair.short_open:
                pnl = pair.get_short_profit(current_price)
                status_parts.append(f"S:{pnl:+.1f}%")
            else:
                status_parts.append(f"S:--")
        
        print(f"\r  {' | '.join(status_parts)}    ", end="", flush=True)
    
    def _cleanup(self):
        """Close all open positions on shutdown."""
        print("\n\nClosing all open positions...")
        current_price = self.exchange.get_price(self.symbol)
        
        for threshold, pair in self.pairs.items():
            if pair.long_open:
                self._close_pair_long(pair, current_price)
            if pair.short_open:
                self._close_pair_short(pair, current_price)
            if not pair.has_open_positions():
                self._log_pair_round(pair)
    
    def _print_final_stats(self):
        """Print final statistics for all thresholds."""
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS BY THRESHOLD")
        print(f"{'='*70}")
        
        for threshold in self.thresholds:
            pair = self.pairs[threshold]
            cumulative = self.cumulative_pnl[threshold]
            emoji = "✅" if cumulative >= 0 else "❌"
            print(f"  {emoji} [{threshold}%] Rounds: {pair.round_number} | Total P&L: {cumulative:+.2f}%")
        
        total_all = sum(self.cumulative_pnl.values())
        print(f"\n  COMBINED TOTAL: {total_all:+.2f}%")
        print(f"{'='*70}\n")
        
        print_session_stats()
