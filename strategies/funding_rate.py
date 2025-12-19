#!/usr/bin/env python3
"""
Funding Rate Farming Strategy

Strategy Logic:
1. Monitor funding rate on Binance Futures
2. When funding rate > entry_threshold:
   - Positive funding: SHORT perp (longs pay shorts)
   - Negative funding: LONG perp (shorts pay longs)
3. Hold position through funding payment (every 8 hours)
4. Exit when funding drops below exit_threshold

Note: This is a simplified version using futures only.
For true delta-neutral, you would also hold spot in opposite direction.
"""

import time
import requests
from datetime import datetime
from typing import Optional

from core.exchange import BinanceExchange
from core.state_manager import StateManager
from config.settings import PRICE_CHECK_INTERVAL


class FundingRateStrategy:
    """
    Funding rate farming strategy.
    Collects funding payments by being on the receiving side.
    """
    
    # Binance Futures API endpoints
    MAINNET_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"
    
    def __init__(
        self,
        exchange: BinanceExchange,
        symbol: str,
        entry_threshold: float = 0.1,  # Enter when funding > 0.1%
        exit_threshold: float = 0.02,  # Exit when funding < 0.02%
        position_size: float = 1.0,
        use_testnet: bool = True
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.base_url = self.TESTNET_URL if use_testnet else self.MAINNET_URL
        
        # State manager for isolated tracking
        self.state = StateManager(mode="funding", symbol=symbol)
        
        # State
        self.running = False
        self.position_side: Optional[str] = None  # 'LONG' or 'SHORT'
        self.entry_price: float = 0.0
        self.entry_time: datetime = None
        self.funding_collected: float = 0.0
        self.funding_payments: int = 0
        self.current_order_id: Optional[str] = None
    
    def run(self):
        """Main strategy loop."""
        self.running = True
        
        self._print_header()
        
        try:
            while self.running:
                # Get current funding rate
                funding_info = self._get_funding_rate()
                
                if funding_info is None:
                    print("Error getting funding rate. Retrying...")
                    time.sleep(10)
                    continue
                
                funding_rate = funding_info['rate']
                next_funding = funding_info['next_time']
                
                # Manage position
                if self.position_side is None:
                    self._check_entry(funding_rate)
                else:
                    self._check_exit(funding_rate, next_funding)
                
                # Status update
                self._print_status(funding_rate, next_funding)
                
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            self._close_position()
        
        self._print_summary()
    
    def _print_header(self):
        print(f"\n{'#'*60}")
        print(f"FUNDING RATE FARMING STRATEGY")
        print(f"{'#'*60}")
        print(f"Symbol:          {self.symbol}")
        print(f"Entry Threshold: >{self.entry_threshold}% (absolute value)")
        print(f"Exit Threshold:  <{self.exit_threshold}%")
        print(f"Position Size:   {self.position_size}")
        print(f"{'#'*60}")
        print(f"\nMonitoring funding rate...")
        print(f"Funding payments occur every 8 hours (00:00, 08:00, 16:00 UTC)\n")
    
    def _get_funding_rate(self) -> Optional[dict]:
        """Get current funding rate from Binance."""
        try:
            url = f"{self.base_url}/fapi/v1/premiumIndex"
            params = {'symbol': self.symbol}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            rate = float(data.get('lastFundingRate', 0)) * 100  # Convert to percentage
            next_time = int(data.get('nextFundingTime', 0))
            mark_price = float(data.get('markPrice', 0))
            
            return {
                'rate': rate,
                'next_time': next_time,
                'mark_price': mark_price
            }
        except Exception as e:
            print(f"Error fetching funding rate: {e}")
            return None
    
    def _check_entry(self, funding_rate: float):
        """Check if we should enter a position."""
        abs_rate = abs(funding_rate)
        
        if abs_rate >= self.entry_threshold:
            # Determine direction
            if funding_rate > 0:
                # Positive funding: longs pay shorts
                # We want to be SHORT to receive payment
                self._open_position('SHORT', funding_rate)
            else:
                # Negative funding: shorts pay longs
                # We want to be LONG to receive payment
                self._open_position('LONG', funding_rate)
    
    def _open_position(self, side: str, funding_rate: float):
        """Open a position."""
        price = self.exchange.get_price(self.symbol)
        
        print(f"\n{'='*60}")
        print(f"📈 ENTERING POSITION")
        print(f"{'='*60}")
        print(f"Instance:     {self.state.instance_id}")
        print(f"Side:         {side}")
        print(f"Price:        ${price:.4f}")
        print(f"Size:         {self.position_size}")
        print(f"Funding Rate: {funding_rate:+.4f}%")
        print(f"Expected:     {abs(funding_rate) * 3:.2f}% daily")
        print(f"{'='*60}")
        
        # Generate unique client order ID
        client_order_id = self.state.generate_client_order_id(side)
        
        # Open on exchange with tracking
        order = self.exchange.open_single_position(
            symbol=self.symbol,
            side=side,
            quantity=self.position_size,
            client_order_id=client_order_id
        )
        
        # Track in state manager
        self.state.add_order(
            client_order_id=client_order_id,
            exchange_order_id=order.get('orderId', 0),
            side=side,
            quantity=self.position_size,
            entry_price=price
        )
        
        self.position_side = side
        self.entry_price = price
        self.entry_time = datetime.now()
        self.current_order_id = client_order_id
        
        print(f"✓ Opened {side}: {self.position_size} {self.symbol}")
        print(f"  Order ID: {client_order_id}")
    
    def _check_exit(self, funding_rate: float, next_funding: int):
        """Check if we should exit the position."""
        abs_rate = abs(funding_rate)
        
        # Check if funding flipped against us
        if self.position_side == 'SHORT' and funding_rate < 0:
            print(f"\n⚠ Funding flipped negative! Exiting position.")
            self._close_position()
            return
        
        if self.position_side == 'LONG' and funding_rate > 0:
            print(f"\n⚠ Funding flipped positive! Exiting position.")
            self._close_position()
            return
        
        # Check if funding dropped below threshold
        if abs_rate < self.exit_threshold:
            print(f"\n📉 Funding below exit threshold ({abs_rate:.4f}% < {self.exit_threshold}%)")
            self._close_position()
            return
        
        # Track funding payments
        self._check_funding_payment(next_funding)
    
    def _check_funding_payment(self, next_funding: int):
        """Check if a funding payment occurred."""
        now = int(datetime.now().timestamp() * 1000)
        
        # Simple check: if we're within 1 minute after funding time
        time_since_funding = now - (next_funding - 8 * 60 * 60 * 1000)
        
        if 0 < time_since_funding < 60000:  # Within 1 minute after payment
            # Estimate funding collected (simplified)
            funding_info = self._get_funding_rate()
            if funding_info:
                estimated_payment = abs(funding_info['rate'])
                self.funding_collected += estimated_payment
                self.funding_payments += 1
                print(f"\n💰 FUNDING PAYMENT #{self.funding_payments}: +{estimated_payment:.4f}%")
                print(f"   Total Collected: {self.funding_collected:.4f}%")
    
    def _close_position(self):
        """Close the current position."""
        if self.position_side is None:
            return
        
        price = self.exchange.get_price(self.symbol)
        
        # Calculate P&L
        if self.position_side == 'LONG':
            price_pnl = ((price - self.entry_price) / self.entry_price) * 100
            self.exchange.close_position_by_quantity(self.symbol, 'LONG', self.position_size)
        else:
            price_pnl = ((self.entry_price - price) / self.entry_price) * 100
            self.exchange.close_position_by_quantity(self.symbol, 'SHORT', self.position_size)
        
        total_pnl = price_pnl + self.funding_collected
        hold_time = datetime.now() - self.entry_time
        
        print(f"\n{'='*60}")
        print(f"📊 POSITION CLOSED")
        print(f"{'='*60}")
        print(f"Instance:        {self.state.instance_id}")
        print(f"Entry:           ${self.entry_price:.4f}")
        print(f"Exit:            ${price:.4f}")
        print(f"Hold Time:       {hold_time}")
        print(f"Price P&L:       {price_pnl:+.2f}%")
        print(f"Funding Collected: {self.funding_collected:+.4f}%")
        print(f"Funding Payments:  {self.funding_payments}")
        print(f"------------------------")
        print(f"TOTAL P&L:       {total_pnl:+.2f}%")
        print(f"{'='*60}")
        
        # Remove from state manager
        if self.current_order_id:
            self.state.remove_order(self.current_order_id)
        
        # Reset state
        self.position_side = None
        self.entry_price = 0.0
        self.entry_time = None
        self.current_order_id = None
    
    def _print_status(self, funding_rate: float, next_funding: int):
        """Print current status."""
        # Calculate time until next funding
        now = int(datetime.now().timestamp() * 1000)
        time_until = (next_funding - now) / 1000 / 60  # minutes
        
        if self.position_side:
            price = self.exchange.get_price(self.symbol)
            if self.position_side == 'LONG':
                price_pnl = ((price - self.entry_price) / self.entry_price) * 100
            else:
                price_pnl = ((self.entry_price - price) / self.entry_price) * 100
            
            print(f"\r  Funding: {funding_rate:+.4f}% | {self.position_side} @ ${self.entry_price:.2f} | "
                  f"P&L: {price_pnl:+.2f}% + {self.funding_collected:.2f}% funding | "
                  f"Next payment: {time_until:.0f}m    ", 
                  end="", flush=True)
        else:
            print(f"\r  Funding: {funding_rate:+.4f}% | No position | "
                  f"Entry threshold: >{self.entry_threshold}% | "
                  f"Next payment: {time_until:.0f}m    ", 
                  end="", flush=True)
    
    def _print_summary(self):
        """Print final summary."""
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Instance ID:             {self.state.instance_id}")
        print(f"Total Funding Collected: {self.funding_collected:.4f}%")
        print(f"Total Payments:          {self.funding_payments}")
        print(f"{'='*60}")
        
        # Cleanup state file if no positions remain
        if not self.state.get_all_orders():
            self.state.cleanup_state_file()
