"""
Binance Futures Testnet API wrapper.
Handles connection, order placement, and position management.
"""

from binance.client import Client
from binance.enums import *
from typing import Optional, Dict, Any
import time

from config.settings import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    TESTNET,
    SYMBOL
)


class BinanceExchange:
    """Wrapper for Binance Futures Testnet API."""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.symbol = SYMBOL
        
    def connect(self) -> bool:
        """
        Initialize connection to Binance Futures Testnet.
        
        Returns:
            True if connection successful
        """
        try:
            self.client = Client(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET,
                testnet=TESTNET
            )
            # Verify connection by fetching account info
            self.client.futures_account()
            print("✓ Connected to Binance Futures Testnet")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return False
    
    def get_price(self, symbol: Optional[str] = None) -> float:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading pair (defaults to configured SYMBOL)
            
        Returns:
            Current price as float
        """
        symbol = symbol or self.symbol
        ticker = self.client.futures_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    
    def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance information.
        
        Returns:
            Dictionary with balance info
        """
        account = self.client.futures_account()
        usdt_balance = next(
            (b for b in account['assets'] if b['asset'] == 'USDT'),
            {'walletBalance': '0', 'availableBalance': '0'}
        )
        return {
            'wallet_balance': float(usdt_balance['walletBalance']),
            'available_balance': float(usdt_balance['availableBalance']),
            'total_unrealized_pnl': float(account['totalUnrealizedProfit'])
        }
    
    def open_long(self, symbol: Optional[str] = None, quantity: float = 0.1) -> Dict[str, Any]:
        """
        Open a long position (BUY).
        
        Args:
            symbol: Trading pair
            quantity: Amount to buy
            
        Returns:
            Order response from Binance
        """
        symbol = symbol or self.symbol
        order = self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"✓ Opened LONG: {quantity} {symbol} @ market")
        return order
    
    def open_short(self, symbol: Optional[str] = None, quantity: float = 0.1) -> Dict[str, Any]:
        """
        Open a short position (SELL).
        
        Args:
            symbol: Trading pair
            quantity: Amount to sell
            
        Returns:
            Order response from Binance
        """
        symbol = symbol or self.symbol
        order = self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"✓ Opened SHORT: {quantity} {symbol} @ market")
        return order
    
    def close_long(self, symbol: Optional[str] = None, quantity: float = 0.1) -> Dict[str, Any]:
        """
        Close a long position by selling.
        
        Args:
            symbol: Trading pair
            quantity: Amount to sell
            
        Returns:
            Order response from Binance
        """
        symbol = symbol or self.symbol
        order = self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity,
            reduceOnly=True
        )
        print(f"✓ Closed LONG: {quantity} {symbol} @ market")
        return order
    
    def close_short(self, symbol: Optional[str] = None, quantity: float = 0.1) -> Dict[str, Any]:
        """
        Close a short position by buying.
        
        Args:
            symbol: Trading pair
            quantity: Amount to buy
            
        Returns:
            Order response from Binance
        """
        symbol = symbol or self.symbol
        order = self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=quantity,
            reduceOnly=True
        )
        print(f"✓ Closed SHORT: {quantity} {symbol} @ market")
        return order
    
    def get_positions(self, symbol: Optional[str] = None) -> list:
        """
        Get current open positions.
        
        Args:
            symbol: Trading pair (optional, returns all if not specified)
            
        Returns:
            List of position dictionaries
        """
        positions = self.client.futures_position_information(symbol=symbol or self.symbol)
        return [p for p in positions if float(p['positionAmt']) != 0]
    
    def set_hedge_mode(self) -> bool:
        """
        Enable hedge mode to allow simultaneous long and short positions.
        
        Returns:
            True if successful
        """
        try:
            self.client.futures_change_position_mode(dualSidePosition=True)
            print("✓ Hedge mode enabled")
            return True
        except Exception as e:
            if "No need to change position side" in str(e):
                print("✓ Hedge mode already enabled")
                return True
            print(f"⚠ Could not set hedge mode: {e}")
            return False
    
    def open_hedged_positions(self, symbol: Optional[str] = None, quantity: float = 1, max_retries: int = 10) -> tuple:
        """
        Open both long and short positions simultaneously.
        
        Args:
            symbol: Trading pair
            quantity: Amount for each position
            max_retries: Number of retries on failure
            
        Returns:
            Tuple of (long_order, short_order)
        """
        symbol = symbol or self.symbol
        quantity = int(quantity)  # Ensure whole number for SOL on testnet
        
        for attempt in range(max_retries):
            try:
                # Open long position
                long_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    positionSide='LONG',
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # Small delay between orders
                time.sleep(1)
                
                # Open short position
                short_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    positionSide='SHORT',
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                price = self.get_price(symbol)
                print(f"✓ Opened HEDGED positions: {quantity} {symbol} @ ~{price:.4f}")
                print(f"  ├─ LONG:  {quantity} {symbol}")
                print(f"  └─ SHORT: {quantity} {symbol}")
                
                return long_order, short_order
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5 + attempt  # Increasing wait: 5, 6, 7, 8... seconds
                    print(f"  ⚠ Order attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
    
    def close_long_hedge(self, symbol: Optional[str] = None, quantity: float = 1, max_retries: int = 3) -> Dict[str, Any]:
        """Close long position in hedge mode."""
        symbol = symbol or self.symbol
        quantity = int(quantity)  # Ensure whole number for SOL on testnet
        
        for attempt in range(max_retries):
            try:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    positionSide='LONG',
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                return order
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise
    
    def close_short_hedge(self, symbol: Optional[str] = None, quantity: float = 1, max_retries: int = 3) -> Dict[str, Any]:
        """Close short position in hedge mode."""
        symbol = symbol or self.symbol
        quantity = int(quantity)  # Ensure whole number for SOL on testnet
        
        for attempt in range(max_retries):
            try:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    positionSide='SHORT',
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                return order
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise
