#!/usr/bin/env python3
"""
Tick Data Fetcher for Binance Data Portal

Downloads tick-level trade data from data.binance.vision for accurate backtesting.
Trades include: id, price, qty, quoteQty, time, isBuyerMaker

Data URL pattern:
    https://data.binance.vision/data/futures/um/monthly/trades/{SYMBOL}/{SYMBOL}-trades-{YYYY-MM}.zip
"""

import os
import csv
import json
import zipfile
import requests
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass


CACHE_DIR = "cache/trades"
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/trades"


@dataclass
class Trade:
    """Single trade record."""
    timestamp: datetime
    price: float
    quantity: float
    is_buyer_maker: bool


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(symbol: str, year: int, month: int) -> str:
    """Get path to cached trade data."""
    return os.path.join(CACHE_DIR, f"{symbol}_{year}_{month:02d}_trades.json")


def get_months_in_range(years: int) -> List[Tuple[int, int]]:
    """Get list of (year, month) tuples for the past N years."""
    months = []
    now = datetime.now()
    
    # Start from 2 months ago (current month data may be incomplete)
    end_date = now.replace(day=1) - timedelta(days=1)
    start_date = end_date - timedelta(days=years * 365)
    
    current = start_date.replace(day=1)
    while current <= end_date:
        months.append((current.year, current.month))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return months


def download_month_trades(symbol: str, year: int, month: int) -> Optional[List[Trade]]:
    """
    Download and parse trades for a specific month.
    
    Returns:
        List of Trade objects, or None if download failed.
    """
    cache_path = get_cache_path(symbol, year, month)
    
    # Check cache first
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return [Trade(
                timestamp=datetime.fromisoformat(t['ts']),
                price=t['p'],
                quantity=t['q'],
                is_buyer_maker=t['m']
            ) for t in data]
        except Exception:
            pass  # Cache corrupted, re-download
    
    # Download from Binance Data Portal
    url = f"{BASE_URL}/{symbol}/{symbol}-trades-{year}-{month:02d}.zip"
    
    try:
        # Stream download for large files with progress
        response = requests.get(url, timeout=300, stream=True)
        if response.status_code == 404:
            return None  # Month not available
        response.raise_for_status()
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        chunks = []
        downloaded = 0
        
        for chunk in response.iter_content(chunk_size=8192):
            chunks.append(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                pct = (downloaded / total_size) * 100
                print(f"\r    Downloading: {mb_down:.1f}/{mb_total:.1f} MB ({pct:.0f}%)    ", end="", flush=True)
        
        content = b''.join(chunks)
        print("\r" + " " * 60 + "\r", end="", flush=True)  # Clear progress line
        
        # Extract ZIP in memory
        trades = []
        with zipfile.ZipFile(BytesIO(content)) as zf:
            # ZIP contains single CSV file
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as csv_file:
                # Skip header
                lines = csv_file.read().decode('utf-8').strip().split('\n')
                
                for line in lines[1:]:  # Skip header
                    parts = line.split(',')
                    if len(parts) >= 6:
                        # CSV format: id, price, qty, quoteQty, time, isBuyerMaker
                        trades.append(Trade(
                            timestamp=datetime.fromtimestamp(int(parts[4]) / 1000),
                            price=float(parts[1]),
                            quantity=float(parts[2]),
                            is_buyer_maker=parts[5].strip().lower() == 'true'
                        ))
        
        # Cache the data (compact format)
        ensure_cache_dir()
        cache_data = [{'ts': t.timestamp.isoformat(), 'p': t.price, 'q': t.quantity, 'm': t.is_buyer_maker} 
                      for t in trades]
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        
        return trades
        
    except Exception as e:
        print(f"Error downloading {symbol} {year}-{month:02d}: {e}")
        return None


def fetch_tick_data(
    symbol: str,
    years: int = 3,
    aggregate_seconds: float = 1.0,
    verbose: bool = True
) -> List[Tuple[datetime, float]]:
    """
    Fetch tick data and aggregate to price points.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        years: Number of years of history
        aggregate_seconds: Aggregate trades within this window (default 1s)
        verbose: Print progress
        
    Returns:
        List of (timestamp, price) tuples, sorted by time.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"FETCHING TICK DATA: {symbol}")
        print(f"{'='*60}")
        print(f"Period: {years} year(s)")
    
    months = get_months_in_range(years)
    all_prices = []
    
    for i, (year, month) in enumerate(months, 1):
        if verbose:
            print(f"  [{i}/{len(months)}] {year}-{month:02d}...", end=" ", flush=True)
        
        trades = download_month_trades(symbol, year, month)
        
        if trades is None:
            if verbose:
                print("not available")
            continue
        
        # Aggregate trades to price points
        # Use VWAP (Volume-Weighted Average Price) within each time window
        if aggregate_seconds > 0:
            aggregated = aggregate_trades(trades, aggregate_seconds)
        else:
            # Use every trade as a price point
            aggregated = [(t.timestamp, t.price) for t in trades]
        
        all_prices.extend(aggregated)
        
        if verbose:
            print(f"✓ {len(trades):,} trades → {len(aggregated):,} prices")
    
    # Sort by timestamp
    all_prices.sort(key=lambda x: x[0])
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Total: {len(all_prices):,} price points")
        if all_prices:
            print(f"Range: {all_prices[0][0].strftime('%Y-%m-%d')} to {all_prices[-1][0].strftime('%Y-%m-%d')}")
            print(f"Price: ${min(p for _, p in all_prices):.2f} - ${max(p for _, p in all_prices):.2f}")
        print(f"{'='*60}")
    
    return all_prices


def aggregate_trades(trades: List[Trade], window_seconds: float) -> List[Tuple[datetime, float]]:
    """
    Aggregate trades using VWAP within time windows.
    
    Each window produces one price point: volume-weighted average price.
    """
    if not trades:
        return []
    
    prices = []
    window_start = trades[0].timestamp
    window_volume = 0.0
    window_value = 0.0  # price * volume
    
    for trade in trades:
        elapsed = (trade.timestamp - window_start).total_seconds()
        
        if elapsed >= window_seconds:
            # Emit VWAP for completed window
            if window_volume > 0:
                vwap = window_value / window_volume
                prices.append((window_start, vwap))
            
            # Start new window
            window_start = trade.timestamp
            window_volume = trade.quantity
            window_value = trade.price * trade.quantity
        else:
            # Accumulate in current window
            window_volume += trade.quantity
            window_value += trade.price * trade.quantity
    
    # Emit final window
    if window_volume > 0:
        vwap = window_value / window_volume
        prices.append((window_start, vwap))
    
    return prices


def get_years_from_user() -> int:
    """Prompt user for number of years of data."""
    print("\n" + "=" * 50)
    print("TICK DATA CONFIGURATION")
    print("=" * 50)
    print("\nHow many years of data for simulation?")
    print("  More years = more accurate results, but longer download")
    print("  Estimated sizes: 1yr ≈ 5GB, 3yr ≈ 15GB, 5yr ≈ 25GB")
    print()
    
    while True:
        try:
            user_input = input("Enter years (1-5) [default: 3]: ").strip()
            if user_input == "":
                return 3
            years = int(user_input)
            if 1 <= years <= 5:
                return years
            print("  Please enter a number between 1 and 5.")
        except ValueError:
            print("  Invalid input. Please enter a number.")


# Test
if __name__ == "__main__":
    years = get_years_from_user()
    print(f"\nFetching {years} year(s) of BTCUSDT tick data...")
    
    prices = fetch_tick_data("BTCUSDT", years=years, aggregate_seconds=1.0)
    
    if prices:
        print(f"\nSuccess! Got {len(prices):,} price points")
        print(f"First: {prices[0]}")
        print(f"Last: {prices[-1]}")
