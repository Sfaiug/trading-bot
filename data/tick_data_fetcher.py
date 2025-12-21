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


CACHE_DIR = os.environ.get("TICK_DATA_CACHE", "cache/trades")
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


def download_and_aggregate_month(
    symbol: str, 
    year: int, 
    month: int,
    aggregate_seconds: float = 1.0
) -> Optional[Tuple[List[Tuple[datetime, float]], int]]:
    """
    Download and aggregate trades for a specific month.
    
    Memory-efficient: aggregates while streaming, never holds all trades.
    
    Returns:
        Tuple of (aggregated_prices, trade_count), or None if download failed.
    """
    url = f"{BASE_URL}/{symbol}/{symbol}-trades-{year}-{month:02d}.zip"
    
    try:
        # Stream download for large files with progress
        response = requests.get(url, timeout=300, stream=True)
        if response.status_code == 404:
            return None
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
        del chunks  # Free download chunks
        print("\r" + " " * 60 + "\r", end="", flush=True)
        
        # Stream through ZIP and aggregate on-the-fly
        # Never hold more than 1 window of trades in memory
        aggregated_prices = []
        trade_count = 0
        
        # VWAP window state
        window_start = None
        window_volume = 0.0
        window_value = 0.0
        
        with zipfile.ZipFile(BytesIO(content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as csv_file:
                first_line = True
                for raw_line in csv_file:
                    if first_line:
                        first_line = False
                        continue
                    
                    try:
                        line = raw_line.decode('utf-8').strip()
                        parts = line.split(',')
                        if len(parts) >= 6:
                            trade_count += 1
                            ts = datetime.fromtimestamp(int(parts[4]) / 1000)
                            price = float(parts[1])
                            qty = float(parts[2])
                            
                            # Initialize window
                            if window_start is None:
                                window_start = ts
                                window_volume = qty
                                window_value = price * qty
                                continue
                            
                            # Check if we should emit a completed window
                            elapsed = (ts - window_start).total_seconds()
                            if elapsed >= aggregate_seconds:
                                # Emit VWAP for completed window
                                if window_volume > 0:
                                    vwap = window_value / window_volume
                                    aggregated_prices.append((window_start, vwap))
                                
                                # Start new window
                                window_start = ts
                                window_volume = qty
                                window_value = price * qty
                            else:
                                # Accumulate in current window
                                window_volume += qty
                                window_value += price * qty
                    except:
                        continue
        
        # Emit final window
        if window_volume > 0 and window_start is not None:
            vwap = window_value / window_volume
            aggregated_prices.append((window_start, vwap))
        
        del content  # Free ZIP content
        
        return (aggregated_prices, trade_count)
        
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
    
    Memory-efficient: processes each month's trades immediately and discards them.
    
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
    
    # Check for pre-aggregated cache first
    agg_cache_path = os.path.join(CACHE_DIR, f"{symbol}_{years}yr_agg{int(aggregate_seconds)}s.json")
    if os.path.exists(agg_cache_path):
        if verbose:
            print(f"  Loading from aggregated cache...")
        try:
            with open(agg_cache_path, 'r') as f:
                data = json.load(f)
            all_prices = [(datetime.fromisoformat(ts), p) for ts, p in data]
            if verbose:
                print(f"  ✓ Loaded {len(all_prices):,} prices from cache")
                print(f"{'='*60}")
            return all_prices
        except Exception:
            pass
    
    for i, (year, month) in enumerate(months, 1):
        if verbose:
            print(f"  [{i}/{len(months)}] {year}-{month:02d}...", end=" ", flush=True)
        
        # Download and aggregate in one pass (memory-efficient)
        result = download_and_aggregate_month(symbol, year, month, aggregate_seconds)
        
        if result is None:
            if verbose:
                print("not available")
            continue
        
        aggregated, trade_count = result
        
        # Extend our results
        all_prices.extend(aggregated)
        
        if verbose:
            print(f"✓ {trade_count:,} trades → {len(aggregated):,} prices")
    
    # Sort by timestamp
    all_prices.sort(key=lambda x: x[0])
    
    # Cache the aggregated result for future runs
    if all_prices:
        try:
            ensure_cache_dir()
            cache_data = [(ts.isoformat(), p) for ts, p in all_prices]
            with open(agg_cache_path, 'w') as f:
                json.dump(cache_data, f)
            if verbose:
                print(f"  ✓ Saved aggregated cache for future runs")
        except Exception as e:
            if verbose:
                print(f"  (Could not save cache: {e})")
    
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
