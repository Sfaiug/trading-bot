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
from enum import Enum
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass


# =============================================================================
# PHASE 3 FIX: Data Granularity Modes
# =============================================================================

class DataGranularity(Enum):
    """
    Data granularity modes for tick data streaming.

    The granularity mode significantly affects backtest accuracy:
    - RAW_TICKS: Maximum accuracy, uses every single trade (SLOW, 50-80GB cache)
    - TIME_SAMPLED: Good balance, samples at fixed time intervals (RECOMMENDED)
    - MOVE_FILTERED: Fastest, only keeps ticks that moved min_move_pct (current default)

    For optimizer: Use TIME_SAMPLED (100ms) for exploration, RAW_TICKS for final validation.
    """
    RAW_TICKS = "raw"        # Every tick (most accurate, slowest)
    TIME_SAMPLED = "time"    # Every N milliseconds (good balance)
    MOVE_FILTERED = "move"   # Only ticks that moved min_move_pct (fastest, least accurate)


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
                                # PHASE 2.1 FIX: Use END of window time, not start
                                # This prevents look-ahead bias - the price at time T
                                # only includes information up to time T, not after
                                if window_volume > 0:
                                    vwap = window_value / window_volume
                                    # Use current timestamp (end of window) for causal correctness
                                    aggregated_prices.append((ts, vwap))

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
        
        # Emit final window (use last trade timestamp for causal correctness)
        if window_volume > 0 and window_start is not None:
            vwap = window_value / window_volume
            # Use the last known timestamp, not window_start (Phase 2.1 fix)
            final_ts = aggregated_prices[-1][0] if aggregated_prices else window_start
            aggregated_prices.append((final_ts, vwap))
        
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
    expected_months = len(months)
    all_prices = []
    
    # Check for pre-aggregated cache first
    agg_cache_path = os.path.join(CACHE_DIR, f"{symbol}_{years}yr_agg{int(aggregate_seconds)}s.json")
    if os.path.exists(agg_cache_path):
        if verbose:
            print(f"  Checking cache...")
        try:
            with open(agg_cache_path, 'r') as f:
                cache = json.load(f)
            
            # Validate cache has metadata and is complete
            if isinstance(cache, dict) and cache.get('complete') == True:
                months_in_cache = cache.get('months_completed', 0)
                if months_in_cache >= expected_months - 2:  # Allow 2 months tolerance (data availability)
                    data = cache.get('prices', [])
                    all_prices = [(datetime.fromisoformat(ts), p) for ts, p in data]
                    if verbose:
                        print(f"  ✓ Loaded {len(all_prices):,} prices from cache ({months_in_cache}/{expected_months} months)")
                        print(f"{'='*60}")
                    return all_prices
                else:
                    if verbose:
                        print(f"  ⚠ Cache incomplete ({months_in_cache}/{expected_months} months), re-downloading...")
            else:
                # Old format or incomplete cache
                if verbose:
                    print(f"  ⚠ Cache invalid or incomplete, re-downloading...")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Cache error: {e}, re-downloading...")
    
    months_completed = 0
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
        months_completed += 1
        
        # Extend our results
        all_prices.extend(aggregated)
        
        if verbose:
            print(f"✓ {trade_count:,} trades → {len(aggregated):,} prices")
    
    # Sort by timestamp
    all_prices.sort(key=lambda x: x[0])
    
    # Cache the aggregated result with metadata
    if all_prices:
        try:
            ensure_cache_dir()
            cache_data = {
                'complete': True,
                'months_expected': expected_months,
                'months_completed': months_completed,
                'years': years,
                'aggregate_seconds': aggregate_seconds,
                'total_prices': len(all_prices),
                'prices': [(ts.isoformat(), p) for ts, p in all_prices]
            }
            with open(agg_cache_path, 'w') as f:
                json.dump(cache_data, f)
            if verbose:
                print(f"  ✓ Saved cache ({months_completed}/{expected_months} months, {len(all_prices):,} prices)")
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


def get_streaming_cache_path(symbol: str, years: int, aggregate_seconds: float) -> str:
    """Get path to streaming cache file (CSV format for line-by-line reading)."""
    return os.path.join(CACHE_DIR, f"{symbol}_{years}yr_agg{int(aggregate_seconds)}s_stream.csv")


def ensure_cached_tick_data(
    symbol: str,
    years: int = 3,
    aggregate_seconds: float = 1.0,
    verbose: bool = True
) -> str:
    """
    Ensure tick data is downloaded and cached to disk in streaming format.

    Downloads month-by-month and writes to a CSV file that can be streamed
    line-by-line without loading all data into memory.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        years: Number of years of history
        aggregate_seconds: Aggregate trades within this window (default 1s)
        verbose: Print progress

    Returns:
        Path to the cache file.
    """
    ensure_cache_dir()
    cache_path = get_streaming_cache_path(symbol, years, aggregate_seconds)

    # Check if cache already exists
    if os.path.exists(cache_path):
        if verbose:
            # Count lines to report size
            with open(cache_path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"  ✓ Using cached data: {line_count:,} prices from {cache_path}")
        return cache_path

    if verbose:
        print(f"\n{'='*60}")
        print(f"DOWNLOADING & CACHING: {symbol}")
        print(f"{'='*60}")
        print(f"Period: {years} year(s), {aggregate_seconds}s aggregation")

    months = get_months_in_range(years)
    total_prices = 0
    total_trades = 0

    # Write directly to disk as we download each month
    with open(cache_path, 'w') as f:
        for i, (year, month) in enumerate(months, 1):
            if verbose:
                print(f"  [{i}/{len(months)}] {year}-{month:02d}...", end=" ", flush=True)

            result = download_and_aggregate_month(symbol, year, month, aggregate_seconds)

            if result is None:
                if verbose:
                    print("not available")
                continue

            aggregated, trade_count = result
            total_trades += trade_count

            # Write each price point to disk immediately
            for ts, price in aggregated:
                f.write(f"{ts.isoformat()},{price}\n")
                total_prices += 1

            # Flush to ensure data is written
            f.flush()

            if verbose:
                print(f"✓ {trade_count:,} trades → {len(aggregated):,} prices")

            # Free memory from this month's data
            del aggregated

    if verbose:
        print(f"\n{'='*60}")
        print(f"Total: {total_prices:,} prices from {total_trades:,} trades")
        print(f"Cached to: {cache_path}")
        cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"Cache size: {cache_size_mb:.1f} MB")
        print(f"{'='*60}")

    return cache_path


def create_price_streamer(
    symbol: str,
    years: int = 3,
    aggregate_seconds: float = 1.0,
    verbose: bool = True
) -> callable:
    """
    Create a function that streams prices from cache.

    This ensures data is cached first, then returns a function that can be
    called multiple times to stream through the prices. Each call creates
    a fresh generator, allowing multiple backtests to iterate through the data.

    Usage:
        streamer = create_price_streamer("BTCUSDT", years=3)

        # For each backtest:
        for timestamp, price in streamer():
            # process price

    Args:
        symbol: Trading pair
        years: Number of years of history
        aggregate_seconds: Aggregation window
        verbose: Print progress during initial cache creation

    Returns:
        A callable that returns a generator of (datetime, price) tuples.
    """
    # Ensure data is cached (downloads if needed)
    cache_path = ensure_cached_tick_data(symbol, years, aggregate_seconds, verbose)

    def stream_prices() -> Generator[Tuple[datetime, float], None, None]:
        """Generator that streams prices from the cache file."""
        with open(cache_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    ts_str, price_str = line.split(',')
                    yield (datetime.fromisoformat(ts_str), float(price_str))

    return stream_prices


def count_cached_prices(symbol: str, years: int, aggregate_seconds: float = 1.0) -> int:
    """Count the number of prices in a cached file without loading all into memory."""
    cache_path = get_streaming_cache_path(symbol, years, aggregate_seconds)
    if not os.path.exists(cache_path):
        return 0
    with open(cache_path, 'r') as f:
        return sum(1 for _ in f)


def get_tick_cache_path(symbol: str, years: int) -> str:
    """Get path to raw tick cache file (no aggregation)."""
    return os.path.join(CACHE_DIR, f"{symbol}_{years}yr_ticks.csv")


def download_raw_ticks_to_file(
    symbol: str,
    year: int,
    month: int,
    output_file
) -> int:
    """
    Download raw trades for a month and write directly to file.

    Memory-efficient: streams from ZIP directly to output file.

    Args:
        symbol: Trading pair
        year: Year
        month: Month
        output_file: Open file handle to write to

    Returns:
        Number of trades written, or -1 if download failed.
    """
    url = f"{BASE_URL}/{symbol}/{symbol}-trades-{year}-{month:02d}.zip"

    try:
        response = requests.get(url, timeout=300, stream=True)
        if response.status_code == 404:
            return -1
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
        del chunks
        print("\r" + " " * 60 + "\r", end="", flush=True)

        # Stream through ZIP and write each trade directly
        trade_count = 0

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
                            # Extract timestamp (ms) and price
                            ts_ms = int(parts[4])
                            price = float(parts[1])

                            # Write as: timestamp_ms,price
                            output_file.write(f"{ts_ms},{price}\n")
                            trade_count += 1
                    except:
                        continue

        del content
        return trade_count

    except Exception as e:
        print(f"Error downloading {symbol} {year}-{month:02d}: {e}")
        return -1


def ensure_cached_tick_data_raw(
    symbol: str,
    years: int = 3,
    verbose: bool = True
) -> str:
    """
    Ensure RAW tick data (no aggregation) is downloaded and cached.

    Downloads all trades and stores them as-is for maximum accuracy.
    Warning: This creates LARGE files (~50-80 GB per coin for 5 years).

    Args:
        symbol: Trading pair
        years: Number of years
        verbose: Print progress

    Returns:
        Path to the cache file.
    """
    ensure_cache_dir()
    cache_path = get_tick_cache_path(symbol, years)

    # Check if cache already exists
    if os.path.exists(cache_path):
        if verbose:
            file_size_gb = os.path.getsize(cache_path) / (1024 ** 3)
            with open(cache_path, 'r') as f:
                # Count lines efficiently
                line_count = sum(1 for _ in f)
            print(f"  ✓ Using cached raw ticks: {line_count:,} trades ({file_size_gb:.1f} GB)")
        return cache_path

    if verbose:
        print(f"\n{'='*60}")
        print(f"DOWNLOADING RAW TICK DATA: {symbol}")
        print(f"{'='*60}")
        print(f"Period: {years} year(s), NO aggregation (tick-by-tick)")
        print(f"⚠️  WARNING: This will create a LARGE cache file!")

    months = get_months_in_range(years)
    total_trades = 0

    # Write directly to disk as we download each month
    with open(cache_path, 'w') as f:
        for i, (year, month) in enumerate(months, 1):
            if verbose:
                print(f"  [{i}/{len(months)}] {year}-{month:02d}...", end=" ", flush=True)

            trade_count = download_raw_ticks_to_file(symbol, year, month, f)

            if trade_count < 0:
                if verbose:
                    print("not available")
                continue

            total_trades += trade_count
            f.flush()

            if verbose:
                print(f"✓ {trade_count:,} raw trades")

    if verbose:
        file_size_gb = os.path.getsize(cache_path) / (1024 ** 3)
        print(f"\n{'='*60}")
        print(f"Total: {total_trades:,} raw trades")
        print(f"Cached to: {cache_path}")
        print(f"Cache size: {file_size_gb:.2f} GB")
        print(f"{'='*60}")

    return cache_path


def create_tick_streamer_raw(
    symbol: str,
    years: int = 3,
    verbose: bool = True
) -> callable:
    """
    Create a function that streams RAW ticks from cache.

    Each call returns a generator that yields every single trade.
    Use for maximum accuracy verification of backtest results.

    Args:
        symbol: Trading pair
        years: Number of years
        verbose: Print progress during cache creation

    Returns:
        Callable that returns a generator of (datetime, price) tuples.
    """
    cache_path = ensure_cached_tick_data_raw(symbol, years, verbose)

    def stream_ticks() -> Generator[Tuple[datetime, float], None, None]:
        """Generator that streams raw ticks from cache."""
        with open(cache_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    ts_ms_str, price_str = line.split(',')
                    ts = datetime.fromtimestamp(int(ts_ms_str) / 1000)
                    yield (ts, float(price_str))

    return stream_ticks


def create_filtered_tick_streamer(
    symbol: str,
    years: int = 3,
    min_move_pct: float = 0.01,
    verbose: bool = True,
    preload_to_memory: bool = True
) -> callable:
    """
    Create a function that streams FILTERED ticks from cache.

    Only yields prices that moved at least min_move_pct from the previous
    yielded price. This dramatically reduces data points while preserving
    all meaningful price movements for strategy simulation.

    MEMORY OPTIMIZATION: When preload_to_memory=True (default), loads all
    filtered ticks into memory ONCE. This avoids re-reading the file for
    each backtest iteration (332K+ file reads → 1 read).

    Args:
        symbol: Trading pair
        years: Number of years
        min_move_pct: Minimum price move % to include (default 0.01% = 1 basis point)
        verbose: Print progress during cache creation
        preload_to_memory: Load all ticks into memory for faster iteration (default True)

    Returns:
        Callable that returns a generator of (datetime, price) tuples.

    Example:
        With min_move_pct=0.01, only keeps prices that moved 0.01% or more:
        $50,000 → $50,005 (kept, +0.01%) → $50,006 (skipped, +0.002%) → $50,010 (kept, +0.01%)
    """
    cache_path = ensure_cached_tick_data_raw(symbol, years, verbose)

    if preload_to_memory:
        # MEMORY-OPTIMIZED: Load all filtered ticks into memory ONCE
        if verbose:
            print(f"  Loading filtered ticks into memory...")

        filtered_ticks = []
        last_price = None
        total_ticks = 0

        with open(cache_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                total_ticks += 1
                ts_ms_str, price_str = line.split(',')
                price = float(price_str)

                if last_price is None:
                    # Always keep first price
                    last_price = price
                    filtered_ticks.append((datetime.fromtimestamp(int(ts_ms_str) / 1000), price))
                else:
                    # Check if price moved enough
                    move_pct = abs((price - last_price) / last_price) * 100
                    if move_pct >= min_move_pct:
                        last_price = price
                        filtered_ticks.append((datetime.fromtimestamp(int(ts_ms_str) / 1000), price))

        if verbose and total_ticks > 0:
            kept_ticks = len(filtered_ticks)
            reduction = (1 - kept_ticks / total_ticks) * 100
            mem_mb = (kept_ticks * 48) / (1024 * 1024)  # Rough estimate: 48 bytes per tuple
            print(f"  ✓ Loaded {kept_ticks:,} ticks into memory (~{mem_mb:.1f} MB)")
            print(f"    Filter: {total_ticks:,} raw → {kept_ticks:,} filtered ({reduction:.1f}% reduction)")

        def stream_from_memory() -> Generator[Tuple[datetime, float], None, None]:
            """Generator that yields from pre-loaded memory list."""
            for tick in filtered_ticks:
                yield tick

        return stream_from_memory

    else:
        # DISK STREAMING: Read from file each time (original behavior)
        def stream_filtered_ticks() -> Generator[Tuple[datetime, float], None, None]:
            """Generator that streams filtered ticks from cache."""
            last_price = None
            total_ticks = 0
            kept_ticks = 0

            with open(cache_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    total_ticks += 1
                    ts_ms_str, price_str = line.split(',')
                    price = float(price_str)

                    if last_price is None:
                        # Always keep first price
                        last_price = price
                        kept_ticks += 1
                        yield (datetime.fromtimestamp(int(ts_ms_str) / 1000), price)
                    else:
                        # Check if price moved enough
                        move_pct = abs((price - last_price) / last_price) * 100
                        if move_pct >= min_move_pct:
                            last_price = price
                            kept_ticks += 1
                            yield (datetime.fromtimestamp(int(ts_ms_str) / 1000), price)

            if verbose and total_ticks > 0:
                reduction = (1 - kept_ticks / total_ticks) * 100
                print(f"    Tick filter: {total_ticks:,} → {kept_ticks:,} ({reduction:.1f}% reduction)")

        return stream_filtered_ticks


def count_filtered_ticks(symbol: str, years: int, min_move_pct: float = 0.01) -> Tuple[int, int]:
    """
    Count filtered ticks without running full simulation.

    Returns:
        Tuple of (total_raw_ticks, filtered_ticks)
    """
    cache_path = get_tick_cache_path(symbol, years)
    if not os.path.exists(cache_path):
        return 0, 0

    last_price = None
    total_ticks = 0
    kept_ticks = 0

    with open(cache_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_ticks += 1
            _, price_str = line.split(',')
            price = float(price_str)

            if last_price is None:
                last_price = price
                kept_ticks += 1
            else:
                move_pct = abs((price - last_price) / last_price) * 100
                if move_pct >= min_move_pct:
                    last_price = price
                    kept_ticks += 1

    return total_ticks, kept_ticks


def count_cached_ticks_raw(symbol: str, years: int) -> int:
    """Count raw ticks in cache without loading into memory."""
    cache_path = get_tick_cache_path(symbol, years)
    if not os.path.exists(cache_path):
        return 0
    with open(cache_path, 'r') as f:
        return sum(1 for _ in f)


def aggregate_ticks_to_interval(
    tick_streamer: callable,
    interval_seconds: float
) -> Generator[Tuple[datetime, float], None, None]:
    """
    Aggregate tick data to custom time intervals.

    Args:
        tick_streamer: Callable that returns generator of (datetime, price) tuples
        interval_seconds: Aggregation interval in seconds.
                         0 = tick-by-tick (pass through all ticks)
                         >0 = aggregate to VWAP for each interval

    Yields:
        (datetime, price) tuples at the specified interval

    Examples:
        interval_seconds=0    -> every tick (live mode)
        interval_seconds=0.1  -> 100ms aggregation
        interval_seconds=1.0  -> 1 second aggregation
        interval_seconds=6.4  -> 6.4 second aggregation
    """
    if interval_seconds == 0:
        # Tick mode - pass through all ticks
        yield from tick_streamer()
        return

    # Aggregate to intervals
    current_window_start = None
    window_prices = []

    for ts, price in tick_streamer():
        # Calculate window start time
        # For sub-second intervals, use microseconds
        if interval_seconds < 1:
            # Convert to microseconds for precision
            total_micros = ts.hour * 3600_000_000 + ts.minute * 60_000_000 + ts.second * 1_000_000 + ts.microsecond
            interval_micros = int(interval_seconds * 1_000_000)
            window_micros = (total_micros // interval_micros) * interval_micros

            window_second = (window_micros // 1_000_000) % 60
            window_minute = (window_micros // 60_000_000) % 60
            window_hour = (window_micros // 3600_000_000) % 24
            window_micro = window_micros % 1_000_000

            window_start = ts.replace(
                hour=window_hour,
                minute=window_minute,
                second=window_second,
                microsecond=window_micro
            )
        else:
            # For >= 1 second intervals
            total_seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
            window_seconds = int((total_seconds // interval_seconds) * interval_seconds)

            window_hour = window_seconds // 3600
            window_minute = (window_seconds % 3600) // 60
            window_second = window_seconds % 60

            window_start = ts.replace(
                hour=window_hour,
                minute=window_minute,
                second=window_second,
                microsecond=0
            )

        if current_window_start is None:
            current_window_start = window_start

        if window_start != current_window_start:
            # Emit average price for completed window
            if window_prices:
                avg_price = sum(window_prices) / len(window_prices)
                yield (current_window_start, avg_price)

            current_window_start = window_start
            window_prices = []

        window_prices.append(price)

    # Emit final window
    if window_prices:
        avg_price = sum(window_prices) / len(window_prices)
        yield (current_window_start, avg_price)


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


# =============================================================================
# PHASE 3 FIX: Configurable Tick Streamer with Granularity Modes
# =============================================================================

def create_configurable_tick_streamer(
    symbol: str,
    years: int = 3,
    granularity: DataGranularity = DataGranularity.TIME_SAMPLED,
    sample_interval_ms: float = 100.0,
    min_move_pct: float = 0.01,
    verbose: bool = True,
    preload_to_memory: bool = False
) -> callable:
    """
    Create a tick streamer with configurable data granularity.

    This is the RECOMMENDED function for creating tick data streamers.
    It provides multiple granularity modes to balance accuracy vs speed.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        years: Number of years of history
        granularity: DataGranularity mode:
            - RAW_TICKS: Every tick (most accurate, 50-80GB cache)
            - TIME_SAMPLED: Every sample_interval_ms (recommended, good balance)
            - MOVE_FILTERED: Only ticks that moved min_move_pct (fastest)
        sample_interval_ms: Time interval for TIME_SAMPLED mode (default 100ms)
        min_move_pct: Minimum move % for MOVE_FILTERED mode (default 0.01%)
        verbose: Print progress
        preload_to_memory: Load all ticks into memory (faster but uses more RAM)

    Returns:
        Callable that returns a generator of (datetime, price) tuples.

    Usage Examples:
        # For optimizer exploration (fast, reasonable accuracy):
        streamer = create_configurable_tick_streamer(
            "BTCUSDT", years=5,
            granularity=DataGranularity.TIME_SAMPLED,
            sample_interval_ms=100  # 100ms = 10 samples/second
        )

        # For final validation (maximum accuracy):
        streamer = create_configurable_tick_streamer(
            "BTCUSDT", years=5,
            granularity=DataGranularity.RAW_TICKS
        )

        # For quick prototyping (fastest):
        streamer = create_configurable_tick_streamer(
            "BTCUSDT", years=5,
            granularity=DataGranularity.MOVE_FILTERED,
            min_move_pct=0.05  # Only 5 basis point moves
        )
    """
    if verbose:
        mode_desc = {
            DataGranularity.RAW_TICKS: "RAW_TICKS (every tick)",
            DataGranularity.TIME_SAMPLED: f"TIME_SAMPLED ({sample_interval_ms}ms)",
            DataGranularity.MOVE_FILTERED: f"MOVE_FILTERED ({min_move_pct}% min move)"
        }
        print(f"  Data granularity: {mode_desc[granularity]}")

    if granularity == DataGranularity.RAW_TICKS:
        # Maximum accuracy: every single tick
        return create_tick_streamer_raw(symbol, years, verbose)

    elif granularity == DataGranularity.TIME_SAMPLED:
        # Time-based sampling: emit one price per interval
        return _create_time_sampled_streamer(
            symbol, years, sample_interval_ms, verbose, preload_to_memory
        )

    elif granularity == DataGranularity.MOVE_FILTERED:
        # Move-based filtering: only emit when price moves enough
        return create_filtered_tick_streamer(
            symbol, years, min_move_pct, verbose, preload_to_memory
        )

    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def _create_time_sampled_streamer(
    symbol: str,
    years: int,
    sample_interval_ms: float,
    verbose: bool,
    preload_to_memory: bool
) -> callable:
    """
    Create a time-sampled tick streamer.

    Emits one price per sample_interval_ms, using the last price in each interval.
    This preserves timing relationships better than move-filtered for strategy testing.
    """
    cache_path = ensure_cached_tick_data_raw(symbol, years, verbose)
    interval_ms = int(sample_interval_ms)

    if preload_to_memory:
        if verbose:
            print(f"  Loading time-sampled ticks into memory ({interval_ms}ms intervals)...")

        sampled_ticks = []
        current_window_ms = None
        last_price_in_window = None
        last_ts_in_window = None
        total_ticks = 0

        with open(cache_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                total_ticks += 1
                ts_ms_str, price_str = line.split(',')
                ts_ms = int(ts_ms_str)
                price = float(price_str)

                # Calculate window
                window_ms = (ts_ms // interval_ms) * interval_ms

                if current_window_ms is None:
                    current_window_ms = window_ms
                    last_price_in_window = price
                    last_ts_in_window = ts_ms
                elif window_ms != current_window_ms:
                    # Emit last price from previous window
                    sampled_ticks.append((
                        datetime.fromtimestamp(last_ts_in_window / 1000),
                        last_price_in_window
                    ))
                    current_window_ms = window_ms
                    last_price_in_window = price
                    last_ts_in_window = ts_ms
                else:
                    # Update last price in current window
                    last_price_in_window = price
                    last_ts_in_window = ts_ms

        # Emit final window
        if last_price_in_window is not None:
            sampled_ticks.append((
                datetime.fromtimestamp(last_ts_in_window / 1000),
                last_price_in_window
            ))

        if verbose and total_ticks > 0:
            kept_ticks = len(sampled_ticks)
            reduction = (1 - kept_ticks / total_ticks) * 100
            mem_mb = (kept_ticks * 48) / (1024 * 1024)
            print(f"  ✓ Loaded {kept_ticks:,} ticks into memory (~{mem_mb:.1f} MB)")
            print(f"    Sampling: {total_ticks:,} raw → {kept_ticks:,} sampled ({reduction:.1f}% reduction)")

        def stream_from_memory() -> Generator[Tuple[datetime, float], None, None]:
            for tick in sampled_ticks:
                yield tick

        return stream_from_memory

    else:
        # Disk streaming mode
        def stream_time_sampled() -> Generator[Tuple[datetime, float], None, None]:
            current_window_ms = None
            last_price_in_window = None
            last_ts_in_window = None

            with open(cache_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    ts_ms_str, price_str = line.split(',')
                    ts_ms = int(ts_ms_str)
                    price = float(price_str)

                    window_ms = (ts_ms // interval_ms) * interval_ms

                    if current_window_ms is None:
                        current_window_ms = window_ms
                        last_price_in_window = price
                        last_ts_in_window = ts_ms
                    elif window_ms != current_window_ms:
                        # Emit last price from previous window
                        yield (
                            datetime.fromtimestamp(last_ts_in_window / 1000),
                            last_price_in_window
                        )
                        current_window_ms = window_ms
                        last_price_in_window = price
                        last_ts_in_window = ts_ms
                    else:
                        last_price_in_window = price
                        last_ts_in_window = ts_ms

            # Emit final window
            if last_price_in_window is not None:
                yield (
                    datetime.fromtimestamp(last_ts_in_window / 1000),
                    last_price_in_window
                )

        return stream_time_sampled


def estimate_tick_counts(symbol: str, years: int) -> dict:
    """
    Estimate tick counts for different granularity modes.

    Useful for planning which granularity to use.

    Returns:
        Dict with estimated counts for each mode.
    """
    cache_path = get_tick_cache_path(symbol, years)
    if not os.path.exists(cache_path):
        return {
            'raw_ticks': 0,
            'time_sampled_100ms': 0,
            'move_filtered_001pct': 0,
            'cache_exists': False
        }

    # Count raw ticks
    total_raw = 0
    with open(cache_path, 'r') as f:
        for _ in f:
            total_raw += 1

    # Estimate based on typical ratios:
    # - 100ms sampling typically reduces by 90-95%
    # - 0.01% move filter typically reduces by 99%+

    return {
        'raw_ticks': total_raw,
        'time_sampled_100ms': int(total_raw * 0.05),  # ~5% of raw
        'time_sampled_1s': int(total_raw * 0.01),     # ~1% of raw
        'move_filtered_001pct': int(total_raw * 0.01),  # ~1% of raw
        'move_filtered_01pct': int(total_raw * 0.001),  # ~0.1% of raw
        'cache_exists': True
    }


# =============================================================================
# PHASE 3.2 FIX: Data Quality Validation
# =============================================================================

@dataclass
class DataQualityReport:
    """
    Report of data quality issues found in tick data.

    Data quality issues can severely impact backtest reliability:
    - Gaps: Missing data means missed trading opportunities or false signals
    - Flash crashes: Extreme moves may trigger false entries/exits
    - Anomalies: Corrupt data leads to impossible P&L calculations
    """
    # Summary stats
    total_ticks: int = 0
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    total_days: float = 0.0

    # Gap analysis
    gap_count: int = 0
    max_gap_hours: float = 0.0
    gaps_over_1h: List[Tuple[datetime, datetime, float]] = None  # (start, end, hours)
    gaps_over_24h: List[Tuple[datetime, datetime, float]] = None
    gaps_over_7d: int = 0

    # Flash crash detection
    flash_crash_count: int = 0
    flash_crashes: List[Tuple[datetime, float, float]] = None  # (time, move_pct, duration_sec)

    # Price anomalies
    zero_price_count: int = 0
    negative_price_count: int = 0
    extreme_price_count: int = 0  # Prices > 10x or < 0.1x median

    # Overall verdict
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.gaps_over_1h is None:
            self.gaps_over_1h = []
        if self.gaps_over_24h is None:
            self.gaps_over_24h = []
        if self.flash_crashes is None:
            self.flash_crashes = []
        if self.warnings is None:
            self.warnings = []

    @property
    def quality_score(self) -> float:
        """
        Calculate overall data quality score (0-100).

        Scoring:
        - Start at 100
        - Deduct 5 points per gap over 1 hour
        - Deduct 20 points per gap over 24 hours
        - Deduct 10 points per flash crash
        - Deduct 50 points if data is invalid
        - Minimum score is 0
        """
        score = 100.0

        # Penalize gaps
        score -= len(self.gaps_over_1h) * 5
        score -= len(self.gaps_over_24h) * 20

        # Penalize flash crashes
        score -= self.flash_crash_count * 10

        # Penalize anomalies
        score -= self.zero_price_count * 20
        score -= self.negative_price_count * 50

        # Invalid data is heavily penalized
        if not self.is_valid:
            score -= 50

        return max(0.0, min(100.0, score))


def validate_data_quality(
    prices: List[Tuple[datetime, float]],
    max_gap_days: float = 7.0,
    flash_crash_threshold_pct: float = 10.0,
    flash_crash_window_sec: float = 60.0,
    verbose: bool = True
) -> DataQualityReport:
    """
    Validate tick data quality before backtesting.

    PHASE 3.2 FIX: Ensures data doesn't have issues that would corrupt backtest results.

    Args:
        prices: List of (timestamp, price) tuples
        max_gap_days: Reject data with gaps larger than this (default 7 days)
        flash_crash_threshold_pct: Flag moves larger than this in short window
        flash_crash_window_sec: Window for flash crash detection
        verbose: Print detailed report

    Returns:
        DataQualityReport with all findings
    """
    report = DataQualityReport()

    if not prices:
        report.is_valid = False
        report.rejection_reason = "No price data provided"
        return report

    report.total_ticks = len(prices)
    report.first_timestamp = prices[0][0]
    report.last_timestamp = prices[-1][0]
    report.total_days = (prices[-1][0] - prices[0][0]).total_seconds() / 86400

    # Calculate median price for anomaly detection
    all_prices = [p[1] for p in prices]
    sorted_prices = sorted(all_prices)
    median_price = sorted_prices[len(sorted_prices) // 2]

    # Scan through data
    prev_ts = prices[0][0]
    prev_price = prices[0][1]

    for i, (ts, price) in enumerate(prices[1:], start=1):
        # Check for price anomalies
        if price <= 0:
            if price == 0:
                report.zero_price_count += 1
            else:
                report.negative_price_count += 1
            continue

        if price > median_price * 10 or price < median_price * 0.1:
            report.extreme_price_count += 1

        # Check for gaps
        gap_seconds = (ts - prev_ts).total_seconds()
        gap_hours = gap_seconds / 3600

        if gap_hours > 1:
            report.gap_count += 1
            report.gaps_over_1h.append((prev_ts, ts, gap_hours))

            if gap_hours > report.max_gap_hours:
                report.max_gap_hours = gap_hours

            if gap_hours > 24:
                report.gaps_over_24h.append((prev_ts, ts, gap_hours))

                if gap_hours > 24 * 7:
                    report.gaps_over_7d += 1

        # Check for flash crashes (large move in short time)
        if gap_seconds <= flash_crash_window_sec and gap_seconds > 0:
            move_pct = abs((price - prev_price) / prev_price) * 100
            if move_pct >= flash_crash_threshold_pct:
                report.flash_crash_count += 1
                report.flash_crashes.append((ts, move_pct, gap_seconds))

        prev_ts = ts
        prev_price = price

    # Generate warnings
    if report.gap_count > 0:
        report.warnings.append(f"Found {report.gap_count} gaps > 1 hour")

    if report.gaps_over_24h:
        report.warnings.append(f"Found {len(report.gaps_over_24h)} gaps > 24 hours")

    if report.flash_crash_count > 0:
        report.warnings.append(
            f"Found {report.flash_crash_count} flash crashes (>{flash_crash_threshold_pct}% in <{flash_crash_window_sec}s)"
        )

    if report.zero_price_count > 0:
        report.warnings.append(f"Found {report.zero_price_count} zero prices")

    if report.extreme_price_count > 0:
        report.warnings.append(f"Found {report.extreme_price_count} extreme prices (>10x or <0.1x median)")

    # Determine validity
    if report.gaps_over_7d > 0:
        report.is_valid = False
        report.rejection_reason = f"Data has {report.gaps_over_7d} gaps longer than 7 days"
    elif report.negative_price_count > 0:
        report.is_valid = False
        report.rejection_reason = f"Data has {report.negative_price_count} negative prices"
    elif report.total_ticks < 1000:
        report.is_valid = False
        report.rejection_reason = f"Insufficient data: only {report.total_ticks} ticks"
    elif report.max_gap_hours > max_gap_days * 24:
        report.is_valid = False
        report.rejection_reason = f"Max gap ({report.max_gap_hours:.1f}h) exceeds limit ({max_gap_days * 24}h)"

    if verbose:
        print("\n" + "=" * 60)
        print("DATA QUALITY VALIDATION REPORT")
        print("=" * 60)
        print(f"Total ticks: {report.total_ticks:,}")
        print(f"Date range: {report.first_timestamp} to {report.last_timestamp}")
        print(f"Total days: {report.total_days:.1f}")
        print(f"\nGap Analysis:")
        print(f"  Gaps > 1 hour: {report.gap_count}")
        print(f"  Gaps > 24 hours: {len(report.gaps_over_24h)}")
        print(f"  Gaps > 7 days: {report.gaps_over_7d}")
        print(f"  Max gap: {report.max_gap_hours:.1f} hours")
        print(f"\nAnomaly Detection:")
        print(f"  Flash crashes: {report.flash_crash_count}")
        print(f"  Zero prices: {report.zero_price_count}")
        print(f"  Extreme prices: {report.extreme_price_count}")

        if report.warnings:
            print(f"\nWarnings:")
            for w in report.warnings:
                print(f"  - {w}")

        print(f"\nVerdict: {'VALID' if report.is_valid else 'REJECTED'}")
        if report.rejection_reason:
            print(f"Reason: {report.rejection_reason}")
        print("=" * 60)

    return report


def filter_anomalous_ticks(
    prices: List[Tuple[datetime, float]],
    remove_zero: bool = True,
    remove_extreme: bool = True,
    extreme_threshold: float = 5.0
) -> List[Tuple[datetime, float]]:
    """
    Filter out anomalous ticks from price data.

    Args:
        prices: List of (timestamp, price) tuples
        remove_zero: Remove ticks with zero price
        remove_extreme: Remove extreme price outliers
        extreme_threshold: Multiple of median to consider extreme

    Returns:
        Cleaned price list
    """
    if not prices:
        return []

    all_prices = [p[1] for p in prices if p[1] > 0]
    if not all_prices:
        return []

    sorted_prices = sorted(all_prices)
    median_price = sorted_prices[len(sorted_prices) // 2]

    filtered = []
    for ts, price in prices:
        # Skip zero prices
        if remove_zero and price <= 0:
            continue

        # Skip extreme prices
        if remove_extreme:
            if price > median_price * extreme_threshold:
                continue
            if price < median_price / extreme_threshold:
                continue

        filtered.append((ts, price))

    return filtered


# Test
if __name__ == "__main__":
    years = get_years_from_user()
    print(f"\nFetching {years} year(s) of BTCUSDT tick data...")

    prices = fetch_tick_data("BTCUSDT", years=years, aggregate_seconds=1.0)

    if prices:
        print(f"\nSuccess! Got {len(prices):,} price points")
        print(f"First: {prices[0]}")
        print(f"Last: {prices[-1]}")
