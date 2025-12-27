"""
Memory-Mapped Price Data for Low-RAM Optimization

This module provides disk-based price storage that never loads the full dataset
into RAM. Uses numpy memory-mapped files so the OS handles paging.

Key Features:
- Downloads data month-by-month, writes directly to disk
- Memory-maps the file for efficient random access
- Slicing returns views, not copies
- Iteration yields tuples without loading all into RAM

Memory Usage:
- Download phase: ~100-200 MB peak (one month at a time)
- Optimization phase: ~50-100 MB (only accessed portions paged in)
- Total dataset on disk: ~1-2 GB for 5 years

Usage:
    prices_file, meta = ensure_prices_on_disk("BTCUSDT", years=5)
    prices = load_prices_memmap(prices_file, meta)

    # Slice for fold (returns view, not copy)
    train_slice = prices[start_idx:end_idx]

    # Convert to iterator for backtest
    price_iter = memmap_to_iterator(train_slice)
    result = run_pyramid_backtest(price_iter, ...)
"""

import os
import gc
import json
import struct
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Iterator, Optional


# Record format: timestamp (int64, 8 bytes) + price (float32, 4 bytes) = 12 bytes
RECORD_DTYPE = np.dtype([('timestamp', '<i8'), ('price', '<f4')])
RECORD_SIZE = 12


def ensure_prices_on_disk(
    symbol: str,
    years: int,
    cache_dir: str = "./cache/memmap",
    verbose: bool = True
) -> Tuple[str, Dict]:
    """
    Ensure tick data is downloaded and cached to disk in memory-mapped format.

    Downloads data month-by-month and writes directly to disk, never holding
    the full dataset in RAM.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        years: Number of years of history
        cache_dir: Directory for cache files
        verbose: Print progress

    Returns:
        Tuple of (prices_file_path, metadata_dict)
    """
    os.makedirs(cache_dir, exist_ok=True)
    prices_file = os.path.join(cache_dir, f"{symbol}_{years}yr.mmap")
    meta_file = os.path.join(cache_dir, f"{symbol}_{years}yr.meta.json")

    # Check if already cached
    if os.path.exists(prices_file) and os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        if verbose:
            print(f"\n  [CACHE HIT] Found existing data for {symbol}")
            print(f"  Using cached data: {meta['total_prices']:,} prices ({meta['file_size_mb']:.1f} MB)")
            print(f"  Cache file: {prices_file}")
        return prices_file, meta

    if verbose:
        print(f"\n  [CACHE MISS] No cached data for {symbol} ({years}yr)")
        print(f"  Will download to: {prices_file}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"DOWNLOADING TO DISK: {symbol}")
        print(f"{'='*60}")
        print(f"Period: {years} year(s)")
        print(f"Output: {prices_file}")
        print()

    # Import data fetcher functions
    from data.tick_data_fetcher import get_months_in_range, download_and_aggregate_month

    months = get_months_in_range(years)
    total_prices = 0
    months_completed = 0

    # First pass: count total prices (to pre-allocate file)
    # Actually, we'll just append and track count

    # Open file for binary writing
    with open(prices_file, 'wb') as f:
        for i, (year, month) in enumerate(months, 1):
            if verbose:
                print(f"  [{i}/{len(months)}] {year}-{month:02d}...", end=" ", flush=True)

            try:
                result = download_and_aggregate_month(symbol, year, month, aggregate_seconds=1.0)
            except Exception as e:
                if verbose:
                    print(f"error: {e}")
                continue

            if result is None:
                if verbose:
                    print("not available")
                continue

            aggregated, trade_count = result
            months_completed += 1

            # Write each record directly to disk (never accumulate in RAM)
            month_prices = 0
            for ts, price in aggregated:
                ts_int = int(ts.timestamp())
                f.write(struct.pack('<qf', ts_int, price))
                month_prices += 1

            total_prices += month_prices

            if verbose:
                print(f"✓ {trade_count:,} trades → {month_prices:,} prices")

            # Explicitly clear memory after each month
            del aggregated
            del result
            gc.collect()

    # Calculate file size
    file_size = os.path.getsize(prices_file)
    file_size_mb = file_size / (1024 * 1024)

    # Save metadata
    meta = {
        'symbol': symbol,
        'years': years,
        'total_prices': total_prices,
        'months_completed': months_completed,
        'months_expected': len(months),
        'record_dtype': str(RECORD_DTYPE),
        'record_size': RECORD_SIZE,
        'file_size_bytes': file_size,
        'file_size_mb': file_size_mb,
        'created': datetime.now().isoformat(),
    }

    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print()
        print(f"{'='*60}")
        print(f"Download complete!")
        print(f"  Total prices: {total_prices:,}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Months: {months_completed}/{len(months)}")
        print(f"{'='*60}")

    return prices_file, meta


def load_prices_memmap(prices_file: str, meta: Dict) -> np.memmap:
    """
    Load prices file as memory-mapped numpy array.

    The returned array can be sliced and iterated without loading
    the full dataset into RAM. The OS handles paging data in/out
    as needed.

    Args:
        prices_file: Path to the .mmap file
        meta: Metadata dict from ensure_prices_on_disk

    Returns:
        Memory-mapped numpy structured array with 'timestamp' and 'price' fields
    """
    total_prices = meta['total_prices']
    return np.memmap(prices_file, dtype=RECORD_DTYPE, mode='r', shape=(total_prices,))


def memmap_to_iterator(
    mmap_slice: np.ndarray,
    sample_rate: int = 1
) -> Iterator[Tuple[datetime, float]]:
    """
    Convert a memmap slice to a price iterator for backtesting.

    Yields (datetime, price) tuples. With sample_rate > 1, yields
    only every Nth price for faster grid search.

    Args:
        mmap_slice: Slice of the memory-mapped price array
        sample_rate: Use every Nth price (1=all, 10=every 10th, etc.)
                    Higher values = faster but less accurate.
                    Recommended: 10 for grid search, 1 for final validation.

    Yields:
        Tuple of (datetime, float) for each selected price point
    """
    if sample_rate <= 1:
        # No sampling - iterate all prices
        for record in mmap_slice:
            ts = datetime.fromtimestamp(int(record['timestamp']))
            yield (ts, float(record['price']))
    else:
        # Sample every Nth price for speed
        for i in range(0, len(mmap_slice), sample_rate):
            record = mmap_slice[i]
            ts = datetime.fromtimestamp(int(record['timestamp']))
            yield (ts, float(record['price']))


def get_fold_indices(
    total_prices: int,
    fold: Dict
) -> Tuple[int, int, int, int]:
    """
    Calculate array indices for a fold's train and validation data.

    Args:
        total_prices: Total number of prices in the dataset
        fold: Fold config with train_start_pct, train_end_pct, etc.

    Returns:
        Tuple of (train_start, train_end, val_start, val_end) indices
    """
    train_start = int(total_prices * fold['train_start_pct'])
    train_end = int(total_prices * fold['train_end_pct'])
    val_start = int(total_prices * fold['val_start_pct'])
    val_end = int(total_prices * fold['val_end_pct'])

    return train_start, train_end, val_start, val_end


def get_holdout_indices(total_prices: int, holdout: Dict) -> Tuple[int, int]:
    """
    Calculate array indices for holdout data.

    Args:
        total_prices: Total number of prices in the dataset
        holdout: Holdout config with start_pct and end_pct

    Returns:
        Tuple of (start, end) indices
    """
    start = int(total_prices * holdout['start_pct'])
    end = int(total_prices * holdout['end_pct'])
    return start, end


def slice_memmap_for_fold(
    prices: np.memmap,
    fold: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice memory-mapped prices for a fold's train and validation data.

    Returns views into the memmap, not copies. Memory usage stays low.

    Args:
        prices: Memory-mapped price array
        fold: Fold config dict

    Returns:
        Tuple of (train_slice, val_slice) - both are views, not copies
    """
    total = len(prices)
    train_start, train_end, val_start, val_end = get_fold_indices(total, fold)

    train_slice = prices[train_start:train_end]
    val_slice = prices[val_start:val_end]

    return train_slice, val_slice


def slice_memmap_for_holdout(prices: np.memmap, holdout: Dict) -> np.ndarray:
    """
    Slice memory-mapped prices for holdout testing.

    Returns a view into the memmap, not a copy.

    Args:
        prices: Memory-mapped price array
        holdout: Holdout config dict

    Returns:
        Holdout slice (view, not copy)
    """
    total = len(prices)
    start, end = get_holdout_indices(total, holdout)
    return prices[start:end]


def get_price_range(mmap_slice: np.ndarray) -> Tuple[datetime, datetime, float, float]:
    """
    Get the time and price range of a memmap slice.

    Only reads first and last records, plus scans prices.

    Args:
        mmap_slice: Slice of memory-mapped price array

    Returns:
        Tuple of (start_time, end_time, min_price, max_price)
    """
    if len(mmap_slice) == 0:
        return None, None, None, None

    start_time = datetime.fromtimestamp(int(mmap_slice[0]['timestamp']))
    end_time = datetime.fromtimestamp(int(mmap_slice[-1]['timestamp']))

    # For min/max, we need to scan (but memmap handles paging)
    prices = mmap_slice['price']
    min_price = float(np.min(prices))
    max_price = float(np.max(prices))

    return start_time, end_time, min_price, max_price


def print_memmap_info(prices: np.memmap, meta: Dict):
    """Print information about the memory-mapped price data."""
    print(f"\n{'='*60}")
    print("MEMORY-MAPPED PRICE DATA INFO")
    print(f"{'='*60}")
    print(f"  Symbol: {meta['symbol']}")
    print(f"  Period: {meta['years']} years")
    print(f"  Total prices: {meta['total_prices']:,}")
    print(f"  File size: {meta['file_size_mb']:.1f} MB")
    print(f"  Months: {meta['months_completed']}/{meta['months_expected']}")

    # Get time range (only reads first/last records)
    start_time = datetime.fromtimestamp(int(prices[0]['timestamp']))
    end_time = datetime.fromtimestamp(int(prices[-1]['timestamp']))
    print(f"  Time range: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")


# =============================================================================
# FUNDING RATES (small enough to stay in RAM)
# =============================================================================

def load_funding_rates(symbol: str, years: int = 5) -> Dict[datetime, float]:
    """
    Load funding rates for a symbol.

    Funding rates are small enough to keep in RAM (~10-50 KB for 5 years).

    Args:
        symbol: Trading pair
        years: Years of history

    Returns:
        Dict mapping datetime to funding rate percentage
    """
    try:
        from data.funding_rate_fetcher import get_funding_rates
        payments, funding_rates = get_funding_rates(symbol, years=years)
        if funding_rates:
            print(f"  Loaded {len(funding_rates)} funding rate entries")
        return funding_rates
    except Exception as e:
        print(f"  Could not load funding rates: {e}")
        return {}


# =============================================================================
# DATA QUALITY VALIDATION
# =============================================================================

def validate_memmap_quality(
    prices: np.memmap,
    sample_size: int = 100000,
    verbose: bool = True
) -> Optional[float]:
    """
    Validate data quality by sampling the memmap.

    Only loads a sample into RAM for validation.

    NOTE: This may report low quality scores for large datasets because
    the sampling creates apparent gaps between records. The actual data
    is still valid - this is just a limitation of sampling.

    Args:
        prices: Memory-mapped price array
        sample_size: Number of records to sample
        verbose: Print results

    Returns:
        Quality score (0-100) or None if validation fails
    """
    try:
        from data.tick_data_fetcher import validate_data_quality

        # Sample evenly across the dataset
        total = len(prices)
        if total <= sample_size:
            indices = range(total)
        else:
            step = total // sample_size
            indices = range(0, total, step)[:sample_size]

        # Convert sample to list of tuples (required by validate_data_quality)
        sample = []
        for i in indices:
            ts = datetime.fromtimestamp(int(prices[i]['timestamp']))
            price = float(prices[i]['price'])
            sample.append((ts, price))

        quality_report = validate_data_quality(sample, verbose=False)

        if quality_report and verbose:
            print(f"  Data quality: {quality_report.quality_score:.1f}%")
            if quality_report.quality_score < 80:
                print("  WARNING: Data quality below 80%")

        return quality_report.quality_score if quality_report else None

    except Exception as e:
        if verbose:
            print(f"  Data quality check failed: {e}")
        return None


if __name__ == "__main__":
    # Test the module
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    years = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"Testing memmap_prices with {symbol}, {years} year(s)")

    # Ensure data is on disk
    prices_file, meta = ensure_prices_on_disk(symbol, years)

    # Load as memmap
    prices = load_prices_memmap(prices_file, meta)
    print_memmap_info(prices, meta)

    # Test slicing
    print("\nTesting fold slicing...")
    from balanced_grid import get_fixed_folds
    folds, holdout = get_fixed_folds()

    for fold in folds:
        train, val = slice_memmap_for_fold(prices, fold)
        print(f"  {fold['name']}: train={len(train):,}, val={len(val):,}")

    holdout_slice = slice_memmap_for_holdout(prices, holdout)
    print(f"  Holdout: {len(holdout_slice):,}")

    # Test iteration
    print("\nTesting iteration (first 5 prices)...")
    for i, (ts, price) in enumerate(memmap_to_iterator(prices[:5])):
        print(f"  {i+1}. {ts} -> ${price:.2f}")

    print("\nDone!")
