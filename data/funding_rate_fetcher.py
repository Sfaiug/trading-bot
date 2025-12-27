#!/usr/bin/env python3
"""
Funding Rate Fetcher Module

Downloads historical funding rate data from Binance for perpetual futures.
Funding is applied every 8 hours (00:00, 08:00, 16:00 UTC).

Funding Rate Impact:
- Positive rate: LONG pays SHORT (bearish sentiment)
- Negative rate: SHORT pays LONG (bullish sentiment)
- Typical range: -0.1% to +0.3% per 8 hours
- Annual impact: 5-20% drag on positions held over time
"""

import os
import requests
import zipfile
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class FundingPayment:
    """Represents a single funding payment event."""
    timestamp: datetime
    funding_rate: float  # Rate as percentage (e.g., 0.01 = 0.01%)
    mark_price: float  # Mark price at funding time


def fetch_funding_rate_history_api(
    symbol: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> List[FundingPayment]:
    """
    Fetch funding rate history from Binance Futures API.

    Note: API limits to 1000 records per request.
    For longer periods, use fetch_funding_rates_bulk().

    Args:
        symbol: Trading pair (e.g., 'SOLUSDT')
        start_time: Start of period (default: 30 days ago)
        end_time: End of period (default: now)
        limit: Max records to fetch (max 1000)

    Returns:
        List of FundingPayment objects
    """
    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"

    params = {
        "symbol": symbol,
        "limit": min(limit, 1000)
    }

    if start_time:
        params["startTime"] = int(start_time.timestamp() * 1000)
    if end_time:
        params["endTime"] = int(end_time.timestamp() * 1000)

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        payments = []
        for item in data:
            # Handle empty or invalid funding rates
            funding_rate_str = item.get("fundingRate", "0")
            if funding_rate_str == "" or funding_rate_str is None:
                funding_rate_str = "0"

            mark_price_str = item.get("markPrice", "0")
            if mark_price_str == "" or mark_price_str is None:
                mark_price_str = "0"

            payment = FundingPayment(
                timestamp=datetime.fromtimestamp(item["fundingTime"] / 1000),
                funding_rate=float(funding_rate_str) * 100,  # Convert to percentage
                mark_price=float(mark_price_str)
            )
            payments.append(payment)

        return sorted(payments, key=lambda x: x.timestamp)

    except Exception as e:
        print(f"Error fetching funding rates for {symbol}: {e}")
        return []


def fetch_funding_rates_bulk(
    symbol: str,
    years: int = 5
) -> List[FundingPayment]:
    """
    Fetch funding rate history for multiple years.

    Makes multiple API calls to get complete history.

    Args:
        symbol: Trading pair (e.g., 'SOLUSDT')
        years: Number of years of history to fetch

    Returns:
        List of FundingPayment objects covering the requested period
    """
    all_payments = []
    end_time = datetime.now()
    start_time = end_time - timedelta(days=years * 365)

    # Binance returns max 1000 records = ~333 days (3 per day)
    current_end = end_time

    while current_end > start_time:
        payments = fetch_funding_rate_history_api(
            symbol=symbol,
            start_time=start_time,
            end_time=current_end,
            limit=1000
        )

        if not payments:
            break

        all_payments.extend(payments)

        # Move window back
        oldest = min(p.timestamp for p in payments)
        if oldest <= start_time:
            break
        current_end = oldest - timedelta(hours=1)

    # Remove duplicates and sort
    seen = set()
    unique_payments = []
    for p in all_payments:
        key = (p.timestamp, p.funding_rate)
        if key not in seen:
            seen.add(key)
            unique_payments.append(p)

    return sorted(unique_payments, key=lambda x: x.timestamp)


def create_funding_rate_lookup(
    payments: List[FundingPayment]
) -> Dict[datetime, float]:
    """
    Create a lookup dictionary for funding rates by funding time.

    Funding times are normalized to 00:00, 08:00, or 16:00 UTC.

    Args:
        payments: List of FundingPayment objects

    Returns:
        Dict mapping funding timestamp to rate percentage
    """
    lookup = {}
    for p in payments:
        # Normalize to funding hour
        normalized = p.timestamp.replace(minute=0, second=0, microsecond=0)
        lookup[normalized] = p.funding_rate
    return lookup


def get_funding_rate_at_time(
    lookup: Dict[datetime, float],
    timestamp: datetime
) -> Optional[float]:
    """
    Get the funding rate that would be applied at a given time.

    Finds the most recent funding time before or at the timestamp.

    Args:
        lookup: Funding rate lookup dictionary
        timestamp: Time to check

    Returns:
        Funding rate percentage, or None if not found
    """
    # Find nearest funding time (00:00, 08:00, or 16:00)
    hour = timestamp.hour
    if hour < 8:
        funding_hour = 0
    elif hour < 16:
        funding_hour = 8
    else:
        funding_hour = 16

    funding_time = timestamp.replace(hour=funding_hour, minute=0, second=0, microsecond=0)
    return lookup.get(funding_time)


def is_funding_time(timestamp: datetime) -> bool:
    """
    Check if the timestamp is at a funding time (00:00, 08:00, or 16:00 UTC).

    Args:
        timestamp: Time to check

    Returns:
        True if this is a funding time
    """
    return timestamp.hour in (0, 8, 16) and timestamp.minute == 0


def calculate_funding_payment(
    position_notional: float,
    funding_rate_pct: float,
    is_long: bool
) -> float:
    """
    Calculate funding payment for a position.

    Args:
        position_notional: Position value in USDT
        funding_rate_pct: Funding rate as percentage (e.g., 0.01 = 0.01%)
        is_long: True for long position, False for short

    Returns:
        Funding payment in USDT (positive = you pay, negative = you receive)
    """
    # Payment = Notional * Funding Rate
    payment = position_notional * (funding_rate_pct / 100)

    # Long pays when rate > 0, receives when rate < 0
    # Short receives when rate > 0, pays when rate < 0
    if is_long:
        return payment  # Long pays positive rate
    else:
        return -payment  # Short receives positive rate


def estimate_annual_funding_cost(
    payments: List[FundingPayment],
    is_long: bool = True
) -> Tuple[float, float, float]:
    """
    Estimate annual funding cost/benefit for a position.

    Args:
        payments: List of funding payments
        is_long: Position direction

    Returns:
        Tuple of (annual_cost_pct, avg_rate_pct, total_periods)
    """
    if not payments:
        return 0.0, 0.0, 0

    rates = [p.funding_rate for p in payments]
    avg_rate = sum(rates) / len(rates)

    # 3 funding periods per day * 365 days = 1095 per year
    annual_periods = 1095

    # For long: positive rate = cost, negative rate = benefit
    # For short: opposite
    if is_long:
        annual_cost_pct = avg_rate * annual_periods
    else:
        annual_cost_pct = -avg_rate * annual_periods

    return annual_cost_pct, avg_rate, len(rates)


def save_funding_rates_cache(
    payments: List[FundingPayment],
    symbol: str,
    cache_dir: str = "./cache/funding"
):
    """Save funding rates to cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_funding.json")

    data = [
        {
            "timestamp": p.timestamp.isoformat(),
            "funding_rate": p.funding_rate,
            "mark_price": p.mark_price
        }
        for p in payments
    ]

    with open(cache_file, 'w') as f:
        json.dump(data, f)

    print(f"Saved {len(payments)} funding rates to {cache_file}")


def load_funding_rates_cache(
    symbol: str,
    cache_dir: str = "./cache/funding"
) -> Optional[List[FundingPayment]]:
    """Load funding rates from cache file."""
    cache_file = os.path.join(cache_dir, f"{symbol}_funding.json")

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)

        payments = [
            FundingPayment(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                funding_rate=item["funding_rate"],
                mark_price=item["mark_price"]
            )
            for item in data
        ]
        return payments

    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def get_funding_rates(
    symbol: str,
    years: int = 5,
    use_cache: bool = True,
    cache_dir: str = "./cache/funding"
) -> Tuple[List[FundingPayment], Dict[datetime, float]]:
    """
    Get funding rate history with caching.

    Args:
        symbol: Trading pair (e.g., 'SOLUSDT')
        years: Years of history to fetch
        use_cache: Whether to use/save cache
        cache_dir: Cache directory

    Returns:
        Tuple of (payment list, lookup dictionary)
    """
    payments = None

    # Try cache first
    if use_cache:
        payments = load_funding_rates_cache(symbol, cache_dir)
        if payments:
            print(f"Loaded {len(payments)} funding rates from cache for {symbol}")

    # Fetch from API if needed
    if payments is None:
        print(f"Fetching funding rate history for {symbol}...")
        payments = fetch_funding_rates_bulk(symbol, years)

        if payments and use_cache:
            save_funding_rates_cache(payments, symbol, cache_dir)

    lookup = create_funding_rate_lookup(payments) if payments else {}
    return payments, lookup


# Test function
if __name__ == "__main__":
    print("Testing Funding Rate Fetcher...")
    print("=" * 60)

    # Test with SOLUSDT
    symbol = "SOLUSDT"

    # Fetch recent funding rates (limited to avoid long API calls)
    print(f"\nFetching recent funding rates for {symbol}...")
    payments = fetch_funding_rate_history_api(symbol, limit=100)

    if payments:
        print(f"Fetched {len(payments)} funding rate records")

        # Show sample
        print(f"\nRecent funding rates:")
        for p in payments[-5:]:
            print(f"  {p.timestamp}: {p.funding_rate:+.4f}%")

        # Estimate annual cost
        annual_cost, avg_rate, periods = estimate_annual_funding_cost(payments, is_long=True)
        print(f"\nFunding Rate Analysis ({periods} periods):")
        print(f"  Average rate: {avg_rate:+.4f}%")
        print(f"  Est. annual cost (LONG): {annual_cost:+.2f}%")
        print(f"  Est. annual cost (SHORT): {-annual_cost:+.2f}%")

        # Create lookup
        lookup = create_funding_rate_lookup(payments)
        print(f"\nCreated lookup with {len(lookup)} entries")

        # Test payment calculation
        notional = 1000  # $1000 position
        rate = payments[-1].funding_rate
        long_payment = calculate_funding_payment(notional, rate, is_long=True)
        short_payment = calculate_funding_payment(notional, rate, is_long=False)

        print(f"\nFunding payment for ${notional} position at {rate:+.4f}%:")
        print(f"  LONG pays: ${long_payment:.4f}")
        print(f"  SHORT pays: ${short_payment:.4f}")

    else:
        print("No funding rate data available (API may be rate limited)")

    print(f"\n{'='*60}")
    print("Phase 1.2 (Funding Rate Fetcher) test completed!")
