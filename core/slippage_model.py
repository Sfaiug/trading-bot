"""
Position-Size-Dependent Slippage Model

This module provides realistic slippage estimates based on position size.
Larger positions experience more slippage due to order book depth limitations.

Slippage Components:
1. Base slippage: 0.01% (market spread for liquid pairs)
2. Size-dependent: Additional slippage based on position size
3. Volatility adjustment: Higher volatility = more slippage

Formula:
    slippage_pct = base + (position_usdt / reference_size) * size_factor

For $100 USDT position: ~0.01% slippage
For $10,000 USDT position: ~0.03% slippage
For $100,000 USDT position: ~0.05% slippage

These estimates are based on Binance Futures order book depth analysis.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SlippageConfig:
    """Configuration for slippage model."""
    base_slippage_pct: float = 0.01  # 0.01% base slippage (1 bp)
    size_factor: float = 0.02  # Additional slippage per $50k
    reference_size_usdt: float = 50000.0  # Reference size for linear scaling
    max_slippage_pct: float = 0.10  # Cap at 0.10% (10 bps)
    volatility_multiplier: float = 1.5  # Multiply slippage in high vol


# Default configurations for major pairs
# Based on typical Binance Futures order book depth
PAIR_CONFIGS: Dict[str, SlippageConfig] = {
    'BTCUSDT': SlippageConfig(
        base_slippage_pct=0.005,  # Very liquid
        size_factor=0.015,
        reference_size_usdt=100000.0,
    ),
    'ETHUSDT': SlippageConfig(
        base_slippage_pct=0.008,
        size_factor=0.018,
        reference_size_usdt=75000.0,
    ),
    'SOLUSDT': SlippageConfig(
        base_slippage_pct=0.01,
        size_factor=0.02,
        reference_size_usdt=50000.0,
    ),
    'XRPUSDT': SlippageConfig(
        base_slippage_pct=0.01,
        size_factor=0.02,
        reference_size_usdt=50000.0,
    ),
    'DOGEUSDT': SlippageConfig(
        base_slippage_pct=0.012,
        size_factor=0.025,
        reference_size_usdt=40000.0,
    ),
}

# Default config for unknown pairs
DEFAULT_CONFIG = SlippageConfig()


def get_slippage_config(symbol: str) -> SlippageConfig:
    """Get slippage configuration for a symbol."""
    return PAIR_CONFIGS.get(symbol.upper(), DEFAULT_CONFIG)


def calculate_slippage_pct(
    position_size_usdt: float,
    symbol: str = "BTCUSDT",
    is_high_volatility: bool = False,
    config: Optional[SlippageConfig] = None
) -> float:
    """
    Calculate expected slippage percentage for a position.

    Args:
        position_size_usdt: Position size in USDT
        symbol: Trading pair symbol
        is_high_volatility: If True, applies volatility multiplier
        config: Optional custom config (uses pair default if None)

    Returns:
        Expected slippage as percentage (e.g., 0.02 = 0.02%)
    """
    if config is None:
        config = get_slippage_config(symbol)

    # Base slippage
    slippage = config.base_slippage_pct

    # Size-dependent component
    size_component = (position_size_usdt / config.reference_size_usdt) * config.size_factor
    slippage += size_component

    # Volatility adjustment
    if is_high_volatility:
        slippage *= config.volatility_multiplier

    # Cap at maximum
    slippage = min(slippage, config.max_slippage_pct)

    return slippage


def calculate_slippage_cost(
    position_size_usdt: float,
    symbol: str = "BTCUSDT",
    is_high_volatility: bool = False
) -> float:
    """
    Calculate slippage cost in USDT for a position.

    Args:
        position_size_usdt: Position size in USDT
        symbol: Trading pair symbol
        is_high_volatility: If True, applies volatility multiplier

    Returns:
        Expected slippage cost in USDT
    """
    slippage_pct = calculate_slippage_pct(position_size_usdt, symbol, is_high_volatility)
    return position_size_usdt * (slippage_pct / 100.0)


def apply_slippage_to_fill_price(
    price: float,
    is_buy: bool,
    position_size_usdt: float,
    symbol: str = "BTCUSDT",
    is_high_volatility: bool = False
) -> float:
    """
    Apply slippage to a fill price.

    For buys, price increases (we pay more).
    For sells, price decreases (we receive less).

    Args:
        price: Original price
        is_buy: True if buying, False if selling
        position_size_usdt: Position size in USDT
        symbol: Trading pair symbol
        is_high_volatility: If True, applies volatility multiplier

    Returns:
        Adjusted fill price with slippage
    """
    slippage_pct = calculate_slippage_pct(position_size_usdt, symbol, is_high_volatility)

    if is_buy:
        # Buys: price goes up (we pay more)
        return price * (1 + slippage_pct / 100.0)
    else:
        # Sells: price goes down (we receive less)
        return price * (1 - slippage_pct / 100.0)


def estimate_round_slippage(
    num_trades: int,
    avg_position_size_usdt: float,
    symbol: str = "BTCUSDT"
) -> float:
    """
    Estimate total slippage for a trading round.

    Args:
        num_trades: Total number of trades (entries + exits + pyramids)
        avg_position_size_usdt: Average position size per trade
        symbol: Trading pair symbol

    Returns:
        Total estimated slippage cost in USDT
    """
    slippage_pct = calculate_slippage_pct(avg_position_size_usdt, symbol)
    per_trade_cost = avg_position_size_usdt * (slippage_pct / 100.0)
    return num_trades * per_trade_cost


def estimate_round_slippage_pct(
    num_pyramids: int,
    position_size_usdt: float,
    symbol: str = "BTCUSDT"
) -> float:
    """
    Estimate total slippage as percentage of position for a pyramid round.

    A typical pyramid round has:
    - 2 initial hedge entries (long + short)
    - 1 losing side exit
    - N pyramid entries
    - 1 final exit (all positions)

    Total trades = 2 + 1 + N + 1 = 4 + N

    Args:
        num_pyramids: Number of pyramid positions added
        position_size_usdt: Position size per entry
        symbol: Trading pair symbol

    Returns:
        Total slippage as percentage of initial position
    """
    # Calculate trades: initial hedge (2) + losing exit (1) + pyramids (N) + final exit (1)
    num_trades = 4 + num_pyramids

    # Slippage per trade
    per_trade_slippage_pct = calculate_slippage_pct(position_size_usdt, symbol)

    # Total slippage scales with number of trades
    total_slippage_pct = num_trades * per_trade_slippage_pct

    return total_slippage_pct


if __name__ == "__main__":
    # Test the slippage model
    print("=" * 60)
    print("SLIPPAGE MODEL TEST")
    print("=" * 60)

    test_sizes = [100, 1000, 5000, 10000, 50000, 100000]

    for symbol in ['BTCUSDT', 'SOLUSDT', 'DOGEUSDT']:
        print(f"\n{symbol}:")
        for size in test_sizes:
            slippage = calculate_slippage_pct(size, symbol)
            cost = calculate_slippage_cost(size, symbol)
            print(f"  ${size:>7,}: {slippage:.4f}% (${cost:.4f})")

    print("\n" + "=" * 60)
    print("PYRAMID ROUND SLIPPAGE TEST")
    print("=" * 60)

    for pyramids in [0, 5, 10, 20, 50]:
        slippage = estimate_round_slippage_pct(pyramids, 100.0, "BTCUSDT")
        print(f"  {pyramids} pyramids: {slippage:.4f}% total slippage")
