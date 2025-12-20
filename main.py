#!/usr/bin/env python3
"""
Automated Crypto Trading Bot
=============================

Two Modes:
1. PYRAMID TRADING - Hedge + Pyramid + Trailing Stop
2. FUNDING RATE - Delta-neutral funding collection

Usage:
    # Interactive mode (prompts for everything)
    python main.py
    
    # Direct mode - Pyramid Trading
    python main.py --mode trading --symbol SOLUSDT \
        --threshold 10 --trailing 1.5 --pyramid 2 --size 1
    
    # Direct mode - Funding Rate
    python main.py --mode funding --symbol SOLUSDT \
        --entry-funding 0.1 --exit-funding 0.02 --size 1
"""

import argparse
import sys

from core.exchange import BinanceExchange
from strategies.pyramid_trading import PyramidTradingStrategy
from strategies.funding_rate import FundingRateStrategy
from config.settings import POSITION_SIZE, SYMBOL


# Available coins
COINS = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "XLMUSDT", "XRPUSDT", "DOGEUSDT"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["trading", "funding"],
        default=None,
        help="Trading mode: 'trading' (pyramid) or 'funding' (funding rate)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading pair (e.g., SOLUSDT)"
    )
    
    parser.add_argument(
        "-s", "--size",
        type=float,
        default=POSITION_SIZE,
        help=f"Position size (default: {POSITION_SIZE})"
    )
    
    parser.add_argument(
        "-r", "--rounds",
        type=int,
        default=None,
        help="Maximum rounds (trading mode only)"
    )
    
    # Pyramid trading args
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold %% - when losing side closes (trading mode)"
    )
    
    parser.add_argument(
        "--trailing",
        type=float,
        default=None,
        help="Trailing %% - when winning side closes (trading mode)"
    )
    
    parser.add_argument(
        "--pyramid",
        type=float,
        default=None,
        help="Pyramid step %% - interval to add positions (trading mode)"
    )
    
    # Funding rate args
    parser.add_argument(
        "--entry-funding",
        type=float,
        default=None,
        help="Entry threshold %% for funding rate (funding mode)"
    )
    
    parser.add_argument(
        "--exit-funding",
        type=float,
        default=None,
        help="Exit threshold %% for funding rate (funding mode)"
    )
    
    parser.add_argument(
        "--leverage",
        type=int,
        default=None,
        help="Leverage to use (1-20, default: 5)"
    )
    
    return parser.parse_args()


def print_banner():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " AUTOMATED CRYPTO TRADING BOT ".center(58) + "║")
    print("║" + " Binance Futures Testnet ".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print("║" + " Select Mode: ".ljust(58) + "║")
    print("║" + "   1. Pyramid Trading (hedge + pyramid + trailing) ".ljust(58) + "║")
    print("║" + "   2. Funding Rate Farming (collect funding payments) ".ljust(58) + "║")
    print("╚" + "═" * 58 + "╝")


def select_mode() -> str:
    while True:
        choice = input("\nEnter choice [1/2]: ").strip()
        if choice == "1":
            return "trading"
        elif choice == "2":
            return "funding"
        print("Invalid choice. Enter 1 or 2.")


def select_coin() -> str:
    print("\nSelect coin:")
    for i, coin in enumerate(COINS, 1):
        print(f"  {i}. {coin}")
    print(f"  {len(COINS) + 1}. Custom")
    
    while True:
        choice = input(f"\nEnter choice [1-{len(COINS) + 1}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(COINS):
                return COINS[idx]
            elif idx == len(COINS):
                return input("Enter custom symbol (e.g., LINKUSDT): ").strip().upper()
        except ValueError:
            pass
        print("Invalid choice.")


def get_float_input(prompt: str, default: float) -> float:
    while True:
        value = input(f"{prompt} [default: {default}]: ").strip()
        if value == "":
            return default
        try:
            return float(value)
        except ValueError:
            print("Invalid number. Try again.")


def get_int_input(prompt: str, default: str = "unlimited") -> int:
    while True:
        value = input(f"{prompt} [default: {default}]: ").strip()
        if value == "":
            return None if default == "unlimited" else int(default)
        try:
            return int(value)
        except ValueError:
            print("Invalid number. Try again.")


def get_leverage_input(args) -> int:
    """Get leverage setting from args or prompt."""
    if args.leverage:
        return min(max(args.leverage, 1), 20)  # Clamp 1-20
    
    print("\nLeverage Settings:")
    print("  Recommended: 3-5x for pyramid strategy")
    print("  Higher = more capital efficient but riskier")
    
    while True:
        value = input("Leverage (1-20) [default: 5]: ").strip()
        if value == "":
            return 5
        try:
            lev = int(value)
            if 1 <= lev <= 20:
                return lev
            print("Leverage must be between 1 and 20.")
        except ValueError:
            print("Invalid number. Try again.")


def configure_trading_mode(args) -> dict:
    """Configure pyramid trading parameters."""
    print("\n" + "=" * 50)
    print("PYRAMID TRADING CONFIGURATION")
    print("=" * 50)
    
    threshold = args.threshold if args.threshold else get_float_input("Threshold % (losing side closes)", 1.0)
    trailing = args.trailing if args.trailing else get_float_input("Trailing % (profit lock)", 1.0)
    pyramid = args.pyramid if args.pyramid else get_float_input("Pyramid step %", 2.0)
    size = args.size if args.size != POSITION_SIZE else get_float_input("Position size", POSITION_SIZE)
    rounds = args.rounds if args.rounds else get_int_input("Max rounds", "unlimited")
    
    return {
        'threshold': threshold,
        'trailing': trailing,
        'pyramid': pyramid,
        'size': size,
        'rounds': rounds
    }


def configure_funding_mode(args) -> dict:
    """Configure funding rate parameters."""
    print("\n" + "=" * 50)
    print("FUNDING RATE CONFIGURATION")
    print("=" * 50)
    
    entry = args.entry_funding if args.entry_funding else get_float_input("Entry threshold % (min funding to enter)", 0.1)
    exit_th = args.exit_funding if args.exit_funding else get_float_input("Exit threshold % (exit when funding below)", 0.02)
    size = args.size if args.size != POSITION_SIZE else get_float_input("Position size", POSITION_SIZE)
    
    return {
        'entry_threshold': entry,
        'exit_threshold': exit_th,
        'size': size
    }


def confirm_settings(mode: str, symbol: str, config: dict, leverage: int) -> bool:
    """Display settings and confirm."""
    print("\n" + "=" * 50)
    print("CONFIRM SETTINGS")
    print("=" * 50)
    print(f"Mode:     {mode.upper()}")
    print(f"Symbol:   {symbol}")
    print(f"Leverage: {leverage}x")
    
    if mode == "trading":
        print(f"Threshold:    {config['threshold']}%")
        print(f"Trailing:     {config['trailing']}%")
        print(f"Pyramid Step: {config['pyramid']}%")
        print(f"Size:         {config['size']}")
        print(f"Max Rounds:   {config['rounds'] or 'Unlimited'}")
    else:
        print(f"Entry Threshold: >{config['entry_threshold']}%")
        print(f"Exit Threshold:  <{config['exit_threshold']}%")
        print(f"Size:            {config['size']}")
    
    print("=" * 50)
    
    confirm = input("\nStart trading? [y/n]: ").strip().lower()
    return confirm in ['y', 'yes']


def main():
    args = parse_args()
    
    # Interactive mode selection if not provided
    if args.mode is None:
        print_banner()
        mode = select_mode()
    else:
        mode = args.mode
    
    # Symbol selection if not provided
    if args.symbol is None:
        symbol = select_coin()
    else:
        symbol = args.symbol
    
    # Mode-specific configuration
    if mode == "trading":
        config = configure_trading_mode(args)
    else:
        config = configure_funding_mode(args)
    
    # Get leverage
    leverage = get_leverage_input(args)
    
    # Connect to exchange
    print("\nConnecting to Binance Futures Testnet...")
    exchange = BinanceExchange()
    
    if not exchange.connect():
        print("Failed to connect. Check API keys in .env file.")
        sys.exit(1)
    
    # Enable hedge mode
    if not exchange.set_hedge_mode():
        print("Warning: Could not enable hedge mode.")
    
    # Set leverage
    exchange.set_leverage(symbol, leverage)
    
    # Show balance
    balance = exchange.get_balance()
    print(f"\nAccount Balance:")
    print(f"  Wallet:    ${balance['wallet_balance']:.2f} USDT")
    print(f"  Available: ${balance['available_balance']:.2f} USDT")
    
    # Show current price
    price = exchange.get_price(symbol)
    print(f"\nCurrent {symbol} Price: ${price:.4f}")
    
    # Confirm settings
    if not confirm_settings(mode, symbol, config, leverage):
        print("\nCancelled.")
        sys.exit(0)
    
    # Run strategy
    if mode == "trading":
        strategy = PyramidTradingStrategy(
            exchange=exchange,
            symbol=symbol,
            threshold_pct=config['threshold'],
            trailing_pct=config['trailing'],
            pyramid_step_pct=config['pyramid'],
            position_size=config['size'],
            max_rounds=config['rounds']
        )
    else:
        strategy = FundingRateStrategy(
            exchange=exchange,
            symbol=symbol,
            entry_threshold=config['entry_threshold'],
            exit_threshold=config['exit_threshold'],
            position_size=config['size']
        )
    
    strategy.run()


if __name__ == "__main__":
    main()
