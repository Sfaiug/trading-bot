#!/usr/bin/env python3
"""
Automated Crypto Trading Bot
=============================

Hedge Trailing Strategy:
- Opens simultaneous LONG and SHORT positions
- Tracks peak (for long) and trough (for short) prices
- Closes each position when price retraces X% from extreme

Usage:
    # Single threshold
    python main.py --threshold 1         # 1% trailing stop
    python main.py --threshold 0.1       # 0.1% trailing stop (fast)
    
    # Multiple thresholds (run simultaneously!)
    python main.py --thresholds 0.1,1,5  # Run all three at once
    python main.py --thresholds 0.5,1,2,5 --rounds 10
"""

import argparse
import sys

from core.exchange import BinanceExchange
from strategies.hedge_trailing import HedgeTrailingStrategy
from strategies.multi_threshold import MultiThresholdStrategy
from config.settings import DEFAULT_TRAILING_STOP_PERCENT, POSITION_SIZE, SYMBOL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hedge Trailing Strategy Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single threshold mode
    python main.py --threshold 1         # 1% trailing stop
    python main.py --threshold 0.5       # 0.5% trailing stop
    
    # Multi-threshold mode (recommended for testing!)
    python main.py --thresholds 0.1,1,5  # Test 3 thresholds at once
    python main.py --thresholds 0.5,1,2  # Custom thresholds
        """
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help=f"Single trailing stop percentage (default: {DEFAULT_TRAILING_STOP_PERCENT}%%)"
    )
    
    parser.add_argument(
        "-T", "--thresholds",
        type=str,
        default=None,
        help="Multiple trailing stop percentages, comma-separated (e.g., '0.1,1,5')"
    )
    
    parser.add_argument(
        "-s", "--size",
        type=float,
        default=POSITION_SIZE,
        help=f"Position size per threshold in SOL (default: {POSITION_SIZE})"
    )
    
    parser.add_argument(
        "-r", "--rounds",
        type=int,
        default=None,
        help="Maximum rounds to run (default: unlimited)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=SYMBOL,
        help=f"Trading pair (default: {SYMBOL})"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine mode: multi-threshold or single threshold
    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
        multi_mode = True
    elif args.threshold:
        thresholds = [args.threshold]
        multi_mode = False
    else:
        thresholds = [DEFAULT_TRAILING_STOP_PERCENT]
        multi_mode = False
    
    print("=" * 70)
    print("AUTOMATED CRYPTO TRADING BOT")
    print("Binance Futures Testnet")
    print("=" * 70)
    print()
    
    # Initialize exchange connection
    print("Connecting to Binance Futures Testnet...")
    exchange = BinanceExchange()
    
    if not exchange.connect():
        print("Failed to connect to exchange. Check API keys in .env file.")
        sys.exit(1)
    
    # Enable hedge mode for simultaneous long/short
    if not exchange.set_hedge_mode():
        print("Warning: Could not enable hedge mode. Strategy may not work correctly.")
    
    # Show account balance
    balance = exchange.get_balance()
    print(f"\nAccount Balance:")
    print(f"  Wallet:    ${balance['wallet_balance']:.2f} USDT")
    print(f"  Available: ${balance['available_balance']:.2f} USDT")
    
    # Show current price
    price = exchange.get_price(args.symbol)
    print(f"\nCurrent {args.symbol} Price: ${price:.4f}")
    
    # Calculate required margin
    total_positions = len(thresholds) * 2  # LONG + SHORT per threshold
    estimated_margin = price * args.size * len(thresholds) * 2 * 0.1  # ~10x leverage assumption
    
    # Confirm settings
    print(f"\nStrategy Settings:")
    print(f"  Mode:       {'MULTI-THRESHOLD' if multi_mode else 'SINGLE THRESHOLD'}")
    print(f"  Symbol:     {args.symbol}")
    print(f"  Size:       {args.size} per threshold")
    print(f"  Thresholds: {', '.join(f'{t}%' for t in thresholds)}")
    print(f"  Positions:  {total_positions} total ({len(thresholds)} LONG + {len(thresholds)} SHORT)")
    print(f"  Est. Margin: ~${estimated_margin:.2f} USDT")
    print(f"  Rounds:     {args.rounds or 'Unlimited'}")
    print()
    
    if estimated_margin > balance['available_balance']:
        print(f"⚠️  Warning: Estimated margin (${estimated_margin:.2f}) exceeds available balance!")
        print(f"    Consider reducing position size or number of thresholds.")
        print()
    
    input("Press Enter to start trading (Ctrl+C to cancel)...")
    
    # Run appropriate strategy
    if multi_mode or len(thresholds) > 1:
        strategy = MultiThresholdStrategy(
            exchange=exchange,
            thresholds=thresholds,
            position_size=args.size,
            symbol=args.symbol,
            max_rounds=args.rounds
        )
    else:
        strategy = HedgeTrailingStrategy(
            exchange=exchange,
            threshold_percent=thresholds[0],
            position_size=args.size,
            symbol=args.symbol,
            max_rounds=args.rounds
        )
    
    strategy.run()


if __name__ == "__main__":
    main()
