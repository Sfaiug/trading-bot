#!/usr/bin/env python3
"""
Parameter sweep for single_pos_multiplier.
Tests 15 different multiplier values on the 2% threshold.
"""

import subprocess
import sys
import re

# Multipliers to test (15 values from 0.1 to 5.0)
multipliers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Fixed parameters
THRESHOLD = "2"  # Best performing threshold
DAYS = "90"
INTERVAL = "1m"

print("=" * 70)
print("MULTIPLIER SWEEP - Finding Optimal Single Position Multiplier")
print("=" * 70)
print(f"Testing {len(multipliers)} multipliers on {THRESHOLD}% threshold")
print(f"Period: {DAYS} days, {INTERVAL} candles")
print("=" * 70)
print()

results = []

for i, mult in enumerate(multipliers, 1):
    print(f"[{i}/{len(multipliers)}] Testing multiplier {mult}...", end=" ", flush=True)
    
    cmd = [
        sys.executable, "backtest.py",
        "--thresholds", THRESHOLD,
        "--days", DAYS,
        "--interval", INTERVAL,
        "--multiplier", str(mult)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        
        # Parse total P&L from output
        match = re.search(r'Total P&L:\s+([\+\-]?\d+\.?\d*)%', output)
        if match:
            pnl = float(match.group(1))
            results.append((mult, pnl))
            emoji = "✅" if pnl > 0 else "❌"
            print(f"{emoji} P&L: {pnl:+.2f}%")
        else:
            # Try alternate format
            match = re.search(r'([\+\-]?\d+\.?\d*)% total', output)
            if match:
                pnl = float(match.group(1))
                results.append((mult, pnl))
                emoji = "✅" if pnl > 0 else "❌"
                print(f"{emoji} P&L: {pnl:+.2f}%")
            else:
                print("Failed to parse P&L")
                
    except subprocess.TimeoutExpired:
        print("Timeout!")
    except Exception as e:
        print(f"Error: {e}")

# Sort by P&L and print results
print()
print("=" * 70)
print("RESULTS - RANKED BY TOTAL P&L")
print("=" * 70)
print()
print(f"{'Rank':<6} {'Multiplier':<12} {'Total P&L':<12}")
print("-" * 40)

results.sort(key=lambda x: x[1], reverse=True)
for i, (mult, pnl) in enumerate(results, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{emoji}{i:<4} {mult:<12.2f} {pnl:+10.2f}%")

print("-" * 40)

if results:
    best_mult, best_pnl = results[0]
    print()
    print(f"🏆 BEST MULTIPLIER: {best_mult}")
    print(f"   Total P&L: {best_pnl:+.2f}%")
    print()
    print(f"   This means when only one position is open,")
    print(f"   use {2 * best_mult:.2f}% trailing stop (base 2% × {best_mult})")
