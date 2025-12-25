#!/usr/bin/env python3
"""
Final Review Checklist for Live Trading Deployment

This script provides an interactive checklist to ensure all validation
stages have passed before deploying to live trading.

Usage:
    python final_review_checklist.py SYMBOL
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Checklist items with their validation criteria
CHECKLIST_ITEMS = [
    # Validation stages
    ("All 7 validation stages pass", "validation", lambda r: r.get('overall_pass', False)),
    ("Monte Carlo P(positive) >= 80%", "monte_carlo", lambda r: r.get('probability_positive', 0) >= 0.80),
    ("Monte Carlo P(ruin) <= 5%", "monte_carlo", lambda r: r.get('probability_ruin', 1) <= 0.05),
    ("Monte Carlo Sharpe 5th pct >= 0.5", "monte_carlo", lambda r: r.get('sharpe_5th_percentile', 0) >= 0.5),
    ("Top 3 parameters all pass", "top_3", lambda r: r.get('top_3_all_pass', False)),

    # Paper trading
    ("Paper trading completed 30+ rounds", "paper_trading", lambda r: r.get('num_rounds', 0) >= 30),
    ("Paper trading P&L within MC 90% CI", "paper_trading", lambda r: r.get('within_mc_confidence', False)),
    ("Paper trading slippage < 2x expected", "paper_trading", lambda r: r.get('slippage_ratio', 99) < 2.0),
    ("Zero liquidation events", "paper_trading", lambda r: r.get('liquidation_events', 1) == 0),

    # Manual checks (require human confirmation)
    ("Parameter values are reasonable", "manual", None),
    ("Risk limits configured (-40% safety halt)", "manual", None),
    ("Starting with 25% of validated size", "manual", None),
    ("Testnet API keys configured correctly", "manual", None),
    ("Emergency stop procedure understood", "manual", None),
]


def load_validation_result(symbol: str) -> Optional[Dict]:
    """Load validation results for a symbol."""
    filepath = f"logs/{symbol}_validation_result.json"
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def check_item(item: Tuple, results: Dict) -> Tuple[bool, str]:
    """
    Check if a checklist item passes.

    Returns:
        Tuple of (passed: bool, reason: str)
    """
    description, category, validator = item

    if validator is None:
        # Manual check - requires user confirmation
        return None, "Requires manual confirmation"

    if category == "validation":
        passed = validator(results)
    elif category == "monte_carlo":
        mc_data = results.get('stage_results', {}).get('monte_carlo', {}).get('details', {}).get('mc_result', {})
        passed = validator(mc_data)
    elif category == "top_3":
        top3_data = results.get('stage_results', {}).get('top_3', {}).get('details', {})
        passed = validator(top3_data)
    elif category == "paper_trading":
        pt_data = results.get('stage_results', {}).get('paper_trading', {}).get('details', {})
        passed = validator(pt_data)
    else:
        passed = False

    return passed, "PASS" if passed else "FAIL"


def print_checklist(symbol: str, results: Optional[Dict]) -> Tuple[int, int, int]:
    """
    Print the final review checklist.

    Returns:
        Tuple of (passed_count, failed_count, manual_count)
    """
    print("\n" + "=" * 70)
    print(f"FINAL REVIEW CHECKLIST - {symbol}")
    print("=" * 70)
    print(f"Review Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    passed_count = 0
    failed_count = 0
    manual_count = 0

    print("\n### AUTOMATED CHECKS ###\n")

    for item in CHECKLIST_ITEMS:
        description, category, validator = item

        if validator is None:
            continue  # Skip manual items for now

        if results:
            passed, reason = check_item(item, results)
        else:
            passed = False
            reason = "No validation results"

        if passed:
            status = "[X] PASS"
            passed_count += 1
        else:
            status = "[ ] FAIL"
            failed_count += 1

        print(f"  {status}: {description}")

    print("\n### MANUAL CHECKS (Confirm Each) ###\n")

    for item in CHECKLIST_ITEMS:
        description, category, validator = item

        if validator is not None:
            continue  # Skip automated items

        print(f"  [ ] {description}")
        manual_count += 1

    return passed_count, failed_count, manual_count


def interactive_review(symbol: str) -> bool:
    """
    Run interactive review process.

    Returns:
        True if all checks pass, False otherwise
    """
    # Load results
    results = load_validation_result(symbol)

    if not results:
        print(f"\n[WARNING] No validation results found for {symbol}")
        print(f"Run: python validation_pipeline.py {symbol}")
        print("\nProceeding with manual-only checklist...\n")

    # Print checklist
    passed, failed, manual = print_checklist(symbol, results)

    print("\n" + "-" * 70)
    print(f"AUTOMATED: {passed} passed, {failed} failed")
    print(f"MANUAL: {manual} items require confirmation")
    print("-" * 70)

    if failed > 0:
        print("\n[BLOCKED] Cannot proceed - automated checks failed")
        print("Fix the issues above before continuing.")
        return False

    # Confirm manual items
    print("\n### MANUAL CONFIRMATION ###")
    print("Please confirm each item below (y/n):\n")

    all_confirmed = True
    for item in CHECKLIST_ITEMS:
        description, category, validator = item

        if validator is not None:
            continue

        while True:
            response = input(f"  {description}? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print(f"    [X] Confirmed")
                break
            elif response in ['n', 'no']:
                print(f"    [ ] NOT confirmed")
                all_confirmed = False
                break
            else:
                print("    Please enter 'y' or 'n'")

    print("\n" + "=" * 70)

    if all_confirmed and failed == 0:
        print("RESULT: ALL CHECKS PASSED")
        print("=" * 70)
        print("""
APPROVED FOR LIVE TRADING

Next steps:
1. Configure live API keys (not testnet)
2. Set position size to 25% of validated amount
3. Enable -40% equity kill switch
4. Start trading and monitor closely
5. Scale up gradually over 4 weeks if profitable

IMPORTANT:
- Monitor performance vs paper trading expectations
- Be prepared to stop if deviation exceeds 2 standard deviations
- Review weekly and document any discrepancies
""")
        return True
    else:
        print("RESULT: NOT APPROVED")
        print("=" * 70)
        if failed > 0:
            print(f"\n{failed} automated check(s) failed")
        if not all_confirmed:
            print("\nNot all manual items were confirmed")
        print("\nDo NOT proceed to live trading until all checks pass.")
        return False


def generate_approval_record(symbol: str, approved: bool) -> None:
    """Generate a JSON record of the approval decision."""
    record = {
        'symbol': symbol,
        'review_timestamp': datetime.now().isoformat(),
        'approved': approved,
        'reviewer': os.environ.get('USER', 'unknown'),
    }

    filepath = f"logs/{symbol}_APPROVAL_RECORD.json"
    with open(filepath, 'w') as f:
        json.dump(record, f, indent=2)

    print(f"\nApproval record saved to: {filepath}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python final_review_checklist.py SYMBOL")
        print("\nExample:")
        print("  python final_review_checklist.py BTCUSDT")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    print("\n" + "=" * 70)
    print("FINAL REVIEW BEFORE LIVE TRADING")
    print("=" * 70)
    print(f"\nSymbol: {symbol}")
    print("\nThis checklist ensures all validation stages have passed")
    print("and risk management is properly configured before going live.\n")

    proceed = input("Ready to begin review? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("Review cancelled.")
        sys.exit(0)

    approved = interactive_review(symbol)
    generate_approval_record(symbol, approved)

    sys.exit(0 if approved else 1)


if __name__ == "__main__":
    main()
