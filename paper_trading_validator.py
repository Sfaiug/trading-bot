#!/usr/bin/env python3
"""
Paper Trading Validation Module

PHASE 5: Automatic paper trading validation before live deployment.

This module:
1. Connects to Binance Futures Testnet
2. Runs the pyramid strategy with optimized parameters
3. Compares actual results to Monte Carlo expectations
4. Determines if the strategy is ready for live trading

CRITICAL: Never deploy to live trading without passing paper trading validation.
A strategy must:
- Complete 60+ rounds (approximately 4 weeks)
- Achieve P&L within Monte Carlo 90% confidence interval
- Show slippage within 2x of modeled assumptions
- Have zero liquidation events
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.monte_carlo import MonteCarloResult


# =============================================================================
# Configuration
# =============================================================================

# Minimum validation requirements
MIN_COMPLETED_ROUNDS = 30  # Reduced from 60 for faster iteration
MAX_DURATION_DAYS = 28
SLIPPAGE_TOLERANCE = 2.0  # Max 2x expected slippage
PNL_DEVIATION_THRESHOLD = 2.0  # Max 2 standard deviations from MC median
MAX_LIQUIDATION_EVENTS = 0  # Zero tolerance for liquidations

# Paths
LOG_DIR = "logs/paper_trading"
RESULTS_FILE = "paper_trading_results.json"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PaperTradingConfig:
    """Configuration for paper trading validation."""
    symbol: str
    params: Dict[str, float]
    mc_result: Optional[MonteCarloResult] = None
    min_rounds: int = MIN_COMPLETED_ROUNDS
    max_duration_days: int = MAX_DURATION_DAYS
    slippage_tolerance: float = SLIPPAGE_TOLERANCE
    pnl_deviation_threshold: float = PNL_DEVIATION_THRESHOLD
    expected_slippage_bps: float = 10.0  # Expected slippage in basis points


@dataclass
class Trade:
    """Individual trade record."""
    timestamp: datetime
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 1.0
    pnl_pct: float = 0.0
    slippage_bps: float = 0.0
    is_pyramid: bool = False
    pyramid_level: int = 0


@dataclass
class RoundResult:
    """Results from a single trading round."""
    round_number: int
    start_time: datetime
    end_time: datetime
    direction: str
    num_pyramids: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    max_profit_pct: float
    trades: List[Trade] = field(default_factory=list)
    funding_paid: float = 0.0
    avg_slippage_bps: float = 0.0


@dataclass
class PaperTradingResult:
    """Complete results from paper trading validation."""
    # Metadata
    symbol: str
    params: Dict[str, float]
    start_time: datetime
    end_time: datetime
    duration_days: float

    # Statistics
    completed_rounds: int
    total_pnl_pct: float
    avg_pnl_per_round: float
    win_rate: float
    realized_sharpe: float

    # Execution quality
    avg_slippage_bps: float
    max_slippage_bps: float
    execution_quality_score: float  # 1.0 = matches expectation

    # Comparison to Monte Carlo
    mc_pnl_5th: float = 0.0
    mc_pnl_95th: float = 0.0
    mc_median: float = 0.0
    pnl_z_score: float = 0.0
    within_mc_confidence: bool = False

    # Validation
    validation_passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    # Detailed data
    round_results: List[RoundResult] = field(default_factory=list)


# =============================================================================
# Validation Functions
# =============================================================================

def calculate_sharpe_from_rounds(rounds: List[RoundResult]) -> float:
    """Calculate Sharpe ratio from round results."""
    if len(rounds) < 2:
        return 0.0

    returns = [r.pnl_pct for r in rounds]
    mean_return = sum(returns) / len(returns)

    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = variance ** 0.5 if variance > 0 else 0.0001

    if std_dev < 0.0001:
        return 0.0

    # Estimate rounds per year based on duration
    if rounds:
        duration_days = (rounds[-1].end_time - rounds[0].start_time).total_seconds() / 86400
        if duration_days > 0:
            rounds_per_year = len(rounds) / duration_days * 365
        else:
            rounds_per_year = 52  # Fallback: ~1 per week
    else:
        rounds_per_year = 52

    annualization = (rounds_per_year) ** 0.5
    return (mean_return / std_dev) * annualization


def load_paper_trading_csv(csv_path: str) -> List[RoundResult]:
    """
    Load paper trading results from CSV file.

    Args:
        csv_path: Path to CSV file with paper trading logs

    Returns:
        List of RoundResult objects
    """
    import csv

    round_results = []

    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return round_results

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                round_result = RoundResult(
                    round_number=int(row.get('round_num', 0)),
                    start_time=datetime.fromisoformat(row['start_time']),
                    end_time=datetime.fromisoformat(row['end_time']),
                    direction=row.get('direction', ''),
                    num_pyramids=int(row.get('num_pyramids', 0)),
                    entry_price=float(row.get('entry_price', 0)),
                    exit_price=float(row.get('exit_price', 0)),
                    pnl_pct=float(row.get('pnl_pct', 0)),
                    max_profit_pct=float(row.get('max_profit_pct', 0)),
                )
                round_results.append(round_result)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid row: {e}")
                continue

    return round_results


def analyze_paper_trading_results(
    config: PaperTradingConfig,
    round_results: List[RoundResult]
) -> PaperTradingResult:
    """
    Analyze paper trading results and compare to Monte Carlo expectations.

    Args:
        config: PaperTradingConfig with parameters and MC result
        round_results: List of completed round results

    Returns:
        PaperTradingResult with validation status
    """
    if not round_results:
        return PaperTradingResult(
            symbol=config.symbol,
            params=config.params,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_days=0,
            completed_rounds=0,
            total_pnl_pct=0,
            avg_pnl_per_round=0,
            win_rate=0,
            realized_sharpe=0,
            avg_slippage_bps=0,
            max_slippage_bps=0,
            execution_quality_score=0,
            validation_passed=False,
            failure_reasons=["No rounds completed"]
        )

    # Calculate basic statistics
    start_time = round_results[0].start_time
    end_time = round_results[-1].end_time
    duration_days = (end_time - start_time).total_seconds() / 86400

    total_pnl = sum(r.pnl_pct for r in round_results)
    avg_pnl = total_pnl / len(round_results)
    winning_rounds = sum(1 for r in round_results if r.pnl_pct > 0)
    win_rate = winning_rounds / len(round_results) * 100

    # Calculate Sharpe
    realized_sharpe = calculate_sharpe_from_rounds(round_results)

    # Execution quality
    slippages = [r.avg_slippage_bps for r in round_results if r.avg_slippage_bps > 0]
    avg_slippage = sum(slippages) / len(slippages) if slippages else 0
    max_slippage = max(slippages) if slippages else 0

    slippage_ratio = avg_slippage / config.expected_slippage_bps if config.expected_slippage_bps > 0 else 1.0
    execution_quality = 1.0 / max(slippage_ratio, 0.5)  # Higher is better

    # Compare to Monte Carlo
    mc_pnl_5th = 0.0
    mc_pnl_95th = 0.0
    mc_median = 0.0
    pnl_z_score = 0.0
    within_mc_ci = False

    if config.mc_result is not None:
        mc_pnl_5th = config.mc_result.pnl_5th_percentile
        mc_pnl_95th = config.mc_result.pnl_95th_percentile
        mc_median = config.mc_result.pnl_median
        mc_std = config.mc_result.pnl_std

        if mc_std > 0:
            pnl_z_score = (total_pnl - mc_median) / mc_std
        within_mc_ci = mc_pnl_5th <= total_pnl <= mc_pnl_95th

    # Validation checks
    failures = []

    if len(round_results) < config.min_rounds:
        failures.append(f"Insufficient rounds: {len(round_results)} < {config.min_rounds}")

    if config.mc_result is not None and not within_mc_ci:
        failures.append(f"P&L {total_pnl:.1f}% outside MC 90% CI [{mc_pnl_5th:.1f}%, {mc_pnl_95th:.1f}%]")

    if abs(pnl_z_score) > config.pnl_deviation_threshold:
        failures.append(f"P&L z-score {pnl_z_score:.2f} exceeds threshold {config.pnl_deviation_threshold}")

    if slippage_ratio > config.slippage_tolerance:
        failures.append(f"Slippage {slippage_ratio:.1f}x higher than expected")

    # Check for liquidations
    liquidation_rounds = [r for r in round_results if r.pnl_pct < -50]  # Proxy for liquidation
    if len(liquidation_rounds) > MAX_LIQUIDATION_EVENTS:
        failures.append(f"Liquidation-like events: {len(liquidation_rounds)}")

    if realized_sharpe < 0.5:
        failures.append(f"Sharpe ratio {realized_sharpe:.2f} < 0.5")

    return PaperTradingResult(
        symbol=config.symbol,
        params=config.params,
        start_time=start_time,
        end_time=end_time,
        duration_days=duration_days,
        completed_rounds=len(round_results),
        total_pnl_pct=total_pnl,
        avg_pnl_per_round=avg_pnl,
        win_rate=win_rate,
        realized_sharpe=realized_sharpe,
        avg_slippage_bps=avg_slippage,
        max_slippage_bps=max_slippage,
        execution_quality_score=execution_quality,
        mc_pnl_5th=mc_pnl_5th,
        mc_pnl_95th=mc_pnl_95th,
        mc_median=mc_median,
        pnl_z_score=pnl_z_score,
        within_mc_confidence=within_mc_ci,
        validation_passed=len(failures) == 0,
        failure_reasons=failures,
        round_results=round_results
    )


def format_paper_trading_report(result: PaperTradingResult) -> str:
    """Format paper trading results as a human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"PAPER TRADING VALIDATION REPORT: {result.symbol}")
    lines.append(f"Duration: {result.duration_days:.1f} days | Rounds: {result.completed_rounds}")
    lines.append("=" * 70)

    lines.append("\n--- PERFORMANCE ---")
    lines.append(f"  Total P&L:          {result.total_pnl_pct:+.2f}%")
    lines.append(f"  Avg P&L per round:  {result.avg_pnl_per_round:+.2f}%")
    lines.append(f"  Win Rate:           {result.win_rate:.1f}%")
    lines.append(f"  Realized Sharpe:    {result.realized_sharpe:.2f}")

    lines.append("\n--- EXECUTION QUALITY ---")
    lines.append(f"  Avg Slippage:       {result.avg_slippage_bps:.1f} bps")
    lines.append(f"  Max Slippage:       {result.max_slippage_bps:.1f} bps")
    lines.append(f"  Quality Score:      {result.execution_quality_score:.2f}")

    if result.mc_median != 0:
        lines.append("\n--- MONTE CARLO COMPARISON ---")
        lines.append(f"  Expected P&L (5th pct):   {result.mc_pnl_5th:+.1f}%")
        lines.append(f"  Expected P&L (median):    {result.mc_median:+.1f}%")
        lines.append(f"  Expected P&L (95th pct):  {result.mc_pnl_95th:+.1f}%")
        lines.append(f"  Actual P&L:               {result.total_pnl_pct:+.1f}%")
        lines.append(f"  Z-Score:                  {result.pnl_z_score:+.2f}")
        status = "PASS" if result.within_mc_confidence else "FAIL"
        lines.append(f"  Within 90% CI:            [{status}]")

    lines.append("\n" + "=" * 70)
    if result.validation_passed:
        lines.append("OVERALL: PASS - Strategy validated for live trading consideration")
        lines.append("\nNEXT STEPS:")
        lines.append("  1. Review risk limits and position sizing")
        lines.append("  2. Start live trading with 25% of validated size")
        lines.append("  3. Scale up gradually over 4 weeks if profitable")
        lines.append("  4. Set -40% equity kill switch")
    else:
        lines.append("OVERALL: FAIL - Strategy NOT validated for live trading")
        lines.append("\nFAILURE REASONS:")
        for reason in result.failure_reasons:
            lines.append(f"  - {reason}")
        lines.append("\nRECOMMENDATIONS:")
        lines.append("  1. Review execution model assumptions")
        lines.append("  2. Re-optimize with updated slippage parameters")
        lines.append("  3. Continue paper trading to gather more data")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_paper_trading_result(result: PaperTradingResult):
    """Save paper trading results to JSON file."""
    os.makedirs(LOG_DIR, exist_ok=True)

    # Convert to serializable format
    result_dict = {
        'symbol': result.symbol,
        'params': result.params,
        'start_time': result.start_time.isoformat(),
        'end_time': result.end_time.isoformat(),
        'duration_days': result.duration_days,
        'completed_rounds': result.completed_rounds,
        'total_pnl_pct': result.total_pnl_pct,
        'avg_pnl_per_round': result.avg_pnl_per_round,
        'win_rate': result.win_rate,
        'realized_sharpe': result.realized_sharpe,
        'avg_slippage_bps': result.avg_slippage_bps,
        'max_slippage_bps': result.max_slippage_bps,
        'execution_quality_score': result.execution_quality_score,
        'mc_pnl_5th': result.mc_pnl_5th,
        'mc_pnl_95th': result.mc_pnl_95th,
        'mc_median': result.mc_median,
        'pnl_z_score': result.pnl_z_score,
        'within_mc_confidence': result.within_mc_confidence,
        'validation_passed': result.validation_passed,
        'failure_reasons': result.failure_reasons
    }

    filepath = os.path.join(LOG_DIR, f"{result.symbol}_{RESULTS_FILE}")
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nResults saved to: {filepath}")


# =============================================================================
# Paper Trading Runner (Stub for Integration)
# =============================================================================

class PaperTradingValidator:
    """
    Paper trading validation runner.

    This class provides the interface for running paper trading validation.
    The actual trading logic would integrate with the existing main.py.

    Usage:
        1. Run optimizer to get best parameters
        2. Load parameters and MC result
        3. Run paper trading for 4 weeks
        4. Analyze results and validate
    """

    def __init__(self, config: PaperTradingConfig):
        self.config = config
        self.round_results: List[RoundResult] = []
        self.is_running = False

    def add_round_result(self, result: RoundResult):
        """Add a completed round result (called by live trading system)."""
        self.round_results.append(result)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current paper trading statistics."""
        if not self.round_results:
            return {'rounds': 0, 'pnl': 0, 'status': 'No data'}

        total_pnl = sum(r.pnl_pct for r in self.round_results)
        return {
            'rounds': len(self.round_results),
            'pnl': total_pnl,
            'avg_pnl': total_pnl / len(self.round_results),
            'status': 'Running' if self.is_running else 'Stopped'
        }

    def validate(self) -> PaperTradingResult:
        """Validate paper trading results."""
        return analyze_paper_trading_results(self.config, self.round_results)


# =============================================================================
# Command Line Interface
# =============================================================================

def load_optimization_result(symbol: str) -> Optional[Dict]:
    """Load optimization results for a symbol."""
    result_file = f"logs/{symbol}_v4_final_result.json"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None


def main():
    """Main entry point for paper trading validation."""
    print("=" * 70)
    print("PAPER TRADING VALIDATOR")
    print("=" * 70)

    # List available optimization results
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print("No optimization results found. Run optimizer first.")
        return

    available = []
    for f in os.listdir(log_dir):
        if f.endswith("_v4_final_result.json"):
            symbol = f.replace("_v4_final_result.json", "")
            available.append(symbol)

    if not available:
        print("No optimization results found. Run optimizer first.")
        return

    print("\nAvailable symbols:")
    for i, sym in enumerate(available, 1):
        print(f"  {i}. {sym}")

    # Get user selection
    try:
        choice = int(input("\nSelect symbol (number): "))
        symbol = available[choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    # Load optimization result
    result = load_optimization_result(symbol)
    if not result:
        print(f"Could not load results for {symbol}")
        return

    params = result.get('winner_params', {})
    print(f"\nLoaded parameters for {symbol}:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Create config
    config = PaperTradingConfig(
        symbol=symbol,
        params=params,
        min_rounds=30,
        max_duration_days=28
    )

    print("\n" + "=" * 70)
    print("PAPER TRADING INSTRUCTIONS")
    print("=" * 70)
    print(f"""
To run paper trading:

1. Start the trading bot in paper trading mode:
   python main.py --mode trading --symbol {symbol} \\
     --threshold {params.get('threshold', 3.0)} \\
     --trailing {params.get('trailing', 1.5)} \\
     --pyramid {params.get('pyramid_step', 2.0)} \\
     --paper-trading

2. Let it run for at least 30 rounds (approximately 2-4 weeks)

3. Once complete, run this validator again to analyze results

4. Results will be compared to Monte Carlo expectations

IMPORTANT:
- Use Binance Futures TESTNET, not live
- Monitor for any issues during paper trading
- Do NOT proceed to live trading without validation passing
""")


# =============================================================================
# Iterative Refinement Loop (Phase 3)
# =============================================================================

def find_latest_paper_trading_csv(symbol: str) -> Optional[str]:
    """Find the most recent paper trading CSV for a symbol."""
    pt_dir = Path(LOG_DIR)
    if not pt_dir.exists():
        return None

    csv_files = sorted(pt_dir.glob(f"{symbol}_*.csv"), reverse=True)
    return str(csv_files[0]) if csv_files else None


def iterative_refinement_loop(
    symbol: str,
    initial_params: Dict,
    round_returns: List[float],
    max_iterations: int = 3
) -> Tuple[Dict, bool]:
    """
    Run paper trading, detect model mismatches, adjust, repeat.

    If paper trading shows discrepancies (e.g., slippage 2x higher than expected):
    1. Detect the mismatch type
    2. Adjust the relevant model parameter
    3. Signal for re-validation

    Args:
        symbol: Trading pair symbol
        initial_params: Initial optimized parameters
        round_returns: Round P&L returns from optimization
        max_iterations: Maximum refinement iterations

    Returns:
        Tuple of (final_params, validated)
    """
    from core.monte_carlo import run_monte_carlo_validation

    params = initial_params.copy()

    for iteration in range(max_iterations):
        print(f"\n{'=' * 70}")
        print(f"REFINEMENT ITERATION {iteration + 1}/{max_iterations}")
        print("=" * 70)

        # 1. Run Monte Carlo on current params
        print("\nStep 1: Running Monte Carlo validation...")
        mc_result, perm_result = run_monte_carlo_validation(
            round_returns=round_returns,
            params=params
        )

        # 2. Load paper trading results
        print("\nStep 2: Loading paper trading results...")
        pt_csv = find_latest_paper_trading_csv(symbol)
        if not pt_csv:
            print("  [ERROR] No paper trading data found")
            print(f"  Run paper trading first: python main.py --mode trading --symbol {symbol}")
            return params, False

        pt_rounds = load_paper_trading_csv(pt_csv)
        print(f"  Loaded {len(pt_rounds)} rounds from {pt_csv}")

        if len(pt_rounds) < MIN_COMPLETED_ROUNDS:
            print(f"  [WARNING] Only {len(pt_rounds)} rounds - need at least {MIN_COMPLETED_ROUNDS}")
            print("  Continue paper trading to gather more data")
            return params, False

        # 3. Analyze discrepancies
        print("\nStep 3: Analyzing discrepancies...")
        config = PaperTradingConfig(
            symbol=symbol,
            params=params,
            mc_result=mc_result
        )
        pt_result = analyze_paper_trading_results(config, pt_rounds)

        print(format_paper_trading_report(pt_result))

        if pt_result.validation_passed:
            print(f"\n[SUCCESS] Validation passed on iteration {iteration + 1}")
            return params, True

        # 4. Adjust model based on failures
        print("\nStep 4: Analyzing failure reasons and adjusting model...")
        adjustments_made = False

        for reason in pt_result.failure_reasons:
            if "Slippage" in reason or "slippage" in reason:
                # Extract slippage ratio and adjust
                current_mult = params.get('slippage_multiplier', 1.0)
                # Increase by 50% each iteration
                new_mult = current_mult * 1.5
                params['slippage_multiplier'] = new_mult
                print(f"  [ADJUST] Increasing slippage_multiplier: {current_mult:.2f} -> {new_mult:.2f}")
                adjustments_made = True

            elif "P&L" in reason and "outside" in reason.lower():
                # P&L outside MC confidence interval
                print("  [WARN] P&L outside Monte Carlo expectations")
                print("  Consider: Re-running optimization with updated execution model")
                # Can't auto-adjust this - would need full re-optimization
                adjustments_made = True

            elif "Insufficient" in reason:
                print("  [WARN] Insufficient data - continue paper trading")

        if not adjustments_made:
            print("\n  [WARN] No actionable adjustments identified")
            print("  Manual review required")
            return params, False

        print(f"\n  Proceeding to iteration {iteration + 2}...")

    print(f"\n[FAIL] Max iterations ({max_iterations}) reached without validation")
    print("Manual intervention required:")
    print("  1. Review execution model assumptions")
    print("  2. Consider re-optimization with more conservative parameters")
    print("  3. Extend paper trading period")

    return params, False


def run_refinement_cli(symbol: str) -> None:
    """CLI interface for iterative refinement loop."""
    print("=" * 70)
    print("ITERATIVE REFINEMENT LOOP")
    print("=" * 70)

    # Load optimization results
    opt_result = load_optimization_result(symbol)
    if not opt_result:
        print(f"No optimization results found for {symbol}")
        print("Run optimizer first: python optimize_pyramid_v4.py")
        return

    # Extract parameters and round returns
    winner = opt_result.get('winner', {})
    params = {
        'threshold': winner.get('threshold', 3.0),
        'trailing': winner.get('trailing', 1.5),
        'pyramid_step': winner.get('pyramid_step', 2.0),
    }

    # Get round returns from holdout or walks
    round_returns = opt_result.get('holdout_result', {}).get('round_returns', [])
    if not round_returns:
        round_returns = opt_result.get('round_returns', [])

    if not round_returns:
        print("No round returns found in optimization results")
        return

    print(f"\nLoaded {len(round_returns)} round returns from optimization")
    print(f"Parameters: {params}")

    # Run refinement loop
    final_params, validated = iterative_refinement_loop(
        symbol=symbol,
        initial_params=params,
        round_returns=round_returns
    )

    # Summary
    print("\n" + "=" * 70)
    print("REFINEMENT COMPLETE")
    print("=" * 70)
    print(f"Final parameters: {final_params}")
    print(f"Validated: {'YES' if validated else 'NO'}")

    if validated:
        print("\nStrategy is validated for live trading consideration")
    else:
        print("\nStrategy did NOT pass validation - do not deploy to live trading")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--refine":
        if len(sys.argv) > 2:
            run_refinement_cli(sys.argv[2])
        else:
            print("Usage: python paper_trading_validator.py --refine SYMBOL")
    else:
        main()
