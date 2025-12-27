"""
Walk-Forward Validation Framework

Walk-forward analysis is the most realistic form of backtesting because it
simulates how a strategy would actually be deployed:

1. Train on historical data (e.g., Years 1-2)
2. Test on future data (e.g., Year 3)
3. Roll forward: Train on Years 1-3, test on Year 4
4. Continue until all data is used

This prevents look-ahead bias and tests strategy stability across regimes.

Key Benefits:
- No data leakage (each test period is truly out-of-sample)
- Tests parameter stability (do optimal params change over time?)
- Detects regime changes (when does strategy stop working?)
- More realistic than fixed train/test split

Usage:
    from core.walk_forward import run_walk_forward_optimization

    results = run_walk_forward_optimization(
        prices=prices,
        funding_rates=funding_rates,
        train_years=2,
        test_years=1,
        step_years=1,
    )
"""

import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import numpy as np

from core.minimal_grid import (
    build_minimal_grid,
    generate_all_combinations,
    check_validation_gate,
    check_bonferroni_significance,
)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    window_id: int
    train_start_pct: float
    train_end_pct: float
    test_start_pct: float
    test_end_pct: float
    train_start_idx: int = 0
    train_end_idx: int = 0
    test_start_idx: int = 0
    test_end_idx: int = 0


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""
    window: WalkForwardWindow
    best_params: Dict[str, Any]
    train_pnl: float
    test_pnl: float
    test_rounds: int
    test_win_rate: float
    test_max_drawdown: float
    params_changed: bool  # Did optimal params change from previous window?
    per_round_returns: List[float] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward analysis."""
    windows: List[WindowResult]
    total_test_pnl: float
    avg_test_pnl: float
    total_test_rounds: int
    avg_test_win_rate: float
    max_test_drawdown: float
    param_stability_score: float  # 0-1, how often did params stay same?
    best_stable_params: Optional[Dict[str, Any]]  # Most common winning params
    all_per_round_returns: List[float] = field(default_factory=list)


def create_walk_forward_windows(
    total_prices: int,
    train_years: int = 2,
    test_years: int = 1,
    step_years: int = 1,
    total_years: int = 5,
) -> List[WalkForwardWindow]:
    """
    Create walk-forward windows for analysis.

    With defaults (train=2, test=1, step=1, total=5):
    - Window 0: Train Y1-Y2, Test Y3
    - Window 1: Train Y1-Y3, Test Y4
    - Window 2: Train Y1-Y4, Test Y5

    Args:
        total_prices: Total number of price records
        train_years: Years of training data per window
        test_years: Years of test data per window
        step_years: How much to advance per window
        total_years: Total years of data

    Returns:
        List of WalkForwardWindow objects
    """
    windows = []
    prices_per_year = total_prices / total_years

    window_id = 0
    test_start_year = train_years

    while test_start_year + test_years <= total_years:
        train_start_year = 0  # Always start from beginning (expanding window)
        train_end_year = test_start_year
        test_end_year = test_start_year + test_years

        # Calculate percentages
        train_start_pct = train_start_year / total_years
        train_end_pct = train_end_year / total_years
        test_start_pct = train_end_year / total_years
        test_end_pct = test_end_year / total_years

        # Calculate indices
        window = WalkForwardWindow(
            window_id=window_id,
            train_start_pct=train_start_pct,
            train_end_pct=train_end_pct,
            test_start_pct=test_start_pct,
            test_end_pct=test_end_pct,
            train_start_idx=int(train_start_pct * total_prices),
            train_end_idx=int(train_end_pct * total_prices),
            test_start_idx=int(test_start_pct * total_prices),
            test_end_idx=int(test_end_pct * total_prices),
        )
        windows.append(window)

        window_id += 1
        test_start_year += step_years

    return windows


def optimize_window(
    prices,
    window: WalkForwardWindow,
    funding_rates: Optional[Dict],
    run_backtest_func: Callable,
    all_combos: List[Dict],
    min_val_pnl_ratio: float = 0.3,
    min_val_rounds: int = 15,
    bonferroni_alpha: float = 0.000078,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], float, Dict]:
    """
    Find optimal parameters for a single walk-forward window.

    Args:
        prices: Memory-mapped price array
        window: Walk-forward window to optimize
        funding_rates: Funding rate dict
        run_backtest_func: Function to run backtest
        all_combos: All parameter combinations
        min_val_pnl_ratio: Minimum val/train PnL ratio
        min_val_rounds: Minimum validation rounds
        bonferroni_alpha: Corrected significance level
        verbose: Print progress

    Returns:
        Tuple of (best_params, best_train_pnl, best_train_result)
    """
    train_prices = prices[window.train_start_idx:window.train_end_idx]

    best_params = None
    best_train_pnl = float('-inf')
    best_result = None

    # Use 80% of train data for training, 20% for validation
    train_80_end = int(len(train_prices) * 0.80)
    actual_train = train_prices[:train_80_end]
    actual_val = train_prices[train_80_end:]

    for i, params in enumerate(all_combos):
        # Train on 80% of window's training data
        train_result = run_backtest_func(params, actual_train, funding_rates)
        train_pnl = train_result.get('total_pnl', 0)

        if train_pnl <= 0:
            continue

        # Validate on remaining 20%
        val_result = run_backtest_func(params, actual_val, funding_rates)

        # Check validation gate
        gate_passed, _ = check_validation_gate(
            train_result, val_result, min_val_pnl_ratio, min_val_rounds
        )

        if not gate_passed:
            continue

        # Track best by train PnL
        if train_pnl > best_train_pnl:
            best_train_pnl = train_pnl
            best_params = params
            best_result = train_result

        if verbose and i > 0 and i % 100 == 0:
            print(f"    Window {window.window_id}: {i}/{len(all_combos)} combos tested...")

    return best_params, best_train_pnl, best_result


def test_params_on_window(
    params: Dict,
    prices,
    window: WalkForwardWindow,
    funding_rates: Optional[Dict],
    run_backtest_func: Callable,
) -> Dict:
    """Test parameters on a window's test period."""
    test_prices = prices[window.test_start_idx:window.test_end_idx]
    return run_backtest_func(params, test_prices, funding_rates)


def run_walk_forward_optimization(
    prices,
    funding_rates: Optional[Dict],
    run_backtest_func: Callable,
    train_years: int = 2,
    test_years: int = 1,
    step_years: int = 1,
    total_years: int = 5,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Run complete walk-forward optimization.

    Args:
        prices: Memory-mapped price array
        funding_rates: Funding rate dict (or None)
        run_backtest_func: Function to run backtest (params, prices, funding) -> result dict
        train_years: Years of training per window
        test_years: Years of testing per window
        step_years: How much to advance per window
        total_years: Total years of data
        verbose: Print detailed progress

    Returns:
        WalkForwardResult with aggregated results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("WALK-FORWARD OPTIMIZATION")
        print("=" * 70)
        print(f"Train window: {train_years} year(s) (expanding)")
        print(f"Test window: {test_years} year(s)")
        print(f"Step: {step_years} year(s)")
        print()

    # Create windows
    windows = create_walk_forward_windows(
        total_prices=len(prices),
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
        total_years=total_years,
    )

    if verbose:
        print(f"Created {len(windows)} walk-forward windows:")
        for w in windows:
            print(f"  Window {w.window_id}: Train {w.train_start_pct*100:.0f}%-{w.train_end_pct*100:.0f}%, "
                  f"Test {w.test_start_pct*100:.0f}%-{w.test_end_pct*100:.0f}%")
        print()

    # Get parameter grid
    grid = build_minimal_grid()
    all_combos = generate_all_combinations(grid)
    n_combos = len(all_combos)
    bonferroni_alpha = 0.05 / n_combos

    if verbose:
        print(f"Testing {n_combos} parameter combinations per window")
        print(f"Bonferroni alpha: {bonferroni_alpha:.6f}")
        print()

    # Process each window
    window_results: List[WindowResult] = []
    prev_best_params = None
    all_returns = []
    param_changes = 0

    for window in windows:
        if verbose:
            print(f"\n--- WINDOW {window.window_id} ---")
            print(f"Training on {window.train_start_pct*100:.0f}%-{window.train_end_pct*100:.0f}% "
                  f"({window.train_end_idx - window.train_start_idx:,} prices)")

        # Optimize on training data
        best_params, train_pnl, train_result = optimize_window(
            prices=prices,
            window=window,
            funding_rates=funding_rates,
            run_backtest_func=run_backtest_func,
            all_combos=all_combos,
            verbose=verbose,
        )

        if best_params is None:
            if verbose:
                print(f"  [WARNING] No valid params found for window {window.window_id}")
            continue

        # Test on out-of-sample period
        if verbose:
            print(f"Testing on {window.test_start_pct*100:.0f}%-{window.test_end_pct*100:.0f}% "
                  f"({window.test_end_idx - window.test_start_idx:,} prices)")

        test_result = test_params_on_window(
            params=best_params,
            prices=prices,
            window=window,
            funding_rates=funding_rates,
            run_backtest_func=run_backtest_func,
        )

        test_pnl = test_result.get('total_pnl', 0)
        test_rounds = test_result.get('total_rounds', 0)
        test_win_rate = test_result.get('win_rate', 0)
        test_max_dd = test_result.get('max_drawdown_pct', 0)
        per_round = test_result.get('per_round_returns', [])

        # Check if params changed from previous window
        params_changed = False
        if prev_best_params is not None:
            # Compare key parameters
            for key in ['threshold', 'trailing', 'pyramid_step']:
                if best_params.get(key) != prev_best_params.get(key):
                    params_changed = True
                    param_changes += 1
                    break

        prev_best_params = best_params.copy()

        if verbose:
            change_str = " (CHANGED)" if params_changed else ""
            print(f"  Best params{change_str}: th={best_params['threshold']}, "
                  f"tr={best_params['trailing']}, ps={best_params['pyramid_step']}")
            print(f"  Train P&L: {train_pnl:+.2f}%")
            print(f"  TEST P&L:  {test_pnl:+.2f}% ({test_rounds} rounds, {test_win_rate:.1f}% win rate)")

        # Store result
        result = WindowResult(
            window=window,
            best_params=best_params,
            train_pnl=train_pnl,
            test_pnl=test_pnl,
            test_rounds=test_rounds,
            test_win_rate=test_win_rate,
            test_max_drawdown=test_max_dd,
            params_changed=params_changed,
            per_round_returns=per_round,
        )
        window_results.append(result)
        all_returns.extend(per_round)

        # Memory cleanup
        gc.collect()

    # Aggregate results
    if not window_results:
        return WalkForwardResult(
            windows=[],
            total_test_pnl=0,
            avg_test_pnl=0,
            total_test_rounds=0,
            avg_test_win_rate=0,
            max_test_drawdown=0,
            param_stability_score=0,
            best_stable_params=None,
            all_per_round_returns=[],
        )

    total_test_pnl = sum(r.test_pnl for r in window_results)
    avg_test_pnl = total_test_pnl / len(window_results)
    total_test_rounds = sum(r.test_rounds for r in window_results)
    avg_win_rate = sum(r.test_win_rate for r in window_results) / len(window_results)
    max_dd = max(r.test_max_drawdown for r in window_results)

    # Parameter stability: 1 - (changes / windows)
    param_stability = 1.0 - (param_changes / len(window_results)) if len(window_results) > 1 else 1.0

    # Find most common params (best stable params)
    from collections import Counter
    param_keys = []
    for r in window_results:
        key = (r.best_params['threshold'], r.best_params['trailing'], r.best_params['pyramid_step'])
        param_keys.append(key)

    if param_keys:
        most_common_key = Counter(param_keys).most_common(1)[0][0]
        best_stable_params = {
            'threshold': most_common_key[0],
            'trailing': most_common_key[1],
            'pyramid_step': most_common_key[2],
        }
        # Add other params from the first result with matching key
        for r in window_results:
            if (r.best_params['threshold'], r.best_params['trailing'], r.best_params['pyramid_step']) == most_common_key:
                best_stable_params.update(r.best_params)
                break
    else:
        best_stable_params = None

    result = WalkForwardResult(
        windows=window_results,
        total_test_pnl=total_test_pnl,
        avg_test_pnl=avg_test_pnl,
        total_test_rounds=total_test_rounds,
        avg_test_win_rate=avg_win_rate,
        max_test_drawdown=max_dd,
        param_stability_score=param_stability,
        best_stable_params=best_stable_params,
        all_per_round_returns=all_returns,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("WALK-FORWARD RESULTS SUMMARY")
        print("=" * 70)
        print(f"Windows analyzed: {len(window_results)}")
        print(f"Total test P&L: {total_test_pnl:+.2f}%")
        print(f"Avg test P&L per window: {avg_test_pnl:+.2f}%")
        print(f"Total test rounds: {total_test_rounds}")
        print(f"Avg test win rate: {avg_win_rate:.1f}%")
        print(f"Max test drawdown: {max_dd:.1f}%")
        print(f"Parameter stability: {param_stability*100:.1f}%")
        if best_stable_params:
            print(f"Most stable params: th={best_stable_params['threshold']}, "
                  f"tr={best_stable_params['trailing']}, ps={best_stable_params['pyramid_step']}")
        print("=" * 70)

    return result


def format_walk_forward_report(result: WalkForwardResult) -> str:
    """Format a detailed walk-forward report."""
    lines = []
    lines.append("=" * 70)
    lines.append("WALK-FORWARD ANALYSIS REPORT")
    lines.append("=" * 70)

    lines.append("\nPER-WINDOW RESULTS:")
    for r in result.windows:
        status = "[+]" if r.test_pnl > 0 else "[-]"
        change = " (changed)" if r.params_changed else ""
        lines.append(f"  Window {r.window.window_id}: {status} Train={r.train_pnl:+.2f}%, "
                     f"Test={r.test_pnl:+.2f}% ({r.test_rounds} rounds){change}")
        lines.append(f"              Params: th={r.best_params['threshold']}, "
                     f"tr={r.best_params['trailing']}, ps={r.best_params['pyramid_step']}")

    lines.append("\nAGGREGATE RESULTS:")
    lines.append(f"  Total test P&L: {result.total_test_pnl:+.2f}%")
    lines.append(f"  Avg test P&L: {result.avg_test_pnl:+.2f}%")
    lines.append(f"  Total test rounds: {result.total_test_rounds}")
    lines.append(f"  Avg win rate: {result.avg_test_win_rate:.1f}%")
    lines.append(f"  Max drawdown: {result.max_test_drawdown:.1f}%")

    lines.append("\nPARAMETER STABILITY:")
    lines.append(f"  Stability score: {result.param_stability_score*100:.1f}%")
    if result.best_stable_params:
        lines.append(f"  Most stable: th={result.best_stable_params['threshold']}, "
                     f"tr={result.best_stable_params['trailing']}, "
                     f"ps={result.best_stable_params['pyramid_step']}")

    lines.append("\nVERDICT:")
    if result.total_test_pnl > 0 and result.param_stability_score >= 0.5:
        lines.append("  [PASS] Strategy shows consistent out-of-sample profits")
        lines.append("         with stable parameter selection across regimes.")
    elif result.total_test_pnl > 0:
        lines.append("  [WARN] Strategy is profitable but params change frequently.")
        lines.append("         Consider using more robust parameter selection.")
    else:
        lines.append("  [FAIL] Strategy not profitable in walk-forward testing.")
        lines.append("         Historical results may be overfit.")

    lines.append("=" * 70)

    return "\n".join(lines)
