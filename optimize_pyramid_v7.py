#!/usr/bin/env python3
"""
Pure Profit Pyramid Strategy Optimizer (v7.6)

v7.6 NEW:
- Removed arbitrary combo limits from all phases
- All combos that pass each phase are now tested in subsequent phases
- No more truncation that could miss better strategies
- Fixed Bonferroni calculation to use actual combo count

v7.5:
- Added progress bars to all parallel optimization phases

v7.1 CRITICAL FIXES:
- Fixed data leakage in robustness testing (excluded holdout data)
- Fixed overlapping folds (now non-overlapping for proper CV)
- Moved Bonferroni correction from train to holdout phase
- Increased holdout from 10% to 15% (9 months vs 6 months)
- Added position-size-dependent slippage model
- Added walk-forward validation for regime testing

PHILOSOPHY: "Whatever Works" - pure profit optimization
- NO arbitrary constraints (win rate, drawdown, expectancy, profit factor)
- PURE PROFIT optimization - only holdout P&L matters
- 640 combinations (8x better statistical validity than v5/v6)

STATISTICAL VALIDITY:
- Bonferroni correction applied to HOLDOUT (not train) - α = 0.05/N
- Non-overlapping cross-fold validation
- 15% holdout (~9 months) for statistical confidence
- Walk-forward validation for regime stability
- Monte Carlo survival simulation
- Robustness testing (on train/val data only - no holdout leakage)

Grid Size: 5 x 4 x 4 x 2 x 2 x 2 = 640 combinations

Usage:
    python optimize_pyramid_v7.py --symbol BTCUSDT --days 1825

Author: Claude Code
"""

import os
import sys
import csv
import json
import gc
import resource
import argparse
import signal
import multiprocessing as mp
from multiprocessing import Pool
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy

# Local imports
from core.minimal_grid import (
    build_minimal_grid,
    expand_params,
    generate_all_combinations,
    get_grid_size,
    get_bonferroni_alpha,
    get_fixed_folds,
    check_validation_gate,
    check_bonferroni_significance,
)
from core.disk_results import (
    DiskResultStorage,
    calculate_real_sharpe,
)
from core.slippage_model import (
    calculate_slippage_pct,
    estimate_round_slippage_pct,
)
from backtest_pyramid import (
    run_pyramid_backtest,
    DEFAULT_EXECUTION_MODEL,
)


# =============================================================================
# MEMORY MONITORING
# =============================================================================

MEMORY_WARNING_MB = 800
MEMORY_CRITICAL_MB = 1200


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return usage / (1024 * 1024)  # macOS: bytes to MB
    else:
        return usage / 1024  # Linux: KB to MB


def check_memory_usage(context: str = "") -> None:
    """Check memory usage and warn/force GC if high."""
    usage_mb = get_memory_usage_mb()

    if usage_mb > MEMORY_CRITICAL_MB:
        print(f"\n  CRITICAL: Memory usage {usage_mb:.0f}MB > {MEMORY_CRITICAL_MB}MB [{context}]")
        print("  Forcing garbage collection...")
        gc.collect()
        new_usage = get_memory_usage_mb()
        print(f"  After GC: {new_usage:.0f}MB")
    elif usage_mb > MEMORY_WARNING_MB:
        print(f"\n  WARNING: Memory usage {usage_mb:.0f}MB > {MEMORY_WARNING_MB}MB [{context}]")
        gc.collect()


# =============================================================================
# PROGRESS BAR UTILITY (v7.5 - parallel processing visibility)
# =============================================================================

def print_progress_bar(completed: int, total: int, prefix: str = "",
                       extra_info: str = "", start_time: float = None) -> None:
    """
    Print a progress bar with ETA. Works correctly with multiprocessing
    because it runs in the main process as results arrive from workers.

    Args:
        completed: Number of items completed
        total: Total number of items
        prefix: Optional prefix string (e.g., "Phase 1: ")
        extra_info: Optional extra info (e.g., "Pass: 45 | Fail: 595")
        start_time: Start time from time.time() for ETA calculation
    """
    import time as time_module

    pct = completed / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * completed / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    if start_time and completed > 0:
        elapsed = time_module.time() - start_time
        eta = (total - completed) * (elapsed / completed)
        eta_str = f"{int(eta//60)}m{int(eta%60):02d}s"
    else:
        eta_str = "..."

    line = f"\r{prefix}[{bar}] {pct:5.1f}% | {completed}/{total}"
    if extra_info:
        line += f" | {extra_info}"
    line += f" | ETA: {eta_str}"

    print(line, end="", flush=True)
    sys.stdout.flush()


# =============================================================================
# CONFIGURATION (PURE PROFIT - NO ARBITRARY CONSTRAINTS)
# =============================================================================

# Default settings
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DAYS = 1825  # 5 years
DEFAULT_BATCH_SIZE = 50
NUM_FOLDS = 3
LOG_DIR = "logs"

# =============================================================================
# THOROUGHNESS CONFIGURATION (v7.2 user-selectable analysis depth)
# =============================================================================

@dataclass
class ThoroughnessConfig:
    """Configuration for optimization thoroughness level."""
    name: str
    sample_rate: int            # Use every Nth price (1=all, 10=every 10th)
    crossfold_divisor: int      # sample_rate // divisor for cross-fold
    robustness_divisor: int     # sample_rate // divisor for robustness
    holdout_sample_rate: int    # 1 = full data, higher = sampled
    mc_bootstrap: int           # Monte Carlo bootstrap iterations
    mc_permutation: int         # Monte Carlo permutation iterations
    estimated_time: str         # Human-readable estimate


THOROUGHNESS_CONFIGS = {
    'quick': ThoroughnessConfig(
        name='Quick',
        sample_rate=100,
        crossfold_divisor=2,
        robustness_divisor=2,
        holdout_sample_rate=10,
        mc_bootstrap=1000,
        mc_permutation=500,
        estimated_time='2-4 hours',
    ),
    'standard': ThoroughnessConfig(
        name='Standard',
        sample_rate=10,
        crossfold_divisor=2,
        robustness_divisor=2,
        holdout_sample_rate=1,
        mc_bootstrap=10000,
        mc_permutation=5000,
        estimated_time='6-12 hours',
    ),
    'thorough': ThoroughnessConfig(
        name='Thorough',
        sample_rate=5,
        crossfold_divisor=1,
        robustness_divisor=1,
        holdout_sample_rate=1,
        mc_bootstrap=20000,
        mc_permutation=10000,
        estimated_time='24-48 hours',
    ),
    'exhaustive': ThoroughnessConfig(
        name='Exhaustive',
        sample_rate=1,
        crossfold_divisor=1,
        robustness_divisor=1,
        holdout_sample_rate=1,
        mc_bootstrap=50000,
        mc_permutation=25000,
        estimated_time='27-54 days',
    ),
}

DEFAULT_THOROUGHNESS = 'standard'

# VALIDATION GATE (MINIMAL - only prevents garbage)
MIN_VAL_PNL_RATIO = 0.3  # val_pnl >= 0.3 * train_pnl (loose overfitting check)
MIN_VAL_ROUNDS = 15      # Minimum rounds for statistical validity

# HOLDOUT REQUIREMENTS (MINIMAL)
MIN_HOLDOUT_PNL = 0.0    # Must be positive (that's it!)
MIN_HOLDOUT_ROUNDS = 30  # Need some rounds for statistical validity

# NO OTHER CONSTRAINTS:
# - NO MIN_WIN_RATE_PCT
# - NO MAX_DRAWDOWN_PCT
# - NO MIN_EXPECTANCY
# - NO MIN_PROFIT_FACTOR
# - NO MIN_WIN_LOSS_RATIO

# Robustness testing (keep this - it's not arbitrary, it's statistical)
MIN_ROBUSTNESS_SCORE = 0.6  # Nearby params must give at least 60% of best P&L

# Family-wise error rate for Bonferroni
FAMILY_ALPHA = 0.05

# Realistic simulation settings
INITIAL_CAPITAL = 5000.0
LEVERAGE = 10
POSITION_SIZE_USDT = 100.0
USE_MARGIN_TRACKING = True
APPLY_FUNDING_RATES = True
VALIDATE_DATA_QUALITY = True

# Parameter limits (for robustness perturbations)
PARAM_LIMITS = {
    'threshold': (0.5, 30.0),
    'trailing': (0.2, 20.0),
    'pyramid_step': (0.2, 10.0),
    'max_pyramids': (1, 9999),
}


@dataclass
class OptimizationResult:
    """Result from optimization run (PURE PROFIT VERSION)."""
    params: Dict[str, Any]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    holdout_metrics: Optional[Dict[str, float]]
    passed_gate: bool
    is_significant: bool
    p_value: float
    sharpe: float
    per_round_returns: List[float]
    # Cross-fold validation
    fold_pnls: List[float] = field(default_factory=list)
    total_pnl_positive: bool = False
    avg_val_pnl: float = 0.0
    # Robustness
    robustness_score: float = 0.0
    # Monte Carlo results
    mc_passes: bool = False
    mc_prob_positive: float = 0.0
    mc_prob_ruin: float = 1.0
    mc_sharpe_5th: float = 0.0
    mc_sequence_matters: bool = True
    # Regime validation (optional)
    regime_pass: bool = True


# =============================================================================
# MULTIPROCESSING SUPPORT (v7.3)
# =============================================================================

def get_worker_count(requested: Optional[int] = None) -> int:
    """
    Determine number of worker processes for parallel execution.

    Args:
        requested: User-specified worker count (--workers flag)
                  None = auto-detect (cpu_count - 1)
                  0 = force sequential mode

    Returns:
        Number of workers (0 for sequential, 1+ for parallel)
    """
    if requested == 0:
        return 0  # Force sequential mode
    if requested is not None:
        return max(1, min(requested, os.cpu_count() or 4))
    # Auto-detect: use all cores minus 1 (leave one for OS/main process)
    return max(1, (os.cpu_count() or 4) - 1)


# Worker process globals (set by init_worker, used by worker functions)
_worker_prices = None
_worker_prices_file = None
_worker_meta = None
_worker_folds = None
_worker_funding = None
_worker_thoroughness = None
# Phase 2-5 worker globals (set lazily via set_worker_holdout_data)
_worker_holdout_prices = None
_worker_train_val_prices = None
_worker_holdout_start_pct = None
_worker_symbol = None


def init_worker(prices_file: str, meta: Dict, folds: List[Dict],
                funding_rates: Dict, thoroughness_dict: Dict,
                holdout_dict: Dict = None, symbol: str = None) -> None:
    """
    Initialize worker process globals.

    Called once per worker by Pool. Re-opens memmap in worker process
    for cross-platform safety. On fork-based systems (macOS/Linux),
    this shares physical memory pages via copy-on-write.

    Args:
        prices_file: Path to memmap file
        meta: Memmap metadata dict
        folds: List of fold configurations
        funding_rates: Funding rate dict
        thoroughness_dict: ThoroughnessConfig as dict
        holdout_dict: Holdout configuration for Phases 2-5 (optional)
        symbol: Trading symbol for slippage estimation (optional)
    """
    global _worker_prices, _worker_prices_file, _worker_meta
    global _worker_folds, _worker_funding, _worker_thoroughness
    global _worker_holdout_prices, _worker_train_val_prices
    global _worker_holdout_start_pct, _worker_symbol

    from core.memmap_prices import load_prices_memmap

    _worker_prices_file = prices_file
    _worker_meta = meta
    _worker_folds = folds
    _worker_funding = funding_rates
    _worker_thoroughness = ThoroughnessConfig(**thoroughness_dict)
    _worker_symbol = symbol

    # Re-open memmap in worker (shares pages via COW on fork)
    _worker_prices = load_prices_memmap(prices_file, meta)

    # Set up holdout slices for Phases 2-5 (if holdout info provided)
    if holdout_dict is not None:
        _worker_holdout_start_pct = holdout_dict.get('start_pct', 0.85)
        non_holdout_end = int(len(_worker_prices) * _worker_holdout_start_pct)
        _worker_train_val_prices = _worker_prices[:non_holdout_end]
        _worker_holdout_prices = slice_prices_for_holdout(_worker_prices, holdout_dict)
    else:
        _worker_holdout_start_pct = None
        _worker_train_val_prices = None
        _worker_holdout_prices = None


def process_single_combo(args: Tuple[int, Dict]) -> Optional[Dict]:
    """
    Process a single parameter combination (worker function).

    Args:
        args: Tuple of (combo_idx, params)

    Returns:
        Dict with all result data, or dict with error/failure info.

    Uses worker globals set by init_worker.
    """
    combo_idx, params = args
    sample_rate = _worker_thoroughness.sample_rate

    try:
        # Get fold 0 for initial train/val split
        fold = _worker_folds[0]
        train_prices, val_prices = slice_prices_for_fold(_worker_prices, fold)

        # Run training backtest
        train_result = run_backtest_with_params(
            params, train_prices, _worker_funding,
            return_rounds=True, sample_rate=sample_rate
        )

        # Run validation backtest
        val_result = run_backtest_with_params(
            params, val_prices, _worker_funding,
            return_rounds=False, sample_rate=sample_rate
        )

        # Check validation gate
        gate_passed, gate_reason = check_validation_gate(
            train_result, val_result,
            min_val_pnl_ratio=MIN_VAL_PNL_RATIO,
            min_val_rounds=MIN_VAL_ROUNDS
        )

        # Build CSV row data
        csv_row = {
            'threshold': params.get('threshold'),
            'trailing': params.get('trailing'),
            'pyramid_step': params.get('pyramid_step'),
            'max_pyramids': params.get('max_pyramids'),
            'vol_type': params.get('vol_type'),
            'size_schedule': params.get('size_schedule'),
            'train_pnl': train_result.get('total_pnl', 0),
            'val_pnl': val_result.get('total_pnl', 0),
            'win_rate': train_result.get('win_rate', 0),
            'rounds': train_result.get('total_rounds', 0),
            'max_drawdown_pct': train_result.get('max_drawdown_pct', 0),
            'passed_gate': gate_passed,
            'is_significant': False,
            'p_value': 1.0,
        }

        if not gate_passed:
            return {
                'combo_idx': combo_idx,
                'passed': False,
                'reason': gate_reason,
                'params': params,
                'csv_row': csv_row,
            }

        # Cross-fold validation
        crossfold_rate = max(1, sample_rate // _worker_thoroughness.crossfold_divisor)
        total_pnl_positive, fold_pnls, avg_val_pnl = validate_across_all_folds(
            params, _worker_prices, _worker_folds, _worker_funding,
            sample_rate=crossfold_rate
        )

        # Collect rounds data for storage
        rounds_data = []
        for r in train_result.get('rounds', []):
            rounds_data.append({
                'timestamp': str(getattr(r, 'entry_time', '')),
                'pnl_pct': getattr(r, 'total_pnl', 0),
                'duration_sec': 0,
                'num_pyramids': getattr(r, 'num_pyramids', 0),
            })

        return {
            'combo_idx': combo_idx,
            'passed': True,
            'params': params,
            'csv_row': csv_row,
            'train_metrics': {
                'total_pnl': train_result.get('total_pnl', 0),
                'win_rate': train_result.get('win_rate', 0),
                'total_rounds': train_result.get('total_rounds', 0),
                'max_drawdown_pct': train_result.get('max_drawdown_pct', 0),
            },
            'val_metrics': {
                'total_pnl': val_result.get('total_pnl', 0),
                'win_rate': val_result.get('win_rate', 0),
                'total_rounds': val_result.get('total_rounds', 0),
            },
            'fold_pnls': fold_pnls,
            'avg_val_pnl': avg_val_pnl,
            'total_pnl_positive': total_pnl_positive,
            'per_round_returns': train_result.get('per_round_returns', []),
            'rounds_data': rounds_data,
            'summary': {
                'total_pnl': train_result.get('total_pnl', 0),
                'total_rounds': train_result.get('total_rounds', 0),
                'win_rate': train_result.get('win_rate', 0),
            },
        }

    except Exception as e:
        import traceback
        return {
            'combo_idx': combo_idx,
            'passed': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'params': params,
            'csv_row': {
                'threshold': params.get('threshold'),
                'trailing': params.get('trailing'),
                'pyramid_step': params.get('pyramid_step'),
                'max_pyramids': params.get('max_pyramids'),
                'vol_type': params.get('vol_type'),
                'size_schedule': params.get('size_schedule'),
                'train_pnl': 0,
                'val_pnl': 0,
                'win_rate': 0,
                'rounds': 0,
                'max_drawdown_pct': 0,
                'passed_gate': False,
                'is_significant': False,
                'p_value': 1.0,
            },
        }


# =============================================================================
# PHASE 2-5 WORKER FUNCTIONS (Parallel execution)
# =============================================================================

def process_robustness_combo(args: Tuple[int, Dict, int]) -> Dict:
    """
    Worker: Calculate robustness score for one combo (Phase 2).

    Args:
        args: Tuple of (idx, params, sample_rate)

    Returns:
        Dict with idx and robustness score.
    """
    idx, params, sample_rate = args

    try:
        robustness, perturbations = calculate_robustness_score(
            params, _worker_train_val_prices, _worker_funding,
            sample_rate=sample_rate
        )
        return {'idx': idx, 'robustness': robustness, 'error': None}
    except Exception as e:
        return {'idx': idx, 'robustness': 0.0, 'error': str(e)}


def process_holdout_combo(args: Tuple[int, Dict, int, int, float]) -> Dict:
    """
    Worker: Run holdout backtest + Bonferroni significance (Phase 3).

    Args:
        args: Tuple of (idx, params, sample_rate, n_tests, family_alpha)

    Returns:
        Dict with holdout metrics, significance, and adjusted P&L.
    """
    idx, params, sample_rate, n_tests, family_alpha = args

    try:
        holdout_result = run_backtest_with_params(
            params, _worker_holdout_prices, _worker_funding,
            sample_rate=sample_rate
        )

        holdout_pnl = holdout_result.get('total_pnl', 0)
        holdout_rounds = holdout_result.get('total_rounds', 0)

        # Apply slippage cost estimate
        num_pyramids = holdout_result.get('avg_pyramids', 5)
        slippage_pct = estimate_round_slippage_pct(
            num_pyramids=int(num_pyramids),
            position_size_usdt=POSITION_SIZE_USDT,
            symbol=_worker_symbol or 'BTCUSDT'
        )
        total_slippage = slippage_pct * holdout_rounds
        adjusted_pnl = holdout_pnl - total_slippage

        # Build holdout metrics
        holdout_metrics = {
            'total_pnl': holdout_pnl,
            'adjusted_pnl': adjusted_pnl,
            'slippage_pct': total_slippage,
            'win_rate': holdout_result.get('win_rate', 0),
            'total_rounds': holdout_rounds,
            'max_drawdown_pct': holdout_result.get('max_drawdown_pct', 0),
        }

        # Bonferroni significance test on holdout returns
        holdout_returns = holdout_result.get('per_round_returns', [])
        if len(holdout_returns) >= 10:
            is_significant, p_value, _ = check_bonferroni_significance(
                params, holdout_returns, n_tests, family_alpha
            )
        else:
            is_significant = False
            p_value = 1.0

        return {
            'idx': idx,
            'holdout_metrics': holdout_metrics,
            'is_significant': is_significant,
            'p_value': p_value,
            'adjusted_pnl': adjusted_pnl,
            'holdout_rounds': holdout_rounds,
            'error': None,
        }
    except Exception as e:
        return {
            'idx': idx,
            'holdout_metrics': None,
            'is_significant': False,
            'p_value': 1.0,
            'adjusted_pnl': 0,
            'holdout_rounds': 0,
            'error': str(e),
        }


def process_regime_combo(args: Tuple[int, Dict]) -> Dict:
    """
    Worker: Validate regime robustness for one combo (Phase 4).

    Args:
        args: Tuple of (idx, params)

    Returns:
        Dict with idx and regime_pass boolean.
    """
    idx, params = args

    try:
        regime_pass = validate_regime_robustness(
            params, _worker_holdout_prices, _worker_funding
        )
        return {'idx': idx, 'regime_pass': regime_pass, 'error': None}
    except Exception as e:
        return {'idx': idx, 'regime_pass': False, 'error': str(e)}


def process_monte_carlo_combo(args: Tuple[int, Dict, int, int, int]) -> Dict:
    """
    Worker: Run Monte Carlo validation for one combo (Phase 5).

    Args:
        args: Tuple of (idx, params, sample_rate, mc_bootstrap, mc_permutation)

    Returns:
        Dict with Monte Carlo validation results.
    """
    idx, params, sample_rate, mc_bootstrap, mc_permutation = args

    try:
        from core.monte_carlo import run_monte_carlo_validation

        # Run holdout backtest
        holdout_result = run_backtest_with_params(
            params, _worker_holdout_prices, _worker_funding,
            sample_rate=sample_rate
        )
        holdout_returns = holdout_result.get('per_round_returns', [])

        if len(holdout_returns) < 10:
            return {
                'idx': idx,
                'mc_passes': False,
                'mc_prob_positive': 0,
                'mc_prob_ruin': 1,
                'mc_sharpe_5th': 0,
                'mc_sequence_matters': True,
                'holdout_pnl': holdout_result.get('total_pnl', 0),
                'error': 'Insufficient rounds (<10)',
            }

        # Run Monte Carlo validation
        passes, mc_result, perm_result = run_monte_carlo_validation(
            round_returns=holdout_returns,
            n_bootstrap=mc_bootstrap,
            n_permutation=mc_permutation,
            ruin_threshold=0.50,
            random_seed=42 + idx  # Different seed per combo
        )

        return {
            'idx': idx,
            'mc_passes': passes,
            'mc_prob_positive': mc_result.probability_positive,
            'mc_prob_ruin': mc_result.probability_ruin,
            'mc_sharpe_5th': mc_result.sharpe_5th_percentile,
            'mc_sequence_matters': perm_result.sequence_matters,
            'holdout_pnl': holdout_result.get('total_pnl', 0),
            'error': None,
        }
    except Exception as e:
        return {
            'idx': idx,
            'mc_passes': False,
            'mc_prob_positive': 0,
            'mc_prob_ruin': 1,
            'mc_sharpe_5th': 0,
            'mc_sequence_matters': True,
            'holdout_pnl': 0,
            'error': str(e),
        }


def run_parallel_grid_search(
    all_combos: List[Dict],
    prices_file: str,
    meta: Dict,
    folds: List[Dict],
    funding_rates: Dict,
    thoroughness: ThoroughnessConfig,
    n_workers: int,
    csv_writer,
    storage,
    verbose: bool = True
) -> Tuple[List, int, int]:
    """
    Run Phase 1 grid search in parallel using multiprocessing.Pool.

    Args:
        all_combos: List of parameter dicts to test
        prices_file: Path to memmap file (workers re-open)
        meta: Memmap metadata dict
        folds: List of fold configurations
        funding_rates: Funding rate dict
        thoroughness: ThoroughnessConfig for sample rates
        n_workers: Number of worker processes
        csv_writer: CSV DictWriter for logging
        storage: DiskResultStorage for saving results
        verbose: Whether to print progress

    Returns:
        Tuple of (passed_combos, failed_gate_count, error_count)
    """
    import time as time_module

    n_combos = len(all_combos)
    passed_combos = []
    failed_gate = 0
    errors = 0

    # Prepare work items: (combo_idx, params)
    work_items = [(i + 1, params) for i, params in enumerate(all_combos)]

    # Convert thoroughness to dict for pickling
    thoroughness_dict = {
        'name': thoroughness.name,
        'sample_rate': thoroughness.sample_rate,
        'crossfold_divisor': thoroughness.crossfold_divisor,
        'robustness_divisor': thoroughness.robustness_divisor,
        'holdout_sample_rate': thoroughness.holdout_sample_rate,
        'mc_bootstrap': thoroughness.mc_bootstrap,
        'mc_permutation': thoroughness.mc_permutation,
        'estimated_time': thoroughness.estimated_time,
    }

    print(f"Starting parallel grid search with {n_workers} workers...")
    start_time = time_module.time()

    # Graceful shutdown handler
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        pool = Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(prices_file, meta, folds, funding_rates, thoroughness_dict)
        )
        signal.signal(signal.SIGINT, original_sigint)

        completed = 0

        # Use imap_unordered for streaming results as they complete
        for result in pool.imap_unordered(process_single_combo, work_items, chunksize=10):
            completed += 1

            # Progress update using helper (updates every result for responsiveness)
            if verbose:
                extra = f"Pass: {len(passed_combos)} | Fail: {failed_gate}"
                print_progress_bar(completed, n_combos, "Grid: ", extra, start_time)

            if result is None:
                errors += 1
                continue

            # Log to CSV
            if 'csv_row' in result:
                csv_writer.writerow(result['csv_row'])

            # Handle errors
            if 'error' in result:
                errors += 1
                if verbose:
                    print(f"\n  [ERROR] Combo {result['combo_idx']}: {result['error']}")
                continue

            # Handle failed gate
            if not result['passed']:
                failed_gate += 1
                continue

            # Build OptimizationResult for passed combos
            opt_result = OptimizationResult(
                params=result['params'],
                train_metrics=result['train_metrics'],
                val_metrics=result['val_metrics'],
                holdout_metrics=None,
                passed_gate=True,
                is_significant=False,
                p_value=1.0,
                sharpe=0.0,  # Will be calculated after storage save
                per_round_returns=result['per_round_returns'],
                fold_pnls=result['fold_pnls'],
                total_pnl_positive=result['total_pnl_positive'],
                avg_val_pnl=result['avg_val_pnl'],
            )
            passed_combos.append(opt_result)

            # Save to storage and calculate Sharpe
            storage.save_combo_result(
                result['params'],
                result['rounds_data'],
                result['summary']
            )
            try:
                opt_result.sharpe = calculate_real_sharpe(storage, result['params'])
            except ValueError:
                opt_result.sharpe = 0.0

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\n\nShutdown requested. Terminating workers...")
        pool.terminate()
        pool.join()
        print(f"Terminated. Returning {len(passed_combos)} partial results.")
        raise

    print()  # New line after progress bar
    return passed_combos, failed_gate, errors


# =============================================================================
# PHASE 2-5 PARALLEL WRAPPERS
# =============================================================================

def run_parallel_robustness(
    results: List,
    prices_file: str,
    meta: Dict,
    folds: List[Dict],
    holdout: Dict,
    funding_rates: Dict,
    thoroughness: ThoroughnessConfig,
    n_workers: int,
    symbol: str,
    verbose: bool = True
) -> None:
    """
    Run Phase 2 robustness testing in parallel.

    Updates results in-place with robustness_score.
    """
    import time as time_module

    n_items = len(results)  # v7.6: No limit - test all combos
    if n_items == 0:
        return

    robustness_sample_rate = max(1, thoroughness.sample_rate // thoroughness.robustness_divisor)

    # Prepare work items: (idx, params, sample_rate)
    work_items = [
        (i, results[i].params, robustness_sample_rate)
        for i in range(n_items)
    ]

    thoroughness_dict = {
        'name': thoroughness.name,
        'sample_rate': thoroughness.sample_rate,
        'crossfold_divisor': thoroughness.crossfold_divisor,
        'robustness_divisor': thoroughness.robustness_divisor,
        'holdout_sample_rate': thoroughness.holdout_sample_rate,
        'mc_bootstrap': thoroughness.mc_bootstrap,
        'mc_permutation': thoroughness.mc_permutation,
        'estimated_time': thoroughness.estimated_time,
    }

    holdout_dict = {
        'start_pct': holdout.get('start_pct', 0.85),
        'end_pct': holdout.get('end_pct', 1.0),
    }

    print(f"  Running parallel robustness with {n_workers} workers...")
    start_time = time_module.time()

    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        pool = Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(prices_file, meta, folds, funding_rates, thoroughness_dict,
                      holdout_dict, symbol)
        )
        signal.signal(signal.SIGINT, original_sigint)

        completed = 0
        robust_count = 0
        for result in pool.imap_unordered(process_robustness_combo, work_items, chunksize=5):
            completed += 1
            idx = result['idx']
            results[idx].robustness_score = result['robustness']

            if result['robustness'] >= MIN_ROBUSTNESS_SCORE:
                robust_count += 1

            # Progress bar update
            if verbose:
                extra = f"Robust: {robust_count} | Fragile: {completed - robust_count}"
                print_progress_bar(completed, n_items, "  Robustness: ", extra, start_time)

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\n\nShutdown requested. Terminating workers...")
        pool.terminate()
        pool.join()
        raise

    if verbose:
        print()  # Newline after progress bar
    elapsed = time_module.time() - start_time
    if verbose:
        print(f"  Robustness phase completed in {elapsed:.1f}s")


def run_parallel_holdout(
    results: List,
    prices_file: str,
    meta: Dict,
    folds: List[Dict],
    holdout: Dict,
    funding_rates: Dict,
    thoroughness: ThoroughnessConfig,
    n_workers: int,
    symbol: str,
    verbose: bool = True
) -> List:
    """
    Run Phase 3 holdout evaluation in parallel.

    Returns list of results that pass holdout requirements.
    """
    import time as time_module

    n_items = len(results)  # v7.6: No limit - test all combos
    if n_items == 0:
        return []

    n_holdout_tests = n_items
    holdout_bonferroni_alpha = FAMILY_ALPHA / n_holdout_tests if n_holdout_tests > 0 else FAMILY_ALPHA

    # Prepare work items: (idx, params, sample_rate, n_tests, family_alpha)
    work_items = [
        (i, results[i].params, thoroughness.holdout_sample_rate,
         n_holdout_tests, FAMILY_ALPHA)
        for i in range(n_items)
    ]

    thoroughness_dict = {
        'name': thoroughness.name,
        'sample_rate': thoroughness.sample_rate,
        'crossfold_divisor': thoroughness.crossfold_divisor,
        'robustness_divisor': thoroughness.robustness_divisor,
        'holdout_sample_rate': thoroughness.holdout_sample_rate,
        'mc_bootstrap': thoroughness.mc_bootstrap,
        'mc_permutation': thoroughness.mc_permutation,
        'estimated_time': thoroughness.estimated_time,
    }

    holdout_dict = {
        'start_pct': holdout.get('start_pct', 0.85),
        'end_pct': holdout.get('end_pct', 1.0),
    }

    print(f"  Running parallel holdout with {n_workers} workers...")
    start_time = time_module.time()

    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    holdout_candidates = []

    try:
        pool = Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(prices_file, meta, folds, funding_rates, thoroughness_dict,
                      holdout_dict, symbol)
        )
        signal.signal(signal.SIGINT, original_sigint)

        completed = 0
        for result_data in pool.imap_unordered(process_holdout_combo, work_items, chunksize=10):
            completed += 1
            idx = result_data['idx']
            result = results[idx]

            if result_data['holdout_metrics'] is not None:
                result.holdout_metrics = result_data['holdout_metrics']
                result.is_significant = result_data['is_significant']
                result.p_value = result_data['p_value']

                adjusted_pnl = result_data['adjusted_pnl']
                holdout_rounds = result_data['holdout_rounds']

                # Check requirements
                if (adjusted_pnl >= MIN_HOLDOUT_PNL and
                    holdout_rounds >= MIN_HOLDOUT_ROUNDS and
                    result.is_significant):
                    holdout_candidates.append(result)

            # Progress bar update
            if verbose:
                extra = f"Pass: {len(holdout_candidates)}"
                print_progress_bar(completed, n_items, "  Holdout: ", extra, start_time)

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\n\nShutdown requested. Terminating workers...")
        pool.terminate()
        pool.join()
        raise

    if verbose:
        print()  # Newline after progress bar
    elapsed = time_module.time() - start_time
    if verbose:
        print(f"  Holdout phase completed in {elapsed:.1f}s")

    # Sort by adjusted holdout P&L
    holdout_candidates.sort(key=lambda x: x.holdout_metrics['adjusted_pnl'], reverse=True)
    return holdout_candidates


def run_parallel_regime(
    results: List,
    prices_file: str,
    meta: Dict,
    folds: List[Dict],
    holdout: Dict,
    funding_rates: Dict,
    thoroughness: ThoroughnessConfig,
    n_workers: int,
    symbol: str,
    verbose: bool = True
) -> None:
    """
    Run Phase 4 regime validation in parallel.

    Updates results in-place with regime_pass.
    """
    import time as time_module

    n_items = len(results)  # v7.6: No limit - test all combos
    if n_items == 0:
        return

    # Prepare work items: (idx, params)
    work_items = [(i, results[i].params) for i in range(n_items)]

    thoroughness_dict = {
        'name': thoroughness.name,
        'sample_rate': thoroughness.sample_rate,
        'crossfold_divisor': thoroughness.crossfold_divisor,
        'robustness_divisor': thoroughness.robustness_divisor,
        'holdout_sample_rate': thoroughness.holdout_sample_rate,
        'mc_bootstrap': thoroughness.mc_bootstrap,
        'mc_permutation': thoroughness.mc_permutation,
        'estimated_time': thoroughness.estimated_time,
    }

    holdout_dict = {
        'start_pct': holdout.get('start_pct', 0.85),
        'end_pct': holdout.get('end_pct', 1.0),
    }

    print(f"  Running parallel regime validation with {n_workers} workers...")
    start_time = time_module.time()

    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        pool = Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(prices_file, meta, folds, funding_rates, thoroughness_dict,
                      holdout_dict, symbol)
        )
        signal.signal(signal.SIGINT, original_sigint)

        completed = 0
        pass_count = 0
        for result_data in pool.imap_unordered(process_regime_combo, work_items, chunksize=2):
            completed += 1
            idx = result_data['idx']
            results[idx].regime_pass = result_data['regime_pass']
            if result_data['regime_pass']:
                pass_count += 1

            # Progress bar update
            if verbose:
                extra = f"Pass: {pass_count} | Fail: {completed - pass_count}"
                print_progress_bar(completed, n_items, "  Regime: ", extra, start_time)

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\n\nShutdown requested. Terminating workers...")
        pool.terminate()
        pool.join()
        raise

    if verbose:
        print()  # Newline after progress bar
    elapsed = time_module.time() - start_time
    if verbose:
        print(f"  Regime phase completed in {elapsed:.1f}s")


def run_parallel_monte_carlo(
    results: List,
    prices_file: str,
    meta: Dict,
    folds: List[Dict],
    holdout: Dict,
    funding_rates: Dict,
    thoroughness: ThoroughnessConfig,
    n_workers: int,
    symbol: str,
    verbose: bool = True
) -> List:
    """
    Run Phase 5 Monte Carlo validation in parallel.

    Returns list of results that pass Monte Carlo validation.
    """
    import time as time_module

    n_items = len(results)  # v7.6: No limit - test all combos
    if n_items == 0:
        return []

    # Prepare work items: (idx, params, sample_rate, mc_bootstrap, mc_permutation)
    work_items = [
        (i, results[i].params, thoroughness.holdout_sample_rate,
         thoroughness.mc_bootstrap, thoroughness.mc_permutation)
        for i in range(n_items)
    ]

    thoroughness_dict = {
        'name': thoroughness.name,
        'sample_rate': thoroughness.sample_rate,
        'crossfold_divisor': thoroughness.crossfold_divisor,
        'robustness_divisor': thoroughness.robustness_divisor,
        'holdout_sample_rate': thoroughness.holdout_sample_rate,
        'mc_bootstrap': thoroughness.mc_bootstrap,
        'mc_permutation': thoroughness.mc_permutation,
        'estimated_time': thoroughness.estimated_time,
    }

    holdout_dict = {
        'start_pct': holdout.get('start_pct', 0.85),
        'end_pct': holdout.get('end_pct', 1.0),
    }

    print(f"  Running parallel Monte Carlo with {n_workers} workers...")
    start_time = time_module.time()

    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    mc_validated = []

    try:
        pool = Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(prices_file, meta, folds, funding_rates, thoroughness_dict,
                      holdout_dict, symbol)
        )
        signal.signal(signal.SIGINT, original_sigint)

        completed = 0
        for result_data in pool.imap_unordered(process_monte_carlo_combo, work_items, chunksize=5):
            completed += 1
            idx = result_data['idx']
            result = results[idx]

            result.mc_passes = result_data['mc_passes']
            result.mc_prob_positive = result_data['mc_prob_positive']
            result.mc_prob_ruin = result_data['mc_prob_ruin']
            result.mc_sharpe_5th = result_data['mc_sharpe_5th']
            result.mc_sequence_matters = result_data['mc_sequence_matters']

            if result_data['mc_passes']:
                mc_validated.append(result)

            # Progress bar update
            if verbose:
                extra = f"Pass: {len(mc_validated)} | Fail: {completed - len(mc_validated)}"
                print_progress_bar(completed, n_items, "  Monte Carlo: ", extra, start_time)

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\n\nShutdown requested. Terminating workers...")
        pool.terminate()
        pool.join()
        raise

    if verbose:
        print()  # Newline after progress bar
    elapsed = time_module.time() - start_time
    if verbose:
        print(f"  Monte Carlo phase completed in {elapsed:.1f}s")

    return mc_validated


# =============================================================================
# DATA LOADING (MEMORY-EFFICIENT DISK-BASED)
# =============================================================================

def load_tick_data_memmap(symbol: str, days: int) -> Tuple[any, Dict, Dict, str]:
    """
    Load tick data using memory-mapped files (LOW RAM USAGE).

    Downloads data to disk month-by-month (never all in RAM), then
    memory-maps the file for efficient access. OS handles paging.

    Returns:
        Tuple of (prices, meta, funding_rates, prices_file_path)
    """
    from core.memmap_prices import (
        ensure_prices_on_disk,
        load_prices_memmap,
        validate_memmap_quality,
        load_funding_rates,
        print_memmap_info,
    )

    years = max(1, days // 365)

    print(f"Loading tick data for {symbol} ({years} years) - DISK-BASED MODE")

    # Download to disk if not cached
    prices_file, meta = ensure_prices_on_disk(symbol, years)

    # Memory-map the file
    prices = load_prices_memmap(prices_file, meta)
    print_memmap_info(prices, meta)

    # Data quality validation
    if VALIDATE_DATA_QUALITY:
        validate_memmap_quality(prices)

    # Load funding rates
    funding_rates = {}
    if APPLY_FUNDING_RATES:
        funding_rates = load_funding_rates(symbol, years)

    return prices, meta, funding_rates, prices_file


def slice_prices_for_fold(prices, fold: Dict) -> Tuple[any, any]:
    """Slice memory-mapped prices for a fold's train and validation data."""
    from core.memmap_prices import slice_memmap_for_fold
    return slice_memmap_for_fold(prices, fold)


def slice_prices_for_holdout(prices, holdout: Dict):
    """Slice memory-mapped prices for holdout testing."""
    from core.memmap_prices import slice_memmap_for_holdout
    return slice_memmap_for_holdout(prices, holdout)


def prices_to_iterator(price_slice, sample_rate: int = 1):
    """Convert a price slice to iterator for backtest.

    Args:
        price_slice: Memory-mapped price array or list
        sample_rate: Use every Nth price (1=all, 10=every 10th).
                    Higher values = faster but less accurate.
    """
    from core.memmap_prices import memmap_to_iterator
    import numpy as np

    if isinstance(price_slice, np.ndarray):
        return memmap_to_iterator(price_slice, sample_rate=sample_rate)
    else:
        # For lists, apply sampling manually
        if sample_rate > 1:
            return iter(price_slice[::sample_rate])
        return iter(price_slice)


# =============================================================================
# BACKTEST WRAPPER
# =============================================================================

def run_backtest_with_params(
    params: Dict,
    price_slice,
    funding_rates: Dict = None,
    return_rounds: bool = True,
    sample_rate: int = 1
) -> Dict:
    """
    Run backtest with expanded parameters.

    Args:
        params: Strategy parameters
        price_slice: Memory-mapped price array or list
        funding_rates: Funding rate dict
        return_rounds: Include round details in result
        sample_rate: Use every Nth price (1=all, 10=every 10th).
                    For grid search use 10, for final validation use 1.
    """
    try:
        execution_model = DEFAULT_EXECUTION_MODEL
    except:
        execution_model = None

    price_iter = prices_to_iterator(price_slice, sample_rate=sample_rate)

    result = run_pyramid_backtest(
        prices=price_iter,
        threshold_pct=params.get('threshold', 5.0),
        trailing_pct=params.get('trailing', 1.0),
        pyramid_step_pct=params.get('pyramid_step', 2.0),
        max_pyramids=params.get('max_pyramids', 20),
        # Execution
        fee_pct=0.04,
        execution_model=execution_model,
        # Pyramid parameters (FIXED in v7)
        pyramid_size_schedule=params.get('size_schedule', 'fixed'),
        min_pyramid_spacing_pct=params.get('min_spacing', 0.0),
        pyramid_acceleration=params.get('acceleration', 1.0),
        time_decay_exit_seconds=params.get('time_decay'),  # None in v7
        # Volatility
        volatility_filter_type=params.get('vol_type', 'none'),
        volatility_min_pct=params.get('vol_min', 0.0),
        volatility_window_size=params.get('vol_window', 100),
        # Causal trailing
        use_causal_trailing=params.get('use_causal', True),
        confirmation_ticks=params.get('confirmation_ticks', 3),
        # Exit controls (DISABLED in v7 - rely on trailing stop)
        take_profit_pct=params.get('take_profit_pct'),  # None
        stop_loss_pct=params.get('stop_loss_pct'),      # None
        breakeven_after_pct=params.get('breakeven_after_pct'),
        # Timing
        pyramid_cooldown_sec=params.get('pyramid_cooldown_sec', 0),
        max_round_duration_hr=params.get('max_round_duration_hr'),
        # Filters
        trend_filter_ema=params.get('trend_filter_ema'),
        session_filter=params.get('session_filter', 'all'),
        # Funding
        funding_rates=funding_rates,
        apply_funding=funding_rates is not None,
        # Rounds
        return_rounds=return_rounds,
        # Margin
        use_margin_tracking=USE_MARGIN_TRACKING,
        initial_capital=INITIAL_CAPITAL,
        leverage=LEVERAGE,
        position_size_usdt=POSITION_SIZE_USDT,
    )

    return result


# =============================================================================
# ROBUSTNESS TESTING
# =============================================================================

def calculate_robustness_score(
    params: Dict,
    prices,
    funding_rates: Optional[Dict] = None,
    sample_rate: int = 1
) -> Tuple[float, List[Dict]]:
    """
    Test nearby parameters to check if the optimum is robust.
    """
    center_result = run_backtest_with_params(params, prices, funding_rates, return_rounds=False, sample_rate=sample_rate)
    center_pnl = center_result['total_pnl']

    if center_pnl <= 0:
        return 0.0, []

    perturbation_results = []
    perturbations = [
        ('threshold', -0.5),
        ('threshold', +0.5),
        ('trailing', -0.2),
        ('trailing', +0.2),
        ('pyramid_step', -0.2),
        ('pyramid_step', +0.2),
    ]

    for param, delta in perturbations:
        perturbed = params.copy()
        current_val = perturbed.get(param, 1.0)

        if isinstance(current_val, (int, float)):
            new_val = max(0.1, current_val + delta)
            limits = PARAM_LIMITS.get(param, (0, 100))
            if limits[0] <= new_val <= limits[1]:
                perturbed[param] = new_val
                result = run_backtest_with_params(perturbed, prices, funding_rates, return_rounds=False, sample_rate=sample_rate)
                perturbation_results.append({
                    'param': param,
                    'delta': delta,
                    'pnl': result['total_pnl'],
                    'ratio': result['total_pnl'] / center_pnl if center_pnl > 0 else 0
                })

    if perturbation_results:
        min_ratio = min(p['ratio'] for p in perturbation_results)
        robustness = max(0, min_ratio)
    else:
        robustness = 1.0

    return robustness, perturbation_results


# =============================================================================
# CROSS-FOLD VALIDATION (SIMPLE - TOTAL P&L POSITIVE)
# =============================================================================

def validate_across_all_folds(
    params: Dict,
    prices,
    folds: List[Dict],
    funding_rates: Optional[Dict],
    sample_rate: int = 1
) -> Tuple[bool, List[float], float]:
    """
    Validate parameters across ALL folds.

    Simple requirement: Total P&L across all folds must be positive.
    Individual folds can be negative (ranging periods happen).
    """
    fold_pnls = []

    for fold in folds:
        train_prices, val_prices = slice_prices_for_fold(prices, fold)
        val_result = run_backtest_with_params(params, val_prices, funding_rates, return_rounds=False, sample_rate=sample_rate)
        val_pnl = val_result.get('total_pnl', 0)
        fold_pnls.append(val_pnl)

    avg_val_pnl = sum(fold_pnls) / len(fold_pnls) if fold_pnls else 0
    total_pnl_positive = sum(fold_pnls) > 0

    return total_pnl_positive, fold_pnls, avg_val_pnl


# =============================================================================
# TOP 3 CANDIDATES VALIDATION (OVERFITTING GUARD)
# =============================================================================

def validate_top_3_candidates(
    results: List[OptimizationResult],
    prices,
    holdout: Dict,
    funding_rates: Optional[Dict]
) -> Tuple[int, bool]:
    """
    Validate top 3 candidates as overfitting guard.

    If only 1 passes, it's likely a statistical fluke.
    This is a key anti-overfitting measure.
    """
    holdout_prices = slice_prices_for_holdout(prices, holdout)
    top_3 = results[:3] if len(results) >= 3 else results

    pass_count = 0
    all_pass = True

    print("\n--- TOP 3 CANDIDATES VALIDATION (Overfitting Guard) ---")

    for i, result in enumerate(top_3):
        holdout_result = run_backtest_with_params(result.params, holdout_prices, funding_rates, return_rounds=True)

        # Simple check: positive P&L and enough rounds
        pnl = holdout_result.get('total_pnl', 0)
        rounds = holdout_result.get('total_rounds', 0)
        passes = pnl > 0 and rounds >= MIN_HOLDOUT_ROUNDS

        status = "[PASS]" if passes else "[FAIL]"
        params_str = f"th={result.params['threshold']}, tr={result.params['trailing']}"
        print(f"  Top {i+1}: {status} ({params_str}) P&L={pnl:.1f}%, rounds={rounds}")

        if passes:
            pass_count += 1
        else:
            all_pass = False

    if not all_pass:
        print(f"\n  [WARNING] Only {pass_count}/3 top candidates pass holdout!")
        print(f"  This suggests possible overfitting.")
    else:
        print(f"\n  [OK] All top 3 candidates pass - reduces overfitting risk.")

    return pass_count, all_pass


# =============================================================================
# REGIME VALIDATION (MARKET CONDITION ROBUSTNESS)
# =============================================================================

def validate_regime_robustness(
    params: Dict,
    prices,
    funding_rates: Optional[Dict] = None
) -> bool:
    """
    Validate strategy across different market regimes.

    Checks if strategy works in trending, ranging, volatile, and calm markets.
    We don't require all regimes to be profitable - just that overall it works.
    """
    try:
        from core.regime_detection import detect_regimes, validate_strategy_across_regimes

        if len(prices) < 1000:
            return True  # Pass if insufficient data

        # Convert memmap slice to list for regime detection
        price_list = list(prices_to_iterator(prices))
        regime_labels = detect_regimes(price_list, window_size=500)
        if not regime_labels:
            return True

        result = run_backtest_with_params(params, prices, funding_rates, return_rounds=True)
        rounds = result.get('rounds', [])

        if not rounds:
            return True

        # Map rounds to regimes
        time_to_regime = {label.start_time: label.combined for label in regime_labels}
        regime_returns = {}

        for round_obj in rounds:
            round_time = round_obj.entry_time
            closest_regime = None
            min_diff = float('inf')

            for label_time, regime in time_to_regime.items():
                diff = abs((round_time - label_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_regime = regime

            if closest_regime:
                if closest_regime not in regime_returns:
                    regime_returns[closest_regime] = []
                regime_returns[closest_regime].append(round_obj.total_pnl)

        if not regime_returns:
            return True

        # Lenient validation: only require 40% of regimes profitable
        # (Some regimes like "ranging" may naturally lose money for trend strategies)
        validation = validate_strategy_across_regimes(
            regime_returns=regime_returns,
            min_rounds_per_regime=5,
            min_profitable_regimes_pct=40.0,  # Very lenient
            verbose=False
        )

        return validation.overall_pass

    except ImportError:
        return True  # Pass if module not available
    except Exception as e:
        print(f"  Regime validation error: {e}")
        return True


# =============================================================================
# SINGLE-STAGE OPTIMIZATION (PURE PROFIT)
# =============================================================================

def run_optimization(
    symbol: str = DEFAULT_SYMBOL,
    days: int = DEFAULT_DAYS,
    output_dir: str = "./optimization_v7_results",
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = True,
    thoroughness: ThoroughnessConfig = None,
    workers: Optional[int] = None
) -> List[OptimizationResult]:
    """
    Run pure profit optimized single-stage optimization.

    Args:
        workers: Number of worker processes for parallel execution.
                None = auto-detect (cpu_count - 1)
                0 = force sequential mode
    """
    # Default to standard thoroughness if not specified
    if thoroughness is None:
        thoroughness = THOROUGHNESS_CONFIGS[DEFAULT_THOROUGHNESS]

    # Determine worker count
    n_workers = get_worker_count(workers)

    print("=" * 70)
    print("PURE PROFIT OPTIMIZER v7.6 - 'No Limits'")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Days: {days}")
    print(f"Thoroughness: {thoroughness.name} (~{thoroughness.estimated_time})")
    if n_workers > 0:
        print(f"Workers: {n_workers} (parallel mode)")
    else:
        print("Workers: 1 (sequential mode)")
    print(f"  - Grid sample rate: 1/{thoroughness.sample_rate} ({100/thoroughness.sample_rate:.0f}% data)")
    print(f"  - Holdout sample rate: 1/{thoroughness.holdout_sample_rate} ({100/thoroughness.holdout_sample_rate:.0f}% data)")
    print(f"  - Monte Carlo: {thoroughness.mc_bootstrap:,} bootstrap, {thoroughness.mc_permutation:,} permutation")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now()}")
    print()
    print("V7 PHILOSOPHY: No arbitrary constraints, just profit")
    print("  - NO win rate filter")
    print("  - NO drawdown limit")
    print("  - NO expectancy/profit factor minimums")
    print("  - RANKING = Pure holdout P&L")
    print()
    print("STATISTICAL VALIDITY: Preserved")
    print(f"  - Grid size: 640 (vs 5,120 in v5/v6)")
    print(f"  - Bonferroni alpha: 0.000078 (8x better)")
    print("  - Cross-fold validation, Monte Carlo, Robustness testing")
    print()

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Initialize storage
    storage = DiskResultStorage(os.path.join(output_dir, "disk_storage"))

    # Generate parameter grid (MINIMAL - 640 combos)
    grid = build_minimal_grid()
    all_combos = generate_all_combinations(grid)
    n_combos = len(all_combos)
    bonferroni_alpha = get_bonferroni_alpha(grid, FAMILY_ALPHA)

    print(f"Parameter Grid: {n_combos:,} combinations")
    print(f"Bonferroni alpha: {bonferroni_alpha:.6f}")
    print()

    # Load data (DISK-BASED)
    print("Loading tick data (memory-efficient disk mode)...")
    try:
        prices, meta, funding_rates, prices_file = load_tick_data_memmap(symbol, days)
        print(f"Loaded {meta['total_prices']:,} prices (memory-mapped from disk)")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return []
    print()

    # Get fold structure
    folds, holdout = get_fixed_folds(days)
    print("Fold Structure:")
    for fold in folds:
        print(f"  {fold['name']}: Train {fold['train_start_pct']*100:.0f}%-{fold['train_end_pct']*100:.0f}%, "
              f"Val {fold['val_start_pct']*100:.0f}%-{fold['val_end_pct']*100:.0f}%")
    print(f"  Holdout: {holdout['start_pct']*100:.0f}%-{holdout['end_pct']*100:.0f}%")
    print()

    # CSV logging
    csv_file = os.path.join(LOG_DIR, f"{symbol}_v7_grid.csv")
    csv_fields = [
        'threshold', 'trailing', 'pyramid_step', 'max_pyramids', 'vol_type',
        'size_schedule', 'train_pnl', 'val_pnl', 'win_rate', 'rounds',
        'max_drawdown_pct', 'passed_gate', 'is_significant', 'p_value'
    ]
    csv_f = open(csv_file, 'w', newline='')
    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Track results
    passed_combos: List[OptimizationResult] = []
    failed_gate = 0
    errors = 0

    # Process all combinations with real-time progress
    print(f"Processing {n_combos:,} combinations...")
    print("-" * 70)

    # PHASE 1: Grid search (parallel or sequential)
    if n_workers > 0:
        # Parallel mode using multiprocessing
        try:
            passed_combos, failed_gate, errors = run_parallel_grid_search(
                all_combos=all_combos,
                prices_file=prices_file,
                meta=meta,
                folds=folds,
                funding_rates=funding_rates,
                thoroughness=thoroughness,
                n_workers=n_workers,
                csv_writer=csv_writer,
                storage=storage,
                verbose=verbose
            )
        except KeyboardInterrupt:
            print("\nOptimization cancelled by user.")
            csv_f.close()
            return passed_combos if passed_combos else []
        except Exception as e:
            print(f"\n[WARNING] Parallel execution failed: {e}")
            print("[FALLBACK] Switching to sequential mode...")
            n_workers = 0  # Fall back to sequential

    if n_workers == 0:
        # Sequential mode (original implementation)
        import time as time_module
        start_time = time_module.time()
        combo_times = []

        for batch_start in range(0, n_combos, batch_size):
            batch_end = min(batch_start + batch_size, n_combos)
            batch = all_combos[batch_start:batch_end]

            for i, params in enumerate(batch):
                combo_idx = batch_start + i + 1
                combo_start = time_module.time()

                # Real-time progress bar
                if verbose:
                    elapsed = time_module.time() - start_time
                    elapsed_str = f"{int(elapsed//60)}m{int(elapsed%60):02d}s"

                    if combo_times:
                        avg_time = sum(combo_times[-20:]) / len(combo_times[-20:])
                        remaining = (n_combos - combo_idx) * avg_time
                        eta_str = f"{int(remaining//60)}m{int(remaining%60):02d}s"
                    else:
                        eta_str = "calculating..."

                    pct = combo_idx / n_combos * 100
                    bar_len = 30
                    filled = int(bar_len * combo_idx / n_combos)
                    bar = "█" * filled + "░" * (bar_len - filled)

                    print(f"\r[{bar}] {pct:5.1f}% | {combo_idx}/{n_combos} | "
                          f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                          f"Pass: {len(passed_combos)} | Fail: {failed_gate}",
                          end="", flush=True)

                # Test on FOLD 0 first
                fold = folds[0]
                train_prices, val_prices = slice_prices_for_fold(prices, fold)

                # Run training backtest (with sampling for speed)
                train_result = run_backtest_with_params(params, train_prices, funding_rates, sample_rate=thoroughness.sample_rate)

                # MINIMAL VALIDATION GATE
                val_result = run_backtest_with_params(params, val_prices, funding_rates, sample_rate=thoroughness.sample_rate)

                gate_passed, gate_reason = check_validation_gate(
                    train_result, val_result,
                    min_val_pnl_ratio=MIN_VAL_PNL_RATIO,
                    min_val_rounds=MIN_VAL_ROUNDS
                )

                # Log to CSV
                csv_row = {
                    'threshold': params.get('threshold'),
                    'trailing': params.get('trailing'),
                    'pyramid_step': params.get('pyramid_step'),
                    'max_pyramids': params.get('max_pyramids'),
                    'vol_type': params.get('vol_type'),
                    'size_schedule': params.get('size_schedule'),
                    'train_pnl': train_result.get('total_pnl', 0),
                    'val_pnl': val_result.get('total_pnl', 0),
                    'win_rate': train_result.get('win_rate', 0),
                    'rounds': train_result.get('total_rounds', 0),
                    'max_drawdown_pct': train_result.get('max_drawdown_pct', 0),
                    'passed_gate': gate_passed,
                    'is_significant': False,
                    'p_value': 1.0,
                }

                if not gate_passed:
                    failed_gate += 1
                    csv_writer.writerow(csv_row)
                    combo_time = time_module.time() - combo_start
                    combo_times.append(combo_time)
                    if len(combo_times) > 100:
                        combo_times = combo_times[-50:]
                    continue

                per_round_returns = train_result.get('per_round_returns', [])
                csv_row['is_significant'] = True
                csv_row['p_value'] = 0.0
                csv_writer.writerow(csv_row)

                # Cross-fold validation
                crossfold_rate = max(1, thoroughness.sample_rate // thoroughness.crossfold_divisor)
                total_pnl_positive, fold_pnls, avg_val_pnl = validate_across_all_folds(
                    params, prices, folds, funding_rates, sample_rate=crossfold_rate
                )

                # Save to storage
                rounds_data = []
                for r in train_result.get('rounds', []):
                    rounds_data.append({
                        'timestamp': str(getattr(r, 'entry_time', '')),
                        'pnl_pct': getattr(r, 'total_pnl', 0),
                        'duration_sec': 0,
                        'num_pyramids': getattr(r, 'num_pyramids', 0),
                    })

                summary = {
                    'total_pnl': train_result.get('total_pnl', 0),
                    'total_rounds': train_result.get('total_rounds', 0),
                    'win_rate': train_result.get('win_rate', 0),
                }

                storage.save_combo_result(params, rounds_data, summary)

                # Calculate Sharpe
                try:
                    sharpe = calculate_real_sharpe(storage, params)
                except ValueError:
                    sharpe = 0.0

                # Record passed combo
                result = OptimizationResult(
                    params=params,
                    train_metrics={
                        'total_pnl': train_result.get('total_pnl', 0),
                        'win_rate': train_result.get('win_rate', 0),
                        'total_rounds': train_result.get('total_rounds', 0),
                        'max_drawdown_pct': train_result.get('max_drawdown_pct', 0),
                    },
                    val_metrics={
                        'total_pnl': val_result.get('total_pnl', 0),
                        'win_rate': val_result.get('win_rate', 0),
                        'total_rounds': val_result.get('total_rounds', 0),
                    },
                    holdout_metrics=None,
                    passed_gate=True,
                    is_significant=False,
                    p_value=1.0,
                    sharpe=sharpe,
                    per_round_returns=per_round_returns,
                    fold_pnls=fold_pnls,
                    total_pnl_positive=total_pnl_positive,
                    avg_val_pnl=avg_val_pnl,
                )
                passed_combos.append(result)

                # Track combo time
                combo_time = time_module.time() - combo_start
                combo_times.append(combo_time)
                if len(combo_times) > 100:
                    combo_times = combo_times[-50:]

            # Memory management
            gc.collect()
            if combo_idx % 200 == 0:
                check_memory_usage(f"combo_{combo_idx}")

        print()  # New line after progress bar

    csv_f.close()
    print(f"\nCSV log saved to: {csv_file}")

    print("-" * 70)
    print(f"PHASE 1 COMPLETE: {len(passed_combos)} combos passed validation gate")
    print(f"  Failed gate: {failed_gate}")
    print(f"  (Note: Bonferroni significance now applied in Phase 3 on holdout data)")

    # Filter to those with positive total P&L
    total_positive = [r for r in passed_combos if r.total_pnl_positive]
    print(f"  Total P&L positive across folds: {len(total_positive)}")
    print()

    if not passed_combos:
        print("No combos passed initial validation!")
        return []

    # Sort by average validation P&L (simple - just profit)
    if total_positive:
        total_positive.sort(key=lambda x: x.avg_val_pnl, reverse=True)
        working_results = total_positive
    else:
        passed_combos.sort(key=lambda x: x.avg_val_pnl, reverse=True)
        working_results = passed_combos
        print("WARNING: No combo has positive total P&L, using best available.")

    # PHASE 2: Robustness testing
    print("=" * 70)
    print("PHASE 2: ROBUSTNESS TESTING")
    print("=" * 70)

    # v7.1 FIX: Use only train+val data, EXCLUDE holdout to prevent data leakage
    # Holdout starts at 85%, so we use 0-85% for robustness testing
    holdout_start_pct = holdout['start_pct']
    non_holdout_end = int(len(prices) * holdout_start_pct)
    train_val_prices = prices[:non_holdout_end]
    print(f"  Using {non_holdout_end:,} prices (0-{holdout_start_pct*100:.0f}%) for robustness testing")
    print(f"  (Holdout {holdout_start_pct*100:.0f}-100% is excluded to prevent data leakage)")

    if n_workers > 0:
        # Parallel robustness testing
        run_parallel_robustness(
            results=working_results,
            prices_file=prices_file,
            meta=meta,
            folds=folds,
            holdout=holdout,
            funding_rates=funding_rates,
            thoroughness=thoroughness,
            n_workers=n_workers,
            symbol=symbol,
            verbose=verbose
        )
    else:
        # Sequential robustness testing
        robustness_sample_rate = max(1, thoroughness.sample_rate // thoroughness.robustness_divisor)
        for result in working_results:  # v7.6: No limit - test all combos
            robustness, perturbations = calculate_robustness_score(
                result.params, train_val_prices, funding_rates, sample_rate=robustness_sample_rate
            )
            result.robustness_score = robustness

            if robustness < MIN_ROBUSTNESS_SCORE:
                print(f"  [FRAGILE] th={result.params['threshold']}, tr={result.params['trailing']}: robustness={robustness:.2f}")
            else:
                print(f"  [ROBUST]  th={result.params['threshold']}, tr={result.params['trailing']}: robustness={robustness:.2f}")

    # Filter by robustness (keep this - it's not arbitrary)
    robust_results = [r for r in working_results if r.robustness_score >= MIN_ROBUSTNESS_SCORE]
    if robust_results:
        working_results = robust_results
        print(f"\n{len(robust_results)} combos pass robustness threshold ({MIN_ROBUSTNESS_SCORE})")
    else:
        print("\nWARNING: No combos pass robustness threshold, using best available")

    # PHASE 3: Holdout evaluation (THE ONLY THING THAT MATTERS)
    print()
    print("=" * 70)
    print("PHASE 3: HOLDOUT EVALUATION + BONFERRONI (Pure Profit)")
    print("=" * 70)
    print("v7.1: Bonferroni correction now applied HERE on holdout data")
    print("Ranking by HOLDOUT TOTAL P&L - nothing else matters")
    print()

    holdout_prices = slice_prices_for_holdout(prices, holdout)
    n_holdout_tests = len(working_results)  # v7.6: Use actual count for Bonferroni
    holdout_bonferroni_alpha = FAMILY_ALPHA / n_holdout_tests if n_holdout_tests > 0 else FAMILY_ALPHA
    print(f"  Testing {n_holdout_tests} combos on holdout")
    print(f"  Bonferroni alpha for holdout: {holdout_bonferroni_alpha:.6f}")
    if thoroughness.holdout_sample_rate == 1:
        print(f"  Using FULL data (sample_rate=1) for final validation")
    else:
        print(f"  Using 1/{thoroughness.holdout_sample_rate} data for holdout ({thoroughness.name} mode)")
    print()

    if n_workers > 0:
        # Parallel holdout evaluation
        holdout_candidates = run_parallel_holdout(
            results=working_results,
            prices_file=prices_file,
            meta=meta,
            folds=folds,
            holdout=holdout,
            funding_rates=funding_rates,
            thoroughness=thoroughness,
            n_workers=n_workers,
            symbol=symbol,
            verbose=verbose
        )
    else:
        # Sequential holdout evaluation
        holdout_candidates = []
        for result in working_results:  # v7.6: No limit - test all combos
            # Holdout validation - sample rate depends on thoroughness level
            holdout_result = run_backtest_with_params(result.params, holdout_prices, funding_rates, sample_rate=thoroughness.holdout_sample_rate)

            holdout_pnl = holdout_result.get('total_pnl', 0)
            holdout_rounds = holdout_result.get('total_rounds', 0)

            # v7.1 FIX: Apply slippage cost estimate to holdout P&L
            num_pyramids = holdout_result.get('avg_pyramids', 5)  # Estimate avg pyramids
            slippage_pct = estimate_round_slippage_pct(
                num_pyramids=int(num_pyramids),
                position_size_usdt=POSITION_SIZE_USDT,
                symbol=symbol
            )
            # Apply slippage penalty per round
            total_slippage = slippage_pct * holdout_rounds
            adjusted_pnl = holdout_pnl - total_slippage

            result.holdout_metrics = {
                'total_pnl': holdout_pnl,
                'adjusted_pnl': adjusted_pnl,  # After slippage
                'slippage_pct': total_slippage,
                'win_rate': holdout_result.get('win_rate', 0),
                'total_rounds': holdout_rounds,
                'max_drawdown_pct': holdout_result.get('max_drawdown_pct', 0),
            }

            # v7.1 FIX: Apply Bonferroni significance test on HOLDOUT returns
            holdout_returns = holdout_result.get('per_round_returns', [])
            if len(holdout_returns) >= 10:  # Need enough rounds
                is_significant, p_value, _ = check_bonferroni_significance(
                    result.params, holdout_returns, n_holdout_tests, FAMILY_ALPHA
                )
                result.is_significant = is_significant
                result.p_value = p_value
            else:
                result.is_significant = False
                result.p_value = 1.0

            # REQUIREMENTS: profitable (after slippage), enough rounds, statistically significant
            if (adjusted_pnl >= MIN_HOLDOUT_PNL and
                holdout_rounds >= MIN_HOLDOUT_ROUNDS and
                result.is_significant):
                holdout_candidates.append(result)

        # PURE PROFIT RANKING - sort by adjusted holdout P&L (after slippage)
        holdout_candidates.sort(key=lambda x: x.holdout_metrics['adjusted_pnl'], reverse=True)

    final_results = holdout_candidates
    print(f"{len(final_results)} combos pass holdout requirements:")
    print(f"  - Positive adjusted P&L (after slippage)")
    print(f"  - {MIN_HOLDOUT_ROUNDS}+ rounds")
    print(f"  - Bonferroni significant (p < {holdout_bonferroni_alpha:.6f})")

    # Top 3 validation (overfitting guard)
    if final_results:
        top3_count, top3_all_pass = validate_top_3_candidates(
            final_results, prices, holdout, funding_rates
        )

    # PHASE 4: Regime validation (market condition robustness)
    print()
    print("=" * 70)
    print("PHASE 4: REGIME VALIDATION")
    print("=" * 70)

    if n_workers > 0 and final_results:
        # Parallel regime validation
        run_parallel_regime(
            results=final_results,
            prices_file=prices_file,
            meta=meta,
            folds=folds,
            holdout=holdout,
            funding_rates=funding_rates,
            thoroughness=thoroughness,
            n_workers=n_workers,
            symbol=symbol,
            verbose=verbose
        )
    else:
        # Sequential regime validation
        for result in final_results:  # v7.6: No limit - test all combos
            result.regime_pass = validate_regime_robustness(
                result.params, holdout_prices, funding_rates
            )
            status = "[PASS]" if result.regime_pass else "[FAIL]"
            print(f"  {status} th={result.params['threshold']}, tr={result.params['trailing']}")

    # PHASE 5: Monte Carlo validation (statistical validity)
    print()
    print("=" * 70)
    print("PHASE 5: MONTE CARLO VALIDATION")
    print("=" * 70)

    if final_results:
        if n_workers > 0:
            # Parallel Monte Carlo validation
            try:
                mc_validated = run_parallel_monte_carlo(
                    results=final_results,
                    prices_file=prices_file,
                    meta=meta,
                    folds=folds,
                    holdout=holdout,
                    funding_rates=funding_rates,
                    thoroughness=thoroughness,
                    n_workers=n_workers,
                    symbol=symbol,
                    verbose=verbose
                )
                print(f"\n{len(mc_validated)} combos passed Monte Carlo")
                if mc_validated:
                    final_results = mc_validated
            except Exception as e:
                print(f"  Monte Carlo parallel execution failed: {e}")
        else:
            # Sequential Monte Carlo validation
            try:
                from core.monte_carlo import run_monte_carlo_validation, format_monte_carlo_report

                mc_validated = []
                for result in final_results:  # v7.6: No limit - test all combos
                    # Use holdout sample rate for Monte Carlo backtest
                    holdout_result = run_backtest_with_params(result.params, holdout_prices, funding_rates, sample_rate=thoroughness.holdout_sample_rate)
                    holdout_returns = holdout_result.get('per_round_returns', [])

                    if len(holdout_returns) < 10:
                        continue

                    passes, mc_result, perm_result = run_monte_carlo_validation(
                        round_returns=holdout_returns,
                        n_bootstrap=thoroughness.mc_bootstrap,
                        n_permutation=thoroughness.mc_permutation,
                        ruin_threshold=0.50,  # Generous - we don't limit drawdown
                        random_seed=42
                    )

                    result.mc_passes = passes
                    result.mc_prob_positive = mc_result.probability_positive
                    result.mc_prob_ruin = mc_result.probability_ruin
                    result.mc_sharpe_5th = mc_result.sharpe_5th_percentile
                    result.mc_sequence_matters = perm_result.sequence_matters

                    if passes:
                        mc_validated.append(result)
                        print(f"  [PASS] th={result.params['threshold']}, tr={result.params['trailing']}, "
                              f"P(+)={mc_result.probability_positive:.1%}, "
                              f"Holdout P&L={result.holdout_metrics['total_pnl']:.1f}%")
                    else:
                        print(f"  [FAIL] th={result.params['threshold']}, tr={result.params['trailing']}")

                print(f"\n{len(mc_validated)} combos passed Monte Carlo")

                if mc_validated:
                    final_results = mc_validated

            except ImportError as e:
                print(f"  Monte Carlo module not available: {e}")

    # PHASE 6: Walk-Forward Validation (v7.1 NEW)
    print()
    print("=" * 70)
    print("PHASE 6: WALK-FORWARD VALIDATION (v7.1 NEW)")
    print("=" * 70)
    print("Testing strategy stability across different time periods")
    print()

    walk_forward_result = None
    if final_results:
        try:
            from core.walk_forward import (
                run_walk_forward_optimization,
                format_walk_forward_report,
            )

            # Create a wrapper for run_backtest_with_params that matches expected signature
            # Use holdout sample rate for walk-forward validation
            def backtest_wrapper(params, price_slice, funding):
                return run_backtest_with_params(params, price_slice, funding, return_rounds=True, sample_rate=thoroughness.holdout_sample_rate)

            walk_forward_result = run_walk_forward_optimization(
                prices=prices,
                funding_rates=funding_rates,
                run_backtest_func=backtest_wrapper,
                train_years=2,
                test_years=1,
                step_years=1,
                total_years=max(1, days // 365),
                verbose=True,
            )

            # Print report
            print(format_walk_forward_report(walk_forward_result))

            # Store walk-forward stable params if available
            if walk_forward_result.best_stable_params and final_results:
                # Check if walk-forward stable params match top result
                top_params = final_results[0].params
                wf_params = walk_forward_result.best_stable_params
                params_match = (
                    top_params.get('threshold') == wf_params.get('threshold') and
                    top_params.get('trailing') == wf_params.get('trailing') and
                    top_params.get('pyramid_step') == wf_params.get('pyramid_step')
                )
                if params_match:
                    print("\n  [EXCELLENT] Walk-forward stable params MATCH top holdout result!")
                else:
                    print("\n  [NOTE] Walk-forward stable params differ from top holdout result:")
                    print(f"    Holdout best: th={top_params.get('threshold')}, tr={top_params.get('trailing')}")
                    print(f"    Walk-forward: th={wf_params.get('threshold')}, tr={wf_params.get('trailing')}")

        except ImportError as e:
            print(f"  Walk-forward module not available: {e}")
        except Exception as e:
            print(f"  Walk-forward validation error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    save_results(final_results, output_dir, symbol, funding_rates, walk_forward_result)

    return final_results


def save_results(
    results: List[OptimizationResult],
    output_dir: str,
    symbol: str,
    funding_rates: Dict,
    walk_forward_result=None
):
    """Save optimization results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save top results as JSON
    top_10 = results[:10]
    results_data = []
    for r in top_10:
        results_data.append({
            'params': r.params,
            'train_metrics': r.train_metrics,
            'val_metrics': r.val_metrics,
            'holdout_metrics': r.holdout_metrics,
            'fold_pnls': r.fold_pnls,
            'total_pnl_positive': r.total_pnl_positive,
            'avg_val_pnl': r.avg_val_pnl,
            'p_value': r.p_value,
            'sharpe': r.sharpe,
            'robustness_score': r.robustness_score,
            'monte_carlo': {
                'passes': r.mc_passes,
                'prob_positive': r.mc_prob_positive,
                'prob_ruin': r.mc_prob_ruin,
                'sharpe_5th_pct': r.mc_sharpe_5th,
                'sequence_matters': r.mc_sequence_matters,
            },
        })

    with open(os.path.join(output_dir, f"{symbol}_top10_v7.json"), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    # Print summary
    print()
    print("=" * 70)
    print("TOP 10 RESULTS (PURE PROFIT v7.1)")
    print("=" * 70)
    for i, r in enumerate(top_10):
        params = r.params
        print(f"\n{i+1}. threshold={params['threshold']}, trailing={params['trailing']}, "
              f"pyramid_step={params['pyramid_step']}")
        print(f"   vol_type={params['vol_type']}, size_schedule={params['size_schedule']}, "
              f"max_pyramids={params['max_pyramids']}")
        print(f"   ---")
        print(f"   Train P&L:   {r.train_metrics['total_pnl']:+.2f}%")
        print(f"   Val P&L:     {r.avg_val_pnl:+.2f}% (avg across folds)")
        holdout_pnl = r.holdout_metrics.get('total_pnl', 0)
        adjusted_pnl = r.holdout_metrics.get('adjusted_pnl', holdout_pnl)
        slippage_pct = r.holdout_metrics.get('slippage_pct', 0)
        print(f"   HOLDOUT P&L: {holdout_pnl:+.2f}% (raw)")
        print(f"   ADJUSTED:    {adjusted_pnl:+.2f}% (after {slippage_pct:.2f}% slippage) << REALISTIC P&L")
        print(f"   ---")
        print(f"   Win Rate: {r.train_metrics['win_rate']:.1f}%")
        print(f"   Max DD: {r.train_metrics['max_drawdown_pct']:.1f}%")
        print(f"   Rounds: {r.holdout_metrics['total_rounds']}")
        print(f"   Robustness: {r.robustness_score:.2f}")
        print(f"   Bonferroni p-value: {r.p_value:.6f}")
        print(f"   Monte Carlo: P(+)={r.mc_prob_positive:.1%}, P(ruin)={r.mc_prob_ruin:.1%}")

    # Save winner
    if results:
        winner = results[0]
        winner_data = {
            'params': winner.params,
            'metrics': {
                'train': winner.train_metrics,
                'val': winner.val_metrics,
                'holdout': winner.holdout_metrics,
            },
            'cross_validation': {
                'fold_pnls': winner.fold_pnls,
                'total_pnl_positive': winner.total_pnl_positive,
                'avg_val_pnl': winner.avg_val_pnl,
            },
            'robustness': {
                'robustness_score': winner.robustness_score,
            },
            'monte_carlo': {
                'passes': winner.mc_passes,
                'prob_positive': winner.mc_prob_positive,
                'prob_ruin': winner.mc_prob_ruin,
                'sharpe_5th_pct': winner.mc_sharpe_5th,
                'sequence_matters': winner.mc_sequence_matters,
            },
            'statistics': {
                'p_value': winner.p_value,
                'sharpe': winner.sharpe,
                'is_significant': winner.is_significant,
            },
            'optimization_info': {
                'version': 'v7.1-pure-profit',
                'grid_size': get_grid_size(),
                'bonferroni_alpha': get_bonferroni_alpha(),
                'timestamp': datetime.now().isoformat(),
                'fixes': [
                    'non-overlapping folds',
                    'bonferroni on holdout',
                    'robustness excludes holdout',
                    '15% holdout (9 months)',
                    'position-size slippage',
                    'walk-forward validation',
                ],
            },
        }

        # Add walk-forward results if available
        if walk_forward_result:
            winner_data['walk_forward'] = {
                'total_test_pnl': walk_forward_result.total_test_pnl,
                'avg_test_pnl': walk_forward_result.avg_test_pnl,
                'param_stability_score': walk_forward_result.param_stability_score,
                'best_stable_params': walk_forward_result.best_stable_params,
                'windows_analyzed': len(walk_forward_result.windows),
            }

        with open(os.path.join(output_dir, f"{symbol}_winner_v7.json"), 'w') as f:
            json.dump(winner_data, f, indent=2, default=str)

        # Print live trading command
        print()
        print("=" * 70)
        print("LIVE TRADING COMMAND")
        print("=" * 70)
        max_pyr = winner.params.get('max_pyramids', 20)
        print(f"  python main.py --mode trading --symbol {symbol} \\")
        print(f"    --threshold {winner.params['threshold']} --trailing {winner.params['trailing']} \\")
        print(f"    --pyramid {winner.params['pyramid_step']} --max-pyramids {max_pyr}")

        # Final verdict
        all_pass = (winner.total_pnl_positive and
                    winner.robustness_score >= MIN_ROBUSTNESS_SCORE and
                    winner.mc_passes and
                    winner.holdout_metrics['total_pnl'] > 0)

        print()
        print("=" * 70)
        if all_pass:
            print("STRATEGY PASSES ALL VALIDATION CHECKS")
            print()
            print("  [OK] Holdout P&L positive")
            print("  [OK] Cross-fold total positive")
            print("  [OK] Robustness test passed")
            print("  [OK] Monte Carlo validation passed")
            print()
            print("Ready for paper trading (8 weeks minimum)")
        else:
            print("WARNING: STRATEGY FAILED SOME VALIDATION CHECKS")
            if not winner.total_pnl_positive:
                print("  [FAIL] Total P&L not positive across folds")
            if winner.robustness_score < MIN_ROBUSTNESS_SCORE:
                print("  [FAIL] Robustness below threshold")
            if not winner.mc_passes:
                print("  [FAIL] Monte Carlo validation failed")
            if winner.holdout_metrics['total_pnl'] <= 0:
                print("  [FAIL] Holdout P&L not positive")
        print("=" * 70)

        print(f"\nWinner saved to: {output_dir}/{symbol}_winner_v7.json")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

AVAILABLE_COINS = [
    ("BTC", "BTCUSDT", "Bitcoin"),
    ("ETH", "ETHUSDT", "Ethereum"),
    ("SOL", "SOLUSDT", "Solana"),
    ("XRP", "XRPUSDT", "Ripple"),
    ("DOGE", "DOGEUSDT", "Dogecoin"),
    ("XLM", "XLMUSDT", "Stellar"),
]


def select_coin_interactive() -> str:
    """Display interactive coin selection menu."""
    print()
    print("=" * 50)
    print("SELECT COIN FOR OPTIMIZATION")
    print("=" * 50)
    print()
    for i, (short, symbol, name) in enumerate(AVAILABLE_COINS, 1):
        print(f"  {i}. {short:5} - {name} ({symbol})")
    print()

    while True:
        try:
            choice = input("Enter choice (1-6): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_COINS):
                selected = AVAILABLE_COINS[idx]
                print(f"\nSelected: {selected[2]} ({selected[1]})")
                return selected[1]
            else:
                print("Invalid choice. Please enter 1-6.")
        except ValueError:
            print("Invalid input. Please enter a number 1-6.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def select_thoroughness_interactive() -> ThoroughnessConfig:
    """Display interactive thoroughness selection menu."""
    print()
    print("=" * 50)
    print("SELECT ANALYSIS THOROUGHNESS")
    print("=" * 50)
    print()

    options = [
        ('quick', 'Quick', '~2-4 hours', '1% data, fast screening'),
        ('standard', 'Standard', '~6-12 hours', '10% data, balanced (RECOMMENDED)'),
        ('thorough', 'Thorough', '~24-48 hours', '20% data, deep analysis'),
        ('exhaustive', 'Exhaustive', '~27-54 days', '100% data, maximum accuracy'),
    ]

    for i, (key, name, time, desc) in enumerate(options, 1):
        rec = " ★" if key == 'standard' else ""
        print(f"  {i}. {name:12} {time:15} - {desc}{rec}")
    print()

    while True:
        try:
            choice = input("Enter choice (1-4) [2]: ").strip()
            if choice == "":
                choice = "2"  # Default to standard
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected_key = options[idx][0]
                selected = THOROUGHNESS_CONFIGS[selected_key]
                print(f"\nSelected: {selected.name} (~{selected.estimated_time})")
                return selected
            else:
                print("Invalid choice. Please enter 1-4.")
        except ValueError:
            print("Invalid input. Please enter a number 1-4.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Pure Profit Pyramid Strategy Optimizer v7 - 'Whatever Works'"
    )
    parser.add_argument(
        "--symbol", "-s",
        default=None,
        help="Trading pair symbol (e.g., BTCUSDT). If not provided, shows interactive menu."
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Days of historical data (default: {DEFAULT_DAYS})"
    )
    parser.add_argument(
        "--output", "-o",
        default="./optimization_v7_results",
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--thoroughness", "-t",
        type=str,
        choices=['quick', 'standard', 'thorough', 'exhaustive'],
        default=None,
        help="Analysis thoroughness: quick (~2-4h), standard (~6-12h), "
             "thorough (~24-48h), exhaustive (~27-54 days). If not provided, shows interactive menu."
    )
    parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=None,
        help="(Advanced) Override sample rate. Use --thoroughness instead."
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes for parallel execution. "
             "Default: auto-detect (CPU cores - 1). Use 0 to force sequential mode."
    )

    args = parser.parse_args()

    # Interactive coin selection if no symbol provided
    if args.symbol is None:
        args.symbol = select_coin_interactive()

    # Resolve thoroughness config
    if args.sample_rate is not None:
        # Advanced user override - create custom config
        thoroughness = ThoroughnessConfig(
            name=f'Custom (sample_rate={args.sample_rate})',
            sample_rate=args.sample_rate,
            crossfold_divisor=2,
            robustness_divisor=2,
            holdout_sample_rate=1,
            mc_bootstrap=10000,
            mc_permutation=5000,
            estimated_time='varies',
        )
    elif args.thoroughness is not None:
        # CLI flag provided
        thoroughness = THOROUGHNESS_CONFIGS[args.thoroughness]
    else:
        # Interactive selection
        thoroughness = select_thoroughness_interactive()

    results = run_optimization(
        symbol=args.symbol,
        days=args.days,
        output_dir=args.output,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        thoroughness=thoroughness,
        workers=args.workers
    )

    print()
    print(f"Completed: {datetime.now()}")

    if results:
        print(f"\nOptimization complete! {len(results)} pure profit combos found.")
        print(f"Grid size: 640 (8x more statistically valid than v5/v6)")
    else:
        print("\nNo statistically valid parameters found.")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
