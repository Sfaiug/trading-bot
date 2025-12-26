"""
Disk-based storage for optimization results.

This module provides efficient disk storage for per-round returns during
parameter optimization. Required for statistically valid Monte Carlo and
Sharpe ratio calculations (vs. synthetic/averaged returns).

Usage:
    storage = DiskResultStorage("./optimization_results")
    storage.save_combo_result(params, rounds, summary)

    # Later retrieval
    result = storage.load_combo_result(params_hash)
    returns = result['rounds']
"""

import os
import json
import hashlib
import gzip
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import struct


@dataclass
class RoundResult:
    """Single trading round result."""
    timestamp: str  # ISO format
    entry_price: float
    exit_price: float
    direction: str  # 'LONG' or 'SHORT'
    pnl_pct: float
    duration_sec: float
    num_pyramids: int
    max_profit_pct: float
    exit_reason: str  # 'TRAILING', 'TAKE_PROFIT', 'STOP_LOSS', etc.


@dataclass
class ComboResult:
    """Complete result for a parameter combination."""
    params: Dict[str, Any]
    rounds: List[RoundResult]
    summary: Dict[str, float]
    created_at: str


def params_to_hash(params: Dict) -> str:
    """
    Generate a unique hash for a parameter combination.
    Used as filename for disk storage.
    """
    # Sort keys for consistent ordering
    sorted_items = sorted(params.items())
    params_str = json.dumps(sorted_items, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:12]


class DiskResultStorage:
    """
    Disk-based storage for optimization results.

    Stores per-round returns for each parameter combination,
    enabling real (non-synthetic) statistical validation.

    Storage format:
    - One gzipped JSON file per parameter combination
    - Files named by params hash for fast lookup
    - Automatic cleanup of old results
    """

    def __init__(self, base_dir: str = "./optimization_results"):
        """
        Initialize disk storage.

        Args:
            base_dir: Directory to store results (created if needed)
        """
        self.base_dir = base_dir
        self.index_file = os.path.join(base_dir, "_index.json")
        os.makedirs(base_dir, exist_ok=True)
        self._load_index()

    def _load_index(self):
        """Load or create the results index."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                'created_at': datetime.now().isoformat(),
                'combos': {},
                'stats': {
                    'total_combos': 0,
                    'total_rounds': 0,
                }
            }

    def _save_index(self):
        """Save the index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _get_filepath(self, params_hash: str) -> str:
        """Get the file path for a params hash."""
        return os.path.join(self.base_dir, f"{params_hash}.json.gz")

    def save_combo_result(
        self,
        params: Dict[str, Any],
        rounds: List[Dict],
        summary: Dict[str, float]
    ) -> str:
        """
        Save a parameter combination result to disk.

        Args:
            params: Parameter dictionary (threshold, trailing, etc.)
            rounds: List of per-round results with actual P&L values
            summary: Summary statistics (total_pnl, win_rate, etc.)

        Returns:
            params_hash: The hash used as file identifier
        """
        params_hash = params_to_hash(params)
        filepath = self._get_filepath(params_hash)

        # Create result structure
        result = {
            'params': params,
            'rounds': rounds,
            'summary': summary,
            'created_at': datetime.now().isoformat(),
        }

        # Save compressed
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(result, f)

        # Update index
        self.index['combos'][params_hash] = {
            'params_summary': f"th={params.get('threshold', '?')}_tr={params.get('trailing', '?')}",
            'total_pnl': summary.get('total_pnl', 0),
            'rounds': len(rounds),
            'created_at': result['created_at'],
        }
        self.index['stats']['total_combos'] += 1
        self.index['stats']['total_rounds'] += len(rounds)
        self._save_index()

        return params_hash

    def load_combo_result(self, params_hash: str) -> Optional[Dict]:
        """
        Load a parameter combination result from disk.

        Args:
            params_hash: Hash returned from save_combo_result

        Returns:
            Result dict or None if not found
        """
        filepath = self._get_filepath(params_hash)
        if not os.path.exists(filepath):
            return None

        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)

    def load_by_params(self, params: Dict) -> Optional[Dict]:
        """Load result by params dict (computes hash internally)."""
        params_hash = params_to_hash(params)
        return self.load_combo_result(params_hash)

    def get_per_round_returns(self, params: Dict) -> Optional[List[float]]:
        """
        Get just the per-round P&L values for a params combo.

        This is the key method for Monte Carlo validation -
        returns ACTUAL per-round returns, not synthetic averages.

        Args:
            params: Parameter dictionary

        Returns:
            List of per-round P&L percentages, or None if not found
        """
        result = self.load_by_params(params)
        if result is None:
            return None

        rounds = result.get('rounds', [])
        return [r['pnl_pct'] for r in rounds if 'pnl_pct' in r]

    def get_round_durations(self, params: Dict) -> Optional[List[float]]:
        """
        Get per-round durations for Sharpe annualization.

        Returns:
            List of round durations in seconds, or None if not found
        """
        result = self.load_by_params(params)
        if result is None:
            return None

        rounds = result.get('rounds', [])
        return [r['duration_sec'] for r in rounds if 'duration_sec' in r]

    def get_top_n(self, n: int = 10, metric: str = 'total_pnl') -> List[Dict]:
        """
        Get the top N parameter combinations by a metric.

        Args:
            n: Number of results to return
            metric: Metric to sort by ('total_pnl', 'rounds', etc.)

        Returns:
            List of {params_hash, params_summary, metric_value}
        """
        combos = list(self.index['combos'].items())
        combos.sort(key=lambda x: x[1].get(metric, 0), reverse=True)

        top_n = []
        for params_hash, info in combos[:n]:
            top_n.append({
                'params_hash': params_hash,
                'params_summary': info['params_summary'],
                metric: info.get(metric, 0),
                'rounds': info.get('rounds', 0),
            })

        return top_n

    def exists(self, params: Dict) -> bool:
        """Check if a params combo has been computed and stored."""
        params_hash = params_to_hash(params)
        return params_hash in self.index['combos']

    def clear(self):
        """Clear all stored results (for fresh optimization run)."""
        import shutil
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.index = {
            'created_at': datetime.now().isoformat(),
            'combos': {},
            'stats': {
                'total_combos': 0,
                'total_rounds': 0,
            }
        }
        self._save_index()

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            'total_combos': self.index['stats']['total_combos'],
            'total_rounds': self.index['stats']['total_rounds'],
            'disk_size_mb': self._get_disk_size() / (1024 * 1024),
        }

    def _get_disk_size(self) -> int:
        """Calculate total disk usage in bytes."""
        total = 0
        for f in os.listdir(self.base_dir):
            filepath = os.path.join(self.base_dir, f)
            if os.path.isfile(filepath):
                total += os.path.getsize(filepath)
        return total


class BatchProcessor:
    """
    Process parameter combinations in batches with disk storage.

    This prevents RAM exhaustion during grid search while
    preserving all per-round return data for validation.
    """

    def __init__(
        self,
        storage: DiskResultStorage,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ):
        """
        Initialize batch processor.

        Args:
            storage: DiskResultStorage instance
            batch_size: Number of combos per batch
            progress_callback: Optional function(completed, total) for progress
        """
        self.storage = storage
        self.batch_size = batch_size
        self.progress_callback = progress_callback

    def process_grid(
        self,
        param_grid: Dict[str, List],
        backtest_func: callable,
        cache_file: str,
        funding_rates: Optional[Dict] = None
    ) -> List[str]:
        """
        Process all parameter combinations in the grid.

        Args:
            param_grid: Dict of param_name -> list of values
            backtest_func: Function(params) -> result dict
            cache_file: Path to tick data cache
            funding_rates: Optional funding rate lookup

        Returns:
            List of params_hashes for all processed combos
        """
        from itertools import product
        import gc

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        all_combos = list(product(*param_values))

        total = len(all_combos)
        processed_hashes = []

        print(f"Processing {total} combinations in batches of {self.batch_size}...")

        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch = all_combos[batch_start:batch_end]

            for i, values in enumerate(batch):
                params = dict(zip(param_names, values))

                # Skip if already computed
                if self.storage.exists(params):
                    params_hash = params_to_hash(params)
                    processed_hashes.append(params_hash)
                    continue

                # Run backtest
                result = backtest_func(params, cache_file, funding_rates)

                # Extract per-round data (use new per_round_returns if available)
                rounds = []
                per_round_returns = result.get('per_round_returns', [])
                round_durations = result.get('round_durations_sec', [])

                if 'rounds' in result and result['rounds']:
                    for idx, r in enumerate(result['rounds']):
                        # Use pre-computed values if available
                        pnl_pct = per_round_returns[idx] if idx < len(per_round_returns) else getattr(r, 'total_pnl', 0)
                        duration_sec = round_durations[idx] if idx < len(round_durations) else 0

                        rounds.append({
                            'timestamp': str(getattr(r, 'entry_time', '')),
                            'entry_price': getattr(r, 'entry_price', 0),
                            'exit_price': getattr(r, 'exit_price', 0),
                            'direction': getattr(r, 'direction', ''),
                            'pnl_pct': pnl_pct,
                            'duration_sec': duration_sec,
                            'num_pyramids': getattr(r, 'num_pyramids', 0),
                            'max_profit_pct': getattr(r, 'max_profit_pct', 0),
                            'exit_reason': '',  # TODO: Add exit reason tracking
                        })

                # Create summary
                summary = {
                    'total_pnl': result.get('total_pnl', 0),
                    'total_rounds': result.get('total_rounds', 0),
                    'win_rate': result.get('win_rate', 0),
                    'avg_pnl': result.get('avg_pnl', 0),
                    'avg_pyramids': result.get('avg_pyramids', 0),
                    'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                }

                # Save to disk
                params_hash = self.storage.save_combo_result(params, rounds, summary)
                processed_hashes.append(params_hash)

            # Progress callback
            completed = batch_end
            if self.progress_callback:
                self.progress_callback(completed, total)
            else:
                pct = completed / total * 100
                print(f"  [{completed}/{total}] {pct:.1f}% complete", end='\r')

            # Force garbage collection between batches
            gc.collect()

        print(f"\nCompleted {len(processed_hashes)} combinations.")
        return processed_hashes


# Convenience functions for direct use

def create_storage(base_dir: str = "./optimization_results") -> DiskResultStorage:
    """Create or connect to a disk storage instance."""
    return DiskResultStorage(base_dir)


def get_real_returns(storage: DiskResultStorage, params: Dict) -> List[float]:
    """
    Get REAL per-round returns for Monte Carlo validation.

    This is the critical function that replaces synthetic returns.

    Args:
        storage: DiskResultStorage instance
        params: Parameter dictionary

    Returns:
        List of actual per-round P&L percentages

    Raises:
        ValueError: If no stored result found (can't use synthetic fallback)
    """
    returns = storage.get_per_round_returns(params)
    if returns is None:
        raise ValueError(
            f"No stored results for params {params}. "
            "Cannot use synthetic returns - run backtest with return_rounds=True first."
        )
    return returns


def calculate_real_sharpe(
    storage: DiskResultStorage,
    params: Dict,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio using REAL per-round returns.

    Unlike the synthetic fallback, this uses actual return distribution.

    Args:
        storage: DiskResultStorage instance
        params: Parameter dictionary
        risk_free_rate: Annual risk-free rate (default 0)

    Returns:
        Annualized Sharpe ratio
    """
    returns = get_real_returns(storage, params)
    durations = storage.get_round_durations(params)

    if len(returns) < 2:
        return 0.0

    # Calculate mean and std of returns
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_return = variance ** 0.5

    if std_return == 0:
        return 0.0

    # Calculate actual trading frequency for annualization
    if durations and len(durations) > 0:
        avg_duration_days = sum(durations) / len(durations) / 86400
        trades_per_year = 365 / avg_duration_days if avg_duration_days > 0 else 10
    else:
        # Fallback: conservative 10 trades/year (not synthetic!)
        trades_per_year = 10

    # Annualized Sharpe
    excess_return = mean_return - (risk_free_rate / trades_per_year)
    sharpe = (excess_return / std_return) * (trades_per_year ** 0.5)

    return sharpe


def run_grid_search(
    param_grid: Dict[str, List],
    backtest_func: callable,
    cache_file: str,
    funding_rates: Optional[Dict] = None,
    storage_dir: str = "./optimization_results",
    batch_size: int = 100,
    clear_existing: bool = False,
    progress_callback: Optional[callable] = None
) -> Tuple[DiskResultStorage, List[str]]:
    """
    High-level convenience function to run a complete grid search.

    This function:
    1. Creates or connects to disk storage
    2. Processes all parameter combinations in batches
    3. Stores results with per-round returns for Monte Carlo validation
    4. Returns storage handle and processed hashes

    Args:
        param_grid: Dict of param_name -> list of values
        backtest_func: Function(params, cache_file, funding_rates) -> result dict
        cache_file: Path to tick data cache file
        funding_rates: Optional funding rate lookup dict
        storage_dir: Directory for disk storage (default: ./optimization_results)
        batch_size: Number of combos per batch (default: 100)
        clear_existing: If True, clears existing results before starting
        progress_callback: Optional function(completed, total) for progress

    Returns:
        Tuple of (DiskResultStorage, list of params_hashes)

    Example:
        ```python
        grid = {
            'threshold': [2, 3, 5],
            'trailing': [0.5, 1.0, 2.0],
            'pyramid_step': [1.0, 2.0],
        }

        def my_backtest(params, cache, funding):
            return run_pyramid_backtest(
                prices=load_prices(cache),
                threshold_pct=params['threshold'],
                trailing_pct=params['trailing'],
                pyramid_step_pct=params['pyramid_step'],
            )

        storage, hashes = run_grid_search(grid, my_backtest, 'btc_ticks.pkl')
        top_combos = storage.get_top_n(10, 'total_pnl')
        ```
    """
    # Create storage
    storage = DiskResultStorage(storage_dir)
    if clear_existing:
        storage.clear()

    # Create processor
    processor = BatchProcessor(
        storage=storage,
        batch_size=batch_size,
        progress_callback=progress_callback
    )

    # Run grid search
    hashes = processor.process_grid(
        param_grid=param_grid,
        backtest_func=backtest_func,
        cache_file=cache_file,
        funding_rates=funding_rates
    )

    return storage, hashes


def validate_combo_with_real_returns(
    storage: DiskResultStorage,
    params: Dict,
    min_rounds: int = 20,
    min_win_rate: float = 40.0,
    min_sharpe: float = 0.5
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a parameter combination using real per-round returns.

    This function performs statistical validation using actual stored
    returns, not synthetic data.

    Args:
        storage: DiskResultStorage instance
        params: Parameter dictionary to validate
        min_rounds: Minimum number of trading rounds required
        min_win_rate: Minimum win rate percentage
        min_sharpe: Minimum annualized Sharpe ratio

    Returns:
        Tuple of (passed: bool, validation_details: dict)

    Example:
        passed, details = validate_combo_with_real_returns(storage, params)
        if passed:
            print("Combo passed validation!")
        else:
            print(f"Failed: {details['failure_reason']}")
    """
    result = storage.load_by_params(params)

    if result is None:
        return False, {'failure_reason': 'No stored result found'}

    rounds = result.get('rounds', [])
    summary = result.get('summary', {})

    details = {
        'total_rounds': len(rounds),
        'total_pnl': summary.get('total_pnl', 0),
        'win_rate': summary.get('win_rate', 0),
        'sharpe': 0.0,
        'passed_rounds': False,
        'passed_win_rate': False,
        'passed_sharpe': False,
    }

    # Check minimum rounds
    if len(rounds) < min_rounds:
        details['failure_reason'] = f'Insufficient rounds: {len(rounds)} < {min_rounds}'
        return False, details
    details['passed_rounds'] = True

    # Check win rate
    win_rate = summary.get('win_rate', 0)
    if win_rate < min_win_rate:
        details['failure_reason'] = f'Win rate too low: {win_rate:.1f}% < {min_win_rate}%'
        return False, details
    details['passed_win_rate'] = True

    # Calculate and check Sharpe using real returns
    try:
        sharpe = calculate_real_sharpe(storage, params)
        details['sharpe'] = sharpe
        if sharpe < min_sharpe:
            details['failure_reason'] = f'Sharpe too low: {sharpe:.2f} < {min_sharpe}'
            return False, details
        details['passed_sharpe'] = True
    except ValueError as e:
        details['failure_reason'] = f'Sharpe calculation failed: {e}'
        return False, details

    details['failure_reason'] = None
    return True, details


def get_combo_statistics(storage: DiskResultStorage, params: Dict) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a parameter combination.

    Returns detailed stats computed from actual per-round returns.

    Args:
        storage: DiskResultStorage instance
        params: Parameter dictionary

    Returns:
        Dictionary with statistics or None if not found
    """
    result = storage.load_by_params(params)
    if result is None:
        return None

    rounds = result.get('rounds', [])
    summary = result.get('summary', {})

    if not rounds:
        return {
            'params': result.get('params', {}),
            'summary': summary,
            'statistics': None,
            'error': 'No rounds data'
        }

    # Extract returns
    returns = [r['pnl_pct'] for r in rounds if 'pnl_pct' in r]
    durations = [r['duration_sec'] for r in rounds if 'duration_sec' in r]

    if len(returns) < 2:
        return {
            'params': result.get('params', {}),
            'summary': summary,
            'statistics': None,
            'error': 'Insufficient returns for statistics'
        }

    # Calculate statistics
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_return = variance ** 0.5

    sorted_returns = sorted(returns)
    median_return = sorted_returns[len(sorted_returns) // 2]

    positive_returns = [r for r in returns if r > 0]
    negative_returns = [r for r in returns if r < 0]

    # Calculate max drawdown from returns sequence
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Calculate Sharpe
    try:
        sharpe = calculate_real_sharpe(storage, params)
    except ValueError:
        sharpe = None

    # Average duration
    avg_duration_hours = (sum(durations) / len(durations) / 3600) if durations else None

    return {
        'params': result.get('params', {}),
        'summary': summary,
        'statistics': {
            'count': len(returns),
            'mean': mean_return,
            'std': std_return,
            'median': median_return,
            'min': min(returns),
            'max': max(returns),
            'sum': sum(returns),
            'wins': len(positive_returns),
            'losses': len(negative_returns),
            'win_rate': len(positive_returns) / len(returns) * 100,
            'avg_win': sum(positive_returns) / len(positive_returns) if positive_returns else 0,
            'avg_loss': sum(negative_returns) / len(negative_returns) if negative_returns else 0,
            'profit_factor': abs(sum(positive_returns) / sum(negative_returns)) if negative_returns else float('inf'),
            'max_consecutive_drawdown': max_drawdown,
            'sharpe': sharpe,
            'avg_duration_hours': avg_duration_hours,
        }
    }
