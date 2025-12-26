"""
Balanced Parameter Grid for Statistical Validity

This module provides a constrained parameter grid (~5,000 combinations)
designed for statistically valid optimization.

Key design decisions:
1. vol_type tested at SAME level as core params (not locked to 'none')
2. Grid size allows Bonferroni correction with α=0.00001
3. Parameters chosen based on theory, not exhaustive search
4. Includes only combinations with physical/market meaning

Grid Size Calculation:
- Core params: 5 × 5 × 5 × 4 = 500 combinations
- vol_type: × 2 = 1,000 combinations
- Extended groups: × 5 = 5,000 combinations

With 5,000 tests and Bonferroni α = 0.05/5000 = 0.00001:
- Expected false positives: 0.05 (controlled)
- Each significant result has 99.999% confidence
"""

from typing import Dict, List, Tuple, Optional, Any
from itertools import product


# =============================================================================
# BALANCED CORE GRID (~5,000 combinations)
# =============================================================================

def build_balanced_grid() -> Dict[str, List]:
    """
    Build a balanced parameter grid with ~5,000 combinations.

    This grid is designed for statistical validity:
    - Tests vol_type at SAME level as core params
    - Grid size allows Bonferroni correction (α=0.00001)
    - Parameters based on trading theory

    Returns:
        Dict mapping param names to lists of values
    """
    return {
        # CORE PARAMETERS
        # Threshold: Entry/exit signal sensitivity
        # Theory: Lower = more signals, higher = fewer but larger moves
        'threshold': [1.5, 2.5, 4.0, 6.0, 10.0],  # 5 values

        # Trailing: Profit protection aggressiveness
        # Theory: Tight = captures more profit, loose = lets winners run
        'trailing': [0.5, 1.0, 2.0, 3.5],  # 4 values

        # Pyramid Step: Position scaling frequency
        # Theory: Tight = aggressive scaling, wide = conservative
        'pyramid_step': [0.5, 1.0, 2.0, 3.5],  # 4 values

        # Max Pyramids: Maximum position scaling
        # Theory: More = higher risk/reward, fewer = conservative
        'max_pyramids': [10, 20, 35, 50],  # 4 values

        # CRITICAL: vol_type now at SAME level (not locked to 'none')
        # Theory: Volatility filter prevents entries in choppy markets
        'vol_type': ['none', 'stddev'],  # 2 values

        # Size Schedule: Position sizing strategy
        # Theory: Decay reduces risk as pyramids accumulate
        'size_schedule': ['fixed', 'exp_decay'],  # 2 values

        # EXIT CONTROLS (grouped because they interact)
        # Group 0: No exit controls (baseline)
        # Group 1: Take profit only
        # Group 2: Stop loss only
        # Group 3: Both TP and SL
        'exit_group': [0, 1, 2, 3],  # 4 smart combinations

        # Fixed parameters (match live trading)
        'poll_interval': [1.0],  # Must match live PRICE_CHECK_INTERVAL
        'use_causal': [True],  # Always use causal trailing (no look-ahead)
        'confirmation_ticks': [3],  # Default confirmation
    }
    # Total: 5 × 4 × 4 × 4 × 2 × 2 × 4 = 5,120 combinations


def expand_exit_group(group_id: int) -> Dict[str, Any]:
    """
    Expand exit_group ID into actual parameter values.

    Groups are designed based on exit strategy theory:
    - Group 0: Pure trailing stop (baseline)
    - Group 1: Take profit at 10% (lock gains)
    - Group 2: Stop loss at 20% (limit downside)
    - Group 3: TP=15%, SL=15% (balanced)
    """
    groups = {
        0: {'take_profit_pct': None, 'stop_loss_pct': None, 'breakeven_after_pct': None},
        1: {'take_profit_pct': 10, 'stop_loss_pct': None, 'breakeven_after_pct': None},
        2: {'take_profit_pct': None, 'stop_loss_pct': 20, 'breakeven_after_pct': None},
        3: {'take_profit_pct': 15, 'stop_loss_pct': 15, 'breakeven_after_pct': None},
    }
    return groups.get(group_id, groups[0])


def expand_params(params: Dict) -> Dict:
    """
    Expand grouped parameters into full parameter dict.

    Takes a params dict with group IDs and expands them
    into individual parameter values.
    """
    expanded = params.copy()

    # Expand exit group
    if 'exit_group' in expanded:
        exit_params = expand_exit_group(expanded.pop('exit_group'))
        expanded.update(exit_params)

    # Add neutral defaults for parameters not in grid
    defaults = {
        'acceleration': 1.0,
        'min_spacing': 0.0,
        'time_decay': None,
        'vol_min': 0.5 if expanded.get('vol_type') != 'none' else 0.0,
        'vol_window': 100,
        'pyramid_cooldown_sec': 0,
        'max_round_duration_hr': None,
        'trend_filter_ema': None,
        'session_filter': 'all',
    }

    for key, value in defaults.items():
        if key not in expanded:
            expanded[key] = value

    return expanded


def generate_all_combinations(grid: Dict[str, List] = None) -> List[Dict]:
    """
    Generate all parameter combinations from the balanced grid.

    Returns:
        List of parameter dictionaries (expanded, ready for backtest)
    """
    if grid is None:
        grid = build_balanced_grid()

    # Get param names and values
    param_names = list(grid.keys())
    param_values = [grid[name] for name in param_names]

    # Generate all combinations
    combinations = []
    for values in product(*param_values):
        params = dict(zip(param_names, values))
        expanded = expand_params(params)
        combinations.append(expanded)

    return combinations


def get_grid_size(grid: Dict[str, List] = None) -> int:
    """Calculate the total number of combinations in the grid."""
    if grid is None:
        grid = build_balanced_grid()

    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def get_bonferroni_alpha(grid: Dict[str, List] = None, family_alpha: float = 0.05) -> float:
    """
    Calculate Bonferroni-corrected significance level.

    Args:
        grid: Parameter grid (uses balanced grid if None)
        family_alpha: Family-wise error rate (default 0.05)

    Returns:
        Corrected alpha for individual tests
    """
    n_tests = get_grid_size(grid)
    return family_alpha / n_tests


# =============================================================================
# CROSS-VALIDATION FOLD STRUCTURE
# =============================================================================

def get_fixed_folds(total_days: int = 1825) -> List[Dict]:
    """
    Get fixed-size fold structure with gaps to prevent look-ahead bias.

    The plan specifies:
    - Fold 0: Train 0%-30%, Val 35%-45% (5% gap)
    - Fold 1: Train 20%-50%, Val 55%-65% (5% gap)
    - Fold 2: Train 40%-70%, Val 75%-85% (5% gap)
    - Holdout: 90%-100%

    Each fold has 30% training, 5% gap, 10% validation.

    Args:
        total_days: Total days in dataset (default 1825 = 5 years)

    Returns:
        List of fold configs with train/val/holdout indices
    """
    folds = [
        {
            'name': 'Fold 0',
            'train_start_pct': 0.0,
            'train_end_pct': 0.30,
            'val_start_pct': 0.35,  # 5% gap
            'val_end_pct': 0.45,
        },
        {
            'name': 'Fold 1',
            'train_start_pct': 0.20,
            'train_end_pct': 0.50,
            'val_start_pct': 0.55,  # 5% gap
            'val_end_pct': 0.65,
        },
        {
            'name': 'Fold 2',
            'train_start_pct': 0.40,
            'train_end_pct': 0.70,
            'val_start_pct': 0.75,  # 5% gap
            'val_end_pct': 0.85,
        },
    ]

    # Convert percentages to day indices
    for fold in folds:
        fold['train_start'] = int(total_days * fold['train_start_pct'])
        fold['train_end'] = int(total_days * fold['train_end_pct'])
        fold['val_start'] = int(total_days * fold['val_start_pct'])
        fold['val_end'] = int(total_days * fold['val_end_pct'])

    # Holdout is always 90%-100%
    holdout = {
        'name': 'Holdout',
        'start_pct': 0.90,
        'end_pct': 1.00,
        'start': int(total_days * 0.90),
        'end': total_days,
    }

    return folds, holdout


# =============================================================================
# VALIDATION GATE (Immediate train→val check)
# =============================================================================

def check_validation_gate(
    train_result: Dict,
    val_result: Dict,
    min_val_pnl_ratio: float = 0.5,
    min_val_win_rate: float = 40.0,
    min_val_rounds: int = 20
) -> Tuple[bool, str]:
    """
    Check if a parameter combo passes the immediate validation gate.

    This is applied IMMEDIATELY after each combo is tested on training,
    before proceeding to the next fold. Only combos that pass proceed.

    Criteria:
    - val_pnl >= min_val_pnl_ratio * train_pnl (no severe overfitting)
    - val_win_rate >= min_val_win_rate
    - val_rounds >= min_val_rounds

    Args:
        train_result: Backtest result on training data
        val_result: Backtest result on validation data
        min_val_pnl_ratio: Minimum val/train P&L ratio
        min_val_win_rate: Minimum validation win rate %
        min_val_rounds: Minimum validation rounds

    Returns:
        Tuple of (passed: bool, reason: str)
    """
    train_pnl = train_result.get('total_pnl', 0)
    val_pnl = val_result.get('total_pnl', 0)
    val_win_rate = val_result.get('win_rate', 0)
    val_rounds = val_result.get('total_rounds', 0)

    # Check minimum rounds
    if val_rounds < min_val_rounds:
        return False, f"Insufficient val rounds: {val_rounds} < {min_val_rounds}"

    # Check win rate
    if val_win_rate < min_val_win_rate:
        return False, f"Val win rate too low: {val_win_rate:.1f}% < {min_val_win_rate}%"

    # Check P&L ratio (only if train_pnl > 0)
    if train_pnl > 0:
        ratio = val_pnl / train_pnl
        if ratio < min_val_pnl_ratio:
            return False, f"Val/Train ratio too low: {ratio:.2f} < {min_val_pnl_ratio}"
    elif val_pnl <= 0:
        # Both train and val are non-positive
        return False, "Both train and val P&L are non-positive"

    return True, "Passed validation gate"


# =============================================================================
# BONFERRONI SIGNIFICANCE CHECK (During selection)
# =============================================================================

def check_bonferroni_significance(
    params: Dict,
    per_round_returns: List[float],
    n_total_tests: int,
    family_alpha: float = 0.05
) -> Tuple[bool, float, float]:
    """
    Check if a combo is statistically significant using Bonferroni correction.

    This is applied DURING selection, not after. Only combos that pass
    at the corrected alpha level proceed.

    Args:
        params: Parameter dictionary (for logging)
        per_round_returns: List of per-round P&L percentages
        n_total_tests: Total number of tests being run
        family_alpha: Family-wise error rate (default 0.05)

    Returns:
        Tuple of (significant: bool, p_value: float, corrected_alpha: float)
    """
    import math

    n = len(per_round_returns)
    if n < 2:
        return False, 1.0, family_alpha / n_total_tests

    # Calculate t-statistic (testing if mean > 0)
    mean = sum(per_round_returns) / n
    variance = sum((r - mean) ** 2 for r in per_round_returns) / (n - 1)
    std_error = (variance / n) ** 0.5

    if std_error == 0:
        return False, 1.0, family_alpha / n_total_tests

    t_stat = mean / std_error

    # Calculate p-value using t-distribution approximation
    # For n > 30, t-distribution ≈ normal
    if n > 30:
        # Using standard normal approximation
        # P(Z > t) = 1 - Φ(t)
        z = t_stat
        # Approximation of 1 - Φ(z) for z > 0
        if z > 0:
            p_value = 0.5 * math.erfc(z / math.sqrt(2))
        else:
            p_value = 1.0 - 0.5 * math.erfc(-z / math.sqrt(2))
    else:
        # For smaller samples, use conservative estimate
        # This is a rough approximation
        if t_stat > 3.5:
            p_value = 0.001
        elif t_stat > 2.5:
            p_value = 0.01
        elif t_stat > 1.8:
            p_value = 0.05
        elif t_stat > 1.3:
            p_value = 0.1
        elif t_stat > 0:
            p_value = 0.2
        else:
            p_value = 0.5 + abs(t_stat) * 0.1

    # Bonferroni correction
    corrected_alpha = family_alpha / n_total_tests
    significant = p_value < corrected_alpha

    return significant, p_value, corrected_alpha


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_grid_summary(grid: Dict[str, List] = None):
    """Print a summary of the balanced grid."""
    if grid is None:
        grid = build_balanced_grid()

    print("=" * 60)
    print("BALANCED PARAMETER GRID SUMMARY")
    print("=" * 60)

    total = 1
    for param, values in grid.items():
        n = len(values)
        total *= n
        if n <= 5:
            print(f"  {param}: {values} ({n})")
        else:
            print(f"  {param}: [{values[0]}...{values[-1]}] ({n})")

    print("-" * 60)
    print(f"  Total combinations: {total:,}")
    print(f"  Bonferroni α (0.05): {0.05/total:.2e}")
    print("=" * 60)


if __name__ == "__main__":
    # Test the module
    print_grid_summary()

    print("\nGenerating first 10 combinations...")
    combos = generate_all_combinations()
    for i, combo in enumerate(combos[:10]):
        print(f"\n{i+1}. threshold={combo['threshold']}, trailing={combo['trailing']}, "
              f"vol_type={combo['vol_type']}, exit_group removed -> "
              f"TP={combo.get('take_profit_pct')}, SL={combo.get('stop_loss_pct')}")

    print(f"\n...and {len(combos) - 10} more.")
    print(f"\nTotal combinations: {len(combos)}")
