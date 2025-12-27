"""
Minimal Parameter Grid for Maximum Statistical Validity (v7)

This module provides a carefully reduced parameter grid (640 combinations)
designed for maximum statistical robustness with minimal overfitting risk.

Key design decisions:
1. Only 6 parameters tested (vs 7 in v5/v6)
2. Removed redundant parameters (acceleration, min_spacing, time_decay)
3. Removed TP/SL (conflicts with trailing stop philosophy)
4. 8x better Bonferroni alpha than v5/v6

Grid Size: 5 x 4 x 4 x 2 x 2 x 2 = 640 combinations
Bonferroni alpha: 0.05/640 = 0.000078 (vs 0.00001 for 5,120)

Parameter Selection Rationale:
- threshold: ESSENTIAL - controls entry timing
- trailing: ESSENTIAL - controls exit timing
- pyramid_step: ESSENTIAL - controls position scaling
- max_pyramids: IMPORTANT - caps risk exposure
- vol_type: IMPORTANT - binary test of volatility filtering
- size_schedule: IMPORTANT - binary test of position sizing

Removed Parameters (with reasoning):
- poll_interval: Fixed at 1s (matches data resolution)
- acceleration: Redundant with pyramid_step
- min_spacing: Redundant with pyramid_step
- time_decay: Band-aid parameter, trailing stop handles exits
- vol_min/vol_window: Fix at sensible defaults, testing these overfits
- exit_group (TP/SL): Conflicts with trailing stop philosophy
"""

from typing import Dict, List, Tuple, Optional, Any
from itertools import product


# =============================================================================
# MINIMAL GRID (640 combinations)
# =============================================================================

def build_minimal_grid() -> Dict[str, List]:
    """
    Build a minimal parameter grid with 640 combinations.

    This grid is designed for maximum statistical robustness:
    - Only essential parameters tested
    - 8x better Bonferroni alpha than v5/v6
    - Lower overfitting risk

    Returns:
        Dict mapping param names to lists of values
    """
    return {
        # ESSENTIAL PARAMETERS (determine strategy behavior)

        # Threshold: When to close losing hedge and start pyramiding
        # Lower = react faster to small moves, higher = wait for larger moves
        'threshold': [1.5, 3.0, 5.0, 8.0, 12.0],  # 5 values

        # Trailing: Profit protection - how much can profit drop before exit
        # Lower = lock in profits early, higher = let winners run
        'trailing': [0.5, 1.0, 2.0, 4.0],  # 4 values

        # Pyramid Step: Distance between pyramid entries
        # Lower = add positions frequently, higher = add positions rarely
        'pyramid_step': [0.5, 1.0, 2.0, 4.0],  # 4 values

        # IMPORTANT MODIFIERS (binary tests)

        # Max Pyramids: Hard cap on position count
        'max_pyramids': [20, 50],  # 2 values

        # Volatility Filter: Only pyramid in volatile markets
        'vol_type': ['none', 'stddev'],  # 2 values

        # Size Schedule: Position sizing strategy
        'size_schedule': ['fixed', 'exp_decay'],  # 2 values
    }
    # Total: 5 x 4 x 4 x 2 x 2 x 2 = 640 combinations


# Fixed parameters (not tested, set to neutral/sensible defaults)
FIXED_PARAMS = {
    # Spacing parameters - simplified to linear
    'acceleration': 1.0,      # Linear spacing (no exponential)
    'min_spacing': 0.0,       # No extra minimum (pyramid_step handles it)

    # Time-based exit - disabled (trailing stop handles exits)
    'time_decay': None,

    # Volatility filter settings (when vol_type='stddev')
    'vol_min': 0.5,           # Sensible default threshold
    'vol_window': 100,        # Standard lookback

    # Exit controls - disabled (pure trailing stop)
    'take_profit_pct': None,  # Let winners run
    'stop_loss_pct': None,    # Rely on trailing stop
    'breakeven_after_pct': None,

    # Timing
    'pyramid_cooldown_sec': 0,
    'max_round_duration_hr': None,

    # Filters
    'trend_filter_ema': None,
    'session_filter': 'all',

    # Execution
    'poll_interval': 1.0,     # Match data resolution
    'use_causal': True,       # No look-ahead bias
    'confirmation_ticks': 3,  # Standard confirmation
}


def expand_params(params: Dict) -> Dict:
    """
    Expand grid parameters into full parameter dict with defaults.

    Takes a params dict from the grid and adds all fixed defaults.
    """
    expanded = params.copy()

    # Add all fixed parameters
    for key, value in FIXED_PARAMS.items():
        if key not in expanded:
            # Special case: vol_min should be 0 if vol_type is 'none'
            if key == 'vol_min' and expanded.get('vol_type') == 'none':
                expanded[key] = 0.0
            else:
                expanded[key] = value

    return expanded


def generate_all_combinations(grid: Dict[str, List] = None) -> List[Dict]:
    """
    Generate all parameter combinations from the minimal grid.

    Returns:
        List of parameter dictionaries (expanded, ready for backtest)
    """
    if grid is None:
        grid = build_minimal_grid()

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
        grid = build_minimal_grid()

    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def get_bonferroni_alpha(grid: Dict[str, List] = None, family_alpha: float = 0.05) -> float:
    """
    Calculate Bonferroni-corrected significance level.

    Args:
        grid: Parameter grid (uses minimal grid if None)
        family_alpha: Family-wise error rate (default 0.05)

    Returns:
        Corrected alpha for individual tests
    """
    n_tests = get_grid_size(grid)
    return family_alpha / n_tests


# =============================================================================
# CROSS-VALIDATION FOLD STRUCTURE (same as balanced_grid)
# =============================================================================

def get_fixed_folds(total_days: int = 1825) -> Tuple[List[Dict], Dict]:
    """
    Get NON-OVERLAPPING fold structure with gaps to prevent look-ahead bias.

    v7.1 FIX: Folds no longer overlap - each training set is independent.
    This ensures proper cross-validation (correlated results from overlapping
    training data was a major statistical flaw in v7.0).

    Structure (with 2% gaps between train and val):
    - Fold 0: Train 0%-20%, Gap 20%-22%, Val 22%-32%
    - Fold 1: Train 32%-52%, Gap 52%-54%, Val 54%-64%
    - Fold 2: Train 64%-76%, Gap 76%-78%, Val 78%-85%
    - Holdout: 85%-100% (15% = ~9 months for 5 years)

    Args:
        total_days: Total days in dataset (default 1825 = 5 years)

    Returns:
        Tuple of (folds list, holdout dict)
    """
    folds = [
        {
            'name': 'Fold 0',
            'train_start_pct': 0.00,
            'train_end_pct': 0.20,
            'val_start_pct': 0.22,  # 2% gap
            'val_end_pct': 0.32,
        },
        {
            'name': 'Fold 1',
            'train_start_pct': 0.32,
            'train_end_pct': 0.52,
            'val_start_pct': 0.54,  # 2% gap
            'val_end_pct': 0.64,
        },
        {
            'name': 'Fold 2',
            'train_start_pct': 0.64,
            'train_end_pct': 0.76,
            'val_start_pct': 0.78,  # 2% gap
            'val_end_pct': 0.85,
        },
    ]

    # Convert percentages to day indices
    for fold in folds:
        fold['train_start'] = int(total_days * fold['train_start_pct'])
        fold['train_end'] = int(total_days * fold['train_end_pct'])
        fold['val_start'] = int(total_days * fold['val_start_pct'])
        fold['val_end'] = int(total_days * fold['val_end_pct'])

    # Holdout is 85%-100% (15% = ~9 months for 5 years)
    # v7.1 FIX: Increased from 10% to 15% for better statistical confidence
    holdout = {
        'name': 'Holdout',
        'start_pct': 0.85,
        'end_pct': 1.00,
        'start': int(total_days * 0.85),
        'end': total_days,
    }

    return folds, holdout


# =============================================================================
# SIMPLIFIED VALIDATION (No arbitrary constraints)
# =============================================================================

def check_validation_gate(
    train_result: Dict,
    val_result: Dict,
    min_val_pnl_ratio: float = 0.3,  # More lenient than v5
    min_val_rounds: int = 15         # More lenient than v5
) -> Tuple[bool, str]:
    """
    Check if a parameter combo passes the validation gate.

    V7 philosophy: Minimal constraints, let profit speak.
    Only checks:
    - val_pnl not severely worse than train_pnl (overfitting check)
    - Minimum rounds for statistical validity

    NO win rate check, NO drawdown check.
    """
    train_pnl = train_result.get('total_pnl', 0)
    val_pnl = val_result.get('total_pnl', 0)
    val_rounds = val_result.get('total_rounds', 0)

    # Check minimum rounds (statistical validity)
    if val_rounds < min_val_rounds:
        return False, f"Insufficient val rounds: {val_rounds} < {min_val_rounds}"

    # Check P&L ratio (overfitting check - only if train profitable)
    if train_pnl > 0:
        ratio = val_pnl / train_pnl
        if ratio < min_val_pnl_ratio:
            return False, f"Val/Train ratio too low: {ratio:.2f} < {min_val_pnl_ratio}"
    elif val_pnl <= 0:
        return False, "Both train and val P&L are non-positive"

    # NO win rate check
    # NO drawdown check

    return True, "Passed validation gate"


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
    """
    import math

    n = len(per_round_returns)
    corrected_alpha = family_alpha / n_total_tests

    if n < 2:
        return False, 1.0, corrected_alpha

    # Calculate t-statistic (testing if mean > 0)
    mean = sum(per_round_returns) / n
    variance = sum((r - mean) ** 2 for r in per_round_returns) / (n - 1)
    std_error = (variance / n) ** 0.5

    if std_error == 0:
        return False, 1.0, corrected_alpha

    t_stat = mean / std_error

    # Calculate p-value
    if n > 30:
        # Normal approximation
        z = t_stat
        if z > 0:
            p_value = 0.5 * math.erfc(z / math.sqrt(2))
        else:
            p_value = 1.0 - 0.5 * math.erfc(-z / math.sqrt(2))
    else:
        # Conservative estimate for small samples
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

    significant = p_value < corrected_alpha
    return significant, p_value, corrected_alpha


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_grid_summary(grid: Dict[str, List] = None):
    """Print a summary of the minimal grid."""
    if grid is None:
        grid = build_minimal_grid()

    print("=" * 60)
    print("MINIMAL PARAMETER GRID (v7)")
    print("=" * 60)

    total = 1
    for param, values in grid.items():
        n = len(values)
        total *= n
        print(f"  {param}: {values} ({n})")

    print("-" * 60)
    print(f"  Total combinations: {total:,}")
    print(f"  Bonferroni alpha (0.05): {0.05/total:.6f}")
    print("-" * 60)
    print("  Fixed parameters:")
    for param, value in list(FIXED_PARAMS.items())[:5]:
        print(f"    {param}: {value}")
    print(f"    ... and {len(FIXED_PARAMS) - 5} more")
    print("=" * 60)


if __name__ == "__main__":
    # Test the module
    print_grid_summary()

    print("\nGenerating first 5 combinations...")
    combos = generate_all_combinations()
    for i, combo in enumerate(combos[:5]):
        print(f"\n{i+1}. th={combo['threshold']}, tr={combo['trailing']}, "
              f"ps={combo['pyramid_step']}, mp={combo['max_pyramids']}, "
              f"vol={combo['vol_type']}, size={combo['size_schedule']}")

    print(f"\n...and {len(combos) - 5} more.")
    print(f"\nTotal combinations: {len(combos)}")
    print(f"Grid size check: {get_grid_size()}")
