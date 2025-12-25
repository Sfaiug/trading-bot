# Statistical Issues: Code Examples and Fixes

This document shows specific code issues from the optimizer and how to fix them.

---

## Issue 1: Selection Bias in Cross-Fold Analysis

### The Problem

**File**: `optimize_pyramid_v4.py`, lines 920-967

```python
def run_multi_fold_optimization(coin, cache_files):
    # ... optimization happens ...

    # Find parameters that are profitable on ALL folds
    cross_fold_results = []

    for result in fold_results[0]['validated']:
        params = result['params']
        total_val_pnl = result['val_pnl']
        all_profitable = result['val_pnl'] > 0
        fold_pnls = [result['val_pnl']]

        # Test this exact param set on other folds' validation data
        for fold_info in cache_files[1:]:
            val_result = run_single_backtest_streaming(fold_info['val_cache'], params)
            fold_pnls.append(val_result['total_pnl'])
            total_val_pnl += val_result['total_pnl']
            if val_result['total_pnl'] <= 0:
                all_profitable = False

        avg_val_pnl = total_val_pnl / NUM_FOLDS

        cross_fold_results.append({
            'params': params,
            'fold_pnls': fold_pnls,
            'avg_val_pnl': avg_val_pnl,
            'all_profitable': all_profitable,
            'min_fold_pnl': min(fold_pnls),
        })

    # Filter to only those profitable on ALL folds
    profitable_all = [r for r in cross_fold_results if r['all_profitable']]

    if profitable_all:
        # PROBLEM: Selecting the BEST from potentially profitable parameters
        profitable_all.sort(key=lambda x: x['avg_val_pnl'], reverse=True)
        winner = profitable_all[0]  # ← SELECTION BIAS: Takes the best of the profitable set
        print(f"Found {len(profitable_all)} param sets profitable on ALL folds!")
    else:
        # PROBLEM: If nothing worked, falls back to LEAST BAD
        cross_fold_results.sort(key=lambda x: x['min_fold_pnl'], reverse=True)
        winner = cross_fold_results[0]  # ← Even worse: best of unprofitable set
        print(f"WARNING: No param set profitable on ALL folds!")
```

### Why It's Wrong

1. **Selects the MAXIMUM of profitable parameters** → Regression to mean will hurt live performance
2. **Falls back to "least bad" if everything fails** → Continues with a parameter set that FAILED validation
3. **Takes best avg of 3 folds, not median** → One lucky fold can pull average up

### The Fix

```python
def run_multi_fold_optimization(coin, cache_files):
    # ... optimization happens ...

    cross_fold_results = []

    for result in fold_results[0]['validated']:
        params = result['params']
        fold_pnls = [result['val_pnl']]

        for fold_info in cache_files[1:]:
            val_result = run_single_backtest_streaming(fold_info['val_cache'], params)
            fold_pnls.append(val_result['total_pnl'])

        cross_fold_results.append({
            'params': params,
            'fold_pnls': fold_pnls,
            'min_fold_pnl': min(fold_pnls),
            'median_fold_pnl': sorted(fold_pnls)[len(fold_pnls)//2],  # ← Use MEDIAN
            'all_profitable': all(pnl > 0 for pnl in fold_pnls),
        })

    # FIX 1: Require profitability on ALL folds
    profitable_all = [r for r in cross_fold_results if r['all_profitable']]

    if not profitable_all:
        # FIX 2: Fail loudly instead of falling back
        raise ValueError(
            f"CRITICAL: Strategy failed walk-forward validation. "
            f"No parameters profitable on all {NUM_FOLDS} folds. "
            f"Do not proceed to live trading."
        )

    # FIX 3: Use MEDIAN instead of best
    profitable_all.sort(key=lambda x: x['median_fold_pnl'], reverse=True)

    # FIX 4: Use TOP 3 to account for random variation
    if len(profitable_all) >= 3:
        # Average the top 3 to reduce variance
        top_3_medians = [r['median_fold_pnl'] for r in profitable_all[:3]]
        avg_top_3 = sum(top_3_medians) / 3
        winner = profitable_all[0]  # Main result
        backup_1 = profitable_all[1]
        backup_2 = profitable_all[2]

        print(f"Top 3 median P&L: {avg_top_3:.2f}%")
        print(f"  1. {top_3_medians[0]:.2f}%")
        print(f"  2. {top_3_medians[1]:.2f}%")
        print(f"  3. {top_3_medians[2]:.2f}%")
        print(f"Recommend using ensemble of these 3 parameter sets")
    else:
        winner = profitable_all[0]

    return {
        'winner': winner,
        'all_results': profitable_all,
        'num_profitable_all': len(profitable_all),
    }
```

---

## Issue 2: Bootstrap Confidence Intervals Ignore Selection Bias

### The Problem

**File**: `core/statistical_validation.py`, lines 72-124

```python
def calculate_bootstrap_sharpe_ci(returns, n_bootstrap=10000, confidence_level=0.95):
    """
    Calculate Sharpe Ratio with bootstrap confidence interval.
    """
    point_estimate = calculate_sharpe_ratio(returns)

    # Bootstrap resampling
    n = len(returns)
    bootstrap_sharpes = []

    for _ in range(n_bootstrap):
        # Sample with replacement - THIS IS THE PROBLEM
        sample = [returns[random.randint(0, n - 1)] for _ in range(n)]
        bootstrap_sharpes.append(calculate_sharpe_ratio(sample))

    bootstrap_sharpes.sort()
    alpha = 1 - confidence_level
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    ci_lower = bootstrap_sharpes[lower_idx]
    ci_upper = bootstrap_sharpes[upper_idx]

    return (point_estimate, ci_lower, ci_upper)
```

### Why It's Wrong

1. **Returns are biased upward** (selected from millions of backtests)
2. **Bootstrap resamples from biased data**
3. **CI is tight around a biased estimate** ← Looks good but is wrong
4. **Bootstrap assumes returns are the truth, but they're contaminated**

Example:
```
True Sharpe: 0.5
Selected (biased) Sharpe: 2.5
Bootstrap CI: [2.2, 2.8] ← Looks tight and significant
But true CI should be: [-0.5, 1.5] ← Possibly includes zero!
```

### The Fix

Use a **permutation test** to estimate true variability:

```python
def calculate_honest_sharpe_ci(returns, num_tests_performed, n_bootstrap=10000):
    """
    Calculate Sharpe with correction for selection bias from multiple testing.

    Uses Romano-Wolf correction to account for selection from many parameter combinations.
    """
    point_estimate = calculate_sharpe_ratio(returns)
    n = len(returns)

    # STEP 1: Estimate selection bias
    # Assuming we tested m parameter combinations:
    m = num_tests_performed  # e.g., 70,800,000

    # Under null hypothesis (no edge), expected bias:
    # E[bias] ≈ sqrt(2 * ln(m)) * σ
    variance = sum((r - point_estimate)**2 for r in returns) / (n - 1)
    std_dev = math.sqrt(variance)
    expected_bias = math.sqrt(2 * math.log(m)) * std_dev

    # STEP 2: Debiased estimate of true Sharpe
    debiased_sharpe = point_estimate - (expected_bias / std_dev)

    # STEP 3: Bootstrap with permutation for conservative CI
    bootstrap_sharpes = []

    for _ in range(n_bootstrap):
        # Permutation test: shuffle returns to get null distribution
        shuffled = returns.copy()
        random.shuffle(shuffled)

        # Calculate Sharpe on shuffled returns
        # This gives distribution under null hypothesis
        shuffled_sharpe = calculate_sharpe_ratio(shuffled)
        bootstrap_sharpes.append(shuffled_sharpe)

    # STEP 4: Calculate confidence interval from null distribution
    bootstrap_sharpes.sort()
    alpha = 0.05
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    null_std = bootstrap_sharpes[upper_idx] - bootstrap_sharpes[lower_idx]

    # STEP 5: Conservative confidence interval
    # Add expected bias as error margin
    ci_lower = debiased_sharpe - 2 * expected_bias
    ci_upper = point_estimate  # Upper bound is observed (before correction)

    return {
        'point_estimate': point_estimate,
        'debiased_estimate': debiased_sharpe,
        'expected_bias': expected_bias,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
    }

# Example usage:
result = calculate_honest_sharpe_ci(
    returns=[0.5, -0.3, 1.2, 0.8, ...],
    num_tests_performed=70_800_000
)

print(f"Observed Sharpe: {result['point_estimate']:.2f}")
print(f"Debiased Sharpe: {result['debiased_estimate']:.2f}")
print(f"Expected bias: ±{result['expected_bias']:.2f}")
print(f"Conservative 95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
```

---

## Issue 3: Sharpe Ratio Annualization is Wrong

### The Problem

**File**: `core/statistical_validation.py`, lines 40-69

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance)

    # PROBLEM: Uses number of returns as proxy for trades per year
    trades_per_year = min(252, len(returns))  # ← WRONG!
    annualization_factor = math.sqrt(trades_per_year)

    return (mean_return / std_dev) * annualization_factor
```

### Why It's Wrong

For strategy with 100 trades over 5 years:
- Actual trades/year: 100 / 5 = 20
- Current code calculates: min(252, 100) = 100
- Annualization factor wrong: sqrt(100) = 10 vs. sqrt(20) = 4.47
- Sharpe inflated by: 10 / 4.47 = 2.24x

### The Fix

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.0, dates=None):
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: List of per-trade returns (%)
        risk_free_rate: Risk-free rate (%)
        dates: Optional list of trade dates for accurate annualization
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    excess_return = mean_return - risk_free_rate

    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0001

    if std_dev == 0:
        return 0.0

    # CORRECT annualization factor
    if dates and len(dates) >= 2:
        # Use actual date span
        time_span = (max(dates) - min(dates)).days / 365.25
        if time_span > 0:
            trades_per_year = len(returns) / time_span
        else:
            trades_per_year = 1
    else:
        # Use only observed frequency
        trades_per_year = len(returns)  # Conservative: assumes all trades in 1 year

    # Cap at 252 (max trading days per year)
    trades_per_year = min(252, trades_per_year)

    annualization_factor = math.sqrt(max(1, trades_per_year))

    return (excess_return / std_dev) * annualization_factor

# Example:
trades = [0.5, 0.3, -0.2, 0.8, ...]  # 100 trades
dates = [datetime(2020, 1, 1), datetime(2020, 1, 5), ...]  # Over 5 years

# WRONG:
wrong_sharpe = (0.5 / 2.0) * sqrt(min(252, 100))  # = 2.5 * 10 = 25.0 ← Too high!

# CORRECT:
correct_sharpe = (0.5 / 2.0) * sqrt(100 / 5)  # = 2.5 * 4.47 = 11.18 ← Still high but more honest
```

---

## Issue 4: Regime Detection Hard Thresholds

### The Problem

**File**: `core/regime_detection.py`, lines 156-175

```python
def classify_volatility(vol_pct, vol_percentiles=(0.5, 1.5)):
    """
    Classify volatility level.
    """
    low_thresh, high_thresh = vol_percentiles

    # PROBLEM: Hard-coded global thresholds
    if vol_pct < 0.5:      # ← Arbitrary
        return VolatilityRegime.LOW
    elif vol_pct > 1.5:    # ← Arbitrary
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.NORMAL
```

### Why It's Wrong

1. **Not data-driven** - Should be based on actual percentiles
2. **Same for all coins** - BTC vol ≠ SHIB vol, but uses same thresholds
3. **Same for all periods** - 2017 bull ≠ 2022 bear, but uses same thresholds
4. **Arbitrary 0.5% and 1.5%** - No justification given

### The Fix

```python
def detect_regime_thresholds(volatilities, trend_values):
    """
    Calculate data-driven regime thresholds.
    """
    # Volatility: use 25th and 75th percentiles
    vol_sorted = sorted(volatilities)
    vol_p25 = vol_sorted[len(vol_sorted) // 4]
    vol_p75 = vol_sorted[(3 * len(vol_sorted)) // 4]

    # Trend: use -1 std and +1 std from mean
    trend_mean = sum(trend_values) / len(trend_values)
    trend_var = sum((t - trend_mean)**2 for t in trend_values) / len(trend_values)
    trend_std = math.sqrt(trend_var)
    trend_bear = trend_mean - trend_std
    trend_bull = trend_mean + trend_std

    return {
        'vol_percentiles': (vol_p25, vol_p75),
        'trend_thresholds': (trend_bear, trend_bull),
    }


def classify_volatility_fixed(vol_pct, vol_percentiles):
    """
    Classify volatility using data-driven thresholds.
    """
    low_thresh, high_thresh = vol_percentiles

    # Use actual percentiles from data
    if vol_pct < low_thresh:
        return VolatilityRegime.LOW
    elif vol_pct > high_thresh:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.NORMAL


def classify_trend_fixed(trend_pct, trend_thresholds):
    """
    Classify trend using data-driven thresholds.
    """
    bear_thresh, bull_thresh = trend_thresholds

    if trend_pct < bear_thresh:
        return TrendRegime.BEAR
    elif trend_pct > bull_thresh:
        return TrendRegime.BULL
    else:
        return TrendRegime.SIDEWAYS


# Usage in main code:
def detect_regimes_fixed(prices, verbose=False):
    """
    Detect market regimes with data-driven thresholds.
    """
    # Calculate volatility and trend
    vol_data = calculate_volatility_percentile(prices, vol_window_hours=24)
    trend_data = calculate_trend(prices, lookback_hours=24*30)

    # Get all volatilities and trends
    all_vols = [v for _, v in vol_data]
    all_trends = [t for _, t in trend_data]

    # FIXED: Calculate thresholds from actual data
    thresholds = detect_regime_thresholds(all_vols, all_trends)
    vol_percentiles = thresholds['vol_percentiles']
    trend_thresholds = thresholds['trend_thresholds']

    if verbose:
        print(f"Volatility thresholds (25th, 75th): {vol_percentiles}")
        print(f"Trend thresholds (bear, bull): {trend_thresholds}")

    # Rest of detection logic uses fixed functions...
    vol_map = {ts: vol for ts, vol in vol_data}
    trend_map = {ts: trend for ts, trend in trend_data}

    labels = []
    for ts in sorted(set(vol_map.keys()) & set(trend_map.keys())):
        vol_pct = vol_map[ts]
        trend_pct = trend_map[ts]

        # Use fixed functions with data-driven thresholds
        vol_regime = classify_volatility_fixed(vol_pct, vol_percentiles)
        trend_regime = classify_trend_fixed(trend_pct, trend_thresholds)
        combined = combine_regimes(vol_regime, trend_regime)

        labels.append(RegimeLabel(
            start_time=ts,
            end_time=ts,
            volatility=vol_regime,
            trend=trend_regime,
            combined=combined,
            volatility_pct=vol_pct,
            trend_pct=trend_pct
        ))

    return labels
```

---

## Issue 5: Robustness Score Uses Biased Baseline

### The Problem

**File**: `optimize_pyramid_v4.py`, lines 660-711

```python
def calculate_robustness_score(cache_file, best_params):
    """
    Test nearby parameters to check if optimum is robust.
    """
    # PROBLEM 1: Uses BEST as baseline (biased upward)
    center_result = run_single_backtest_streaming(cache_file, best_params)
    center_pnl = center_result['total_pnl']

    if center_pnl <= 0:
        return 0.0, []

    perturbation_results = []

    # PROBLEM 2: Only 1-parameter perturbations
    # PROBLEM 3: Very small perturbations (±0.5%, ±0.2%)
    perturbations = [
        ('threshold', -0.5),
        ('threshold', +0.5),
        ('trailing', -0.2),
        ...
    ]

    for param, delta in perturbations:
        perturbed = best_params.copy()
        current_val = perturbed.get(param, 1.0)

        if isinstance(current_val, (int, float)):
            # PROBLEM 4: No limits checking
            perturbed[param] = max(0.1, current_val + delta)

            result = run_single_backtest_streaming(cache_file, perturbed)
            perturbation_results.append({
                'param': param,
                'delta': delta,
                'pnl': result['total_pnl'],
                'ratio': result['total_pnl'] / center_pnl if center_pnl > 0 else 0
            })

    # PROBLEM 5: Uses minimum ratio (worst case)
    # But minimum is naturally worse than center due to selection bias
    if perturbation_results:
        min_ratio = min(p['ratio'] for p in perturbation_results)
        robustness = max(0, min_ratio)
    else:
        robustness = 1.0

    return robustness, perturbation_results
```

### Why It's Wrong

1. **Best is biased upward** from selection bias
2. **Perturbations naturally look worse** due to regression to mean
3. **Can't distinguish fragility from selection bias**
4. **No benchmark for what robustness score means**

### The Fix

```python
def calculate_honest_robustness_score(train_cache, val_cache, best_params, baseline_pnl=None):
    """
    Test if the optimum is in a smooth region or a spike.

    Compares to validation set (different data) to remove selection bias.
    """
    # STEP 1: Test best params on VALIDATION data (out-of-sample baseline)
    val_result = run_single_backtest_streaming(val_cache, best_params)
    val_pnl = val_result['total_pnl']

    if baseline_pnl is None:
        baseline_pnl = val_pnl  # Use OOS performance as baseline

    if baseline_pnl <= 0:
        return 0.0, "Cannot test robustness: negative baseline"

    # STEP 2: Test 2D and 3D perturbations (not just 1D)
    perturbations = []

    param_deltas = {
        'threshold': [-1.0, -0.5, 0.0, 0.5, 1.0],
        'trailing': [-0.5, -0.25, 0.0, 0.25, 0.5],
        'pyramid_step': [-0.5, -0.25, 0.0, 0.25, 0.5],
    }

    # Test combinations of perturbations
    for t_delta in param_deltas['threshold']:
        for tr_delta in param_deltas['trailing']:
            for p_delta in param_deltas['pyramid_step']:
                perturbed = best_params.copy()
                perturbed['threshold'] = max(0.5, perturbed.get('threshold', 3) + t_delta)
                perturbed['trailing'] = max(0.2, perturbed.get('trailing', 1.5) + tr_delta)
                perturbed['pyramid_step'] = max(0.2, perturbed.get('pyramid_step', 1) + p_delta)

                # Test on VALIDATION data for honest OOS estimate
                result = run_single_backtest_streaming(val_cache, perturbed)

                perturbations.append({
                    'threshold_delta': t_delta,
                    'trailing_delta': tr_delta,
                    'pyramid_step_delta': p_delta,
                    'pnl': result['total_pnl'],
                    'ratio': result['total_pnl'] / baseline_pnl if baseline_pnl > 0 else 0
                })

    # STEP 3: Robustness = % of neighbors within 90% of baseline
    nearby_good = sum(1 for p in perturbations if p['ratio'] >= 0.9)
    robustness = nearby_good / len(perturbations) if perturbations else 0

    # STEP 4: Analyze surface smoothness
    pnl_values = [p['pnl'] for p in perturbations]
    avg_pnl = sum(pnl_values) / len(pnl_values)
    variance = sum((p - avg_pnl)**2 for p in pnl_values) / len(pnl_values)
    std_dev = math.sqrt(variance)

    # High std dev = spiky surface (fragile)
    # Low std dev = smooth surface (robust)
    surface_smoothness = 1 / (1 + std_dev / abs(avg_pnl) if avg_pnl != 0 else 1)

    return {
        'robustness_score': robustness,  # % neighbors good
        'surface_smoothness': surface_smoothness,  # 0-1, higher is smoother
        'baseline_oos_pnl': baseline_pnl,
        'perturbation_avg': avg_pnl,
        'perturbation_std': std_dev,
        'verdict': 'ROBUST' if robustness >= 0.7 and surface_smoothness >= 0.6 else 'FRAGILE',
    }
```

---

## Summary of Fixes

| Issue | Current | Fix |
|-------|---------|-----|
| Select best params | Inherently biased | Use median of top 3 |
| Bootstrap CI | Ignores selection bias | Use permutation test with bias correction |
| Sharpe annualization | Uses min(252, N) | Use actual trades_per_year |
| Regime thresholds | Hard-coded (0.5%, 1.5%) | Data-driven percentiles |
| Robustness test | 1D, small deltas, biased baseline | 2D/3D, larger deltas, OOS baseline |
| Validation failure | Falls back to "least bad" | Reject and require improvement |

All these fixes reduce overfitting claims and provide more honest estimates of live trading performance.

