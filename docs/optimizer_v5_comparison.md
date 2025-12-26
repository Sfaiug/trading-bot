# Optimizer v5 vs v4 Comparison

## Executive Summary

The v5 optimizer addresses fundamental statistical validity issues identified in v4 that were causing false positives from data mining.

## Key Differences

| Aspect | v4 (Old) | v5 (New) |
|--------|----------|----------|
| **Grid Size** | ~86 million combinations | 5,120 combinations |
| **Search Structure** | Hierarchical (Phase A → B → C) | Single-stage (all at once) |
| **vol_type Testing** | Locked to 'none' in Phase A | Tested with core params |
| **Bonferroni** | Applied AFTER selection | Applied DURING selection |
| **Validation Gate** | None (post-hoc only) | Immediate train→val check |
| **Per-Round Returns** | Optional (return_rounds=False) | Always stored |
| **Fold Structure** | Expanding windows | Fixed-size with 5% gaps |
| **Expected False Positives** | ~170,000 | 0.05 (controlled) |

## Statistical Validity Improvements

### 1. Grid Size Reduction (17,000x smaller)
- **v4**: Tests 86M+ combinations
- **v5**: Tests 5,120 combinations
- **Why**: With 86M tests at α=0.05, expect 4.3M false positives. With 5K tests, expect 256.

### 2. Single-Stage Search
- **v4**: Phase A finds "best" with vol_type='none', then Phase B refines
- **v5**: All parameters tested simultaneously
- **Why**: Parameters interact. Optimizing core params with vol_type='none' finds noise-tuned optima.

### 3. Bonferroni During Selection
- **v4**: Run all tests → pick winner → apply Bonferroni to winner only
- **v5**: After each combo, check if p-value < α/5120
- **Why**: Post-hoc Bonferroni is rationalization, not protection.

### 4. Immediate Validation Gate
- **v4**: No validation until final holdout test
- **v5**: Each combo tested on validation immediately after training
- **Why**: Combos that overfit training are rejected before proceeding.

### 5. Fixed-Size Folds with Gaps
```
v5 Fold Structure:
Fold 0: Train 0%-30%, [5% gap], Val 35%-45%
Fold 1: Train 20%-50%, [5% gap], Val 55%-65%
Fold 2: Train 40%-70%, [5% gap], Val 75%-85%
Holdout: 90%-100%
```
- **v4**: Expanding windows with potential overlap
- **v5**: Fixed 30% training, 5% gap, 10% validation
- **Why**: Gaps prevent look-ahead bias between train/val.

### 6. Real Per-Round Returns
- **v4**: `return_rounds=False` generates synthetic `[avg_pnl] * n`
- **v5**: Always stores actual round P&L to disk
- **Why**: Monte Carlo on synthetic returns is meaningless validation.

## Running the v5 Optimizer

```bash
# Full 5-year optimization
python optimize_pyramid_v5.py --symbol BTCUSDT --days 1825

# Quick test (1 year)
python optimize_pyramid_v5.py --symbol BTCUSDT --days 365 --output ./test_results

# Multiple symbols
for symbol in BTCUSDT ETHUSDT SOLUSDT; do
    python optimize_pyramid_v5.py --symbol $symbol --days 1825
done
```

## Output Files

```
optimization_v5_results/
├── disk_storage/           # Per-round returns for each combo
│   ├── _index.json
│   └── {hash}.json.gz
├── BTCUSDT_top10_v5.json   # Top 10 parameter combinations
└── BTCUSDT_winner_v5.json  # Best parameters with statistics
```

## Interpreting Results

### Significant Combo
```json
{
  "params": {"threshold": 4.0, "trailing": 1.0, "vol_type": "stddev", ...},
  "metrics": {
    "train": {"total_pnl": 45.2, "win_rate": 58.3},
    "val": {"total_pnl": 28.1, "win_rate": 52.1},
    "holdout": {"total_pnl": 31.5, "win_rate": 54.7}
  },
  "statistics": {
    "p_value": 2.3e-08,
    "sharpe": 1.82,
    "is_significant": true
  }
}
```

### What to Look For
1. **p-value < 9.77e-06** (Bonferroni threshold for 5,120 tests)
2. **val_pnl >= 50% of train_pnl** (no severe overfitting)
3. **holdout_pnl > 0** (profitable on unseen data)
4. **Sharpe > 0.5** (risk-adjusted returns)

## Expected Outcomes

With proper statistical controls, expect:
- **Fewer "winning" combos**: 10-50 instead of thousands
- **Lower reported returns**: No more data-mined 500% returns
- **Higher confidence**: Each winner is statistically significant
- **Better live performance**: Parameters that survive rigorous validation

## Migration from v4

If you have existing v4 results:
1. Keep v4 results for comparison
2. Run v5 on same symbol/timeframe
3. Compare winning parameters
4. v5 winners should be subset of v4 (only truly robust ones survive)

## Files Modified

- `core/disk_results.py` - New disk storage module
- `core/balanced_grid.py` - New balanced grid module
- `backtest_pyramid.py` - Always returns per-round data
- `optimize_pyramid_v5.py` - New statistically valid optimizer
