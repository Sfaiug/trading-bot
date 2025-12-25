# Validation Workflow Guide

This guide explains how to use the validation pipeline to ensure your trading strategy is ready for live deployment.

## Overview

The validation pipeline consists of 7 stages that must all pass before deploying to live trading:

1. **Robustness** - Parameter perturbation testing
2. **Statistical** - Risk constraint compliance
3. **Holdout** - Out-of-sample performance
4. **Regime** - Performance across market conditions
5. **Monte Carlo** - Bootstrap stress testing
6. **Top 3** - Overfitting guard
7. **Paper Trading** - Live testnet validation

## Quick Start

```bash
# Step 1: Run optimization
python optimize_pyramid_v4.py --coins BTCUSDT

# Step 2: Run validation pipeline
python validation_pipeline.py BTCUSDT

# Step 3: Run paper trading (4 weeks)
python main.py --mode trading --symbol BTCUSDT

# Step 4: Re-run validation with paper trading data
python validation_pipeline.py BTCUSDT

# Step 5: Final review
python final_review_checklist.py BTCUSDT
```

## Detailed Workflow

### Stage 1: Optimization

Run the optimizer to find optimal parameters:

```bash
python optimize_pyramid_v4.py --coins BTCUSDT
```

This will:
- Test 16,500+ parameter combinations
- Use 3-fold walk-forward validation
- Test on 20% holdout data
- Save results to `logs/BTCUSDT_v4_final_result.json`

### Stage 2: Validation Pipeline

Run the validation pipeline:

```bash
python validation_pipeline.py BTCUSDT
```

To skip paper trading validation (if not yet available):

```bash
python validation_pipeline.py BTCUSDT --skip-paper
```

### Stage 3: Paper Trading

Run paper trading on Binance Futures Testnet:

```bash
python main.py --mode trading --symbol BTCUSDT \
  --threshold 3.0 --trailing 1.5 --pyramid 2.0
```

Paper trading logs are automatically saved to:
- `logs/paper_trading/BTCUSDT_YYYYMMDD_HHMMSS.csv`

**Requirements:**
- Complete at least 30 rounds (approximately 2-4 weeks)
- Monitor for any issues during trading
- Use testnet, NOT live trading

### Stage 4: Iterative Refinement

If paper trading shows discrepancies, run refinement:

```bash
python paper_trading_validator.py --refine BTCUSDT
```

This will:
- Compare paper trading results to Monte Carlo expectations
- Identify model mismatches (e.g., higher slippage)
- Suggest parameter adjustments
- Re-validate with updated model

### Stage 5: Final Review

Run the final review checklist:

```bash
python final_review_checklist.py BTCUSDT
```

This interactive checklist confirms:
- All automated validation stages pass
- Manual risk management checks
- Proper configuration before live deployment

## Validation Criteria

### Monte Carlo Requirements
- P(positive P&L) >= 80%
- P(ruin) <= 5% (ruin = -40% drawdown)
- Sharpe ratio 5th percentile >= 0.5

### Paper Trading Requirements
- Complete 30+ rounds
- P&L within Monte Carlo 90% confidence interval
- Slippage ratio < 2x expected
- Zero liquidation events

### Risk Management
- Safety halt at -40% account equity
- Start with 25% of validated position size
- Scale up gradually over 4 weeks

## Troubleshooting

### "No optimization results found"
Run the optimizer first:
```bash
python optimize_pyramid_v4.py --coins BTCUSDT
```

### "Insufficient rounds for Monte Carlo"
The strategy needs more trading rounds. Ensure you've run paper trading for sufficient time.

### "Monte Carlo P(ruin) too high"
The strategy has too much drawdown risk. Consider:
- Reducing position size
- Increasing trailing stop
- Using more conservative parameters

### "Top 3 validation failed"
Only one parameter set passes, suggesting overfitting. Consider:
- Using more historical data
- Reducing parameter grid complexity
- Using stricter validation criteria

### "Paper trading P&L outside MC confidence interval"
Actual results don't match backtest. Consider:
- Adjusting slippage assumptions
- Re-running optimization with updated execution model
- Extending paper trading period

## Output Files

| File | Description |
|------|-------------|
| `logs/SYMBOL_v4_final_result.json` | Optimization results |
| `logs/SYMBOL_validation_result.json` | Validation pipeline results |
| `logs/paper_trading/SYMBOL_*.csv` | Paper trading logs |
| `logs/SYMBOL_APPROVAL_RECORD.json` | Final approval record |

## Safety Reminders

1. **Never skip paper trading** - Always validate on testnet first
2. **Start small** - Use 25% of validated size initially
3. **Monitor closely** - Watch for deviation from expectations
4. **Set kill switches** - Configure -40% equity safety halt
5. **Document everything** - Keep records of all decisions

## Contact

For issues or questions, check the logs and validation results first.
If problems persist, review the parameter reasonableness and consider re-optimization.
