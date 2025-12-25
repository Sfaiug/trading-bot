# Statistical Validation Checklist

Use this checklist before deploying any parameters from the optimizer to live trading.

---

## Pre-Deployment Assessment

### Phase 1: Honest About Overfitting Risk

- [ ] **Read the analysis files**
  - [ ] STATISTICAL_ANALYSIS_SUMMARY.txt (overview)
  - [ ] STATISTICAL_VALIDITY_REPORT.md (deep dive)
  - [ ] STATISTICAL_ISSUES_CODE_EXAMPLES.md (code fixes)

- [ ] **Acknowledge the risks**
  - [ ] Understand that 70.8M combinations tested = high overfitting risk
  - [ ] Acknowledge that selection bias could inflate results 10-20x
  - [ ] Accept that confidence in reported edge is <15%

- [ ] **Do NOT assume**
  - [ ] ❌ "The parameters will work as-is on live data"
  - [ ] ❌ "The backtest P&L is my expected live P&L"
  - [ ] ❌ "The confidence interval is tight enough to trust"
  - [ ] ❌ "The regime validation proves robustness"

---

## Phase 2: Code Improvements (Minimal Effort)

Complete these before using results at all:

- [ ] **Fix Sharpe Annualization** (30 minutes)
  - [ ] Replace `min(252, len(returns))` with actual trades_per_year
  - [ ] Calculate actual trading frequency from dates
  - [ ] Recalculate Sharpe ratio - expect it to drop 40-60%
  - [ ] File: `core/statistical_validation.py`, lines 40-69

- [ ] **Create Held-Out Test Set** (30 minutes)
  - [ ] Reserve final year of data
  - [ ] NEVER use this year in ANY optimization phase
  - [ ] Optimize only on first 4 years
  - [ ] Test ONLY on final year for honest evaluation
  - [ ] Files: `optimize_pyramid_v4.py` main flow

- [ ] **Use Median Not Best** (15 minutes)
  - [ ] Find top 3 parameter sets that work on ALL folds
  - [ ] Use median of top 3, not the single best
  - [ ] Calculate expected performance as: median ± std of top 3
  - [ ] File: `optimize_pyramid_v4.py`, lines 951-964

- [ ] **Stricter Validation Failure** (15 minutes)
  - [ ] If no parameters work on ALL folds, raise exception
  - [ ] Remove the fallback to "least bad" parameters
  - [ ] Require explicit approval to proceed without this
  - [ ] File: `optimize_pyramid_v4.py`, lines 1230-1238

---

## Phase 3: Medium-Term Improvements (1-2 weeks)

Implement these for better confidence:

- [ ] **Fix Regime Thresholds** (2 hours)
  - [ ] Replace hard-coded (0.5%, 1.5%) with data-driven percentiles
  - [ ] Calculate 25th/75th percentiles from actual volatility
  - [ ] Make thresholds coin-specific and period-specific
  - [ ] File: `core/regime_detection.py`, lines 156-175

- [ ] **Improve Robustness Testing** (2 hours)
  - [ ] Test 2D/3D parameter perturbations, not just 1D
  - [ ] Use validation set (different data) as baseline
  - [ ] Test larger perturbations (±10-20% of parameter range)
  - [ ] File: `optimize_pyramid_v4.py`, lines 660-711

- [ ] **Require Minimum Regime Samples** (1 hour)
  - [ ] Verify minimum 30 trades per regime
  - [ ] Skip regimes with <30 samples
  - [ ] Fail validation if any regime is untested
  - [ ] File: `core/regime_detection.py`, lines 322-408

- [ ] **Data-Driven Regime Thresholds** (2 hours)
  - [ ] Calculate vol/trend percentiles from data
  - [ ] Use 25th/75th percentiles instead of hard values
  - [ ] File: `core/regime_detection.py`, entire file

---

## Phase 4: Paper Trading Preparation

Before touching real money:

- [ ] **Select Parameters Carefully**
  - [ ] Get median of top 3 (not the single best)
  - [ ] Check robustness score (need >0.7)
  - [ ] Verify regime coverage (profitable in 6+ of 9 regimes)
  - [ ] Review cross-fold validation (works on all 3 folds)

- [ ] **Set Conservative Position Sizes**
  - [ ] Start with 1/10 of recommended size
  - [ ] If backtest recommends 10 BTC, start with 1 BTC
  - [ ] Gradually increase only if live results match backtest

- [ ] **Prepare Monitoring Dashboard**
  - [ ] Track daily P&L vs. expected
  - [ ] Track monthly P&L vs. backtest
  - [ ] Track maximum drawdown
  - [ ] Set hard stop at -50% from expected baseline

- [ ] **Create Exit Plan**
  - [ ] If live P&L < backtest P&L × 0.2, stop trading
  - [ ] If drawdown > expected max_dd × 2, stop trading
  - [ ] If consecutive losses exceed 3× backtest avg loss, review
  - [ ] Kill switch if any risk metrics breached

- [ ] **Set Up Alerts**
  - [ ] Daily P&L vs. expected: Alert if >±50% deviation
  - [ ] Win rate vs. expected: Alert if <50% of expected
  - [ ] Trade frequency vs. expected: Alert if >2x expected

---

## Phase 5: Paper Trading (6-12 months minimum)

Do NOT skip this:

- [ ] **Month 1-2: Baseline**
  - [ ] Run paper trading for 2 full months
  - [ ] Compare daily P&L to backtest predictions
  - [ ] Track deviation from expected returns
  - [ ] Expected: ±30-40% variance in monthly returns

- [ ] **Month 3-6: Pattern Recognition**
  - [ ] Look for consistent deviations from backtest
  - [ ] Are certain markets worse than expected?
  - [ ] Are certain regimes different than predicted?
  - [ ] Is volatility higher/lower than backtest?

- [ ] **Month 6-12: Validation**
  - [ ] If cumulative = expected: Can consider live
  - [ ] If cumulative < 0.5× expected: Parameters likely overfit
  - [ ] If cumulative > expected: Possible data snooping in backtest
  - [ ] If results diverge by regime: Regime parameters wrong

- [ ] **Track Key Metrics**
  - [ ] Win rate: Compare to backtest prediction
  - [ ] Avg win/loss: Compare to backtest prediction
  - [ ] Sharpe ratio: Compare to backtest prediction
  - [ ] Max drawdown: Compare to backtest prediction

---

## Phase 6: Live Trading Decision

Only proceed if conditions met:

### APPROVED for live trading if ALL true:
- [ ] Code improvements (Phase 2) completed
- [ ] Medium improvements (Phase 3) completed
- [ ] 6+ months paper trading completed
- [ ] Paper trading P&L >= 70% of backtest P&L
- [ ] All regimes profitable in paper trading
- [ ] No systemic deviations from expected P&L
- [ ] Risk management systems verified
- [ ] Hard stops and kills switches tested

### DO NOT proceed if ANY true:
- [ ] Paper trading P&L < 30% of backtest
- [ ] Regime profitability reversed (e.g., bullish became bearish)
- [ ] Volatility environment drastically different
- [ ] Drawdown exceeded backtest worst-case by 2x
- [ ] Win rate dropped below 40%
- [ ] Consecutive loss streaks exceeded 5 trades

### ALTERNATIVE: Reduce and Monitor

If paper trading results are 50-70% of backtest:
- [ ] Proceed with 1/20 of recommended position
- [ ] Add additional hard stops
- [ ] Increase monitoring frequency
- [ ] Plan review cycle: Weekly, not monthly

---

## Live Trading Protocol

### Pre-Trade Checklist
Before going live, verify:

- [ ] Parameters set correctly in code
- [ ] Position size correct (conservative estimate)
- [ ] Risk management active (hard stops, heat checks)
- [ ] Monitoring dashboard live
- [ ] Alerts configured and tested
- [ ] Kill switch accessible and practiced
- [ ] Funding plan clear (no over-leverage)
- [ ] Team/advisor notified of plan

### Daily Operations
- [ ] Check P&L vs. expected range
- [ ] Verify all positions correctly sized
- [ ] Review any regime changes
- [ ] Log any unusual market conditions
- [ ] Flag any algorithmic errors

### Weekly Review
- [ ] Compare weekly P&L to backtest
- [ ] Review drawdown
- [ ] Check regime distribution
- [ ] Verify all systems operational

### Monthly Review
- [ ] Compare monthly P&L to backtest
- [ ] Calculate Sharpe ratio
- [ ] Assess regime-by-regime performance
- [ ] Decision: Continue, reduce, or stop?

---

## Red Flags (Stop Immediately If Any Occur)

- [ ] **Drawdown exceeds 2× backtest maximum**
  - Action: Close all positions, investigate
  - Don't re-enter without understanding why

- [ ] **P&L trends consistently below 50% of expected**
  - Action: Stop trading, rerun analysis
  - Likely indicates overfitting or changed market

- [ ] **Win rate drops below 30%**
  - Action: Halt trading immediately
  - Indicates strategy fundamentally broken

- [ ] **Consecutive losses exceed 5 trades**
  - Action: Close positions, verify parameters
  - May signal parameter drift or market change

- [ ] **Volatility environment changed drastically**
  - Action: Reduce position size immediately
  - Backtests may be invalid in new regimes

- [ ] **Any systematic error detected**
  - Action: Fix bug, pause trading
  - Resume only after verification

- [ ] **Market disruption (flash crash, exchange issue)**
  - Action: Assess impact, pause trading
  - Resume only if systems verified

---

## Post-Mortem Analysis (If Things Go Wrong)

If live trading underperforms significantly:

- [ ] **Collect Data**
  - [ ] All trades with entry/exit prices
  - [ ] Market conditions (volatility, trends)
  - [ ] Regime at each trade
  - [ ] Slippage and fees paid

- [ ] **Compare to Backtest**
  - [ ] Are regimes different? (e.g., more high-vol-bear)
  - [ ] Is slippage different? (expect 2-5x more in live)
  - [ ] Are fees different? (maker/taker spreads matter)
  - [ ] Did execution match expectations?

- [ ] **Identify Root Cause**
  - [ ] Overfitting: P&L < 30% of backtest
  - [ ] Changed market conditions: Wrong regime mix
  - [ ] Parameter drift: Best params no longer optimal
  - [ ] Implementation error: Code bug or wrong order type

- [ ] **Decide Next Steps**
  - [ ] If overfitting: Stop trading, improve methodology
  - [ ] If regime change: Reoptimize on recent data
  - [ ] If implementation error: Fix and retry
  - [ ] If genuine market change: Reduce size and monitor

---

## Document Checklist

Keep this documentation:

- [ ] **Backtest Configuration**
  - [ ] Date range used
  - [ ] Symbol tested
  - [ ] Parameter ranges
  - [ ] Walk-forward fold boundaries

- [ ] **Results Documentation**
  - [ ] Best parameters found
  - [ ] Training P&L (per fold)
  - [ ] Validation P&L (per fold)
  - [ ] Cross-fold analysis

- [ ] **Risk Assessment**
  - [ ] Maximum drawdown
  - [ ] Win rate
  - [ ] Sharpe ratio
  - [ ] Per-regime performance

- [ ] **Paper Trading Results**
  - [ ] Monthly returns (actual vs. expected)
  - [ ] Regime distribution
  - [ ] Drawdowns experienced
  - [ ] Actual vs. predicted Sharpe

- [ ] **Live Trading Journal**
  - [ ] Start date and parameters
  - [ ] Position sizes
  - [ ] Daily/weekly P&L tracking
  - [ ] Any issues encountered
  - [ ] Stop date (if applicable)

---

## Timeline Summary

```
Recommended Timeline for Going Live:

NOW:
  ├─ Read analysis documents (2-3 hours)
  └─ Decide: Improve code or accept risks

WEEK 1: Code Improvements (if proceeding)
  ├─ Fix Sharpe annualization (30 min)
  ├─ Create held-out test set (30 min)
  ├─ Use median not best (15 min)
  └─ Stricter validation (15 min)

WEEK 2-4: Medium Improvements (optional)
  ├─ Data-driven regime thresholds (2 hours)
  ├─ Improved robustness testing (2 hours)
  └─ Minimum regime samples (1 hour)

MONTH 2-3: Paper Trading Setup
  ├─ Select conservative parameters
  ├─ Set position sizes (1/10 recommended)
  ├─ Create monitoring dashboard
  └─ Prepare risk management

MONTH 3-8: Paper Trading Phase 1
  ├─ Run 6 months of paper trading
  ├─ Weekly monitoring and review
  └─ Compare to backtest predictions

MONTH 8-9: Decision Point
  ├─ Analyze paper trading results
  ├─ Decide: Live, adjust, or stop?
  └─ If live: final preparation

MONTH 10+: Potential Live Trading
  ├─ Small position size initially (1/10 recommended)
  ├─ Daily monitoring
  ├─ Monthly review
  └─ Scale up slowly if results match backtest

Total: 10 months minimum before considering live deployment
```

---

## Key Question to Answer Before Proceeding

**"If this strategy doesn't work in live trading, can I afford the loss?"**

If NO: Don't proceed. Wait until methodology improved.
If YES: Can only afford 1/10 of maximum position? Use only that.

**Default assumption**: Backtest results will degrade 50-80% in live trading.
Plan accordingly.

---

## Final Checklist

- [ ] I have read all analysis documents
- [ ] I understand the selection bias issue
- [ ] I understand why bootstrap CI is unreliable
- [ ] I understand the held-out test set importance
- [ ] I will NOT deploy at full position size
- [ ] I will paper trade for 6+ months first
- [ ] I have exit plan and kill switch ready
- [ ] I can afford potential loss
- [ ] I have backup funding if strategy fails
- [ ] I am prepared for 50-80% drawdown from expected

**If you checked all boxes**: You are ready to proceed cautiously.
**If you unchecked any box**: Do NOT deploy to live trading.

---

Date of deployment: _______________
Initial position size: _______________
Conservative position size (1/10): _______________
Maximum acceptable loss: _______________
Kill switch trigger (P&L level): _______________

Signature / Approval: _______________

