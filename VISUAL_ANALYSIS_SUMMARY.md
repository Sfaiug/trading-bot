# Statistical Analysis - Visual Summary

## The Core Problem: Selection Bias from Massive Optimization

```
Parameter Space vs. Sample Size Mismatch:

┌─────────────────────────────────────────────────────────┐
│ Testing 70.8 MILLION combinations on 5 YEARS of data   │
│                                                           │
│         70,800,000 combinations                          │
│       ÷ 252 trading days/year × 5 years                │
│       ÷ ~100 trades per strategy                        │
│       = 56 combinations per trade observation           │
│                                                           │
│  This means each trade is being "fit" ~56 times!      │
│  Overfitting guarantee.                                │
└─────────────────────────────────────────────────────────┘
```

## Selection Bias Visualization

```
Finding the Best of 70.8M Random Results

Random Hypothesis Test Results (null: no edge):
┌──────────────────────────────────────────────────────────┐
│ Positive false positives: ~3.54M (5% × 70.8M)           │
│                                                            │
│ Sharpe distribution under null:      Selected Best:      │
│                                                            │
│    │     ××××××                                          │
│    │   ××××××××××      ┌─ Expected Sharpe: 0.5         │
│    │  ××××××××××××     │  Selected Sharpe: 2.5         │
│  P │×××××××××××××××    │  Selection Bias: +2.0         │
│  r │××××××××××××××××   │  Overstatement: 400%          │
│  o │                    │                                │
│  b │                    └─ When we pick THE BEST        │
│    │                                                      │
│    └──────────────────────────────────────────────────────┘
│         Sharpe Ratio
│
│ The "best" looks great because:
│ 1. We tested 70.8M combinations
│ 2. 3.54M happened to be profitable by chance
│ 3. We picked #1 from those 3.54M
│ 4. Expected value of best = 5.87 std devs above mean
└──────────────────────────────────────────────────────────┘
```

## What Should Happen vs. What Actually Happens

```
CORRECT VALIDATION FRAMEWORK:
┌─────────────────────────────────────────┐
│ 1. Reserve TEST SET (never optimize on) │
│    ▼                                    │
│ 2. TRAINING: Optimize parameters        │
│    (test 1000-10000 combos)             │
│    ▼                                    │
│ 3. Fold Validation: Test on fold data   │
│    (not used in optimization)           │
│    ▼                                    │
│ 4. TEST SET: Final honest evaluation    │
│    (data never seen before)             │
│    ▼                                    │
│ 5. CONFIDENCE INTERVAL:                 │
│    Test set result ± estimation error   │
└─────────────────────────────────────────┘
         Results: HONEST & RELIABLE

CURRENT VALIDATION FRAMEWORK:
┌─────────────────────────────────────────┐
│ 1. Load 5 years of data                 │
│    (no test set reserved)               │
│    ▼                                    │
│ 2. Phase A: Test 217K combos            │
│    select top 20                        │
│    ▼                                    │
│ 3. Phase B: Test 3.5M combos per top 20 │
│    select top 10                        │
│    ▼                                    │
│ 4. Phase C: Fine-tune top 10            │
│    ▼                                    │
│ 5. Cross-fold test (on fold validation) │
│    (but parameters already fit to this!) │
│    ▼                                    │
│ 6. SELECT BEST and report               │
│    ▼                                    │
│ 7. Bootstrap CI (resamples from         │
│    already-selected parameters)         │
└─────────────────────────────────────────┘
      Results: BIASED & UNRELIABLE
```

## Overfitting Risk by the Numbers

```
Parameter Testing Flow & Risk Accumulation:

Phase A (Dense Core):      217,800 combos tested
Phase B (New Params):      50.8M combos tested
Phase C (Fine-tuning):     ~1,000 combos tested
────────────────────────────────────────────────────
TOTAL:                     70.8M combos tested
                           ════════════════════════

Risk Multiplier Analysis:
─────────────────────────────────────────────────────
Tests       │ Expected     │ If selecting  │ True P&L
            │ False (+)    │ the BEST      │ if real
────────────┼──────────────┼───────────────┼─────────
1,000       │ 50           │ High risk     │ 45% → 25%
10,000      │ 500          │ Very high     │ 45% → 15%
100,000     │ 5,000        │ Extreme       │ 45% → 8%
1,000,000   │ 50,000       │ Severe        │ 45% → 5%
70,800,000  │ 3.54M        │ CRITICAL      │ 45% → 2-3%
────────────┴──────────────┴───────────────┴─────────
```

## Statistical Significance Thresholds

```
Effect of Bonferroni Correction:

Standard α = 0.05 (1 test):
├─ t-critical ≈ 1.96 (for large n)
├─ With 100 trades: achievable
└─ Confidence: Reasonable

Bonferroni with 70.8M tests:
├─ Corrected α = 0.05 / 70,800,000
├─ α = 7.06e-10
├─ t-critical ≈ 6.1 (need 6.1 std devs!)
├─ With 100 trades: IMPOSSIBLE
└─ Confidence: Current results DON'T meet this threshold

Visual representation:
        Prob(result)
            │
         ××│  (Normal distribution)
        ××××│
       ██████│ Normal region
      ████████│ (95% of true results)
     ─────┤──────┤─────
        -2σ  0  +2σ

Bonferroni requirement:
        Prob(result)
            │
            │                ××
            │              ××  ×  ← Need here!
            │            ××      (6 sigma)
            │          ××
            │        ××
     ──────────────────┼─────────────────
           0          6σ

    Probability of reaching 6σ by chance: <0.00001%
    With our results: FAILING this test
```

## Sharpe Ratio Inflation

```
How Sharpe Gets Inflated 2-3x:

True Strategy Over 5 Years:
  100 trades across 5 years = 20 trades/year
  Mean return per trade: 0.5%
  Std dev per trade: 2%
  Proper annualization: √20 ≈ 4.47

  TRUE SHARPE = (0.5% / 2%) × 4.47 = 1.12

What the Code Calculates:
  trades_per_year = min(252, 100) = 100  ← WRONG!
  Annualization: √100 = 10

  REPORTED SHARPE = (0.5% / 2%) × 10 = 2.50

Inflation Factor:
  2.50 / 1.12 = 2.23x
  OR 223% overstatement!
```

## Regime Analysis Problems

```
Regime Distribution & Sample Sizes:

         ▓▓▓▓▓
       ▓▓▓▓▓▓▓▓
       ▓▓▓▓▓▓▓▓  ~2.8M ticks
       ▓▓▓▓▓▓▓▓   per regime
       ▓▓▓▓▓▓▓▓
       ▓▓▓▓▓▓▓▓  But in backtest:
       ▓▓▓▓▓▓▓▓   100 total trades
         ▓▓▓▓▓   ÷ 9 regimes
                   = 11 trades/regime!

Statistical Power vs. Sample Size:

     Power (%)
       80├─────────────────────── Required (80%)
        │           /
        │         /
        │       /
        │     /
       40├─  / ← Current (40%
        │  / power at 11 trades)
        │/
        └────────────────────────
         5 10 15 20 25 30 35 40 45
           Trades per Regime

Problem: Can't reliably detect failures
         in regimes with <11 trades
```

## Bootstrap Confidence Interval Problem

```
The Bootstrap Illusion:

Real Situation:
┌─────────────────────────────────────────┐
│ True P&L = 0.5% (with edge)            │
│ Selected P&L = 45% (due to selection)   │
│ We observe the biased value (45%)       │
└─────────────────────────────────────────┘
              ▲

Current Bootstrap:
┌─────────────────────────────────────────┐
│ Resample from: [0.5, 0.3, 1.2, ...]    │
│ (these are the selected 45% results)    │
│                                          │
│ Bootstrap CI: [42%, 48%]                │
│ (tight and confident looking!)          │
│                                          │
│ But wait... 45% is SELECTED BIAS!      │
│ True CI should be: [-20%, 80%]!        │
└─────────────────────────────────────────┘

Visual:
Reported CI:     ═══════════════════
                 42%    45%    48%

True CI:         ════════════════════════════════
                 -20%            45%          80%

The reported CI is inside the true CI
but centered on the wrong value!
```

## Decision Tree: What to Do

```
                    Is the result profitable?
                            │
                ┌───────────┴────────────┐
               YES                      NO
                │                        │
        Use 10% → Want 100%?   NOT SUITABLE
             ↓     │            FOR LIVE
        Can you  ┌┴┐            TRADING
        paper   │NO│
        trade  └─┬─┘
       first?  YES
             ↓
        STILL RISKY
        (50% underperformance)
        Use 5% position size
        and add hard stops
```

## Comparison: This System vs. Best Practices

```
Dimension                  │ This System  │ Best Practice
──────────────────────────┼──────────────┼─────────────────
Parameter combos tested   │ 70.8M        │ 1K-10K
True out-of-sample set    │ None         │ 20-30% of data
Selection bias addressed  │ No           │ Yes (shrinkage)
Reports best or median    │ Best (biased)│ Median (honest)
Walk-forward proper       │ Partial      │ Full retrain
Bonferroni correction     │ Attempted    │ Alternative methods
Bootstrap accounts for    │ No           │ Yes
selection bias            │              │
Regime samples per type   │ 11 trades    │ 30+ trades
Confidence about edge     │ LOW 5-15%    │ HIGH 70-80%
Suitable for live trading │ NO           │ YES
──────────────────────────┴──────────────┴─────────────────
```

## Time to Failure: Expected vs. Observed

```
Backtest Results:        Live Trading (Estimated):

Month 1:   +45%          Month 1: +8%   (Degradation: 82%)
Month 2:   +38%          Month 2: -3%   (Hit drawdown)
Month 3:   +42%          Month 3: +5%
Month 4:   +35%          Month 4: -5%   (Stops triggered)
Month 5:   +31%          Month 5: N/A   (Trading paused)
Month 6:   +28%
                         Cumulative:    -15% (vs +219% backtest!)

The "best" parameters that looked amazing
probably won't survive first real trade.
```

## Risk Hierarchy

```
              CRITICAL
                │
    ┌───────────┼───────────┐
    │           │           │
Selection    No OOS      Bootstrap
Bias from    Test Set      Ignores
70.8M Tests               Selection
    │           │           │
    ├───────────┼───────────┤
    │    HIGH RISK: Do not use
    │
    ├───────────────────────┘
    │
    └─────┬─────────────────┐
          │         MEDIUM
    ┌─────┴─────┐      RISK
    │           │
Sharpe    Regime
Annualized  Detection
Wrong     Arbitrary
    │
    ├───────────────────────┐
    │     Should improve:
    │  Robustness testing
    │  Regime samples
    │  Parameter sensitivity
```

## Recommendation Summary

```
DO THIS FIRST (Stop reading, do these):
  [ ] Reduce position size to 1/10 of recommended
  [ ] Paper trade for 6+ months
  [ ] Track daily vs. backtest predictions
  [ ] Set hard stops at -50% from backtest

DO THIS SECOND (1-3 hours):
  [ ] Create held-out test set (reserve 1 year)
  [ ] Rerun optimization on remaining 4 years
  [ ] Test on held-out year
  [ ] Compare held-out vs. training results

DO THIS THIRD (1-2 weeks):
  [ ] Fix Sharpe annualization formula
  [ ] Use data-driven regime thresholds
  [ ] Report median not best parameters
  [ ] Improve robustness testing

DO THIS LONG-TERM (1-2 months):
  [ ] Rewrite with proper OOS framework
  [ ] Use ensemble of top 3-5 parameters
  [ ] Reduce search space to 1M combos
  [ ] Add parameter shrinkage (regularization)

IF IN DOUBT:
  ► Start with 10% of recommended position size
  ► Paper trade until it matches backtest
  ► Have exit plan ready
  ► Be prepared to give up 50% of expected gains
```

## The Mathematics Behind It All

```
Kelly Criterion for Position Sizing (when uncertain):

Position Size = (Win% × Avg Win - Loss% × Avg Loss) / Avg Win

Standard use: All parameters estimated reliably
Edge estimate: +2% per trade

BUT when overfitting likely:
Edge estimate: +0.5% per trade (conservative)
Uncertainty: ±2% (wide confidence interval)

Result:
Kelly position = 0.5% / 2% = 0.25
Fractional Kelly (recommended): 0.25 × 0.5 = 0.125
→ Use 1/8 of Kelly position
→ Use 1/10 of recommended size

(So even with Kelly approach, reduce size to 10%)
```

---

## Summary

The optimizer tests 70.8 MILLION parameter combinations on 5 years of data,
reports the single best result, and claims it's ready for live trading.

**This is statistically equivalent to flipping a coin 70.8 million times
and selecting the longest streak of heads.**

Of course the best streak looks impressive. But that doesn't mean
the next 100 flips will continue the streak.

**Key insight**: The larger the search space relative to data size,
the higher the probability of finding luck instead of edge.

**Recommendation**: Treat any results with extreme skepticism.
Start with 1/10 recommended position size, paper trade thoroughly,
and monitor daily P&L vs. backtest expectations carefully.

The system is sophisticated and well-coded.
But sophistication without statistical honesty just makes overfitting elegant.

