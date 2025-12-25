#!/usr/bin/env python3
"""
Validation Pipeline Orchestrator

PHASE 2: End-to-end validation pipeline that ties together all validation stages.

This module orchestrates:
1. Robustness validation (parameter perturbation)
2. Statistical validation (risk constraints)
3. Holdout validation (out-of-sample testing)
4. Regime robustness (performance across market conditions)
5. Monte Carlo validation (bootstrap + permutation)
6. Top 3 validation (overfitting guard)
7. Paper trading validation (live testnet comparison)

USAGE:
    from validation_pipeline import ValidationPipeline

    pipeline = ValidationPipeline(coin="BTCUSDT")
    results = pipeline.run_all_validations()

    if results['overall_pass']:
        print("Strategy approved for live trading!")
    else:
        print(f"Failed stages: {results['failed_stages']}")
"""

import os
import sys
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import validation modules
from core.monte_carlo import (
    run_monte_carlo_validation,
    MonteCarloResult,
    PermutationTestResult,
    format_monte_carlo_report
)
from paper_trading_validator import (
    PaperTradingConfig,
    PaperTradingResult,
    RoundResult,
    analyze_paper_trading_results,
    load_paper_trading_csv
)


# =============================================================================
# Configuration
# =============================================================================

# Validation thresholds
MIN_ROBUSTNESS_SCORE = 0.80       # 80% of perturbations must be profitable
MIN_WIN_RATE = 45.0               # Minimum 45% win rate
MAX_DRAWDOWN = 40.0               # Maximum -40% drawdown
MIN_SHARPE = 0.5                  # Minimum Sharpe ratio
MIN_ROUNDS = 50                   # Minimum trading rounds
TOP_N_VALIDATION = 3              # Require top N params to pass

# Monte Carlo thresholds
MC_MIN_PROB_POSITIVE = 0.80       # 80%+ probability of positive P&L
MC_MAX_PROB_RUIN = 0.05           # <5% probability of ruin
MC_MIN_SHARPE_5TH = 0.5           # 5th percentile Sharpe > 0.5
MC_PERMUTATION_P_VALUE = 0.05     # p-value >= 0.05 (sequence doesn't matter)

# Paper trading thresholds
PT_MIN_ROUNDS = 30                # Minimum paper trading rounds
PT_MAX_SLIPPAGE_RATIO = 2.0       # Max 2x expected slippage
PT_MAX_PNL_Z_SCORE = 2.0          # P&L within 2 std of MC median

# Paths
LOGS_DIR = "logs"
RESULTS_FILE = "validation_results.json"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StageResult:
    """Result from a single validation stage."""
    stage_name: str
    passed: bool
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


@dataclass
class ValidationResult:
    """Complete validation pipeline result."""
    coin: str
    params: Dict[str, float]
    timestamp: datetime
    overall_pass: bool
    passed_stages: List[str] = field(default_factory=list)
    failed_stages: List[str] = field(default_factory=list)
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    estimated_live_probability: float = 0.0
    recommendation: str = ""
    total_execution_time_ms: float = 0.0


# =============================================================================
# Validation Pipeline
# =============================================================================

class ValidationPipeline:
    """
    End-to-end validation pipeline for trading strategy parameters.

    Orchestrates all validation stages and produces a final decision.
    """

    def __init__(
        self,
        coin: str,
        params: Optional[Dict[str, float]] = None,
        round_returns: Optional[List[float]] = None,
        optimization_json: Optional[str] = None
    ):
        """
        Initialize the validation pipeline.

        Args:
            coin: Trading pair symbol (e.g., "BTCUSDT")
            params: Strategy parameters (if not loading from file)
            round_returns: P&L per round (if not loading from file)
            optimization_json: Path to optimization results JSON
        """
        self.coin = coin
        self.params = params or {}
        self.round_returns = round_returns or []
        self.optimization_json = optimization_json
        self.results: Dict[str, StageResult] = {}
        self._load_results_if_needed()

    def _load_results_if_needed(self):
        """Load optimization results from JSON if not provided."""
        if self.params and self.round_returns:
            return

        # Try to find optimization results
        json_path = self.optimization_json or f"{LOGS_DIR}/{self.coin}_v4_final_result.json"

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract parameters
            if 'winner' in data:
                winner = data['winner']
                self.params = {
                    'threshold': winner.get('threshold', 0),
                    'trailing': winner.get('trailing', 0),
                    'pyramid_step': winner.get('pyramid_step', 0),
                }

            # Extract round returns if available
            if 'holdout_result' in data and 'round_returns' in data['holdout_result']:
                self.round_returns = data['holdout_result']['round_returns']
            elif 'round_returns' in data:
                self.round_returns = data['round_returns']

            print(f"[Pipeline] Loaded results from {json_path}")
            print(f"[Pipeline] Parameters: {self.params}")
            print(f"[Pipeline] Round returns: {len(self.round_returns)} rounds")
        else:
            print(f"[Pipeline] Warning: No optimization results found at {json_path}")

    # =========================================================================
    # Stage 1: Robustness Validation
    # =========================================================================

    def validate_robustness(
        self,
        robustness_score: Optional[float] = None,
        min_threshold: float = MIN_ROBUSTNESS_SCORE
    ) -> StageResult:
        """
        Validate robustness score from parameter perturbation.

        A high robustness score means the strategy performs well even with
        ±10% parameter changes, reducing overfitting risk.
        """
        start_time = datetime.now()
        stage_name = "robustness"

        if robustness_score is None:
            # Try to load from optimization results
            json_path = self.optimization_json or f"{LOGS_DIR}/{self.coin}_v4_final_result.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                robustness_score = data.get('robustness_score', 0)

        # Handle missing data
        reasons = []
        if robustness_score is None:
            passed = False
            robustness_score = 0
            reasons.append("No robustness score available - run optimization first")
        else:
            passed = robustness_score >= min_threshold
            if not passed:
                reasons.append(f"Robustness score {robustness_score:.1%} < {min_threshold:.1%}")

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        result = StageResult(
            stage_name=stage_name,
            passed=passed,
            score=robustness_score or 0,
            details={'min_threshold': min_threshold},
            failure_reasons=reasons,
            execution_time_ms=elapsed
        )
        self.results[stage_name] = result
        return result

    # =========================================================================
    # Stage 2: Statistical Validation
    # =========================================================================

    def validate_statistical(
        self,
        total_pnl: Optional[float] = None,
        win_rate: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        num_rounds: Optional[int] = None
    ) -> StageResult:
        """
        Validate risk constraint compliance.

        Checks:
        - Positive P&L
        - Win rate >= MIN_WIN_RATE
        - Drawdown <= MAX_DRAWDOWN
        - Sufficient rounds
        """
        start_time = datetime.now()
        stage_name = "statistical"
        reasons = []

        # Load from optimization results if not provided
        if total_pnl is None:
            json_path = self.optimization_json or f"{LOGS_DIR}/{self.coin}_v4_final_result.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                winner = data.get('winner', {})
                total_pnl = winner.get('total_pnl', 0)
                win_rate = winner.get('win_rate', 0)
                max_drawdown = winner.get('max_drawdown_pct', 0)
                num_rounds = winner.get('rounds', 0)

        # Validation checks
        if total_pnl is not None and total_pnl <= 0:
            reasons.append(f"Total P&L {total_pnl:.2f}% is not positive")

        if win_rate is not None and win_rate < MIN_WIN_RATE:
            reasons.append(f"Win rate {win_rate:.1f}% < {MIN_WIN_RATE}%")

        if max_drawdown is not None and max_drawdown > MAX_DRAWDOWN:
            reasons.append(f"Max drawdown {max_drawdown:.1f}% > {MAX_DRAWDOWN}%")

        if num_rounds is not None and num_rounds < MIN_ROUNDS:
            reasons.append(f"Only {num_rounds} rounds < {MIN_ROUNDS} minimum")

        passed = len(reasons) == 0
        score = 1.0 if passed else (1.0 - len(reasons) / 4)  # Partial score

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        result = StageResult(
            stage_name=stage_name,
            passed=passed,
            score=score,
            details={
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'num_rounds': num_rounds
            },
            failure_reasons=reasons,
            execution_time_ms=elapsed
        )
        self.results[stage_name] = result
        return result

    # =========================================================================
    # Stage 3: Holdout Validation
    # =========================================================================

    def validate_holdout(
        self,
        holdout_pnl: Optional[float] = None,
        holdout_sharpe: Optional[float] = None
    ) -> StageResult:
        """
        Validate performance on holdout (out-of-sample) data.

        The holdout period (typically last 20% of data) was never seen
        during optimization, providing a true test of generalization.
        """
        start_time = datetime.now()
        stage_name = "holdout"
        reasons = []

        # Load from optimization results if not provided
        if holdout_pnl is None:
            json_path = self.optimization_json or f"{LOGS_DIR}/{self.coin}_v4_final_result.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                holdout = data.get('holdout_result', {})
                holdout_pnl = holdout.get('total_pnl', 0)
                holdout_sharpe = holdout.get('sharpe', 0)

        # Validation checks
        if holdout_pnl is not None and holdout_pnl <= 0:
            reasons.append(f"Holdout P&L {holdout_pnl:.2f}% is not positive")

        if holdout_sharpe is not None and holdout_sharpe < MIN_SHARPE:
            reasons.append(f"Holdout Sharpe {holdout_sharpe:.2f} < {MIN_SHARPE}")

        passed = len(reasons) == 0

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        result = StageResult(
            stage_name=stage_name,
            passed=passed,
            score=holdout_pnl if holdout_pnl else 0,
            details={
                'holdout_pnl': holdout_pnl,
                'holdout_sharpe': holdout_sharpe
            },
            failure_reasons=reasons,
            execution_time_ms=elapsed
        )
        self.results[stage_name] = result
        return result

    # =========================================================================
    # Stage 4: Regime Robustness
    # =========================================================================

    def validate_regime_robustness(
        self,
        regime_performance: Optional[Dict[str, float]] = None,
        min_regime_pnl: float = -5.0  # Allow up to -5% in worst regime
    ) -> StageResult:
        """
        Validate performance across different market regimes.

        The strategy should not have catastrophic losses in any regime.
        """
        start_time = datetime.now()
        stage_name = "regime"
        reasons = []

        # Load from optimization results if not provided
        if regime_performance is None:
            json_path = self.optimization_json or f"{LOGS_DIR}/{self.coin}_v4_final_result.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                regime_performance = data.get('regime_performance', {})

        if regime_performance:
            worst_regime = min(regime_performance.items(), key=lambda x: x[1])
            if worst_regime[1] < min_regime_pnl:
                reasons.append(
                    f"Worst regime '{worst_regime[0]}': {worst_regime[1]:.1f}% < {min_regime_pnl}%"
                )

        passed = len(reasons) == 0

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        result = StageResult(
            stage_name=stage_name,
            passed=passed,
            score=1.0 if passed else 0.5,
            details={'regime_performance': regime_performance},
            failure_reasons=reasons,
            execution_time_ms=elapsed
        )
        self.results[stage_name] = result
        return result

    # =========================================================================
    # Stage 5: Monte Carlo Validation
    # =========================================================================

    def validate_monte_carlo(
        self,
        round_returns: Optional[List[float]] = None,
        ruin_threshold: float = 0.40
    ) -> StageResult:
        """
        Run Monte Carlo stress testing on round returns.

        Uses bootstrap resampling and permutation testing to validate
        that the strategy's edge is statistically robust.
        """
        start_time = datetime.now()
        stage_name = "monte_carlo"
        reasons = []

        returns = round_returns or self.round_returns

        if not returns or len(returns) < 10:
            reasons.append(f"Insufficient rounds for Monte Carlo: {len(returns)} < 10")
            result = StageResult(
                stage_name=stage_name,
                passed=False,
                score=0,
                details={},
                failure_reasons=reasons,
                execution_time_ms=0
            )
            self.results[stage_name] = result
            return result

        # Run Monte Carlo validation
        mc_result, perm_result = run_monte_carlo_validation(
            round_returns=returns,
            params=self.params,
            n_bootstrap=10_000,
            n_permutation=5_000,
            ruin_threshold=ruin_threshold
        )

        # Check criteria
        if mc_result.probability_positive < MC_MIN_PROB_POSITIVE:
            reasons.append(
                f"P(positive) {mc_result.probability_positive:.1%} < {MC_MIN_PROB_POSITIVE:.1%}"
            )

        if mc_result.probability_ruin > MC_MAX_PROB_RUIN:
            reasons.append(
                f"P(ruin) {mc_result.probability_ruin:.1%} > {MC_MAX_PROB_RUIN:.1%}"
            )

        if mc_result.sharpe_5th_percentile < MC_MIN_SHARPE_5TH:
            reasons.append(
                f"Sharpe 5th pct {mc_result.sharpe_5th_percentile:.2f} < {MC_MIN_SHARPE_5TH}"
            )

        if perm_result.p_value < MC_PERMUTATION_P_VALUE:
            reasons.append(
                f"Permutation p-value {perm_result.p_value:.3f} < {MC_PERMUTATION_P_VALUE}"
            )

        passed = len(reasons) == 0

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        result = StageResult(
            stage_name=stage_name,
            passed=passed,
            score=mc_result.probability_positive,
            details={
                'mc_result': asdict(mc_result) if hasattr(mc_result, '__dataclass_fields__') else {},
                'perm_result': asdict(perm_result) if hasattr(perm_result, '__dataclass_fields__') else {},
            },
            failure_reasons=reasons,
            execution_time_ms=elapsed
        )
        self.results[stage_name] = result
        return result

    # =========================================================================
    # Stage 6: Top 3 Validation
    # =========================================================================

    def validate_top_3(
        self,
        top_3_all_pass: Optional[bool] = None,
        top_3_pass_count: Optional[int] = None
    ) -> StageResult:
        """
        Validate that top 3 parameter sets all pass risk constraints.

        If only the "best" parameter set passes, it may be an outlier due
        to selection bias from testing millions of combinations.
        """
        start_time = datetime.now()
        stage_name = "top_3"
        reasons = []

        # Load from optimization results if not provided
        if top_3_all_pass is None:
            json_path = self.optimization_json or f"{LOGS_DIR}/{self.coin}_v4_final_result.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                wf_result = data.get('walk_forward_result', {})
                top_3_all_pass = wf_result.get('top_3_all_pass', False)
                top_3_pass_count = wf_result.get('top_3_pass_count', 0)

        if top_3_all_pass is False:
            reasons.append(
                f"Only {top_3_pass_count}/3 top candidates pass risk constraints"
            )
            reasons.append("Possible overfitting - 'best' result may be statistical fluke")

        passed = top_3_all_pass is True

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        result = StageResult(
            stage_name=stage_name,
            passed=passed,
            score=1.0 if passed else top_3_pass_count / 3 if top_3_pass_count else 0,
            details={
                'top_3_all_pass': top_3_all_pass,
                'top_3_pass_count': top_3_pass_count
            },
            failure_reasons=reasons,
            execution_time_ms=elapsed
        )
        self.results[stage_name] = result
        return result

    # =========================================================================
    # Stage 7: Paper Trading Validation
    # =========================================================================

    def validate_paper_trading(
        self,
        paper_trading_csv: Optional[str] = None,
        mc_result: Optional[MonteCarloResult] = None
    ) -> StageResult:
        """
        Validate paper trading results against Monte Carlo expectations.

        Compares actual testnet performance to simulated expectations,
        checking for execution quality and P&L alignment.
        """
        start_time = datetime.now()
        stage_name = "paper_trading"
        reasons = []

        # Find most recent paper trading CSV
        if paper_trading_csv is None:
            pt_dir = Path("logs/paper_trading")
            if pt_dir.exists():
                csv_files = sorted(pt_dir.glob(f"{self.coin}_*.csv"), reverse=True)
                if csv_files:
                    paper_trading_csv = str(csv_files[0])

        if not paper_trading_csv or not os.path.exists(paper_trading_csv):
            reasons.append("No paper trading data found - run paper trading first")
            result = StageResult(
                stage_name=stage_name,
                passed=False,
                score=0,
                details={},
                failure_reasons=reasons,
                execution_time_ms=0
            )
            self.results[stage_name] = result
            return result

        # Load paper trading rounds
        round_results = load_paper_trading_csv(paper_trading_csv)

        if len(round_results) < PT_MIN_ROUNDS:
            reasons.append(
                f"Only {len(round_results)} paper trading rounds < {PT_MIN_ROUNDS} minimum"
            )

        # Get Monte Carlo result if not provided
        if mc_result is None and 'monte_carlo' in self.results:
            mc_details = self.results['monte_carlo'].details
            if 'mc_result' in mc_details and mc_details['mc_result']:
                # Reconstruct MC result for comparison
                mc_data = mc_details['mc_result']
                mc_result = MonteCarloResult(
                    iterations=mc_data.get('iterations', 0),
                    sample_size=mc_data.get('sample_size', 0),
                    pnl_median=mc_data.get('pnl_median', 0),
                    pnl_std=mc_data.get('pnl_std', 1),
                    pnl_5th_percentile=mc_data.get('pnl_5th_percentile', 0),
                    pnl_95th_percentile=mc_data.get('pnl_95th_percentile', 0),
                )

        if mc_result and round_results:
            # Compare to MC expectations
            config = PaperTradingConfig(
                symbol=self.coin,
                params=self.params,
                mc_result=mc_result
            )
            pt_result = analyze_paper_trading_results(config, round_results)

            if not pt_result.within_mc_confidence:
                reasons.append(
                    f"P&L {pt_result.total_pnl_pct:.1f}% outside MC 90% CI "
                    f"[{mc_result.pnl_5th_percentile:.1f}%, {mc_result.pnl_95th_percentile:.1f}%]"
                )

            if pt_result.avg_slippage_bps > config.expected_slippage_bps * PT_MAX_SLIPPAGE_RATIO:
                reasons.append(
                    f"Slippage {pt_result.avg_slippage_bps:.1f} bps > "
                    f"{config.expected_slippage_bps * PT_MAX_SLIPPAGE_RATIO:.1f} bps tolerance"
                )

        passed = len(reasons) == 0

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        result = StageResult(
            stage_name=stage_name,
            passed=passed,
            score=1.0 if passed else 0.5,
            details={
                'csv_file': paper_trading_csv,
                'num_rounds': len(round_results) if round_results else 0
            },
            failure_reasons=reasons,
            execution_time_ms=elapsed
        )
        self.results[stage_name] = result
        return result

    # =========================================================================
    # Run All Validations
    # =========================================================================

    def run_all_validations(
        self,
        skip_paper_trading: bool = False
    ) -> ValidationResult:
        """
        Run all validation stages and produce final result.

        Args:
            skip_paper_trading: Skip paper trading validation if not available

        Returns:
            ValidationResult with overall pass/fail and stage details
        """
        start_time = datetime.now()
        print("\n" + "=" * 70)
        print(f"VALIDATION PIPELINE - {self.coin}")
        print("=" * 70)

        # Run each stage
        stages = [
            ("Stage 1: Robustness", lambda: self.validate_robustness()),
            ("Stage 2: Statistical", lambda: self.validate_statistical()),
            ("Stage 3: Holdout", lambda: self.validate_holdout()),
            ("Stage 4: Regime", lambda: self.validate_regime_robustness()),
            ("Stage 5: Monte Carlo", lambda: self.validate_monte_carlo()),
            ("Stage 6: Top 3", lambda: self.validate_top_3()),
        ]

        if not skip_paper_trading:
            stages.append(
                ("Stage 7: Paper Trading", lambda: self.validate_paper_trading())
            )

        passed_stages = []
        failed_stages = []

        for stage_name, stage_func in stages:
            print(f"\n{stage_name}...")
            try:
                result = stage_func()
                status = "[PASS]" if result.passed else "[FAIL]"
                print(f"  {status} (score: {result.score:.2f})")
                if result.passed:
                    passed_stages.append(result.stage_name)
                else:
                    failed_stages.append(result.stage_name)
                    for reason in result.failure_reasons:
                        print(f"    - {reason}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                failed_stages.append(stage_name)

        # Calculate overall result
        overall_pass = len(failed_stages) == 0
        total_elapsed = (datetime.now() - start_time).total_seconds() * 1000

        # Estimate live profitability
        stage_weights = {
            'robustness': 0.15,
            'statistical': 0.15,
            'holdout': 0.15,
            'regime': 0.10,
            'monte_carlo': 0.25,
            'top_3': 0.10,
            'paper_trading': 0.10
        }

        weighted_score = sum(
            self.results.get(stage, StageResult(stage, False)).score * weight
            for stage, weight in stage_weights.items()
        )
        estimated_probability = min(0.95, weighted_score)

        # Generate recommendation
        if overall_pass:
            recommendation = "APPROVED: Strategy passes all validation stages. Ready for live trading."
        elif len(failed_stages) <= 2:
            recommendation = f"MARGINAL: Review failed stages ({', '.join(failed_stages)}) before live trading."
        else:
            recommendation = f"REJECTED: Too many failures ({len(failed_stages)} stages). Do not deploy."

        # Print summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
        print(f"Passed stages: {len(passed_stages)}/{len(stages)}")
        print(f"Failed stages: {failed_stages if failed_stages else 'None'}")
        print(f"Estimated live probability: {estimated_probability:.1%}")
        print(f"Recommendation: {recommendation}")
        print(f"Total time: {total_elapsed:.0f}ms")

        result = ValidationResult(
            coin=self.coin,
            params=self.params,
            timestamp=datetime.now(),
            overall_pass=overall_pass,
            passed_stages=passed_stages,
            failed_stages=failed_stages,
            stage_results=self.results,
            estimated_live_probability=estimated_probability,
            recommendation=recommendation,
            total_execution_time_ms=total_elapsed
        )

        # Save results
        self._save_results(result)

        return result

    def _save_results(self, result: ValidationResult):
        """Save validation results to JSON."""
        os.makedirs(LOGS_DIR, exist_ok=True)
        output_path = f"{LOGS_DIR}/{self.coin}_validation_result.json"

        # Convert to serializable dict
        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: serialize(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            return obj

        with open(output_path, 'w') as f:
            json.dump(serialize(result), f, indent=2)

        print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Run validation pipeline from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Validation Pipeline for Trading Strategy")
    parser.add_argument("coin", help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument("--json", help="Path to optimization results JSON")
    parser.add_argument("--skip-paper", action="store_true", help="Skip paper trading validation")

    args = parser.parse_args()

    pipeline = ValidationPipeline(
        coin=args.coin,
        optimization_json=args.json
    )

    result = pipeline.run_all_validations(skip_paper_trading=args.skip_paper)

    # Exit with appropriate code
    sys.exit(0 if result.overall_pass else 1)


if __name__ == "__main__":
    main()
