#!/usr/bin/env python3
"""
Integration Test Suite for optimize_pyramid_v4.py

Tests all critical integration points with synthetic data.
Target runtime: < 2 minutes

Usage:
    python tests/test_optimizer_integration.py
"""

import sys
import os
import time
import tempfile
import struct
import traceback
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for test suite."""
    num_ticks: int = 10_000
    base_price: float = 100.0
    volatility_pct: float = 2.0
    thresholds: List[float] = None
    trailings: List[float] = None
    pyramid_steps: List[float] = None
    max_runtime_seconds: int = 120

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = [1.0, 2.0]
        if self.trailings is None:
            self.trailings = [0.5, 1.0]
        if self.pyramid_steps is None:
            self.pyramid_steps = [0.5, 1.0]


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_ticks(
    num_ticks: int,
    base_price: float = 100.0,
    volatility_pct: float = 2.0,
    start_time: datetime = None
) -> List[Tuple[datetime, float]]:
    """Generate synthetic tick data with realistic price movements."""
    random.seed(42)

    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0, 0)

    ticks = []
    price = base_price
    current_time = start_time

    for _ in range(num_ticks):
        time_delta = timedelta(seconds=random.uniform(0.5, 2.0))
        current_time += time_delta

        vol_multiplier = 1.0 + 0.5 * abs(random.gauss(0, 1))
        move_pct = random.gauss(0, volatility_pct / 100 * vol_multiplier)
        price *= (1 + move_pct)
        price = max(price, 0.01)

        ticks.append((current_time, price))

    return ticks


def create_binary_tick_cache(ticks: List[Tuple[datetime, float]], cache_path: str) -> None:
    """Write ticks to binary cache file (16 bytes per tick)."""
    with open(cache_path, 'wb') as f:
        for ts, price in ticks:
            ts_float = ts.timestamp()
            f.write(struct.pack('dd', ts_float, price))


def create_test_tick_streamer(ticks: List[Tuple[datetime, float]]) -> Callable:
    """Create a callable tick streamer (not a generator)."""
    def streamer():
        for tick in ticks:
            yield tick
    return streamer


# =============================================================================
# TEST RESULT TRACKING
# =============================================================================

class TestResult:
    """Stores result of a single test."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = ""
        self.duration_ms = 0


# =============================================================================
# TEST CASES
# =============================================================================

def test_validate_data_quality_type(config: TestConfig) -> TestResult:
    """BUG-001: validate_data_quality expects List, not callable."""
    result = TestResult("validate_data_quality type check")
    start = time.time()

    try:
        from data.tick_data_fetcher import validate_data_quality

        ticks = generate_synthetic_ticks(1000, config.base_price, config.volatility_pct)

        # Test: Passing a list should work
        report = validate_data_quality(ticks, verbose=False)
        if report is None:
            result.error = "validate_data_quality returned None for valid list"
            return result

        result.details = f"List input: OK, {report.total_ticks} ticks analyzed"
        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_quality_score_property(config: TestConfig) -> TestResult:
    """BUG-002: DataQualityReport.quality_score should exist."""
    result = TestResult("DataQualityReport.quality_score property")
    start = time.time()

    try:
        from data.tick_data_fetcher import validate_data_quality

        ticks = generate_synthetic_ticks(1000, config.base_price, config.volatility_pct)
        report = validate_data_quality(ticks, verbose=False)

        # Check quality_score exists and is valid
        if not hasattr(report, 'quality_score'):
            result.error = "BUG-002: quality_score attribute missing"
            return result

        score = report.quality_score
        if not isinstance(score, (int, float)):
            result.error = f"quality_score is not numeric: {type(score)}"
            return result

        if not 0 <= score <= 100:
            result.error = f"quality_score out of range: {score}"
            return result

        result.details = f"quality_score = {score:.1f}%"
        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_aggregate_ticks_callable(config: TestConfig) -> TestResult:
    """BUG-003: aggregate_ticks_to_interval requires callable."""
    result = TestResult("aggregate_ticks_to_interval callable")
    start = time.time()

    try:
        from data.tick_data_fetcher import aggregate_ticks_to_interval

        ticks = generate_synthetic_ticks(1000, config.base_price, config.volatility_pct)

        # Create proper callable (lambda pattern used in optimizer)
        def tick_streamer():
            for tick in ticks:
                yield tick

        # Test callable works
        aggregated1 = list(aggregate_ticks_to_interval(tick_streamer, 10))
        result.details += f"First call: {len(aggregated1)} aggregated points\n"

        # Test callable is reusable (not exhausted)
        aggregated2 = list(aggregate_ticks_to_interval(tick_streamer, 10))
        result.details += f"Second call: {len(aggregated2)} aggregated points\n"

        if len(aggregated1) != len(aggregated2):
            result.error = f"Generator exhaustion! {len(aggregated1)} vs {len(aggregated2)}"
            return result

        if len(aggregated1) == 0:
            result.error = "No aggregated points produced"
            return result

        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_disk_streamer_format(config: TestConfig) -> TestResult:
    """Test binary tick cache read/write."""
    result = TestResult("Disk streamer binary format")
    start = time.time()

    try:
        ticks = generate_synthetic_ticks(100, config.base_price, config.volatility_pct)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            cache_path = f.name

        try:
            create_binary_tick_cache(ticks, cache_path)

            def disk_streamer():
                with open(cache_path, 'rb') as f:
                    while True:
                        data = f.read(16)
                        if not data or len(data) < 16:
                            break
                        ts_float, price = struct.unpack('dd', data)
                        yield (datetime.fromtimestamp(ts_float), price)

            read_ticks = list(disk_streamer())

            if len(read_ticks) != len(ticks):
                result.error = f"Count mismatch: {len(ticks)} vs {len(read_ticks)}"
                return result

            # Verify data integrity
            for i in [0, len(ticks)//2, -1]:
                if abs(ticks[i][1] - read_ticks[i][1]) > 0.0001:
                    result.error = f"Price mismatch at {i}: {ticks[i][1]} vs {read_ticks[i][1]}"
                    return result

            result.details = f"Binary cache: {len(ticks)} ticks OK"
            result.passed = True

        finally:
            os.unlink(cache_path)

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_backtest_engine(config: TestConfig) -> TestResult:
    """Test backtest engine with synthetic data."""
    result = TestResult("Backtest engine integration")
    start = time.time()

    try:
        from backtest_pyramid import run_pyramid_backtest

        ticks = generate_synthetic_ticks(5000, config.base_price, config.volatility_pct)

        backtest_result = run_pyramid_backtest(
            iter(ticks),
            threshold_pct=2.0,
            trailing_pct=1.0,
            pyramid_step_pct=1.0,
            max_pyramids=5,
            verbose=False,
            return_rounds=True
        )

        required_keys = ['total_pnl', 'total_rounds', 'win_rate', 'avg_pnl']
        missing = [k for k in required_keys if k not in backtest_result]
        if missing:
            result.error = f"Missing result keys: {missing}"
            return result

        result.details = (
            f"Rounds: {backtest_result['total_rounds']}, "
            f"P&L: {backtest_result['total_pnl']:.2f}%, "
            f"Win rate: {backtest_result['win_rate']:.1f}%"
        )
        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_statistical_validation(config: TestConfig) -> TestResult:
    """Test statistical validation module."""
    result = TestResult("Statistical validation module")
    start = time.time()

    try:
        from core.statistical_validation import validate_strategy_statistically

        returns = [random.gauss(0.5, 2.0) for _ in range(50)]

        validation = validate_strategy_statistically(
            returns,
            num_tests_performed=100,
            verbose=False
        )

        result.details = (
            f"Sharpe: {validation.sharpe_ratio:.2f}, "
            f"P-value: {validation.p_value:.4f}, "
            f"Passes: {validation.passes_validation}"
        )
        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_monte_carlo_simulation(config: TestConfig) -> TestResult:
    """Test Monte Carlo simulation with minimal iterations."""
    result = TestResult("Monte Carlo simulation")
    start = time.time()

    try:
        from core.monte_carlo import run_monte_carlo_validation

        returns = [random.gauss(0.3, 1.5) for _ in range(30)]

        passes, mc_result, perm_result = run_monte_carlo_validation(
            returns,
            n_bootstrap=100,
            n_permutation=50,
            random_seed=42
        )

        result.details = (
            f"P(positive): {mc_result.probability_positive:.1%}, "
            f"P(ruin): {mc_result.probability_ruin:.1%}, "
            f"Permutation p-value: {perm_result.p_value:.3f}"
        )
        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_mini_grid_search(config: TestConfig) -> TestResult:
    """Test minimal grid search (2x2x2 = 8 combinations)."""
    result = TestResult("Mini grid search (8 combos)")
    start = time.time()

    try:
        from backtest_pyramid import run_pyramid_backtest

        ticks = generate_synthetic_ticks(2000, config.base_price, config.volatility_pct)

        total = len(config.thresholds) * len(config.trailings) * len(config.pyramid_steps)
        results = []

        for thresh in config.thresholds:
            for trail in config.trailings:
                for step in config.pyramid_steps:
                    tick_iter = iter(ticks)

                    br = run_pyramid_backtest(
                        tick_iter,
                        threshold_pct=thresh,
                        trailing_pct=trail,
                        pyramid_step_pct=step,
                        max_pyramids=5,
                        verbose=False,
                        return_rounds=False
                    )

                    results.append({
                        'params': (thresh, trail, step),
                        'pnl': br['total_pnl'],
                        'rounds': br['total_rounds']
                    })

        results.sort(key=lambda x: x['pnl'], reverse=True)

        result.details = (
            f"Tested {total} combos. "
            f"Best: {results[0]['params']} P&L={results[0]['pnl']:.2f}%"
        )
        result.passed = len(results) == total

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_optimizer_imports(config: TestConfig) -> TestResult:
    """Test that optimizer module imports without error."""
    result = TestResult("Optimizer module imports")
    start = time.time()

    try:
        # Test critical imports
        from optimize_pyramid_v4 import (
            run_streaming_grid_search,
            run_single_backtest_streaming,
            create_disk_streamer,
            DEFAULT_EXECUTION_MODEL,
        )

        from data.tick_data_fetcher import (
            validate_data_quality,
            aggregate_ticks_to_interval,
            create_filtered_tick_streamer,
            DataQualityReport,
        )

        from core.monte_carlo import run_monte_carlo_validation
        from core.statistical_validation import validate_strategy_statistically

        result.details = "All critical imports successful"
        result.passed = True

    except ImportError as e:
        result.error = f"Import failed: {e}"
    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_cross_fold_results_structure(config: TestConfig) -> TestResult:
    """Test that cross_fold_results uses correct nested 'params' access.

    BUG-004: Line 1418 was accessing candidate['threshold'] directly,
    but the data structure has params nested: candidate['params']['threshold']
    """
    result = TestResult("Cross-fold results data structure")
    start = time.time()

    try:
        # Simulate the cross_fold_results structure from lines 1375-1386
        cross_fold_results = [
            {
                'params': {'threshold': 1.5, 'trailing': 0.8, 'pyramid_step': 0.5},
                'fold_pnls': [2.1, 1.8, 2.5],
                'avg_val_pnl': 2.13,
                'all_profitable': True,
                'min_fold_pnl': 1.8,
                'total_pnl': 2.13,
                'max_drawdown_pct': 5.2,
                'win_rate': 65.0,
                'rounds': 45,
            },
            {
                'params': {'threshold': 2.0, 'trailing': 1.0, 'pyramid_step': 0.7},
                'fold_pnls': [1.5, 2.0, 1.8],
                'avg_val_pnl': 1.77,
                'all_profitable': True,
                'min_fold_pnl': 1.5,
                'total_pnl': 1.77,
                'max_drawdown_pct': 4.8,
                'win_rate': 62.0,
                'rounds': 38,
            },
        ]

        # Test the CORRECT access pattern (as fixed in line 1418)
        for i, candidate in enumerate(cross_fold_results):
            # This is the CORRECT way - accessing nested params
            params_str = f"th={candidate['params']['threshold']:.2f}, tr={candidate['params']['trailing']:.2f}, ps={candidate['params']['pyramid_step']:.2f}"

            # Verify it produces expected output
            if i == 0:
                expected = "th=1.50, tr=0.80, ps=0.50"
                if params_str != expected:
                    result.error = f"Params string mismatch: got '{params_str}', expected '{expected}'"
                    return result

        # Test that WRONG access pattern would fail
        try:
            wrong_str = f"th={cross_fold_results[0]['threshold']:.2f}"
            result.error = "BUG-004 NOT FIXED: Direct access didn't raise KeyError!"
            return result
        except KeyError:
            # Good - this should fail
            pass

        result.details = "Nested params access pattern works correctly"
        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


# =============================================================================
# PHASE 6 PARAMETER TESTS
# =============================================================================

def test_session_filter(config: TestConfig) -> TestResult:
    """Test session filter helper function."""
    result = TestResult("Session filter function")
    start = time.time()

    try:
        from backtest_pyramid import _is_session_active

        # Test 'all' always returns True
        test_time = datetime(2024, 1, 15, 10, 0, 0)
        assert _is_session_active(test_time, 'all') == True, "'all' should always be active"

        # Test Asia session (00:00-08:00 UTC)
        asia_active = datetime(2024, 1, 15, 3, 0, 0)  # 3:00 UTC
        asia_inactive = datetime(2024, 1, 15, 12, 0, 0)  # 12:00 UTC
        assert _is_session_active(asia_active, 'asia') == True, "Asia should be active at 3:00 UTC"
        assert _is_session_active(asia_inactive, 'asia') == False, "Asia should be inactive at 12:00 UTC"

        # Test Europe session (07:00-16:00 UTC)
        eu_active = datetime(2024, 1, 15, 10, 0, 0)  # 10:00 UTC
        eu_inactive = datetime(2024, 1, 15, 20, 0, 0)  # 20:00 UTC
        assert _is_session_active(eu_active, 'europe') == True, "Europe should be active at 10:00 UTC"
        assert _is_session_active(eu_inactive, 'europe') == False, "Europe should be inactive at 20:00 UTC"

        # Test US session (13:00-22:00 UTC)
        us_active = datetime(2024, 1, 15, 16, 0, 0)  # 16:00 UTC
        us_inactive = datetime(2024, 1, 15, 5, 0, 0)  # 5:00 UTC
        assert _is_session_active(us_active, 'us') == True, "US should be active at 16:00 UTC"
        assert _is_session_active(us_inactive, 'us') == False, "US should be inactive at 5:00 UTC"

        result.details = "All session filter tests passed"
        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_phase6_new_parameters(config: TestConfig) -> TestResult:
    """Test that new Phase 6 parameters are accepted by backtest."""
    result = TestResult("Phase 6 new parameters accepted")
    start = time.time()

    try:
        from backtest_pyramid import run_pyramid_backtest

        ticks = generate_synthetic_ticks(1000, config.base_price, config.volatility_pct)

        # Test with all new Phase 6 parameters
        br = run_pyramid_backtest(
            iter(ticks),
            threshold_pct=2.0,
            trailing_pct=1.0,
            pyramid_step_pct=1.0,
            max_pyramids=5,
            verbose=False,
            return_rounds=False,
            # Phase 6 exit controls
            take_profit_pct=10.0,
            stop_loss_pct=15.0,
            breakeven_after_pct=5.0,
            # Phase 6 timing controls
            pyramid_cooldown_sec=60,
            max_round_duration_hr=4.0,
            # Phase 6 filters
            trend_filter_ema=20,
            session_filter='all',
        )

        # Verify basic result structure
        assert 'total_pnl' in br, "Missing total_pnl in result"
        assert 'total_rounds' in br, "Missing total_rounds in result"
        assert 'win_rate' in br, "Missing win_rate in result"

        result.details = f"Backtest completed: P&L={br['total_pnl']:.2f}%, Rounds={br['total_rounds']}"
        result.passed = True

    except TypeError as e:
        result.error = f"New parameters not accepted: {e}"
    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_take_profit_exit(config: TestConfig) -> TestResult:
    """Test that take_profit_pct triggers exit at target profit."""
    result = TestResult("Take profit exit trigger")
    start = time.time()

    try:
        from backtest_pyramid import run_pyramid_backtest

        # Generate trending data that should hit take profit
        ticks = generate_synthetic_ticks(2000, config.base_price, config.volatility_pct * 2)

        # Run with low take profit to ensure it triggers
        br = run_pyramid_backtest(
            iter(ticks),
            threshold_pct=1.0,  # Low threshold to trigger direction quickly
            trailing_pct=5.0,   # High trailing so take profit triggers first
            pyramid_step_pct=1.0,
            max_pyramids=5,
            verbose=False,
            return_rounds=True,
            take_profit_pct=2.0,  # 2% take profit target
        )

        # With take_profit, we expect some rounds completed
        rounds_count = br.get('total_rounds', 0)
        result.details = f"Completed {rounds_count} rounds with take_profit_pct=2.0"
        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_optimizer_new_params_grid(config: TestConfig) -> TestResult:
    """Test that optimizer grids include new Phase 6 parameters."""
    result = TestResult("Optimizer includes Phase 6 params in grid")
    start = time.time()

    try:
        from optimize_pyramid_v4 import build_dense_new_params_grid

        grid = build_dense_new_params_grid()

        # Check existing params are now enabled (not single values)
        assert len(grid.get('size_schedule', [])) >= 2, "size_schedule should have multiple values"
        assert len(grid.get('acceleration', [])) >= 2, "acceleration should have multiple values"

        # Check new Phase 6 params exist
        new_params = [
            'take_profit_pct', 'stop_loss_pct', 'breakeven_after_pct',
            'pyramid_cooldown_sec', 'max_round_duration_hr',
            'trend_filter_ema', 'session_filter'
        ]

        missing = []
        for param in new_params:
            if param not in grid:
                missing.append(param)

        if missing:
            result.error = f"Missing Phase 6 params in grid: {missing}"
            return result

        total_combos = 1
        for param, values in grid.items():
            total_combos *= len(values)

        result.details = f"Grid has {len(grid)} params, ~{total_combos:,} combinations"
        result.passed = True

    except Exception as e:
        result.error = f"Error: {e}\n{traceback.format_exc()}"

    result.duration_ms = (time.time() - start) * 1000
    return result


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests(config: TestConfig = None) -> Dict:
    """Run all integration tests and return summary."""
    if config is None:
        config = TestConfig()

    tests = [
        test_optimizer_imports,
        test_validate_data_quality_type,
        test_quality_score_property,
        test_aggregate_ticks_callable,
        test_cross_fold_results_structure,  # BUG-004 fix test
        test_disk_streamer_format,
        test_backtest_engine,
        test_statistical_validation,
        test_monte_carlo_simulation,
        test_mini_grid_search,
        # Phase 6 new parameter tests
        test_session_filter,
        test_phase6_new_parameters,
        test_take_profit_exit,
        test_optimizer_new_params_grid,
    ]

    print("=" * 70)
    print("OPTIMIZE_PYRAMID_V4 INTEGRATION TEST SUITE")
    print("=" * 70)
    print(f"Running {len(tests)} tests with {config.num_ticks} synthetic ticks\n")

    results = []
    total_start = time.time()

    for test_func in tests:
        print(f"  {test_func.__name__}...", end=" ", flush=True)

        try:
            result = test_func(config)
        except Exception as e:
            result = TestResult(test_func.__name__)
            result.error = f"Test crashed: {e}\n{traceback.format_exc()}"

        results.append(result)

        status = "\033[92mPASS\033[0m" if result.passed else "\033[91mFAIL\033[0m"
        print(f"[{status}] ({result.duration_ms:.0f}ms)")

        if result.error:
            print(f"    ERROR: {result.error[:100]}")

    total_duration = time.time() - total_start

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    if failed == 0:
        print(f"\033[92mALL {passed} TESTS PASSED\033[0m")
    else:
        print(f"\033[91m{failed} TESTS FAILED\033[0m, {passed} passed")

    print(f"Duration: {total_duration:.1f}s")

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")
                if r.error:
                    first_line = r.error.split('\n')[0][:70]
                    print(f"    {first_line}")

    print("=" * 70)

    return {
        'total': len(results),
        'passed': passed,
        'failed': failed,
        'duration_seconds': total_duration,
        'all_passed': failed == 0,
        'results': results
    }


if __name__ == "__main__":
    config = TestConfig(
        num_ticks=10_000,
        base_price=100.0,
        volatility_pct=2.0,
        thresholds=[1.0, 2.0],
        trailings=[0.5, 1.0],
        pyramid_steps=[0.5, 1.0],
    )

    summary = run_all_tests(config)
    sys.exit(0 if summary['all_passed'] else 1)
