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
        test_disk_streamer_format,
        test_backtest_engine,
        test_statistical_validation,
        test_monte_carlo_simulation,
        test_mini_grid_search,
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
