#!/usr/bin/env python3
"""
Pyramid Strategy Backtest Module

Strategy Logic:
1. Open hedge at entry price (1 LONG + 1 SHORT)
2. Losing side closes when price moves threshold% against it
3. Pyramid reference = price where losing side closed
4. Add positions every pyramid_step% from pyramid reference
5. All positions close when profit drops trailing% from CONFIRMED max (causal)

IMPORTANT: This module uses CAUSAL trailing stops to avoid look-ahead bias.
A peak is only confirmed after price reverses, matching real trading behavior.
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Iterator, Union, Iterable, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics

# Import funding rate functions (Phase 1.2)
try:
    from data.funding_rate_fetcher import (
        calculate_funding_payment,
        is_funding_time,
        get_funding_rate_at_time
    )
    FUNDING_AVAILABLE = True
except ImportError:
    FUNDING_AVAILABLE = False


# =============================================================================
# PHASE 1 FIX: Causal Trailing Stop (removes look-ahead bias)
# =============================================================================

@dataclass
class CausalTrailingState:
    """
    State for causal (non-look-ahead) trailing stop.

    The key insight: In live trading, you can't know a price is the "max"
    until AFTER it starts declining. This class tracks candidate peaks
    and only confirms them after sufficient reversal.
    """
    # Confirmed peak (safe to use for trailing stop)
    confirmed_peak_pct: float = 0.0
    confirmed_peak_price: float = 0.0

    # Candidate peak (might become confirmed)
    candidate_peak_pct: float = 0.0
    candidate_peak_price: float = 0.0
    candidate_peak_time: datetime = None

    # Tracking
    ticks_since_candidate: int = 0
    trailing_stop_active: bool = False


def update_causal_trailing_state(
    state: CausalTrailingState,
    current_price: float,
    current_profit_pct: float,
    current_time: datetime,
    trailing_pct: float,
    confirmation_ticks: int = 3,
    confirmation_reversal_pct: float = 0.0
) -> bool:
    """
    Update causal trailing stop state. Returns True if stop is triggered.

    A peak is confirmed when EITHER:
    1. Price has reversed by confirmation_reversal_pct from candidate peak, OR
    2. confirmation_ticks have passed since candidate was set

    This ensures we only use information available in live trading.

    Args:
        state: CausalTrailingState to update
        current_price: Current market price
        current_profit_pct: Current profit percentage from entry
        current_time: Current timestamp
        trailing_pct: Trailing stop percentage
        confirmation_ticks: Ticks to wait before confirming peak
        confirmation_reversal_pct: Confirm if price reverses this much (0 = disabled)

    Returns:
        True if trailing stop is triggered, False otherwise
    """
    # Check if we have a new candidate peak
    if current_profit_pct > state.candidate_peak_pct:
        # New candidate peak - reset confirmation counter
        state.candidate_peak_pct = current_profit_pct
        state.candidate_peak_price = current_price
        state.candidate_peak_time = current_time
        state.ticks_since_candidate = 0
    else:
        # Price didn't make new high - increment counter
        state.ticks_since_candidate += 1

        # Check for confirmation via reversal
        if state.candidate_peak_pct > 0 and confirmation_reversal_pct > 0:
            reversal = state.candidate_peak_pct - current_profit_pct
            if reversal >= confirmation_reversal_pct:
                # Confirm peak via reversal
                if state.candidate_peak_pct > state.confirmed_peak_pct:
                    state.confirmed_peak_pct = state.candidate_peak_pct
                    state.confirmed_peak_price = state.candidate_peak_price
                    state.trailing_stop_active = True

        # Check for confirmation via tick count
        if state.ticks_since_candidate >= confirmation_ticks:
            # Confirm candidate peak
            if state.candidate_peak_pct > state.confirmed_peak_pct:
                state.confirmed_peak_pct = state.candidate_peak_pct
                state.confirmed_peak_price = state.candidate_peak_price
                state.trailing_stop_active = True

    # Check if trailing stop is triggered (using CONFIRMED peak only)
    if state.trailing_stop_active and state.confirmed_peak_pct > 0:
        trigger_level = state.confirmed_peak_pct - trailing_pct
        if current_profit_pct <= trigger_level:
            return True

    return False


# =============================================================================
# PHASE 1.3 FIX: Dynamic Execution Model (Realistic Costs)
# =============================================================================

import math


@dataclass
class ExecutionModel:
    """
    Models realistic execution costs including fees, slippage, and spread.

    PHASE 1.3 IMPROVEMENTS:
    - Slippage scales with sqrt(order_size / ADV) - market impact model
    - Volatility impact is exponential (vol^2), not linear
    - Exit costs are 2.5x worse during drawdowns (adverse selection)
    - Spread widens dynamically with volatility (2-20 bps range)

    Default values based on Binance Futures market conditions:
    - Maker fee: 0.02% (with BNB discount)
    - Taker fee: 0.04% (with BNB discount)
    - Slippage: 0.05-0.50% depending on volatility, size, and urgency
    - Spread: 2-20 basis points depending on volatility
    """
    # Base fees
    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.04

    # Slippage model parameters
    base_slippage_bps: float = 5.0  # Base slippage in basis points
    size_impact_coefficient: float = 0.1  # Market impact coefficient
    normal_volatility_pct: float = 2.0  # Baseline volatility for comparison
    volatility_slippage_exponent: float = 2.0  # Exponential (not linear) impact

    # Spread dynamics
    base_spread_bps: float = 2.0  # Normal spread in basis points
    max_spread_bps: float = 20.0  # Maximum spread during extreme volatility
    volatility_spread_sensitivity: float = 3.0  # How much spread widens with vol

    # Exit urgency
    exit_stress_multiplier: float = 2.5  # Exits during drawdown cost more
    trailing_stop_premium: float = 1.5  # Premium for trailing stop exits

    # Latency (for future use)
    latency_ms: float = 100.0  # Network + exchange latency

    # Average daily volume for size impact (default $100M for major pairs)
    avg_daily_volume_usd: float = 100_000_000.0

    def calculate_slippage(
        self,
        order_size_usd: float = 100.0,
        volatility_pct: float = 0.0,
        is_exit: bool = False,
        is_under_stress: bool = False,
        is_trailing_stop: bool = False
    ) -> float:
        """
        Calculate realistic slippage using market impact model.

        Formula: base + size_impact + volatility_impact + stress_premium

        Args:
            order_size_usd: Order size in USDT
            volatility_pct: Current volatility (std dev of returns)
            is_exit: True if this is an exit order
            is_under_stress: True if exiting during drawdown
            is_trailing_stop: True if triggered by trailing stop

        Returns:
            Slippage as percentage (e.g., 0.10 = 0.10%)
        """
        # Base slippage
        base = self.base_slippage_bps / 100

        # Size impact: sqrt scaling (market impact model)
        # Impact = coefficient * sqrt(size / ADV)
        size_ratio = order_size_usd / self.avg_daily_volume_usd
        size_impact = self.size_impact_coefficient * math.sqrt(max(size_ratio, 0)) * 100

        # Volatility impact: EXPONENTIAL (not linear)
        # High volatility means much worse execution
        if volatility_pct > 0 and self.normal_volatility_pct > 0:
            vol_ratio = volatility_pct / self.normal_volatility_pct
            vol_impact = base * (vol_ratio ** self.volatility_slippage_exponent - 1)
        else:
            vol_impact = 0.0

        # Stress multiplier for exits
        stress_mult = 1.0
        if is_exit:
            if is_under_stress:
                # Exiting during drawdown - worst execution
                stress_mult = self.exit_stress_multiplier
            elif is_trailing_stop:
                # Trailing stop - moderate premium
                stress_mult = self.trailing_stop_premium

        # Total slippage with cap
        total_slippage = (base + size_impact + vol_impact) * stress_mult
        return min(total_slippage, 0.5)  # Cap at 0.5%

    def calculate_spread(self, volatility_pct: float = 0.0) -> float:
        """
        Calculate dynamic bid-ask spread that widens with volatility.

        Returns:
            Half spread (cost to cross) as percentage
        """
        if volatility_pct > 0 and self.normal_volatility_pct > 0:
            vol_ratio = volatility_pct / self.normal_volatility_pct
            spread_bps = min(
                self.max_spread_bps,
                self.base_spread_bps * (1 + (vol_ratio - 1) * self.volatility_spread_sensitivity)
            )
        else:
            spread_bps = self.base_spread_bps

        # Return half spread (cost to cross)
        return spread_bps / 100 / 2

    def calculate_entry_cost(
        self,
        volatility_pct: float = 0.0,
        is_maker: bool = False,
        order_size_usd: float = 100.0
    ) -> float:
        """
        Calculate total cost for entering a position.

        Args:
            volatility_pct: Current market volatility
            is_maker: True if using limit order (maker)
            order_size_usd: Order size in USDT

        Returns:
            Total cost as percentage (e.g., 0.15 = 0.15%)
        """
        # Base fee
        fee = self.maker_fee_pct if is_maker else self.taker_fee_pct

        # Slippage (taker orders only)
        if is_maker:
            slippage = 0.0
        else:
            slippage = self.calculate_slippage(
                order_size_usd=order_size_usd,
                volatility_pct=volatility_pct,
                is_exit=False
            )

        # Spread cost
        spread_cost = self.calculate_spread(volatility_pct)

        return fee + slippage + spread_cost

    def calculate_exit_cost(
        self,
        volatility_pct: float = 0.0,
        is_maker: bool = False,
        order_size_usd: float = 100.0,
        is_under_stress: bool = False,
        is_trailing_stop: bool = False
    ) -> float:
        """
        Calculate total cost for exiting a position.

        IMPORTANT: Exit costs can be significantly higher than entry costs,
        especially during volatile conditions or when exiting under stress.

        Args:
            volatility_pct: Current market volatility
            is_maker: True if using limit order
            order_size_usd: Order size in USDT
            is_under_stress: True if exiting during drawdown
            is_trailing_stop: True if triggered by trailing stop

        Returns:
            Total cost as percentage
        """
        # Base fee
        fee = self.maker_fee_pct if is_maker else self.taker_fee_pct

        # Slippage (with exit premiums)
        if is_maker:
            slippage = 0.0
        else:
            slippage = self.calculate_slippage(
                order_size_usd=order_size_usd,
                volatility_pct=volatility_pct,
                is_exit=True,
                is_under_stress=is_under_stress,
                is_trailing_stop=is_trailing_stop
            )

        # Spread cost (often wider during exits)
        spread_cost = self.calculate_spread(volatility_pct)
        if is_under_stress:
            spread_cost *= 1.5  # Spread widens when you need to exit

        return fee + slippage + spread_cost

    def calculate_round_trip_cost(
        self,
        volatility_pct: float = 0.0,
        order_size_usd: float = 100.0,
        is_trailing_stop_exit: bool = False
    ) -> float:
        """Calculate total round-trip cost (entry + exit as taker)."""
        entry = self.calculate_entry_cost(
            volatility_pct=volatility_pct,
            is_maker=False,
            order_size_usd=order_size_usd
        )
        exit_cost = self.calculate_exit_cost(
            volatility_pct=volatility_pct,
            is_maker=False,
            order_size_usd=order_size_usd,
            is_trailing_stop=is_trailing_stop_exit
        )
        return entry + exit_cost


# Default execution model - realistic for Binance Futures
DEFAULT_EXECUTION_MODEL = ExecutionModel(
    maker_fee_pct=0.02,
    taker_fee_pct=0.04,
    base_slippage_bps=5.0,
    size_impact_coefficient=0.1,
    normal_volatility_pct=2.0,
    volatility_slippage_exponent=2.0,
    base_spread_bps=2.0,
    max_spread_bps=20.0,
    exit_stress_multiplier=2.5
)

# Conservative execution model - for validation
CONSERVATIVE_EXECUTION_MODEL = ExecutionModel(
    maker_fee_pct=0.02,
    taker_fee_pct=0.05,
    base_slippage_bps=10.0,
    size_impact_coefficient=0.15,
    normal_volatility_pct=2.0,
    volatility_slippage_exponent=2.5,
    base_spread_bps=3.0,
    max_spread_bps=30.0,
    exit_stress_multiplier=3.0
)

# Optimistic execution model - best case scenario
OPTIMISTIC_EXECUTION_MODEL = ExecutionModel(
    maker_fee_pct=0.02,
    taker_fee_pct=0.04,
    base_slippage_bps=3.0,
    size_impact_coefficient=0.05,
    normal_volatility_pct=2.0,
    volatility_slippage_exponent=1.5,
    base_spread_bps=1.5,
    max_spread_bps=10.0,
    exit_stress_multiplier=1.5
)


# =============================================================================
# PHASE 1.1 FIX: Leverage and Liquidation Modeling
# =============================================================================

@dataclass
class MarginState:
    """
    Tracks margin, leverage, and liquidation state for realistic simulation.

    With 10x leverage:
    - $1,000 capital = $10,000 max position value
    - 10% adverse move = 100% equity loss = LIQUIDATION

    This class ensures the simulation cannot show impossible recoveries
    that would have triggered liquidation in real trading.
    """
    # Account settings
    initial_capital: float = 10000.0  # Starting capital in USDT
    leverage: int = 10  # Leverage multiplier (1-125x on Binance)
    maintenance_margin_pct: float = 0.5  # Liquidation threshold (0.5% = 200x leverage point)

    # Current state
    current_equity: float = 0.0  # Initial capital + unrealized PnL
    used_margin: float = 0.0  # Margin currently used by positions
    unrealized_pnl: float = 0.0  # Current unrealized profit/loss in USDT
    realized_pnl: float = 0.0  # Cumulative realized profit/loss in USDT

    # Tracking
    peak_equity: float = 0.0  # For drawdown calculation
    max_drawdown_pct: float = 0.0  # Maximum drawdown observed
    total_notional: float = 0.0  # Total position notional value
    liquidation_events: int = 0  # Count of liquidation events

    def __post_init__(self):
        """Initialize equity to initial capital."""
        self.current_equity = self.initial_capital
        self.peak_equity = self.initial_capital

    def get_available_margin(self) -> float:
        """Calculate margin available for new positions."""
        return max(0, self.current_equity - self.used_margin)

    def get_max_position_notional(self) -> float:
        """Calculate maximum position notional value allowed."""
        return self.current_equity * self.leverage

    def can_open_position(self, notional_value: float) -> bool:
        """
        Check if there's enough margin to open a position.

        Args:
            notional_value: Position value in USDT (size * price)

        Returns:
            True if position can be opened, False otherwise
        """
        required_margin = notional_value / self.leverage
        return required_margin <= self.get_available_margin()

    def open_position(self, notional_value: float) -> bool:
        """
        Reserve margin for a new position.

        Args:
            notional_value: Position value in USDT

        Returns:
            True if position opened, False if insufficient margin
        """
        required_margin = notional_value / self.leverage
        if required_margin > self.get_available_margin():
            return False

        self.used_margin += required_margin
        self.total_notional += notional_value
        return True

    def close_position(self, notional_value: float, pnl_usdt: float):
        """
        Release margin and record P&L for a closed position.

        Args:
            notional_value: Position value that was opened
            pnl_usdt: Realized profit/loss in USDT
        """
        margin_released = notional_value / self.leverage
        self.used_margin = max(0, self.used_margin - margin_released)
        self.total_notional = max(0, self.total_notional - notional_value)
        self.realized_pnl += pnl_usdt
        self.current_equity += pnl_usdt

        # Update peak equity and drawdown
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        else:
            current_dd = ((self.peak_equity - self.current_equity) / self.peak_equity) * 100
            self.max_drawdown_pct = max(self.max_drawdown_pct, current_dd)

    def update_unrealized_pnl(self, unrealized_pnl: float):
        """
        Update unrealized P&L and check for liquidation.

        Args:
            unrealized_pnl: Current unrealized P&L in USDT

        Note: Call this every tick to track margin ratio
        """
        self.unrealized_pnl = unrealized_pnl
        # Current equity = initial + realized + unrealized
        self.current_equity = self.initial_capital + self.realized_pnl + self.unrealized_pnl

    def is_liquidated(self) -> bool:
        """
        Check if position should be liquidated.

        Liquidation occurs when:
        - Equity falls below maintenance margin requirement
        - Margin ratio = (equity / used_margin) * 100 <= maintenance_margin_pct

        For most positions, this is approximately when equity approaches zero
        (100% loss of initial margin).
        """
        if self.used_margin <= 0:
            return False

        # Margin ratio as percentage
        margin_ratio = (self.current_equity / self.used_margin) * 100

        # Liquidation when margin ratio falls below maintenance requirement
        # At 0.5% maintenance margin with 10x leverage, liquidation at ~10% adverse move
        return margin_ratio <= self.maintenance_margin_pct * self.leverage

    def get_margin_ratio(self) -> float:
        """Get current margin ratio as percentage."""
        if self.used_margin <= 0:
            return 100.0
        return (self.current_equity / self.used_margin) * 100

    def get_liquidation_price_move_pct(self) -> float:
        """
        Calculate how much price can move against us before liquidation.

        Returns:
            Percentage move to liquidation (e.g., 10.0 = 10% adverse move)
        """
        if self.total_notional <= 0:
            return float('inf')

        # How much can we lose before liquidation?
        # Liquidation when equity = maintenance_margin_pct * used_margin / 100
        max_loss = self.current_equity - (self.maintenance_margin_pct * self.used_margin / 100)

        # Convert to percentage of total position
        return (max_loss / self.total_notional) * 100

    def reset(self):
        """Reset state for new simulation."""
        self.current_equity = self.initial_capital
        self.used_margin = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.peak_equity = self.initial_capital
        self.max_drawdown_pct = 0.0
        self.total_notional = 0.0
        self.liquidation_events = 0


def calculate_position_notional(price: float, size: float) -> float:
    """Calculate notional value of a position."""
    return price * size


def calculate_pnl_usdt(
    entry_price: float,
    exit_price: float,
    size: float,
    is_long: bool
) -> float:
    """
    Calculate P&L in USDT for a closed position.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        size: Position size in base asset
        is_long: True for long, False for short

    Returns:
        P&L in USDT (positive = profit, negative = loss)
    """
    if is_long:
        return (exit_price - entry_price) * size
    else:
        return (entry_price - exit_price) * size


# =============================================================================
# PHASE 3.1 FIX: Risk Management Framework
# =============================================================================

@dataclass
class RiskLimits:
    """
    Risk limit configuration for the simulation.

    These limits prevent catastrophic losses and ensure the simulation
    reflects realistic trading constraints. Without limits, a strategy
    could show fantastic returns while taking 80% drawdowns that would
    be unacceptable in real trading.
    """
    # Maximum drawdown before stopping all trading
    max_drawdown_pct: float = 20.0  # Stop if equity drops 20% from peak

    # Maximum daily loss before pausing
    max_daily_loss_pct: float = 5.0  # Stop trading for the day at 5% loss

    # Maximum loss per trading round
    max_round_loss_pct: float = 10.0  # Force-close round if losing > 10%

    # Maximum number of consecutive losses before pause
    max_consecutive_losses: int = 5  # Pause after 5 consecutive losing rounds

    # Maximum position size as percentage of equity
    max_position_pct: float = 50.0  # No single position > 50% of equity

    # Risk limits active
    enabled: bool = True


@dataclass
class RiskState:
    """
    Tracks current risk state during simulation.

    This class monitors all risk metrics in real-time and flags
    when limits are breached.
    """
    # Daily tracking
    daily_pnl: float = 0.0  # Today's cumulative P&L
    daily_start_equity: float = 0.0  # Equity at start of day
    current_date: str = ""  # Current trading date (for day boundary detection)

    # Drawdown tracking
    current_drawdown_pct: float = 0.0  # Current drawdown from peak

    # Round tracking
    current_round_pnl_pct: float = 0.0  # Current round's P&L percentage

    # Consecutive loss tracking
    consecutive_losses: int = 0  # Current streak of losing rounds
    max_consecutive_losses: int = 0  # Worst streak observed

    # Breach tracking
    drawdown_breached: bool = False  # True if max drawdown hit
    daily_loss_breached: bool = False  # True if max daily loss hit
    round_loss_breached: bool = False  # True if max round loss hit
    consec_loss_breached: bool = False  # True if max consecutive losses hit

    # Statistics
    total_rounds_stopped: int = 0  # Rounds force-closed due to risk
    total_days_paused: int = 0  # Days paused due to daily loss
    total_breach_events: int = 0  # Total times any limit was breached

    def reset(self):
        """Reset state for new simulation."""
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.current_date = ""
        self.current_drawdown_pct = 0.0
        self.current_round_pnl_pct = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.drawdown_breached = False
        self.daily_loss_breached = False
        self.round_loss_breached = False
        self.consec_loss_breached = False
        self.total_rounds_stopped = 0
        self.total_days_paused = 0
        self.total_breach_events = 0

    def update_daily_tracking(self, current_date: str, current_equity: float):
        """
        Update daily tracking when date changes.

        Args:
            current_date: Current trading date (YYYY-MM-DD format)
            current_equity: Current equity value
        """
        if current_date != self.current_date:
            # New day - reset daily tracking
            self.current_date = current_date
            self.daily_pnl = 0.0
            self.daily_start_equity = current_equity
            self.daily_loss_breached = False  # Reset daily breach flag

    def update_pnl(self, pnl: float, current_equity: float, peak_equity: float):
        """
        Update P&L tracking and check for breaches.

        Args:
            pnl: P&L from this trade/update
            current_equity: Current total equity
            peak_equity: Peak equity value for drawdown calc

        Returns:
            True if any limit was breached
        """
        self.daily_pnl += pnl

        # Update drawdown
        if peak_equity > 0:
            self.current_drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100

        # Check breaches (will be checked by check_limits)
        return False

    def record_round_result(self, pnl_pct: float):
        """
        Record the result of a completed trading round.

        Args:
            pnl_pct: P&L percentage for the completed round
        """
        if pnl_pct < 0:
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(
                self.max_consecutive_losses,
                self.consecutive_losses
            )
        else:
            self.consecutive_losses = 0

    def check_limits(self, limits: 'RiskLimits', current_equity: float) -> Tuple[bool, List[str]]:
        """
        Check if any risk limits are breached.

        Args:
            limits: RiskLimits configuration
            current_equity: Current equity value

        Returns:
            Tuple of (any_breached, list of breach reasons)
        """
        if not limits.enabled:
            return False, []

        breaches = []

        # Check max drawdown
        if self.current_drawdown_pct >= limits.max_drawdown_pct:
            if not self.drawdown_breached:
                self.drawdown_breached = True
                self.total_breach_events += 1
            breaches.append(
                f"Max drawdown breached: {self.current_drawdown_pct:.1f}% >= {limits.max_drawdown_pct}%"
            )

        # Check daily loss
        if self.daily_start_equity > 0:
            daily_loss_pct = ((self.daily_start_equity - current_equity) / self.daily_start_equity) * 100
            if daily_loss_pct >= limits.max_daily_loss_pct:
                if not self.daily_loss_breached:
                    self.daily_loss_breached = True
                    self.total_days_paused += 1
                    self.total_breach_events += 1
                breaches.append(
                    f"Daily loss limit breached: {daily_loss_pct:.1f}% >= {limits.max_daily_loss_pct}%"
                )

        # Check round loss (needs to be set externally via current_round_pnl_pct)
        if self.current_round_pnl_pct <= -limits.max_round_loss_pct:
            if not self.round_loss_breached:
                self.round_loss_breached = True
                self.total_rounds_stopped += 1
                self.total_breach_events += 1
            breaches.append(
                f"Round loss limit breached: {self.current_round_pnl_pct:.1f}% <= -{limits.max_round_loss_pct}%"
            )

        # Check consecutive losses
        if self.consecutive_losses >= limits.max_consecutive_losses:
            if not self.consec_loss_breached:
                self.consec_loss_breached = True
                self.total_breach_events += 1
            breaches.append(
                f"Consecutive losses limit: {self.consecutive_losses} >= {limits.max_consecutive_losses}"
            )

        return len(breaches) > 0, breaches

    def should_stop_trading(self, limits: 'RiskLimits') -> bool:
        """
        Check if trading should be stopped based on current state.

        Returns:
            True if trading should stop (max drawdown breached)
        """
        if not limits.enabled:
            return False

        # Stop completely if max drawdown is hit
        return self.drawdown_breached

    def should_skip_round(self, limits: 'RiskLimits') -> bool:
        """
        Check if the next round should be skipped.

        Returns:
            True if should skip (daily loss or consecutive losses breached)
        """
        if not limits.enabled:
            return False

        return self.daily_loss_breached or self.consec_loss_breached


# Default risk limits - conservative for retail traders
DEFAULT_RISK_LIMITS = RiskLimits(
    max_drawdown_pct=20.0,
    max_daily_loss_pct=5.0,
    max_round_loss_pct=10.0,
    max_consecutive_losses=5,
    max_position_pct=50.0,
    enabled=True
)

# Aggressive risk limits - for testing
AGGRESSIVE_RISK_LIMITS = RiskLimits(
    max_drawdown_pct=50.0,
    max_daily_loss_pct=10.0,
    max_round_loss_pct=20.0,
    max_consecutive_losses=10,
    max_position_pct=100.0,
    enabled=True
)

# No limits - for comparing with/without risk management
NO_RISK_LIMITS = RiskLimits(enabled=False)


# =============================================================================
# Position and Round Data Classes
# =============================================================================

@dataclass
class PyramidPosition:
    """A single position in the pyramid."""
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    size: float = 1.0
    is_open: bool = True
    exit_price: float = 0.0
    pnl_percent: float = 0.0


@dataclass 
class PyramidRound:
    """Tracks a complete pyramid round."""
    entry_price: float
    entry_time: datetime
    direction: str = ''  # 'LONG' or 'SHORT' (winning direction)
    pyramid_reference: float = 0.0  # Price where losing side closed
    positions: List[PyramidPosition] = field(default_factory=list)
    max_profit_pct: float = 0.0
    exit_price: float = 0.0
    exit_time: datetime = None
    total_pnl: float = 0.0
    num_pyramids: int = 0


def calculate_profit_pct(entry_price: float, current_price: float, is_long: bool) -> float:
    """Calculate profit percentage."""
    if is_long:
        return ((current_price - entry_price) / entry_price) * 100
    else:
        return ((entry_price - current_price) / entry_price) * 100


def _calculate_pyramid_size(level: int, schedule: str) -> float:
    """
    Calculate position size based on pyramid level and schedule.

    Args:
        level: Pyramid level (1-based)
        schedule: 'fixed', 'linear_decay', or 'exp_decay'

    Returns:
        Position size multiplier
    """
    if schedule == 'fixed':
        return 1.0
    elif schedule == 'linear_decay':
        # Size decreases linearly: 1.0, 0.8, 0.6, 0.4, 0.2 (min 0.2)
        return max(1.0 - (level - 1) * 0.2, 0.2)
    elif schedule == 'exp_decay':
        # Size decreases exponentially: 1.0, 0.7, 0.49, 0.34...
        return 0.7 ** (level - 1)
    return 1.0


def _calculate_volatility(prices: deque, method: str) -> float:
    """
    Calculate volatility from recent prices.

    Args:
        prices: Recent price history
        method: 'none', 'stddev', or 'range'

    Returns:
        Volatility measure (higher = more volatile)
    """
    if method == 'none' or len(prices) < 10:
        return float('inf')  # Always pass filter

    prices_list = list(prices)

    if method == 'stddev':
        # Standard deviation of returns
        returns = [
            (prices_list[i] - prices_list[i-1]) / prices_list[i-1] * 100
            for i in range(1, len(prices_list))
        ]
        return statistics.stdev(returns) if len(returns) > 1 else 0.0

    elif method == 'range':
        # Range as % of mean
        min_price = min(prices_list)
        max_price = max(prices_list)
        avg_price = sum(prices_list) / len(prices_list)
        return ((max_price - min_price) / avg_price) * 100

    return float('inf')


def run_pyramid_backtest(
    prices: Iterable[Tuple[datetime, float]],
    threshold_pct: float,
    trailing_pct: float,
    pyramid_step_pct: float,
    fee_pct: float = 0.04,
    max_pyramids: int = 20,
    verbose: bool = False,
    # PYRAMID PARAMETERS:
    pyramid_size_schedule: str = 'fixed',
    min_pyramid_spacing_pct: float = 0.0,
    pyramid_acceleration: float = 1.0,
    time_decay_exit_seconds: Optional[float] = None,
    volatility_filter_type: str = 'none',
    volatility_min_pct: float = 0.0,
    volatility_window_size: int = 100,
    # CAUSAL TRAILING STOP PARAMETERS (Phase 1 fix):
    use_causal_trailing: bool = True,  # Enable causal trailing stop (no look-ahead)
    confirmation_ticks: int = 3,  # Ticks to confirm peak
    confirmation_reversal_pct: float = 0.0,  # Or confirm via reversal (0 = disabled)
    # EXECUTION MODEL PARAMETERS (Phase 2 fix):
    execution_model: Optional[ExecutionModel] = None,  # None = use fee_pct only
    # MARGIN AND LEVERAGE PARAMETERS (Phase 1.1 fix):
    initial_capital: float = 10000.0,  # Starting capital in USDT
    leverage: int = 10,  # Leverage multiplier (1-125x)
    position_size_usdt: float = 100.0,  # Position size per unit in USDT
    use_margin_tracking: bool = True,  # Enable margin/liquidation tracking
    # FUNDING RATE PARAMETERS (Phase 1.2 fix):
    funding_rates: Optional[Dict[datetime, float]] = None,  # Funding rate lookup {time: rate%}
    apply_funding: bool = True,  # Apply funding rate costs when available
    # MEMORY OPTIMIZATION:
    return_rounds: bool = True  # Set False to skip accumulating round objects (saves memory)
) -> Dict:
    """
    Run pyramid strategy backtest.

    Args:
        prices: Iterable of (timestamp, price) tuples (list or generator for streaming)
        threshold_pct: % move that closes the losing hedge side
        trailing_pct: % drop from max profit that closes all positions
        pyramid_step_pct: % interval to add new positions (from pyramid reference)
        fee_pct: Trading fee per trade (used if execution_model is None)
        max_pyramids: Maximum number of pyramid positions
        verbose: Print detailed output

        # PYRAMID PARAMETERS:
        pyramid_size_schedule: Position sizing ('fixed', 'linear_decay', 'exp_decay')
        min_pyramid_spacing_pct: Minimum % between pyramid entries (0 = disabled)
        pyramid_acceleration: Pyramid spacing multiplier (1.0 = linear, >1 = exponential)
        time_decay_exit_seconds: Force exit after N seconds (None = disabled)
        volatility_filter_type: Volatility filter ('none', 'stddev', 'range')
        volatility_min_pct: Minimum volatility to add pyramids (0 = disabled)
        volatility_window_size: Number of prices for volatility calculation

        # CAUSAL TRAILING STOP (Phase 1 fix - removes look-ahead bias):
        use_causal_trailing: If True, use causal trailing stop (peak confirmed after reversal)
        confirmation_ticks: Number of ticks after peak to confirm it (default: 3)
        confirmation_reversal_pct: Alternative: confirm peak after this % reversal (0 = disabled)

        # EXECUTION MODEL (Phase 2 fix - realistic costs):
        execution_model: ExecutionModel instance for realistic costs, or None for simple fee_pct

        # MARGIN AND LEVERAGE (Phase 1.1 fix - realistic margin tracking):
        initial_capital: Starting capital in USDT (default: 10000)
        leverage: Leverage multiplier, 1-125x (default: 10)
        position_size_usdt: Position size per unit in USDT (default: 100)
        use_margin_tracking: If True, track margin and check liquidation (default: True)

        # FUNDING RATES (Phase 1.2 fix - perpetual futures funding):
        funding_rates: Dict mapping datetime to funding rate percentage
        apply_funding: If True and funding_rates provided, deduct funding from P&L

        # MEMORY:
        return_rounds: If False, skip accumulating round objects (saves memory in grid search)

    Returns:
        Dictionary with backtest results including:
        - Standard metrics (total_pnl, win_rate, etc.)
        - Margin metrics (final_equity, max_drawdown_pct, liquidation_events) if use_margin_tracking
        - Funding metrics (total_funding_paid, net_funding_impact) if funding_rates provided
    """
    # Only accumulate rounds if requested (saves memory in grid search)
    rounds: List[PyramidRound] = [] if return_rounds else None

    # Initialize margin state (Phase 1.1 fix)
    margin_state: Optional[MarginState] = None
    if use_margin_tracking:
        margin_state = MarginState(
            initial_capital=initial_capital,
            leverage=leverage
        )

    # Helper function to get execution cost (Phase 1.3 - dynamic model)
    def get_cost(
        volatility: float = 0.0,
        is_entry: bool = True,
        order_size_usd: float = 100.0,
        is_trailing_stop: bool = False,
        is_under_stress: bool = False
    ) -> float:
        """
        Get execution cost using dynamic model or simple fee.

        Phase 1.3: Now considers order size, volatility, and exit conditions
        for more realistic cost estimation.
        """
        if execution_model is not None:
            if is_entry:
                return execution_model.calculate_entry_cost(
                    volatility_pct=volatility,
                    is_maker=False,
                    order_size_usd=order_size_usd
                )
            else:
                return execution_model.calculate_exit_cost(
                    volatility_pct=volatility,
                    is_maker=False,
                    order_size_usd=order_size_usd,
                    is_under_stress=is_under_stress,
                    is_trailing_stop=is_trailing_stop
                )
        return fee_pct

    # State tracking
    long_pos: PyramidPosition = None
    short_pos: PyramidPosition = None
    pyramid_positions: List[PyramidPosition] = []
    pyramid_reference: float = 0.0
    direction: str = ''
    max_profit_pct: float = 0.0  # Still tracked for reporting, but not used for trailing if causal
    next_pyramid_level: int = 1
    round_entry_price: float = 0.0
    round_entry_time: datetime = None

    # Causal trailing stop state (Phase 1 fix)
    trailing_state: CausalTrailingState = None

    # Additional state for parameters
    last_pyramid_price: float = 0.0  # For min_pyramid_spacing
    round_start_time: datetime = None  # For time_decay_exit
    recent_prices: deque = deque(maxlen=volatility_window_size)  # For volatility filter
    current_volatility: float = 0.0  # For execution model

    # Track position notionals for margin (Phase 1.1)
    position_notionals: Dict[int, float] = {}  # position_id -> notional_value

    total_pnl = 0.0
    total_fees = 0.0
    total_rounds = 0
    winning_rounds = 0
    total_pyramids = 0  # Track for avg_pyramids when return_rounds=False
    skipped_pyramids_margin = 0  # Track pyramids skipped due to insufficient margin
    liquidation_exits = 0  # Track rounds ended by liquidation

    # Funding rate tracking (Phase 1.2)
    total_funding_paid = 0.0  # Total funding paid (positive = cost)
    total_funding_received = 0.0  # Total funding received (negative = benefit)
    funding_events = 0  # Number of funding events applied
    last_funding_time: Optional[datetime] = None  # Track last funding application
    
    for timestamp, price in prices:
        # Track recent prices for volatility calculation
        recent_prices.append(price)

        # Update volatility estimate for execution model
        if len(recent_prices) >= 10:
            current_volatility = _calculate_volatility(recent_prices, 'stddev')
            if current_volatility == float('inf'):
                current_volatility = 0.0

        # === PHASE 1.2: Apply funding rate every 8 hours ===
        # PHASE 5 FIX: More precise funding timing
        # Binance applies funding at exactly 00:00, 08:00, 16:00 UTC
        # Only positions open at that instant are charged
        if apply_funding and funding_rates and FUNDING_AVAILABLE and pyramid_positions:
            # Check if we should apply funding (every 8 hours)
            should_apply_funding = False

            # Calculate the nearest funding time (00:00, 08:00, or 16:00 UTC)
            funding_hour = (timestamp.hour // 8) * 8  # 0, 8, or 16
            funding_time = timestamp.replace(hour=funding_hour, minute=0, second=0, microsecond=0)

            if last_funding_time is None:
                # First funding check - apply if we've crossed a funding boundary
                # and the position was open before the funding time
                if timestamp.hour in (0, 8, 16) and timestamp.minute == 0:
                    # Exactly at funding time
                    should_apply_funding = True
                elif timestamp > funding_time:
                    # We're past a funding time - check if position was open before it
                    # For simplicity, apply funding if this is the first check after a boundary
                    if timestamp.hour >= funding_hour and timestamp.minute < 5:
                        should_apply_funding = True
            else:
                # Check if we've crossed a funding boundary since last check
                # Calculate next expected funding time after last_funding_time
                next_funding_hour = ((last_funding_time.hour // 8) + 1) * 8
                if next_funding_hour >= 24:
                    next_funding_hour = 0
                    next_funding_day = last_funding_time.date() + timedelta(days=1)
                else:
                    next_funding_day = last_funding_time.date()

                next_funding_time = datetime.combine(
                    next_funding_day,
                    datetime.min.time()
                ).replace(hour=next_funding_hour)

                if timestamp >= next_funding_time:
                    should_apply_funding = True

            if should_apply_funding:
                # Get funding rate at this time
                funding_rate = get_funding_rate_at_time(funding_rates, timestamp)

                if funding_rate is not None:
                    # Calculate total position notional
                    is_long = (direction == 'LONG')
                    total_notional = sum(
                        position_size_usdt * pos.size for pos in pyramid_positions
                    )

                    # Calculate funding payment
                    funding_payment = calculate_funding_payment(
                        position_notional=total_notional,
                        funding_rate_pct=funding_rate,
                        is_long=is_long
                    )

                    # Track funding
                    if funding_payment > 0:
                        total_funding_paid += funding_payment
                    else:
                        total_funding_received += abs(funding_payment)

                    funding_events += 1
                    last_funding_time = timestamp

                    # Deduct from margin equity if tracking
                    if margin_state is not None:
                        margin_state.realized_pnl -= funding_payment
                        margin_state.current_equity -= funding_payment

                    if verbose and abs(funding_payment) > 0.001:
                        print(f"[{timestamp}] FUNDING: {funding_rate:+.4f}% | "
                              f"Notional: ${total_notional:.2f} | "
                              f"Payment: ${funding_payment:+.4f}")

        # === PHASE 1: Open initial hedge if no positions ===
        if long_pos is None and short_pos is None and not pyramid_positions:
            # Calculate notional for hedge (2 positions)
            hedge_notional = position_size_usdt * 2

            # Check margin availability (Phase 1.1)
            if margin_state is not None:
                if not margin_state.can_open_position(hedge_notional):
                    # Insufficient margin - skip this round
                    if verbose:
                        print(f"[{timestamp}] HEDGE SKIPPED - Insufficient margin "
                              f"(need ${hedge_notional:.2f}, have ${margin_state.get_available_margin():.2f})")
                    continue
                # Reserve margin for hedge positions
                margin_state.open_position(hedge_notional)
                position_notionals[0] = position_size_usdt  # Long
                position_notionals[1] = position_size_usdt  # Short

            long_pos = PyramidPosition(side='LONG', entry_price=price, entry_time=timestamp)
            short_pos = PyramidPosition(side='SHORT', entry_price=price, entry_time=timestamp)
            round_entry_price = price
            round_entry_time = timestamp
            round_start_time = timestamp  # Track for time_decay_exit
            direction = ''
            pyramid_reference = 0.0
            max_profit_pct = 0.0
            next_pyramid_level = 1
            last_pyramid_price = 0.0  # Reset for min_pyramid_spacing

            # Initialize causal trailing state (Phase 1 fix)
            trailing_state = CausalTrailingState()

            # Entry fees using execution model or simple fee
            entry_cost = get_cost(current_volatility, is_entry=True)
            total_fees += 2 * entry_cost  # Entry fees for both positions

            if verbose:
                margin_info = ""
                if margin_state:
                    margin_info = f" | Margin used: ${margin_state.used_margin:.2f}"
                print(f"\n[{timestamp}] OPENED HEDGE @ ${price:.2f} (cost: {entry_cost:.3f}% each){margin_info}")
            continue
        
        # === PHASE 2: Check if losing side should close (determines direction) ===
        if direction == '':
            # Calculate profits for both sides
            long_profit = calculate_profit_pct(long_pos.entry_price, price, is_long=True)
            short_profit = calculate_profit_pct(short_pos.entry_price, price, is_long=False)

            # Get exit cost for this trade
            exit_cost = get_cost(current_volatility, is_entry=False)

            # Check if LONG loses (price dropped threshold%)
            if long_profit <= -threshold_pct:
                # SHORT wins, LONG loses
                direction = 'SHORT'
                pyramid_reference = price
                long_pos.is_open = False
                long_pos.exit_price = price
                long_pos.pnl_percent = long_profit - exit_cost
                total_fees += exit_cost

                # Update margin state for closed LONG position (Phase 1.1)
                if margin_state is not None and 0 in position_notionals:
                    notional = position_notionals[0]
                    pnl_usdt = (long_profit / 100) * notional  # Convert % to USDT
                    margin_state.close_position(notional, pnl_usdt)
                    del position_notionals[0]

                # SHORT becomes first pyramid position
                pyramid_positions = [short_pos]
                short_pos = None
                long_pos = None
                max_profit_pct = short_profit

                # Initialize trailing state with current profit as first candidate
                if use_causal_trailing:
                    trailing_state.candidate_peak_pct = short_profit
                    trailing_state.candidate_peak_price = price
                    trailing_state.candidate_peak_time = timestamp

                if verbose:
                    print(f"[{timestamp}] LONG CLOSED @ ${price:.2f} ({long_profit:+.2f}%) → Direction: SHORT")

            # Check if SHORT loses (price rose threshold%)
            elif short_profit <= -threshold_pct:
                # LONG wins, SHORT loses
                direction = 'LONG'
                pyramid_reference = price
                short_pos.is_open = False
                short_pos.exit_price = price
                short_pos.pnl_percent = short_profit - exit_cost
                total_fees += exit_cost

                # Update margin state for closed SHORT position (Phase 1.1)
                if margin_state is not None and 1 in position_notionals:
                    notional = position_notionals[1]
                    pnl_usdt = (short_profit / 100) * notional  # Convert % to USDT
                    margin_state.close_position(notional, pnl_usdt)
                    del position_notionals[1]

                # LONG becomes first pyramid position
                pyramid_positions = [long_pos]
                long_pos = None
                short_pos = None
                max_profit_pct = long_profit

                # Initialize trailing state with current profit as first candidate
                if use_causal_trailing:
                    trailing_state.candidate_peak_pct = long_profit
                    trailing_state.candidate_peak_price = price
                    trailing_state.candidate_peak_time = timestamp

                if verbose:
                    print(f"[{timestamp}] SHORT CLOSED @ ${price:.2f} ({short_profit:+.2f}%) → Direction: LONG")
        
        # === PHASE 3: Direction established - manage pyramid ===
        if direction != '' and pyramid_positions:
            is_long = (direction == 'LONG')

            # Calculate profit from ORIGINAL ENTRY PRICE
            if is_long:
                profit_from_entry = ((price - round_entry_price) / round_entry_price) * 100
            else:
                profit_from_entry = ((round_entry_price - price) / round_entry_price) * 100

            # Update max profit for reporting (still tracked even in causal mode)
            if profit_from_entry > max_profit_pct:
                max_profit_pct = profit_from_entry

            # Check for new pyramid level
            if len(pyramid_positions) < max_pyramids:
                # Calculate how far price has moved from pyramid reference
                if is_long:
                    move_from_ref = ((price - pyramid_reference) / pyramid_reference) * 100
                else:
                    move_from_ref = ((pyramid_reference - price) / pyramid_reference) * 100

                # Calculate pyramid threshold with acceleration
                if pyramid_acceleration == 1.0:
                    # Linear spacing (original behavior)
                    pyramid_threshold = next_pyramid_level * pyramid_step_pct
                else:
                    # Exponential spacing: sum of geometric series
                    pyramid_threshold = pyramid_step_pct * sum(
                        pyramid_acceleration ** i for i in range(next_pyramid_level)
                    )

                # Check if we've reached next pyramid level
                if move_from_ref >= pyramid_threshold:
                    # Check minimum spacing from last pyramid
                    spacing_ok = True
                    if last_pyramid_price > 0 and min_pyramid_spacing_pct > 0:
                        if is_long:
                            spacing = ((price - last_pyramid_price) / last_pyramid_price) * 100
                        else:
                            spacing = ((last_pyramid_price - price) / last_pyramid_price) * 100
                        spacing_ok = spacing >= min_pyramid_spacing_pct

                    # Check volatility filter
                    current_vol = _calculate_volatility(recent_prices, volatility_filter_type)
                    volatility_ok = current_vol >= volatility_min_pct

                    if spacing_ok and volatility_ok:
                        # Add new pyramid position with calculated size
                        pyramid_level = len(pyramid_positions) + 1
                        pos_size = _calculate_pyramid_size(pyramid_level, pyramid_size_schedule)
                        pyramid_notional = position_size_usdt * pos_size

                        # Check margin availability (Phase 1.1)
                        margin_ok = True
                        if margin_state is not None:
                            if not margin_state.can_open_position(pyramid_notional):
                                margin_ok = False
                                skipped_pyramids_margin += 1
                                if verbose:
                                    print(f"[{timestamp}] PYRAMID SKIPPED - Insufficient margin "
                                          f"(need ${pyramid_notional:.2f}, have ${margin_state.get_available_margin():.2f})")

                        if margin_ok:
                            # PHASE 5 FIX: Add micro-slippage for pyramid entry
                            # In real trading, order takes 100-200ms to execute
                            # Price moves during this time, especially in volatile conditions
                            pyramid_entry_cost = get_cost(current_volatility, is_entry=True)

                            # Convert half of entry cost to price slippage
                            entry_slippage_pct = pyramid_entry_cost * 0.3 / 100  # 30% of cost is slippage

                            # Apply slippage to entry price
                            # Long: slippage pushes entry price UP (worse for you)
                            # Short: slippage pushes entry price DOWN (worse for you)
                            if is_long:
                                actual_entry_price = price * (1 + entry_slippage_pct)
                            else:
                                actual_entry_price = price * (1 - entry_slippage_pct)

                            new_pos = PyramidPosition(
                                side=direction,
                                entry_price=actual_entry_price,  # Use slipped price
                                entry_time=timestamp,
                                size=pos_size
                            )
                            pyramid_positions.append(new_pos)
                            last_pyramid_price = actual_entry_price  # Track for spacing check

                            # Reserve margin for pyramid position (Phase 1.1)
                            if margin_state is not None:
                                margin_state.open_position(pyramid_notional)
                                # PHASE 5 FIX: Use consistent key scheme
                                # pyramid_1 = 10, pyramid_2 = 11, pyramid_3 = 12, etc.
                                position_notionals[len(pyramid_positions) + 9] = pyramid_notional

                            total_fees += pyramid_entry_cost
                            next_pyramid_level += 1

                            if verbose:
                                margin_info = ""
                                if margin_state:
                                    margin_info = f" | Margin: ${margin_state.used_margin:.2f}"
                                print(f"[{timestamp}] PYRAMID #{len(pyramid_positions)} @ ${price:.2f} "
                                      f"(+{move_from_ref:.1f}% from ref, size={pos_size:.2f}, cost={pyramid_entry_cost:.3f}%){margin_info}")
                    elif verbose and not spacing_ok:
                        print(f"[{timestamp}] PYRAMID SKIPPED (spacing {spacing:.2f}% < min {min_pyramid_spacing_pct}%)")
                    elif verbose and not volatility_ok:
                        print(f"[{timestamp}] PYRAMID SKIPPED (vol {current_vol:.2f}% < min {volatility_min_pct}%)")

            # === PHASE 1.1: Check for liquidation ===
            liquidation_exit = False
            if margin_state is not None and pyramid_positions:
                # Calculate unrealized P&L for all open positions
                unrealized_pnl = 0.0
                for i, pos in enumerate(pyramid_positions):
                    pos_pnl_pct = calculate_profit_pct(pos.entry_price, price, is_long)
                    # PHASE 5 FIX: Consistent key scheme
                    # pyramid_1 = 10, pyramid_2 = 11, etc. (i is 0-indexed)
                    pos_key = i + 10
                    if pos_key in position_notionals:
                        pos_notional = position_notionals[pos_key]
                    else:
                        pos_notional = position_size_usdt * pos.size
                    unrealized_pnl += (pos_pnl_pct / 100) * pos_notional

                margin_state.update_unrealized_pnl(unrealized_pnl)

                if margin_state.is_liquidated():
                    liquidation_exit = True
                    liquidation_exits += 1
                    margin_state.liquidation_events += 1
                    if verbose:
                        print(f"[{timestamp}] *** LIQUIDATION *** @ ${price:.2f} | "
                              f"Equity: ${margin_state.current_equity:.2f} | "
                              f"Margin ratio: {margin_state.get_margin_ratio():.1f}%")

            # Check time-based exit
            time_exit = False
            if time_decay_exit_seconds is not None and round_start_time is not None:
                time_elapsed = (timestamp - round_start_time).total_seconds()
                time_exit = time_elapsed >= time_decay_exit_seconds

            # Check trailing stop - CAUSAL vs LEGACY mode
            trail_exit = False
            if use_causal_trailing:
                # PHASE 1 FIX: Use causal trailing stop (no look-ahead bias)
                # Peak is only confirmed AFTER price reverses, matching real trading
                trail_exit = update_causal_trailing_state(
                    state=trailing_state,
                    current_price=price,
                    current_profit_pct=profit_from_entry,
                    current_time=timestamp,
                    trailing_pct=trailing_pct,
                    confirmation_ticks=confirmation_ticks,
                    confirmation_reversal_pct=confirmation_reversal_pct
                )
            else:
                # Legacy mode: instant max tracking (has look-ahead bias)
                trigger_profit = max_profit_pct - trailing_pct
                trail_exit = profit_from_entry <= trigger_profit

            if time_exit or trail_exit or liquidation_exit:
                # Close all pyramid positions
                round_pnl = 0.0
                exit_cost = get_cost(current_volatility, is_entry=False)

                # PHASE 5 FIX: Convert exit cost to actual price impact
                # Half of the exit cost is due to price moving against you (slippage)
                # The other half is fees (deducted from P&L separately)
                slippage_impact_pct = exit_cost * 0.5 / 100  # Convert to decimal

                # For liquidation, apply cascade slippage (5-15% typical)
                # This represents the market impact of forced liquidation
                if liquidation_exit:
                    cascade_slippage_pct = 0.10  # 10% cascade effect
                    slippage_impact_pct = cascade_slippage_pct

                # Calculate the actual exit price with market impact
                # Long positions: slippage pushes exit price DOWN (worse for you)
                # Short positions: slippage pushes exit price UP (worse for you)
                if is_long:
                    actual_exit_price = price * (1 - slippage_impact_pct)
                else:
                    actual_exit_price = price * (1 + slippage_impact_pct)

                for i, pos in enumerate(pyramid_positions):
                    pos.is_open = False
                    pos.exit_price = actual_exit_price  # Use impacted price
                    pos_pnl_pct = calculate_profit_pct(pos.entry_price, actual_exit_price, is_long) - exit_cost
                    pos.pnl_percent = pos_pnl_pct
                    round_pnl += pos_pnl_pct
                    total_fees += exit_cost

                    # Update margin state for closed position (Phase 1.1)
                    if margin_state is not None:
                        # PHASE 5 FIX: Consistent key scheme (matches entry keys)
                        pos_key = i + 10
                        if pos_key in position_notionals:
                            notional = position_notionals[pos_key]
                        else:
                            notional = position_size_usdt * pos.size
                        pnl_usdt = (pos_pnl_pct / 100) * notional
                        margin_state.close_position(notional, pnl_usdt)

                # Add the losing hedge P&L (already closed earlier and margin updated)
                losing_exit_cost = get_cost(current_volatility, is_entry=False)
                losing_pnl = -threshold_pct - losing_exit_cost
                round_pnl += losing_pnl

                # Clear position notionals for this round (Phase 1.1)
                position_notionals.clear()

                total_pnl += round_pnl
                total_rounds += 1
                total_pyramids += len(pyramid_positions)  # Track for avg calculation
                if round_pnl > 0:
                    winning_rounds += 1

                # For verbose output, show which peak was used
                if verbose:
                    if liquidation_exit:
                        exit_reason = "LIQUIDATION"
                    elif time_exit:
                        exit_reason = "TIME DECAY"
                    else:
                        exit_reason = "TRAILING STOP"

                    margin_info = ""
                    if margin_state:
                        margin_info = f" | Equity: ${margin_state.current_equity:.2f}"

                    if use_causal_trailing:
                        confirmed_max = trailing_state.confirmed_peak_pct
                        print(f"[{timestamp}] {exit_reason} @ ${price:.2f} | Pyramids: {len(pyramid_positions)} | "
                              f"Confirmed Max: {confirmed_max:+.2f}% | Profit: {profit_from_entry:+.2f}% | Round P&L: {round_pnl:+.2f}%{margin_info}")
                    else:
                        print(f"[{timestamp}] {exit_reason} @ ${price:.2f} | Pyramids: {len(pyramid_positions)} | "
                              f"Max: {max_profit_pct:+.2f}% | Profit: {profit_from_entry:+.2f}% | Round P&L: {round_pnl:+.2f}%{margin_info}")

                # Record round (only if return_rounds=True to save memory)
                if return_rounds:
                    rounds.append(PyramidRound(
                        entry_price=round_entry_price,
                        entry_time=round_entry_time,
                        direction=direction,
                        pyramid_reference=pyramid_reference,
                        positions=pyramid_positions.copy(),
                        max_profit_pct=max_profit_pct,
                        exit_price=price,
                        exit_time=timestamp,
                        total_pnl=round_pnl,
                        num_pyramids=len(pyramid_positions)
                    ))

                # Reset for next round
                pyramid_positions = []
                direction = ''
                pyramid_reference = 0.0
                max_profit_pct = 0.0
                next_pyramid_level = 1
                last_pyramid_price = 0.0  # Reset for min_pyramid_spacing
    
    # Calculate statistics
    if total_rounds == 0:
        result = {
            'threshold': threshold_pct,
            'trailing': trailing_pct,
            'pyramid_step': pyramid_step_pct,
            'total_rounds': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'win_rate': 0,
            'avg_pyramids': 0,
            'total_fees': total_fees
        }
        # Add margin metrics (Phase 1.1)
        if margin_state is not None:
            result['initial_capital'] = initial_capital
            result['final_equity'] = margin_state.current_equity
            result['total_return_pct'] = 0.0
            result['max_drawdown_pct'] = margin_state.max_drawdown_pct
            result['liquidation_events'] = margin_state.liquidation_events
            result['skipped_pyramids_margin'] = skipped_pyramids_margin
        # Add funding metrics (Phase 1.2)
        if funding_rates is not None:
            result['total_funding_paid'] = total_funding_paid
            result['total_funding_received'] = total_funding_received
            result['net_funding_impact'] = total_funding_received - total_funding_paid
            result['funding_events'] = funding_events
        if return_rounds:
            result['rounds'] = []
        return result

    # Calculate avg_pyramids from tracked total (works regardless of return_rounds)
    avg_pyramids = total_pyramids / total_rounds if total_rounds > 0 else 0

    result = {
        'threshold': threshold_pct,
        'trailing': trailing_pct,
        'pyramid_step': pyramid_step_pct,
        'total_rounds': total_rounds,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / total_rounds,
        'win_rate': (winning_rounds / total_rounds) * 100,
        'avg_pyramids': avg_pyramids,
        'total_fees': total_fees,
    }

    # Add margin metrics (Phase 1.1)
    if margin_state is not None:
        result['initial_capital'] = initial_capital
        result['final_equity'] = margin_state.current_equity
        result['total_return_pct'] = ((margin_state.current_equity - initial_capital) / initial_capital) * 100
        result['max_drawdown_pct'] = margin_state.max_drawdown_pct
        result['liquidation_events'] = margin_state.liquidation_events
        result['skipped_pyramids_margin'] = skipped_pyramids_margin
        result['leverage'] = leverage

    # Add funding metrics (Phase 1.2)
    if funding_rates is not None:
        result['total_funding_paid'] = total_funding_paid
        result['total_funding_received'] = total_funding_received
        result['net_funding_impact'] = total_funding_received - total_funding_paid
        result['funding_events'] = funding_events
        # Funding-adjusted P&L (what you actually keep after funding costs)
        net_funding_usdt = total_funding_received - total_funding_paid
        if margin_state is not None:
            result['funding_adjusted_return_pct'] = (
                (margin_state.current_equity - initial_capital) / initial_capital
            ) * 100
        else:
            # Approximate funding impact as % of total P&L
            result['funding_impact_pct'] = (net_funding_usdt / initial_capital) * 100 if initial_capital > 0 else 0

    # Only include rounds list if requested (saves memory in grid search)
    if return_rounds:
        result['rounds'] = rounds

    return result


# Test function
if __name__ == "__main__":
    # Simple test with synthetic data
    print("Testing pyramid backtest with margin tracking...")
    print("=" * 60)

    # Create test prices: $100 -> $110 -> $105
    from datetime import timedelta

    test_prices = []
    base_time = datetime.now()

    # Uptrend
    for i in range(100):
        price = 100 + i * 0.1
        test_prices.append((base_time + timedelta(minutes=i), price))

    # Reversal
    for i in range(50):
        price = 110 - i * 0.1
        test_prices.append((base_time + timedelta(minutes=100+i), price))

    # Test WITH margin tracking (Phase 1.1)
    print("\n--- Test with Margin Tracking (10x leverage, $1000 capital) ---")
    result = run_pyramid_backtest(
        test_prices,
        threshold_pct=1.0,
        trailing_pct=1.0,
        pyramid_step_pct=2.0,
        verbose=True,
        # Margin parameters
        initial_capital=1000.0,
        leverage=10,
        position_size_usdt=50.0,
        use_margin_tracking=True
    )

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Total P&L: {result['total_pnl']:+.2f}%")
    print(f"  Rounds: {result['total_rounds']}")
    print(f"  Avg pyramids: {result['avg_pyramids']:.1f}")
    print(f"  Win rate: {result['win_rate']:.1f}%")

    if 'final_equity' in result:
        print(f"\nMARGIN METRICS:")
        print(f"  Initial capital: ${result['initial_capital']:.2f}")
        print(f"  Final equity: ${result['final_equity']:.2f}")
        print(f"  Total return: {result['total_return_pct']:+.2f}%")
        print(f"  Max drawdown: {result['max_drawdown_pct']:.2f}%")
        print(f"  Liquidation events: {result['liquidation_events']}")
        print(f"  Skipped pyramids (margin): {result['skipped_pyramids_margin']}")
        print(f"  Leverage: {result['leverage']}x")

    # Test WITHOUT margin tracking (legacy mode)
    print("\n--- Test without Margin Tracking (legacy mode) ---")
    result_legacy = run_pyramid_backtest(
        test_prices,
        threshold_pct=1.0,
        trailing_pct=1.0,
        pyramid_step_pct=2.0,
        verbose=False,
        use_margin_tracking=False
    )
    print(f"  Legacy P&L: {result_legacy['total_pnl']:+.2f}%")
    print(f"  (No margin metrics in legacy mode)")

    print(f"\n{'='*60}")
    print("Phase 1.1 (Margin & Liquidation) test completed!")
