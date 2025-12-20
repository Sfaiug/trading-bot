"""
Rich Terminal Dashboard for Trading Bot

Provides a live-updating terminal UI that refreshes in place
with real-time trading stats, positions, and activity log.
"""

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class SessionStats:
    """Tracks session-wide statistics."""
    rounds: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    best_round: float = 0.0
    worst_round: float = 0.0
    
    @property
    def win_rate(self) -> float:
        if self.rounds == 0:
            return 0.0
        return (self.wins / self.rounds) * 100
    
    @property
    def avg_pnl(self) -> float:
        if self.rounds == 0:
            return 0.0
        return self.total_pnl / self.rounds


@dataclass  
class RoundState:
    """Tracks current round state."""
    number: int = 0
    phase: str = "WAITING"  # HEDGE, PYRAMIDING, CLOSED
    direction: str = ""
    pyramids: int = 0
    max_pyramids: int = 10
    current_profit: float = 0.0
    max_profit: float = 0.0
    trigger_profit: float = 0.0
    entry_price: float = 0.0


@dataclass
class Position:
    """Represents a single position."""
    side: str
    entry_price: float
    current_price: float
    size: float
    number: int = 1
    
    @property
    def pnl_pct(self) -> float:
        if self.side == "LONG":
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100


class TradingDashboard:
    """Live-updating terminal dashboard for trading."""
    
    def __init__(
        self,
        mode: str,
        symbol: str,
        leverage: int,
        dollar_amount: float,
        threshold_pct: float = 0.0,
        trailing_pct: float = 0.0,
        pyramid_step_pct: float = 0.0,
        entry_threshold: float = 0.0,
        exit_threshold: float = 0.0,
        max_pyramids: int = 10
    ):
        self.mode = mode
        self.symbol = symbol
        self.leverage = leverage
        self.dollar_amount = dollar_amount
        self.threshold_pct = threshold_pct
        self.trailing_pct = trailing_pct
        self.pyramid_step_pct = pyramid_step_pct
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_pyramids = max_pyramids
        
        self.console = Console()
        self.live: Optional[Live] = None
        
        # State
        self.current_price: float = 0.0
        self.wallet_balance: float = 0.0
        self.available_balance: float = 0.0
        self.stats = SessionStats()
        self.round = RoundState(max_pyramids=max_pyramids)
        self.positions: List[Position] = []
        self.logs: List[str] = []
        
        # Funding mode specific
        self.funding_rate: float = 0.0
        self.next_funding_mins: int = 0
        self.funding_collected: float = 0.0
        self.funding_payments: int = 0
    
    def start(self):
        """Start the live display."""
        self.live = Live(
            self._generate_layout(),
            console=self.console,
            refresh_per_second=2,
            screen=True
        )
        self.live.start()
    
    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()
    
    def update(self):
        """Refresh the display."""
        if self.live:
            self.live.update(self._generate_layout())
    
    def set_price(self, price: float):
        """Update current price."""
        self.current_price = price
        self._update_position_prices(price)
        self.update()
    
    def set_balance(self, wallet: float, available: float):
        """Update balance."""
        self.wallet_balance = wallet
        self.available_balance = available
        self.update()
    
    def set_funding_info(self, rate: float, next_mins: int):
        """Update funding rate info."""
        self.funding_rate = rate
        self.next_funding_mins = next_mins
        self.update()
    
    def start_round(self, round_num: int, entry_price: float):
        """Start a new trading round."""
        self.round = RoundState(
            number=round_num,
            phase="HEDGE",
            entry_price=entry_price,
            max_pyramids=self.max_pyramids
        )
        self.positions = []
        self.add_log(f"Round #{round_num} started @ ${entry_price:.4f}")
        self.update()
    
    def set_direction(self, direction: str, price: float):
        """Set the round direction."""
        self.round.direction = direction
        self.round.phase = "PYRAMIDING"
        emoji = "📈" if direction == "LONG" else "📉"
        self.add_log(f"{emoji} Direction: {direction} @ ${price:.4f}")
        self.update()
    
    def add_pyramid(self, price: float, pyramid_num: int):
        """Add a pyramid position."""
        self.round.pyramids = pyramid_num
        self.add_log(f"🔺 Pyramid #{pyramid_num} @ ${price:.4f}")
        self.update()
    
    def set_profits(self, current: float, max_profit: float, trigger: float):
        """Update profit tracking."""
        self.round.current_profit = current
        self.round.max_profit = max_profit
        self.round.trigger_profit = trigger
        self.update()
    
    def end_round(self, pnl: float, is_win: bool):
        """End the current round."""
        self.stats.rounds += 1
        if is_win:
            self.stats.wins += 1
        else:
            self.stats.losses += 1
        self.stats.total_pnl += pnl
        
        if pnl > self.stats.best_round:
            self.stats.best_round = pnl
        if pnl < self.stats.worst_round or self.stats.worst_round == 0:
            self.stats.worst_round = pnl
        
        emoji = "✅" if is_win else "❌"
        self.add_log(f"{emoji} Round #{self.round.number} closed: {pnl:+.2f}%")
        self.round.phase = "CLOSED"
        self.positions = []
        self.update()
    
    def add_position(self, side: str, entry_price: float, size: float, number: int = 1):
        """Add a position to track."""
        self.positions.append(Position(
            side=side,
            entry_price=entry_price,
            current_price=self.current_price,
            size=size,
            number=number
        ))
        self.update()
    
    def clear_positions(self):
        """Clear all positions."""
        self.positions = []
        self.update()
    
    def add_log(self, message: str):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[dim]{timestamp}[/dim] {message}")
        # Keep only last 8 logs
        if len(self.logs) > 8:
            self.logs = self.logs[-8:]
        self.update()
    
    def _update_position_prices(self, price: float):
        """Update all position current prices."""
        for pos in self.positions:
            pos.current_price = price
    
    def _generate_layout(self) -> Layout:
        """Generate the dashboard layout."""
        layout = Layout()
        
        # Create main sections
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Body has two columns
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        # Left column: stats + positions
        layout["left"].split_column(
            Layout(name="stats", size=12),
            Layout(name="positions")
        )
        
        # Right column: round + logs
        layout["right"].split_column(
            Layout(name="round", size=12),
            Layout(name="logs")
        )
        
        # Populate sections
        layout["header"].update(self._make_header())
        layout["stats"].update(self._make_stats())
        layout["round"].update(self._make_round())
        layout["positions"].update(self._make_positions())
        layout["logs"].update(self._make_logs())
        layout["footer"].update(self._make_footer())
        
        return layout
    
    def _make_header(self) -> Panel:
        """Create header panel."""
        title = "PYRAMID TRADING" if self.mode == "trading" else "FUNDING RATE FARMING"
        
        if self.mode == "trading":
            info = (
                f"Symbol: [bold cyan]{self.symbol}[/] | "
                f"Leverage: [bold]{self.leverage}x[/] | "
                f"Price: [bold green]${self.current_price:.4f}[/] | "
                f"Balance: [bold]${self.wallet_balance:.2f}[/]"
            )
        else:
            rate_color = "green" if self.funding_rate > 0 else "red" if self.funding_rate < 0 else "white"
            info = (
                f"Symbol: [bold cyan]{self.symbol}[/] | "
                f"Funding: [bold {rate_color}]{self.funding_rate:+.4f}%[/] | "
                f"Next: [bold]{self.next_funding_mins}m[/] | "
                f"Balance: [bold]${self.wallet_balance:.2f}[/]"
            )
        
        return Panel(
            Text.from_markup(info, justify="center"),
            title=f"[bold white]🤖 {title} BOT[/]",
            border_style="blue",
            box=box.DOUBLE
        )
    
    def _make_stats(self) -> Panel:
        """Create session stats panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value", justify="right")
        
        pnl_color = "green" if self.stats.total_pnl >= 0 else "red"
        
        table.add_row("Rounds", f"[bold]{self.stats.rounds}[/]")
        table.add_row("Wins", f"[green]{self.stats.wins}[/] ({self.stats.win_rate:.1f}%)")
        table.add_row("Losses", f"[red]{self.stats.losses}[/]")
        table.add_row("─" * 12, "─" * 10)
        table.add_row("Total P&L", f"[bold {pnl_color}]{self.stats.total_pnl:+.2f}%[/]")
        table.add_row("Avg P&L", f"{self.stats.avg_pnl:+.2f}%")
        table.add_row("Best", f"[green]{self.stats.best_round:+.2f}%[/]")
        table.add_row("Worst", f"[red]{self.stats.worst_round:+.2f}%[/]")
        
        return Panel(table, title="[bold]📊 SESSION STATS[/]", border_style="cyan")
    
    def _make_round(self) -> Panel:
        """Create current round panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value", justify="right")
        
        if self.mode == "trading":
            dir_emoji = "📈" if self.round.direction == "LONG" else "📉" if self.round.direction == "SHORT" else "⏳"
            dir_color = "green" if self.round.direction == "LONG" else "red" if self.round.direction == "SHORT" else "white"
            
            table.add_row("Round", f"[bold]#{self.round.number}[/]")
            table.add_row("Phase", f"[yellow]{self.round.phase}[/]")
            table.add_row("Direction", f"[{dir_color}]{dir_emoji} {self.round.direction or 'WAITING'}[/]")
            table.add_row("Pyramids", f"[bold]{self.round.pyramids}[/] / {self.max_pyramids}")
            table.add_row("─" * 12, "─" * 10)
            
            profit_color = "green" if self.round.current_profit >= 0 else "red"
            table.add_row("Profit", f"[bold {profit_color}]{self.round.current_profit:+.2f}%[/]")
            table.add_row("Max", f"[green]{self.round.max_profit:+.2f}%[/]")
            table.add_row("Trigger", f"[yellow]{self.round.trigger_profit:+.2f}%[/]")
        else:
            # Funding mode
            dir_emoji = "📉" if self.round.direction == "SHORT" else "📈" if self.round.direction == "LONG" else "⏳"
            dir_color = "red" if self.round.direction == "SHORT" else "green" if self.round.direction == "LONG" else "white"
            
            table.add_row("Status", f"[bold]{self.round.phase}[/]")
            table.add_row("Position", f"[{dir_color}]{dir_emoji} {self.round.direction or 'NONE'}[/]")
            table.add_row("Entry", f"${self.round.entry_price:.4f}" if self.round.entry_price else "-")
            table.add_row("─" * 12, "─" * 10)
            table.add_row("Payments", f"[bold]{self.funding_payments}[/]")
            table.add_row("Collected", f"[green]{self.funding_collected:+.4f}%[/]")
            table.add_row("Price P&L", f"{self.round.current_profit:+.2f}%")
        
        return Panel(table, title="[bold]🎯 CURRENT ROUND[/]", border_style="magenta")
    
    def _make_positions(self) -> Panel:
        """Create positions table."""
        table = Table(box=box.SIMPLE, padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        table.add_column("Side", width=6)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("Size", justify="right", width=8)
        
        if not self.positions:
            table.add_row("", "[dim]No positions[/]", "", "", "")
        else:
            for pos in self.positions:
                pnl_color = "green" if pos.pnl_pct >= 0 else "red"
                side_color = "green" if pos.side == "LONG" else "red"
                table.add_row(
                    str(pos.number),
                    f"[{side_color}]{pos.side}[/]",
                    f"${pos.entry_price:.4f}",
                    f"[{pnl_color}]{pos.pnl_pct:+.2f}%[/]",
                    f"{pos.size:.2f}"
                )
        
        return Panel(table, title="[bold]📋 POSITIONS[/]", border_style="green")
    
    def _make_logs(self) -> Panel:
        """Create activity log panel."""
        if not self.logs:
            content = "[dim]Waiting for activity...[/]"
        else:
            content = "\n".join(self.logs)
        
        return Panel(
            Text.from_markup(content),
            title="[bold]📜 ACTIVITY LOG[/]",
            border_style="yellow"
        )
    
    def _make_footer(self) -> Panel:
        """Create footer panel."""
        if self.mode == "trading":
            settings = (
                f"Threshold: {self.threshold_pct}% | "
                f"Trailing: {self.trailing_pct}% | "
                f"Pyramid: {self.pyramid_step_pct}% | "
                f"Size: ${self.dollar_amount:.0f}"
            )
        else:
            settings = (
                f"Entry: >{self.entry_threshold}% | "
                f"Exit: <{self.exit_threshold}% | "
                f"Size: ${self.dollar_amount:.0f}"
            )
        
        return Panel(
            Text.from_markup(f"[dim]{settings}[/]  |  [bold]Press Ctrl+C to stop[/]", justify="center"),
            border_style="dim"
        )


# Simple test
if __name__ == "__main__":
    import time
    
    dashboard = TradingDashboard(
        mode="trading",
        symbol="SOLUSDT",
        leverage=5,
        dollar_amount=10,
        threshold_pct=2.0,
        trailing_pct=5.0,
        pyramid_step_pct=0.5
    )
    
    dashboard.set_balance(5000, 4500)
    dashboard.set_price(125.50)
    dashboard.start()
    
    try:
        dashboard.start_round(1, 125.50)
        time.sleep(2)
        
        dashboard.add_position("LONG", 125.50, 1.0, 1)
        dashboard.add_position("SHORT", 125.50, 1.0, 1)
        time.sleep(2)
        
        dashboard.set_direction("LONG", 126.00)
        dashboard.clear_positions()
        dashboard.add_position("LONG", 125.50, 1.0, 1)
        time.sleep(2)
        
        for i in range(3):
            dashboard.set_price(126.50 + i)
            dashboard.set_profits(2.0 + i, 3.0 + i, 1.0)
            dashboard.add_pyramid(126.50 + i * 0.5, i + 2)
            dashboard.add_position("LONG", 126.50 + i * 0.5, 1.0, i + 2)
            time.sleep(1)
        
        dashboard.end_round(5.5, True)
        time.sleep(3)
        
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()
