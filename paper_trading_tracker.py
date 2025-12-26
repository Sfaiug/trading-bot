#!/usr/bin/env python3
"""
Paper Trading Tracker for v5 Optimizer Results

This script tracks paper trading performance over the required 8-week
validation period before going live.

Usage:
    # Initialize tracking for a winner
    python paper_trading_tracker.py init --params-file optimization_v5_results/BTCUSDT_winner_v5.json

    # Log a completed round
    python paper_trading_tracker.py log --pnl 2.5 --duration 3600 --direction LONG

    # View current status
    python paper_trading_tracker.py status

    # Check go/no-go criteria
    python paper_trading_tracker.py evaluate
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


# =============================================================================
# GO/NO-GO CRITERIA (From the plan)
# =============================================================================

GO_LIVE_CRITERIA = {
    'min_weeks': 8,
    'min_pnl_pct': 0.0,  # Must be net positive
    'min_win_rate': 40.0,
    'max_drawdown_pct': 25.0,
    'min_sharpe': 0.5,
    'min_rounds': 20,
    'min_positive_weeks': 4,  # At least 4 of 8 weeks must be positive
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PaperRound:
    """A single paper trading round."""
    timestamp: str
    pnl_pct: float
    duration_sec: float
    direction: str
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    num_pyramids: int = 0
    notes: str = ""


@dataclass
class PaperTradingSession:
    """Paper trading session data."""
    symbol: str
    params: Dict[str, Any]
    start_date: str
    target_end_date: str  # 8 weeks from start
    rounds: List[Dict] = field(default_factory=list)
    weekly_pnl: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# TRACKER CLASS
# =============================================================================

class PaperTradingTracker:
    """Track and evaluate paper trading performance."""

    def __init__(self, data_file: str = "./paper_trading_data.json"):
        self.data_file = data_file
        self.session: Optional[PaperTradingSession] = None
        self._load()

    def _load(self):
        """Load existing session data if available."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.session = PaperTradingSession(**data)

    def _save(self):
        """Save session data to disk."""
        if self.session:
            with open(self.data_file, 'w') as f:
                json.dump(asdict(self.session), f, indent=2, default=str)

    def init_session(self, symbol: str, params: Dict[str, Any]):
        """Initialize a new paper trading session."""
        start = datetime.now()
        end = start + timedelta(weeks=8)

        self.session = PaperTradingSession(
            symbol=symbol,
            params=params,
            start_date=start.isoformat(),
            target_end_date=end.isoformat(),
            rounds=[],
            weekly_pnl={}
        )
        self._save()

        print(f"Paper trading session initialized for {symbol}")
        print(f"Start: {start.strftime('%Y-%m-%d')}")
        print(f"Target end: {end.strftime('%Y-%m-%d')} (8 weeks)")
        print(f"Parameters: {json.dumps(params, indent=2, default=str)}")

    def log_round(
        self,
        pnl_pct: float,
        duration_sec: float,
        direction: str,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        num_pyramids: int = 0,
        notes: str = ""
    ):
        """Log a completed trading round."""
        if not self.session:
            print("Error: No active session. Run 'init' first.")
            return

        round_data = {
            'timestamp': datetime.now().isoformat(),
            'pnl_pct': pnl_pct,
            'duration_sec': duration_sec,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'num_pyramids': num_pyramids,
            'notes': notes
        }
        self.session.rounds.append(round_data)

        # Update weekly P&L
        week_key = datetime.now().strftime('%Y-W%W')
        if week_key not in self.session.weekly_pnl:
            self.session.weekly_pnl[week_key] = 0.0
        self.session.weekly_pnl[week_key] += pnl_pct

        self._save()

        print(f"Logged round: {direction} {pnl_pct:+.2f}% ({duration_sec/3600:.1f}h)")
        print(f"Total rounds: {len(self.session.rounds)}")

    def get_status(self) -> Dict[str, Any]:
        """Get current paper trading status."""
        if not self.session:
            return {'error': 'No active session'}

        rounds = self.session.rounds
        if not rounds:
            return {
                'symbol': self.session.symbol,
                'start_date': self.session.start_date,
                'target_end_date': self.session.target_end_date,
                'total_rounds': 0,
                'total_pnl': 0.0,
                'weeks_elapsed': 0,
            }

        # Calculate metrics
        pnls = [r['pnl_pct'] for r in rounds]
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = (wins / len(pnls)) * 100 if pnls else 0

        # Calculate drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Calculate Sharpe
        if len(pnls) >= 2:
            mean = total_pnl / len(pnls)
            variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
            std = variance ** 0.5
            avg_duration = sum(r['duration_sec'] for r in rounds) / len(rounds)
            trades_per_year = 365 * 24 * 3600 / avg_duration if avg_duration > 0 else 10
            sharpe = (mean / std) * (trades_per_year ** 0.5) if std > 0 else 0
        else:
            sharpe = 0

        # Weeks elapsed
        start = datetime.fromisoformat(self.session.start_date)
        weeks_elapsed = (datetime.now() - start).days / 7

        # Positive weeks
        positive_weeks = sum(1 for w in self.session.weekly_pnl.values() if w > 0)

        return {
            'symbol': self.session.symbol,
            'start_date': self.session.start_date,
            'target_end_date': self.session.target_end_date,
            'weeks_elapsed': weeks_elapsed,
            'total_rounds': len(rounds),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'positive_weeks': positive_weeks,
            'total_weeks_tracked': len(self.session.weekly_pnl),
            'weekly_pnl': self.session.weekly_pnl,
        }

    def evaluate_go_live(self) -> Dict[str, Any]:
        """Evaluate if strategy meets go/no-go criteria for live trading."""
        status = self.get_status()

        if 'error' in status:
            return {'decision': 'NO_SESSION', 'reason': status['error']}

        checks = {}
        all_pass = True

        # Check 1: Minimum weeks
        weeks = status['weeks_elapsed']
        checks['min_weeks'] = {
            'required': GO_LIVE_CRITERIA['min_weeks'],
            'actual': weeks,
            'pass': weeks >= GO_LIVE_CRITERIA['min_weeks']
        }
        if not checks['min_weeks']['pass']:
            all_pass = False

        # Check 2: Minimum P&L
        pnl = status['total_pnl']
        checks['min_pnl'] = {
            'required': f"> {GO_LIVE_CRITERIA['min_pnl_pct']}%",
            'actual': f"{pnl:.2f}%",
            'pass': pnl > GO_LIVE_CRITERIA['min_pnl_pct']
        }
        if not checks['min_pnl']['pass']:
            all_pass = False

        # Check 3: Win rate
        wr = status['win_rate']
        checks['win_rate'] = {
            'required': f">= {GO_LIVE_CRITERIA['min_win_rate']}%",
            'actual': f"{wr:.1f}%",
            'pass': wr >= GO_LIVE_CRITERIA['min_win_rate']
        }
        if not checks['win_rate']['pass']:
            all_pass = False

        # Check 4: Max drawdown
        dd = status['max_drawdown']
        checks['max_drawdown'] = {
            'required': f"< {GO_LIVE_CRITERIA['max_drawdown_pct']}%",
            'actual': f"{dd:.1f}%",
            'pass': dd < GO_LIVE_CRITERIA['max_drawdown_pct']
        }
        if not checks['max_drawdown']['pass']:
            all_pass = False

        # Check 5: Sharpe ratio
        sharpe = status['sharpe']
        checks['sharpe'] = {
            'required': f"> {GO_LIVE_CRITERIA['min_sharpe']}",
            'actual': f"{sharpe:.2f}",
            'pass': sharpe > GO_LIVE_CRITERIA['min_sharpe']
        }
        if not checks['sharpe']['pass']:
            all_pass = False

        # Check 6: Minimum rounds
        rounds = status['total_rounds']
        checks['min_rounds'] = {
            'required': f">= {GO_LIVE_CRITERIA['min_rounds']}",
            'actual': rounds,
            'pass': rounds >= GO_LIVE_CRITERIA['min_rounds']
        }
        if not checks['min_rounds']['pass']:
            all_pass = False

        # Check 7: Positive weeks (4 of 8)
        pos_weeks = status['positive_weeks']
        checks['positive_weeks'] = {
            'required': f">= {GO_LIVE_CRITERIA['min_positive_weeks']} of 8",
            'actual': f"{pos_weeks} of {status['total_weeks_tracked']}",
            'pass': pos_weeks >= GO_LIVE_CRITERIA['min_positive_weeks']
        }
        if not checks['positive_weeks']['pass']:
            all_pass = False

        decision = 'GO' if all_pass else 'NO_GO'

        return {
            'decision': decision,
            'checks': checks,
            'status': status,
            'recommendation': (
                "Strategy APPROVED for live trading with real capital."
                if all_pass else
                "Strategy NOT APPROVED. Continue paper trading or re-optimize."
            )
        }

    def print_status(self):
        """Print formatted status report."""
        status = self.get_status()

        if 'error' in status:
            print(f"Error: {status['error']}")
            return

        print("=" * 60)
        print("PAPER TRADING STATUS REPORT")
        print("=" * 60)
        print(f"Symbol: {status['symbol']}")
        print(f"Started: {status['start_date'][:10]}")
        print(f"Target End: {status['target_end_date'][:10]}")
        print(f"Weeks Elapsed: {status['weeks_elapsed']:.1f} / 8")
        print()
        print(f"Total Rounds: {status['total_rounds']}")
        print(f"Total P&L: {status['total_pnl']:+.2f}%")
        print(f"Win Rate: {status['win_rate']:.1f}%")
        print(f"Max Drawdown: {status['max_drawdown']:.1f}%")
        print(f"Sharpe Ratio: {status['sharpe']:.2f}")
        print()
        print("Weekly P&L:")
        for week, pnl in sorted(status['weekly_pnl'].items()):
            print(f"  {week}: {pnl:+.2f}%")
        print("=" * 60)

    def print_evaluation(self):
        """Print go/no-go evaluation."""
        result = self.evaluate_go_live()

        print("=" * 60)
        print("GO/NO-GO EVALUATION")
        print("=" * 60)

        if result['decision'] == 'NO_SESSION':
            print(f"Error: {result['reason']}")
            return

        for check_name, check_data in result['checks'].items():
            status = "PASS" if check_data['pass'] else "FAIL"
            print(f"[{status}] {check_name}: {check_data['actual']} (required: {check_data['required']})")

        print()
        print(f"DECISION: {result['decision']}")
        print(f"Recommendation: {result['recommendation']}")
        print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper Trading Tracker")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize paper trading session')
    init_parser.add_argument('--params-file', required=True, help='Path to winner params JSON')
    init_parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')

    # Log command
    log_parser = subparsers.add_parser('log', help='Log a completed round')
    log_parser.add_argument('--pnl', type=float, required=True, help='P&L percentage')
    log_parser.add_argument('--duration', type=float, required=True, help='Duration in seconds')
    log_parser.add_argument('--direction', required=True, choices=['LONG', 'SHORT'])
    log_parser.add_argument('--entry', type=float, help='Entry price')
    log_parser.add_argument('--exit', type=float, help='Exit price')
    log_parser.add_argument('--pyramids', type=int, default=0, help='Number of pyramids')
    log_parser.add_argument('--notes', default='', help='Notes')

    # Status command
    subparsers.add_parser('status', help='Show current status')

    # Evaluate command
    subparsers.add_parser('evaluate', help='Evaluate go/no-go criteria')

    args = parser.parse_args()

    tracker = PaperTradingTracker()

    if args.command == 'init':
        with open(args.params_file, 'r') as f:
            data = json.load(f)
        params = data.get('params', data)
        tracker.init_session(args.symbol, params)

    elif args.command == 'log':
        tracker.log_round(
            pnl_pct=args.pnl,
            duration_sec=args.duration,
            direction=args.direction,
            entry_price=args.entry,
            exit_price=args.exit,
            num_pyramids=args.pyramids,
            notes=args.notes
        )

    elif args.command == 'status':
        tracker.print_status()

    elif args.command == 'evaluate':
        tracker.print_evaluation()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
