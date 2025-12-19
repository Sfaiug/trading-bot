#!/usr/bin/env python3
"""
State Manager - Tracks orders and positions per bot instance

Provides isolated tracking of orders so multiple bot instances
can run on the same coin without interfering with each other.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


STATE_DIR = "state"


@dataclass
class TrackedOrder:
    """A single tracked order."""
    client_order_id: str
    exchange_order_id: int
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    created_at: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrackedOrder':
        return cls(**data)


class StateManager:
    """
    Manages state for a single bot instance.
    Tracks orders with unique client order IDs.
    Persists state to disk for crash recovery.
    """
    
    def __init__(self, mode: str = "trading", symbol: str = "SOLUSDT"):
        """
        Initialize state manager.
        
        Args:
            mode: Bot mode ('trading' or 'funding')
            symbol: Trading pair
        """
        self.instance_id = self._generate_instance_id()
        self.mode = mode
        self.symbol = symbol
        self.orders: Dict[str, TrackedOrder] = {}
        self.created_at = datetime.now().isoformat()
        
        # Ensure state directory exists
        os.makedirs(STATE_DIR, exist_ok=True)
        
        print(f"📋 State Manager initialized")
        print(f"   Instance ID: {self.instance_id}")
        print(f"   State file:  {self._state_file}")
    
    def _generate_instance_id(self) -> str:
        """Generate a unique 8-character instance ID."""
        return str(uuid.uuid4())[:8]
    
    @property
    def _state_file(self) -> str:
        """Path to state file for this instance."""
        return os.path.join(STATE_DIR, f"{self.instance_id}.json")
    
    def generate_client_order_id(self, side: str) -> str:
        """
        Generate a unique client order ID for an order.
        
        Format: {instance_id}_{side}_{timestamp}
        Example: a1b2c3d4_LONG_1703001234567
        """
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"{self.instance_id}_{side}_{timestamp}"
    
    def add_order(
        self,
        client_order_id: str,
        exchange_order_id: int,
        side: str,
        quantity: float,
        entry_price: float
    ) -> TrackedOrder:
        """
        Add a new order to track.
        
        Args:
            client_order_id: Our custom order ID
            exchange_order_id: Binance's order ID
            side: 'LONG' or 'SHORT'
            quantity: Position size
            entry_price: Entry price
            
        Returns:
            The created TrackedOrder
        """
        order = TrackedOrder(
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            created_at=datetime.now().isoformat()
        )
        
        self.orders[client_order_id] = order
        self._save_state()
        
        return order
    
    def remove_order(self, client_order_id: str) -> Optional[TrackedOrder]:
        """
        Remove an order from tracking.
        
        Args:
            client_order_id: The order to remove
            
        Returns:
            The removed order, or None if not found
        """
        order = self.orders.pop(client_order_id, None)
        if order:
            self._save_state()
        return order
    
    def get_orders_by_side(self, side: str) -> List[TrackedOrder]:
        """Get all orders for a specific side."""
        return [o for o in self.orders.values() if o.side == side]
    
    def get_all_orders(self) -> List[TrackedOrder]:
        """Get all tracked orders."""
        return list(self.orders.values())
    
    def get_total_quantity(self, side: str) -> float:
        """Get total quantity for a side."""
        return sum(o.quantity for o in self.get_orders_by_side(side))
    
    def clear_all_orders(self):
        """Clear all tracked orders."""
        self.orders.clear()
        self._save_state()
    
    def _save_state(self):
        """Save current state to disk."""
        state = {
            'instance_id': self.instance_id,
            'mode': self.mode,
            'symbol': self.symbol,
            'created_at': self.created_at,
            'last_updated': datetime.now().isoformat(),
            'orders': [o.to_dict() for o in self.orders.values()]
        }
        
        with open(self._state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self) -> bool:
        """
        Load state from disk.
        
        Returns:
            True if state was loaded, False otherwise
        """
        if not os.path.exists(self._state_file):
            return False
        
        try:
            with open(self._state_file, 'r') as f:
                state = json.load(f)
            
            self.mode = state.get('mode', self.mode)
            self.symbol = state.get('symbol', self.symbol)
            self.created_at = state.get('created_at', self.created_at)
            
            self.orders = {}
            for order_data in state.get('orders', []):
                order = TrackedOrder.from_dict(order_data)
                self.orders[order.client_order_id] = order
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not load state: {e}")
            return False
    
    def cleanup_state_file(self):
        """Delete the state file (call when instance is done)."""
        if os.path.exists(self._state_file):
            os.remove(self._state_file)
    
    def print_status(self):
        """Print current tracked orders."""
        print(f"\n📋 Instance {self.instance_id} Orders:")
        if not self.orders:
            print("   No orders tracked")
        else:
            for order in self.orders.values():
                print(f"   {order.side}: {order.quantity} @ ${order.entry_price:.4f}")
                print(f"      ID: {order.client_order_id}")


# Utility to list all active instances
def list_active_instances() -> List[str]:
    """List all active instance IDs from state files."""
    if not os.path.exists(STATE_DIR):
        return []
    
    instances = []
    for filename in os.listdir(STATE_DIR):
        if filename.endswith('.json'):
            instances.append(filename.replace('.json', ''))
    
    return instances


def cleanup_stale_instances(max_age_hours: int = 24):
    """Remove state files older than max_age_hours."""
    if not os.path.exists(STATE_DIR):
        return
    
    now = datetime.now()
    
    for filename in os.listdir(STATE_DIR):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(STATE_DIR, filename)
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            last_updated = datetime.fromisoformat(state.get('last_updated', ''))
            age_hours = (now - last_updated).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                os.remove(filepath)
                print(f"Cleaned up stale instance: {filename}")
                
        except Exception:
            pass


if __name__ == "__main__":
    # Test the state manager
    print("Testing State Manager...")
    
    sm = StateManager(mode="trading", symbol="SOLUSDT")
    
    # Add some orders
    cid1 = sm.generate_client_order_id("LONG")
    sm.add_order(cid1, 12345, "LONG", 1.0, 124.50)
    
    cid2 = sm.generate_client_order_id("LONG")
    sm.add_order(cid2, 12346, "LONG", 1.0, 126.00)
    
    sm.print_status()
    
    print(f"\nTotal LONG quantity: {sm.get_total_quantity('LONG')}")
    
    # Remove an order
    sm.remove_order(cid1)
    print(f"\nAfter removing first order:")
    sm.print_status()
    
    # Cleanup
    sm.cleanup_state_file()
    print("\n✓ State file cleaned up")
