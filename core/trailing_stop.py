"""
Pure logic for trailing stop calculations.
All percentages are calculated relative to entry price.
"""


def calculate_profit_pct(entry_price: float, current_price: float, is_long: bool) -> float:
    """
    Calculate current profit percentage relative to entry price.
    
    Args:
        entry_price: Price at which position was opened
        current_price: Current market price
        is_long: True for long positions, False for short
    
    Returns:
        Profit as a percentage (e.g., 5.0 for 5% profit, -2.0 for 2% loss)
    """
    if entry_price <= 0:
        return 0.0
    
    if is_long:
        # Long profits when price goes up
        return ((current_price - entry_price) / entry_price) * 100
    else:
        # Short profits when price goes down
        return ((entry_price - current_price) / entry_price) * 100


def should_close_long(
    entry_price: float,
    current_price: float,
    max_profit_pct: float,
    threshold_percent: float
) -> bool:
    """
    Determine if a long position should be closed.
    
    A long position closes when current profit drops X% below max profit.
    All percentages are relative to entry price.
    
    Example (1% threshold, entry $100):
        - Price at $105 = +5% profit, max_profit = 5%
        - Price drops to $104 = +4% profit
        - 4% < 5% - 1% = 4%, so trigger = True
    
    Args:
        entry_price: Price at which position was opened
        current_price: Current market price
        max_profit_pct: Highest profit % reached since position opened
        threshold_percent: Drop from max profit that triggers close (e.g., 1.0 for 1%)
    
    Returns:
        True if position should be closed
    """
    current_profit = calculate_profit_pct(entry_price, current_price, is_long=True)
    trigger_profit = max_profit_pct - threshold_percent
    
    return current_profit <= trigger_profit


def should_close_short(
    entry_price: float,
    current_price: float,
    max_profit_pct: float,
    threshold_percent: float
) -> bool:
    """
    Determine if a short position should be closed.
    
    A short position closes when current profit drops X% below max profit.
    All percentages are relative to entry price.
    
    Example (1% threshold, entry $100):
        - Price at $99 = +1% profit (for short), triggers close if this is 1% against entry
        - Actually: short closes immediately when price rises 1% from entry
    
    Args:
        entry_price: Price at which position was opened
        current_price: Current market price  
        max_profit_pct: Highest profit % reached since position opened
        threshold_percent: Drop from max profit that triggers close (e.g., 1.0 for 1%)
    
    Returns:
        True if position should be closed
    """
    current_profit = calculate_profit_pct(entry_price, current_price, is_long=False)
    trigger_profit = max_profit_pct - threshold_percent
    
    return current_profit <= trigger_profit


def update_max_profit(
    entry_price: float,
    current_price: float,
    current_max_profit: float,
    is_long: bool
) -> float:
    """
    Update the maximum profit percentage if current profit exceeds it.
    
    Args:
        entry_price: Price at which position was opened
        current_price: Current market price
        current_max_profit: Current maximum profit percentage
        is_long: True for long positions, False for short
    
    Returns:
        Updated max profit percentage
    """
    current_profit = calculate_profit_pct(entry_price, current_price, is_long)
    return max(current_max_profit, current_profit)
