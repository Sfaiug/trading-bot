"""
Configuration settings for the trading bot.
Load from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Binance API Credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Trading Settings
SYMBOL = "SOLUSDT"  # Trading pair
POSITION_SIZE = float(os.getenv("POSITION_SIZE", "1"))  # Minimum 1 SOL on testnet

# Trailing Stop Settings (can be overridden via command line)
DEFAULT_TRAILING_STOP_PERCENT = 1.0  # Default 1%

# Testnet Configuration
TESTNET = True  # Always use testnet for this bot
TESTNET_BASE_URL = "https://testnet.binancefuture.com"

# Polling interval
PRICE_CHECK_INTERVAL = 1  # seconds

# Logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
TRADE_LOG_FILE = os.path.join(LOG_DIR, "trades.csv")
