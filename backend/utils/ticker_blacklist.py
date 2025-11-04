"""Ticker blacklist management utilities."""

from pathlib import Path
from typing import Set

_blacklist_cache: Set[str] = set()
_blacklist_loaded = False

def load_blacklist() -> Set[str]:
    """
    Load ticker blacklist from config/ticker_blacklist.txt.
    
    Returns:
        Set of blacklisted ticker symbols (uppercase)
    """
    global _blacklist_cache, _blacklist_loaded
    
    if _blacklist_loaded:
        return _blacklist_cache
    
    # Try multiple path resolutions for flexibility
    paths_to_try = [
        Path(__file__).parent.parent.parent / 'config' / 'ticker_blacklist.txt',  # From backend/utils/
        Path(__file__).parent.parent / 'config' / 'ticker_blacklist.txt',  # From backend/
        Path.cwd() / 'config' / 'ticker_blacklist.txt'  # From project root
    ]
    
    blacklist_file = None
    for path in paths_to_try:
        if path.exists():
            blacklist_file = path
            break
    
    if blacklist_file is None:
        _blacklist_loaded = True
        return _blacklist_cache
    
    with open(blacklist_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Extract ticker (before # comment if present)
            ticker = line.split('#')[0].strip().upper()
            
            if ticker:
                _blacklist_cache.add(ticker)
    
    _blacklist_loaded = True
    return _blacklist_cache

def is_blacklisted(ticker: str) -> bool:
    """
    Check if a ticker is blacklisted.
    
    Args:
        ticker: Ticker symbol to check
        
    Returns:
        True if ticker is blacklisted, False otherwise
    """
    blacklist = load_blacklist()
    return ticker.upper() in blacklist

def filter_blacklisted(tickers: list[str]) -> list[str]:
    """
    Filter out blacklisted tickers from a list.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        List with blacklisted tickers removed
    """
    blacklist = load_blacklist()
    return [t for t in tickers if t.upper() not in blacklist]

def get_blacklist_count() -> int:
    """Get the number of blacklisted tickers."""
    return len(load_blacklist())
