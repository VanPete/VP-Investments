"""
Sector ETF Mapping Utilities

Maps GICS sectors to their corresponding Select Sector SPDR ETFs for sector momentum analysis.
"""

from typing import Dict, Optional

# GICS Sector to Select Sector SPDR ETF Mapping
SECTOR_TO_ETF: Dict[str, str] = {
    'Technology': 'XLK',                    # Technology Select Sector SPDR Fund
    'Healthcare': 'XLV',                    # Health Care Select Sector SPDR Fund
    'Financial Services': 'XLF',           # Financial Select Sector SPDR Fund
    'Financials': 'XLF',                   # Alternative name for Financial Services
    'Consumer Cyclical': 'XLY',            # Consumer Discretionary Select Sector SPDR Fund
    'Consumer Discretionary': 'XLY',       # Alternative name
    'Industrials': 'XLI',                  # Industrial Select Sector SPDR Fund
    'Communication Services': 'XLC',       # Communication Services Select Sector SPDR Fund
    'Consumer Defensive': 'XLP',           # Consumer Staples Select Sector SPDR Fund
    'Consumer Staples': 'XLP',             # Alternative name
    'Energy': 'XLE',                       # Energy Select Sector SPDR Fund
    'Basic Materials': 'XLB',              # Materials Select Sector SPDR Fund
    'Materials': 'XLB',                    # Alternative name
    'Real Estate': 'XLRE',                 # Real Estate Select Sector SPDR Fund
    'Utilities': 'XLU',                    # Utilities Select Sector SPDR Fund
}

# Fallback to S&P 500 for unknown sectors
DEFAULT_SECTOR_ETF = 'SPY'


def get_sector_etf(sector: str) -> str:
    """
    Get the ETF ticker for a given GICS sector.
    
    Args:
        sector: The sector name (e.g., 'Technology', 'Healthcare')
        
    Returns:
        ETF ticker symbol (e.g., 'XLK', 'XLV')
        Returns 'SPY' (S&P 500) as fallback for unknown sectors
        
    Examples:
        >>> get_sector_etf('Technology')
        'XLK'
        >>> get_sector_etf('Healthcare')
        'XLV'
        >>> get_sector_etf('Unknown Sector')
        'SPY'
    """
    return SECTOR_TO_ETF.get(sector, DEFAULT_SECTOR_ETF)


def get_all_sector_etfs() -> Dict[str, str]:
    """
    Get all sector to ETF mappings.
    
    Returns:
        Dictionary mapping sector names to ETF tickers
    """
    return SECTOR_TO_ETF.copy()


def get_unique_sectors() -> list[str]:
    """
    Get list of all supported sector names.
    
    Returns:
        List of sector names
    """
    # Get unique sectors (remove duplicates from alternative names)
    unique = set()
    sector_map = {
        'XLK': 'Technology',
        'XLV': 'Healthcare', 
        'XLF': 'Financial Services',
        'XLY': 'Consumer Cyclical',
        'XLI': 'Industrials',
        'XLC': 'Communication Services',
        'XLP': 'Consumer Defensive',
        'XLE': 'Energy',
        'XLB': 'Basic Materials',
        'XLRE': 'Real Estate',
        'XLU': 'Utilities'
    }
    return list(sector_map.values())


def validate_sector(sector: Optional[str]) -> bool:
    """
    Check if a sector name is recognized.
    
    Args:
        sector: Sector name to validate
        
    Returns:
        True if sector is recognized, False otherwise
    """
    if sector is None:
        return False
    return sector in SECTOR_TO_ETF
