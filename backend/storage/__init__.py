"""Storage package - Database interfaces and repositories"""

from .database import DatabaseInterface, get_database

__all__ = ["DatabaseInterface", "get_database"]