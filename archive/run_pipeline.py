"""
VP Investments Pipeline Runner
==============================

Simple script to run the full pipeline with auto-discovery mode.

Usage:
    python run_pipeline.py
"""

import asyncio
import sys
from backend.pipeline import run_pipeline

if __name__ == "__main__":
    print("Starting VP Investments Pipeline v3.1...")
    print("Mode: Auto-Discovery (Reddit + News)")
    print("-" * 500)
    
    # Run pipeline without tickers (auto-discovery mode)
    asyncio.run(run_pipeline(tickers=None))
    
    print("\n" + "=" * 500)
    print("Pipeline execution complete!")
    print("Check results/ folder for JSON export")
    print("Check logs/ folder for factor monitoring report")
    print("=" * 500)
