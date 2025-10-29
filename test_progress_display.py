"""
Test Script for Progress Display

Demonstrates the new Rich progress bars in action.
"""

import asyncio
import time
from backend.utils.progress_display import PipelineProgress
from backend.utils.log_config import configure_pipeline_logging


async def simulate_phase1(progress: PipelineProgress):
    """Simulate Phase 1: Fetch"""
    phase_name = "Phase 1: Fetch Data"
    progress.start_phase(phase_name, total_items=100, description="[bold blue]Phase 1:[/] Fetch Data")
    
    # Simulate fetching Reddit
    reddit_task = progress.add_sub_task(phase_name, "Reddit", total=30)
    for i in range(30):
        await asyncio.sleep(0.05)
        progress.update_sub_task(phase_name, "Reddit", advance=1, status=f"Post {i+1}")
    progress.complete_sub_task(phase_name, "Reddit", "✓ 30 posts")
    
    # Simulate fetching News
    news_task = progress.add_sub_task(phase_name, "News API", total=40)
    for i in range(40):
        await asyncio.sleep(0.03)
        progress.update_sub_task(phase_name, "News API", advance=1, status=f"Article {i+1}")
    progress.complete_sub_task(phase_name, "News API", "✓ 40 articles")
    
    # Simulate fetching YFinance
    yf_task = progress.add_sub_task(phase_name, "YFinance", total=30)
    for i in range(30):
        await asyncio.sleep(0.04)
        ticker = f"TICKER{i+1}"
        progress.update_sub_task(phase_name, "YFinance", advance=1, status=ticker)
    progress.complete_sub_task(phase_name, "YFinance", "✓ 10 tickers")
    
    progress.complete_phase(phase_name)


async def simulate_phase2(progress: PipelineProgress):
    """Simulate Phase 2: Calculate"""
    phase_name = "Phase 2: Calculate"
    progress.start_phase(phase_name, total_items=10, description="[bold green]Phase 2:[/] Calculate Factors")
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    for i, ticker in enumerate(tickers):
        await asyncio.sleep(0.2)
        progress.update_phase(phase_name, advance=1, status=f"Calculating {ticker}...")
    
    progress.complete_phase(phase_name, "10 tickers calculated")


async def simulate_phase3(progress: PipelineProgress):
    """Simulate Phase 3: Normalize"""
    phase_name = "Phase 3: Normalize"
    progress.start_phase(phase_name, total_items=3, description="[bold yellow]Phase 3:[/] Normalize Scores")
    
    steps = ["Winsorize outliers", "Z-score transform", "Cap extremes"]
    
    for step in steps:
        await asyncio.sleep(0.3)
        progress.update_phase(phase_name, advance=1, status=step)
    
    progress.complete_phase(phase_name)


async def simulate_phase4(progress: PipelineProgress):
    """Simulate Phase 4: Score"""
    phase_name = "Phase 4: Assemble Scores"
    progress.start_phase(phase_name, total_items=10, description="[bold magenta]Phase 4:[/] Assemble Scores")
    
    for i in range(10):
        await asyncio.sleep(0.15)
        progress.update_phase(phase_name, advance=1, status=f"Ticker {i+1}/10")
    
    progress.complete_phase(phase_name, "10 signals generated")


async def simulate_phase5(progress: PipelineProgress):
    """Simulate Phase 5: Persist"""
    phase_name = "Phase 5: Persist"
    progress.start_phase(phase_name, total_items=4, description="[bold cyan]Phase 5:[/] Save to Database")
    
    steps = [
        "Insert run metadata",
        "Bulk insert signals",
        "Update factor metrics",
        "Save to JSON"
    ]
    
    for step in steps:
        await asyncio.sleep(0.4)
        progress.update_phase(phase_name, advance=1, status=step)
    
    progress.complete_phase(phase_name)


async def main():
    """Test the progress display."""
    # Configure logging (quiet mode for clean test)
    logger = configure_pipeline_logging(verbose=0, quiet=False)
    
    print("Testing Rich Progress Display...")
    print("=" * 80)
    print()
    
    with PipelineProgress(verbose=False) as progress:
        progress.show_header()
        
        # Simulate all phases
        await simulate_phase1(progress)
        await simulate_phase2(progress)
        await simulate_phase3(progress)
        await simulate_phase4(progress)
        await simulate_phase5(progress)
        
        # Show summary
        progress.show_summary({
            "total_duration": 15.2,
            "tickers_processed": 10,
            "signals_generated": 10,
            "success_rate": 0.944
        })
    
    print()
    print("=" * 80)
    print("✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
