"""
Rich-based Progress Display for VP Investments Pipeline

Provides clean, visual progress tracking with:
- Live-updating progress bars
- Phase tracking with timing
- Hierarchical display (phases -> sub-tasks)
- ETA calculations
- Color-coded status indicators
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskID
)
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


class PipelineProgress:
    """
    Manages progress display for the entire pipeline.
    
    Features:
    - Live-updating progress bars for each phase
    - Hierarchical task tracking (phase -> sub-tasks)
    - Automatic ETA calculation
    - Clean visual formatting with colors
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize pipeline progress display.
        
        Args:
            verbose: If True, show detailed per-item progress
        """
        self.verbose = verbose
        self.start_time = datetime.now()
        self.phase_times: Dict[str, float] = {}
        
        # Create progress bar with custom columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="left"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[status]}", justify="left"),
            TimeElapsedColumn(),
            console=console,
            transient=False,  # Keep completed bars visible
            expand=False
        )
        
        self.tasks: Dict[str, TaskID] = {}
        self.current_phase: Optional[str] = None
        
    def show_header(self):
        """Display pipeline header with title."""
        console.print(Panel(
            "[bold white]VP Investments Pipeline v3.2[/]",
            style="bold white on blue",
            box=box.ROUNDED
        ))
        console.print()
        
    def start_phase(self, phase_name: str, total_items: int = 100, description: str = None) -> TaskID:
        """
        Start a new phase with progress tracking.
        
        Args:
            phase_name: Name of the phase (e.g., "Phase 1: Fetch")
            total_items: Total number of items to process
            description: Optional detailed description
            
        Returns:
            Task ID for updating progress
        """
        self.current_phase = phase_name
        self.phase_times[phase_name] = datetime.now().timestamp()
        
        display_name = description or phase_name
        
        task_id = self.progress.add_task(
            display_name,
            total=total_items,
            status="Initializing..."
        )
        
        self.tasks[phase_name] = task_id
        return task_id
        
    def update_phase(self, phase_name: str, advance: int = 1, status: str = None, **kwargs):
        """
        Update progress for a phase.
        
        Args:
            phase_name: Name of the phase to update
            advance: Number of items completed (default: 1)
            status: Status message to display
            **kwargs: Additional fields to update
        """
        task_id = self.tasks.get(phase_name)
        if task_id is not None:
            update_kwargs = {"advance": advance}
            if status:
                update_kwargs["status"] = status
            update_kwargs.update(kwargs)
            self.progress.update(task_id, **update_kwargs)
            
    def complete_phase(self, phase_name: str, summary: str = None):
        """
        Mark a phase as complete.
        
        Args:
            phase_name: Name of the phase
            summary: Optional summary message
        """
        task_id = self.tasks.get(phase_name)
        if task_id is not None:
            # Calculate duration
            duration = datetime.now().timestamp() - self.phase_times.get(phase_name, 0)
            
            status = f"✓ Complete ({duration:.1f}s)"
            if summary:
                status = f"✓ {summary}"
                
            self.progress.update(task_id, completed=True, status=status)
            
    def add_sub_task(self, parent_phase: str, name: str, total: int = 100) -> TaskID:
        """
        Add a sub-task under a phase (e.g., "Reddit", "YFinance" under "Phase 1").
        
        Args:
            parent_phase: Parent phase name
            name: Sub-task name
            total: Total items for this sub-task
            
        Returns:
            Task ID for the sub-task
        """
        task_id = self.progress.add_task(
            f"  ├─ {name}",
            total=total,
            status="Pending..."
        )
        
        # Store with composite key
        key = f"{parent_phase}:{name}"
        self.tasks[key] = task_id
        
        return task_id
        
    def update_sub_task(self, parent_phase: str, name: str, advance: int = 1, status: str = None):
        """Update a sub-task."""
        key = f"{parent_phase}:{name}"
        task_id = self.tasks.get(key)
        if task_id is not None:
            update_kwargs = {"advance": advance}
            if status:
                update_kwargs["status"] = status
            self.progress.update(task_id, **update_kwargs)
            
    def complete_sub_task(self, parent_phase: str, name: str, summary: str = None):
        """Complete a sub-task."""
        key = f"{parent_phase}:{name}"
        task_id = self.tasks.get(key)
        if task_id is not None:
            status = summary or "✓ Complete"
            self.progress.update(task_id, completed=True, status=status)
            
    def show_summary(self, results: Dict[str, Any]):
        """
        Display final pipeline summary.
        
        Args:
            results: Dictionary with pipeline results and statistics
        """
        console.print()
        console.print(Panel(
            "[bold green]Pipeline Complete[/]",
            style="bold green",
            box=box.ROUNDED
        ))
        
        # Create summary table
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")
        
        # Add metrics
        if "total_duration" in results:
            table.add_row("Total Duration", f"{results['total_duration']:.1f}s")
        if "tickers_processed" in results:
            table.add_row("Tickers Processed", str(results["tickers_processed"]))
        if "signals_generated" in results:
            table.add_row("Signals Generated", str(results["signals_generated"]))
        if "success_rate" in results:
            table.add_row("Success Rate", f"{results['success_rate']:.1%}")
            
        console.print(table)
        console.print()
        
    def __enter__(self):
        """Context manager entry."""
        self.progress.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.progress.stop()
        
        # Show error if exception occurred
        if exc_type is not None:
            console.print()
            console.print(Panel(
                f"[bold red]Pipeline Failed: {exc_val}[/]",
                style="bold red",
                box=box.ROUNDED
            ))
        
        return False  # Don't suppress exceptions
