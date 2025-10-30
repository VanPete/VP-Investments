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
            
            # Get the task and set completed to total to reach 100%
            task = self.progress.tasks[task_id]
            self.progress.update(task_id, completed=task.total, status=status)
            
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


def show_error_summary(error_buffer, phase1_results: Optional[Dict] = None) -> None:
    """
    Display consolidated error summary at end of pipeline run.
    
    Args:
        error_buffer: ErrorBuffer instance with captured errors
        phase1_results: Optional Phase 1 results for additional context
    """
    if not error_buffer.has_errors():
        return
    
    from rich.text import Text
    
    error_count = error_buffer.count()
    errors_by_phase = error_buffer.group_by_phase()
    
    # Build error summary content
    content_lines = []
    content_lines.append(f"[bold]{error_count} error{'s' if error_count != 1 else ''} occurred during pipeline execution:[/]\n")
    
    # Group and display by phase
    phase_names = {
        "phase1": "Phase 1: Fetch Data",
        "phase2": "Phase 2: Calculate Factors",
        "phase3": "Phase 3: Normalize Scores",
        "phase4": "Phase 4: Assemble Scores",
        "phase5": "Phase 5: Database Persistence",
        "phase6": "Phase 6: Performance Tracking",
        "phase7": "Phase 7: Analytics",
        "unknown": "Unknown Phase"
    }
    
    for idx, (phase, errors) in enumerate(sorted(errors_by_phase.items()), 1):
        phase_name = phase_names.get(phase, phase)
        content_lines.append(f"[bold cyan]{phase_name}[/] ({len(errors)} error{'s' if len(errors) != 1 else ''})\n")
        
        # Group errors by ticker
        ticker_errors = {}
        general_errors = []
        
        for error in errors:
            if error.ticker:
                if error.ticker not in ticker_errors:
                    ticker_errors[error.ticker] = []
                ticker_errors[error.ticker].append(error)
            else:
                general_errors.append(error)
        
        # Show ticker-specific errors
        if ticker_errors:
            for ticker, t_errors in sorted(ticker_errors.items())[:10]:  # Limit to 10 tickers
                content_lines.append(f"  [yellow]{ticker}[/]:")
                for err in t_errors[:3]:  # Limit to 3 errors per ticker
                    # Add helpful context based on error message
                    tip = _get_error_tip(err.message, phase)
                    content_lines.append(f"    └─ {err.message}")
                    if tip:
                        content_lines.append(f"    └─ [dim]Tip: {tip}[/dim]")
                if len(t_errors) > 3:
                    content_lines.append(f"    └─ [dim]... and {len(t_errors) - 3} more[/dim]")
            
            if len(ticker_errors) > 10:
                content_lines.append(f"  [dim]... and {len(ticker_errors) - 10} more tickers with errors[/dim]")
        
        # Show general errors
        for err in general_errors[:5]:  # Limit to 5 general errors per phase
            tip = _get_error_tip(err.message, phase)
            content_lines.append(f"  • {err.message}")
            if tip:
                content_lines.append(f"    └─ [dim]Tip: {tip}[/dim]")
        
        if len(general_errors) > 5:
            content_lines.append(f"  [dim]... and {len(general_errors) - 5} more errors[/dim]")
        
        content_lines.append("")  # Blank line between phases
    
    # Add reference to detailed logs
    content_lines.append("[dim]See logs/ directory for complete error details[/dim]")
    
    # Display in Rich panel
    console.print()
    console.print(Panel(
        "\n".join(content_lines),
        title="[bold red]⚠ Errors Encountered[/]",
        border_style="red",
        box=box.ROUNDED,
        padding=(1, 2)
    ))
    
    # Display factor monitoring summary if available
    _show_factor_monitoring_summary()
    
    console.print()


def _show_factor_monitoring_summary():
    """Display comprehensive factor monitoring summary from the latest monitoring JSON file."""
    import json
    from pathlib import Path
    from rich.table import Table
    
    try:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return
        
        # Find most recent factor monitoring file
        monitoring_files = sorted(logs_dir.glob("factor_monitoring_*.json"), reverse=True)
        if not monitoring_files:
            return
        
        with open(monitoring_files[0], 'r') as f:
            data = json.load(f)
        
        # Extract key metrics
        total_factors = data.get('total_factors', 0)
        total_calculations = data.get('total_calculations', 0)
        overall_success_rate = data.get('overall_success_rate', 0) * 100
        duration = data.get('duration_seconds', 0)
        
        # Build header content
        content_lines = []
        content_lines.append(f"[bold]Overall Metrics[/]")
        content_lines.append(f"Total Factors: {total_factors}")
        content_lines.append(f"Total Calculations: {total_calculations:,}")
        content_lines.append(f"Overall Success Rate: [green]{overall_success_rate:.1f}%[/green]")
        content_lines.append(f"Duration: {duration:.1f}s\n")
        
        # Group summary table
        group_summary = data.get('group_summary', {})
        if group_summary:
            content_lines.append("[bold]Success Rate by Group:[/]")
            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("Group", style="cyan")
            table.add_column("Factors", justify="right")
            table.add_column("Success Rate", justify="right")
            table.add_column("Issues", justify="right")
            
            group_names = {
                'technical': 'Technical',
                'fundamental': 'Fundamental',
                'news_macro': 'News/Macro',
                'social_alternative': 'Social',
                'risk_stability': 'Risk',
                'institutional_smart_money': 'Institutional'
            }
            
            for key, name in group_names.items():
                if key in group_summary:
                    group_data = group_summary[key]
                    factors = group_data.get('total_factors', 0)
                    success = group_data.get('avg_success_rate', 0) * 100
                    problematic = group_data.get('problematic_count', 0)
                    
                    # Color code success rate
                    if success >= 90:
                        success_str = f"[green]{success:.1f}%[/green]"
                    elif success >= 70:
                        success_str = f"[yellow]{success:.1f}%[/yellow]"
                    else:
                        success_str = f"[red]{success:.1f}%[/red]"
                    
                    issue_str = f"[red]{problematic}[/red]" if problematic > 0 else "[green]0[/green]"
                    table.add_row(name, str(factors), success_str, issue_str)
            
            from io import StringIO
            table_output = StringIO()
            temp_console = Console(file=table_output, force_terminal=True, width=70)
            temp_console.print(table)
            content_lines.append(table_output.getvalue())
        
        # Show problematic factors
        problematic = data.get('problematic_factors', [])
        if problematic:
            content_lines.append(f"\n[yellow]⚠ Problematic Factors (<70% success rate):[/]")
            for idx, factor in enumerate(problematic[:10], 1):
                factor_name = factor['factor']
                success_rate = factor['success_rate'] * 100
                attempts = factor['attempts']
                
                # Get top error type
                top_errors = factor.get('top_errors', {})
                error_type = list(top_errors.keys())[0] if top_errors else 'unknown'
                error_count = list(top_errors.values())[0] if top_errors else 0
                
                content_lines.append(
                    f"  {idx:2}. [cyan]{factor_name}[/]: "
                    f"[red]{success_rate:.0f}%[/red] success "
                    f"({error_count}/{attempts} {error_type})"
                )
            
            if len(problematic) > 10:
                content_lines.append(f"  [dim]... and {len(problematic) - 10} more problematic factors[/dim]")
        
        content_lines.append(f"\n[dim]Full report: logs/{monitoring_files[0].name}[/dim]")
        
        # Display panel
        console.print(Panel(
            "\n".join(content_lines),
            title="[bold blue]📊 Factor Quality Monitoring[/]",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2)
        ))
        
    except Exception as e:
        # Log error for debugging but don't crash
        console.print(f"[dim]Could not load factor monitoring: {e}[/dim]")


def _get_error_tip(message: str, phase: str) -> Optional[str]:
    """
    Provide helpful troubleshooting tips based on error message patterns.
    
    Args:
        message: Error message text
        phase: Pipeline phase where error occurred
        
    Returns:
        Helpful tip string or None
    """
    message_lower = message.lower()
    
    # Reddit/subreddit errors
    if "redirect" in message_lower or "subreddit" in message_lower:
        return "Subreddit may be private, banned, or doesn't exist"
    
    # YFinance/data fetch errors
    if "missing price history" in message_lower:
        return "Check if ticker is valid and actively traded (may be delisted)"
    
    if "timeout" in message_lower:
        return "Yahoo Finance may be rate limiting - try reducing concurrent requests"
    
    if "404" in message_lower or "not found" in message_lower:
        return "Ticker may be invalid or recently delisted"
    
    if "delisted" in message_lower:
        return "Ticker has been delisted - remove from analysis"
    
    # Database errors
    if phase == "phase5" and ("connection" in message_lower or "database" in message_lower):
        return "Check database credentials and network connection"
    
    # Calculation errors
    if phase == "phase2" and ("division" in message_lower or "nan" in message_lower):
        return "Missing or invalid data for calculation - check factor coverage"
    
    return None
