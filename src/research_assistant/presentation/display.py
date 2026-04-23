"""Display helpers for rich CLI output (subtopics, reports, costs, errors)."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from research_assistant.core.state import CostSummary, FinalReport


def get_console() -> Console:
    """Return a singleton-style Console for the application."""
    return Console()


def show_banner(console: Console, topic: str) -> None:
    """Display the program banner with the research topic."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Research Assistant[/bold cyan]\n[dim]Topic:[/dim] [white]{topic}[/white]",
            border_style="cyan",
        )
    )
    console.print()


def show_subtopics_table(console: Console, subtopics: list[dict]) -> None:
    """Display the list of subtopics in a formatted table.

    Args:
        subtopics: list of dicts with 'index', 'title', 'description' keys
            (from the human_review interrupt payload).
    """
    table = Table(
        title="📋 Subtopics identified by the Investigator",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("#", style="cyan bold", width=4, justify="right")
    table.add_column("Title", style="white bold", min_width=30)
    table.add_column("Description", style="dim white", min_width=40)

    for st in subtopics:
        table.add_row(
            str(st["index"]),
            st["title"],
            st["description"],
        )

    console.print(table)
    console.print()


def show_command_help(console: Console) -> None:
    """Display help text with examples of valid commands."""
    help_text = (
        "[bold yellow]Available commands:[/bold yellow]\n\n"
        "  [cyan]approve[/cyan] [white]<indices>[/white]              "
        "e.g. [dim]approve 1,3[/dim]\n"
        "  [cyan]approve all[/cyan]                    "
        "[dim](shortcut to approve every subtopic)[/dim]\n"
        "  [cyan]reject[/cyan] [white]<indices>[/white]               "
        "e.g. [dim]reject 2[/dim]\n"
        "  [cyan]add[/cyan] [white]'<title>'[/white]                  "
        "e.g. [dim]add 'AI safety concerns'[/dim]\n"
        "  [cyan]modify[/cyan] [white]<index>[/white] [cyan]to[/cyan] "
        "[white]'<title>'[/white]   "
        "e.g. [dim]modify 1 to 'AI ethics frameworks'[/dim]\n\n"
        "[dim]You can combine commands with commas:[/dim]\n"
        "  [dim]reject 2, add 'safety', modify 3 to 'new title'[/dim]"
    )
    console.print(Panel(help_text, title="💡 Help", border_style="yellow"))
    console.print()


def show_status(console: Console, message: str, *, style: str = "cyan") -> None:
    """Print a status line with an arrow and styled message."""
    console.print(f"[{style}]→[/{style}] {message}")


def show_success(console: Console, message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✅[/green] {message}")


def show_warning(console: Console, message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow]  {message}")


def show_error(console: Console, message: str) -> None:
    """Print an error message."""
    console.print(f"[red bold]❌[/red bold] {message}")


def show_final_report(console: Console, report: FinalReport) -> None:
    """Render the final report as Markdown in the console."""
    console.print()
    console.print(
        Panel(
            "[bold green]📄 Final Report[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    console.print()
    console.print(Markdown(report.to_markdown()))
    console.print()


def show_cost_summary(console: Console, summary: CostSummary) -> None:
    """Display the cost summary in a formatted table."""
    table = Table(
        title="💰 Cost Summary",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Metric", style="cyan", min_width=24)
    table.add_column("Value", style="white", justify="right")

    table.add_row("Total LLM calls", str(summary.total_calls))
    table.add_row("Total input tokens", f"{summary.total_input_tokens:,}")
    table.add_row("Total output tokens", f"{summary.total_output_tokens:,}")
    table.add_row(
        "Total cost (USD)",
        f"[bold green]${summary.total_cost_usd:.6f}[/bold green]",
    )

    console.print()
    console.print(table)

    if summary.calls_by_agent:
        agent_table = Table(
            title="📞 Calls by agent",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
        )
        agent_table.add_column("Agent", style="cyan", min_width=15)
        agent_table.add_column("Calls", style="white", justify="right")
        for agent, count in summary.calls_by_agent.items():
            agent_table.add_row(agent, str(count))
        console.print()
        console.print(agent_table)
    console.print()


def show_cancelled(console: Console) -> None:
    """Display the message shown when the user cancels with Ctrl+C."""
    console.print()
    console.print("[yellow]👋 Cancelled by user. No charges incurred for incomplete runs.[/yellow]")
    console.print()
