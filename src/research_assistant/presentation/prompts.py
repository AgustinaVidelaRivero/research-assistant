"""Interactive prompts for the CLI: command input with re-prompt loop."""

from __future__ import annotations

from rich.console import Console

from research_assistant.human_input.parser import (
    CommandParseError,
    parse_human_input,
)
from research_assistant.presentation.display import (
    show_command_help,
    show_error,
)


def prompt_for_commands(
    console: Console,
    *,
    total_subtopics: int,
    max_attempts: int = 5,
) -> str:
    """Prompt the user for a command string, re-prompting on parse errors.

    The user can also type 'help' to see usage examples (does not count as an attempt).

    Args:
        console: Rich console for output.
        total_subtopics: number of subtopics, used to validate indices.
        max_attempts: max parse attempts before giving up.

    Returns:
        The raw (validated) command string. The string is guaranteed to be parseable
        according to parse_human_input(); the actual parsing happens again inside
        the human_review_node, but here we validate early to give friendlier errors.

    Raises:
        RuntimeError: if max_attempts is reached without valid input.
        KeyboardInterrupt: re-raised if the user hits Ctrl+C.
    """
    for _attempt in range(1, max_attempts + 1):
        try:
            console.print(
                "[bold cyan]Your command(s):[/bold cyan] ",
                end="",
            )
            raw = input().strip()
        except EOFError as e:
            raise RuntimeError("No input available (stdin closed)") from e

        if raw.lower() in {"help", "?", "h"}:
            show_command_help(console)
            continue

        if not raw:
            show_error(console, "Empty input. Type 'help' for examples.")
            continue

        try:
            parse_human_input(raw, total_subtopics=total_subtopics)
        except CommandParseError as e:
            show_error(console, f"{e}")
            console.print("[dim]Try again, or type 'help' for examples.[/dim]")
            continue

        return raw

    raise RuntimeError(
        f"No valid command after {max_attempts} attempts. Aborting.",
    )
