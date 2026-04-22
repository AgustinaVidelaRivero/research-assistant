"""Parse human HITL command strings into structured HumanCommand models (no LLM)."""

from __future__ import annotations

import re
from typing import Final, cast

from research_assistant.core.state import (
    AddCommand,
    ApproveCommand,
    HumanCommand,
    ModifyCommand,
    RejectCommand,
)

_COMMAND_LOOKAHEAD: Final[re.Pattern[str]] = re.compile(
    r"^\s*(approve|reject|add|modify)\b",
    re.IGNORECASE,
)

_APPROVE_ALL: Final[re.Pattern[str]] = re.compile(r"^approve\s+all$", re.IGNORECASE)
_APPROVE: Final[re.Pattern[str]] = re.compile(r"^approve\s+(.+)$", re.IGNORECASE)
_REJECT: Final[re.Pattern[str]] = re.compile(r"^reject\s+(.+)$", re.IGNORECASE)
_ADD: Final[re.Pattern[str]] = re.compile(
    r"^add\s+(['\"])(.*?)\1$",
    re.IGNORECASE,
)
_MODIFY: Final[re.Pattern[str]] = re.compile(
    r"^modify\s+(\d+)\s+to\s+(['\"])(.*?)\2$",
    re.IGNORECASE,
)


class CommandParseError(ValueError):
    """Raised when a human input string cannot be parsed into HumanCommand objects."""


def _split_command_chunks(raw: str) -> list[str]:
    """Split on top-level commas only when the next segment starts a new command."""
    if not raw:
        return []
    result: list[str] = []
    current: list[str] = []
    in_quote: str | None = None
    n = len(raw)
    i = 0
    while i < n:
        c = raw[i]
        if in_quote is not None:
            current.append(c)
            if c == in_quote:
                in_quote = None
            i += 1
            continue
        if c in ("'", '"'):
            in_quote = c
            current.append(c)
            i += 1
            continue
        if c == ",":
            rest = raw[i + 1 :]
            if _COMMAND_LOOKAHEAD.match(rest) is not None:
                seg = "".join(current).strip()
                if seg:
                    result.append(seg)
                current = []
                i += 1
                continue
        current.append(c)
        i += 1
    tail = "".join(current).strip()
    if tail:
        result.append(tail)
    return result


def _parse_index_group(indices_part: str, *, chunk: str) -> list[int]:
    out: list[int] = []
    for part in indices_part.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except ValueError as e:
            msg = f"Invalid index '{s}' in '{chunk}'"
            raise CommandParseError(msg) from e
    if not out:
        msg = f"Invalid index '' in '{chunk}'"
        raise CommandParseError(msg)
    return out


def _validate_indices(indices: list[int], *, total_subtopics: int) -> list[int]:
    for i in indices:
        if not 1 <= i <= total_subtopics:
            msg = f"Index {i} out of range; valid range is 1..{total_subtopics}"
            raise CommandParseError(msg)
    return indices


def parse_human_input(raw: str, *, total_subtopics: int) -> list[HumanCommand]:
    """Parse a raw human input string into a list of HumanCommand objects.

    Supported syntax (commands can be combined; indices are 1-based):
        approve <indices>          e.g. "approve 1,3" or "approve 1, 2, 3"
        approve all                shortcut for approving every subtopic
        reject <indices>           e.g. "reject 2"
        add '<title>'              e.g. "add 'AI safety concerns'"
        modify <index> to '<title>'  e.g. "modify 1 to 'AI ethics'"

    Multiple commands: "reject 2, add 'AI safety'" (comma only starts a new command
    when the following text begins with approve | reject | add | modify).

    Args:
        raw: the raw user input string.
        total_subtopics: subtopic count; used to validate index range and to expand
            "approve all".

    Returns:
        HumanCommand instances in the order they appear.

    Raises:
        CommandParseError: if the input is malformed, invalid, or out of range.
    """
    s = raw.strip()
    if not s:
        raise CommandParseError("Empty input")

    out: list[HumanCommand] = []
    for chunk in _split_command_chunks(s):
        c = chunk.strip()
        m_all = _APPROVE_ALL.match(c)
        if m_all is not None:
            out.append(
                ApproveCommand(
                    subtopic_indices=list(range(1, total_subtopics + 1)),
                )
            )
            continue

        m_app = _APPROVE.match(c)
        if m_app is not None:
            idx_text = m_app.group(1).strip()
            indices = _parse_index_group(idx_text, chunk=c)
            _validate_indices(indices, total_subtopics=total_subtopics)
            out.append(ApproveCommand(subtopic_indices=indices))
            continue

        m_rej = _REJECT.match(c)
        if m_rej is not None:
            idx_text = m_rej.group(1).strip()
            indices = _parse_index_group(idx_text, chunk=c)
            _validate_indices(indices, total_subtopics=total_subtopics)
            out.append(RejectCommand(subtopic_indices=indices))
            continue

        m_add = _ADD.match(c)
        if m_add is not None:
            title = m_add.group(2).strip()
            if not title:
                raise CommandParseError("Title cannot be empty")
            out.append(AddCommand(new_title=title))
            continue

        m_mod = _MODIFY.match(c)
        if m_mod is not None:
            sub_idx = int(m_mod.group(1))
            new_title = m_mod.group(3).strip()
            if not 1 <= sub_idx <= total_subtopics:
                msg = f"Index {sub_idx} out of range; valid range is 1..{total_subtopics}"
                raise CommandParseError(msg)
            if not new_title:
                raise CommandParseError("Title cannot be empty")
            out.append(ModifyCommand(subtopic_index=sub_idx, new_title=new_title))
            continue

        msg = f"Unknown command: '{c}'. Valid commands: approve, reject, add, modify."
        raise CommandParseError(msg)
    if not out:
        raise CommandParseError("Empty input")
    return cast(list[HumanCommand], out)
