"""Apply parsed HumanCommand lists to a list of subtopics (pure transformation, no LLM)."""

from __future__ import annotations

from research_assistant.core.state import (
    AddCommand,
    ApproveCommand,
    HumanCommand,
    ModifyCommand,
    RejectCommand,
    Subtopic,
    SubtopicStatus,
)


def apply_commands(
    subtopics: list[Subtopic],
    commands: list[HumanCommand],
) -> list[Subtopic]:
    """Apply a list of HumanCommand to the original subtopic list.

    Returns a NEW list (does not mutate the input). The semantics are:
    - APPROVE indices: those subtopics get status=APPROVED.
    - REJECT indices: those subtopics get status=REJECTED.
    - MODIFY index to '<title>': that subtopic gets status=MODIFIED, original_title
      preserved (only set if still unset), title updated to new_title.
    - ADD: a new Subtopic is appended with status=APPROVED, user-generated
      description, and empty sources. ``id`` comes from the Subtopic default.

    Indices in commands are 1-based and refer to the ORIGINAL subtopic list
    (``subtopics`` as passed in). Unmentioned subtopics keep status=PENDING.
    When several commands target the same index, the last one wins.

    The applier does not filter; all subtopics (including PENDING/REJECTED) are
    returned. Downstream curation is responsible for filtering.
    """
    result = [s.model_copy() for s in subtopics]

    for cmd in commands:
        if isinstance(cmd, ApproveCommand):
            for i in cmd.subtopic_indices:
                idx = i - 1
                st = result[idx]
                st.status = SubtopicStatus.APPROVED
        elif isinstance(cmd, RejectCommand):
            for i in cmd.subtopic_indices:
                idx = i - 1
                st = result[idx]
                st.status = SubtopicStatus.REJECTED
        elif isinstance(cmd, ModifyCommand):
            idx = cmd.subtopic_index - 1
            st = result[idx]
            if st.original_title is None:
                st.original_title = st.title
            st.title = cmd.new_title
            st.status = SubtopicStatus.MODIFIED
        elif isinstance(cmd, AddCommand):
            t = cmd.new_title
            result.append(
                Subtopic(
                    title=t,
                    description=f"User-added subtopic: {t}",
                    sources=[],
                    status=SubtopicStatus.APPROVED,
                ),
            )
    return result
