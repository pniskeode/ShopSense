"""Bounded working memory with a tested rolling-summary policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


RECENT_KEEP = 4
TOKEN_BUDGET = 220


@dataclass(frozen=True)
class Turn:
    """One customer or assistant message retained in the active session."""

    role: str
    content: str


def token_count(text: str) -> int:
    """Count tokens when available, with a deterministic offline fallback."""

    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class RollingConversation:
    """Keep recent turns verbatim and merge older turns into a short summary."""

    recent_keep: int = RECENT_KEEP
    token_budget: int = TOKEN_BUDGET
    summary: str = ""
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        content = content.strip()
        if content:
            self.turns.append(Turn(role=role, content=content))

    def total_tokens(self) -> int:
        summary_tokens = token_count(self.summary) if self.summary else 0
        return sum(token_count(turn.content) for turn in self.turns) + summary_tokens

    def compact(self, summarize: Callable[[list[Turn], str], str]) -> bool:
        """Summarize only when older turns exist beyond the configured budget."""

        if self.total_tokens() <= self.token_budget or len(self.turns) <= self.recent_keep:
            return False

        old_turns = self.turns[: -self.recent_keep]
        updated_summary = summarize(old_turns, self.summary).strip()
        if not updated_summary:
            raise RuntimeError("Rolling-summary model returned an empty summary")

        self.summary = updated_summary
        self.turns = self.turns[-self.recent_keep :]
        return True

    def format_context(self) -> str:
        """Render context without presenting historical text as model instructions."""

        parts: list[str] = []
        if self.summary:
            parts.append(f"Running summary: {self.summary}")
        if self.turns:
            rendered = "\n".join(f"{turn.role}: {turn.content}" for turn in self.turns)
            parts.append(f"Recent session turns:\n{rendered}")
        return "\n\n".join(parts)
