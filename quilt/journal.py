"""The journal: an append-only log of conversation turns, replicated to all.

The leader is the only writer. Each entry has a global sequence number, so a
member can tell exactly how far behind it is, and a newly elected leader
knows it holds the latest state any surviving member has. Sessions are views
over the journal — the entries that share a ``session`` id, in order.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Entry:
    seq: int
    session: str
    role: str  # "user" | "assistant" | "system"
    text: str
    tokens: list[int]
    ts: float = field(default_factory=time.time)
    term: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Entry:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class Journal:
    def __init__(self, path: Path | None = None):
        self.entries: list[Entry] = []
        self.path = path
        if path and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.entries.append(Entry.from_dict(json.loads(line)))

    @property
    def seq(self) -> int:
        return self.entries[-1].seq if self.entries else 0

    def append(self, session: str, role: str, text: str, tokens: list[int], term: int) -> Entry:
        e = Entry(self.seq + 1, session, role, text, list(tokens), term=term)
        self._store(e)
        return e

    def accept(self, e: Entry) -> bool:
        """Apply a replicated entry. Returns False if it is not the next one."""
        if e.seq <= self.seq:
            return True  # duplicate; already have it
        if e.seq != self.seq + 1:
            return False
        self._store(e)
        return True

    def replace(self, entries: list[Entry]) -> None:
        self.entries = sorted(entries, key=lambda e: e.seq)
        if self.path:
            self.path.write_text("".join(json.dumps(e.to_dict()) + "\n" for e in self.entries))

    def _store(self, e: Entry) -> None:
        self.entries.append(e)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(e.to_dict()) + "\n")

    def session(self, session_id: str) -> list[Entry]:
        return [e for e in self.entries if e.session == session_id]

    def session_tokens(self, session_id: str) -> list[int]:
        out: list[int] = []
        for e in self.session(session_id):
            out.extend(e.tokens)
        return out

    def sessions(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.entries:
            seen.setdefault(e.session, None)
        return list(seen)

    def since(self, seq: int) -> list[Entry]:
        return [e for e in self.entries if e.seq > seq]
