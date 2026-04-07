"""
TTL-based in-memory session store.
Keyed by user_id. Each session holds a list of {role, content} message dicts.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str    # "user" | "assistant" | "system"
    content: str


class _Session:
    def __init__(self, ttl_seconds: int, max_history: int) -> None:
        self.history: list[ChatMessage] = []
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.ttl_seconds = ttl_seconds
        self.max_history = max_history

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > self.ttl_seconds

    def touch(self) -> None:
        self.last_active = time.time()

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        # Keep within max_history (pairs of user+assistant = 1 turn)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]
        self.touch()


class SessionStore:
    def __init__(self, ttl_minutes: int = 60, max_history: int = 20) -> None:
        self._store: dict[str, _Session] = {}
        self._lock = Lock()
        self.ttl_seconds = ttl_minutes * 60
        self.max_history = max_history

    def _get_or_create(self, user_id: str) -> _Session:
        with self._lock:
            session = self._store.get(user_id)
            if session is None or session.is_expired:
                session = _Session(
                    ttl_seconds=self.ttl_seconds,
                    max_history=self.max_history,
                )
                self._store[user_id] = session
            return session

    def get_history(self, user_id: str) -> list[ChatMessage]:
        return list(self._get_or_create(user_id).history)

    def add_turn(self, user_id: str, role: str, content: str) -> None:
        self._get_or_create(user_id).add(role, content)

    def clear(self, user_id: str) -> None:
        with self._lock:
            self._store.pop(user_id, None)

    def cleanup_expired(self) -> int:
        """Remove expired sessions; return count removed."""
        with self._lock:
            expired = [uid for uid, s in self._store.items() if s.is_expired]
            for uid in expired:
                del self._store[uid]
            return len(expired)
