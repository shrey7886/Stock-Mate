"""
WatchlistStore: per-user in-memory watchlist (survives for process lifetime).
Users can add/remove tickers they want to track even if they don't hold them.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, List, Set


class WatchlistStore:
    """Thread-safe in-memory watchlist keyed by user_id."""

    def __init__(self) -> None:
        self._store: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()

    def add(self, user_id: str, symbol: str) -> None:
        with self._lock:
            self._store[user_id].add(symbol.upper().strip())

    def remove(self, user_id: str, symbol: str) -> bool:
        with self._lock:
            sym = symbol.upper().strip()
            if sym in self._store[user_id]:
                self._store[user_id].discard(sym)
                return True
            return False

    def get(self, user_id: str) -> List[str]:
        with self._lock:
            return sorted(self._store[user_id])

    def clear(self, user_id: str) -> None:
        with self._lock:
            self._store[user_id].clear()

    def contains(self, user_id: str, symbol: str) -> bool:
        with self._lock:
            return symbol.upper().strip() in self._store[user_id]


# Module-level singleton
watchlist_store = WatchlistStore()
