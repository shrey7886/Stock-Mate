"""
GoalStore: per-user in-memory investment goal tracker.
Users can set a target corpus amount and timeline; the chatbot
will calculate SIP needs and track progress.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class InvestmentGoal:
    target_amount: float          # ₹ target corpus
    current_amount: float         # ₹ current portfolio value
    years: float                  # time horizon in years
    label: str = "Financial Goal" # user-defined label
    expected_return_pct: float = 12.0  # assumed annualized return %


class GoalStore:
    """Thread-safe in-memory goal store (one active goal per user for now)."""

    def __init__(self) -> None:
        self._store: Dict[str, InvestmentGoal] = {}
        self._lock = threading.Lock()

    def set_goal(
        self,
        user_id: str,
        *,
        target_amount: float,
        current_amount: float,
        years: float,
        label: str = "Financial Goal",
        expected_return_pct: float = 12.0,
    ) -> InvestmentGoal:
        goal = InvestmentGoal(
            target_amount=target_amount,
            current_amount=current_amount,
            years=years,
            label=label,
            expected_return_pct=expected_return_pct,
        )
        with self._lock:
            self._store[user_id] = goal
        return goal

    def get_goal(self, user_id: str) -> Optional[InvestmentGoal]:
        with self._lock:
            return self._store.get(user_id)

    def clear_goal(self, user_id: str) -> None:
        with self._lock:
            self._store.pop(user_id, None)

    def update_current_amount(self, user_id: str, current_amount: float) -> None:
        with self._lock:
            if user_id in self._store:
                self._store[user_id].current_amount = current_amount


# Module-level singleton
goal_store = GoalStore()
