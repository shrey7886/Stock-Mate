from __future__ import annotations

import base64
import hmac
import hashlib
import sqlite3
from pathlib import Path

from cryptography.fernet import Fernet

from backend_api.core.config import settings

DB_PATH = Path(__file__).resolve().parent / "backend.db"


def _build_fernet_key(secret: str) -> bytes:
    raw = secret.encode("utf-8")
    raw = (raw + b"0" * 32)[:32]
    return base64.urlsafe_b64encode(raw)


fernet = Fernet(_build_fernet_key(settings.broker_token_secret))


def _hash_state(state: str) -> str:
    return hmac.HMAC(
        settings.oauth_state_hmac_secret.encode("utf-8"),
        state.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        # ── Users table ────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_broker_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                account_id TEXT,
                access_token_enc TEXT NOT NULL,
                public_token_enc TEXT,
                scopes TEXT,
                is_primary INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, provider, account_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oauth_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                state_hash TEXT NOT NULL UNIQUE,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oauth_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                provider TEXT NOT NULL,
                state_hash TEXT,
                event TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Backward-compatible migration: old schema had `state` instead of `state_hash`
        cols = [r[1] for r in conn.execute("PRAGMA table_info(oauth_states)").fetchall()]
        if "state_hash" not in cols and "state" in cols:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS oauth_states_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    state_hash TEXT NOT NULL UNIQUE,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            rows = conn.execute(
                "SELECT id, user_id, provider, state, expires_at, created_at FROM oauth_states"
            ).fetchall()

            for row in rows:
                old_id, user_id, provider, state, expires_at, created_at = row
                conn.execute(
                    """
                    INSERT OR IGNORE INTO oauth_states_new (id, user_id, provider, state_hash, expires_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (old_id, user_id, provider, _hash_state(state), expires_at, created_at),
                )

            conn.execute("DROP TABLE oauth_states")
            conn.execute("ALTER TABLE oauth_states_new RENAME TO oauth_states")

        # Backward-compatible migration: old broker schema supported only one account per provider
        broker_cols = [r[1] for r in conn.execute("PRAGMA table_info(user_broker_accounts)").fetchall()]
        if "is_primary" not in broker_cols:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_broker_accounts_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    account_id TEXT,
                    access_token_enc TEXT NOT NULL,
                    public_token_enc TEXT,
                    scopes TEXT,
                    is_primary INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, provider, account_id)
                )
                """
            )
            rows = conn.execute(
                """
                SELECT id, user_id, provider, account_id, access_token_enc, public_token_enc, scopes, created_at, updated_at
                FROM user_broker_accounts
                ORDER BY id ASC
                """
            ).fetchall()

            seen_primary: set[tuple[str, str]] = set()
            for row in rows:
                (
                    old_id,
                    user_id,
                    provider,
                    account_id,
                    access_token_enc,
                    public_token_enc,
                    scopes,
                    created_at,
                    updated_at,
                ) = row
                key = (user_id, provider)
                is_primary = 0 if key in seen_primary else 1
                seen_primary.add(key)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO user_broker_accounts_new
                    (id, user_id, provider, account_id, access_token_enc, public_token_enc, scopes, is_primary, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        old_id,
                        user_id,
                        provider,
                        account_id,
                        access_token_enc,
                        public_token_enc,
                        scopes,
                        is_primary,
                        created_at,
                        updated_at,
                    ),
                )

            conn.execute("DROP TABLE user_broker_accounts")
            conn.execute("ALTER TABLE user_broker_accounts_new RENAME TO user_broker_accounts")

        conn.execute(
            """
            UPDATE user_broker_accounts
            SET is_primary = 1
            WHERE id IN (
                SELECT MIN(id)
                FROM user_broker_accounts
                GROUP BY user_id, provider
            )
            """
        )

        conn.commit()


def upsert_broker_tokens(
    *,
    user_id: str,
    provider: str,
    account_id: str | None,
    access_token: str,
    public_token: str | None,
    scopes: str | None = None,
) -> None:
    access_enc = fernet.encrypt(access_token.encode("utf-8")).decode("utf-8")
    public_enc = (
        fernet.encrypt(public_token.encode("utf-8")).decode("utf-8") if public_token else None
    )

    with sqlite3.connect(DB_PATH) as conn:
        if account_id is None:
            raise ValueError("account_id is required for broker token upsert")

        existing = conn.execute(
            """
            SELECT id FROM user_broker_accounts
            WHERE user_id = ? AND provider = ?
            LIMIT 1
            """,
            (user_id, provider),
        ).fetchone()
        is_primary = 1 if existing is None else 0

        conn.execute(
            """
            INSERT INTO user_broker_accounts (user_id, provider, account_id, access_token_enc, public_token_enc, scopes, is_primary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, provider, account_id) DO UPDATE SET
                access_token_enc=excluded.access_token_enc,
                public_token_enc=excluded.public_token_enc,
                scopes=excluded.scopes,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, provider, account_id, access_enc, public_enc, scopes, is_primary),
        )
        conn.commit()


def save_oauth_state(*, user_id: str, provider: str, state: str, ttl_minutes: int = 15) -> None:
    state_hash = _hash_state(state)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO oauth_states (user_id, provider, state_hash, expires_at)
            VALUES (?, ?, ?, datetime('now', ?))
            """,
            (user_id, provider, state_hash, f"+{ttl_minutes} minutes"),
        )
        conn.execute(
            """
            INSERT INTO oauth_audit_log (user_id, provider, state_hash, event)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, provider, state_hash, "state_created"),
        )
        conn.commit()


def consume_oauth_state(*, provider: str, state: str) -> str | None:
    state_hash = _hash_state(state)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT user_id FROM oauth_states
            WHERE provider = ? AND state_hash = ?
              AND expires_at >= datetime('now')
            """,
            (provider, state_hash),
        ).fetchone()

        conn.execute("DELETE FROM oauth_states WHERE state_hash = ?", (state_hash,))
        conn.execute(
            """
            INSERT INTO oauth_audit_log (user_id, provider, state_hash, event)
            VALUES (?, ?, ?, ?)
            """,
            (row[0] if row else None, provider, state_hash, "state_consumed" if row else "state_invalid"),
        )
        conn.commit()

    return row[0] if row else None


def consume_latest_oauth_state(*, provider: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT user_id, state_hash FROM oauth_states
            WHERE provider = ?
              AND expires_at >= datetime('now')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (provider,),
        ).fetchone()

        if not row:
            conn.execute(
                """
                INSERT INTO oauth_audit_log (user_id, provider, state_hash, event)
                VALUES (?, ?, ?, ?)
                """,
                (None, provider, None, "state_missing_and_no_pending_state"),
            )
            conn.commit()
            return None

        user_id, state_hash = row
        conn.execute("DELETE FROM oauth_states WHERE state_hash = ?", (state_hash,))
        conn.execute(
            """
            INSERT INTO oauth_audit_log (user_id, provider, state_hash, event)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, provider, state_hash, "state_consumed_without_callback_state"),
        )
        conn.commit()

    return user_id


def has_broker_account(*, user_id: str, provider: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM user_broker_accounts
            WHERE user_id = ? AND provider = ?
            LIMIT 1
            """,
            (user_id, provider),
        ).fetchone()
    return row is not None


def get_broker_account(*, user_id: str, provider: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT user_id, provider, account_id, scopes, is_primary, created_at, updated_at
            FROM user_broker_accounts
            WHERE user_id = ? AND provider = ?
            ORDER BY is_primary DESC, updated_at DESC
            LIMIT 1
            """,
            (user_id, provider),
        ).fetchone()

    if not row:
        return None

    return {
        "user_id": row[0],
        "provider": row[1],
        "account_id": row[2],
        "scopes": row[3],
        "is_primary": bool(row[4]),
        "created_at": row[5],
        "updated_at": row[6],
    }


def list_broker_accounts(*, user_id: str, provider: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT user_id, provider, account_id, scopes, is_primary, created_at, updated_at
            FROM user_broker_accounts
            WHERE user_id = ? AND provider = ?
            ORDER BY is_primary DESC, updated_at DESC
            """,
            (user_id, provider),
        ).fetchall()

    return [
        {
            "user_id": row[0],
            "provider": row[1],
            "account_id": row[2],
            "scopes": row[3],
            "is_primary": bool(row[4]),
            "created_at": row[5],
            "updated_at": row[6],
        }
        for row in rows
    ]


def get_decrypted_broker_tokens(*, user_id: str, provider: str, account_id: str | None = None) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        if account_id:
            row = conn.execute(
                """
                SELECT account_id, access_token_enc, public_token_enc, scopes, is_primary
                FROM user_broker_accounts
                WHERE user_id = ? AND provider = ? AND account_id = ?
                LIMIT 1
                """,
                (user_id, provider, account_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT account_id, access_token_enc, public_token_enc, scopes, is_primary
                FROM user_broker_accounts
                WHERE user_id = ? AND provider = ?
                ORDER BY is_primary DESC, updated_at DESC
                LIMIT 1
                """,
                (user_id, provider),
            ).fetchone()

    if not row:
        return None

    access_token = fernet.decrypt(row[1].encode("utf-8")).decode("utf-8")
    public_token = (
        fernet.decrypt(row[2].encode("utf-8")).decode("utf-8") if row[2] else None
    )

    return {
        "account_id": row[0],
        "access_token": access_token,
        "public_token": public_token,
        "scopes": row[3],
        "is_primary": bool(row[4]),
    }


def delete_broker_account(*, user_id: str, provider: str, account_id: str | None = None) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        if account_id:
            cur = conn.execute(
                """
                DELETE FROM user_broker_accounts
                WHERE user_id = ? AND provider = ? AND account_id = ?
                """,
                (user_id, provider, account_id),
            )
        else:
            cur = conn.execute(
                """
                DELETE FROM user_broker_accounts
                WHERE user_id = ? AND provider = ?
                """,
                (user_id, provider),
            )

        remaining = conn.execute(
            """
            SELECT id FROM user_broker_accounts
            WHERE user_id = ? AND provider = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id, provider),
        ).fetchone()

        conn.execute(
            """
            UPDATE user_broker_accounts
            SET is_primary = 0
            WHERE user_id = ? AND provider = ?
            """,
            (user_id, provider),
        )
        if remaining:
            conn.execute(
                """
                UPDATE user_broker_accounts
                SET is_primary = 1
                WHERE id = ?
                """,
                (remaining[0],),
            )

        conn.execute(
            """
            INSERT INTO oauth_audit_log (user_id, provider, event)
            VALUES (?, ?, ?)
            """,
            (user_id, provider, "account_unlinked" if account_id else "all_accounts_unlinked"),
        )
        conn.commit()
    return cur.rowcount > 0


def set_primary_broker_account(*, user_id: str, provider: str, account_id: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        exists = conn.execute(
            """
            SELECT id FROM user_broker_accounts
            WHERE user_id = ? AND provider = ? AND account_id = ?
            LIMIT 1
            """,
            (user_id, provider, account_id),
        ).fetchone()
        if not exists:
            return False

        conn.execute(
            """
            UPDATE user_broker_accounts
            SET is_primary = 0
            WHERE user_id = ? AND provider = ?
            """,
            (user_id, provider),
        )
        conn.execute(
            """
            UPDATE user_broker_accounts
            SET is_primary = 1, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ? AND provider = ? AND account_id = ?
            """,
            (user_id, provider, account_id),
        )
        conn.execute(
            """
            INSERT INTO oauth_audit_log (user_id, provider, event)
            VALUES (?, ?, ?)
            """,
            (user_id, provider, "primary_account_set"),
        )
        conn.commit()
        return True


# ── User CRUD ──────────────────────────────────────────────────────────────

def create_user(*, user_id: str, email: str, password_hash: str, display_name: str | None = None) -> bool:
    """Insert a new user. Returns True on success, False if duplicate."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, email, password_hash, display_name)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, email, password_hash, display_name),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def update_user_password(*, email: str, new_password_hash: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
            WHERE email = ?
            """,
            (new_password_hash, email),
        )
        conn.commit()
    return cur.rowcount > 0


def get_user_by_email(email: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT user_id, email, password_hash, display_name FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    if not row:
        return None
    return {"user_id": row[0], "email": row[1], "password_hash": row[2], "display_name": row[3]}


def get_user_by_id(user_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT user_id, email, password_hash, display_name FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return {"user_id": row[0], "email": row[1], "password_hash": row[2], "display_name": row[3]}
