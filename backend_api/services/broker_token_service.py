from __future__ import annotations

from backend_api.database.token_store import (
    consume_oauth_state,
    consume_latest_oauth_state,
    delete_broker_account,
    get_decrypted_broker_tokens,
    get_broker_account,
    has_broker_account,
    init_db,
    list_broker_accounts,
    save_oauth_state,
    set_primary_broker_account,
    upsert_broker_tokens,
)


class BrokerTokenService:
    def __init__(self) -> None:
        init_db()

    def save_zerodha_tokens(
        self,
        *,
        user_id: str,
        account_id: str | None,
        access_token: str,
        public_token: str | None,
    ) -> None:
        upsert_broker_tokens(
            user_id=user_id,
            provider="zerodha",
            account_id=account_id,
            access_token=access_token,
            public_token=public_token,
            scopes="read_holdings,read_orders",
        )

    def save_oauth_state(self, *, user_id: str, provider: str, state: str, ttl_minutes: int = 15) -> None:
        save_oauth_state(user_id=user_id, provider=provider, state=state, ttl_minutes=ttl_minutes)

    def consume_oauth_state(self, *, provider: str, state: str) -> str | None:
        return consume_oauth_state(provider=provider, state=state)

    def consume_latest_oauth_state(self, *, provider: str) -> str | None:
        return consume_latest_oauth_state(provider=provider)

    def has_linked_account(self, *, user_id: str, provider: str) -> bool:
        return has_broker_account(user_id=user_id, provider=provider)

    def get_linked_account(self, *, user_id: str, provider: str) -> dict | None:
        return get_broker_account(user_id=user_id, provider=provider)

    def list_linked_accounts(self, *, user_id: str, provider: str) -> list[dict]:
        return list_broker_accounts(user_id=user_id, provider=provider)

    def unlink_account(self, *, user_id: str, provider: str, account_id: str | None = None) -> bool:
        return delete_broker_account(user_id=user_id, provider=provider, account_id=account_id)

    def set_primary_account(self, *, user_id: str, provider: str, account_id: str) -> bool:
        return set_primary_broker_account(user_id=user_id, provider=provider, account_id=account_id)

    def get_linked_tokens(
        self,
        *,
        user_id: str,
        provider: str,
        account_id: str | None = None,
    ) -> dict | None:
        return get_decrypted_broker_tokens(user_id=user_id, provider=provider, account_id=account_id)


broker_token_service = BrokerTokenService()
