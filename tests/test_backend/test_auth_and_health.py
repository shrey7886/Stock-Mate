from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_api.database import token_store
from backend_api.routes.auth import router as auth_router
from backend_api.routes.health import health


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(token_store, "DB_PATH", tmp_path / "auth.db")
    token_store.init_db()
    app = FastAPI()
    app.include_router(auth_router)
    return TestClient(app)


ACCOUNT = {"email": "New.User@Example.com", "password": "secret123"}


def test_new_user_registers_then_signs_in(client):
    reg = client.post("/api/auth/register", json=ACCOUNT)
    assert reg.status_code == 200
    login = client.post("/api/auth/login", json={"email": "new.user@example.com", "password": "secret123"})
    assert login.status_code == 200
    assert login.json()["access_token"]


def test_sign_in_without_account_says_register(client):
    res = client.post("/api/auth/login", json=ACCOUNT)
    assert res.status_code == 404
    assert res.json()["detail"] == "Account does not exist. Please register first."


def test_register_existing_account_says_sign_in(client):
    client.post("/api/auth/register", json=ACCOUNT)
    res = client.post("/api/auth/register", json=ACCOUNT)
    assert res.status_code == 409
    assert res.json()["detail"] == "An account with this email already exists. Please sign in."


def test_wrong_password_is_rejected(client):
    client.post("/api/auth/register", json=ACCOUNT)
    res = client.post("/api/auth/login", json={**ACCOUNT, "password": "wrong-pass"})
    assert res.status_code == 401
    assert "Incorrect password" in res.json()["detail"]


def test_health_includes_broker_configuration_status():
    res = health()
    assert res.status in {"ok", "degraded"}
    assert "zerodha" in res.checks
    assert "upstox" in res.checks
