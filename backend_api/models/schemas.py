from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "backend_api"


class ZerodhaStartResponse(BaseModel):
    login_url: str = Field(..., description="Kite login URL")
    state: str = Field(..., description="CSRF state token")


class ZerodhaCallbackResponse(BaseModel):
    success: bool
    message: str
    user_id: str | None = None
    account_id: str | None = None


class LinkedBrokerAccount(BaseModel):
    account_id: str | None = None
    scopes: str | None = None
    is_primary: bool = False
    updated_at: str | None = None


class ZerodhaStatusResponse(BaseModel):
    linked: bool
    provider: str = "zerodha"
    linked_accounts_count: int = 0
    accounts: list[LinkedBrokerAccount] = Field(default_factory=list)


class ActionResponse(BaseModel):
    success: bool
    message: str


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=1)


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)
    display_name: str | None = None


class RegisterResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    message: str = "Account created successfully"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None, description="Optional session override")


class ChatResponse(BaseModel):
    user_id: str
    message: str
    answer: str
    action_tag: str = Field(..., description="Hold | Trim | Add | Watch | Rebalance | None")
    why: str = Field(..., description="One-sentence reasoning")
    risk_note: str = Field(default="", description="Risk caveat or disclaimer")
    confidence: str = Field(..., description="low | medium | high")
    detected_intent: str = Field(default="unknown", description="Detected user intent")
    confidence_score: float = Field(default=0.0, description="0.0 to 1.0 confidence score")
    next_steps: list[str] = Field(default_factory=list, description="Suggested follow-up prompts")
    proactive_insights: list[dict] = Field(default_factory=list, description="Proactive alerts and insights")
    portfolio_health: dict | None = Field(default=None, description="Portfolio health score breakdown")


class WatchlistAddRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, description="NSE/BSE ticker symbol")


class WatchlistResponse(BaseModel):
    user_id: str
    watchlist: list[str]
    count: int


class GoalSetRequest(BaseModel):
    target_amount: float = Field(..., gt=0, description="Target corpus in INR")
    years: float = Field(..., gt=0, description="Time horizon in years")
    label: str = Field(default="Financial Goal", description="Goal label")
    expected_return_pct: float = Field(default=12.0, description="Expected annualized return %")


class GoalResponse(BaseModel):
    user_id: str
    label: str
    target_amount: float
    current_amount: float
    years: float
    progress_pct: float
    monthly_sip_needed: float
    sip_note: str


class PortfolioSummaryResponse(BaseModel):
    user_id: str
    linked: bool
    account_id: str | None = None
    holdings_count: int = 0
    total_invested: float | None = None
    total_current_value: float | None = None
    total_pnl: float | None = None
    total_pnl_pct: float | None = None
    health_score: float | None = None
    holdings: list[dict] = []
    data_status: str | None = None
    action_required: str | None = None
    link_endpoint: str | None = None
    message: str
