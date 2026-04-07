from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend_api.core.config import settings
from backend_api.routes.auth import router as auth_router
from backend_api.routes.chat import router as chat_router
from backend_api.routes.health import router as health_router
from backend_api.routes.portfolio import router as portfolio_router
from backend_api.routes.user import router as user_router
from backend_api.routes.zerodha import router as zerodha_router
from backend_api.routes.inference import router as inference_router


def create_app() -> FastAPI:
	app = FastAPI(title=settings.app_name)

	app.add_middleware(
		CORSMiddleware,
		allow_origins=[
			"http://localhost:5173",
			"http://127.0.0.1:5173",
		],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	app.include_router(auth_router)
	app.include_router(health_router)
	app.include_router(user_router)
	app.include_router(portfolio_router)
	app.include_router(chat_router)
	app.include_router(zerodha_router)
	app.include_router(inference_router)
	return app


app = create_app()
