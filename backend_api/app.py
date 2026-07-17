from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import threading

from backend_api.core.config import settings
from backend_api.routes.auth import router as auth_router
from backend_api.routes.chat import router as chat_router
from backend_api.routes.health import router as health_router
from backend_api.routes.portfolio import router as portfolio_router
from backend_api.routes.user import router as user_router
from backend_api.routes.zerodha import router as zerodha_router
from backend_api.routes.upstox import router as upstox_router
from backend_api.routes.inference import router as inference_router

logger = logging.getLogger(__name__)


def _warmup_chat_stack() -> None:
	"""Preload heavy chatbot dependencies to reduce first-query latency."""
	try:
		from llm_orchestrator.rag.retriever import rag_retriever
		rag_retriever.warmup()
		logger.info("Chat stack warmup finished")
	except Exception as exc:
		logger.warning("Chat stack warmup failed: %s", exc)


def create_app() -> FastAPI:
	app = FastAPI(title=settings.app_name)

	frontend_origin = settings.frontend_url.rstrip("/")
	frontend_origin_local = None
	if frontend_origin.startswith("http://localhost"):
		frontend_origin_local = frontend_origin.replace("localhost", "127.0.0.1")
	elif frontend_origin.startswith("http://127.0.0.1"):
		frontend_origin_local = frontend_origin.replace("127.0.0.1", "localhost")

	allow_origins = [frontend_origin]
	if frontend_origin_local:
		allow_origins.append(frontend_origin_local)

	app.add_middleware(
		CORSMiddleware,
		allow_origins=allow_origins,
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
	app.include_router(upstox_router)
	app.include_router(inference_router)

	@app.on_event("startup")
	def _startup_warmup() -> None:
		threading.Thread(target=_warmup_chat_stack, daemon=True).start()

	return app


app = create_app()
