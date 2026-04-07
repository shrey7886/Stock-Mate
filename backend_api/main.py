import uvicorn

from backend_api.app import app
from backend_api.core.config import settings


if __name__ == "__main__":
	uvicorn.run(app, host=settings.app_host, port=settings.app_port)
