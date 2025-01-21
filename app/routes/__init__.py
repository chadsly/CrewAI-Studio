from .routes import router as main_router
from .db_routes import router as db_router

# Combine routers
routers = [main_router, db_router]

__all__ = ["routers"]
