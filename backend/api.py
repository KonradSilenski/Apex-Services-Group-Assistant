from fastapi import FastAPI
from backend.auth.controller import router as auth_router
from backend.users.controller import router as users_router
from backend.codes.controller import router as codes_router
from backend.jobs.controller import router as jobs_router
from backend.details.controller import router as details_router
from backend.webhook.controller import router as webhook_router
from backend.statistics.controller import router as stats_router

def register_routes(app: FastAPI):
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(codes_router)
    app.include_router(jobs_router)
    app.include_router(details_router)
    app.include_router(webhook_router)
    app.include_router(stats_router)