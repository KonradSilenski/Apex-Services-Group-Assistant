from fastapi import FastAPI, Request, APIRouter
from fastapi.staticfiles import StaticFiles
from .database.core import engine, Base
from .entities.user import User
from .api import register_routes
from fastapi.middleware.cors import CORSMiddleware
from sorCodeModel.learnLayer.train_learn_models import train_all_models

app = FastAPI(
    root_path="/api"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://51.195.203.116"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app)

#train_all_models()

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    
@app.get("/app")
def read_main(request: Request):
    return {"message": "Hello World", "root_path": request.scope.get("root_path")}
