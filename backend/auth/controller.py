from typing import Annotated
from fastapi import APIRouter, Depends, Request
from starlette import status
from . import model
from . import service
from .service import CurrentUser
from fastapi.security import OAuth2PasswordRequestForm
from ..database.core import Dbsession
from ..rate_limiting import limiter

router = APIRouter(
    prefix='/auth',
    tags=['auth']
)

@router.post("/", status_code=status.HTTP_201_CREATED)
@limiter.limit("5/hour")
async def register_user(request: Request, db: Dbsession,
                        register_user_request: model.RegisterUserRequest):
    service.register_user(db, register_user_request)

@router.post("/token", response_model=model.Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Dbsession):
    return service.login_for_access_token(form_data, db)

@router.get("/verify", status_code=status.HTTP_200_OK)
async def verify_token_route(current_user: CurrentUser):
    return {"message": "Token is valid"}