from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from starlette import status
from . import service
from ..auth.service import CurrentUser
from ..database.core import Dbsession
from uuid import UUID
from typing import List

router = APIRouter(
    prefix="/stats",
    tags=["Statistics"]
)

@router.get("/visits", status_code=status.HTTP_200_OK)
def get_vists_number(current_user: CurrentUser, db: Dbsession):
    result = service.get_vists_number(db)
    return result

@router.get("/codes", status_code=status.HTTP_200_OK)
def get_codes_number(current_user: CurrentUser, db: Dbsession):
    result = service.get_codes_number(db)
    return result

@router.get("/revenue", status_code=status.HTTP_200_OK)
def get_revenue(current_user: CurrentUser, db: Dbsession):
    result = service.get_revenue(db)
    return result

@router.post("/add", status_code=status.HTTP_200_OK)
def add_value_to_visits(current_user: CurrentUser, db: Dbsession):
    service.proliferate_visits(db)
    
@router.post("/add_codes", status_code=status.HTTP_200_OK)
def add_value_to_visits(current_user: CurrentUser, db: Dbsession, value: int):
    service.proliferate_codes(db, value)