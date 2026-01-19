from fastapi import APIRouter

from ..database.core import Dbsession
from .import model
from . import service
from typing import Optional
from starlette import status
from ..auth.service import CurrentUser

router = APIRouter(
    prefix="/sor",
    tags=["SOR"]
)

@router.get("/id", response_model=model.SORRequest)
def get_code_id(current_user: CurrentUser, id: str, db: Dbsession):
    return service.get_sor_by_id(db, id)

@router.get("/rows")
def get_rows(current_user: CurrentUser, db: Dbsession, limit: int, sort_by: str, sort_ord: str, column: str, search: Optional[str] = None):
    return service.get_rows_by_column(db, limit, sort_by, sort_ord, column, search)

@router.post("/change_counter_value", status_code=status.HTTP_200_OK)
def change_counter(current_user: CurrentUser, db: Dbsession, id: str, value: int):
    service.proliferate_code_counter(db, id, value)
    
@router.get("/most_used", status_code=status.HTTP_200_OK)
def get_most_used(current_user: CurrentUser, db: Dbsession, limit: int):
    return service.get_most_used(db, limit)