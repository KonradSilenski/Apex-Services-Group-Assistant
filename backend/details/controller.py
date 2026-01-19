from fastapi import APIRouter
from . import service
from ..database.core import Dbsession
from ..auth.service import CurrentUser
from fastapi import HTTPException
from uuid import UUID
from .model import DetailsResponse

router = APIRouter(
    prefix="/details",
    tags=["Details"]
)

@router.get("/", response_model=DetailsResponse)
def get_details(id: UUID, db: Dbsession, current_user: CurrentUser):
    details = service.get_details_by_id(db, id)
    if details is None:
        raise HTTPException(status_code=404, detail="Details not found")
    return details