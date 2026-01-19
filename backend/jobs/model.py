from pydantic import BaseModel
from typing import List
from uuid import UUID
from ..codes.model import JobSORRequest, SORRequest

class JobResponse(BaseModel):
    id: UUID
    propeller_id: str
    client: str
    property: str
    date: str
    sor_completed: List[str]
    sor_future: List[str]

class JobDetailResponse(BaseModel):
    id: UUID
    propeller_id: str
    client: str
    property: str
    date: str
    sor_codes: List[JobSORRequest]

    model_config = {"from_attributes": True}

class JobSORChange(BaseModel):
    new_sor_codes: List[str]
    quantity: float
    type: str