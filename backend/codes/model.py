from pydantic import BaseModel
from uuid import UUID
from typing import Optional

class SORRequest(BaseModel):
    id: Optional[str] = None
    job_type: Optional[str] = None
    short: Optional[str] = None
    element: Optional[str] = None
    work_categories: Optional[str] = None
    work_sub_categories: Optional[str] = None
    work_sub_categories_attributes: Optional[str] = None
    medium: Optional[str] = None
    counter: Optional[int] = None

    model_config = {"from_attributes": True}

class JobSORRequest(BaseModel):
    sor: SORRequest
    quantity: float
    type: str

    model_config = {"from_attributes": True}