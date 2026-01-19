from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from starlette import status
from . import model
from . import service
from ..database.core import Dbsession
from ..auth.service import CurrentUser
from uuid import UUID
from typing import List
from pathlib import Path

router = APIRouter(
    prefix="/jobs",
    tags=["Jobs"]
)

@router.get("/images/{job_id}")
def list_images(db: Dbsession, job_id: str):
    images = service.get_job_images(db, job_id)
    return images

@router.post("/", status_code=status.HTTP_201_CREATED)
def create_job(current_user: CurrentUser, request: model.JobResponse, db: Dbsession):
        service.create_job(db, request)

@router.get("/id", response_model=model.JobDetailResponse)
def read_job(current_user: CurrentUser, job_id: UUID, db: Dbsession):
    job = service.get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    sor_quantity = []
    for assoc in job.job_sor:
        sor = assoc.sor
        sor_model = model.SORRequest.model_validate(sor)
        sor_quantity.append(model.JobSORRequest(sor=sor_model, quantity=assoc.quantity, type=assoc.type))

    return model.JobDetailResponse(
        id=job.id,
        propeller_id=job.propeller_id,
        client=job.client,
        property=job.property,
        date=job.date,
        sor_codes=sor_quantity
    )

@router.get("/all", response_model=List[model.JobDetailResponse])
def read_all_jobs(current_user: CurrentUser, db: Dbsession):
    jobs = service.get_all_jobs(db)
    job_responses = []
    for job in jobs:
        sor_quantity = []
        for assoc in job.job_sor:
            sor = assoc.sor
            sor_model = model.SORRequest.model_validate(sor)
            sor_quantity.append(model.JobSORRequest(sor=sor_model, quantity=assoc.quantity, type=assoc.type))
 
        job_response = model.JobDetailResponse(
            id=job.id,
            propeller_id=job.propeller_id,
            client=job.client,
            property=job.property,
            date=job.date,
            sor_codes=sor_quantity
        )
        job_responses.append(job_response)

    return job_responses

@router.delete("/remove_by_id", status_code=status.HTTP_200_OK)
def delete_job(current_user: CurrentUser, job_id: UUID, db: Dbsession):
    job = service.get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    service.delete_job_by_id(db, job_id)

@router.put("/change_sor", status_code=status.HTTP_200_OK)
def change_sor(
     current_user: CurrentUser,
     job_id: UUID, 
     new_sor: model.JobSORChange,
     db: Dbsession):
     service.change_sor_codes(db, job_id, new_sor)

@router.delete("/remove_sor", status_code=status.HTTP_200_OK)
def remove_sor(
    current_user: CurrentUser,
    job_id: UUID,
    sor: str,
    db: Dbsession
):
    service.remove_sor_code(db, job_id, sor)

@router.put("/change_code_quantity", status_code=status.HTTP_200_OK)
def change_quantity(
    current_user: CurrentUser,
    job_id: UUID,
    sor: str,
    quantity: float,
    db: Dbsession
):
    service.change_sor_quantity(db, job_id, sor, quantity)
    
@router.put("/change_date", status_code=status.HTTP_200_OK)
def change_date(
    current_user: CurrentUser,
    job_id: UUID,
    date: str,
    db: Dbsession
):
    service.change_job_date(job_id, date, db)
    
@router.post("/submit", status_code=status.HTTP_200_OK)
def submit_to_propeller(
    current_user: CurrentUser,
    job_id: UUID,
    propeller_id: str,
    client: str,
    db: Dbsession
):          
    service.submit_job(job_id, propeller_id, client, db)