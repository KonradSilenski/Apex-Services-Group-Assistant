from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from backend.entities.job import Job
from backend.entities.sor import SOR
from backend.associations.job_to_sor import job_to_sor
from backend.entities.details import Details
from backend.statistics.service import proliferate_codes, proliferate_visits
from backend.codes.service import proliferate_code_counter
from . import model
import logging
from fastapi import HTTPException
import glob
import base64
import requests
import os
import datetime
from dotenv import load_dotenv
import shutil


load_dotenv()

def create_job(db: Session, register_job_request: model.JobResponse) -> None:
    try:
        # Create the Job itself
        create_job_model = Job(
            id=register_job_request.id,
            propeller_id=register_job_request.propeller_id,
            client=register_job_request.client,
            property=register_job_request.property,
            date=register_job_request.date,
        )

        print(f"Create job id: {create_job_model.id}")

        # Handle completed SOR codes
        if register_job_request.sor_completed:
            sor_objs = db.query(SOR).filter(SOR.id.in_(register_job_request.sor_completed)).all()
            sor_map = {sor.id: sor for sor in sor_objs}
            for sor_id in register_job_request.sor_completed:
                if sor_id in sor_map:
                    assoc = job_to_sor(
                        job=create_job_model,
                        sor=sor_map[sor_id],
                        quantity=0,   # default, will be updated later
                        type="completed"
                    )
                    create_job_model.job_sor.append(assoc)

        # Handle future SOR codes
        if register_job_request.sor_future:
            sor_objs = db.query(SOR).filter(SOR.id.in_(register_job_request.sor_future)).all()
            sor_map = {sor.id: sor for sor in sor_objs}
            for sor_id in register_job_request.sor_future:
                if sor_id in sor_map:
                    assoc = job_to_sor(
                        job=create_job_model,
                        sor=sor_map[sor_id],
                        quantity=0,   # default, will be updated later
                        type="future"
                    )
                    create_job_model.job_sor.append(assoc)

        db.add(create_job_model)
        db.commit()
        logging.info(f"Successfully created job with ID: {register_job_request.id}")

    except Exception as e:
        db.rollback()
        logging.error(f"Failed to create a job: {register_job_request.id}. Error: {str(e)}")
        raise


def get_job_by_id(db: Session, job_id: UUID):
    return db.query(Job).filter(Job.id == job_id).first()

def get_all_jobs(db: Session):
    return db.query(Job).all()

def delete_job_by_id(db: Session, job_id: UUID):
    try:
        db.query(job_to_sor).filter(job_to_sor.job_id == job_id).delete(synchronize_session=False)

        deleted_count = db.query(Job).filter(Job.id == job_id).delete(synchronize_session=False)

        if deleted_count == 0:
            logging.warning(f"No job found with id: {job_id}")
            
        detail = db.query(Details).filter_by(job_id=job_id).first()
        
        folder_dir = f'/app/data/{job_id}'
        shutil.rmtree(folder_dir)

        db.delete(detail)
        db.commit()

    except Exception as e:
        db.rollback()
        logging.error(f"Failed to delete job: {job_id}. Error: {str(e)}")
        raise


def change_sor_codes(db: Session, job_id: UUID, job_sor_change: model.JobSORChange) -> None:
    try:
        job = db.query(Job).filter(Job.id == job_id).first()

        quantity = job_sor_change.quantity
        type = job_sor_change.type

        if not job:
            logging.warning(f"Job with ID {job_id} not found.")
            return

        sor_objs = db.query(SOR).filter(SOR.id.in_(job_sor_change.new_sor_codes)).all()

        sor_map = {sor.id: sor for sor in sor_objs}

        for sor_id in job_sor_change.new_sor_codes:
            if sor_id in sor_map:
                assoc = job_to_sor(job=job, sor=sor_map[sor_id], quantity=quantity, type=type)
                job.job_sor.append(assoc)

        db.commit()
        logging.info(f"Successfully changed the SOR codes for job ID: {job_id}")
    except Exception as e:
        logging.error(f"Error during SOR code change for job ID: {job_id}. Error: {str(e)}")
        raise

def remove_sor_code(db: Session, job_id: UUID, sor_id: str):
    job_sor_entry = db.query(job_to_sor).filter_by(job_id=job_id, sor_id=sor_id).first()
    
    if not job_sor_entry:
        raise HTTPException(status_code=404, detail=f"No SOR entry found for job_id {job_id} and sor_id {sor_id}")
    
    db.delete(job_sor_entry)
    db.commit()

def change_sor_quantity(db: Session, job_id: UUID, sor_id: str, quantity: float):
    print(
    f"[DEBUG] Updating FUTURE SOR | "
    f"job_id={job_id} | sor_id='{sor_id}' | "
    f"qty={quantity} | type={type(sor_id)}"
    )

    job_sor_entry = db.query(job_to_sor).filter_by(job_id=job_id, sor_id=sor_id).first()
    
    if not job_sor_entry:
        raise HTTPException(status_code=404, detail=f"No SOR entry found for job_id {job_id} and sor_id {sor_id}")
    
    job_sor_entry.quantity = quantity
    
    db.commit()
    
def get_job_images(db: Session, job_id: UUID):
    image_list = []
    folder_dir = '/app/data'
    
    for image in glob.iglob(f'{folder_dir}/{job_id}/*'):
        
        with open(f'{image}', "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            image_list.append(encoded_string)
    
    return image_list

def change_job_date(job_id: UUID, date: str, db: Session):
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
            logging.warning(f"Job with ID {job_id} not found.")
            return
    
    job.date = date
    db.commit()

def create_quote_payload(job_id, items, notes="Quote Notes", completed=True, use_deposit=True, deposit=0.0, payable_by=0, due_date=None):
    if due_date is None:
        due_date = (datetime.date.today() + datetime.timedelta(days=7)).isoformat()
        
    if completed == True:
        notes = "Completed Works Quote"
    else:
        notes = "Future Works Quote"
    
    payload = {
        "Quote": {
            "dueDate": due_date,
            "useDeposit": use_deposit,
            "deposit": deposit,
            "payableBy": payable_by,
            "notes": notes,
            "quoteItems": []
        },
        "JobId": job_id
    }
    
    for item in items:
        quote_item = {
            "existingPropellerPart": True,
            "code": item["code"],
            "id": item["id"],
            "quantity": item["quantity"],
            "description": item["description"],
            "unitPrice": item["unitPrice"],
            "taxCode": item["taxCode"],
            "discount": 0.0
        }
        payload["Quote"]["quoteItems"].append(quote_item)
    
    return payload

def submit_job(job_id: UUID, propeller_id: str, client: str, db: Session):
        
    API_URL = os.getenv("API_URL")
    SEARCH_URL = os.getenv("SEARCH_URL")
    
    auth_payload = {
        "username": os.getenv("USERNAME"),
        "password": os.getenv("PASSWORD"),
        "grant_type": os.getenv("GRANTTYPE"),
        "organisationcode": os.getenv("ORGCODE")
    }
    token_resp = requests.post(os.getenv("LIVEAPI"), data=auth_payload)
    token_resp.raise_for_status()
    token = token_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    sor_quantity = []
    for assoc in job.job_sor:
        sor = assoc.sor
        sor_model = model.SORRequest.model_validate(sor)
        sor_quantity.append(model.JobSORRequest(sor=sor_model, quantity=assoc.quantity, type=assoc.type))
    
    items_complete = []
    items_future = []
    
    for jq in sor_quantity:
        if jq.sor.short and jq.type == "completed":
            search_payload={
                "code": jq.sor.id,
                "name": "",
                "category": client,
                "subcategory": ""
            }
            response = requests.get(SEARCH_URL, json=search_payload, headers=headers)
            data = response.json()
            
            for entry in data:
                item = {
                "code": jq.sor.id,              # from jq
                "id": entry["id"],              # from API response
                "quantity": jq.quantity,        # from jq
                "description": entry["name"],   # from API response
                "unitPrice": entry["amount"],   # from API response
                "taxCode": entry["taxcode"]     # from API response
                }
                items_complete.append(item)
        elif jq.sor.short and jq.type == "future":
            search_payload={
                "code": jq.sor.id,
                "name": "",
                "category": client,
                "subcategory": ""
            }
                        
            response = requests.get(SEARCH_URL, json=search_payload, headers=headers)
            data = response.json()
                        
            for entry in data:
                item = {
                "code": jq.sor.id,              # from jq
                "id": entry["id"],              # from API response
                "quantity": jq.quantity,        # from jq
                "description": entry["name"],   # from API response
                "unitPrice": entry["amount"],   # from API response
                "taxCode": entry["taxcode"]     # from API response
                }
                items_future.append(item)
    
    payload_completed = create_quote_payload(propeller_id, items_complete, completed=True)
    
    payload_future = create_quote_payload(propeller_id, items_future, completed=False)
    
    print(f"Payload completed: {items_complete}")
    print(f"Payload future: {items_future}")
    
    response_completed = requests.put(API_URL, json=payload_completed, headers=headers)
    response_future = None
        
    if response_completed.status_code == 200:
        print("Quote posted successfully!")
        print(response_completed.json())
    else:
        print(f"Failed to post quote. Status: {response_completed.status_code}")
        print(response_completed.text)
    
    if len(items_future) != 0:
        response_future = requests.put(API_URL, json=payload_future, headers=headers)
    
        if response_future.status_code == 200:
            print("Quote posted successfully!")
            print(response_future.json())
        else:
            print(f"Failed to post quote. Status: {response_future.status_code}")
            print(response_future.text)
    
    if response_future != None:
        if response_completed.status_code == 200 and response_future.status_code == 200:
            delete_job_by_id(db, job_id)
            proliferate_visits(db)
            for jq in sor_quantity:
                if jq.sor.short and jq.type == "completed":
                    proliferate_code_counter(db, jq.sor.id, jq.quantity)
                        
    else:
        if response_completed.status_code == 200:
            delete_job_by_id(db, job_id)
            proliferate_visits(db)
            for jq in sor_quantity:
                if jq.sor.short and jq.type == "completed":
                    proliferate_code_counter(db, jq.sor.id, jq.quantity)
        