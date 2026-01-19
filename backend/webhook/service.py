from uuid import uuid4
from backend.database.core import get_session
from backend.jobs.service import create_job, change_sor_quantity
from backend.jobs.model import JobResponse
from backend.details.service import create_details
from apiModuleCode.apiModule import extract_instance_from_visit
from sorCodeModel.learnLayer.training_ops import run_with_memory_enabled_once

def create_job_entry(new_propeller_id, new_date):

    folder_dir = '/app/data/'
    new_local_id=uuid4()

    details, newList, images = extract_instance_from_visit(
    visit_id=new_propeller_id,
    save_dir=f'{folder_dir}',
    local_id=new_local_id
    )
    
    print("Details: ", details,  str(new_local_id))

    new_client = list(details.values())[1]
    new_property = list(details.values())[2]
    
    model_output = run_with_memory_enabled_once(new_propeller_id, "sorCodeModel/dataSOR/processedSORCodes2.csv", "/app/model/operator_feedback.csv")
    

    job_model = JobResponse(
        id=new_local_id, 
        propeller_id=str(new_propeller_id),
        client=str(new_client),
        property=str(new_property),
        date=new_date,
        sor_completed=model_output["completed_codes"],
        sor_future=model_output["future_codes"]
    )

    with get_session() as db:
        create_job(db, job_model)
        create_details(db, new_local_id, newList)
        for code in job_model.sor_future:
            qty = model_output["future_quantities"].get(code, 0)
            change_sor_quantity(db, job_model.id, code, qty)
        
        for code in job_model.sor_completed:
            qty = model_output["completed_quantities"].get(code, 0)
            change_sor_quantity(db, job_model.id, code, qty)
          