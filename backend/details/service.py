from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from backend.entities.details import Details
from backend.details.model import DetailsResponse

def get_details_by_id(db: Session, id: UUID):
    return db.query(Details).filter(Details.job_id == id).first()

def create_details(db: Session, id: UUID, register_details_request: DetailsResponse):
    
    create_details_model = Details(
        job_id=id,
    
        visit_type=register_details_request.visit_type,
        work_desc=register_details_request.work_desc,
        property_type=register_details_request.property_type,

        # Scaffolding
        scaffold_required=register_details_request.scaffold_required,
        scaffold_type=register_details_request.scaffold_type,
        elevation_measurement=register_details_request.elevation_measurement,

        # Roof
        roof_type=register_details_request.roof_type,
        coverings_type=register_details_request.coverings_type,
        tile_size=register_details_request.tile_size,
        roof_measurement=register_details_request.roof_measurement,

        # Ridge Tile
        ridge_tile=register_details_request.ridge_tile,
        ridge_tile_type=register_details_request.ridge_tile_type,
        ridge_job=register_details_request.ridge_job,
        ridge_measurement=register_details_request.ridge_measurement,
        
        #Leadwork
        leadwork=register_details_request.leadwork,
        leadwork_measurement=register_details_request.leadwork_measurement,
        leadwork_comment=register_details_request.leadwork_comment,
        
        #Chimney
        chimney=register_details_request.chimney,
        chimney_measurement=register_details_request.chimney_measurement,
        chimney_comment=register_details_request.chimney_comment,

        # Roofline
        fascia=register_details_request.fascia,
        fascia_measurement=register_details_request.fascia_measurement,
        soffit=register_details_request.soffit,
        soffit_measurement=register_details_request.soffit_measurement,

        # Rainwater goods
        guttering=register_details_request.guttering,
        guttering_replace=register_details_request.guttering_replace,
        guttering_replace_measurement=register_details_request.guttering_replace_measurement,
        guttering_clean=register_details_request.guttering_clean,
        rwp=register_details_request.rwp,
        rwp_replace=register_details_request.rwp_replace,
        rwp_replace_measurement=register_details_request.rwp_replace_measurement,
        
        # Other Works
        other_works_completed=register_details_request.other_works_completed,
        other_works_needed=register_details_request.other_works_needed,

        # Access
        access_key=register_details_request.access_key,
        wall_notice=register_details_request.wall_notice,

        # Other issues
        issues_present=register_details_request.issues_present,
        issues_comments=register_details_request.issues_comments,

        # Customer vulnerability
        customer_vuln=register_details_request.customer_vuln,
        customer_comments=register_details_request.customer_comments,
        )
    
    print(f"Details id: {create_details_model.job_id}")
    
    db.add(create_details_model)
    db.commit()