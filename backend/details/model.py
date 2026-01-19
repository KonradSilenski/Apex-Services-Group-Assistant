from pydantic import BaseModel
from uuid import UUID
from typing import Optional

class DetailsResponse(BaseModel):
    
    job_id: UUID
    
    # General
    visit_type: Optional[str] = None
    work_desc: Optional[str] = None
    property_type: Optional[str] = None

    # Scaffolding
    scaffold_required: Optional[str] = None
    scaffold_type: Optional[str] = None             # New Field, defines storeys of building
    elevation_measurement: Optional[str] = None     # New Field, defines measurements of side of building worked on

    # Roof
    roof_type: Optional[str] = None
    coverings_type: Optional[str] = None
    tile_size: Optional[str] = None
    roof_measurement: Optional[str] = None

    # Ridge Tile
    ridge_tile: Optional[str] = None
    ridge_tile_type: Optional[str] = None
    ridge_job: Optional[str] = None
    ridge_measurement: Optional[str] = None
    
    #Leadwork
    leadwork: Optional[str] = None
    leadwork_measurement: Optional[str] = None
    leadwork_comment: Optional[str] = None
    
    #Chimney
    chimney: Optional[str] = None
    chimney_measurement: Optional[str] = None
    chimney_comment: Optional[str] = None

    # Roofline
    fascia: Optional[str] = None
    fascia_measurement: Optional[str] = None
    soffit: Optional[str] = None
    soffit_measurement: Optional[str] = None

    # Rainwater goods
    guttering: Optional[str] = None
    guttering_replace: Optional[str] = None
    guttering_replace_measurement: Optional[str] = None
    guttering_clean: Optional[str] = None                   # New Field, defines as a gutter clean only
          

    rwp: Optional[str] = None
    rwp_replace: Optional[str] = None
    rwp_replace_measurement: Optional[str] = None
    
    # Other Works
    other_works_completed: Optional[str] = None
    other_works_needed: Optional[str] = None

    # Access
    access_key: Optional[str] = None
    wall_notice: Optional[str] = None

    # Other issues
    issues_present: Optional[str] = None
    issues_comments: Optional[str] = None

    # Customer vulnerability
    customer_vuln: Optional[str] = None
    customer_comments: Optional[str] = None