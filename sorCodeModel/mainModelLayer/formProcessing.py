from typing import Optional
from typing import Optional
from pydantic import BaseModel
from uuid import UUID
#123
def flatten_survey_data(instance) -> dict:
    return {
# General 
"1.1_Visit_Type": instance.visit_type,
"1.2_Work_Description": instance.work_desc,
"1.3_Property_Type": instance.property_type,

# Scaffolding
"2.1_Scaffold_Required": instance.scaffold_required,
"2.2_Scaffold_Type": instance.scaffold_type,

# Elevations
"3.1_Front_Elevation_Measurement": instance.elevation_measurement,
"3.2_Rear_Elevation_Measurement": instance.elevation_measurement,
"3.3_Gable_Elevation_Measurement": instance.elevation_measurement,

# Roof - Pitched
"4.1_Roof_Type": instance.roof_type,
"4.2_Coverings_Type": instance.coverings_type,
"4.3_Tile_Size": instance.tile_size,
"4.4_Roof_Measurement": instance.roof_measurement,

# Roof - Flat
"4.5_Flat_Coverings_Type": instance.coverings_type,
"4.6_Flat_Tile_Size": instance.tile_size,
"4.7_Flat_Measurement": instance.roof_measurement,

# Roof - Other
"4.8_Other_Coverings_Type": instance.coverings_type,
"4.9_Other_Tile_Size": instance.tile_size,
"4.10_Other_Measurement": instance.roof_measurement,

# Ridge Tile
"5.1_Ridge_Tile": instance.ridge_tile,
"5.2_Ridge_Tile_Type": instance.ridge_tile_type,
"5.3_Ridge_Job": instance.ridge_job,
"5.4_Ridge_Measurement": instance.ridge_measurement,

# Leadwork
"6.1_Leadwork": instance.leadwork,
"6.2_Leadwork_Measurement": instance.leadwork_measurement,
"6.3_Leadwork_Comment": instance.leadwork_comment,

# Chimney
"7.1_Chimney": instance.chimney,
"7.2_Chimney_Measurement": instance.chimney_measurement,
"7.3_Chimney_Comment": instance.chimney_comment,
"7.4_Chimney_Flaunch_Measurement": instance.chimney_measurement,
"7.5_Chimney_Flaunch_Comment": instance.chimney_comment,

# Roofline
"8.1_Fascia": instance.fascia,
"8.2_Fascia_Measurement": instance.fascia_measurement,
"8.3_Soffit": instance.soffit,
"8.4_Soffit_Measurement": instance.soffit_measurement,

# Rainwater Goods - Guttering
"9.1_Guttering": instance.guttering,
"9.2_Guttering_Replace": instance.guttering_replace,
"9.3_Guttering_Replace_Measurement": instance.guttering_replace_measurement,
"9.4_Guttering_Refix": instance.guttering_replace,
"9.5_Guttering_Refix_Measurement": instance.guttering_replace_measurement,
"9.6_Guttering_Clean": instance.guttering_clean,
"9.7_Guttering_Num_Elevations": instance.guttering_replace_measurement,

# Rainwater Goods - RWP
"10.1_RWP": instance.rwp,
"10.2_RWP_Replace": instance.rwp_replace,
"10.3_RWP_Replace_Measurement": instance.rwp_replace_measurement,
"10.4_RWP_Refix": instance.rwp_replace,
"10.5_RWP_Refix_Measurement": instance.rwp_replace_measurement,

# Other Works
"11.1_Other_Works_Completed": instance.other_works_completed,
"11.2_Other_Works_Needed": instance.other_works_needed,

# Access
"12.1_Access_Key": instance.access_key,
"12.2_Wall_Notice": instance.wall_notice,

# Other Issues
"13.1_Issues_Present": instance.issues_present,
"13.2_Issues_Comments": instance.issues_comments,

# Customer Vulnerability
"14.1_Customer_Vuln": instance.customer_vuln,
"14.2_Customer_Comments": instance.customer_comments,
    }
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