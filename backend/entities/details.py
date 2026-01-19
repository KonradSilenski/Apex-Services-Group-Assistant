from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from ..database.core import Base

class Details(Base):
    __tablename__ = 'details'

    job_id = Column(UUID(as_uuid=True), primary_key=True)
    
    # General
    visit_type = Column(String, nullable=True)
    work_desc = Column(String, nullable=True)
    property_type = Column(String, nullable=True)

    # Scaffolding
    scaffold_required = Column(String, nullable=True)
    scaffold_type = Column(String, nullable=True)
    elevation_measurement = Column(String, nullable=True)

    # Roof
    roof_type = Column(String, nullable=True)
    coverings_type = Column(String, nullable=True)
    tile_size = Column(String, nullable=True)
    roof_measurement = Column(String, nullable=True)

    # Ridge Tile
    ridge_tile = Column(String, nullable=True)
    ridge_tile_type = Column(String, nullable=True)
    ridge_job = Column(String, nullable=True)
    ridge_measurement = Column(String, nullable=True)
    
    #Leadwork
    leadwork = Column(String, nullable=True)
    leadwork_measurement = Column(String, nullable=True)
    leadwork_comment = Column(String, nullable=True)
    
    #Chimney
    chimney = Column(String, nullable=True)
    chimney_measurement = Column(String, nullable=True)
    chimney_comment = Column(String, nullable=True)

    # Roofline
    fascia = Column(String, nullable=True)
    fascia_measurement = Column(String, nullable=True)
    soffit = Column(String, nullable=True)
    soffit_measurement = Column(String, nullable=True)

    # Rainwater goods
    guttering = Column(String, nullable=True)
    guttering_replace = Column(String, nullable=True)
    guttering_replace_measurement = Column(String, nullable=True)
    guttering_clean = Column(String, nullable=True)
    rwp = Column(String, nullable=True)
    rwp_replace = Column(String, nullable=True)
    rwp_replace_measurement = Column(String, nullable=True)
    
    # Other Works
    other_works_completed = Column(String, nullable=True)
    other_works_needed = Column(String, nullable=True)

    # Access
    access_key = Column(String, nullable=True)
    wall_notice = Column(String, nullable=True)

    # Other issues
    issues_present = Column(String, nullable=True)
    issues_comments = Column(String, nullable=True)

    # Customer vulnerability
    customer_vuln = Column(String, nullable=True)
    customer_comments = Column(String, nullable=True)

    def __repr__(self):
        return f"{self.job_id}"