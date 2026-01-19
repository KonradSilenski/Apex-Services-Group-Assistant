from sqlalchemy import Column, ForeignKey, Integer, String, Float 
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..database.core import Base

class job_to_sor(Base):
    __tablename__ = 'job_sor'

    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'), primary_key=True)
    sor_id = Column(String, ForeignKey('sor.id'), primary_key=True)
    type = Column(String, nullable=False, primary_key=True)   # <-- make type part of PK
    quantity = Column(Float, nullable=False)

    job = relationship("Job", back_populates="job_sor")
    sor = relationship("SOR", back_populates="job_sor")
