from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID
import uuid
from ..database.core import Base
from sqlalchemy.orm import relationship
from ..associations.job_to_sor import job_to_sor
from .sor import SOR


class Job(Base):
    __tablename__ = 'jobs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    propeller_id = Column(String, nullable=False)
    client = Column(String, nullable=False)
    property = Column(String, nullable=False)
    date = Column(String, nullable=False)

    job_sor = relationship("job_to_sor", back_populates="job", cascade="all, delete-orphan")
    sor_codes = relationship("SOR", secondary='job_sor', overlaps="job_sor,job")

    def __repr__(self):
        return f"<Job(id={self.id}, client={self.client}, property={self.property})>"