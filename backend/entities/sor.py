from sqlalchemy import Column, String, DOUBLE_PRECISION
from sqlalchemy.dialects.postgresql import UUID
import uuid
from ..database.core import Base
from sqlalchemy.orm import relationship
from ..associations.job_to_sor import job_to_sor


class SOR(Base):
    __tablename__ = 'sor'

    id = Column(String, primary_key=True)
    job_type = Column(String, nullable=True)
    short = Column(String, nullable=True)
    element = Column(String, nullable=True)
    work_categories = Column(String, nullable=True)
    work_sub_categories = Column(String, nullable=True)
    work_sub_categories_attributes = Column(String, nullable=True)
    medium = Column(String, nullable=True)
    counter = Column(DOUBLE_PRECISION, nullable=True)

    job_sor = relationship("job_to_sor", back_populates="sor")

    def __repr__(self):
        return f"<SOR(id='{self.id}', short='{self.short}', element='{self.element}')>"