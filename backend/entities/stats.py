from sqlalchemy import Column, BigInteger
from ..database.core import Base

class Stats(Base):
    __tablename__ = 'stats'

    visits = Column(BigInteger, primary_key=True)
    codes = Column(BigInteger)
    revenue = Column(BigInteger)

    def __repr__(self):
        return f"<Statistics(visits='{self.vists}', codes='{self.codes}', revenue='{self.revenue}'])>"