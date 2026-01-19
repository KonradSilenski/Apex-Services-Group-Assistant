from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from backend.entities.stats import Stats
import logging
from fastapi import HTTPException
import glob
import base64

def get_vists_number(db: Session):
    result = db.query(Stats).one()
    return str(result.visits)

def get_codes_number(db: Session):
    result = db.query(Stats).one()
    return str(result.codes)

def get_revenue(db: Session):
    result = db.query(Stats).one()
    return str(result.revenue)

def proliferate_visits(db: Session):
    result = db.query(Stats).one()
    result.visits = result.visits + 1
    db.commit()
    
def proliferate_codes(db: Session, value: int):
    result = db.query(Stats).one()
    result.codes = result.codes + value
    db.commit()
    
def proliferate_revenue(db: Session, value: int):
    result = db.query(Stats).one()
    result.revenue = result.revenue + value
    db.commit()
