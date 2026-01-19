from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc
from sqlalchemy.sql import text
from . import model
from backend.entities.sor import SOR
from backend.exceptions import UserNotFoundError
import logging
from typing import Optional
from sqlalchemy import desc, nullslast, func

def get_sor_by_id(db: Session, sor_id: str) -> model.SORRequest:
    sor = db.query(SOR).filter(SOR.id == sor_id).first()
    if not sor:
        logging.warning(f"SOR Code with {sor_id} ID not found.")
        raise UserNotFoundError(sor_id)
    logging.info(f"Successfully retrieved SOR code with ID: {sor_id}")
    return sor

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc, or_

def get_rows_by_column(
    db: Session,
    limit: int,
    sort_by: str,
    sort_ord: str,
    column: str,
    search: Optional[str] = None
):
    query = db.query(SOR)

    # Apply search filter
    if search and column:
        if hasattr(SOR, column):
            col_attr = getattr(SOR, column)

            if column == "id":
                # First try exact match
                exact_match = query.filter(col_attr == search).all()
                if exact_match:
                    return exact_match[:limit]
                # Otherwise, find closest matches (LIKE)
                query = db.query(SOR).filter(col_attr.like(f"%{search}%"))
            else:
                # Normal exact match for other columns
                query = query.filter(col_attr == search)
        else:
            raise ValueError(f"Invalid column name: {column}")

    # Apply sorting
    if sort_by and hasattr(SOR, sort_by):
        sort_col = getattr(SOR, sort_by)
        query = query.order_by(asc(sort_col) if sort_ord.lower() == "asc" else desc(sort_col))

    return query.limit(limit).all()


def proliferate_code_counter(db: Session, sor_id: str, value: int):
    sor = db.query(SOR).filter(SOR.id == sor_id).first()
    if not sor.counter:
        sor.counter = value
    else:
        sor.counter = sor.counter + value
    db.commit()
    
def get_most_used(db: Session, limit: int):
    return (
        db.query(SOR)
        .order_by(nullslast(func.coalesce(SOR.counter, 0).desc()))
        .order_by(SOR.id.desc())
        .limit(limit)
        .all()
    )