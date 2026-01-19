import os
import csv
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pydantic import BaseModel, ValidationError
from typing import Optional

# Load environment variables
load_dotenv()

# === Database Setup ===
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# === SQLAlchemy ORM Model ===
class SOR(Base):
    __tablename__ = "sor"

    id = Column(String, primary_key=True)
    job_type = Column(String, nullable=True)
    short = Column(String, nullable=True)
    element = Column(String, nullable=True)
    work_categories = Column(String, nullable=True)
    work_sub_categories = Column(String, nullable=True)
    work_sub_categories_attributes = Column(String, nullable=True)
    medium = Column(String, nullable=True)


# === Pydantic Model ===
class RegisterSORRequest(BaseModel):
    id: Optional[str] = None
    job_type: Optional[str] = None
    short: Optional[str] = None
    element: Optional[str] = None
    work_categories: Optional[str] = None
    work_sub_categories: Optional[str] = None
    work_sub_categories_attributes: Optional[str] = None
    medium: Optional[str] = None


# === Get DB Session ===
def get_db() -> Session: # type: ignore
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# === Insert One Entry ===
def register_sor(db: Session, register_sor_request: RegisterSORRequest) -> None:
    try:
        sor_entry = SOR(**register_sor_request.dict())
        db.add(sor_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"Failed to register SOR {register_sor_request.id}: {str(e)}")
        raise


def load_csv_to_db(csv_file_path: str):
    Base.metadata.create_all(bind=engine)

    csv_to_model_map = {
        "id": "id",
        "Job_Type": "job_type",
        "Short": "short",
        "Element": "element",
        "Work_Categories": "work_categories",
        "Work_Sub_Category_Types": "work_sub_categories",
        "Work_Sub_Category_Attributes": "work_sub_categories_attributes",
        "Medium": "medium"
    }

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        db = next(get_db())

        for row in reader:
            try:
                filtered_row = {
                    csv_to_model_map[k]: v for k, v in row.items()
                    if k in csv_to_model_map
                }
                sor_data = RegisterSORRequest(**filtered_row)
                register_sor(db, sor_data)
                print(f"✔️ Added: {sor_data.id}")
            except ValidationError as ve:
                print(f"⚠️ Validation error for row {row.get('id')}: {ve}")
            except Exception as e:
                print(f"❌ Failed to add row {row.get('id')}: {e}")




# === Main Entry ===
if __name__ == "__main__":

    csv_path = '/home/konrad/Desktop/Apex Assistant/backend/codes/processedSORCodes2.csv'
    load_csv_to_db(csv_path)
 