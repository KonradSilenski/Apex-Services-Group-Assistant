from fastapi import FastAPI, Request, APIRouter
from datetime import datetime
from . import service
import os
import traceback

app = FastAPI()

router = APIRouter(
    prefix="/webhook",
    tags=["Webhook"]
)

@router.post("/")
async def receive_webhook(request: Request):
    payload = await request.json()
    visitid = payload["events"][0]["visitid"]
    
    if str(visitid) == "0":
        return {"status": "missing visit id"}
    
    date = payload["events"][0]["eventdate"]
    print("Received webhook:", payload)
    print("Webhook details:", visitid, date[:10])
    
    time_created = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        service.create_job_entry(visitid, time_created)
    except Exception:
        print(f"There was an error when processing webhook data: {Exception}")
        traceback.print_exc()
        return{"status": "ok"}
    else:
        return{"status": "ok"}
    finally:
        log_path = "/app/logs/webhook_logs.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time_created}] {payload}\n")