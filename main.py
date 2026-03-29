from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
SUPABASE_URL = "https://nwyzlwjvmbgyfiqcbcgz.supabase.co"
SUPABASE_KEY = "sb_publishable_NO9Q9eAW6aWRM-9D0Qipog_Y4c-2EgT"
from fastapi.middleware.cors import CORSMiddleware



supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Event(BaseModel):
    email: str
    ip: str
    device_id: str

def calculate_risk(event):
    score = 0
    reasons = []

    if "vpn" in event.ip:
        score += 20
        reasons.append("VPN detected")

    if "new" in event.device_id:
        score += 15
        reasons.append("New device")

    if "temp" in event.email:
        score += 25
        reasons.append("Temporary email")

    return score, reasons

def decision(score):
    if score <= 30:
        return "APPROVE"
    elif score <= 69:
        return "REVIEW"
    else:
        return "BLOCK"

def ai_explanation(reasons):
    return "This activity is suspicious because: " + ", ".join(reasons)

@app.post("/analyze")
def analyze(event: Event):
    score, reasons = calculate_risk(event)
    action = decision(score)
    explanation = ai_explanation(reasons)

    try:
        result = supabase.table("Events").insert({
    "email": event.email,
    "ip": event.ip,
    "device_id": event.device_id,
    "risk_score": score,
    "decision": action,
    "reasons": reasons,
    "ai_explanation": explanation
}).execute()

        print("DB Insert Success:", result)

    except Exception as e:
        print("DB Insert Error:", e)

    return {
        "risk_score": score,
        "decision": action,
        "reasons": reasons,
        "ai_explanation": explanation
    }
@app.get("/events")
def get_events():
    try:
        data = supabase.table("Events").select("*").order("id", desc=True).execute()
        return data.data
    except Exception as e:
        print("Fetch Error:", e)
        return []