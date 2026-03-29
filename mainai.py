"""
Stage 3 — AI: Complete Fraud Detection API
============================================
Builds on Stage 2 by adding:
  Module 7:  ML risk scoring (XGBoost)
  Module 8:  Claude AI explanations (Anthropic API)
  Module 9:  Analyst chat interface
  Module 10: Feedback loop + model retraining

Setup:
    pip install fastapi uvicorn xgboost scikit-learn pandas anthropic

Run the trainer first:
    python module7_train_model.py

Set your Anthropic API key:
    export ANTHROPIC_API_KEY=your_key_here

Start the API:
    uvicorn main:app --reload

Open Swagger UI:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import sqlite3
import pickle
import os
import anthropic

app = FastAPI(
    title="Fraud Detection API — Stage 3 (AI)",
    description="Rules + ML scoring + Claude AI explanations + analyst chat + feedback loop"
)

# ─── DATABASE SETUP ──────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect("fraud_events.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT,
            email           TEXT,
            ip              TEXT,
            device_id       TEXT,
            amount          REAL DEFAULT 0,
            hour_of_day     INTEGER DEFAULT 12,
            failed_logins   INTEGER DEFAULT 0,
            rules_score     INTEGER,
            ml_score        REAL,
            final_score     INTEGER,
            decision        TEXT,
            reasons         TEXT,
            ai_explanation  TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id    INTEGER,
            label       TEXT,
            analyst     TEXT,
            created_at  TEXT,
            FOREIGN KEY (event_id) REFERENCES events(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ─── MODULE 7: LOAD ML MODEL ─────────────────────────────────────────────────

MODEL_PATH  = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"
model  = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("ML model loaded successfully.")
else:
    print("WARNING: No ML model found. Run module7_train_model.py first.")
    print("         API will use rules-only scoring until model is trained.")

def get_ml_score(is_vpn, is_tor, is_new_device, hour, amount, failed_logins) -> float:
    """Returns fraud probability (0.0 to 1.0) from the XGBoost model."""
    if model is None or scaler is None:
        return 0.0
    import numpy as np
    features = np.array([[is_vpn, is_tor, is_new_device, hour, amount, failed_logins]])
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0][1]
    return round(float(probability), 4)

# ─── MODULE 8: CLAUDE API SETUP ──────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

def get_claude_explanation(transaction: dict, rules_score: int, ml_score: float, decision: str, reasons: list) -> str:
    """Call Claude to generate a natural language fraud explanation."""
    if not claude_client:
        return "AI explanation unavailable — set ANTHROPIC_API_KEY environment variable."

    prompt = f"""You are a fraud analyst assistant. Analyse this payment transaction and explain 
the risk assessment in 2-3 clear sentences suitable for a non-technical fraud analyst.

Transaction details:
- Email: {transaction['email']}
- IP address: {transaction['ip']}
- Device: {transaction['device_id']}
- Amount: ${transaction.get('amount', 0):.2f}
- Hour of day: {transaction.get('hour_of_day', 12)}:00
- Failed login attempts: {transaction.get('failed_logins', 0)}

Risk assessment:
- Rules-based score: {rules_score}/100
- ML model fraud probability: {ml_score:.1%}
- Decision: {decision}
- Risk signals detected: {', '.join(reasons) if reasons else 'None'}

Write a concise explanation of WHY this transaction received this risk assessment. 
Focus on the specific signals present. Be direct and factual."""

    message = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ─── RULES ENGINE ────────────────────────────────────────────────────────────

def run_rules_engine(ip: str, device_id: str, failed_logins: int) -> tuple:
    """Original rules engine from Stage 2 — still runs alongside ML."""
    score = 0
    reasons = []

    if ip.upper().startswith("VPN"):
        score += 40
        reasons.append("VPN IP detected")

    if ip.upper().startswith("TOR"):
        score += 50
        reasons.append("TOR exit node detected")

    if "new_device" in device_id.lower():
        score += 15
        reasons.append("New device")

    if failed_logins >= 3:
        score += 20
        reasons.append(f"Multiple failed logins ({failed_logins})")

    if not reasons:
        reasons.append("No rule-based signals detected")

    return min(score, 100), reasons

# ─── DATA MODELS ─────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    email:         str
    ip:            str
    device_id:     str
    amount:        float = 50.0
    hour_of_day:   int   = 12
    failed_logins: int   = 0

class ChatRequest(BaseModel):
    event_id: int
    question: str
    analyst:  str = "Analyst"

class FeedbackRequest(BaseModel):
    event_id: int
    label:    str   # "fraud" or "legitimate"
    analyst:  str = "Analyst"

# ─── MODULE 7 + 8: POST /analyze ─────────────────────────────────────────────

@app.post("/analyze", summary="Analyse a transaction (Rules + ML + Claude AI)")
def analyze(request: TransactionRequest):
    """
    Runs three layers of analysis:
    1. Rules engine (from Stage 2)
    2. XGBoost ML model (Module 7)
    3. Claude AI explanation (Module 8)
    Returns a fused risk score and decision.
    """
    # Layer 1: Rules engine
    is_vpn    = 1 if request.ip.upper().startswith("VPN") else 0
    is_tor    = 1 if request.ip.upper().startswith("TOR") else 0
    is_new    = 1 if "new_device" in request.device_id.lower() else 0
    rules_score, reasons = run_rules_engine(request.ip, request.device_id, request.failed_logins)

    # Layer 2: ML model score
    ml_score = get_ml_score(
        is_vpn, is_tor, is_new,
        request.hour_of_day,
        request.amount,
        request.failed_logins
    )

    # Fusion: combine rules score + ML probability into final score
    ml_contribution = int(ml_score * 60)            # ML contributes up to 60 points
    final_score     = min(rules_score + ml_contribution, 100)

    # Decision threshold
    decision = "DECLINE" if final_score >= 60 else "APPROVE"

    # Layer 3: Claude AI explanation
    transaction_dict = request.dict()
    ai_explanation = get_claude_explanation(
        transaction_dict, rules_score, ml_score, decision, reasons
    )

    # Store in database
    conn = sqlite3.connect("fraud_events.db")
    cursor = conn.execute("""
        INSERT INTO events (created_at, email, ip, device_id, amount, hour_of_day,
                            failed_logins, rules_score, ml_score, final_score,
                            decision, reasons, ai_explanation)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat() + "+00:00",
        request.email, request.ip, request.device_id,
        request.amount, request.hour_of_day, request.failed_logins,
        rules_score, ml_score, final_score,
        decision, ", ".join(reasons), ai_explanation
    ))
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "id":             event_id,
        "created_at":     datetime.utcnow().isoformat(),
        "email":          request.email,
        "ip":             request.ip,
        "device_id":      request.device_id,
        "rules_score":    rules_score,
        "ml_score":       ml_score,
        "final_score":    final_score,
        "decision":       decision,
        "reasons":        reasons,
        "ai_explanation": ai_explanation
    }

# ─── GET /events ─────────────────────────────────────────────────────────────

@app.get("/events", summary="Retrieve all fraud events")
def get_events():
    conn = sqlite3.connect("fraud_events.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM events ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ─── MODULE 9: POST /chat ─────────────────────────────────────────────────────

@app.post("/chat", summary="Ask Claude about a specific fraud event (Module 9)")
def chat(request: ChatRequest):
    """
    Analyst chat interface. Ask any question about a specific fraud event.
    Claude has full context of the transaction and its risk assessment.
    """
    if not claude_client:
        raise HTTPException(status_code=503, detail="Claude API not configured. Set ANTHROPIC_API_KEY.")

    # Fetch the event from the database
    conn = sqlite3.connect("fraud_events.db")
    conn.row_factory = sqlite3.Row
    event = conn.execute("SELECT * FROM events WHERE id = ?", (request.event_id,)).fetchone()
    conn.close()

    if not event:
        raise HTTPException(status_code=404, detail=f"Event {request.event_id} not found.")

    event = dict(event)

    # Build context-rich prompt
    system_prompt = """You are an expert fraud analyst assistant helping a bank analyst 
investigate payment fraud. You have access to the full transaction record and its 
AI-generated risk assessment. Answer questions clearly and concisely. 
If you are uncertain, say so. Never make up information not in the transaction data."""

    user_message = f"""Transaction record (Event ID: {event['id']}):
- Email: {event['email']}
- IP: {event['ip']}
- Device: {event['device_id']}
- Amount: ${event.get('amount', 0):.2f}
- Hour: {event.get('hour_of_day', 'unknown')}:00
- Failed logins: {event.get('failed_logins', 0)}
- Rules score: {event['rules_score']}/100
- ML fraud probability: {float(event['ml_score']):.1%}
- Final score: {event['final_score']}/100
- Decision: {event['decision']}
- Risk signals: {event['reasons']}
- AI explanation: {event['ai_explanation']}

Analyst question: {request.question}"""

    message = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    return {
        "event_id": request.event_id,
        "analyst":  request.analyst,
        "question": request.question,
        "answer":   message.content[0].text
    }

# ─── MODULE 10: POST /feedback ───────────────────────────────────────────────

@app.post("/feedback", summary="Submit analyst feedback — true fraud or false positive (Module 10)")
def submit_feedback(request: FeedbackRequest):
    """
    Analysts confirm whether the AI decision was correct.
    Labels: 'fraud' or 'legitimate'
    These labels are stored and used to retrain the model via POST /retrain.
    """
    if request.label not in ("fraud", "legitimate"):
        raise HTTPException(status_code=400, detail="Label must be 'fraud' or 'legitimate'.")

    conn = sqlite3.connect("fraud_events.db")
    event = conn.execute("SELECT id FROM events WHERE id = ?", (request.event_id,)).fetchone()
    if not event:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Event {request.event_id} not found.")

    conn.execute("""
        INSERT INTO feedback (event_id, label, analyst, created_at)
        VALUES (?,?,?,?)
    """, (request.event_id, request.label, request.analyst, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    return {
        "message":  f"Feedback recorded for event {request.event_id}",
        "event_id": request.event_id,
        "label":    request.label,
        "analyst":  request.analyst
    }

# ─── MODULE 10: POST /retrain ─────────────────────────────────────────────────

@app.post("/retrain", summary="Retrain the ML model using analyst feedback labels (Module 10)")
def retrain_model():
    """
    Rebuilds the XGBoost model using analyst-labelled events from the feedback table.
    Requires at least 20 labelled events. Saves the new model to disk and reloads it.
    """
    global model, scaler

    import numpy as np
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    conn = sqlite3.connect("fraud_events.db")
    conn.row_factory = sqlite3.Row

    # Join events with their analyst labels
    rows = conn.execute("""
        SELECT e.ip, e.device_id, e.amount, e.hour_of_day, e.failed_logins, f.label
        FROM events e
        JOIN feedback f ON e.id = f.event_id
    """).fetchall()
    conn.close()

    if len(rows) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 20 labelled events to retrain. Currently have {len(rows)}."
        )

    # Rebuild features from labelled events
    X, y = [], []
    for r in rows:
        X.append([
            1 if str(r["ip"]).upper().startswith("VPN") else 0,
            1 if str(r["ip"]).upper().startswith("TOR") else 0,
            1 if "new_device" in str(r["device_id"]).lower() else 0,
            r["hour_of_day"] or 12,
            r["amount"] or 50.0,
            r["failed_logins"] or 0
        ])
        y.append(1 if r["label"] == "fraud" else 0)

    X = np.array(X)
    y = np.array(y)

    new_scaler = StandardScaler()
    X_scaled   = new_scaler.fit_transform(X)

    new_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=max(1, (y == 0).sum() / max(1, (y == 1).sum())),
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    new_model.fit(X_scaled, y)

    # Save and hot-reload
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(new_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(new_scaler, f)

    model  = new_model
    scaler = new_scaler

    fraud_count = int(sum(y))
    legit_count = len(y) - fraud_count

    return {
        "message":      "Model retrained successfully",
        "training_rows": len(rows),
        "fraud_labels":  fraud_count,
        "legit_labels":  legit_count,
        "model_file":    MODEL_PATH
    }