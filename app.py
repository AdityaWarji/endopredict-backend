import random
import time
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI(title="EndoPredict AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= MODEL LOADING =================

model = None
scaler = None

try:
    with open("pcos_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("WARNING: pcos_model.pkl not found. Predictions will return mock data.")

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("WARNING: scaler.pkl not found.")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    risk_percentage: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or scaler is None:
        return PredictionResponse(
            risk_percentage=float(np.random.uniform(5.0, 85.0))
        )

    features = np.array(request.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    probability = model.predict_proba(scaled_features)[0][1]
    return PredictionResponse(
        risk_percentage=round(probability * 100, 2)
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ================= OTP SECTION =================

otp_store = {}
users_db = {}
history_db = {}
OTP_EXPIRY_SECONDS = 300

class OTPRequest(BaseModel):
    email: str
    name: str
    password: str | None = None

class OTPVerify(BaseModel):
    email: str
    otp: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleLogin(BaseModel):
    token: str
    email: str
    name: str


# ✅ RESEND EMAIL FUNCTION
def send_email(to_email, name, otp):
    api_key = os.getenv("RESEND_API_KEY")

    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": "onboarding@resend.dev",
            "to": [to_email],
            "subject": "Your Secure Access Code - EndoPredict AI",
            "html": f"""
                <h2>Hello {name},</h2>
                <p>Your OTP is:</p>
                <h1>{otp}</h1>
                <p>This code expires in 5 minutes.</p>
            """,
        },
    )

    if response.status_code >= 400:
        print(response.text)
        raise HTTPException(status_code=500, detail="Email sending failed")


@app.post("/auth/send-otp")
async def send_otp(req: OTPRequest):
    otp = str(random.randint(100000, 999999))
    expires_at = time.time() + OTP_EXPIRY_SECONDS

    otp_store[req.email] = {
        "otp": otp,
        "expires_at": expires_at,
        "name": req.name,
    }

    send_email(req.email, req.name, otp)

    return {"status": "success", "message": "OTP sent successfully"}


@app.post("/auth/verify-otp")
async def verify_otp(req: OTPVerify):
    record = otp_store.get(req.email)

    if not record:
        raise HTTPException(status_code=400, detail="OTP not found or expired")

    if time.time() > record["expires_at"]:
        del otp_store[req.email]
        raise HTTPException(status_code=400, detail="OTP expired")

    if record["otp"] != req.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    name = record["name"]
    del otp_store[req.email]

    users_db[req.email] = {
        "name": name,
        "email": req.email,
        "password": req.password,
    }

    return {
        "status": "success",
        "token": f"mock-jwt-token-{req.email}",
        "user": {"email": req.email, "name": name},
    }


@app.post("/auth/login")
async def login_user(req: LoginRequest):
    user = users_db.get(req.email)

    if not user:
        raise HTTPException(status_code=400, detail="Account not found.")

    if user["password"] != req.password:
        raise HTTPException(status_code=400, detail="Incorrect password.")

    return {
        "status": "success",
        "token": f"mock-jwt-token-{req.email}",
        "user": {"email": user["email"], "name": user["name"]},
    }


@app.post("/auth/google")
async def google_login(req: GoogleLogin):
    return {
        "status": "success",
        "token": f"mock-jwt-google-{req.email}",
        "user": {"email": req.email, "name": req.name},
    }

# ================= HISTORY =================

class HistoryItemRequest(BaseModel):
    email: str
    risk_percentage: float
    date: str

@app.post("/history")
async def save_history(req: HistoryItemRequest):
    if req.email not in history_db:
        history_db[req.email] = []

    history_db[req.email].append({
        "risk_percentage": req.risk_percentage,
        "date": req.date,
    })

    return {"status": "success", "message": "History saved"}

@app.get("/history/{email}")
async def get_history(email: str):
    history = history_db.get(email, [])
    return {"status": "success", "history": history[::-1]}