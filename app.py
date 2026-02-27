import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import time
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle

# Load environment variables from .env file (for SMTP credentials)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

app = FastAPI(title="EndoPredict AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler (with fallback if missing during dev)
model = None
scaler = None
try:
    with open("pcos_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Loaded pcos_model.pkl successfully.")
except FileNotFoundError:
    print("WARNING: pcos_model.pkl not found. Predictions will return mock data.")

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Loaded scaler.pkl successfully.")
except FileNotFoundError:
    print("WARNING: scaler.pkl not found.")


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    risk_percentage: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or scaler is None:
        # Mock prediction if models are missing
        risk_percentage = float(np.random.uniform(5.0, 85.0))
        return PredictionResponse(risk_percentage=risk_percentage)

    features = np.array(request.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    probability = model.predict_proba(scaled_features)[0][1]
    risk_percentage = round(probability * 100, 2)
    return PredictionResponse(risk_percentage=risk_percentage)


@app.get("/health")
async def health():
    return {"status": "healthy"}

# --- AUTHENTICATION & OTP ---
otp_store = {}
users_db = {} # Mock Database
history_db = {} # Mock Database for History: { email: [HistoryItem] }
OTP_EXPIRY_SECONDS = 300 # 5 minutes

class OTPRequest(BaseModel):
    email: str
    name: str
    password: str = None 


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

@app.post("/auth/send-otp")
async def send_otp(req: OTPRequest):
    otp = str(random.randint(100000, 999999))
    expires_at = time.time() + OTP_EXPIRY_SECONDS
    otp_store[req.email] = {"otp": otp, "expires_at": expires_at, "name": req.name}
    
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    
    if smtp_user and smtp_pass:
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = req.email
            msg['Subject'] = "Your Secure Access Code - EndoPredict AI"
            
            html_body = f"""
            <html>
            <body style="font-family: 'Inter', Helvetica, Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 40px 0;">
                <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
                    <div style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); padding: 30px; text-align: center;">
                        <h1 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 600; letter-spacing: 0.5px;">EndoPredict AI</h1>
                    </div>
                    <div style="padding: 40px 30px;">
                        <h2 style="color: #0f172a; font-size: 20px; font-weight: 600; margin-top: 0;">Hello {req.name},</h2>
                        <p style="color: #475569; font-size: 16px; line-height: 1.6; margin-bottom: 24px;">
                            You requested to securely log in to your EndoPredict AI dashboard. Please use the verification code below to complete your sign-in.
                        </p>
                        <div style="background-color: #f1f5f9; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 24px;">
                            <span style="font-family: monospace; font-size: 36px; font-weight: 700; color: #0284c7; letter-spacing: 8px;">{otp}</span>
                        </div>
                        <p style="color: #64748b; font-size: 14px; text-align: center; margin-bottom: 0;">
                            This code will expire in <b>5 minutes</b>. If you did not request this, please safely ignore this email.
                        </p>
                    </div>
                    <div style="background-color: #f8fafc; padding: 20px; text-align: center; border-top: 1px solid #e2e8f0;">
                        <p style="color: #94a3b8; font-size: 12px; margin: 0;">
                            &copy; 2026 EndoPredict AI. All rights reserved.<br>
                            Empowering hormonal health with AI.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"Error sending email: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Fallback print for local dev if credentials are not configured
        print(f"--- MOCK EMAIL --- To: {req.email}, OTP: {otp}")

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
    
    # Save the user with their password
    users_db[req.email] = {
        "name": name,
        "email": req.email,
        "password": req.password
    }
    
    return {
        "status": "success", 
        "token": f"mock-jwt-token-{req.email}", 
        "user": {"email": req.email, "name": name}
    }


@app.post("/auth/login")
async def login_user(req: LoginRequest):
    user = users_db.get(req.email)
    if not user:
        raise HTTPException(status_code=400, detail="Account not found. Please sign up.")
        
    if user["password"] != req.password:
        raise HTTPException(status_code=400, detail="Incorrect email or password.")
        
    return {
        "status": "success",
        "token": f"mock-jwt-token-{req.email}",
        "user": {"email": user["email"], "name": user["name"]}
    }


@app.post("/auth/google")
async def google_login(req: GoogleLogin):
    # Mocking JWT validation for the sake of the requested build
    return {
        "status": "success",
        "token": f"mock-jwt-google-{req.email}",
        "user": {"email": req.email, "name": req.name}
    }


# --- HISTORY ---
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
        "date": req.date
    })
    
    return {"status": "success", "message": "History saved"}

@app.get("/history/{email}")
async def get_history(email: str):
    history = history_db.get(email, [])
    # Return history reversed so newest is first
    return {"status": "success", "history": history[::-1]}
