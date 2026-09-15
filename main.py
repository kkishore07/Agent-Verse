import os
import shutil
import json
import uuid
import hashlib
import base64
import math
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import config
import pandas as pd
from database import db
from rag.retriever import index_file, retrieve_context
from agents.coordinator import coordinate_agents
from agents.strategy_agent import generate_strategy
from forecasting.predictor import forecast_sales, predict_inventory_exhaustion
from forecasting.customer_analytics import initialize_customers_and_transactions, get_customer_insights
from whatsapp.twilio_bot import send_whatsapp_message, get_whatsapp_logs, clear_whatsapp_logs, log_whatsapp_message

# Initialize FastAPI App
app = FastAPI(title="AegisAI API", description="Backend services for Multilingual SME Autonomous Copilot")

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler — catches ANY unhandled error (including DB failures)
# and returns a JSON 500/503 response that STILL includes CORS headers
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )

# ---------------------------------------------------------------------------
# Password hashing helpers
# ---------------------------------------------------------------------------
# Database Paths
USERS_DB_PATH = os.path.join(config.DATA_DIR, "users.json")
SESSIONS_DB_PATH = os.path.join(config.DATA_DIR, "sessions.json")
SCHEMES_DB_PATH = os.path.join(config.DATA_DIR, "government_schemes.json")

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return pwd_hash, salt

# ---------------------------------------------------------------------------
# MongoDB DB helper functions
# ---------------------------------------------------------------------------

def get_user_by_email(email: str) -> dict | None:
    """Fetch a user document by email from MongoDB."""
    return db.users.find_one({"email": email}, {"_id": 0})

def save_user(user_doc: dict):
    """Upsert a user document in MongoDB."""
    db.users.replace_one({"email": user_doc["email"]}, user_doc, upsert=True)

def get_email_by_token(token: str) -> str | None:
    """Look up which email owns a session token."""
    rec = db.sessions.find_one({"token": token}, {"_id": 0})
    return rec["email"] if rec else None

def create_session(token: str, email: str):
    """Store a session token → email mapping in MongoDB."""
    db.sessions.replace_one({"token": token}, {"token": token, "email": email}, upsert=True)

def delete_session(token: str):
    """Remove a session token from MongoDB."""
    db.sessions.delete_one({"token": token})

# Data Importers & Mappers
def parse_uploaded_file(file: UploadFile) -> pd.DataFrame:
    ext = os.path.splitext(file.filename)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file.file)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel.")

def map_and_save_products(df: pd.DataFrame, output_path: str):
    expected_cols = ["ProductID", "ProductName", "Category", "StockLevel", "ReorderLevel", "UnitPrice", "RetailPrice", "Supplier"]
    mapped_df = pd.DataFrame()
    df_cols_lower = {col.lower().replace(" ", "").replace("_", ""): col for col in df.columns}
    
    for target in expected_cols:
        target_clean = target.lower()
        if target_clean in df_cols_lower:
            mapped_df[target] = df[df_cols_lower[target_clean]]
        else:
            if target in ["StockLevel", "ReorderLevel"]:
                mapped_df[target] = 0
            elif target in ["UnitPrice", "RetailPrice"]:
                mapped_df[target] = 0.0
            else:
                mapped_df[target] = "Unknown"
                
    mapped_df.to_csv(output_path, index=False)

def map_and_save_transactions(df: pd.DataFrame, output_path: str):
    expected_cols = ["Date", "TransactionID", "CustomerID", "CustomerName", "ProductID", "ProductName", "Category", "Quantity", "Price", "Type", "Amount", "PaymentMode"]
    mapped_df = pd.DataFrame()
    df_cols_lower = {col.lower().replace(" ", "").replace("_", ""): col for col in df.columns}
    
    for target in expected_cols:
        target_clean = target.lower()
        if target_clean in df_cols_lower:
            mapped_df[target] = df[df_cols_lower[target_clean]]
        else:
            if target == "Date":
                mapped_df[target] = datetime.now().strftime("%Y-%m-%d")
            elif target == "Quantity":
                mapped_df[target] = 1
            elif target in ["Price", "Amount"]:
                mapped_df[target] = 0.0
            elif target == "Type":
                mapped_df[target] = "Sale"
            elif target == "PaymentMode":
                mapped_df[target] = "Cash"
            elif target in ["CustomerID", "CustomerName"]:
                mapped_df[target] = ""
            else:
                mapped_df[target] = "Unknown"
                
    mapped_df.to_csv(output_path, index=False)

def initialize_empty_workspace(workspace_dir: str):
    inv_path = os.path.join(workspace_dir, "inventory.csv")
    tx_path = os.path.join(workspace_dir, "transactions.csv")
    
    inv_cols = ["ProductID", "ProductName", "Category", "StockLevel", "ReorderLevel", "UnitPrice", "RetailPrice", "Supplier"]
    tx_cols = ["Date", "TransactionID", "CustomerID", "CustomerName", "ProductID", "ProductName", "Category", "Quantity", "Price", "Type", "Amount", "PaymentMode"]
    
    pd.DataFrame(columns=inv_cols).to_csv(inv_path, index=False)
    pd.DataFrame(columns=tx_cols).to_csv(tx_path, index=False)
    
    initialize_default_suppliers(workspace_dir)
    
    # Initialize customer registry and rich mock sales context
    initialize_customers_and_transactions(workspace_dir)

# --- SUPPLIER & PROCUREMENT HELPERS ---

def initialize_default_suppliers(workspace_id: str):
    """Seed default suppliers in MongoDB if none exist for this workspace."""
    if db.suppliers.count_documents({"workspace_id": workspace_id}) == 0:
        default_sups = [
            {
                "workspace_id": workspace_id,
                "id": "S101",
                "name": "Sri Balaji Traders",
                "phone": "+919876543211",
                "email": "balaji@traders.com",
                "reliability": 95,
                "payment_terms": "Net 15",
                "catalog": [
                    {"product_id": "P101", "unit_price": 350.0, "min_order_qty": 10, "lead_time_days": 3},
                    {"product_id": "P104", "unit_price": 220.0, "min_order_qty": 15, "lead_time_days": 4}
                ]
            },
            {
                "workspace_id": workspace_id,
                "id": "S102",
                "name": "Vignesh Wholesalers",
                "phone": "+919876543212",
                "email": "vignesh@wholesalers.com",
                "reliability": 88,
                "payment_terms": "COD",
                "catalog": [
                    {"product_id": "P102", "unit_price": 110.0, "min_order_qty": 20, "lead_time_days": 2},
                    {"product_id": "P107", "unit_price": 95.0, "min_order_qty": 10, "lead_time_days": 5},
                    {"product_id": "P108", "unit_price": 115.0, "min_order_qty": 12, "lead_time_days": 3}
                ]
            },
            {
                "workspace_id": workspace_id,
                "id": "S103",
                "name": "Tirupur Distributors",
                "phone": "+919876543213",
                "email": "tirupur@distributors.com",
                "reliability": 91,
                "payment_terms": "Net 30",
                "catalog": [
                    {"product_id": "P103", "unit_price": 20.0, "min_order_qty": 30, "lead_time_days": 6},
                    {"product_id": "P106", "unit_price": 12.0, "min_order_qty": 50, "lead_time_days": 2},
                    {"product_id": "P109", "unit_price": 20.0, "min_order_qty": 25, "lead_time_days": 3}
                ]
            },
            {
                "workspace_id": workspace_id,
                "id": "S104",
                "name": "Raja Pulses",
                "phone": "+919876543214",
                "email": "raja@pulses.com",
                "reliability": 84,
                "payment_terms": "UPI",
                "catalog": [
                    {"product_id": "P105", "unit_price": 140.0, "min_order_qty": 10, "lead_time_days": 7}
                ]
            }
        ]
        db.suppliers.insert_many(default_sups)

def sync_product_with_supplier(workspace_id: str, product_id: str, unit_price: float, supplier_name: str):
    """Upsert a product entry in a supplier's catalog within MongoDB."""
    if not supplier_name or not isinstance(supplier_name, str) or supplier_name.strip() == "" or supplier_name == "Unknown":
        return

    initialize_default_suppliers(workspace_id)

    # Try to find existing supplier
    target_sup = db.suppliers.find_one(
        {"workspace_id": workspace_id, "name": {"$regex": f"^{supplier_name}$", "$options": "i"}},
        {"_id": 0}
    )

    if not target_sup:
        count = db.suppliers.count_documents({"workspace_id": workspace_id})
        sup_id = f"S{100 + count + 1}"
        phone = f"+9198765{10000 + count}"
        target_sup = {
            "workspace_id": workspace_id,
            "id": sup_id,
            "name": supplier_name,
            "phone": phone,
            "email": f"{supplier_name.lower().replace(' ', '')}@wholesaler.com",
            "reliability": 85,
            "payment_terms": "COD",
            "catalog": []
        }
        db.suppliers.insert_one(target_sup)
        # Reload without _id
        target_sup = db.suppliers.find_one(
            {"workspace_id": workspace_id, "id": sup_id}, {"_id": 0}
        )

    # Update catalog entry
    catalog = target_sup.get("catalog", [])
    product_in_catalog = False
    for item in catalog:
        if item["product_id"] == product_id:
            item["unit_price"] = unit_price
            product_in_catalog = True
            break

    if not product_in_catalog:
        catalog.append({
            "product_id": product_id,
            "unit_price": unit_price,
            "min_order_qty": 10,
            "lead_time_days": 3
        })

    db.suppliers.update_one(
        {"workspace_id": workspace_id, "id": target_sup["id"]},
        {"$set": {"catalog": catalog}}
    )

# ---------------------------------------------------------------------------
# Security Dependency
# ---------------------------------------------------------------------------

def get_current_user_workspace_info(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized - Missing or invalid token format")
    token = authorization.split(" ")[1]

    # Resolve session token → email via MongoDB
    email = get_email_by_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Session expired")

    # Fetch user from MongoDB
    user_record = get_user_by_email(email)
    if not user_record:
        raise HTTPException(status_code=401, detail="User not found")

    workspace_id = user_record.get("workspace_id")
    if not workspace_id:
        workspace_id = str(uuid.uuid4())
        user_record["workspace_id"] = workspace_id
        save_user(user_record)

    workspace_dir = os.path.join(config.WORKSPACES_DIR, workspace_id)
    os.makedirs(workspace_dir, exist_ok=True)

    # Ensure default suppliers exist in MongoDB for this workspace
    initialize_default_suppliers(workspace_id)
    initialize_customers_and_transactions(workspace_dir)

    # Read onboarding from MongoDB
    onb_doc = db.onboarding.find_one({"workspace_id": workspace_id}, {"_id": 0})
    business_details = None
    is_onboarded = False
    if onb_doc:
        business_details = {k: v for k, v in onb_doc.items() if k != "workspace_id"}
        is_onboarded = True

    # Read logo base64 if present (still stored on disk)
    logo_base64 = None
    for ext in ['.png', '.jpg', '.jpeg', '.svg']:
        logo_path = os.path.join(workspace_dir, f"logo{ext}")
        if os.path.exists(logo_path):
            try:
                with open(logo_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                mime = "image/png"
                if ext == ".jpg" or ext == ".jpeg": mime = "image/jpeg"
                elif ext == ".svg": mime = "image/svg+xml"
                logo_base64 = f"data:{mime};base64,{encoded_string}"
                break
            except Exception:
                pass

    return {
        "email": email,
        "fullName": user_record.get("full_name"),
        "mobile": user_record.get("mobile"),
        "preferredLanguage": user_record.get("preferred_language"),
        "workspace_id": workspace_id,
        "workspace_dir": workspace_dir,
        "is_onboarded": is_onboarded,
        "business": business_details,
        "logo": logo_base64
    }

# Pydantic request structures
class SignupRequest(BaseModel):
    fullName: str
    email: str
    mobile: str
    password: str
    preferredLanguage: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    query: str
    use_rag: bool = True
    provider: str = "groq"

class AlertRequest(BaseModel):
    custom_recipient: Optional[str] = None
    provider: str = "gemini"

class ProductAddRequest(BaseModel):
    ProductID: str
    ProductName: str
    Category: str
    StockLevel: int
    ReorderLevel: int
    UnitPrice: float
    RetailPrice: float
    Supplier: str

class SupplierAddRequest(BaseModel):
    name: str
    phone: str
    email: str
    paymentTerms: str

class CustomerAddRequest(BaseModel):
    name: str
    email: str
    phone: str

class CampaignRequest(BaseModel):
    segment: Optional[str] = None
    customer_id: Optional[str] = None
    custom_offer: Optional[str] = None
    provider: str = "gemini"

class POApproveRequest(BaseModel):
    productId: str
    supplierId: str
    quantity: int

class POReceiveRequest(BaseModel):
    poId: str

class StrategyInputsRequest(BaseModel):
    businessType: str
    targetAudience: str
    goals: str
    competitors: Optional[str] = ""
    language: Optional[str] = "english"


# Endpoints
@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to AegisAI SME Copilot Backend"}

# --- AUTHENTICATION & ONBOARDING ROUTES ---

@app.post("/api/auth/signup")
def signup(req: SignupRequest):
    email_clean = req.email.strip().lower()

    # Check duplicate via MongoDB
    if get_user_by_email(email_clean):
        raise HTTPException(status_code=400, detail="User with this email already exists")

    pwd_hash, salt = hash_password(req.password)
    workspace_id = str(uuid.uuid4())

    user_doc = {
        "full_name": req.fullName,
        "email": email_clean,
        "mobile": req.mobile,
        "password_hash": pwd_hash,
        "salt": salt,
        "preferred_language": req.preferredLanguage,
        "workspace_id": workspace_id,
        "created_at": datetime.now().isoformat()
    }
    save_user(user_doc)

    # Initialize workspace folder & default suppliers in MongoDB
    workspace_dir = os.path.join(config.WORKSPACES_DIR, workspace_id)
    os.makedirs(workspace_dir, exist_ok=True)
    initialize_default_suppliers(workspace_id)

    return {"success": True, "message": "User registered successfully"}

@app.post("/api/auth/login")
def login(req: LoginRequest):
    email_clean = req.email.strip().lower()

    # Fetch user from MongoDB
    user_record = get_user_by_email(email_clean)
    if not user_record:
        raise HTTPException(status_code=400, detail="Invalid email or password")

    pwd_hash, _ = hash_password(req.password, user_record["salt"])
    if pwd_hash != user_record["password_hash"]:
        raise HTTPException(status_code=400, detail="Invalid email or password")

    # Generate session token and store in MongoDB
    token = str(uuid.uuid4())
    create_session(token, email_clean)

    # Check onboarding status via MongoDB
    workspace_id = user_record["workspace_id"]
    is_onboarded = db.onboarding.find_one({"workspace_id": workspace_id}) is not None

    return {
        "success": True,
        "token": token,
        "user": {
            "fullName": user_record["full_name"],
            "email": email_clean,
            "mobile": user_record["mobile"],
            "preferredLanguage": user_record["preferred_language"],
            "isOnboarded": is_onboarded
        }
    }

@app.get("/api/auth/me")
def get_me(user_info: dict = Depends(get_current_user_workspace_info)):
    return {
        "authenticated": True,
        "user": {
            "fullName": user_info["fullName"],
            "email": user_info["email"],
            "mobile": user_info["mobile"],
            "preferredLanguage": user_info["preferredLanguage"],
            "isOnboarded": user_info["is_onboarded"]
        },
        "business": user_info["business"],
        "logo": user_info["logo"]
    }

@app.post("/api/auth/onboard")
async def onboard(
    businessName: str = Form(...),
    businessCategory: str = Form(...),
    businessLocation: str = Form(...),
    currency: str = Form("₹"),
    merchantWhatsapp: str = Form(...),
    enableInventory: str = Form("false"),
    enableWhatsapp: str = Form("false"),
    startFresh: str = Form("true"),
    businessLogo: UploadFile = File(None),
    productsFile: UploadFile = File(None),
    transactionsFile: UploadFile = File(None),
    authorization: str = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]

    # Resolve session via MongoDB
    email = get_email_by_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Session expired")

    user_record = get_user_by_email(email)
    if not user_record:
        raise HTTPException(status_code=401, detail="User not found")

    workspace_id = user_record.get("workspace_id")
    workspace_dir = os.path.join(config.WORKSPACES_DIR, workspace_id)
    os.makedirs(workspace_dir, exist_ok=True)

    initialize_default_suppliers(workspace_id)

    # Save optional logo (still on disk — binary blobs stay local)
    if businessLogo and businessLogo.filename:
        logo_ext = os.path.splitext(businessLogo.filename)[1].lower()
        if logo_ext in ['.png', '.jpg', '.jpeg', '.svg']:
            logo_path = os.path.join(workspace_dir, f"logo{logo_ext}")
            for ext in ['.png', '.jpg', '.jpeg', '.svg']:
                prev_logo = os.path.join(workspace_dir, f"logo{ext}")
                if os.path.exists(prev_logo):
                    os.remove(prev_logo)
            with open(logo_path, "wb") as buffer:
                shutil.copyfileobj(businessLogo.file, buffer)

    # Save onboarding config to MongoDB
    onboarding_data = {
        "workspace_id": workspace_id,
        "businessName": businessName,
        "businessCategory": businessCategory,
        "businessLocation": businessLocation,
        "currency": currency,
        "merchantWhatsapp": merchantWhatsapp,
        "enableInventory": enableInventory == "true",
        "enableWhatsapp": enableWhatsapp == "true",
        "startFresh": startFresh == "true"
    }
    db.onboarding.replace_one(
        {"workspace_id": workspace_id}, onboarding_data, upsert=True
    )

    # Set up CSV workspace files (still used by pandas analytics)
    if startFresh == "true":
        initialize_empty_workspace(workspace_dir)
    else:
        inv_path = os.path.join(workspace_dir, "inventory.csv")
        if productsFile and productsFile.filename:
            try:
                prod_df = parse_uploaded_file(productsFile)
                map_and_save_products(prod_df, inv_path)

                # Sync unique suppliers from imported catalog to MongoDB
                for _, row in prod_df.iterrows():
                    p_id = str(row.get("ProductID", ""))
                    p_price = float(row.get("UnitPrice", 0.0))
                    p_sup = str(row.get("Supplier", "Unknown"))
                    if p_id and p_sup != "Unknown":
                        sync_product_with_supplier(workspace_id, p_id, p_price, p_sup)
            except Exception as e:
                pd.DataFrame(columns=["ProductID", "ProductName", "Category", "StockLevel", "ReorderLevel", "UnitPrice", "RetailPrice", "Supplier"]).to_csv(inv_path, index=False)
                print(f"Products import failed: {e}")
        else:
            pd.DataFrame(columns=["ProductID", "ProductName", "Category", "StockLevel", "ReorderLevel", "UnitPrice", "RetailPrice", "Supplier"]).to_csv(inv_path, index=False)

        tx_path = os.path.join(workspace_dir, "transactions.csv")
        if transactionsFile and transactionsFile.filename:
            try:
                tx_df = parse_uploaded_file(transactionsFile)
                map_and_save_transactions(tx_df, tx_path)
            except Exception as e:
                pd.DataFrame(columns=["Date", "TransactionID", "ProductID", "ProductName", "Category", "Quantity", "Price", "Type", "Amount", "PaymentMode"]).to_csv(tx_path, index=False)
                print(f"Transactions import failed: {e}")
        else:
            pd.DataFrame(columns=["Date", "TransactionID", "ProductID", "ProductName", "Category", "Quantity", "Price", "Type", "Amount", "PaymentMode"]).to_csv(tx_path, index=False)

    return {"success": True, "message": "Onboarding completed successfully"}

def load_schemes() -> list:
    if not os.path.exists(SCHEMES_DB_PATH):
        return []
    try:
        with open(SCHEMES_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_schemes(schemes: list):
    with open(SCHEMES_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(schemes, f, indent=4)

# Pydantic models for Eligibility
class EligibilityCheckRequest(BaseModel):
    businessName: str
    businessType: str
    state: str
    district: str
    businessStartDate: str
    annualTurnover: float
    gstStatus: str
    udyamStatus: str
    enterpriseCategory: str
    employeeCount: int
    businessSector: str
    loanRequirement: float
    previousAssistance: str
    socialCategory: str
    ownerGender: str
    area: str
    language: Optional[str] = "english"

class SchemeRulesModel(BaseModel):
    enterprise_categories: list[str]
    sectors: list[str]
    max_turnover: Optional[float] = None
    min_loan_requirement: Optional[float] = None
    max_loan_requirement: Optional[float] = None
    requires_udyam: bool
    requires_gst: bool
    owner_gender: Optional[list[str]] = None
    owner_social_category: Optional[list[str]] = None

class DocumentModel(BaseModel):
    id: str
    name_en: str
    name_ta: str
    name_hi: str

class SchemeModel(BaseModel):
    id: str
    name_en: str
    name_ta: str
    name_hi: str
    description_en: str
    description_ta: str
    description_hi: str
    benefits_en: str
    benefits_ta: str
    benefits_hi: str
    official_link: str
    required_documents: list[DocumentModel]
    rules: SchemeRulesModel

# --- GOVERNMENT SCHEME ELIGIBILITY ROUTES ---

@app.get("/api/eligibility/schemes")
def get_eligibility_schemes():
    return load_schemes()

@app.get("/api/eligibility/profile")
def get_eligibility_profile(user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_dir = user_info["workspace_dir"]
    profile_path = os.path.join(workspace_dir, "business_profile.json")
    if not os.path.exists(profile_path):
        return {"success": False, "message": "No profile found"}
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        return {"success": True, "profile": profile}
    except Exception:
        return {"success": False, "message": "Failed to load profile"}

@app.post("/api/eligibility/check")
def check_eligibility(req: EligibilityCheckRequest, user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_dir = user_info["workspace_dir"]
    profile_path = os.path.join(workspace_dir, "business_profile.json")
    
    # Save business profile
    profile = req.dict()
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4)
        
    # Evaluate against schemes
    schemes = load_schemes()
    from agents.eligibility_agent import evaluate_eligibility, generate_groq_explanation
    
    results = []
    for s in schemes:
        eval_res = evaluate_eligibility(profile, s)
        explanation = generate_groq_explanation(profile, eval_res, s.get(f"name_{req.language}", s["name_en"]), req.language)
        results.append({
            "scheme_id": s["id"],
            "name": {
                "en": s["name_en"],
                "ta": s["name_ta"],
                "hi": s["name_hi"]
            },
            "description": {
                "en": s["description_en"],
                "ta": s["description_ta"],
                "hi": s["description_hi"]
            },
            "benefits": {
                "en": s["benefits_en"],
                "ta": s["benefits_ta"],
                "hi": s["benefits_hi"]
            },
            "official_link": s["official_link"],
            "required_documents": s["required_documents"],
            "status": eval_res["status"],
            "matched_conditions": eval_res["matched_conditions"],
            "missing_requirements": eval_res["missing_requirements"],
            "explanation": explanation
        })
        
    return {
        "success": True,
        "profile": profile,
        "results": results
    }

@app.post("/api/eligibility/verify-document")
async def verify_scheme_document(
    schemeId: str = Form(...),
    documentId: str = Form(...),
    file: UploadFile = File(...),
    user_info: dict = Depends(get_current_user_workspace_info)
):
    workspace_dir = user_info["workspace_dir"]
    profile_path = os.path.join(workspace_dir, "business_profile.json")
    
    if not os.path.exists(profile_path):
        raise HTTPException(status_code=400, detail="Please complete the Quick Eligibility Check form first.")
        
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load your business profile.")
        
    # Save the file
    upload_dir = os.path.join(workspace_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Extract text content
    from rag.retriever import extract_text_from_file
    extracted_text, file_type = extract_text_from_file(file_path)
    
    # Send extracted text to LLM to parse structured JSON
    prompt = (
        f"Analyze the following OCR text extracted from a business/identity document:\n\n"
        f"{extracted_text}\n\n"
        "Identify the document type (e.g. Udyam Registration, GST Certificate, Aadhaar Card, Caste Certificate, Project Report, Address Proof).\n"
        "Extract the following details if present (return null if not found):\n"
        "1. Business Name\n"
        "2. GSTIN\n"
        "3. Udyam Number\n"
        "4. State\n"
        "5. District\n"
        "6. Owner Name\n"
        "7. Caste/Category (SC, ST, OBC, General)\n"
        "8. Gender (Male, Female, Other)\n\n"
        "Format the output strictly as a JSON object with keys: "
        "\"document_type\", \"business_name\", \"gstin\", \"udyam_number\", \"state\", \"district\", \"owner_name\", \"caste\", \"gender\""
    )
    
    import re
    from agents.base_agent import call_llm
    
    extracted_data = {}
    try:
        system_instruction = "You are a document extraction assistant. Respond ONLY with valid JSON."
        provider = "gemini" if config.is_gemini_available() else "groq"
        response_text = call_llm(system_instruction, prompt, provider=provider)
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        extracted_data = json.loads(response_text)
    except Exception as e:
        print(f"Error parsing document: {e}")
        # Fallback regex parsing
        extracted_data = {
            "document_type": "Unknown",
            "business_name": None,
            "gstin": None,
            "udyam_number": None,
            "state": None,
            "district": None,
            "owner_name": None,
            "caste": None,
            "gender": None
        }
        text_lower = extracted_text.lower()
        if "udyam" in text_lower:
            extracted_data["document_type"] = "Udyam Registration"
        elif "gstin" in text_lower or "goods and services" in text_lower:
            extracted_data["document_type"] = "GST Registration"
        elif "aadhaar" in text_lower:
            extracted_data["document_type"] = "Aadhaar Card"
        
        gst_match = re.search(r"\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}", extracted_text)
        if gst_match:
            extracted_data["gstin"] = gst_match.group(0)
            
        udyam_match = re.search(r"UDYAM-[A-Z]{2}-\d{2}-\d{7}", extracted_text, re.IGNORECASE)
        if udyam_match:
            extracted_data["udyam_number"] = udyam_match.group(0).upper()
            
    # Perform verification check against profile
    matches = []
    mismatches = []
    missing = []
    
    # 1. Document Type Check
    doc_type = extracted_data.get("document_type", "Unknown")
    
    # 2. Field Comparisons
    # Business Name Check
    doc_biz_name = extracted_data.get("business_name")
    prof_biz_name = profile.get("businessName")
    if doc_biz_name and prof_biz_name:
        if prof_biz_name.lower().replace(" ", "") in doc_biz_name.lower().replace(" ", "") or doc_biz_name.lower().replace(" ", "") in prof_biz_name.lower().replace(" ", ""):
            matches.append({
                "field": "Business Name",
                "expected": prof_biz_name,
                "found": doc_biz_name
            })
        else:
            mismatches.append({
                "field": "Business Name",
                "expected": prof_biz_name,
                "found": doc_biz_name,
                "reason": "Business name on document does not match entered profile."
            })
            
    # GSTIN check
    doc_gstin = extracted_data.get("gstin")
    prof_gst_status = profile.get("gstStatus")
    if doc_gstin:
        if prof_gst_status == "Registered":
            matches.append({
                "field": "GST Registration",
                "expected": "Registered",
                "found": f"GSTIN: {doc_gstin}"
            })
        else:
            mismatches.append({
                "field": "GST Registration",
                "expected": "Not Registered",
                "found": f"GSTIN: {doc_gstin}",
                "reason": "Document contains a GSTIN but profile lists business as Not Registered."
            })
            
    # Udyam number check
    doc_udyam = extracted_data.get("udyam_number")
    prof_udyam_status = profile.get("udyamStatus")
    if doc_udyam:
        if prof_udyam_status == "Registered":
            matches.append({
                "field": "Udyam/MSME Registration",
                "expected": "Registered",
                "found": f"Udyam No: {doc_udyam}"
            })
        else:
            mismatches.append({
                "field": "Udyam/MSME Registration",
                "expected": "Not Registered",
                "found": f"Udyam No: {doc_udyam}",
                "reason": "Document contains Udyam number but profile lists business as Not Registered."
            })
            
    # State check
    doc_state = extracted_data.get("state")
    prof_state = profile.get("state")
    if doc_state and prof_state:
        if prof_state.lower().replace(" ", "") in doc_state.lower().replace(" ", "") or doc_state.lower().replace(" ", "") in prof_state.lower().replace(" ", ""):
            matches.append({
                "field": "State",
                "expected": prof_state,
                "found": doc_state
            })
        else:
            mismatches.append({
                "field": "State",
                "expected": prof_state,
                "found": doc_state,
                "reason": "State on document does not match business profile state."
            })
            
    # Owner Name Check
    doc_owner = extracted_data.get("owner_name")
    prof_owner = user_info.get("fullName")
    if doc_owner and prof_owner:
        if prof_owner.lower().replace(" ", "") in doc_owner.lower().replace(" ", "") or doc_owner.lower().replace(" ", "") in prof_owner.lower().replace(" ", ""):
            matches.append({
                "field": "Owner / Applicant Name",
                "expected": prof_owner,
                "found": doc_owner
            })
        else:
            mismatches.append({
                "field": "Owner / Applicant Name",
                "expected": prof_owner,
                "found": doc_owner,
                "reason": "Owner name on document does not match registered user name."
            })
            
    # Caste Category Check
    doc_caste = extracted_data.get("caste")
    prof_caste = profile.get("socialCategory")
    if doc_caste and prof_caste:
        if prof_caste.lower() in doc_caste.lower() or doc_caste.lower() in prof_caste.lower():
            matches.append({
                "field": "Social Category",
                "expected": prof_caste,
                "found": doc_caste
            })
        else:
            mismatches.append({
                "field": "Social Category",
                "expected": prof_caste,
                "found": doc_caste,
                "reason": "Caste/Category on certificate does not match entered profile."
            })
            
    # Gender check
    doc_gender = extracted_data.get("gender")
    prof_gender = profile.get("ownerGender")
    if doc_gender and prof_gender:
        if prof_gender.lower() in doc_gender.lower() or doc_gender.lower() in prof_gender.lower():
            matches.append({
                "field": "Owner Gender",
                "expected": prof_gender,
                "found": doc_gender
            })
        else:
            mismatches.append({
                "field": "Owner Gender",
                "expected": prof_gender,
                "found": doc_gender,
                "reason": "Gender on document does not match entered profile."
            })

    # Recalculate missing items if any
    schemes = load_schemes()
    selected_scheme = next((s for s in schemes if s["id"] == schemeId), None)
    if selected_scheme:
        rules = selected_scheme.get("rules", {})
        if rules.get("requires_udyam") and not doc_udyam and prof_udyam_status != "Registered":
            missing.append("Udyam Certificate Registration number was not found.")
        if rules.get("requires_gst") and not doc_gstin and prof_gst_status != "Registered":
            missing.append("GSTIN number was not found.")

    return {
        "success": True,
        "document_type_detected": doc_type,
        "extracted_data": extracted_data,
        "matches": matches,
        "mismatches": mismatches,
        "missing": missing
    }

# --- ADMIN GOVERNMENT SCHEMES ROUTES ---

@app.get("/api/admin/schemes")
def admin_get_schemes(user_info: dict = Depends(get_current_user_workspace_info)):
    return load_schemes()

@app.post("/api/admin/schemes")
def admin_add_scheme(req: SchemeModel, user_info: dict = Depends(get_current_user_workspace_info)):
    schemes = load_schemes()
    if any(s["id"] == req.id for s in schemes):
        raise HTTPException(status_code=400, detail=f"Scheme with ID '{req.id}' already exists.")
        
    schemes.append(req.dict())
    save_schemes(schemes)
    return {"success": True, "message": f"Scheme '{req.name_en}' added successfully."}

@app.put("/api/admin/schemes/{scheme_id}")
def admin_update_scheme(scheme_id: str, req: SchemeModel, user_info: dict = Depends(get_current_user_workspace_info)):
    schemes = load_schemes()
    idx = -1
    for i, s in enumerate(schemes):
        if s["id"] == scheme_id:
            idx = i
            break
            
    if idx == -1:
        raise HTTPException(status_code=404, detail="Scheme not found.")
        
    schemes[idx] = req.dict()
    save_schemes(schemes)
    return {"success": True, "message": f"Scheme '{req.name_en}' updated successfully."}

@app.delete("/api/admin/schemes/{scheme_id}")
def admin_delete_scheme(scheme_id: str, user_info: dict = Depends(get_current_user_workspace_info)):
    schemes = load_schemes()
    new_schemes = [s for s in schemes if s["id"] != scheme_id]
    if len(new_schemes) == len(schemes):
        raise HTTPException(status_code=404, detail="Scheme not found.")
        
    save_schemes(new_schemes)
    return {"success": True, "message": f"Scheme deleted successfully."}

# --- WORKSPACE PROTECTED COPILOT ROUTES ---

@app.get("/api/finance/summary")
def get_finance_summary(user_info: dict = Depends(get_current_user_workspace_info)):
    try:
        from forecasting.predictor import TRANSACTIONS_PATH
        tx_path = os.path.join(user_info["workspace_dir"], 'transactions.csv')
        
        if not os.path.exists(tx_path):
            return {
                "total_sales": 0.0,
                "total_expenses": 0.0,
                "net_profit": 0.0,
                "is_empty": True
            }
            
        tx_df = pd.read_csv(tx_path)
        if tx_df.empty:
            return {
                "total_sales": 0.0,
                "total_expenses": 0.0,
                "net_profit": 0.0,
                "is_empty": True
            }
            
        sales = float(tx_df[tx_df['Type'] == 'Sale']['Amount'].sum())
        expenses = float(tx_df[tx_df['Type'] == 'Expense']['Amount'].sum())
        profit = sales - expenses
        
        return {
            "total_sales": sales,
            "total_expenses": expenses,
            "net_profit": profit,
            "is_empty": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(req: ChatRequest, user_info: dict = Depends(get_current_user_workspace_info)):
    context = ""
    if req.use_rag:
        try:
            context = retrieve_context(req.query, k=3, workspace_dir=user_info["workspace_dir"])
        except Exception as e:
            print(f"RAG retrieval failed: {e}")
            context = "Unable to retrieve document context."
            
    try:
        result = coordinate_agents(req.query, context, provider=req.provider, workspace_dir=user_info["workspace_dir"])
        
        if "hello" in req.query.lower() or "hi" in req.query.lower() or "வணக்கம்" in req.query.lower():
            greeting = f"Welcome back, {user_info['fullName']}! It is great to assist you with {user_info['business']['businessName'] if user_info['business'] else 'your business'} today.\n\n"
            result["response"] = greeting + result["response"]
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent coordination error: {str(e)}")

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_upload_dir = os.path.join(user_info["workspace_dir"], "uploads")
    os.makedirs(workspace_upload_dir, exist_ok=True)
    
    file_path = os.path.join(workspace_upload_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        index_result = index_file(file_path, workspace_dir=user_info["workspace_dir"])
        return {"success": True, "message": index_result, "filename": file.filename}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.get("/api/forecast")
def get_forecast(product_id: str = Query(None), days: int = Query(30), user_info: dict = Depends(get_current_user_workspace_info)):
    try:
        results = forecast_sales(product_id=product_id, days_ahead=days, workspace_dir=user_info["workspace_dir"])
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

@app.get("/api/inventory/depletion")
def get_depletion(user_info: dict = Depends(get_current_user_workspace_info)):
    try:
        exhaustion_data = predict_inventory_exhaustion(workspace_dir=user_info["workspace_dir"])
        return exhaustion_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory depletion calculation error: {str(e)}")

@app.post("/api/inventory/add")
def add_inventory_product(req: ProductAddRequest, user_info: dict = Depends(get_current_user_workspace_info)):
    try:
        workspace_dir = user_info["workspace_dir"]
        inv_path = os.path.join(workspace_dir, "inventory.csv")
        
        if os.path.exists(inv_path):
            df = pd.read_csv(inv_path)
        else:
            df = pd.DataFrame(columns=["ProductID", "ProductName", "Category", "StockLevel", "ReorderLevel", "UnitPrice", "RetailPrice", "Supplier"])
            
        if req.ProductID in df["ProductID"].values:
            raise HTTPException(status_code=400, detail=f"Product with ID {req.ProductID} already exists.")
            
        new_row = {
            "ProductID": req.ProductID,
            "ProductName": req.ProductName,
            "Category": req.Category,
            "StockLevel": req.StockLevel,
            "ReorderLevel": req.ReorderLevel,
            "UnitPrice": req.UnitPrice,
            "RetailPrice": req.RetailPrice,
            "Supplier": req.Supplier
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(inv_path, index=False)
        
        # Sync product with supplier catalog in MongoDB
        sync_product_with_supplier(user_info["workspace_id"], req.ProductID, req.UnitPrice, req.Supplier)
        
        return {"success": True, "message": f"Product {req.ProductName} successfully added to inventory."}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add product: {str(e)}")

@app.post("/api/alerts/trigger-low-stock")
def trigger_low_stock_alerts(req: AlertRequest = None, user_info: dict = Depends(get_current_user_workspace_info)):
    try:
        depletion = predict_inventory_exhaustion(workspace_dir=user_info["workspace_dir"])
        low_stock_items = [item for item in depletion if item["Status"] == "Low Stock"]
        
        if not low_stock_items:
            return {"success": True, "alert_sent": False, "message": "All stock levels are currently healthy. No alerts triggered."}
            
        alerts_sent = []
        recipient = (req.custom_recipient if req else None) or (user_info["business"]["merchantWhatsapp"] if user_info["business"] else None) or config.USER_WHATSAPP_NUMBER
        provider = req.provider if req else "gemini"
        
        for item in low_stock_items:
            alert_query = (
                f"Draft a short, urgent low stock WhatsApp alert notification for this product: "
                f"{item['ProductName']} (ID: {item['ProductID']}). "
                f"Current Stock: {item['CurrentStock']} units. Reorder limit: {item['ReorderLevel']}. "
                f"We are selling {item['DailyVelocity']} units per day, and will run out in {item['DaysRemaining']} days. "
                f"Supplier is {item['Supplier']}. Recommend restocking {item['ReorderRecommendation']} units."
            )
            
            context = f"Product details: {str(item)}"
            agent_result = coordinate_agents(alert_query, context, provider=provider, workspace_dir=user_info["workspace_dir"])
            alert_draft = agent_result["response"]
            
            send_result = send_whatsapp_message(alert_draft, to_number=recipient, workspace_dir=user_info["workspace_dir"])
            alerts_sent.append(send_result)
            
        return {
            "success": True,
            "alert_sent": True,
            "count": len(alerts_sent),
            "alerts": alerts_sent
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert dispatch error: {str(e)}")

@app.get("/api/whatsapp/logs")
def get_message_logs(user_info: dict = Depends(get_current_user_workspace_info)):
    return get_whatsapp_logs(workspace_dir=user_info["workspace_dir"])

@app.post("/api/whatsapp/clear")
def clear_message_logs(user_info: dict = Depends(get_current_user_workspace_info)):
    clear_whatsapp_logs(workspace_dir=user_info["workspace_dir"])
    return {"success": True, "message": "Logs cleared."}

# --- NEW SUPPLIER & PROCUREMENT DASHBOARD ROUTES ---

@app.get("/api/suppliers")
def get_suppliers(user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_id = user_info["workspace_id"]
    initialize_default_suppliers(workspace_id)
    try:
        suppliers = list(db.suppliers.find({"workspace_id": workspace_id}, {"_id": 0}))
        return suppliers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read suppliers: {str(e)}")

@app.post("/api/suppliers/add")
def add_supplier(req: SupplierAddRequest, user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_id = user_info["workspace_id"]
    try:
        # Check duplicate
        existing = db.suppliers.find_one(
            {"workspace_id": workspace_id, "name": {"$regex": f"^{req.name}$", "$options": "i"}}
        )
        if existing:
            raise HTTPException(status_code=400, detail="Supplier name already exists.")

        count = db.suppliers.count_documents({"workspace_id": workspace_id})
        sup_id = f"S{100 + count + 1}"
        new_sup = {
            "workspace_id": workspace_id,
            "id": sup_id,
            "name": req.name,
            "phone": req.phone,
            "email": req.email,
            "reliability": 85,
            "payment_terms": req.paymentTerms,
            "catalog": []
        }
        db.suppliers.insert_one(new_sup)
        return {"success": True, "message": f"Supplier {req.name} added successfully."}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add supplier: {str(e)}")

@app.get("/api/procurement/recommendations")
def get_procurement_recommendations(user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_dir = user_info["workspace_dir"]
    
    # 1. Fetch inventory stock alerts
    depletion = predict_inventory_exhaustion(workspace_dir)
    low_stock = [item for item in depletion if item["Status"] in ["Low Stock", "Approaching Outage"]]
    
    if not low_stock:
        return {"recommendations": []}
        
    # Load suppliers from MongoDB
    workspace_id = user_info["workspace_id"]
    suppliers = list(db.suppliers.find({"workspace_id": workspace_id}, {"_id": 0}))
    if not suppliers:
        initialize_default_suppliers(workspace_id)
        suppliers = list(db.suppliers.find({"workspace_id": workspace_id}, {"_id": 0}))
        
    recommendations = []
    
    for item in low_stock:
        p_id = item["ProductID"]
        p_name = item["ProductName"]
        c_stock = item["CurrentStock"]
        r_level = item["ReorderLevel"]
        days_left = item["DaysRemaining"]
        velocity = item["DailyVelocity"]
        if velocity <= 0:
            velocity = 0.4
            
        # Reorder quantity to cover 30 days of sales
        qty_needed = max(5, int(math.ceil(velocity * 30 + r_level - c_stock)))
        
        # Check capable suppliers
        capable = []
        for s in suppliers:
            for catalog_item in s.get("catalog", []):
                if catalog_item["product_id"] == p_id:
                    capable.append({
                        "supplier_id": s["id"],
                        "supplier_name": s["name"],
                        "unit_price": catalog_item["unit_price"],
                        "min_order_qty": catalog_item["min_order_qty"],
                        "lead_time_days": catalog_item["lead_time_days"],
                        "reliability": s["reliability"]
                    })
                    break
                    
        if not capable:
            continue
            
        # Compute scores
        min_price = min(x["unit_price"] for x in capable)
        
        for c in capable:
            base_score = 100.0
            
            # Price Penalty
            price_penalty = 0.0
            if min_price > 0:
                price_diff_percent = (c["unit_price"] - min_price) / min_price
                price_penalty = 30.0 * price_diff_percent
                
            # Lead Time Penalty: Outage risk check
            lead_time_penalty = 0.0
            if c["lead_time_days"] > days_left:
                lead_time_penalty = 40.0 # High penalty for delivery delay past stockout
                
            c["procurement_score"] = round(max(0.0, (base_score - price_penalty - lead_time_penalty) * (c["reliability"] / 100.0)), 1)
            
        # Sort capable suppliers
        capable.sort(key=lambda x: x["procurement_score"], reverse=True)
        recommended = capable[0]
        
        # Determine quantity based on MOQ
        rec_qty = max(qty_needed, recommended["min_order_qty"])
        rec_price = recommended["unit_price"]
        expected_cost = rec_qty * rec_price
        reorder_span_days = round(rec_qty / velocity, 1)
        
        # Build comparison explanation text
        explanation_lines = []
        explanation_lines.append(f"Recommended Supplier: {recommended['supplier_name']} (Procurement Score: {recommended['procurement_score']}%).")
        explanation_lines.append(f"Order {rec_qty} units at Rs. {rec_price}/unit (Est. cost: Rs. {expected_cost:,.2f}), which will replenish your stock for approx {reorder_span_days} days.")
        
        if len(capable) > 1:
            alternatives = [x for x in capable if x["supplier_id"] != recommended["supplier_id"]]
            alt = alternatives[0]
            if alt["unit_price"] < recommended["unit_price"]:
                explanation_lines.append(f"Tradeoff: {alt['supplier_name']} offers a lower price (Rs. {alt['unit_price']} vs Rs. {recommended['unit_price']}), but their lead time is {alt['lead_time_days']} days. This is past your predicted {days_left} days depletion horizon, making them too slow to prevent a stockout.")
            else:
                explanation_lines.append(f"Comparison: Alternative supplier {alt['supplier_name']} scored lower ({alt['procurement_score']}%) due to higher unit pricing (Rs. {alt['unit_price']} vs Rs. {recommended['unit_price']}) or lower reliability ({alt['reliability']}%).")
                
        recommendations.append({
            "product_id": p_id,
            "product_name": p_name,
            "current_stock": c_stock,
            "reorder_level": r_level,
            "days_remaining": days_left,
            "recommended_supplier_id": recommended["supplier_id"],
            "recommended_supplier_name": recommended["supplier_name"],
            "recommended_quantity": rec_qty,
            "recommended_unit_price": rec_price,
            "expected_cost": expected_cost,
            "days_reorder_will_last": reorder_span_days,
            "capable_suppliers": capable,
            "reasoning": " ".join(explanation_lines)
        })
        
    return {"recommendations": recommendations}

@app.get("/api/procurement/orders")
def get_procurement_orders(user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_id = user_info["workspace_id"]
    try:
        orders = list(db.purchase_orders.find({"workspace_id": workspace_id}, {"_id": 0}))
        return orders
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read POs: {str(e)}")

@app.post("/api/procurement/orders/approve")
def approve_recommendation(req: POApproveRequest, user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_id = user_info["workspace_id"]
    workspace_dir = user_info["workspace_dir"]
    inv_path = os.path.join(workspace_dir, "inventory.csv")

    try:
        # Load supplier from MongoDB
        supplier = db.suppliers.find_one(
            {"workspace_id": workspace_id, "id": req.supplierId}, {"_id": 0}
        )
        if not supplier:
            raise HTTPException(status_code=400, detail="Supplier not found.")

        catalog_item = next(
            (item for item in supplier.get("catalog", []) if item["product_id"] == req.productId), None
        )
        if not catalog_item:
            raise HTTPException(status_code=400, detail="Supplier does not supply this product.")

        # Read inventory for product name
        df_inv = pd.read_csv(inv_path)
        prod_row = df_inv[df_inv["ProductID"] == req.productId]
        if prod_row.empty:
            raise HTTPException(status_code=400, detail="Product not found in inventory.")
        p_name = prod_row["ProductName"].values[0]
        p_cat = prod_row["Category"].values[0]

        # Build PO
        po_count = db.purchase_orders.count_documents({"workspace_id": workspace_id})
        po_id = f"PO_{1000 + po_count + 1}"
        expected_delivery = (datetime.now() + timedelta(days=catalog_item["lead_time_days"])).strftime("%Y-%m-%d")
        total_amount = req.quantity * catalog_item["unit_price"]

        new_po = {
            "workspace_id": workspace_id,
            "po_id": po_id,
            "supplier_id": req.supplierId,
            "supplier_name": supplier["name"],
            "date_created": datetime.now().strftime("%Y-%m-%d"),
            "status": "Pending",
            "items": [
                {
                    "product_id": req.productId,
                    "product_name": p_name,
                    "quantity": req.quantity,
                    "unit_price": catalog_item["unit_price"]
                }
            ],
            "total_amount": total_amount,
            "expected_delivery": expected_delivery,
            "actual_delivery": None,
            "fulfillment_score": None
        }
        db.purchase_orders.insert_one(new_po)
        # Remove MongoDB-injected _id before returning
        new_po.pop("_id", None)

        # Generate WhatsApp draft via agent
        alert_query = (
            f"Draft a professional WhatsApp purchase order message to {supplier['name']}. "
            f"We want to order {req.quantity} units of {p_name} at Rs. {catalog_item['unit_price']} each. "
            f"Total amount is Rs. {total_amount}. Delivery requested by {expected_delivery}. "
            f"Our business is {user_info['business']['businessName']}."
        )
        context = f"Supplier: {supplier['name']}, Phone: {supplier['phone']}, Terms: {supplier['payment_terms']}"
        agent_result = coordinate_agents(alert_query, context, provider="gemini", workspace_dir=workspace_dir)
        whatsapp_draft = agent_result["response"]

        log_whatsapp_message(
            to_number=supplier["phone"],
            body=f"Draft PO order text generated for {supplier['name']}: \n{whatsapp_draft}",
            status="Draft Prepared",
            workspace_dir=workspace_dir
        )

        return {
            "success": True,
            "message": f"Purchase Order {po_id} created successfully.",
            "po": {k: v for k, v in new_po.items() if k != "workspace_id"},
            "whatsapp_draft": whatsapp_draft
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve PO: {str(e)}")

@app.post("/api/procurement/orders/receive")
def receive_purchase_order(req: POReceiveRequest, user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_id = user_info["workspace_id"]
    workspace_dir = user_info["workspace_dir"]
    inv_path = os.path.join(workspace_dir, "inventory.csv")
    tx_path = os.path.join(workspace_dir, "transactions.csv")

    try:
        # Load PO from MongoDB
        po = db.purchase_orders.find_one(
            {"workspace_id": workspace_id, "po_id": req.poId}, {"_id": 0}
        )
        if not po:
            raise HTTPException(status_code=404, detail="Purchase Order not found.")

        if po["status"] == "Delivered":
            return {"success": True, "message": "Purchase Order already marked as delivered."}

        # Calculate fulfillment score
        date_created = datetime.strptime(po["date_created"], "%Y-%m-%d")
        expected_delivery = datetime.strptime(po["expected_delivery"], "%Y-%m-%d")
        actual_delivery = datetime.now()
        expected_days = (expected_delivery - date_created).days
        actual_days = (actual_delivery - date_created).days

        if actual_days <= expected_days:
            fulfillment_score = 100
        else:
            delay = actual_days - expected_days
            fulfillment_score = max(50, 100 - (delay * 10))

        # Update PO in MongoDB
        db.purchase_orders.update_one(
            {"workspace_id": workspace_id, "po_id": req.poId},
            {"$set": {
                "status": "Delivered",
                "actual_delivery": actual_delivery.strftime("%Y-%m-%d"),
                "fulfillment_score": fulfillment_score
            }}
        )

        # Update supplier reliability in MongoDB
        supplier = db.suppliers.find_one(
            {"workspace_id": workspace_id, "id": po["supplier_id"]}
        )
        if supplier:
            old_r = supplier["reliability"]
            new_r = int(round(old_r * 0.7 + fulfillment_score * 0.3))
            db.suppliers.update_one(
                {"workspace_id": workspace_id, "id": po["supplier_id"]},
                {"$set": {"reliability": max(50, min(100, new_r))}}
            )

        # Update inventory CSV stock levels
        df_inv = pd.read_csv(inv_path)
        for item in po["items"]:
            prod_id = item["product_id"]
            qty = item["quantity"]
            df_inv.loc[df_inv["ProductID"] == prod_id, "StockLevel"] += qty
        df_inv.to_csv(inv_path, index=False)

        # Log restocking as Expense in transactions CSV
        if os.path.exists(tx_path):
            df_tx = pd.read_csv(tx_path)
        else:
            df_tx = pd.DataFrame(columns=["Date", "TransactionID", "ProductID", "ProductName", "Category", "Quantity", "Price", "Type", "Amount", "PaymentMode"])

        for item in po["items"]:
            prod_id = item["product_id"]
            p_name = item["product_name"]
            qty = item["quantity"]
            price = item["unit_price"]
            prod_info = df_inv[df_inv["ProductID"] == prod_id]
            p_cat = prod_info["Category"].values[0] if not prod_info.empty else "Operations"
            new_row = {
                "Date": actual_delivery.strftime("%Y-%m-%d"),
                "TransactionID": po["po_id"],
                "ProductID": prod_id,
                "ProductName": f"Restock - {p_name}",
                "Category": p_cat,
                "Quantity": qty,
                "Price": price,
                "Type": "Expense",
                "Amount": qty * price,
                "PaymentMode": "UPI"
            }
            df_tx = pd.concat([df_tx, pd.DataFrame([new_row])], ignore_index=True)

        df_tx.to_csv(tx_path, index=False)

        return {"success": True, "message": f"Purchase Order {po['po_id']} delivered. Inventory restocked and recorded."}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to receive PO: {str(e)}")

# --- UNPROTECTED INTEGRATION INBOUND WEBHOOK ---

@app.post("/api/whatsapp/webhook")
async def twilio_webhook(From: str = Form(...), Body: str = Form(...)):
    clean_sender = From.replace("whatsapp:", "").strip()
    print(f"Incoming WhatsApp message from {clean_sender}: '{Body}'")
    
    # Resolve workspace by matching sender number to merchant mobile (MongoDB)
    workspace_dir = None
    matched_workspace_id = None

    # 1. Match users by mobile number
    clean_mob = clean_sender.replace(" ", "").replace("-", "")
    user_doc = db.users.find_one(
        {"mobile": {"$regex": clean_mob[-10:]}}, {"_id": 0}
    )
    if user_doc:
        matched_workspace_id = user_doc.get("workspace_id")

    # 2. Match by merchantWhatsapp in onboarding collection
    if not matched_workspace_id:
        onb_doc = db.onboarding.find_one(
            {"merchantWhatsapp": {"$regex": clean_mob[-10:]}}, {"_id": 0}
        )
        if onb_doc:
            matched_workspace_id = onb_doc.get("workspace_id")
                    
    if matched_workspace_id:
        workspace_dir = os.path.join(config.WORKSPACES_DIR, matched_workspace_id)
        
    # 2. Retrieve context
    context = ""
    try:
        context = retrieve_context(Body, k=3, workspace_dir=workspace_dir)
    except Exception as e:
        print(f"Webhook RAG failed: {e}")
        
    # 3. Query Agent
    try:
        agent_result = coordinate_agents(Body, context, workspace_dir=workspace_dir)
        reply_body = agent_result["response"]
    except Exception as e:
        reply_body = f"Sorry, I encountered an error processing your query: {str(e)}"
        
    # Log messages inside active workspace
    log_whatsapp_message(From, f"Merchant asked: {Body}", "Received", workspace_dir)
    send_whatsapp_message(reply_body, to_number=From, workspace_dir=workspace_dir)
    
    # Return TwiML response to Twilio
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Message>{reply_body}</Message>
    </Response>"""
    
    return Response(content=twiml_response, media_type="application/xml")

# --- NEW CUSTOMER INSIGHTS ROUTES ---

@app.get("/api/customers")
def get_customers(user_info: dict = Depends(get_current_user_workspace_info)):
    try:
        from forecasting.customer_analytics import get_customer_insights
        insights = get_customer_insights(user_info["workspace_dir"])
        return insights["customers"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch customers: {str(e)}")

@app.post("/api/customers/add")
def add_customer(req: CustomerAddRequest, user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_id = user_info["workspace_id"]
    workspace_dir = user_info.get("workspace_dir")

    try:
        req_email = req.email.strip().lower()
        req_phone = req.phone.strip()

        # Load existing customers from customers.json
        customers_json_path = os.path.join(workspace_dir, "customers.json") if workspace_dir else ""
        c_list = []
        if customers_json_path and os.path.exists(customers_json_path):
            try:
                with open(customers_json_path, "r", encoding="utf-8") as f:
                    c_list = json.load(f)
            except Exception:
                c_list = []

        # Load existing customers from MongoDB
        mongo_custs = []
        try:
            mongo_custs = list(db.customers.find({"workspace_id": workspace_id}, {"_id": 0}))
        except Exception as dbe:
            print(f"MongoDB query warning: {dbe}")

        all_existing = c_list + mongo_custs

        # Check duplicate
        for c in all_existing:
            if (c.get("email") and c.get("email").strip().lower() == req_email) or \
               (c.get("phone") and c.get("phone").strip() == req_phone):
                raise HTTPException(status_code=400, detail="Customer with this email or phone already exists.")

        # Determine non-conflicting unique ID (e.g. C111, C112, C113...)
        max_num = 100
        for c in all_existing:
            cid = str(c.get("id", ""))
            if cid.startswith("C") and cid[1:].isdigit():
                max_num = max(max_num, int(cid[1:]))
        new_id = f"C{max_num + 1}"

        new_cust = {
            "workspace_id": workspace_id,
            "id": new_id,
            "name": req.name.strip(),
            "email": req_email,
            "phone": req_phone
        }

        try:
            db.customers.insert_one(new_cust)
        except Exception as dbe:
            print(f"MongoDB insert error: {dbe}")

        new_cust.pop("_id", None)
        return_cust = {k: v for k, v in new_cust.items() if k != "workspace_id"}

        # Sync to workspace customers.json so customer registry & analytics reflect immediately
        if customers_json_path:
            try:
                c_list.append(return_cust)
                with open(customers_json_path, "w", encoding="utf-8") as f:
                    json.dump(c_list, f, indent=4)
            except Exception as sync_err:
                print(f"Failed syncing customer to customers.json: {sync_err}")

        return {"success": True, "message": f"Customer {req.name} added successfully.", "customer": return_cust}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add customer: {str(e)}")

@app.get("/api/customer/insights")
def get_customer_insights_dashboard(inactive_days: int = 60, user_info: dict = Depends(get_current_user_workspace_info)):
    try:
        from forecasting.customer_analytics import get_customer_insights
        insights = get_customer_insights(user_info["workspace_dir"], config_inactive_days=inactive_days)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Customer analytics error: {str(e)}")

@app.post("/api/customer/campaign")
def generate_customer_campaign(req: CampaignRequest, user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_dir = user_info["workspace_dir"]
    try:
        from forecasting.customer_analytics import get_customer_insights
        insights = get_customer_insights(workspace_dir)
        
        target_description = ""
        discount = req.custom_offer or "10% off checkout"
        
        # Check specific customer or segment targeting
        customer = None
        if req.customer_id:
            customer = next((c for c in insights["customers"] if c["id"] == req.customer_id), None)
            if not customer:
                raise HTTPException(status_code=404, detail="Customer not found")
            target_description = f"customer {customer['name']} (phone: {customer['phone']}) who is in the {customer['segment']} segment and prefers {', '.join(customer['pref_products'] or ['Grains'])}"
        elif req.segment:
            target_description = f"all customers in the '{req.segment}' segment"
        else:
            target_description = "our repeat customers"
            
        query = (
            f"Draft a personalized marketing WhatsApp message campaign targeting {target_description}. "
            f"Provide them a special discount of: {discount}. Keep it brief, friendly, with relevant emojis, "
            f"and make sure it mentions our business: {user_info['business']['businessName'] if user_info['business'] else 'AegisAI Partners'}."
        )
        
        context = f"Company business details: {str(user_info['business'])}"
        result = coordinate_agents(query, context, provider=req.provider, workspace_dir=workspace_dir)
        
        # Log campaign message in the Twilio simulation database
        target_phone = customer["phone"] if customer else "+919999999999"
        log_whatsapp_message(
            to_number=target_phone,
            body=f"Draft campaign for {target_description}:\n\n{result['response']}",
            status="Campaign Drafted",
            workspace_dir=workspace_dir
        )
        
        return {
            "success": True,
            "campaign_draft": result["response"],
            "target": target_description,
            "offer": discount
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate campaign: {str(e)}")

@app.get("/api/strategy/profile")
def get_strategy_profile(user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_dir = user_info["workspace_dir"]
    profile_path = os.path.join(workspace_dir, "business_profile.json")
    
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            return {"success": True, "profile": profile}
        except Exception:
            pass
            
    biz = user_info.get("business") or {}
    fallback_profile = {
        "businessName": biz.get("businessName", "Small Business"),
        "businessType": biz.get("businessCategory", "Small Business"),
        "state": "Local",
        "district": biz.get("businessLocation", "Region"),
        "annualTurnover": 0.0,
        "enterpriseCategory": "Micro",
        "employeeCount": 0,
        "businessSector": biz.get("businessCategory", "Retail"),
        "loanRequirement": 0.0,
        "previousAssistance": "No",
        "socialCategory": "General",
        "ownerGender": "Male",
        "area": "Urban"
    }
    return {"success": True, "profile": fallback_profile}

@app.post("/api/strategy/generate")
def generate_business_strategy(req: StrategyInputsRequest, user_info: dict = Depends(get_current_user_workspace_info)):
    workspace_dir = user_info["workspace_dir"]
    profile_path = os.path.join(workspace_dir, "business_profile.json")
    
    profile = {}
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception:
            pass
            
    if not profile:
        biz = user_info.get("business") or {}
        profile = {
            "businessName": biz.get("businessName", "Small Business"),
            "businessType": biz.get("businessCategory", "Small Business"),
            "businessSector": biz.get("businessCategory", "Retail"),
            "enterpriseCategory": "Micro",
            "state": "Local",
            "district": biz.get("businessLocation", "Region"),
            "annualTurnover": "Undisclosed",
            "employeeCount": "N/A"
        }
    
    try:
        strategy_inputs = {
            "businessType": req.businessType,
            "targetAudience": req.targetAudience,
            "goals": req.goals,
            "competitors": req.competitors
        }
        
        lang = req.language or user_info.get("preferredLanguage") or "english"
        strategy_markdown = generate_strategy(profile, strategy_inputs, language=lang)
        
        return {
            "success": True,
            "strategy": strategy_markdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate strategy: {str(e)}")


async def check_inventory_and_alert_loop():
    """
    Background loop that runs every 6 hours to check all workspaces/users'
    inventories and send WhatsApp alerts for products below their reorder limits.
    """
    print("[BG LOOP] Starting background inventory checker task...")
    await asyncio.sleep(10)  # Wait for application startup to complete
    while True:
        print(f"[BG LOOP] Running check at {datetime.now()}...")
        try:
            users = list(db.users.find({}, {"_id": 0}))
            print(f"[BG LOOP] Loaded {len(users)} users.")
            for user_record in users:
                email = user_record.get("email", "unknown")
                workspace_id = user_record.get("workspace_id")
                if not workspace_id:
                    print(f"[BG LOOP] No workspace ID for user {email}")
                    continue
                workspace_dir = os.path.join(config.WORKSPACES_DIR, workspace_id)
                if not os.path.exists(workspace_dir):
                    print(f"[BG LOOP] Workspace directory {workspace_dir} does not exist for user {email}")
                    continue
                
                print(f"[BG LOOP] Processing workspace {workspace_id} for user {email}")
                
                # Check onboarding to retrieve target number
                recipient = None
                onb_doc = db.onboarding.find_one({"workspace_id": workspace_id}, {"_id": 0})
                if onb_doc:
                    recipient = onb_doc.get("merchantWhatsapp")
                
                # Fallback path prioritizing configured USER_WHATSAPP_NUMBER
                recipient = config.USER_WHATSAPP_NUMBER or recipient or user_record.get("mobile")
                print(f"[BG LOOP] Target recipient phone: {recipient}")
                if not recipient:
                    continue
                
                # Predict inventory velocity and low stock levels
                try:
                    depletion = predict_inventory_exhaustion(workspace_dir=workspace_dir)
                except Exception as e:
                    print(f"Error predicting inventory exhaustion for workspace {workspace_id}: {e}")
                    continue
                
                low_stock_items = [item for item in depletion if item["Status"] == "Low Stock"]
                print(f"[BG LOOP] Found {len(low_stock_items)} low stock items.")
                if not low_stock_items:
                    continue
                # Build one combined alert message locally — no LLM calls
                lines = [f"🚨 *Low Stock Alert* — {user_record.get('full_name', 'Merchant')}\n"]
                for item in low_stock_items:
                    lines.append(
                        f"📦 *{item['ProductName']}* (ID: {item['ProductID']})\n"
                        f"   Stock: {item['CurrentStock']} units | Reorder at: {item['ReorderLevel']}\n"
                        f"   Velocity: {item['DailyVelocity']} u/day | Runs out in: {item['DaysRemaining']} days\n"
                        f"   Supplier: {item['Supplier']} | Restock: {item['ReorderRecommendation']} units"
                    )
                alert_draft = "\n\n".join(lines)
                try:
                    send_whatsapp_message(alert_draft, to_number=recipient, workspace_dir=workspace_dir)
                    print(f"[BG LOOP] Alert sent for {len(low_stock_items)} items in workspace {workspace_id}")
                except Exception as ae:
                    print(f"Failed to send background alert: {ae}")
                await asyncio.sleep(10)  # gap between workspaces
        except Exception as e:
            print(f"Error in background alert scheduler loop: {e}")
            
        await asyncio.sleep(21600)  # Sleep for 6 hours

@app.on_event("startup")
async def startup_event():
    pass  # Background alert loop disabled — alerts are sent manually via UI button

# Run Uvicorn if file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
