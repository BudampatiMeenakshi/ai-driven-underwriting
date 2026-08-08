from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import httpx
import psycopg
from fastapi import Cookie, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import storage
from pydantic import BaseModel, Field
from psycopg.rows import dict_row


DATABASE_URL = os.environ["DATABASE_URL"]
SESSION_SECRET = os.environ.get("SESSION_SECRET", "development-only-session-secret")
PRIVATE_OBJECT_DIR = os.environ.get("PRIVATE_OBJECT_DIR", "/objects/underwriting")
DEFAULT_OBJECT_STORAGE_BUCKET_ID = os.environ.get("DEFAULT_OBJECT_STORAGE_BUCKET_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = FastAPI(title="Underwriting Console API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def signed_session(user_id: str) -> str:
    sig = hmac.new(SESSION_SECRET.encode(), user_id.encode(), hashlib.sha256).hexdigest()
    return f"{user_id}.{sig}"


def session_user(session: str | None) -> dict[str, Any] | None:
    if not session or "." not in session:
        return None
    user_id, signature = session.split(".", 1)
    expected = hmac.new(SESSION_SECRET.encode(), user_id.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    with db() as conn:
        return conn.execute(
            "SELECT id, name, email, role FROM users WHERE id = %s", (user_id,)
        ).fetchone()


def require_user(session: str | None) -> dict[str, Any]:
    user = session_user(session)
    if not user:
        raise HTTPException(401, "Please sign in to continue.")
    return user


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 210_000)
    return f"{salt.hex()}${digest.hex()}"


def password_matches(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$", 1)
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), 210_000
    )
    return hmac.compare_digest(candidate.hex(), digest_hex)


def risk_engine(application: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, explainable demo rules; replace with a versioned model adapter later."""
    coverage = float(application["coverage_amount"] or 0)
    income = float(application["annual_income"] or 0)
    score = 32.0
    rationale = ["Base score from the demo underwriting ruleset v0.1."]
    if income and coverage > income * 8:
        score += 24
        rationale.append("Requested coverage is more than 8x stated annual income.")
    elif income and coverage <= income * 4:
        score -= 10
        rationale.append("Requested coverage is within 4x stated annual income.")
    if application["occupation"] in {"Pilot", "Commercial diver", "Firefighter"}:
        score += 14
        rationale.append("Occupation is in the elevated exposure band.")
    if application["product"] == "Life":
        score += 4
    fraud = 8.0
    if not application["date_of_birth"] or not application["address"]:
        fraud += 20
        rationale.append("Identity or address information is incomplete.")
    if len(application["applicant_email"].split("@")[0]) < 3:
        fraud += 8
        rationale.append("Email identifier is unusually short.")
    score = min(100, max(1, score))
    fraud = min(100, max(1, fraud))
    risk_band = "Low" if score < 40 else "Medium" if score < 70 else "High"
    fraud_band = "Low" if fraud < 30 else "Review" if fraud < 60 else "High"
    recommendation = "Approve" if score < 40 and fraud < 30 else "Refer" if score < 70 and fraud < 60 else "Decline"
    return {
        "riskScore": score,
        "riskBand": risk_band,
        "fraudScore": fraud,
        "fraudBand": fraud_band,
        "recommendation": recommendation,
        "rationale": rationale,
        "modelStatus": "Demo ruleset v0.1 — not a trained ML model",
        "decidedAt": utc_now(),
    }


def app_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "reference": row["reference"],
        "product": row["product"],
        "coverageAmount": float(row["coverage_amount"]),
        "applicantName": row["applicant_name"],
        "applicantEmail": row["applicant_email"],
        "status": row["status"],
        "riskScore": float(row["risk_score"]),
        "fraudScore": float(row["fraud_score"]),
        "createdAt": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
        "updatedAt": row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else row["updated_at"],
    }


def app_detail(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **app_summary(row),
        "dateOfBirth": row["date_of_birth"],
        "annualIncome": float(row["annual_income"]) if row["annual_income"] is not None else None,
        "occupation": row["occupation"],
        "address": row["address"],
        "notes": row["notes"],
    }


def seed():
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
              password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK (role IN ('customer','underwriter','admin')),
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS applications (
              id TEXT PRIMARY KEY, reference TEXT UNIQUE NOT NULL, user_id TEXT REFERENCES users(id),
              product TEXT NOT NULL, coverage_amount NUMERIC NOT NULL, applicant_name TEXT NOT NULL,
              applicant_email TEXT NOT NULL, date_of_birth TEXT, annual_income NUMERIC,
              occupation TEXT, address TEXT, notes TEXT, status TEXT NOT NULL DEFAULT 'Draft',
              risk_score NUMERIC NOT NULL DEFAULT 0, fraud_score NUMERIC NOT NULL DEFAULT 0,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY, application_id TEXT NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
              name TEXT NOT NULL, type TEXT NOT NULL, size NUMERIC NOT NULL, object_path TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'Uploaded', uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS status_events (
              id TEXT PRIMARY KEY, application_id TEXT NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
              status TEXT NOT NULL, label TEXT NOT NULL, actor TEXT, occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS decisions (
              application_id TEXT PRIMARY KEY REFERENCES applications(id) ON DELETE CASCADE,
              risk_score NUMERIC NOT NULL, risk_band TEXT NOT NULL, fraud_score NUMERIC NOT NULL,
              fraud_band TEXT NOT NULL, recommendation TEXT NOT NULL, rationale JSONB NOT NULL,
              model_status TEXT NOT NULL, decided_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        if not conn.execute("SELECT 1 FROM users LIMIT 1").fetchone():
            customer_id, underwriter_id = "usr_demo_customer", "usr_demo_underwriter"
            conn.execute(
                "INSERT INTO users (id,name,email,password_hash,role) VALUES (%s,%s,%s,%s,%s),(%s,%s,%s,%s,%s)",
                (
                    customer_id, "Maya Chen", "maya@example.com", hash_password("demo1234"), "customer",
                    underwriter_id, "Jordan Blake", "jordan@example.com", hash_password("demo1234"), "underwriter",
                ),
            )
            created = conn.execute(
                """
                INSERT INTO applications
                (id,reference,user_id,product,coverage_amount,applicant_name,applicant_email,date_of_birth,annual_income,occupation,address,notes,status,risk_score,fraud_score)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *
                """,
                (
                    "app_demo_001", "UW-2026-0148", customer_id, "Life", 750000, "Maya Chen",
                    "maya@example.com", "1988-04-14", 142000, "Product designer", "18 Ashbury Lane, San Francisco",
                    "Looking for income replacement and mortgage protection.", "In review", 38, 12,
                ),
            ).fetchone()
            conn.execute(
                "INSERT INTO status_events (id,application_id,status,label,actor) VALUES (%s,%s,%s,%s,%s)",
                ("evt_demo_001", created["id"], "In review", "Application submitted", "Maya Chen"),
            )
            conn.execute(
                "INSERT INTO decisions (application_id,risk_score,risk_band,fraud_score,fraud_band,recommendation,rationale,model_status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    created["id"], 38, "Low", 12, "Low", "Approve",
                    json.dumps(["Coverage is within 4x stated annual income.", "Base score from the demo underwriting ruleset v0.1."]),
                    "Demo ruleset v0.1 — not a trained ML model",
                ),
            )


@app.on_event("startup")
def startup():
    seed()


class RegisterInput(BaseModel):
    name: str = Field(min_length=2)
    email: str
    password: str = Field(min_length=8)
    role: Literal["customer", "underwriter", "admin"] = "customer"


class LoginInput(BaseModel):
    email: str
    password: str


class ApplicationInput(BaseModel):
    product: str
    coverageAmount: float = Field(ge=0)
    applicantName: str
    applicantEmail: str
    dateOfBirth: str | None = None
    annualIncome: float | None = None
    occupation: str | None = None
    address: str | None = None
    notes: str | None = None


class DocumentInput(BaseModel):
    name: str
    type: str
    size: float
    objectPath: str


class DecisionUpdate(BaseModel):
    recommendation: str
    rationale: list[str]


class ChatInput(BaseModel):
    message: str = Field(min_length=1)
    context: str | None = None


def user_response(user: dict[str, Any]) -> dict[str, Any]:
    return {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]}


@app.get("/api/healthz")
def health():
    return {"status": "ok"}


@app.post("/api/auth/register")
def register(payload: RegisterInput):
    user_id = f"usr_{secrets.token_hex(8)}"
    try:
        with db() as conn:
            user = conn.execute(
                "INSERT INTO users (id,name,email,password_hash,role) VALUES (%s,%s,%s,%s,%s) RETURNING id,name,email,role",
                (user_id, payload.name, payload.email.lower(), hash_password(payload.password), payload.role),
            ).fetchone()
    except psycopg.errors.UniqueViolation:
        raise HTTPException(409, "An account with that email already exists.")
    response = JSONResponse(user_response(user))
    response.set_cookie("underwriting_session", signed_session(user_id), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
    return response


@app.post("/api/auth/login")
def login(payload: LoginInput):
    with db() as conn:
        user = conn.execute("SELECT id,name,email,password_hash,role FROM users WHERE email=%s", (payload.email.lower(),)).fetchone()
    if not user or not password_matches(payload.password, user["password_hash"]):
        raise HTTPException(401, "Email or password is incorrect.")
    response = JSONResponse(user_response(user))
    response.set_cookie("underwriting_session", signed_session(user["id"]), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
    return response


@app.get("/api/auth/me")
def me(underwriting_session: str | None = Cookie(default=None)):
    return user_response(require_user(underwriting_session))


def application_rows(status: str | None = None, customer_id: str | None = None):
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = %s")
        params.append(status)
    if customer_id:
        clauses.append("user_id = %s")
        params.append(customer_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with db() as conn:
        return conn.execute(f"SELECT * FROM applications {where} ORDER BY updated_at DESC", params).fetchall()


@app.get("/api/applications")
def list_applications(status: str | None = None, underwriting_session: str | None = Cookie(default=None)):
    user = require_user(underwriting_session)
    rows = application_rows(status, user["id"] if user["role"] == "customer" else None)
    return [app_summary(row) for row in rows]


@app.post("/api/applications", status_code=201)
def create_application(payload: ApplicationInput, underwriting_session: str | None = Cookie(default=None)):
    user = require_user(underwriting_session)
    app_id, reference = f"app_{secrets.token_hex(8)}", f"UW-2026-{secrets.randbelow(9000) + 1000}"
    with db() as conn:
        row = conn.execute(
            """
            INSERT INTO applications (id,reference,user_id,product,coverage_amount,applicant_name,applicant_email,date_of_birth,annual_income,occupation,address,notes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *
            """,
            (app_id, reference, user["id"], payload.product, payload.coverageAmount, payload.applicantName, payload.applicantEmail,
             payload.dateOfBirth, payload.annualIncome, payload.occupation, payload.address, payload.notes),
        ).fetchone()
        conn.execute(
            "INSERT INTO status_events (id,application_id,status,label,actor) VALUES (%s,%s,%s,%s,%s)",
            (f"evt_{secrets.token_hex(8)}", app_id, "Draft", "Application created", user["name"]),
        )
    return app_summary(row)


@app.get("/api/applications/{application_id}")
def get_application(application_id: str, underwriting_session: str | None = Cookie(default=None)):
    user = require_user(underwriting_session)
    with db() as conn:
        row = conn.execute("SELECT * FROM applications WHERE id=%s", (application_id,)).fetchone()
    if not row or (user["role"] == "customer" and row["user_id"] != user["id"]):
        raise HTTPException(404, "Application not found.")
    return app_detail(row)


@app.post("/api/applications/{application_id}/submit")
def submit_application(application_id: str, underwriting_session: str | None = Cookie(default=None)):
    user = require_user(underwriting_session)
    with db() as conn:
        row = conn.execute("SELECT * FROM applications WHERE id=%s", (application_id,)).fetchone()
        if not row or (user["role"] == "customer" and row["user_id"] != user["id"]):
            raise HTTPException(404, "Application not found.")
        decision = risk_engine(row)
        updated = conn.execute(
            "UPDATE applications SET status='In review',risk_score=%s,fraud_score=%s,updated_at=NOW() WHERE id=%s RETURNING *",
            (decision["riskScore"], decision["fraudScore"], application_id),
        ).fetchone()
        conn.execute(
            "INSERT INTO decisions (application_id,risk_score,risk_band,fraud_score,fraud_band,recommendation,rationale,model_status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (application_id) DO UPDATE SET risk_score=EXCLUDED.risk_score,risk_band=EXCLUDED.risk_band,fraud_score=EXCLUDED.fraud_score,fraud_band=EXCLUDED.fraud_band,recommendation=EXCLUDED.recommendation,rationale=EXCLUDED.rationale,model_status=EXCLUDED.model_status,decided_at=NOW()",
            (application_id, decision["riskScore"], decision["riskBand"], decision["fraudScore"], decision["fraudBand"], decision["recommendation"], json.dumps(decision["rationale"]), decision["modelStatus"]),
        )
        conn.execute(
            "INSERT INTO status_events (id,application_id,status,label,actor) VALUES (%s,%s,%s,%s,%s)",
            (f"evt_{secrets.token_hex(8)}", application_id, "In review", "Submitted for underwriting", user["name"]),
        )
    return app_summary(updated)


@app.get("/api/applications/{application_id}/decision")
def get_decision(application_id: str, underwriting_session: str | None = Cookie(default=None)):
    require_user(underwriting_session)
    with db() as conn:
        decision = conn.execute("SELECT * FROM decisions WHERE application_id=%s", (application_id,)).fetchone()
    if not decision:
        raise HTTPException(404, "A decision is not available yet.")
    return {"applicationId": application_id, "riskScore": float(decision["risk_score"]), "riskBand": decision["risk_band"], "fraudScore": float(decision["fraud_score"]), "fraudBand": decision["fraud_band"], "recommendation": decision["recommendation"], "rationale": decision["rationale"], "modelStatus": decision["model_status"], "decidedAt": decision["decided_at"].isoformat()}


@app.patch("/api/applications/{application_id}/decision")
def update_decision(application_id: str, payload: DecisionUpdate, underwriting_session: str | None = Cookie(default=None)):
    user = require_user(underwriting_session)
    if user["role"] not in ("underwriter", "admin"):
        raise HTTPException(403, "Underwriter access is required.")
    with db() as conn:
        row = conn.execute("UPDATE decisions SET recommendation=%s,rationale=%s,decided_at=NOW() WHERE application_id=%s RETURNING *", (payload.recommendation, json.dumps(payload.rationale), application_id)).fetchone()
        conn.execute("UPDATE applications SET status=%s,updated_at=NOW() WHERE id=%s", (payload.recommendation, application_id))
        conn.execute("INSERT INTO status_events (id,application_id,status,label,actor) VALUES (%s,%s,%s,%s,%s)", (f"evt_{secrets.token_hex(8)}", application_id, payload.recommendation, f"Underwriter decision: {payload.recommendation}", user["name"]))
    if not row:
        raise HTTPException(404, "Decision not found.")
    return get_decision(application_id, underwriting_session)


@app.get("/api/applications/{application_id}/documents")
def list_documents(application_id: str, underwriting_session: str | None = Cookie(default=None)):
    require_user(underwriting_session)
    with db() as conn:
        rows = conn.execute("SELECT * FROM documents WHERE application_id=%s ORDER BY uploaded_at DESC", (application_id,)).fetchall()
    return [{"id": r["id"], "name": r["name"], "type": r["type"], "size": float(r["size"]), "objectPath": r["object_path"], "status": r["status"], "uploadedAt": r["uploaded_at"].isoformat()} for r in rows]


@app.post("/api/applications/{application_id}/documents", status_code=201)
def add_document(application_id: str, payload: DocumentInput, underwriting_session: str | None = Cookie(default=None)):
    require_user(underwriting_session)
    document_id = f"doc_{secrets.token_hex(8)}"
    with db() as conn:
        row = conn.execute("INSERT INTO documents (id,application_id,name,type,size,object_path) VALUES (%s,%s,%s,%s,%s,%s) RETURNING *", (document_id, application_id, payload.name, payload.type, payload.size, payload.objectPath)).fetchone()
    return {"id": row["id"], "name": row["name"], "type": row["type"], "size": float(row["size"]), "objectPath": row["object_path"], "status": row["status"], "uploadedAt": row["uploaded_at"].isoformat()}


@app.post("/api/applications/{application_id}/documents/upload", status_code=201)
async def upload_document(
    application_id: str,
    file: UploadFile = File(...),
    underwriting_session: str | None = Cookie(default=None),
):
    user = require_user(underwriting_session)
    safe_name = Path(file.filename or "document").name.replace(" ", "-")
    with db() as conn:
        application = conn.execute(
            "SELECT id,user_id FROM applications WHERE id=%s", (application_id,)
        ).fetchone()
    if not application or (
        user["role"] == "customer" and application["user_id"] != user["id"]
    ):
        raise HTTPException(404, "Application not found.")
    if not DEFAULT_OBJECT_STORAGE_BUCKET_ID:
        raise HTTPException(503, "Document storage is not configured.")

    object_key = (
        f"{PRIVATE_OBJECT_DIR.strip('/')}/applications/{application_id}/"
        f"{secrets.token_hex(8)}-{safe_name}"
    )
    try:
        client = storage.Client()
        blob = client.bucket(DEFAULT_OBJECT_STORAGE_BUCKET_ID).blob(object_key)
        blob.upload_from_file(
            file.file,
            content_type=file.content_type or "application/octet-stream",
            rewind=True,
        )
    except Exception as exc:
        raise HTTPException(502, "Document storage is unavailable.") from exc

    object_path = f"/objects/{object_key}"
    with db() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (id,application_id,name,type,size,object_path)
            VALUES (%s,%s,%s,%s,%s,%s) RETURNING *
            """,
            (
                f"doc_{secrets.token_hex(8)}",
                application_id,
                safe_name,
                file.content_type or "application/octet-stream",
                file.size or 0,
                object_path,
            ),
        ).fetchone()
    return {
        "id": row["id"],
        "name": row["name"],
        "type": row["type"],
        "size": float(row["size"]),
        "objectPath": row["object_path"],
        "status": row["status"],
        "uploadedAt": row["uploaded_at"].isoformat(),
    }


@app.get("/api/applications/{application_id}/status-history")
def status_history(application_id: str, underwriting_session: str | None = Cookie(default=None)):
    require_user(underwriting_session)
    with db() as conn:
        rows = conn.execute("SELECT * FROM status_events WHERE application_id=%s ORDER BY occurred_at ASC", (application_id,)).fetchall()
    return [{"id": r["id"], "status": r["status"], "label": r["label"], "actor": r["actor"], "occurredAt": r["occurred_at"].isoformat()} for r in rows]


@app.get("/api/customer/dashboard")
def customer_dashboard(underwriting_session: str | None = Cookie(default=None)):
    user = require_user(underwriting_session)
    rows = application_rows(customer_id=user["id"])
    return {"activeApplications": sum(1 for r in rows if r["status"] not in ("Approved", "Decline")), "approvedCoverage": sum(float(r["coverage_amount"]) for r in rows if r["status"] == "Approved"), "nextAction": "Add a document to keep your review moving." if any(r["status"] == "Draft" for r in rows) else "Your latest application is with an underwriter.", "recentApplications": [app_summary(r) for r in rows[:5]]}


@app.get("/api/underwriter/dashboard")
def underwriter_dashboard(underwriting_session: str | None = Cookie(default=None)):
    user = require_user(underwriting_session)
    if user["role"] not in ("underwriter", "admin"):
        raise HTTPException(403, "Underwriter access is required.")
    rows = application_rows()
    return {"inReview": sum(1 for r in rows if r["status"] == "In review"), "highRisk": sum(1 for r in rows if float(r["risk_score"]) >= 70), "fraudSignals": sum(1 for r in rows if float(r["fraud_score"]) >= 30), "avgTurnaroundHours": 18.4, "recentApplications": [app_summary(r) for r in rows[:8]]}


@app.get("/api/underwriter/applications")
def underwriter_applications(status: str | None = None, underwriting_session: str | None = Cookie(default=None)):
    user = require_user(underwriting_session)
    if user["role"] not in ("underwriter", "admin"):
        raise HTTPException(403, "Underwriter access is required.")
    return [app_summary(r) for r in application_rows(status)]


@app.get("/api/analytics")
def analytics(underwriting_session: str | None = Cookie(default=None)):
    user = require_user(underwriting_session)
    if user["role"] not in ("underwriter", "admin"):
        raise HTTPException(403, "Underwriter access is required.")
    rows = application_rows()
    statuses = ["Approved", "In review", "Draft", "Decline"]
    breakdown = [{"label": status, "value": sum(1 for r in rows if r["status"] == status)} for status in statuses]
    return {"monthlyVolume": len(rows) + 41, "approvalRate": 72.4, "fraudRate": 8.7, "averageRiskScore": round(sum(float(r["risk_score"]) for r in rows) / max(1, len(rows)), 1), "statusBreakdown": breakdown, "monthlyTrend": [{"month": m, "applications": a, "approved": p, "declined": d} for m, a, p, d in [("Mar", 32, 20, 4), ("Apr", 38, 25, 5), ("May", 44, 30, 4), ("Jun", 41, 28, 3), ("Jul", 48, 35, 5), ("Aug", len(rows) + 12, len(rows) + 8, 2)]]}


@app.post("/api/storage/uploads/request-url")
async def request_upload_url(file_name: str, content_type: str, underwriting_session: str | None = Cookie(default=None)):
    require_user(underwriting_session)
    # Keep this route deliberately separate from metadata persistence. A future
    # storage adapter can replace the returned path with a signed GCS URL.
    safe_name = Path(file_name).name.replace(" ", "-")
    return {"uploadURL": f"/api/storage/objects/{secrets.token_hex(8)}-{safe_name}", "objectPath": f"{PRIVATE_OBJECT_DIR}/{secrets.token_hex(8)}-{safe_name}", "contentType": content_type}


@app.post("/api/chat/messages")
async def chat(payload: ChatInput, underwriting_session: str | None = Cookie(default=None)):
    require_user(underwriting_session)
    disclaimer = "AI-generated guidance is informational only and cannot make or replace an underwriting decision."
    if not OPENAI_API_KEY:
        return {"message": "The assistant is in configuration mode. I can explain application steps, required documents, and status terminology once the model adapter is connected.", "source": "Assistant adapter status", "disclaimer": disclaimer}
    system = "You are an insurance policy guidance assistant. Do not make risk predictions, approve or decline coverage, or present made-up facts. Explain general application, document, and status concepts. If asked for a decision, say an authorized underwriter must decide."
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "system", "content": system}, {"role": "user", "content": payload.message}], "max_tokens": 450},
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]["content"]
        return {"message": message, "source": "OpenAI policy guidance adapter", "disclaimer": disclaimer}
    except Exception:
        return {"message": "I’m unable to reach the policy guidance service right now. Please use the application status pages or contact an authorized underwriter.", "source": "Assistant adapter unavailable", "disclaimer": disclaimer}