import os
import re
import json
# OAuth & HTTP
import base64, secrets, hashlib
from urllib.parse import urlencode
import httpx
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from supabase import create_client, Client
from dotenv import load_dotenv
import jwt
import requests
# imports (near the top)
from supabase.client import ClientOptions
import datetime
from fastapi import Request
from fastapi.responses import RedirectResponse
import fitz  # PyMuPDF
# Safe import for different storage3 versions
try:
    from storage3.utils import UploadFileOptions  # v2+
except Exception:  # older SDKs won't have it
    UploadFileOptions = None

# secure hashing (recommended)
# -------- Password hashing (Passlib preferred; safe fallback) --------
try:
    from passlib.context import CryptContext  # pip install passlib[bcrypt]
    pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")

    def _hash_password(p: str) -> str:
        return pwd_context.hash(p)

    def _verify_password(p: str, h: str) -> bool:
        return pwd_context.verify(p, h)

except Exception:
    # Fallback: salted sha256 (only if passlib isn't installed) – less secure
    import os, base64, hashlib

    def _hash_password(p: str) -> str:
        salt = os.urandom(16)
        h = hashlib.sha256(salt + p.encode("utf-8")).digest()
        return "sha256$" + base64.b64encode(salt + h).decode("utf-8")

    def _verify_password(p: str, h: str) -> bool:
        if not h.startswith("sha256$"):
            return False
        raw = base64.b64decode(h.split("sha256$", 1)[1].encode("utf-8"))
        salt, digest = raw[:16], raw[16:]
        return hashlib.sha256(salt + p.encode("utf-8")).digest() == digest

# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------
load_dotenv()

ENV = os.getenv("ENV", "local").lower()

if ENV in ("local", "dev", "development"):
    # local HTTP testing
    COOKIE_SAMESITE = "Lax"    # cookies still sent for top-level GETs
    COOKIE_SECURE = False      # allow http://localhost
else:
    # Render / production (must be HTTPS)
    COOKIE_SAMESITE = "None"   # allow cross-site redirects
    COOKIE_SECURE = True       # required when SameSite=None


app = FastAPI(title="Recruitment AI API", version="1.0.0")

# CORS
origins_env = os.getenv("ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000", "https://recruit-ai-gms.netlify.app","https://recruit-ai-e055.onrender.com","http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI")
FRONTEND_SUCCESS_URL = os.getenv("FRONTEND_SUCCESS_URL", "https://recruit-ai-gms.netlify.app")
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment")

# BEFORE
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# AFTER
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")  # <- add this
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment")
if not SUPABASE_ANON_KEY:
    raise RuntimeError("Missing SUPABASE_ANON_KEY in environment")

admin_sb: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    options=ClientOptions(auto_refresh_token=False, persist_session=False),
)

public_sb: Client = create_client(
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
    options=ClientOptions(auto_refresh_token=False, persist_session=False),
)

# Keep all your DB/storage code working without edits:
supabase: Client = admin_sb

INVITE_REDIRECT = os.getenv("INTERVIEWER_INVITE_REDIRECT")
RESET_REDIRECT  = os.getenv("PASSWORD_RESET_REDIRECT")
# ---------- NEW: storage + TTL settings ----------
RESUME_BUCKET = os.getenv("RESUME_BUCKET", "resumes")
RESUME_TTL_HOURS = int(os.getenv("RESUME_TTL_HOURS", "36"))
# ---------------------------------------------------------------------------
# Ollama (cloud-first)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

security = HTTPBearer()
# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class CompanyRequest(BaseModel):
    company_name: str
    recruiter_name: str
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    description: Optional[str] = None

class JDCreateRequest(BaseModel):
    job_title: str
    min_experience: Optional[float] = None
    max_experience: Optional[float] = None
    company_name: str
    company_description: Optional[str] = None
    employment_type: str
    work_mode: str
    skills_must_have: List[str] = Field(default_factory=list)
    skills_nice_to_have: List[str] = Field(default_factory=list)
    requirements: str
    location: Optional[str] = None

class JDIngestText(BaseModel):
    jd_text: str

class JDIngestAnswer(BaseModel):
    original_jd_text: str
    parsed: Dict[str, Any]
    answers: Dict[str, Any]

class UserIdentity(BaseModel):
    user_id: str
    email: str

class ParseOneRequest(BaseModel):
    storage_key: Optional[str] = None
    file_hash: Optional[str] = None
    email_hint: Optional[str] = None  # optional override

class InterviewerInvite(BaseModel):
    name: str
    email: EmailStr
    is_active: Optional[bool] = True  # default True

class InterviewerUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None

class SendResetOut(BaseModel):
    action_link: str  # helpful in dev; in prod you’ll email it, but still return it for UI fallback

class InterviewerOut(BaseModel):
    interviewer_id: str
    name: str
    email: EmailStr
    company_id: Optional[str] = None
    is_active: bool = True

# ---- Pydantic for interviewer auth ----
class InterviewerSignIn(BaseModel):
    email: EmailStr
    password: str

class InterviewerSessionOut(BaseModel):
    access_token: str
    refresh_token: str
    user: Dict[str, Any]
    interviewer: Dict[str, Any]

class BulkShortlistRequest(BaseModel):
    resume_ids: Optional[List[str]] = None          # shortlist by resume_id(s)
    emails: Optional[List[EmailStr]] = None         # or by candidate email(s)
    only_if_status: str = "PARSED"                  # safety: only update when current status == PARSED

class ScheduleInterviewIn(BaseModel):
    job_id: str
    resume_id: str
    interviewer_id: str
    start_iso: str
    end_iso: str
    timezone: str = "Asia/Kolkata"
    external_id: Optional[str] = None  # for idempotency

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def verify_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except Exception:
        return None

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserIdentity:
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id or not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    return UserIdentity(user_id=user_id, email=email)

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
def _is_cloud_host(base_url: str) -> bool:
    return base_url.startswith("https://ollama.com")

def _normalize_model_tag(model: str, cloud: bool) -> str:
    if cloud and model.endswith("-cloud"):
        return model[:-6]
    return model

def generate_jd_with_ollama(prompt: str) -> str:
    try:
        is_cloud = _is_cloud_host(OLLAMA_BASE_URL)
        model = _normalize_model_tag(OLLAMA_MODEL, is_cloud)

        url = f"{OLLAMA_BASE_URL.rstrip('/')}/generate"
        headers = {"Content-Type": "application/json"}
        if is_cloud:
            if not OLLAMA_API_KEY:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Missing OLLAMA_API_KEY for Ollama Cloud",
                )
            headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

        payload = {"model": model, "prompt": prompt, "stream": False}
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response")
        if not text:
            raise HTTPException(status_code=502, detail="Ollama returned empty response")
        return text
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")

# ---------------------------------------------------------------------------
# JD prompt
# ---------------------------------------------------------------------------
def create_jd_prompt(req: JDCreateRequest) -> str:
    skills_must = ", ".join(req.skills_must_have) if req.skills_must_have else ""
    skills_nice = ", ".join(req.skills_nice_to_have) if req.skills_nice_to_have else ""

    if req.min_experience is not None or req.max_experience is not None:
        min_exp = req.min_experience if req.min_experience is not None else 0
        max_exp = req.max_experience if req.max_experience is not None else "+"
        exp_range = f"{min_exp}-{max_exp} years"
    else:
        exp_range = ""

    work_mode_map = {"online": "Remote", "offline": "Onsite", "hybrid": "Hybrid"}
    work_mode = work_mode_map.get((req.work_mode or "").lower(), req.work_mode or "")

    prompt = f"""
You are an expert technical recruiter and job-description writer. Using the inputs below, write a polished, LinkedIn-ready job description.
Do not invent facts. If something isn’t provided, omit it—never write “TBD” or “put your name here.” Return only the final JD text.

Inputs:
- Job Title: {req.job_title}
- Company: {req.company_name}
- Company Description: {req.company_description or ""}
- Location: {req.location or ""}
- Employment Type: {req.employment_type}
- Work Mode: {work_mode}
- Experience Required: {exp_range}
- Must-Have Skills: {skills_must}
- Nice-to-Have Skills: {skills_nice}
- Additional Requirements: {req.requirements}

Write in a professional, inclusive tone. Use strong action verbs and avoid jargon. Structure the JD exactly as follows (markdown headings + bullets).
If “Company Description” is empty, omit the “About {req.company_name}” section.
Include “Tools & Technologies” only if the skills contain concrete technologies; otherwise omit it.

**{req.job_title} — {req.company_name}**

**About {req.company_name}**
(2–3 concise sentences using Company Description.)

**Role Overview**
(3–5 sentences that weave in Work Mode, Location, Employment Type, and Experience Required.)

**Key Responsibilities**
- 7–9 bullets tailored to the role and Additional Requirements.

**Required Qualifications**
- Bullets that reflect Experience Required and all Must-Have Skills.

**Preferred Qualifications**
- Bullets derived from Nice-to-Have Skills.

**Tools & Technologies** (optional)
- Comma-separated list from the skills above (only if meaningful).

Formatting rules:
- No preamble, no code fences, no instructions—only the final JD.
- Don’t state salary/benefits unless they appear in the inputs; otherwise omit those sections.
- Keep sentences crisp; avoid filler; no placeholders.
""".strip()
    return prompt

# ---------------------------------------------------------------------------
# tiny JD text parser for /jd/ingest-text
# ---------------------------------------------------------------------------
def normalize_work_mode(raw: Optional[str]) -> str:
    """Map a free-text work mode to one of: online | offline | hybrid."""
    if not raw:
        return "online"
    r = raw.strip().lower()

    # common remote patterns
    if any(k in r for k in ["remote", "wfh", "from home", "anywhere"]):
        return "online"

    # common onsite patterns
    if any(k in r for k in ["onsite", "on-site", "office", "in office"]):
        return "offline"

    # hybrid
    if "hybrid" in r:
        return "hybrid"

    # fallback
    return "online"

def normalize_employment_type(raw: Optional[str]) -> str:
    if not raw:
        return "Full-time"
    r = raw.strip().lower()
    if "part" in r:
        return "Part-time"
    if "contract" in r or "freelance" in r:
        return "Contract"
    if "intern" in r or "trainee" in r:
        return "Internship"
    return "Full-time"

def _extract_skills(block: str) -> List[str]:
    if not block:
        return []
    parts = re.split(r"[;,]", block)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # drop leading "Must have", etc
        p = re.sub(r"^(must|nice|good)\s*(to)?\s*have:?","",p,flags=re.I).strip()
        if p:
            out.append(p)
    return out

def _extract_fields_from_jd_text(jd_text: str) -> Dict[str, Any]:
    """Very small heuristic parser for a pasted JD."""
    text = jd_text.strip()

    # title: first sentence or "Senior X at Y"
    title = None
    m = re.search(r"(?i)^(?:we\s+are\s+looking\s+for\s+a|hiring\s+a|role:)?\s*([A-Z][\w\s\/\-\+]{4,50})", text)
    if m:
        title = m.group(1).strip(" .,")

    # company
    company = None
    m = re.search(r"(?i)(?:at|@)\s+([A-Z][\w\s\.\-&]{2,60})", text)
    if m:
        company = m.group(1).strip(" .,")

    # location
    location = None
    m = re.search(r"(?i)location\s*[:\-]\s*([^\n\.]+)", text)
    if m:
        location = m.group(1).strip()

    # employment
    employment = None
    m = re.search(r"(?i)(employment|type)\s*[:\-]\s*([^\n\.]+)", text)
    if m:
        employment = m.group(2).strip()

    # work mode
    work_mode = None
    m = re.search(r"(?i)work\s*mode\s*[:\-]\s*([^\n\.]+)", text)
    if m:
        work_mode = m.group(1).strip()
    else:
        # fallback: see if "Remote" appears anywhere
        if re.search(r"(?i)\bremote\b", text):
            work_mode = "remote"
        elif re.search(r"(?i)\bhybrid\b", text):
            work_mode = "hybrid"
        elif re.search(r"(?i)\b(?:onsite|on-site|office)\b", text):
            work_mode = "onsite"

    # experience
    min_exp = None
    max_exp = None
    m = re.search(r"(?i)(\d+)\+?\s*(?:years|yrs)\s+of\s+experience", text)
    if m:
        min_exp = float(m.group(1))

    # must have / nice to have
    must = []
    nice = []

    m = re.search(r"(?i)must\s+have\s*[:\-]\s*(.+)", text)
    if m:
        must = _extract_skills(m.group(1))

    m = re.search(r"(?i)nice\s+to\s+have\s*[:\-]\s*(.+)", text)
    if m:
        nice = _extract_skills(m.group(1))

    return {
        "job_title": title,
        "company_name": company,
        "location": location,
        "employment_type": employment,
        "work_mode": work_mode,
        "min_experience": min_exp,
        "max_experience": max_exp,
        "skills_must_have": must,
        "skills_nice_to_have": nice,
        "jd_text": text,
    }

def _detect_missing_fields(parsed: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return a list of questions we should ask the recruiter."""
    q = []

    if not parsed.get("job_title"):
        q.append({"field": "job_title", "question": "What's the job title for this posting?"})

    if not parsed.get("company_name"):
        q.append({"field": "company_name", "question": "Which company is this role for?"})

    if not parsed.get("employment_type"):
        q.append({
            "field": "employment_type",
            "question": "Is it Full-time, Part-time, Contract or Internship?",
            "options": ["Full-time", "Part-time", "Contract", "Internship"],
        })

    if not parsed.get("work_mode"):
        q.append({
            "field": "work_mode",
            "question": "What's the work mode? (Remote / Onsite / Hybrid)",
            "options": ["online (remote)", "offline (onsite)", "hybrid"],
        })

    if parsed.get("min_experience") is None:
        q.append({"field": "min_experience", "question": "What is the minimum experience required (in years)?"})

    return q

def _download_from_storage(key: str) -> bytes:
    bucket = supabase.storage.from_(RESUME_BUCKET)
    return bucket.download(key)

def _get_job_owned(job_id: str, user: UserIdentity) -> dict:
    job = (
        supabase.table("jobs")
        .select("*")
        .eq("job_id", job_id)
        .eq("created_by", user.user_id)
        .single()
        .execute()
        .data
    )
    if not job:
        raise HTTPException(404, "Job not found or unauthorized")
    return job

def _get_company_id_for_user(user: UserIdentity) -> Optional[str]:
    rec = (
        supabase.table("recruiters")
        .select("company_id")
        .eq("user_id", user.user_id)
        .limit(1)
        .execute()
        .data
    )
    return rec[0]["company_id"] if rec else None

def _pkce_challenge(verifier: str) -> str:
    dig = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(dig).rstrip(b"=").decode()

def _save_google_tokens(user_id: str, token: dict):
    # upsert token for this user
    supabase.table("google_oauth_tokens").upsert({
        "user_id": user_id,
        "refresh_token": token.get("refresh_token"),
        "access_token": token.get("access_token"),
        "token_expiry": datetime.datetime.utcfromtimestamp(
            datetime.datetime.utcnow().timestamp() + int(token.get("expires_in", 3600))
        ).isoformat() + "Z",
        "scope": token.get("scope"),
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }).execute()

def _get_google_refresh_token(user_id: str) -> str | None:
    rows = (
        supabase.table("google_oauth_tokens")
        .select("refresh_token")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
        .data
    )
    return rows[0]["refresh_token"] if rows else None


def _google_service_for_user(user_id: str):
    rtok = _get_google_refresh_token(user_id)
    if not rtok:
        raise HTTPException(400, "Google is not connected for this account")

    creds = Credentials(
        token=None,
        refresh_token=rtok,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=GOOGLE_SCOPES,
    )
    # optional proactive refresh
    # creds.refresh(GARequest())
    return build("calendar", "v3", credentials=creds, cache_discovery=False)

def _first_row(resp):
    """Works across supabase-py versions: returns the first row or None."""
    if resp is None:
        return None
    data = getattr(resp, "data", None)
    if isinstance(data, list):
        return data[0] if data else None
    return data  # may already be a dict or None

# ---------------------------------------------------------------------------
# routes (YOUR ORIGINAL ROUTES — UNCHANGED)
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Recruitment AI API is running", "status": "active"}

# ---------- auth ----------
@app.post("/auth/signup")
async def signup(request: SignUpRequest):
    try:
        resp = public_sb.auth.sign_up({"email": request.email, "password": request.password})
        if resp.user:
            return {"message": "User created successfully", "user": {"id": resp.user.id, "email": resp.user.email}}
        raise HTTPException(status_code=400, detail="Failed to create user")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/signin")
async def signin(request: SignInRequest):
    try:
        resp = public_sb.auth.sign_in_with_password({"email": request.email, "password": request.password})
        if resp.session:
            return {
                "access_token": resp.session.access_token,
                "refresh_token": resp.session.refresh_token,
                "user": {"id": resp.user.id, "email": resp.user.email},
            }
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password")

# Start OAuth (must be called by a logged-in user)
@app.get("/auth/google/start")
async def google_start(current_user: UserIdentity = Depends(get_current_user)):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not OAUTH_REDIRECT_URI:
        raise HTTPException(500, "Google OAuth not configured")

    # PKCE + state
    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(64)

    # first redirect to our own /auth/google/go, where we build the Google URL
    resp = RedirectResponse(url="/auth/google/go")

    # store state, PKCE verifier, and user id in httpOnly cookies
    resp.set_cookie(
        "g_state",
        state,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
    )
    resp.set_cookie(
        "g_verifier",
        verifier,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
    )
    resp.set_cookie(
        "sb_uid",
        current_user.user_id,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
    )
    return resp

@app.get("/auth/google/go")
def google_go(request: Request):
    state = request.cookies.get("g_state")
    verifier = request.cookies.get("g_verifier")
    if not state or not verifier:
        raise HTTPException(400, "Missing PKCE cookies")

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(GOOGLE_SCOPES),
        "state": state,
        "access_type": "offline",   # get refresh token
        "prompt": "consent",        # ensure refresh token even if previously consented
        "code_challenge": _pkce_challenge(verifier),
        "code_challenge_method": "S256",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(url)

# Google's redirect URI (MUST match in Google console)
@app.get("/oauth2/callback")
async def oauth_callback(request: Request, code: str, state: str):
    st = request.cookies.get("g_state")
    verifier = request.cookies.get("g_verifier")
    user_id = request.cookies.get("sb_uid")
    if not st or st != state or not verifier or not user_id:
        raise HTTPException(400, "Invalid state or session")

    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "code_verifier": verifier,
        "grant_type": "authorization_code",
        "redirect_uri": OAUTH_REDIRECT_URI,
    }
    async with httpx.AsyncClient() as client:
        r = await client.post("https://oauth2.googleapis.com/token", data=data, timeout=30)

    try:
        token = r.json()
    except Exception:
        raise HTTPException(400, f"Token HTTP {r.status_code}: {r.text}")

    if "error" in token:
        # surfaces 'invalid_client', 'redirect_uri_mismatch', etc.
        raise HTTPException(400, f"Google token error: {token.get('error')} - {token.get('error_description')}")

    saved_rt = _get_google_refresh_token(user_id)
    if "refresh_token" not in token:
        if saved_rt:
            token["refresh_token"] = saved_rt
        else:
            raise HTTPException(400, "Missing refresh_token (ask user to check 'consent' and offline access)")

    _save_google_tokens(user_id, token)

    # clean cookies and send back to frontend
    resp = RedirectResponse(url=FRONTEND_SUCCESS_URL)
    resp.delete_cookie("g_state")
    resp.delete_cookie("g_verifier")
    resp.delete_cookie("sb_uid")
    return resp

@app.get("/auth/google/status")
def google_status(current_user: UserIdentity = Depends(get_current_user)):
    return {"connected": bool(_get_google_refresh_token(current_user.user_id))}

# ---------- me ----------
@app.get("/me")
async def get_me(current_user: UserIdentity = Depends(get_current_user)):
    recruiter_resp = (
        supabase.table("recruiters")
        .select("user_id, full_name, company_id, company_email")
        .eq("user_id", current_user.user_id)
        .execute()
    )

    if recruiter_resp.data:
        recruiter = recruiter_resp.data[0]
        company = None
        if recruiter.get("company_id"):
            company_resp = (
                supabase.table("companies")
                .select("company_id, company_name, description, location, linkedin_url")
                .eq("company_id", recruiter["company_id"])
                .execute()
            )
            if company_resp.data:
                company = company_resp.data[0]
        return {
            "user": {"id": current_user.user_id, "email": current_user.email},
            "recruiter": recruiter,
            "company": company,
            "has_company": bool(company),
        }

    return {
        "user": {"id": current_user.user_id, "email": current_user.email},
        "recruiter": None,
        "company": None,
        "has_company": False,
    }

# ---------- company ----------
@app.post("/company/create")
async def create_company(request: CompanyRequest, current_user: UserIdentity = Depends(get_current_user)):
    try:
        existing_company = (
            supabase.table("companies").select("*").eq("company_name", request.company_name).execute()
        )

        if existing_company.data:
            company_id = existing_company.data[0]["company_id"]
        else:
            company_result = supabase.table("companies").insert(
                {
                    "company_name": request.company_name,
                    "location": request.location,
                    "linkedin_url": request.linkedin_url,
                    "description": request.description,
                }
            ).execute()
            company_id = company_result.data[0]["company_id"]

        recruiter_data = {
            "user_id": current_user.user_id,
            "company_email": current_user.email,
            "full_name": request.recruiter_name,
            "company_id": company_id,
        }

        existing_recruiter = (
            supabase.table("recruiters").select("*").eq("user_id", current_user.user_id).execute()
        )
        if existing_recruiter.data:
            recruiter_result = (
                supabase.table("recruiters")
                .update(recruiter_data)
                .eq("user_id", current_user.user_id)
                .execute()
            )
        else:
            recruiter_result = supabase.table("recruiters").insert(recruiter_data).execute()

        return {
            "message": "Company profile created successfully",
            "company_id": company_id,
            "recruiter": recruiter_result.data[0],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating company: {e}")

# ---------- jd/create (UI flow) ----------
@app.post("/jd/create")
async def create_jd(request: JDCreateRequest, current_user: UserIdentity = Depends(get_current_user)):
    try:
        recruiter = (
            supabase.table("recruiters")
            .select("company_id")
            .eq("user_id", current_user.user_id)
            .execute()
        )

        if not recruiter.data:
            raise HTTPException(status_code=400, detail="Please create company profile first")

        company_id = recruiter.data[0]["company_id"]

        prompt = create_jd_prompt(request)
        jd_text = generate_jd_with_ollama(prompt)

        job_data = {
            "company_id": company_id,
            "created_by": current_user.user_id,
            "role": request.job_title,
            "location": request.location,
            "employment_type": request.employment_type,
            "work_mode": request.work_mode,
            "min_years": request.min_experience,
            "max_years": request.max_experience,
            "skills_must_have": json.dumps(request.skills_must_have),
            "skills_nice_to_have": json.dumps(request.skills_nice_to_have),
            "jd_text": jd_text,
            "status": "draft",
        }

        job_result = supabase.table("jobs").insert(job_data).execute()
        return {"message": "JD generated successfully", "job_id": job_result.data[0]["job_id"], "jd_text": jd_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating JD: {e}")

# ---------- jd/regenerate ----------
@app.post("/jd/regenerate/{job_id}")
async def regenerate_jd(job_id: str, current_user: UserIdentity = Depends(get_current_user)):
    try:
        job = (
            supabase.table("jobs")
            .select("*")
            .eq("job_id", job_id)
            .eq("created_by", current_user.user_id)
            .execute()
        )
        if not job.data:
            raise HTTPException(status_code=404, detail="Job not found")

        job_data = job.data[0]

        comp = (
            supabase.table("companies")
            .select("company_name, description")
            .eq("company_id", job_data["company_id"])
            .execute()
        )
        company_name = comp.data[0].get("company_name") if comp.data else ""
        company_desc = comp.data[0].get("description") if comp.data else None

        req_obj = JDCreateRequest(
            job_title=job_data["role"],
            min_experience=job_data.get("min_years"),
            max_experience=job_data.get("max_years"),
            company_name=company_name,
            company_description=company_desc,
            employment_type=job_data["employment_type"],
            work_mode=job_data["work_mode"],
            skills_must_have=json.loads(job_data.get("skills_must_have", "[]")),
            skills_nice_to_have=json.loads(job_data.get("skills_nice_to_have", "[]")),
            requirements="Regenerate based on previous requirements",
            location=job_data.get("location"),
        )

        prompt = create_jd_prompt(req_obj)
        jd_text = generate_jd_with_ollama(prompt)

        supabase.table("jobs").update({"jd_text": jd_text}).eq("job_id", job_id).execute()

        return {"message": "JD regenerated successfully", "job_id": job_id, "jd_text": jd_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error regenerating JD: {e}")

# ---------- jd/edit ----------
@app.put("/jd/edit/{job_id}")
async def edit_jd(job_id: str, jd_text: str, current_user: UserIdentity = Depends(get_current_user)):
    try:
        result = (
            supabase.table("jobs")
            .update({"jd_text": jd_text})
            .eq("job_id", job_id)
            .eq("created_by", current_user.user_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Job not found or unauthorized")
        return {"message": "JD updated successfully", "job_id": job_id, "jd_text": jd_text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating JD: {e}")

# ---------- jobs/my-jobs ----------
@app.get("/jobs/my-jobs")
async def get_my_jobs(current_user: UserIdentity = Depends(get_current_user)):
    try:
        jobs = (
            supabase.table("jobs")
            .select("*")
            .eq("created_by", current_user.user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return {"jobs": jobs.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching jobs: {e}")

# ---------- NEW: recruiter pastes full JD text ----------
@app.post("/jd/ingest-text")
async def ingest_jd_text(
    payload: JDIngestText,
    current_user: UserIdentity = Depends(get_current_user),
):
    """
    Recruiter pastes JD text.
    1. ensure user has recruiter/company
    2. parse JD text
    3. if missing fields -> return questions (no insert)
    4. otherwise -> insert into jobs as draft
    """
    # 1) ensure recruiter & company
    recruiter_resp = (
        supabase.table("recruiters")
        .select("company_id, full_name")
        .eq("user_id", current_user.user_id)
        .execute()
    )
    if not recruiter_resp.data:
        # recruiter hasn't created company yet
        return JSONResponse(
            status_code=200,
            content={
                "status": "needs_company",
                "message": "Please create your company profile first.",
            },
        )

    company_id = recruiter_resp.data[0]["company_id"]

    # 2) parse JD text
    parsed = _extract_fields_from_jd_text(payload.jd_text)

    # 3) detect missing
    questions = _detect_missing_fields(parsed)
    if questions:
        return JSONResponse(
            status_code=200,
            content={
                "status": "needs_input",
                "parsed": parsed,
                "questions": questions,
            },
        )

    # 4) normalize values so they pass DB constraints
    work_mode = normalize_work_mode(parsed.get("work_mode"))
    employment_type = normalize_employment_type(parsed.get("employment_type"))

    # 5) insert
    job_payload = {
        "company_id": company_id,
        "created_by": current_user.user_id,
        "role": parsed.get("job_title") or "Untitled Role",
        "location": parsed.get("location"),
        "employment_type": employment_type,
        "work_mode": work_mode,  # <--- guaranteed online/offline/hybrid now
        "min_years": parsed.get("min_experience"),
        "max_years": parsed.get("max_experience"),
        "skills_must_have": json.dumps(parsed.get("skills_must_have") or []),
        "skills_nice_to_have": json.dumps(parsed.get("skills_nice_to_have") or []),
        "jd_text": parsed.get("jd_text"),
        "status": "draft",
    }

    try:
        job_res = supabase.table("jobs").insert(job_payload).execute()
    except Exception as e:
        # If for some reason DB still rejects, tell frontend what we tried
        raise HTTPException(
            status_code=500,
            detail=f"Could not insert JD into jobs: {e}",
        )

    job_id = job_res.data[0]["job_id"]
    return {
        "status": "ok",
        "job_id": job_id,
        "job": job_res.data[0],
    }

@app.post("/jd/finalize/{job_id}")
async def finalize_jd(job_id: str, current_user: UserIdentity = Depends(get_current_user)):
    try:
        result = (
            supabase.table("jobs")
            .update({"status": "published"})
            .eq("job_id", job_id)
            .eq("created_by", current_user.user_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Job not found or unauthorized")
        return {"message": "JD finalized and published", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error finalizing JD: {e}")

@app.post("/jd/ingest-answer")
async def ingest_answer(
    payload: JDIngestAnswer,
    current_user: UserIdentity = Depends(get_current_user),
):
    # still need a company
    recruiter_resp = (
        supabase.table("recruiters")
        .select("company_id")
        .eq("user_id", current_user.user_id)
        .execute()
    )
    if not recruiter_resp.data:
        return {
            "status": "needs_company",
            "message": "Please create your company profile first.",
        }

    company_id = recruiter_resp.data[0]["company_id"]

    parsed = dict(payload.parsed or {})
    # merge answers
    for k, v in (payload.answers or {}).items():
        parsed[k] = v

    # re-check missing
    missing = _detect_missing_fields(parsed)
    if missing:
        return {
            "status": "needs_input",
            "parsed": parsed,
            "questions": missing,
        }

    # normalize
    work_mode = normalize_work_mode(parsed.get("work_mode"))
    employment_type = normalize_employment_type(parsed.get("employment_type"))

    job_payload = {
        "company_id": company_id,
        "created_by": current_user.user_id,
        "role": parsed.get("job_title") or "Untitled Role",
        "location": parsed.get("location"),
        "employment_type": employment_type,
        "work_mode": work_mode,
        "min_years": parsed.get("min_experience"),
        "max_years": parsed.get("max_experience"),
        "skills_must_have": json.dumps(parsed.get("skills_must_have") or []),
        "skills_nice_to_have": json.dumps(parsed.get("skills_nice_to_have") or []),
        "jd_text": parsed.get("jd_text") or payload.original_jd_text,
        "status": "draft",
    }

    job_res = supabase.table("jobs").insert(job_payload).execute()
    return {
        "status": "ok",
        "job_id": job_res.data[0]["job_id"],
        "job": job_res.data[0],
    }

# ---------------------------------------------------------------------------
# ----------------------- NEW: RESUME PIPELINE ------------------------------
# ---------------------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _extract_text_pdf(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join([p.get_text("text") for p in doc])

def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9+\-.#]", "", (s or "").lower())

def _canonicalize(sk: str) -> str:
    m = {
        "reactjs": "react", "react.js": "react",
        "nodejs": "node", "node.js": "node",
        "ts": "typescript", "py": "python",
        "c++": "cpp", "c#": "csharp",
    }
    x = _norm_token(sk)
    return m.get(x, x)

def _skill_groups(must: List[str], nice: List[str]) -> Dict[str, set]:
    groups: Dict[str, set] = {}
    for s in (must or []) + (nice or []):
        if not s:
            continue
        can = _canonicalize(s)
        if not can:
            continue
        groups.setdefault(can, set()).add(can)
    if "javascript" in groups:
        groups["javascript"].update({"js"})
    if "html" in groups:
        groups["html"].update({"html5"})
    if "css" in groups:
        groups["css"].update({"css3"})
    return groups

def _flatten_resume_tokens(text: str) -> set:
    toks = {_canonicalize(t) for t in re.findall(r"[A-Za-z0-9+.#\-]{2,}", text or "")}
    return {t for t in toks if t}

def _years_from_text(text: str) -> float:
    y = 0.0
    for m in re.finditer(r"(\d{4})\s*[-–]\s*(\d{4}|present|current)", (text or "").lower()):
        a, b = m.group(1), m.group(2)
        try:
            aa = int(a)
            bb = datetime.datetime.utcnow().year if ("present" in b or "current" in b) else int(b)
            if bb >= aa:
                y += (bb - aa)
        except Exception:
            pass
    return round(y, 2)

def _experience_match(actual: float, miny: Optional[float], maxy: Optional[float]) -> int:
    if actual is None:
        return 0
    if miny is None and maxy is None:
        return 0
    if miny is not None and actual < miny:
        return max(0, round(100 * (actual / max(miny, 0.01))))
    return 100

def _score(text: str, must: List[str], nice: List[str], miny: Optional[float], maxy: Optional[float]) -> Dict[str, Any]:
    groups = _skill_groups(must, nice)
    res_tokens = _flatten_resume_tokens(text)
    matched = [g for g, aliases in groups.items() if res_tokens & aliases]
    total_groups = max(1, len(groups))
    skill_match = round(100.0 * len(matched) / total_groups, 2)
    yoe = _years_from_text(text)
    exp_score = _experience_match(yoe, miny, maxy)
    jd_match = round(0.65 * skill_match + 0.35 * exp_score, 2)
    return {
        "jd_match_score": jd_match,
        "meta": {
            "total_experience_years": yoe,
            "skill_groups_total": total_groups,
            "skill_groups_matched": matched,
            "skill_match_score": skill_match,
            "experience_match_score": exp_score,
            "top_matched_skills": matched[:10],
        }
    }
# ====== LLM helpers for resume parsing & summary (ADD THIS BLOCK) ======

# We’ll just reuse your Ollama caller.
def _ollama_generate(prompt: str) -> str:
    return generate_jd_with_ollama(prompt)

# Prompts
ALLOWED_RESUME_STATUSES = {"PENDING", "PARSED", "REJECTED","SHORTLISTED"}

EXTRACT_PROMPT = (
    "You are a resume parsing assistant. Return ONLY JSON with these exact top-level keys: "
    "first_name, last_name, full_name, email, phone, location, links, skills, experience, education, projects, certifications.\n"
    "Formatting rules:\n"
    "- links: object; optional keys: linkedin, github, portfolio, other (strings).\n"
    "- skills: object; keys: languages, frameworks, databases, tools, soft_skills (arrays of strings).\n"
    "- experience: array of objects with keys: company, title, start_date, end_date, description, technologies (array).\n"
    "- education: array of objects with keys: degree, institution, start_year, end_year, score.\n"
    "- projects: array of objects with keys: name, description, technologies (array), link, impact.\n"
    "- certifications: array of strings.\n"
    "No comments, no trailing commas.\n"
    "RESUME TEXT:\n{resume_text}"
)
# --- Rich JD↔︎Resume analysis (markdown) ---
ANALYSIS_PROMPT = """
You are a technical recruiter assistant. Compare the JOB DESCRIPTION and the CANDIDATE RESUME and write a concise analysis in Markdown.

Return ONLY Markdown with these sections (use the exact headings):
### Verdict
(one of: Strong Fit, Fit, Borderline, Not a Fit) with a one-sentence rationale.

### Key Matches
- Map must-have requirements to concrete evidence from the resume (role, dates, tech).

### Gaps
- List missing or weak items; mark each as **major** or **minor**.

### Experience Check
- Years parsed: {yoe} vs JD min: {miny} / max: {maxy}. Brief comment.

### Risk Flags
- Any red flags (employment gaps, very short stints, mismatched titles, etc.). If none, write "None noted."

### Scores
- Skill match: {skill_score} / 100
- Experience match: {exp_score} / 100
- Overall JD match: {overall_score} / 100

### Recommendation
- One of: Proceed to phone screen / Proceed to technical screen / Hold for backup / Reject (with reason).

DATA:
JD TITLE: {role}
MUST-HAVES: {must}
NICE-TO-HAVES: {nice}

JOB DESCRIPTION (truncated):
{jd}

CANDIDATE (parsed & truncated):
{cand}

Only use information present above. Be specific, avoid fluff. Keep total under ~250 words.
""".strip()

def generate_resume_analysis(
    candidate: dict,
    meta: dict,
    jd_text: str,
    role: Optional[str],
    must: List[str],
    nice: List[str],
    miny: Optional[float],
    maxy: Optional[float],
) -> str:
    # compact the candidate payload to keep prompt small
    cand_compact = {
        "name": candidate.get("full_name")
                or f"{candidate.get('first_name','')} {candidate.get('last_name','')}".strip(),
        "location": candidate.get("location"),
        "links": candidate.get("links"),
        "skills": candidate.get("skills"),
        "experience": (candidate.get("experience") or [])[:5],
        "education": (candidate.get("education") or [])[:3],
        "projects": (candidate.get("projects") or [])[:3],
        "certifications": candidate.get("certifications") or [],
    }

    overall = round(
        0.65 * float(meta.get("skill_match_score", 0)) +
        0.35 * float(meta.get("experience_match_score", 0)), 2
    )

    prompt = ANALYSIS_PROMPT.format(
        role=role or "Software Engineer",
        must=", ".join(must or []),
        nice=", ".join(nice or []),
        jd=(jd_text or "")[:2000],
        cand=json.dumps(cand_compact, ensure_ascii=False)[:2500],
        yoe=meta.get("total_experience_years", "n/a"),
        miny=miny if miny is not None else "n/a",
        maxy=maxy if maxy is not None else "n/a",
        skill_score=meta.get("skill_match_score", 0),
        exp_score=meta.get("experience_match_score", 0),
        overall_score=overall,
    )
    try:
        return _ollama_generate(prompt).strip()[:3000]
    except Exception:
        return "Analysis unavailable."

# JSON tidying for slightly-invalid LLM JSON
def _json_fragment(s: str) -> str:
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s

def _tidy_json(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)
    return s

# Normalize skills dict to consistent arrays
_DEF_SK_KEYS = ["languages", "frameworks", "databases", "tools", "soft_skills"]

def _normalize_skills_block(sk: dict) -> dict:
    out = {k: [] for k in _DEF_SK_KEYS}
    if isinstance(sk, dict):
        for k in _DEF_SK_KEYS:
            v = sk.get(k)
            if isinstance(v, list):
                out[k] = [str(x).strip() for x in v if str(x).strip()]
            elif isinstance(v, str):
                out[k] = [x.strip() for x in re.split(r",|/|\n", v) if x.strip()]
    return out

def parse_resume_structured(resume_text: str) -> dict:
    """LLM-based extractor → dict. Safe against minor JSON formatting issues."""
    try:
        raw = _ollama_generate(EXTRACT_PROMPT.format(resume_text=resume_text[:8000]))
        frag = _json_fragment(raw)
        try:
            data = json.loads(frag)
        except Exception:
            data = json.loads(_tidy_json(frag))
    except Exception:
        data = {}

    # Ensure shapes
    data = data or {}
    data.setdefault("links", {})
    data.setdefault("skills", {})
    data.setdefault("experience", [])
    data.setdefault("education", [])
    data.setdefault("projects", [])
    data.setdefault("certifications", [])
    data["skills"] = _normalize_skills_block(data.get("skills") or {})
    return data

# ====== end LLM helpers block ======

def _safe_json_list(x) -> List[str]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return []

def _ensure_bucket():
    # idempotent
    try:
        supabase.storage.create_bucket(RESUME_BUCKET, {"public": False})
    except Exception:
        # bucket may already exist
        pass

sb = supabase
def _upload_to_storage(job_id: str, sha: str, filename: str, content: bytes) -> str:
    """
    Upload bytes to Supabase Storage with the correct content-type.
    Works across storage3 versions (uses UploadFileOptions if available).
    """
    _ensure_bucket()  # just in case
    bucket = supabase.storage.from_(RESUME_BUCKET)

    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', filename or 'resume.pdf')
    key = f"{job_id}/{sha}_{safe_name}"

    # Newer storage3 supports UploadFileOptions
    if UploadFileOptions is not None:
        opts = UploadFileOptions(
            content_type="application/pdf",  # IMPORTANT (avoids 'text/plain is not supported')
            cache_control="3600",
            upsert=True,
        )
        bucket.upload(key, content, file_options=opts)  # use keyword arg
    else:
        # Older storage3: pass a headers-like dict via file_options (must be lowercase strings)
        bucket.upload(
            key,
            content,
            file_options={
                "content-type": "application/pdf",
                "cache-control": "3600",
                "x-upsert": "true",
            },
        )

    return key

@app.post("/jobs/{job_id}/parse_upload")
async def parse_upload(
    job_id: str,
    body: ParseOneRequest,
    current_user: UserIdentity = Depends(get_current_user),
):
    job = _get_job_owned(job_id, current_user)

    if not (body.storage_key or body.file_hash):
        raise HTTPException(400, "Provide storage_key or file_hash")

    # resolve storage_key if only file_hash was given
    storage_key = body.storage_key
    filename_hint = None
    if not storage_key:
        row = (
            supabase.table("resume_uploads")
            .select("storage_key, filename")
            .eq("job_id", job_id)
            .eq("file_hash", body.file_hash)
            .limit(1)
            .execute()
            .data
        )
        if not row:
            raise HTTPException(404, "Upload not found for this file_hash")
        storage_key = row[0]["storage_key"]
        filename_hint = row[0].get("filename")

    try:
        data = _download_from_storage(storage_key)
        pdf_text = _extract_text_pdf(data)
    except Exception as e:
        # store error and return
        supabase.table("resume_uploads").update(
            {"last_error": f"download/parse failed: {e}"}
        ).eq("job_id", job_id).eq("storage_key", storage_key).execute()
        raise HTTPException(400, f"Download/parse failed: {e}")

    # infer email from filename if possible
    email_hint = body.email_hint
    if not email_hint:
        fn = filename_hint or storage_key.split("/", 1)[-1]
        m = re.search(r"([\w\.-]+@[\w\.-]+\.\w+)", fn or "")
        if m:
            email_hint = m.group(1)

    _insert_or_update_resume(job, pdf_text, email_hint)

    supabase.table("resume_uploads").update(
        {"parsed_at": datetime.datetime.utcnow().isoformat() + "Z", "last_error": None}
    ).eq("job_id", job_id).eq("storage_key", storage_key).execute()

    return {"status": "ok", "storage_key": storage_key}

@app.post("/jobs/{job_id}/parse_pending")
async def parse_pending(
    job_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: UserIdentity = Depends(get_current_user),
):
    job = _get_job_owned(job_id, current_user)
    # fetch unparsed uploads
    pending = (
        supabase.table("resume_uploads")
        .select("storage_key, filename")
        .eq("job_id", job_id)
        .is_("parsed_at", "null")  # supabase-py supports .is_ for IS NULL
        .limit(limit)
        .execute()
        .data
        or []
    )

    parsed, errors = 0, []

    for row in pending:
        key = row["storage_key"]
        try:
            data = _download_from_storage(key)
            pdf_text = _extract_text_pdf(data)

            # filename-based email hint
            email_hint = None
            m = re.search(r"([\w\.-]+@[\w\.-]+\.\w+)", (row.get("filename") or key) )
            if m:
                email_hint = m.group(1)

            _insert_or_update_resume(job, pdf_text, email_hint)

            supabase.table("resume_uploads").update(
                {"parsed_at": datetime.datetime.utcnow().isoformat() + "Z", "last_error": None}
            ).eq("job_id", job_id).eq("storage_key", key).execute()
            parsed += 1

        except Exception as e:
            msg = f"{key}: {e}"
            errors.append(msg)
            supabase.table("resume_uploads").update(
                {"last_error": msg}
            ).eq("job_id", job_id).eq("storage_key", key).execute()

    return {"status": "ok", "parsed": parsed, "errors": errors, "remaining_estimate": max(0, len(pending) - parsed)}

def _insert_or_update_resume(job: dict, pdf_text: str, email_hint: Optional[str]):
    # ---- email (unchanged) ----
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", pdf_text or "")
    email = (email_hint or "").strip().lower() or (m.group(0).lower() if m else None)
    if not email:
        email = hashlib.sha1((pdf_text or "")[:200].encode("utf-8")).hexdigest()[:16] + "@noemail.local"

    # ---- skills arrays may be json strings in your DB – handle both (unchanged) ----
    must = _safe_json_list(job.get("skills_must_have"))
    nice = _safe_json_list(job.get("skills_nice_to_have"))

    # ---- keep your existing scoring exactly as-is ----
    scored = _score(pdf_text or "", must, nice, job.get("min_years"), job.get("max_years"))
    meta = scored["meta"]
    jd_match = scored["jd_match_score"]

    # ---- NEW: LLM parse for richer candidate/resume fields (not used for scoring) ----
    cand = parse_resume_structured(pdf_text)  # dict: names, links, skills, experience, etc.

    # ---- upsert candidate (email unique) with richer fields ----
    supabase.table("candidates").upsert(
        {
            "email": email,
            "first_name": cand.get("first_name"),
            "last_name": cand.get("last_name"),
            "full_name": cand.get("full_name"),
            "phone": cand.get("phone"),
            "location": cand.get("location"),
            "links": cand.get("links") or {},
        },
        on_conflict="email",
    ).execute()
    cand_row = supabase.table("candidates").select("candidate_id").eq("email", email).execute().data
    candidate_id = cand_row and cand_row[0].get("candidate_id")
    # gather job skill lists in Python form
    must = _safe_json_list(job.get("skills_must_have"))
    nice = _safe_json_list(job.get("skills_nice_to_have"))

    # ... after you compute meta/jd_match and cand ...
    ai_analysis = generate_resume_analysis(
        cand,
        meta,
        job.get("jd_text") or "",
        job.get("role"),
        must,
        nice,
        job.get("min_years"),
        job.get("max_years"),
    )

    # ---- upsert resume row (unique: job_id + email) ----
    supabase.table("resumes").upsert(
        {
            "job_id": job["job_id"],
            "candidate_id": candidate_id,
            "email": email,
            "status": "PARSED",

            # structured fields from parser
            "first_name": cand.get("first_name"),
            "last_name": cand.get("last_name"),
            "full_name": cand.get("full_name"),
            "phone": cand.get("phone"),
            "location": cand.get("location"),
            "links": cand.get("links") or {},
            "skills": cand.get("skills") or {},
            "experience": cand.get("experience") or [],
            "education": cand.get("education") or [],
            "projects": cand.get("projects") or [],
            "certifications": cand.get("certifications") or [],

            # keep your existing meta + scores
            "meta": meta,
            "raw_text": pdf_text,
            "role": job.get("role"),
            "ai_summary": ai_analysis,
            "jd_match_score": jd_match,
            "skill_match_score": meta.get("skill_match_score"),
            "experience_match_score": meta.get("experience_match_score"),
            "education_match_score": meta.get("education_score", 0),
        },
        on_conflict="job_id,email",
    ).execute()

def _ensure_company(user: UserIdentity) -> Dict[str, Any]:
    rec = (
        supabase.table("recruiters")
        .select("company_id")
        .eq("user_id", user.user_id)
        .single()
        .execute()
        .data
    )
    if not rec:
        raise HTTPException(400, "Please create company profile first")
    return rec

@app.get("/jobs/{job_id}/ranked")
async def ranked(job_id: str, limit: int = Query(50, ge=1, le=200), current_user: UserIdentity = Depends(get_current_user)):
    _ = _get_job_owned(job_id, current_user)  # verify ownership
    rows = (
        supabase.table("resumes")
        .select("*")
        .eq("job_id", job_id)
        .order("jd_match_score", desc=True)
        .limit(limit)
        .execute()
        .data
    )
    return {"job_id": job_id, "count": len(rows), "rows": rows}

@app.post("/jobs/{job_id}/rescore_existing")
async def rescore_existing(job_id: str, current_user: UserIdentity = Depends(get_current_user)):
    job = _get_job_owned(job_id, current_user)
    must = _safe_json_list(job.get("skills_must_have"))
    nice = _safe_json_list(job.get("skills_nice_to_have"))

    res = (
        supabase.table("resumes")
        .select("resume_id, raw_text")
        .eq("job_id", job_id)
        .execute()
        .data
        or []
    )
    updated = 0
    for r in res:
        scored = _score(r.get("raw_text") or "", must, nice, job.get("min_years"), job.get("max_years"))
        meta = scored["meta"]; jd_match = scored["jd_match_score"]

        # reparse candidate (small cost, keeps analysis fresh)
        cand = parse_resume_structured(r.get("raw_text") or "")
        ai_analysis = generate_resume_analysis(
            cand,
            meta,
            job.get("jd_text") or "",
            job.get("role"),
            must,
            nice,
            job.get("min_years"),
            job.get("max_years"),
        )

        supabase.table("resumes").update(
            {
                "jd_match_score": jd_match,
                "skill_match_score": meta.get("skill_match_score"),
                "experience_match_score": meta.get("experience_match_score"),
                "meta": meta,
                "ai_summary": ai_analysis,   # keep UI up to date
            }
        ).eq("resume_id", r["resume_id"]).execute()
        updated += 1

    return {"status": "ok", "rescored": updated}

@app.post("/storage/cleanup")
async def cleanup(current_user: UserIdentity = Depends(get_current_user)):
    now = datetime.datetime.utcnow().isoformat() + "Z"
    expired = supabase.table("resume_uploads").select("*").lt("expires_at", now).execute().data or []
    bucket = supabase.storage.from_(RESUME_BUCKET)
    deleted = 0
    for row in expired:
        try:
            bucket.remove([row["storage_key"]])
        except Exception:
            pass
        supabase.table("resume_uploads").delete().eq("job_id", row["job_id"]).eq("file_hash", row["file_hash"]).execute()
        deleted += 1
    return {"deleted": deleted}

@app.post("/jobs/{job_id}/upload_resumes")
async def upload_resumes(
    job_id: str,
    files: List[UploadFile] = File(...),
    current_user: UserIdentity = Depends(get_current_user),
):
    job = _get_job_owned(job_id, current_user)

    if not files:
        raise HTTPException(400, "No files provided")

    results = []
    for f in files:
        try:
            data = await f.read()
            if not data:
                results.append({"file_name": f.filename, "status": "skipped_empty"})
                continue

            ct = (f.content_type or "").lower()
            is_pdf_ct = ct in ("application/pdf", "application/octet-stream", "binary/octet-stream")
            is_pdf_magic = data[:5] == b"%PDF-"
            if not (is_pdf_ct or is_pdf_magic):
                results.append({
                    "file_name": f.filename,
                    "status": "skipped_invalid_type",
                    "reason": f"Not a PDF (content_type={f.content_type})"
                })
                continue

            sha = _sha256_bytes(data)
            exists = (
                supabase.table("resume_uploads")
                .select("storage_key")
                .eq("job_id", job_id)
                .eq("file_hash", sha)
                .execute()
                .data
            )
            if exists:
                results.append({
                    "file_name": f.filename,
                    "file_hash": sha,
                    "storage_key": exists[0]["storage_key"],
                    "status": "duplicate"
                })
                continue

            storage_key = _upload_to_storage(job_id, sha, f.filename or "resume.pdf", data)
            expires_at = (datetime.datetime.utcnow() + datetime.timedelta(hours=RESUME_TTL_HOURS)).isoformat() + "Z"

            supabase.table("resume_uploads").upsert(
                {
                    "job_id": job_id,
                    "file_hash": sha,
                    "storage_key": storage_key,
                    "uploader_id": current_user.user_id,
                    "expires_at": expires_at,
                    "parsed_at": None,
                    "filename": f.filename,
                    "content_type": f.content_type,
                    "last_error": None,
                },
                on_conflict="job_id,file_hash",
            ).execute()

            results.append({"file_name": f.filename, "file_hash": sha, "storage_key": storage_key, "status": "ok"})

        except Exception as e:
            results.append({"file_name": getattr(f, "filename", None), "status": "error", "reason": str(e)})

    return {"status": "ok", "count": len(results), "results": results}

@app.get("/jobs/{job_id}/resumes")
async def list_resumes(
    job_id: str,
    status: Optional[str] = Query(None, description="Filter by status: PENDING | PARSED | REJECTED"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: UserIdentity = Depends(get_current_user),
):
    """
    List resumes for a job with optional status filter and simple pagination.
    - Sorts by jd_match_score DESC, created_at DESC (if present).
    - Returns total count (for client-side paging) and next_offset cursor.
    """
    # Ownership check
    _ = _get_job_owned(job_id, current_user)

    # Validate status (if provided)
    if status is not None and status not in ALLOWED_RESUME_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status. Use one of {sorted(ALLOWED_RESUME_STATUSES)}")

    q = (
        supabase.table("resumes")
        .select("*", count="exact")
        .eq("job_id", job_id)
    )
    if status:
        q = q.eq("status", status)

    # Primary sort by match score, fallback by created_at if you have that column
    # Supabase allows multiple .order(...) calls
    q = q.order("jd_match_score", desc=True)
    try:
        q = q.order("created_at", desc=True)  # safe if column exists; otherwise ignore/remove
    except Exception:
        pass

    # Pagination
    # Supabase "range" is inclusive on both ends
    end = offset + limit - 1
    q = q.range(offset, end)

    resp = q.execute()
    rows = resp.data or []
    total = getattr(resp, "count", None)

    # next_offset-style cursor
    next_offset = None
    if len(rows) == limit:
        next_offset = offset + limit

    return {
        "job_id": job_id,
        "status_filter": status or "ANY",
        "total": total,           # may be None on older libs; if so, omit in UI
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "rows": rows,
    }

# ===================== INTERVIEWERS CRUD =====================

# Create
@app.post("/interviewers", response_model=InterviewerOut)
async def invite_interviewer(
    body: InterviewerInvite,
    current_user: UserIdentity = Depends(get_current_user),
):
    rec = _ensure_company(current_user)
    company_id = rec["company_id"]

    # enforce uniqueness per company
    existing = (
        supabase.table("interviewers")
        .select("interviewer_id")
        .eq("company_id", company_id)
        .eq("email", body.email.lower())
        .limit(1)
        .execute()
        .data
    )
    if existing:
        raise HTTPException(409, "Interviewer with this email already exists in this company")

    admin = admin_sb.auth.admin

    # 1) Send an invite email (Supabase sends it via SMTP)
    # NOTE: 'options' shape varies with library versions, we handle both.
    try:
        invited = admin.invite_user_by_email(
            body.email.lower(),
            { "redirect_to": INVITE_REDIRECT }
        )
    except Exception:
        # older signatures sometimes use 'data' kw; this keeps it robust
        invited = admin.invite_user_by_email(body.email.lower(), {"redirect_to": INVITE_REDIRECT})

    auth_user_id = invited.user.id

    # 2) Set metadata (invite API only sets user_metadata by default)
    admin.update_user_by_id(auth_user_id, {
        "app_metadata":  {"role": "interviewer", "company_id": company_id},
        "user_metadata": {"name": body.name},
    })

    # 3) Insert your shadow row
    row = (
        supabase.table("interviewers")
        .insert({
            "company_id": company_id,
            "auth_user_id": auth_user_id,
            "name": body.name.strip(),
            "email": body.email.lower(),
            "is_active": bool(body.is_active),
            "created_by": current_user.user_id,
        })
        .execute()
        .data[0]
    )

    return {
        "interviewer_id": row["interviewer_id"],
        "name": row["name"],
        "email": row["email"],
        "company_id": row.get("company_id"),
        "is_active": row.get("is_active", True),
    }

@app.get("/interviewers", response_model=List[InterviewerOut])
async def list_interviewers(
    q: Optional[str] = Query(None, description="search by name/email"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: UserIdentity = Depends(get_current_user),
):
    company_id = _get_company_id_for_user(current_user)
    if not company_id:
        raise HTTPException(400, "Please create company profile first")

    query = (
        supabase.table("interviewers")
        .select("interviewer_id, name, email, company_id, is_active")
        .eq("company_id", company_id)
    )

    # Optional order if column exists
    try:
        query = query.order("created_at", desc=True)
    except Exception:
        pass

    if q:
        # Search both name and email (PostgREST OR syntax)
        try:
            query = query.or_(f"email.ilike.%{q}%,name.ilike.%{q}%")
        except Exception:
            # Minimal fallback: filter by email only
            query = query.filter("email", "ilike", f"%{q}%")

    end = offset + limit - 1
    rows = query.range(offset, end).execute().data or []
    return [
        {
            "interviewer_id": r["interviewer_id"],
            "name": r["name"],
            "email": r["email"],
            "company_id": r.get("company_id"),
            "is_active": r.get("is_active", True),
        }
        for r in rows
    ]

# Read one
@app.get("/interviewers/{interviewer_id}", response_model=InterviewerOut)
async def get_interviewer(
    interviewer_id: str,
    current_user: UserIdentity = Depends(get_current_user),
):
    company_id = _get_company_id_for_user(current_user)
    row = (
        supabase.table("interviewers")
        .select("interviewer_id, name, email, company_id, is_active")
        .eq("interviewer_id", interviewer_id)
        .eq("company_id", company_id)
        .single()
        .execute()
        .data
    )
    if not row:
        raise HTTPException(404, "Interviewer not found")
    return row

# Update
@app.put("/interviewers/{interviewer_id}", response_model=InterviewerOut)
async def update_interviewer(
    interviewer_id: str,
    body: InterviewerUpdate,
    current_user: UserIdentity = Depends(get_current_user),
):
    rec = _ensure_company(current_user)
    company_id = rec["company_id"]

    # 1) Fetch current row to get auth_user_id
    existing = (
        supabase.table("interviewers")
        .select("interviewer_id, auth_user_id, name, email, company_id, is_active")
        .eq("interviewer_id", interviewer_id)
        .eq("company_id", company_id)
        .single()
        .execute()
        .data
    )
    if not existing:
        raise HTTPException(404, "Interviewer not found")

    updates: Dict[str, Any] = {}
    if body.name is not None:
        updates["name"] = body.name.strip()
    if body.email is not None:
        updates["email"] = body.email.lower()
    if body.is_active is not None:
        updates["is_active"] = bool(body.is_active)

    if not updates:
        return existing  # nothing to do

    # 2) Update Auth user (only if name/email changed)
    admin = admin_sb.auth.admin
    auth_updates: Dict[str, Any] = {}
    if "name" in updates:
        auth_updates.setdefault("user_metadata", {})["name"] = updates["name"]
    if "email" in updates:
        auth_updates["email"] = updates["email"]

    if auth_updates:
        admin.update_user_by_id(existing["auth_user_id"], auth_updates)

    # 3) Update your row
    row = (
        supabase.table("interviewers")
        .update(updates)
        .eq("interviewer_id", interviewer_id)
        .eq("company_id", company_id)
        .execute()
        .data[0]
    )
    return {
        "interviewer_id": row["interviewer_id"],
        "name": row["name"],
        "email": row["email"],
        "company_id": row.get("company_id"),
        "is_active": row.get("is_active", True),
    }

@app.post("/interviewers/{interviewer_id}/deactivate")
async def deactivate_interviewer(
    interviewer_id: str,
    current_user: UserIdentity = Depends(get_current_user),
):
    rec = _ensure_company(current_user)
    company_id = rec["company_id"]

    row = (
        supabase.table("interviewers")
        .select("auth_user_id")
        .eq("interviewer_id", interviewer_id)
        .eq("company_id", company_id)
        .single()
        .execute()
        .data
    )
    if not row:
        raise HTTPException(404, "Interviewer not found")

    admin = admin_sb.auth.admin
    # Mark blocked in app_metadata (enforce in RLS or app logic)
    admin.update_user_by_id(row["auth_user_id"], {"app_metadata": {"blocked": True}})
    supabase.table("interviewers").update({"is_active": False}).eq("interviewer_id", interviewer_id).execute()
    return {"message": "Interviewer deactivated"}

@app.post("/interviewers/{interviewer_id}/reactivate")
async def reactivate_interviewer(
    interviewer_id: str,
    current_user: UserIdentity = Depends(get_current_user),
):
    rec = _ensure_company(current_user)
    company_id = rec["company_id"]

    row = (
        supabase.table("interviewers")
        .select("auth_user_id")
        .eq("interviewer_id", interviewer_id)
        .eq("company_id", company_id)
        .single()
        .execute()
        .data
    )
    if not row:
        raise HTTPException(404, "Interviewer not found")

    admin = admin_sb.auth.admin
    admin.update_user_by_id(row["auth_user_id"], {"app_metadata": {"blocked": False}})
    supabase.table("interviewers").update({"is_active": True}).eq("interviewer_id", interviewer_id).execute()
    return {"message": "Interviewer reactivated"}

@app.delete("/interviewers/{interviewer_id}")
async def delete_interviewer(
    interviewer_id: str,
    hard: bool = Query(False, description="set true to hard delete"),
    current_user: UserIdentity = Depends(get_current_user),
):
    rec = _ensure_company(current_user)
    company_id = rec["company_id"]

    row = (
        supabase.table("interviewers")
        .select("auth_user_id")
        .eq("interviewer_id", interviewer_id)
        .eq("company_id", company_id)
        .single()
        .execute()
        .data
    )
    if not row:
        raise HTTPException(404, "Interviewer not found")

    if hard:
        supabase.auth.admin.delete_user(row["auth_user_id"])
        supabase.table("interviewers").delete().eq("interviewer_id", interviewer_id).eq("company_id", company_id).execute()
        return {"message": "Interviewer deleted", "hard": True}

    # soft delete
    supabase.table("interviewers").update({"is_active": False}).eq("interviewer_id", interviewer_id).eq("company_id", company_id).execute()
    supabase.auth.admin.update_user_by_id(row["auth_user_id"], {"app_metadata": {"blocked": True}})
    return {"message": "Interviewer deactivated", "hard": False}

@app.post("/interviewers/{interviewer_id}/send-reset", response_model=SendResetOut)
async def send_reset_link(interviewer_id: str, current_user: UserIdentity = Depends(get_current_user)):
    rec = _ensure_company(current_user)
    company_id = rec["company_id"]

    row = (
        supabase.table("interviewers")
        .select("email")
        .eq("interviewer_id", interviewer_id)
        .eq("company_id", company_id)
        .single()
        .execute()
        .data
    )
    if not row:
        raise HTTPException(404, "Interviewer not found")

    email = row["email"].lower()
    try:
        public_sb.auth.reset_password_for_email(email, options={"redirect_to": RESET_REDIRECT})
    except AttributeError:
        public_sb.auth.reset_password_for_email(email, {"redirect_to": RESET_REDIRECT})

    # <-- return must be outside the try/except
    return {"action_link": f"(email sent) redirect_to={RESET_REDIRECT}"}

@app.post("/auth/interviewer/signin", response_model=InterviewerSessionOut)
async def interviewer_signin(body: InterviewerSignIn):
    try:
        # 1) sign in with email/password
        resp = public_sb.auth.sign_in_with_password({"email": body.email, "password": body.password})
        if not resp.session or not resp.user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user = resp.user
        # 2) optional safety: honor app_metadata.blocked
        app_meta = (getattr(user, "app_metadata", None) or {})
        if app_meta.get("blocked") is True:
            raise HTTPException(status_code=403, detail="Account is blocked")

        # 3) find the interviewer shadow row by auth_user_id (preferred), fallback by email
        row = (
            supabase.table("interviewers")
            .select("*")
            .eq("auth_user_id", user.id)
            .single()
            .execute()
            .data
        )
        if not row:
            row = (
                supabase.table("interviewers")
                .select("*")
                .eq("email", body.email.lower())
                .single()
                .execute()
                .data
            )

        if not row:
            raise HTTPException(status_code=403, detail="Not an interviewer")
        if not row.get("is_active", True):
            raise HTTPException(status_code=403, detail="Interviewer is deactivated")

        return {
            "access_token": resp.session.access_token,
            "refresh_token": resp.session.refresh_token,
            "user": {"id": user.id, "email": user.email},
            "interviewer": {
                "interviewer_id": row["interviewer_id"],
                "name": row.get("name"),
                "email": row.get("email"),
                "company_id": row.get("company_id"),
                "is_active": row.get("is_active", True),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Login failed: {e}")

@app.get("/interviewer/me")
async def interviewer_me(current_user: UserIdentity = Depends(get_current_user)):
    row = (
        supabase.table("interviewers")
        .select("interviewer_id, name, email, company_id, is_active")
        .eq("auth_user_id", current_user.user_id)
        .single()
        .execute()
        .data
    )
    if not row:
        raise HTTPException(status_code=404, detail="Interviewer profile not found")
    if not row.get("is_active", True):
        raise HTTPException(status_code=403, detail="Interviewer is deactivated")
    return {
        "user": {"id": current_user.user_id, "email": current_user.email},
        "interviewer": row,
    }

@app.post("/jobs/{job_id}/resumes/shortlist")
async def shortlist_resumes(
    job_id: str,
    payload: BulkShortlistRequest,
    current_user: UserIdentity = Depends(get_current_user),
):
    """
    Bulk-shortlist candidates for a job.
    - Accepts either 'resume_ids' or 'emails' (or both).
    - Default behavior updates rows currently in status == PARSED.
    - Returns counts for transparency.
    """
    # Verify recruiter owns the job
    _ = _get_job_owned(job_id, current_user)

    ids = [i for i in (payload.resume_ids or []) if i]
    emails = [e.lower() for e in (payload.emails or []) if e]
    if not ids and not emails:
        raise HTTPException(status_code=400, detail="Provide resume_ids and/or emails")

    updated_total = 0
    skipped = 0
    errors: List[Dict[str, Any]] = []

    # What we will set
    new_status = "SHORTLISTED"
    if new_status not in ALLOWED_RESUME_STATUSES:
        raise HTTPException(status_code=500, detail="Server not configured for status SHORTLISTED")

    # Update by resume_ids
    if ids:
        try:
            resp = (
                supabase.table("resumes")
                .update({"status": new_status})
                .in_("resume_id", ids)
                .eq("job_id", job_id)
                .eq("status", payload.only_if_status)
                .execute()
            )
            updated_total += len(resp.data or [])
        except Exception as e:
            errors.append({"by": "resume_ids", "error": str(e)})

    # Update by emails
    if emails:
        try:
            resp = (
                supabase.table("resumes")
                .update({"status": new_status})
                .in_("email", emails)
                .eq("job_id", job_id)
                .eq("status", payload.only_if_status)
                .execute()
            )
            updated_total += len(resp.data or [])
        except Exception as e:
            errors.append({"by": "emails", "error": str(e)})

    # For visibility, report how many in the selection are currently NOT in only_if_status
    # (best-effort estimate)
    try:
        selected_count = 0
        if ids:
            selected_count += (
                supabase.table("resumes")
                .select("resume_id", count="exact")
                .in_("resume_id", ids).eq("job_id", job_id)
                .execute().count or 0
            )
        if emails:
            selected_count += (
                supabase.table("resumes")
                .select("email", count="exact")
                .in_("email", emails).eq("job_id", job_id)
                .execute().count or 0
            )
        skipped = max(0, selected_count - updated_total)
    except Exception:
        # ignore count errors, still return updated_total
        pass

    return {
        "job_id": job_id,
        "new_status": new_status,
        "updated": updated_total,
        "skipped_non_matching_status_or_missing": skipped,
        "errors": errors,
    }

from postgrest.exceptions import APIError as PgAPIError

@app.post("/interviews/schedule")
async def schedule_interview(
    body: ScheduleInterviewIn,
    current_user: UserIdentity = Depends(get_current_user),
):
    # Verify recruiter owns the job
    _ = _get_job_owned(body.job_id, current_user)

    # ---------- candidate (resume) ----------
    res_row = _first_row(
        supabase.table("resumes")
        .select("email, full_name")
        .eq("resume_id", body.resume_id)
        .eq("job_id", body.job_id)
        .limit(1)
        .execute()
    )
    if not res_row:
        raise HTTPException(404, "Resume not found for job")
    candidate_email = res_row["email"]

    # ---------- interviewer ----------
    intv = _first_row(
        supabase.table("interviewers")
        .select("email, name")
        .eq("interviewer_id", body.interviewer_id)
        .limit(1)
        .execute()
    )
    if not intv:
        raise HTTPException(404, "Interviewer not found")
    interviewer_email = intv["email"]

    # ---------- job (and optional company) ----------
    job = _first_row(
        supabase.table("jobs")
        .select("role, company_id")
        .eq("job_id", body.job_id)
        .limit(1)
        .execute()
    )
    if not job:
        raise HTTPException(404, "Job not found")

    comp = None
    if job.get("company_id"):
        comp = _first_row(
            supabase.table("companies")
            .select("company_name")
            .eq("company_id", job["company_id"])
            .limit(1)
            .execute()
        )

    role = job.get("role") or "Interview"
    company_name = (comp or {}).get("company_name") or ""
    # ✅ this was missing before; you referenced `title` without defining it
    title = f"{role} Interview" + (f" — {company_name}" if company_name else "")

    # ---------- idempotency check ----------
    if body.external_id:
        existing = _first_row(
            supabase.table("interviews")
            .select("*")
            .eq("external_id", body.external_id)
            .limit(1)
            .execute()
        )
        if existing and existing.get("google_event_id"):
            return {
                "status": existing["status"],
                "calendarId": existing.get("google_html_link"),
                "meetLink": existing.get("google_meet_link"),
                "interview_id": existing["interview_id"],
            }

    # ---------- Google Calendar ----------
    svc = _google_service_for_user(current_user.user_id)

    # Best-effort free/busy check (skip on error)
    try:
        fb_req = {
            "timeMin": body.start_iso,
            "timeMax": body.end_iso,
            "timeZone": body.timezone,
            "items": [{"id": interviewer_email}, {"id": candidate_email}],
        }
        fb = svc.freebusy().query(body=fb_req).execute()
        busy = []
        for cal in fb.get("calendars", {}).values():
            busy.extend(cal.get("busy", []))
        if busy:
            return {"status": "CONFLICT", "busy": busy}
    except Exception:
        pass

    description = (
        f"Interview for {role}"
        + (f" at {company_name}" if company_name else "")
        + ".\n\nPlease join using the Google Meet link. "
          "If you need to reschedule, reply to this email.\n\n"
          "Agenda: 5 min intro, 35 min technical Q&A, 10 min wrap-up.\n"
          "Kindly accept or decline from the calendar invite."
    )

    event = {
        "summary": title,
        "description": description,
        "start": {"dateTime": body.start_iso, "timeZone": body.timezone},
        "end":   {"dateTime": body.end_iso,   "timeZone": body.timezone},
        "attendees": [
            {"email": candidate_email},
            {"email": interviewer_email},
            {"email": current_user.email},  # recruiter/admin
        ],
        "extendedProperties": {"shared": {"externalId": body.external_id or secrets.token_hex(8)}},
        "conferenceData": {
            "createRequest": {
                "requestId": secrets.token_hex(8),
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
    }

    created = svc.events().insert(
        calendarId="primary",
        body=event,
        conferenceDataVersion=1,  # required to create Meet
        sendUpdates="all"         # email all attendees
    ).execute()

    meet = created.get("hangoutLink") or (
        (created.get("conferenceData", {}).get("entryPoints") or [{}])[0].get("uri")
    )

    # ---------- persist interview ----------
    row = (
        supabase.table("interviews")
        .insert({
            "job_id": body.job_id,
            "resume_id": body.resume_id,
            "interviewer_id": body.interviewer_id,
            "candidate_email": candidate_email,
            "interviewer_email": interviewer_email,
            "recruiter_email": current_user.email,
            "status": "SCHEDULED",  # your single main status
            "google_event_id": created["id"],
            "google_html_link": created.get("htmlLink"),
            "google_meet_link": meet,
            "external_id": (body.external_id or event["extendedProperties"]["shared"]["externalId"]),
            "start_at": body.start_iso,
            "end_at": body.end_iso,
            "created_by": current_user.user_id,
        })
        .execute()
        .data[0]
    )

    return {
        "interview_id": row["interview_id"],
        "status": row["status"],
        "calendarId": row["google_html_link"],
        "meetLink": row["google_meet_link"],
        "flags": {
            # ✅ compare against the actual stored value
            "scheduled": row["status"] == "SCHEDULED",
            "interview_done": row["status"] == "INTERVIEW_DONE",
            "select": row["status"] == "SELECTED",
            "reject": row["status"] == "REJECTED",
            "review": row["status"] == "REVIEW",
            "hired": row["status"] == "HIRED",
            "absent": row["status"] == "ABSENT",
        },
        "attendees": created.get("attendees", []),
    }

class InterviewStatusIn(BaseModel):
    status: str  # validate against the enum in code too

@app.post("/interviews/{interview_id}/status")
async def update_interview_status(interview_id: str, body: InterviewStatusIn,
                                  current_user: UserIdentity = Depends(get_current_user)):
    allowed = {"INTERVIEW_SCHEDULED","INTERVIEW_DONE","SELECTED","REJECTED","REVIEW","HIRED","ABSENT","CANCELLED"}
    if body.status not in allowed:
        raise HTTPException(400, "Invalid status")
    row = supabase.table("interviews").update({"status": body.status}).eq("interview_id", interview_id).execute().data
    if not row:
        raise HTTPException(404, "Interview not found")
    return {"interview_id": interview_id, "status": body.status}
# ---------------------------------------------------------------------------
# run (local)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
