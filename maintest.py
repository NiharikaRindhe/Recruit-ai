import os
import re
import json
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

import hashlib
import datetime
import fitz  # PyMuPDF
# Safe import for different storage3 versions
try:
    from storage3.utils import UploadFileOptions  # v2+
except Exception:  # older SDKs won't have it
    UploadFileOptions = None

# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------
load_dotenv()

app = FastAPI(title="Recruitment AI API", version="1.0.0")

# CORS
origins_env = os.getenv("ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
# ---------- NEW: storage + TTL settings ----------
RESUME_BUCKET = os.getenv("RESUME_BUCKET", "resumes")
RESUME_TTL_HOURS = int(os.getenv("RESUME_TTL_HOURS", "36"))
# ---------------------------------------------------------------------------
# Ollama (cloud-first)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
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
        elif re.search(r"(?i)\bonsite|on-site|office\b", text):
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
        resp = supabase.auth.sign_up({"email": request.email, "password": request.password})
        if resp.user:
            return {"message": "User created successfully", "user": {"id": resp.user.id, "email": resp.user.email}}
        raise HTTPException(status_code=400, detail="Failed to create user")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/signin")
async def signin(request: SignInRequest):
    try:
        resp = supabase.auth.sign_in_with_password({"email": request.email, "password": request.password})
        if resp.session:
            return {
                "access_token": resp.session.access_token,
                "refresh_token": resp.session.refresh_token,
                "user": {"id": resp.user.id, "email": resp.user.email},
            }
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password")

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
ALLOWED_RESUME_STATUSES = {"PENDING", "PARSED", "REJECTED"}

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

SUMMARY_PROMPT = (
    "Write a <=100-word hiring summary stating whether {name} fits the role \"{role}\". "
    "Mention verified skills ({skills}) and total experience ({yoe} years). "
    "Conclude with the JD match score: {score}. Plain text only. JD: {jd}"
)

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

def generate_resume_summary(candidate: dict, meta: dict, jd_text: str, role: str, score: float) -> str:
    try:
        name = (
            candidate.get("full_name")
            or (candidate.get("first_name") or "") + " " + (candidate.get("last_name") or "")
        ).strip() or "Candidate"
        top_sk = ", ".join((meta.get("top_matched_skills") or [])[:6]) or "n/a"
        yoe = meta.get("total_experience_years", "n/a")
        prompt = SUMMARY_PROMPT.format(
            name=name, role=role or "Software Engineer", skills=top_sk, yoe=yoe, score=score, jd=(jd_text or "")[:1500]
        )
        return _ollama_generate(prompt).strip()[:800]
    except Exception:
        return "Concise fit summary unavailable."
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
        .is_("parsed_at", None)  # supabase-py supports .is_ for IS NULL
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

    # ---- NEW: AI summary (uses your JD text + your meta/scores) ----
    ai_sum = generate_resume_summary(
        cand,
        {**meta, "primary_role": job.get("role")},
        job.get("jd_text") or "",
        job.get("role"),
        jd_match,
    )

    # ---- upsert resume row (unique: job_id + email) ----
    supabase.table("resumes").upsert(
        {
            "job_id": job["job_id"],
            "candidate_id": candidate_id,
            "email": email,
            "status": "PENDING",

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
            "ai_summary": ai_sum,
            "jd_match_score": jd_match,
            "skill_match_score": meta.get("skill_match_score"),
            "experience_match_score": meta.get("experience_match_score"),
            "education_match_score": meta.get("education_score", 0),
        },
        on_conflict="job_id,email",
    ).execute()

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
    for r in res:
        scored = _score(r.get("raw_text") or "", must, nice, job.get("min_years"), job.get("max_years"))
        meta = scored["meta"]; jd_match = scored["jd_match_score"]
        supabase.table("resumes").update(
            {
                "jd_match_score": jd_match,
                "skill_match_score": meta.get("skill_match_score"),
                "experience_match_score": meta.get("experience_match_score"),
                "meta": meta,
            }
        ).eq("resume_id", r["resume_id"]).execute()
    return {"status": "ok", "rescored": len(res)}

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
# ---------------------------------------------------------------------------
# run (local)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
