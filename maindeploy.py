import os
import re
import json
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middlewaclass JDIngestAnswer(BaseModel):
    original_jd_text: str
    parsed: Dict[str, Any]
    answers: Dict[str, Any]
re.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from supabase import create_client, Client
from dotenv import load_dotenv
import jwt
import requests

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


class JDIngestAnswer(BaseModel):
    original_jd_text: str
    parsed: Dict[str, Any]
    answers: Dict[str, Any]


class UserIdentity(BaseModel):
    user_id: str
    email: str


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


# ---------------------------------------------------------------------------
# routes
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
# run (local)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

