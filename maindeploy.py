# maindeploy.py
import os
import json
import requests
import jwt
from typing import Optional, List, Dict, Any

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from supabase import create_client, Client
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load env
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# FastAPI app & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Recruitment AI API", version="1.1.0")

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
# Ollama (Cloud OR local)
# ---------------------------------------------------------------------------
# Cloud:
#   OLLAMA_BASE_URL=https://ollama.com/api
#   OLLAMA_MODEL=gpt-oss:20b
#   OLLAMA_API_KEY=...
# Local:
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=gemma3:4b
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# ---------------------------------------------------------------------------
# Security / auth
# ---------------------------------------------------------------------------
security = HTTPBearer()


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


class JDFromTextRequest(BaseModel):
    jd_text: str


class JDEditAnswers(BaseModel):
    # [{ "field": "work_mode", "answer": "Remote" }]
    answers: List[Dict[str, Any]]


class UserIdentity(BaseModel):
    user_id: str
    email: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def verify_token(token: str) -> Optional[dict]:
    """
    Verify Supabase JWT token (here we don't verify signature because Supabase already did).
    If you later want to verify, fetch Supabase JWKS and verify properly.
    """
    try:
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded
    except Exception:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserIdentity:
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    return UserIdentity(user_id=user_id, email=email)


def _is_cloud_host(base_url: str) -> bool:
    return base_url.startswith("https://ollama.com")


def _normalize_model_tag(model: str, cloud: bool) -> str:
    # Cloud wants tag without "-cloud"
    if cloud and model.endswith("-cloud"):
        return model[:-6]
    return model


def generate_jd_with_ollama(prompt: str) -> str:
    """
    Shared utility: call Ollama (cloud or local) and return plain text.
    """
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

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response")
        if not text:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Ollama returned empty response",
            )
        return text

    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ollama request failed: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating JD with AI: {str(e)}",
        )


def create_jd_prompt(req: JDCreateRequest) -> str:
    # normalize
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
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Recruitment AI API is running", "status": "active"}


# ---------------- Auth ----------------
@app.post("/auth/signup")
async def signup(request: SignUpRequest):
    try:
        response = supabase.auth.sign_up(
            {
                "email": request.email,
                "password": request.password,
            }
        )

        if response.user:
            return {
                "message": "User created successfully",
                "user": {
                    "id": response.user.id,
                    "email": response.user.email,
                },
            }

        raise HTTPException(status_code=400, detail="Failed to create user")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/signin")
async def signin(request: SignInRequest):
    try:
        response = supabase.auth.sign_in_with_password(
            {
                "email": request.email,
                "password": request.password,
            }
        )
        if response.session:
            return {
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "user": {
                    "id": response.user.id,
                    "email": response.user.email,
                },
            }
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password")


# ---------------- /me ----------------
@app.get("/me")
async def get_me(current_user: UserIdentity = Depends(get_current_user)):
    """
    FE uses this to decide:
    - if recruiter/company exists -> go directly to JD screen
    - else -> show "create company"
    """
    recruiter_resp = (
        supabase.table("recruiters")
        .select("user_id, full_name, company_id, company_email")
        .eq("user_id", current_user.user_id)
        .execute()
    )

    if recruiter_resp.data:
        recruiter = recruiter_resp.data[0]
        company = None
        company_id = recruiter.get("company_id")
        if company_id:
            company_resp = (
                supabase.table("companies")
                .select("company_id, company_name, description, location, linkedin_url")
                .eq("company_id", company_id)
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

    # first time user
    return {
        "user": {"id": current_user.user_id, "email": current_user.email},
        "recruiter": None,
        "company": None,
        "has_company": False,
    }


# ---------------- Company ----------------
@app.post("/company/create")
async def create_company(
    request: CompanyRequest,
    current_user: UserIdentity = Depends(get_current_user),
):
    try:
        existing_company = (
            supabase.table("companies")
            .select("*")
            .eq("company_name", request.company_name)
            .execute()
        )

        if existing_company.data:
            company_id = existing_company.data[0]["company_id"]
        else:
            company_data = {
                "company_name": request.company_name,
                "location": request.location,
                "linkedin_url": request.linkedin_url,
                "description": request.description,
            }
            company_result = supabase.table("companies").insert(company_data).execute()
            company_id = company_result.data[0]["company_id"]

        recruiter_data = {
            "user_id": current_user.user_id,
            "company_email": current_user.email,
            "full_name": request.recruiter_name,
            "company_id": company_id,
        }

        existing_recruiter = (
            supabase.table("recruiters")
            .select("*")
            .eq("user_id", current_user.user_id)
            .execute()
        )

        if existing_recruiter.data:
            recruiter_result = (
                supabase.table("recruiters")
                .update(recruiter_data)
                .eq("user_id", current_user.user_id)
                .execute()
            )
        else:
            recruiter_result = (
                supabase.table("recruiters").insert(recruiter_data).execute()
            )

        return {
            "message": "Company profile created successfully",
            "company_id": company_id,
            "recruiter": recruiter_result.data[0],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating company: {str(e)}")


# ---------------- JD create (AI) ----------------
@app.post("/jd/create")
async def create_jd(
    request: JDCreateRequest, current_user: UserIdentity = Depends(get_current_user)
):
    try:
        recruiter = (
            supabase.table("recruiters")
            .select("company_id")
            .eq("user_id", current_user.user_id)
            .execute()
        )

        if not recruiter.data:
            raise HTTPException(
                status_code=400, detail="Please create company profile first"
            )

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

        return {
            "message": "JD generated successfully",
            "job_id": job_result.data[0]["job_id"],
            "jd_text": jd_text,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating JD: {str(e)}"
        )


# ---------------- JD ingest from plain text (NEW) ----------------
@app.post("/jd/ingest-text")
async def ingest_jd_text(
    req: JDFromTextRequest,
    current_user: UserIdentity = Depends(get_current_user),
):
    """
    Recruiter pastes JD text.
    We:
      1. ensure user has company
      2. best-effort AI extract
      3. insert job with status='draft'
      4. return follow-up questions if key fields are missing
    """
    # 1) user must have company
    recruiter = (
        supabase.table("recruiters")
        .select("company_id")
        .eq("user_id", current_user.user_id)
        .execute()
    )
    if not recruiter.data:
        raise HTTPException(
            status_code=400, detail="Please create company profile first"
        )
    company_id = recruiter.data[0]["company_id"]

    # optional: get company name
    company_name = ""
    comp = (
        supabase.table("companies")
        .select("company_name, description")
        .eq("company_id", company_id)
        .execute()
    )
    if comp.data:
        company_name = comp.data[0].get("company_name") or ""

    # base parsed
    parsed: Dict[str, Any] = {
        "job_title": None,
        "employment_type": None,
        "work_mode": None,
        "location": None,
        "min_experience": None,
        "max_experience": None,
        "skills_must_have": [],
        "skills_nice_to_have": [],
        "requirements": req.jd_text,
    }

    # 2) best-effort AI extraction
    try:
        is_cloud = _is_cloud_host(OLLAMA_BASE_URL)
        model = _normalize_model_tag(OLLAMA_MODEL, is_cloud)
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/generate"
        headers = {"Content-Type": "application/json"}
        if is_cloud and OLLAMA_API_KEY:
            headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

        extract_prompt = f"""
You are an expert recruiter.
From the JD below, extract these keys as JSON only:
job_title, employment_type, work_mode, location, min_experience, max_experience,
skills_must_have, skills_nice_to_have, requirements.
Return null/[] for anything missing.

JD:
{req.jd_text}
""".strip()

        r = requests.post(
            url,
            headers=headers,
            json={"model": model, "prompt": extract_prompt, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            raw = raw[s : e + 1]
        extracted = json.loads(raw)
        parsed.update(extracted)
    except Exception:
        # don't fail – we still insert
        pass

    # 3) figure out missing → questions for FE
    must_fields = ["job_title", "employment_type", "work_mode"]
    missing = [f for f in must_fields if not parsed.get(f)]
    questions: List[str] = []
    for f in missing:
        if f == "job_title":
            questions.append(
                "What is the job title? (e.g. Senior Backend Engineer, Data Scientist)"
            )
        elif f == "employment_type":
            questions.append(
                "What is the employment type? (Full-time, Part-time, Contract, Internship)"
            )
        elif f == "work_mode":
            questions.append(
                "What is the work mode? (Remote, Onsite, Hybrid)"
            )

    # 4) insert job with a VALID status
    job_payload = {
        "company_id": company_id,
        "created_by": current_user.user_id,
        "role": parsed.get("job_title"),
        "location": parsed.get("location"),
        "employment_type": parsed.get("employment_type"),
        "work_mode": parsed.get("work_mode"),
        "min_years": parsed.get("min_experience"),
        "max_years": parsed.get("max_experience"),
        "skills_must_have": json.dumps(parsed.get("skills_must_have") or []),
        "skills_nice_to_have": json.dumps(parsed.get("skills_nice_to_have") or []),
        "jd_text": req.jd_text,
        "status": "draft",  # ✅ important – your table already knows this
    }

    job_res = supabase.table("jobs").insert(job_payload).execute()
    job_id = job_res.data[0]["job_id"]

    return {
        "message": "JD ingested",
        "job_id": job_id,
        "parsed": parsed,
        "questions": questions,
        "done": len(questions) == 0,
    }


# ---------------- JD clarify (NEW) ----------------
@app.post("/jd/clarify/{job_id}")
async def clarify_jd(
    job_id: str,
    payload: JDEditAnswers,
    current_user: UserIdentity = Depends(get_current_user),
):
    # fetch job
    job_res = (
        supabase.table("jobs")
        .select("*")
        .eq("job_id", job_id)
        .eq("created_by", current_user.user_id)
        .execute()
    )
    if not job_res.data:
        raise HTTPException(status_code=404, detail="Job not found")

    updates: Dict[str, Any] = {}
    for item in payload.answers:
        field = item.get("field")
        answer = item.get("answer")
        if not field:
            continue
        if field == "job_title":
            updates["role"] = answer
        elif field == "employment_type":
            updates["employment_type"] = answer
        elif field == "work_mode":
            updates["work_mode"] = answer
        elif field == "location":
            updates["location"] = answer
        elif field == "min_experience":
            updates["min_years"] = float(answer) if answer is not None else None
        elif field == "max_experience":
            updates["max_years"] = float(answer) if answer is not None else None

    if updates:
        supabase.table("jobs").update(updates).eq("job_id", job_id).execute()

    # re-fetch to see what's still missing
    refreshed = (
        supabase.table("jobs")
        .select("*")
        .eq("job_id", job_id)
        .eq("created_by", current_user.user_id)
        .single()
        .execute()
        .data
    )

    still_missing: List[str] = []
    if not refreshed.get("role"):
        still_missing.append(
            "What is the job title? (e.g. Senior Backend Engineer, Data Scientist)"
        )
    if not refreshed.get("employment_type"):
        still_missing.append(
            "What is the employment type? (Full-time, Part-time, Contract, Internship)"
        )
    if not refreshed.get("work_mode"):
        still_missing.append(
            "What is the work mode? (Remote, Onsite, Hybrid)"
        )

    return {
        "job_id": job_id,
        "done": len(still_missing) == 0,
        "questions": still_missing,
    }


# ---------------- JD regenerate ----------------
@app.post("/jd/regenerate/{job_id}")
async def regenerate_jd(
    job_id: str, current_user: UserIdentity = Depends(get_current_user)
):
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
        raise HTTPException(status_code=500, detail=f"Error regenerating JD: {str(e)}")


# ---------------- JD edit ----------------
@app.put("/jd/edit/{job_id}")
async def edit_jd(
    job_id: str,
    jd_text: str = Query(..., description="New JD text"),
    current_user: UserIdentity = Depends(get_current_user),
):
    try:
        result = (
            supabase.table("jobs")
            .update({"jd_text": jd_text})
            .eq("job_id", job_id)
            .eq("created_by", current_user.user_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(
                status_code=404, detail="Job not found or unauthorized"
            )
        return {
            "message": "JD updated successfully",
            "job_id": job_id,
            "jd_text": jd_text,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating JD: {str(e)}")


# ---------------- finalize ----------------
@app.post("/jd/finalize/{job_id}")
async def finalize_jd(
    job_id: str, current_user: UserIdentity = Depends(get_current_user)
):
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
        raise HTTPException(status_code=400, detail=f"Error finalizing JD: {str(e)}")


# ---------------- my-jobs ----------------
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
        raise HTTPException(status_code=400, detail=f"Error fetching jobs: {str(e)}")


# ---------------- helper: list ollama tags ----------------
@app.get("/ollama/tags")
async def list_ollama_tags():
    is_cloud = _is_cloud_host(OLLAMA_BASE_URL)
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/tags"
    headers = {}
    if is_cloud and OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch tags: {e}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
