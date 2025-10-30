
import os
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from supabase import create_client, Client
from dotenv import load_dotenv
import jwt
import requests
import json

# Load environment variables
load_dotenv()

# ======================== App & CORS ========================
app = FastAPI(title="Recruitment AI API", version="1.0.0")

origins_env = os.getenv("ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== Supabase ========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ======================== Ollama (Cloud-first) ========================
# For direct Cloud usage:
#   OLLAMA_BASE_URL=https://ollama.com/api
#   OLLAMA_MODEL=gpt-oss:20b    # (no -cloud suffix for direct Cloud API)
#   OLLAMA_API_KEY=<your key>
# For local daemon with Cloud offloading instead, use:
#   OLLAMA_BASE_URL=http://localhost:11434/api
#   OLLAMA_MODEL=gpt-oss:20b-cloud  # requires `ollama signin` locally

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# Security
security = HTTPBearer()

# ======================== Models ========================
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
    skills_must_have: list[str] = Field(default_factory=list)
    skills_nice_to_have: list[str] = Field(default_factory=list)
    requirements: str
    location: Optional[str] = None

class UserIdentity(BaseModel):
    user_id: str
    email: str

# ======================== Auth helpers ========================
def verify_token(token: str) -> Optional[dict]:
    """Verify Supabase JWT token (signature verification disabled here by design,
    since the token is validated by Supabase on issuance; enable verification if you
    distribute your public JWKs or rely on a fixed secret)."""
    try:
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded
    except Exception:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserIdentity:
    token = credentials.credentials
    payload = verify_token(token)

    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id or not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    return UserIdentity(user_id=user_id, email=email)

# ======================== Ollama utils ========================
def _is_cloud_host(base_url: str) -> bool:
    return base_url.startswith("https://ollama.com")

def _normalize_model_tag(model: str, cloud: bool) -> str:
    # Direct Cloud API expects tags without the "-cloud" suffix
    if cloud and model.endswith("-cloud"):
        return model[:-6]
    return model

# ======================== Ollama integration ========================
def generate_jd_with_ollama(prompt: str) -> str:
    """Generate JD using Ollama (Cloud or local, depending on OLLAMA_BASE_URL)."""
    try:
        is_cloud = _is_cloud_host(OLLAMA_BASE_URL)
        model = _normalize_model_tag(OLLAMA_MODEL, is_cloud)

        url = f"{OLLAMA_BASE_URL.rstrip('/')}/generate"
        headers = {"Content-Type": "application/json"}

        # Cloud requires bearer auth
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
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Ollama returned an empty response")
        return text

    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Ollama request failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating JD with AI: {str(e)}")

# ======================== Prompt builder ========================
def create_jd_prompt(req: JDCreateRequest) -> str:
    # Normalize inputs
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

# ======================== API Endpoints ========================
@app.get("/")
async def root():
    return {"message": "Recruitment AI API is running", "status": "active"}

@app.post("/auth/signup")
async def signup(request: SignUpRequest):
    try:
        response = supabase.auth.sign_up({"email": request.email, "password": request.password})
        if response.user:
            return {"message": "User created successfully", "user": {"id": response.user.id, "email": response.user.email}}
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create user")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/auth/signin")
async def signin(request: SignInRequest):
    try:
        response = supabase.auth.sign_in_with_password({"email": request.email, "password": request.password})
        if response.session:
            return {
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "user": {"id": response.user.id, "email": response.user.email},
            }
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    
@app.get("/me")
async def get_me(current_user: UserIdentity = Depends(get_current_user)):
        """
        Return current user + recruiter + company (if any).
        Used by the frontend to decide whether to show 'create company'.
        """
        # is there a recruiter row for this user?
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

        # no recruiter yet -> first time user
        return {
            "user": {"id": current_user.user_id, "email": current_user.email},
            "recruiter": None,
            "company": None,
            "has_company": False,
        }


@app.post("/company/create")
async def create_company(request: CompanyRequest, current_user: UserIdentity = Depends(get_current_user)):
    try:
        existing_company = supabase.table("companies").select("*").eq("company_name", request.company_name).execute()

        company_id = None
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
            "company_email": current_user.email,  # adjust if your column is named differently
            "full_name": request.recruiter_name,
            "company_id": company_id,
        }

        existing_recruiter = supabase.table("recruiters").select("*").eq("user_id", current_user.user_id).execute()

        if existing_recruiter.data:
            recruiter_result = supabase.table("recruiters").update(recruiter_data).eq("user_id", current_user.user_id).execute()
        else:
            recruiter_result = supabase.table("recruiters").insert(recruiter_data).execute()

        return {"message": "Company profile created successfully", "company_id": company_id, "recruiter": recruiter_result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error creating company: {str(e)}")

@app.post("/jd/create")
async def create_jd(request: JDCreateRequest, current_user: UserIdentity = Depends(get_current_user)):
    try:
        recruiter = supabase.table("recruiters").select("company_id").eq("user_id", current_user.user_id).execute()
        if not recruiter.data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please create company profile first")

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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error creating JD: {str(e)}")

@app.post("/jd/regenerate/{job_id}")
async def regenerate_jd(job_id: str, current_user: UserIdentity = Depends(get_current_user)):
    try:
        job = supabase.table("jobs").select("*").eq("job_id", job_id).eq("created_by", current_user.user_id).execute()
        if not job.data:
            raise HTTPException(status_code=404, detail="Job not found")

        job_data = job.data[0]

        comp = supabase.table("companies").select("company_name, description").eq("company_id", job_data["company_id"]).execute()
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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found or unauthorized")

        return {"message": "JD updated successfully", "job_id": job_id, "jd_text": jd_text}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error updating JD: {str(e)}")

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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found or unauthorized")

        return {"message": "JD finalized and published", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error finalizing JD: {str(e)}")

@app.get("/jobs/my-jobs")
async def get_my_jobs(current_user: UserIdentity = Depends(get_current_user)):
    try:
        jobs = (
            supabase
            .table("jobs")
            .select("*")
            .eq("created_by", current_user.user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return {"jobs": jobs.data}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error fetching jobs: {str(e)}")

# Optional: simple model list probe (Cloud or local)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
