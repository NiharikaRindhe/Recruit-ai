# resume.py — Gemma3:4b local only, fixed prompts, robust JSON, Supabase schema-safe
# - Fixes KeyError '"skill"' by removing/escaping JSON braces in prompts
# - Uses ONLY local Ollama at http://localhost:11434 (no cloud, no API key needed)
# - Adds resilient JSON parsing + fallback heuristic judge (no LLM needed if JSON fails)
# - Matches your Supabase schema exactly (NO 'file_name' column)
# - Fields preserved: candidates + resumes columns you shared
#
# Run (PowerShell):
#   $env:SUPABASE_URL="https://...supabase.co"
#   $env:SUPABASE_SERVICE_ROLE_KEY="<service-role>"
#   $env:JOB_ID="2b1ab202-1713-4801-a571-2a4510655379"
#   $env:OLLAMA_BASE_URL="http://localhost:11434"
#   $env:OLLAMA_MODEL="gemma3:4b"
#   python resume.py --resume_dir .\resumes --jd_path .\jd.txt --job_id $env:JOB_ID

import os, re, json, glob, logging, hashlib
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import fitz  # PyMuPDF
import phonenumbers
import requests, yaml

from pydantic import BaseModel, Field, ValidationError, ConfigDict
from supabase import create_client

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from dateutil import parser as dtparser
from rapidfuzz import fuzz

# --------------------------------------------------------------------------------------
# Env & constants
# --------------------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
JOBS_TABLE = os.getenv("JOBS_TABLE", "jobs")

RESUME_DIR_DEFAULT = "./resumes"
JD_PATH_DEFAULT = "./jd.txt"
DATA_DIR = Path("./data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

LINGUIST_YAML_URL = os.getenv(
    "LINGUIST_YAML_URL",
    "https://raw.githubusercontent.com/github-linguist/linguist/HEAD/lib/linguist/languages.yml",
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_AS_OF = date.today()

# --------------------------------------------------------------------------------------
# Schema models (keeps your fields intact)
# --------------------------------------------------------------------------------------
class ExperienceItem(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)

class EducationItem(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    degree: Optional[str] = None
    institution: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    score: Optional[str] = None

class ProjectItem(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    link: Optional[str] = None
    impact: Optional[str] = None

class CandidateRecord(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    links: Dict[str, Any] = Field(default_factory=dict)
    skills: Dict[str, List[str]] = Field(default_factory=dict)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

# --------------------------------------------------------------------------------------
# PDF → text
# --------------------------------------------------------------------------------------

def extract_text_pymupdf(pdf_path: str) -> str:
    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts)

# --------------------------------------------------------------------------------------
# Heuristics
# --------------------------------------------------------------------------------------

def extract_email(text: str) -> Optional[str]:
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return m.group(0) if m else None


def extract_phone(text: str, region: str = "IN") -> Optional[str]:
    for match in phonenumbers.PhoneNumberMatcher(text, region):
        return phonenumbers.format_number(
            match.number, phonenumbers.PhoneNumberFormat.E164
        )
    return None


def split_name(full_name: Optional[str]):
    if not full_name:
        return None, None
    parts = re.split(r"\s+", full_name.strip())
    return (parts[0], parts[-1]) if parts else (None, None)

# --------------------------------------------------------------------------------------
# LLM setup + prompts (no JSON literals inside the template!)
# --------------------------------------------------------------------------------------

# Keep generation small for speed; Gemma:4b local.
llm_options = {
    "num_ctx": 3072,
    "num_predict": 800,  # extraction
    "temperature": 0.2,
}

llm_kwargs = {
    "model": OLLAMA_MODEL,
    "base_url": OLLAMA_BASE_URL,
    "temperature": 0.2,
    "options": llm_options,
}

# A second LLM with smaller output for judge/summary
llm_judge = None
try:
    llm_judge = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        options={"num_ctx": 2048, "num_predict": 400}
    )
except Exception:
    pass

llm = OllamaLLM(**llm_kwargs)

# ----- extraction prompt (no curly-brace JSON examples) -----
EXTRACT_PROMPT = PromptTemplate.from_template(
    (
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
)

# ----- judge prompt (carefully avoids any inline { ... } blocks) -----
JUDGE_PROMPT = PromptTemplate.from_template(
    (
        "Return ONLY JSON with two keys: per_skill and rationale.\n"
        "- per_skill: an array; each element is an object with the keys: skill (string), present (boolean), evidence (array of short quotes), synonyms_used (array of strings).\n"
        "- rationale: short string.\n\n"
        "Mark present true ONLY if the resume text explicitly mentions the skill or a close synonym.\n"
        "Provide 1–2 tiny evidence snippets per skill when present.\n\n"
        "JD_SKILLS (canonical list):\n{jd_skills}\n\n"
        "SYNONYM_MAP (token -> canonical):\n{syn_map}\n\n"
        "RESUME_TEXT (truncated):\n{resume_text}"
    )
)

# ----- tiny summary prompt -----
SUMMARY_PROMPT = PromptTemplate.from_template(
    (
        "Write a <=100-word hiring summary stating whether {name} fits the role \"{role}\". "
        "Mention verified skills ({skills}) and total experience ({yoe} years). "
        "Conclude with the JD match score: {score}. Plain text only. JD: {jd}"
    )
)

# --------------------------------------------------------------------------------------
# Safer JSON calling
# --------------------------------------------------------------------------------------

def _json_fragment(s: str) -> str:
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s


def _tidy_json(s: str) -> str:
    # Remove trailing commas, fix common json-ish tokens
    s = re.sub(r",\s*([}\]])", r"\\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # strip control chars
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)
    return s


def call_json(prompt: PromptTemplate, prefer_judge_llm: bool = False, **kwargs) -> dict:
    text = prompt.format(**kwargs)
    engines = [llm_judge, llm] if prefer_judge_llm and llm_judge else [llm]
    last_err = None
    for engine in engines:
        for _ in range(2):
            try:
                raw = engine.invoke(text).strip()
            except Exception as e:
                last_err = e
                continue
            frag = _json_fragment(raw)
            try:
                return json.loads(frag)
            except json.JSONDecodeError:
                pass
            frag2 = _tidy_json(frag)
            try:
                return json.loads(frag2)
            except Exception as e:
                last_err = e
                continue
    if last_err:
        raise last_err
    return {}

# --------------------------------------------------------------------------------------
# Linguist-based skill aliasing
# --------------------------------------------------------------------------------------

_FALLBACK_ALIASES = {
    "reactjs": "react", "react.js": "react",
    "nodejs": "node", "node.js": "node",
    "ts": "typescript", "py": "python",
}
_ALIAS_MAP: Dict[str, str] = {}


def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9+\-.#]", "", s.lower())


def _build_alias_map() -> Dict[str, str]:
    global _ALIAS_MAP
    if _ALIAS_MAP:
        return _ALIAS_MAP
    try:
        r = requests.get(LINGUIST_YAML_URL, timeout=10)
        r.raise_for_status()
        data = yaml.safe_load(r.text) or {}
        for lang, cfg in data.items():
            main = _norm_token(lang)
            _ALIAS_MAP[main] = main
            for a in (cfg or {}).get("aliases", []) or []:
                _ALIAS_MAP[_norm_token(a)] = main
    except Exception:
        logging.warning("Linguist alias fetch failed; using fallback map.")
    _ALIAS_MAP.update(_FALLBACK_ALIASES)
    return _ALIAS_MAP


def _canonicalize_skill(s: str) -> str:
    m = _build_alias_map()
    key = _norm_token(s)
    return m.get(key, key)


def extract_jd_skills(jd_text: str) -> List[str]:
    tokens = set(_canonicalize_skill(t) for t in re.findall(r"[A-Za-z0-9+.#\-]{2,}", jd_text))
    common = [
        "react","node","python","java","spring","django","flask","kubernetes","docker","aws","gcp","azure",
        "postgresql","mysql","mongodb","redis","graphql","rest","nextjs","typescript","terraform","ansible","jenkins"
    ]
    for c in common:
        if fuzz.QRatio(c, jd_text) >= 85:
            tokens.add(c)
    stop = {"with","and","the","a","an","or","to","in","on","for","by","of","at","as"}
    return sorted([t for t in tokens if t and t not in stop])

# --------------------------------------------------------------------------------------
# Deterministic scoring helpers
# --------------------------------------------------------------------------------------


def _parse_year(date_like: Optional[str]) -> Optional[int]:
    if not date_like:
        return None
    s = str(date_like)
    m = re.search(r"(20\d{2}|19\d{2})", s)
    if m:
        return int(m.group(1))
    try:
        return dtparser.parse(s, default=dtparser.parse("2000-01-01")).year
    except Exception:
        return None


def _total_experience_years(cand: CandidateRecord) -> float:
    total_months = 0
    for e in cand.experience or []:
        sy = _parse_year(e.start_date) or _parse_year(e.description)
        ey = _parse_year(e.end_date) or _AS_OF.year
        if sy and ey and ey >= sy:
            total_months += (ey - sy) * 12
    return round(total_months / 12.0, 2)

_EDU_ORDER = ["", "diploma", "associate", "bachelor", "be", "btech", "bs",
              "postgraduate", "master", "me", "mtech", "ms", "mba", "phd", "doctorate"]


def _edu_level(text: Optional[str]) -> int:
    if not text:
        return 0
    t = text.lower()
    best = 0
    for i, tag in enumerate(_EDU_ORDER):
        if tag and tag in t:
            best = max(best, i)
    return best


def highest_education_level(cand: CandidateRecord) -> Tuple[int, Optional[str]]:
    best, label = 0, None
    for e in cand.education or []:
        lvl = _edu_level((e.degree or "") + " " + (e.institution or ""))
        if lvl > best:
            best, label = lvl, e.degree
    return best, label


def education_match_score(required_text: Optional[str], candidate: CandidateRecord) -> Optional[int]:
    if not required_text:
        return 0
    req_lvl = _edu_level(required_text)
    cand_lvl, _ = highest_education_level(candidate)
    if cand_lvl == 0:
        return None
    diff = cand_lvl - req_lvl
    if diff > 0: return 100
    if diff == 0: return 75
    if diff == -1: return 50
    if diff == -2: return 25
    return 0


def project_relevance(candidate: CandidateRecord, jd_skills: Set[str]) -> float:
    hits = 0
    for p in candidate.projects or []:
        toks = set(_canonicalize_skill(t) for t in (p.technologies or []))
        if toks & jd_skills:
            hits += 1
    if not candidate.projects:
        return 0.0
    return round(100.0 * hits / max(1, len(candidate.projects)), 2)


def testing_devops_score(candidate: CandidateRecord) -> float:
    bag = " ".join(sum([e.technologies for e in candidate.experience or []], []))
    bag += " " + " ".join(sum([p.technologies for p in candidate.projects or []], []))
    bag = bag.lower()
    hits = sum(int(k in bag) for k in ["pytest","unittest","jest","cypress","selenium","ci/cd","jenkins","github actions","kubernetes","docker"])
    return min(100.0, hits * 20.0)


def comm_ownership_score(candidate: CandidateRecord) -> float:
    text = " ".join([(e.description or "") for e in candidate.experience or []]).lower()
    hits = sum(int(w in text) for w in ["led","owned","mentored","stakeholder","communicat","collaborat","presented"])
    return min(100.0, hits * 15.0)


def skill_match_score_from_meta(meta: dict) -> float:
    per = meta.get("per_skill") or []
    if not per:
        return 0.0
    jd_count = len(per)
    matched = sum(1 for p in per if p.get("present"))
    return round(100.0 * matched / max(1, jd_count), 2)


def experience_match_score(actual_yoe: float, min_years: float = None, max_years: float = None) -> Optional[int]:
    if actual_yoe is None:
        return None
    if min_years is None and max_years is None:
        return None
    if min_years is not None:
        if actual_yoe >= min_years and (max_years is None or actual_yoe <= max_years):
            return 100
        if actual_yoe < min_years:
            return max(0, round(100 * (actual_yoe / max(min_years, 0.01))))
        if max_years is not None and actual_yoe > max_years:
            return 100
    if max_years is not None:
        return 100 if actual_yoe <= max_years else 100
    return None

# --------------------------------------------------------------------------------------
# LLM-guarded scoring with heuristic fallback
# --------------------------------------------------------------------------------------


def _heuristic_judge(jd_skills: List[str], amap: Dict[str,str], resume_text: str) -> Dict[str, Any]:
    """No-LLM fallback judge: marks present if canonical or alias token appears."""
    lo = resume_text.lower()
    per = []
    for s in jd_skills:
        canon = _canonicalize_skill(s)
        syns = [k for k, v in amap.items() if v == canon][:8]
        present = False
        evidence = []
        for tok in [canon] + syns:
            if tok and tok in lo:
                present = True
                # crude evidence: the token itself
                evidence.append(tok)
                if len(evidence) >= 2:
                    break
        per.append({"skill": canon, "present": bool(present), "evidence": evidence, "synonyms_used": syns[:4]})
    return {"per_skill": per, "rationale": "Heuristic match on canonical tokens and aliases."}


def score_candidate_llm_guarded(candidate: CandidateRecord, jd_text: str, resume_text: str) -> Dict[str, Any]:
    jd_skills = extract_jd_skills(jd_text)
    amap = _build_alias_map()
    syn_map = {k: v for k, v in amap.items() if v in jd_skills or k in jd_skills}

    # Try LLM judge first, then fallback
    try:
        judged = call_json(
            JUDGE_PROMPT, prefer_judge_llm=True,
            jd_skills=json.dumps(jd_skills, ensure_ascii=False),
            syn_map=json.dumps(syn_map, ensure_ascii=False),
            resume_text=resume_text[:6000],
        )
        per = judged.get("per_skill") or []
        if not isinstance(per, list):
            raise ValueError("per_skill not a list")
    except Exception:
        judged = _heuristic_judge(jd_skills, amap, resume_text)
        per = judged.get("per_skill") or []

    jd_skill_set = set(jd_skills)

    total_yoe = _total_experience_years(candidate)
    proj_rel  = project_relevance(candidate, jd_skill_set)
    tdev      = testing_devops_score(candidate)
    comm      = comm_ownership_score(candidate)

    patt = r"\b(\d+%|\d+\s*(x|hrs|ms|s|k|m)|improv|increase|reduc|optimi)"
    impact_hits = 0
    for e in candidate.experience or []:
        if e.description and re.search(patt, (e.description or "").lower()):
            impact_hits += 1
    for p in candidate.projects or []:
        if p.impact and re.search(patt, (p.impact or "").lower()):
            impact_hits += 1
    impact_score = min(100.0, impact_hits * 20.0)

    skill_match_percent = skill_match_score_from_meta({"per_skill": per})

    jd_match_score = round(
        0.45 * skill_match_percent +
        0.20 * impact_score +
        0.15 * proj_rel +
        0.10 * tdev +
        0.10 * comm, 2
    )

    meta = {
        "total_experience_years": round(total_yoe, 2),
        "primary_role": candidate.meta.get("primary_role") or "Software Engineer",
        "top_matched_skills": [p.get("skill") for p in per if p.get("present")][:10],
        "skill_match_percent": skill_match_percent,
        "education_score": None,
        "project_relevance_score": round(proj_rel, 2),
        "testing_devops_score": round(tdev, 2),
        "communication_ownership_score": round(comm, 2),
        "per_skill": per,
        "rationale": f"{sum(1 for p in per if p.get('present'))}/{len(per) or 1} JD skills verified; YOE={total_yoe} yrs as of {_AS_OF}.",
    }
    return {"meta": meta, "jd_match_score": jd_match_score}

# --------------------------------------------------------------------------------------
# AI summary (<=100 words)
# --------------------------------------------------------------------------------------

def generate_ai_summary(candidate: CandidateRecord, meta: dict, jd_text: str, jd_match_score: float) -> str:
    try:
        name = candidate.full_name or f"{candidate.first_name or ''} {candidate.last_name or ''}".strip() or "Candidate"
        primary_role = meta.get("primary_role") or "Software Engineer"
        yoe = meta.get("total_experience_years")
        top_skills = ", ".join((meta.get("top_matched_skills") or [])[:6]) or "n/a"
        out = (llm_judge or llm).invoke(SUMMARY_PROMPT.format(
            name=name, role=primary_role, skills=top_skills, yoe=yoe, score=jd_match_score, jd=jd_text[:1500]
        ))
        return (out or "").strip()[:800]
    except Exception:
        return "Concise fit summary unavailable."

# --------------------------------------------------------------------------------------
# Supabase helpers
# --------------------------------------------------------------------------------------

def supabase_client():
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in env.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def stable_pseudo_email(seed: str) -> str:
    pseudo = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"{pseudo}@noemail.local"


def upsert_candidate(sb, cand: CandidateRecord, raw_text: str) -> Tuple[str, str]:
    """Upsert into public.candidates by unique email. Returns (candidate_id, final_email)."""
    email = cand.email or extract_email(raw_text)
    if not email:
        email = stable_pseudo_email((cand.full_name or "") + raw_text[:200])

    fn = cand.first_name; ln = cand.last_name
    if not (fn and ln):
        sfn, sln = split_name(cand.full_name)
        fn = fn or sfn
        ln = ln or sln

    row = {
        "email": email,
        "first_name": fn,
        "last_name": ln,
        "full_name": cand.full_name,
        "phone": cand.phone or extract_phone(raw_text),
        "location": cand.location,
        "links": cand.links or {},
    }
    sb.table("candidates").upsert(row, on_conflict="email").execute()
    got = sb.table("candidates").select("candidate_id").eq("email", email).single().execute().data
    candidate_id = got and got.get("candidate_id")
    if not candidate_id:
        raise RuntimeError("Failed to fetch candidate_id after upsert.")
    return candidate_id, email


def upsert_resume(sb, resume_row: dict):
    # STRICT: do not include columns not present in schema (e.g., file_name)
    sb.table("resumes").upsert(resume_row, on_conflict="job_id,email").execute()

# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------

def _normalize_skills(sk: Any) -> Dict[str, List[str]]:
    out = {"languages": [], "frameworks": [], "databases": [], "tools": [], "soft_skills": []}
    if isinstance(sk, dict):
        for k in out:
            v = sk.get(k)
            if isinstance(v, list):
                out[k] = [str(x).strip() for x in v if str(x).strip()]
            elif isinstance(v, str):
                out[k] = [x.strip() for x in re.split(r",|/|\n", v) if x.strip()]
    return out


def run_pipeline(resume_dir: str, jd_path: str, job_id: str):
    # Load JD
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read().strip()

    sb = supabase_client()

    # Fetch job context
    job = sb.table(JOBS_TABLE).select("*").eq("job_id", job_id).single().execute().data
    if not job:
        raise RuntimeError(f"No job found for job_id={job_id}")
    role = job.get("role") or "Software Engineer"
    min_years = job.get("min_years")
    max_years = job.get("max_years")
    education_req_text = job.get("education_req") or ""

    files = sorted(glob.glob(os.path.join(resume_dir, "*.pdf")))
    if not files:
        logging.warning(f"No PDFs found in {resume_dir}")
        return

    _ = _build_alias_map()  # prime

    for pdf in tqdm(files, desc=f"Processing resumes for job {role}"):
        file_name = os.path.basename(pdf)
        try:
            # 1) Read
            resume_text = extract_text_pymupdf(pdf)

            # 2) Parse via LLM
            parsed = call_json(EXTRACT_PROMPT, resume_text=resume_text)
            # Hardening: coerce minimal structure
            if not isinstance(parsed, dict):
                parsed = {}
            parsed.setdefault("skills", {})
            parsed.setdefault("experience", [])
            parsed.setdefault("education", [])
            parsed.setdefault("projects", [])
            parsed.setdefault("links", {})
            parsed.setdefault("certifications", [])

            # 2b) Normalize to schema types
            parsed["skills"] = _normalize_skills(parsed.get("skills"))

            # Map into CandidateRecord
            candidate = CandidateRecord(**parsed)

            # 3) Judge + composite score
            scored = score_candidate_llm_guarded(candidate, jd_text, resume_text)
            meta = scored.get("meta", {})
            jd_match_score = scored.get("jd_match_score", 0)

            # 4) Deterministic sub-scores
            edu_score = education_match_score(education_req_text, candidate)
            meta["education_score"] = edu_score
            skill_score = skill_match_score_from_meta(meta)
            exp_score = experience_match_score(meta.get("total_experience_years"), min_years, max_years)

            # 5) Summary
            ai_sum = generate_ai_summary(candidate, meta, jd_text, jd_match_score)

            # 6) Upsert candidate, get candidate_id
            candidate_id, final_email = upsert_candidate(sb, candidate, resume_text)

            # 7) Upsert resume row keyed by (job_id,email)
            resume_row = {
                "job_id": job_id,
                "candidate_id": candidate_id,
                "email": final_email,
                "status": "PENDING",
                "first_name": candidate.first_name,
                "last_name": candidate.last_name,
                "full_name": candidate.full_name,
                "phone": candidate.phone or extract_phone(resume_text),
                "location": candidate.location,
                "links": candidate.links or {},
                "skills": candidate.skills or {},
                "experience": [e.model_dump() for e in (candidate.experience or [])],
                "education": [e.model_dump() for e in (candidate.education or [])],
                "projects": [p.model_dump() for p in (candidate.projects or [])],
                "certifications": candidate.certifications or [],
                "meta": meta,
                "raw_text": resume_text,
                "role": role,
                "ai_summary": ai_sum,
                "jd_match_score": jd_match_score,
                "skill_match_score": skill_score,
                "experience_match_score": exp_score,
                "education_match_score": edu_score,
            }
            # IMPORTANT: Do NOT add file_name — your schema doesn't have it.
            upsert_resume(sb, resume_row)
            logging.info(
                f"Stored: {file_name} → candidate_id={candidate_id}, jd_match={jd_match_score}, skill={skill_score}, edu={edu_score}, exp={exp_score}"
            )

        except ValidationError as ve:
            logging.error(f"Validation error for {file_name}: {ve}")
        except Exception as e:
            logging.exception(f"Failed for {file_name}: {e}")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Resume Parser → Supabase (candidates + resumes)")
    parser.add_argument("--resume_dir", default=RESUME_DIR_DEFAULT)
    parser.add_argument("--jd_path", default=JD_PATH_DEFAULT)
    parser.add_argument("--job_id", default=os.getenv("JOB_ID"))
    args = parser.parse_args()

    if not args.job_id:
        raise SystemExit("Set --job_id or JOB_ID in env.")
    if not os.path.exists(args.jd_path):
        raise SystemExit("Create jd.txt with your Job Description (or pass --jd_path).")
    os.makedirs(args.resume_dir, exist_ok=True)

    run_pipeline(args.resume_dir, args.jd_path, args.job_id)
