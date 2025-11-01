# resume.py — Gemma3:4b local only, fixed prompts, Supabase schema-safe
# PARSING KEPT SAME
# META CHANGED: now only (1) skill-based marking (from files, grouped) (2) experience marking
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
SUPABASE_SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
)
JOBS_TABLE = os.getenv("JOBS_TABLE", "jobs")

RESUME_DIR_DEFAULT = "./resumes"
JD_PATH_DEFAULT = "./jd.txt"
DATA_DIR = Path("./data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

# we keep the old env var, but we will prefer local files first
LINGUIST_YAML_URL = os.getenv(
    "LINGUIST_YAML_URL",
    "https://raw.githubusercontent.com/github-linguist/linguist/HEAD/lib/linguist/languages.yml",
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_AS_OF = date.today()

# NEW: where we look for *your* skills / aliases first
SKILL_FILES = [
    Path(os.getenv("SKILLS_FILE", "")),    # user override
    Path("./skills.yaml"),
    Path("./skills.yml"),
    Path("./skills.json"),
    Path("/mnt/data/skills.yaml"),
    Path("/mnt/data/skills.yml"),
    Path("/mnt/data/skills.json"),
]
LOCAL_LINGUIST_FILES = [
    Path("/mnt/data/linguist.yml"),
    Path("/mnt/data/linguist.yaml"),
]

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
# LLM setup + prompts (unchanged parsing)
# --------------------------------------------------------------------------------------
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
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
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
# Aliasing (we prefer local; if not, old remote; then tiny fallback)
# --------------------------------------------------------------------------------------
_FALLBACK_ALIASES = {
    "reactjs": "react", "react.js": "react",
    "nodejs": "node", "node.js": "node",
    "ts": "typescript", "py": "python",
}
_ALIAS_MAP: Dict[str, str] = {}

def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9+\-.#]", "", s.lower())

def _try_load_local_linguist() -> Dict[str, str]:
    for p in LOCAL_LINGUIST_FILES:
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                out: Dict[str, str] = {}
                for lang, cfg in data.items():
                    can = _norm_token(lang)
                    if can:
                        out[can] = can
                    aliases = (cfg or {}).get("aliases") or []
                    for a in aliases:
                        na = _norm_token(a)
                        if na:
                            out[na] = can
                return out
            except Exception as e:
                logging.warning(f"Local linguist {p} failed: {e}")
    return {}

def _build_alias_map() -> Dict[str, str]:
    global _ALIAS_MAP
    if _ALIAS_MAP:
        return _ALIAS_MAP

    # 1) FAST PATH: local linguist
    local_aliases = _try_load_local_linguist()
    if local_aliases:
        _ALIAS_MAP = {**local_aliases, **_FALLBACK_ALIASES}
        return _ALIAS_MAP

    # 2) old remote behaviour (kept for compatibility)
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

# --------------------------------------------------------------------------------------
# JD fallback skill extractor (kept – only used if skill files missing)
# --------------------------------------------------------------------------------------
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
# Experience helpers (same)
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
# NEW: load required skills from file (your attached skills.*)
# --------------------------------------------------------------------------------------
def _load_required_skills_from_files() -> List[str]:
    for p in SKILL_FILES:
        if not p or not str(p):
            continue
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8")
            if p.suffix.lower() in (".yaml", ".yml"):
                data = yaml.safe_load(txt) or {}
                if isinstance(data, dict):
                    raw = data.get("skills") or []
                else:
                    raw = data
            elif p.suffix.lower() == ".json":
                data = json.loads(txt)
                if isinstance(data, dict):
                    raw = data.get("skills") or []
                else:
                    raw = data
            else:
                raw = []
        except Exception as e:
            logging.warning(f"Failed to read skills from {p}: {e}")
            continue
        skills = [str(x).strip() for x in (raw or []) if str(x).strip()]
        if skills:
            logging.info(f"Loaded {len(skills)} required skills from {p}")
            return skills
    return []

# --------------------------------------------------------------------------------------
# NEW: build grouped skills: {canonical: {aliases...}}
# --------------------------------------------------------------------------------------
def _build_skill_groups(required_skills: List[str]) -> Dict[str, Set[str]]:
    alias_map = _build_alias_map()  # already preferring local
    groups: Dict[str, Set[str]] = {}
    # 1) create groups from your file
    for s in required_skills:
        can = _canonicalize_skill(s)
        if not can:
            continue
        groups.setdefault(can, set()).add(can)
    # 2) add aliases that map to these canonicals
    for alias, can in alias_map.items():
        if can in groups:
            groups[can].add(alias)
    # small extra merges
    extra_pairs = {
        "html5": "html",
        "css3": "css",
        "js": "javascript",
    }
    for a, c in extra_pairs.items():
        na, nc = _norm_token(a), _norm_token(c)
        if nc in groups:
            groups[nc].add(na)
    return groups

# --------------------------------------------------------------------------------------
# NEW: flatten resume → tokens
# --------------------------------------------------------------------------------------
def _flatten_resume_tokens(candidate: CandidateRecord, resume_text: str) -> Set[str]:
    tokens: Set[str] = set()

    # from structured skills
    for _, vals in (candidate.skills or {}).items():
        for v in vals or []:
            tokens.add(_canonicalize_skill(v))

    # from experience technologies + description
    for e in candidate.experience or []:
        for t in (e.technologies or []):
            tokens.add(_canonicalize_skill(t))
        if e.description:
            for tok in re.findall(r"[A-Za-z0-9+.#\-]{2,}", e.description):
                tokens.add(_canonicalize_skill(tok))

    # from projects
    for p in candidate.projects or []:
        for t in (p.technologies or []):
            tokens.add(_canonicalize_skill(t))
        if p.description:
            for tok in re.findall(r"[A-Za-z0-9+.#\-]{2,}", p.description):
                tokens.add(_canonicalize_skill(tok))

    # from raw resume text
    if resume_text:
        for tok in re.findall(r"[A-Za-z0-9+.#\-]{2,}", resume_text):
            tokens.add(_canonicalize_skill(tok))

    # drop empties
    tokens = {t for t in tokens if t}
    return tokens

# --------------------------------------------------------------------------------------
# NEW: score = skill groups + experience only
# --------------------------------------------------------------------------------------
def score_candidate_skills_and_experience(
    candidate: CandidateRecord,
    resume_text: str,
    skill_groups: Dict[str, Set[str]],
    min_years: Optional[float],
    max_years: Optional[float],
) -> Dict[str, Any]:
    resume_tokens = _flatten_resume_tokens(candidate, resume_text)

    matched_groups: List[str] = []
    for can, aliases in (skill_groups or {}).items():
        if resume_tokens & aliases:
            matched_groups.append(can)

    total_groups = len(skill_groups) or 1
    skill_match_score = round(100.0 * len(matched_groups) / total_groups, 2)

    total_yoe = _total_experience_years(candidate)
    exp_score = experience_match_score(total_yoe, min_years, max_years) or 0

    # combine (tweak weights here)
    final_score = round(0.65 * skill_match_score + 0.35 * exp_score, 2)

    meta = {
        "total_experience_years": total_yoe,
        "skill_groups_total": total_groups,
        "skill_groups_matched": matched_groups,
        "skill_match_score": skill_match_score,
        "experience_match_score": exp_score,
        "rationale": f"{len(matched_groups)}/{total_groups} required skill groups matched; YOE={total_yoe}, min={min_years}, max={max_years}",
    }
    # keep this because your UI/DB expects it
    meta["top_matched_skills"] = matched_groups[:10]
    # education_score stays but we will set it outside to 0
    return {"meta": meta, "jd_match_score": final_score}

# --------------------------------------------------------------------------------------
# AI summary (same, just uses new meta)
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
    sb.table("resumes").upsert(resume_row, on_conflict="job_id,email").execute()

# --------------------------------------------------------------------------------------
# Normalize skills (same)
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

# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def run_pipeline(resume_dir: str, jd_path: str, job_id: str):
    # 1) load JD (still used for summary / fallback skills)
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read().strip()

    # 2) supabase
    sb = supabase_client()

    # 3) job context
    job = sb.table(JOBS_TABLE).select("*").eq("job_id", job_id).single().execute().data
    if not job:
        raise RuntimeError(f"No job found for job_id={job_id}")
    role = job.get("role") or "Software Engineer"
    min_years = job.get("min_years")
    max_years = job.get("max_years")

    # 4) resumes
    files = sorted(glob.glob(os.path.join(resume_dir, "*.pdf")))
    if not files:
        logging.warning(f"No PDFs found in {resume_dir}")
        return

    # 5) build alias map once
    _ = _build_alias_map()

    # 6) load required skills (file first, then JD)
    required_skills = _load_required_skills_from_files()
    if not required_skills:
        required_skills = extract_jd_skills(jd_text)
        logging.info(f"No skill file found; using JD-extracted skills: {required_skills}")
    skill_groups = _build_skill_groups(required_skills)

    for pdf in tqdm(files, desc=f"Processing resumes for job {role}"):
        file_name = os.path.basename(pdf)
        try:
            # 1) Read
            resume_text = extract_text_pymupdf(pdf)

            # 2) Parse via LLM (UNCHANGED)
            parsed = call_json(EXTRACT_PROMPT, resume_text=resume_text)
            if not isinstance(parsed, dict):
                parsed = {}
            parsed.setdefault("skills", {})
            parsed.setdefault("experience", [])
            parsed.setdefault("education", [])
            parsed.setdefault("projects", [])
            parsed.setdefault("links", {})
            parsed.setdefault("certifications", [])

            parsed["skills"] = _normalize_skills(parsed.get("skills"))
            candidate = CandidateRecord(**parsed)

            # 3) NEW scoring: only skills (from file/grouped) + experience
            scored = score_candidate_skills_and_experience(
                candidate,
                resume_text,
                skill_groups,
                min_years,
                max_years,
            )
            meta = scored.get("meta", {})
            jd_match_score = scored.get("jd_match_score", 0)

            # 4) Education → you said not needed, we just keep 0 to fill column
            edu_score = 0
            meta["education_score"] = edu_score

            # 5) Summary
            ai_sum = generate_ai_summary(candidate, meta, jd_text, jd_match_score)

            # 6) Upsert candidate
            candidate_id, final_email = upsert_candidate(sb, candidate, resume_text)

            # 7) Upsert resume (same columns as old code)
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
                "jd_match_score": jd_match_score,  # ← now this is (skills+exp) only
                "skill_match_score": meta.get("skill_match_score"),
                "experience_match_score": meta.get("experience_match_score"),
                "education_match_score": edu_score,
            }
            upsert_resume(sb, resume_row)
            logging.info(
                f"Stored: {file_name} → candidate_id={candidate_id}, skill={meta.get('skill_match_score')}, exp={meta.get('experience_match_score')}"
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
    parser = argparse.ArgumentParser(description="Unified Resume Parser → Supabase (skills+experience only)")
    parser.add_argument("--resume_dir", default=RESUME_DIR_DEFAULT)
    parser.add_argument("--jd_path", default=JD_PATH_DEFAULT)
    parser.add_argument("--job_id", default=os.getenv("JOB_ID"))
    parser.add_argument("--skills_file", default=None, help="override skills file to use for scoring")
    args = parser.parse_args()

    if args.skills_file:
        SKILL_FILES.insert(0, Path(args.skills_file))

    if not args.job_id:
        raise SystemExit("Set --job_id or JOB_ID in env.")
    if not os.path.exists(args.jd_path):
        raise SystemExit("Create jd.txt with your Job Description (or pass --jd_path).")
    os.makedirs(args.resume_dir, exist_ok=True)

    run_pipeline(args.resume_dir, args.jd_path, args.job_id)
