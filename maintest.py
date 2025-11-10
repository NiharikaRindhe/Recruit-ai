Absolutely—let’s add a **bulk shortlist** flow so a recruiter can select one or more candidates and flip their status from `PARSED` → `SHORTLISTED`.

Below are drop-in changes for your current file.

---

## 1) Allow the new status

Find this line and **add `SHORTLISTED`**:

```python
ALLOWED_RESUME_STATUSES = {"PENDING", "PARSED", "REJECTED", "SHORTLISTED"}
```

(Also update the docstring on `/jobs/{job_id}/resumes` if you want the filter text to mention it.)

---

## 2) Request model

Add this Pydantic model near your other models:

```python
class BulkShortlistRequest(BaseModel):
    resume_ids: Optional[List[str]] = None          # shortlist by resume_id(s)
    emails: Optional[List[EmailStr]] = None         # or by candidate email(s)
    only_if_status: str = "PARSED"                  # safety: only update when current status == PARSED
```

---

## 3) Endpoint: bulk shortlist

Paste this endpoint (I’d place it right below `/jobs/{job_id}/resumes`):

```python
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
```

> Notes
> • This only updates rows that currently match `only_if_status` (defaults to `PARSED`).
> • If you want to record who/when did the shortlist, add `shortlisted_by` (uuid) and `shortlisted_at` (timestamptz) columns to `resumes`, then include them in the `update({...})` dict.
> • The endpoint is idempotent (calling again on already-shortlisted rows won’t increase `updated` because of the `only_if_status` guard).

---

## 4) (Optional) DB migration

If `resumes.status` is a Postgres **enum** or has a **CHECK** constraint, update it to allow `SHORTLISTED`.

* **If enum** (example name: `resume_status`):

```sql
ALTER TYPE resume_status ADD VALUE IF NOT EXISTS 'SHORTLISTED';
```

* **If CHECK constraint**:

```sql
ALTER TABLE resumes DROP CONSTRAINT IF EXISTS resumes_status_check;
ALTER TABLE resumes
  ADD CONSTRAINT resumes_status_check
  CHECK (status IN ('PENDING','PARSED','REJECTED','SHORTLISTED'));
```

---

## 5) Postman tests (quick)

1. **List current resumes**
   GET `/jobs/{job_id}/resumes?status=PARSED`
   Header: `Authorization: Bearer <recruiter_access_token>`

2. **Shortlist by ids**
   POST `/jobs/{job_id}/resumes/shortlist`
   Header: `Authorization: Bearer <recruiter_access_token>`
   Body:

```json
{
  "resume_ids": ["<uuid-1>", "<uuid-2>"]
}
```

3. **Shortlist by emails**

```json
{
  "emails": ["cand1@example.com", "cand2@example.com"]
}
```

4. **Verify**
   GET `/jobs/{job_id}/resumes?status=SHORTLISTED`

---

If you later want **bulk reject** or **revert to PARSED**, you can generalize the endpoint to `/jobs/{job_id}/resumes/status` with a `status` field in the body. For now, the above keeps it clean and focused on shortlisting.
