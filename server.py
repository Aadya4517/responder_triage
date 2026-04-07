"""
FastAPI server exposing the EmailTriageEnv over HTTP.

Endpoints:
  GET  /              serves the frontend UI
  POST /reset         reset(task_id) -> Observation
  POST /step          step(action)   -> {observation, reward, done, info}
  GET  /state         state()        -> dict
  GET  /health        health check   -> 200
  GET  /hint          hint for current email
  POST /leaderboard   save a score entry
  GET  /leaderboard   get top scores
  GET  /analytics     per-category accuracy stats
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, time, json, re
from pathlib import Path
from collections import defaultdict

from env import EmailTriageEnv
from env.models import Action
from env.dataset import ALL_EMAIL_MAP

app = FastAPI(title="Email Triage OpenEnv", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env = EmailTriageEnv()

# ── Leaderboard persistence ────────────────────────────────────────────────
LEADERBOARD_FILE = Path("leaderboard.json")

def _load_lb() -> list:
    if LEADERBOARD_FILE.exists():
        try:
            return json.loads(LEADERBOARD_FILE.read_text())
        except Exception:
            return []
    return []

def _save_lb(data: list):
    LEADERBOARD_FILE.write_text(json.dumps(data, indent=2))

# ── Analytics accumulator ─────────────────────────────────────────────────
# {category: {correct: int, total: int}}
_analytics: dict = defaultdict(lambda: {"correct": 0, "total": 0})
# {priority: {correct: int, total: int}}
_priority_analytics: dict = defaultdict(lambda: {"correct": 0, "total": 0})

# ── Hint templates ────────────────────────────────────────────────────────
_HINTS = {
    "e001": "The subject says URGENT and mentions production being down — that's a critical ops issue.",
    "e002": "Gift card promos from unknown domains are classic spam patterns.",
    "e003": "An overdue invoice from a vendor needs prompt financial attention.",
    "e004": "Internal meeting agendas are routine — not urgent.",
    "e005": "A 200-engineer company evaluating your platform is a high-value sales lead.",
    "e006": "A user locked out for 2 hours is a high-priority support issue.",
    "e007": "Casual internal chatter about lunch is low priority.",
    "e008": "A refund request is a billing matter — normal priority since the package is unopened.",
    "e009": "A security researcher reporting an SQL injection is extremely urgent — treat as critical.",
    "e010": "SaaS discount emails are promotional spam.",
    "adv001": "Look at the sender domain carefully — 'paypa1' with a number is a typosquat.",
    "adv002": "Corporate IT never asks you to reset passwords via email links.",
    "adv003": "Requests to change bank account details mid-invoice are a classic BEC (Business Email Compromise) attack.",
    "adv004": "This is a legitimate Stripe billing confirmation — no action needed.",
}

_GENERIC_HINTS = [
    "Check the sender domain carefully for typos or lookalike characters.",
    "Urgency language combined with an unknown sender is a red flag.",
    "Internal emails from known colleagues are usually low priority.",
    "Customer-facing issues (support/billing) generally rank higher than internal ones.",
]


# ── Models ────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "easy"

class StepRequest(BaseModel):
    priority: str
    category: str
    reply: Optional[str] = None
    is_phishing: Optional[bool] = None
    confidence: Optional[float] = None   # 0.0–1.0 from UI slider
    time_taken: Optional[float] = None   # seconds

class LeaderboardEntry(BaseModel):
    name: str
    task_id: str
    score: float
    grade: str
    steps: int
    time_total: Optional[float] = None


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "env": "email-triage-env", "version": "2.0.0"}

@app.get("/")
def index():
    return FileResponse(os.path.join("static", "index.html"))

@app.post("/reset")
def reset(req: ResetRequest):
    obs = _env.reset(req.task_id)
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
    try:
        action = Action(
            priority=req.priority,
            category=req.category,
            reply=req.reply,
            is_phishing=req.is_phishing,
        )
        next_obs, reward, done, info = _env.step(action)

        # Update analytics
        state = _env.state()
        email_id = info["email_id"]
        email = ALL_EMAIL_MAP.get(email_id, {})
        if email:
            cat = email.get("expected_category", "")
            pri = email.get("expected_priority", "")
            _analytics[cat]["total"] += 1
            if reward.breakdown.get("category", 0) == 1.0:
                _analytics[cat]["correct"] += 1
            _priority_analytics[pri]["total"] += 1
            if reward.breakdown.get("priority", 0) == 1.0:
                _priority_analytics[pri]["correct"] += 1

        return {
            "observation": next_obs.model_dump() if next_obs else None,
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    return _env.state()

@app.get("/hint")
def hint():
    """Return a contextual hint for the current email."""
    s = _env.state()
    if s["done"] or s["step"] >= s["total_steps"]:
        return {"hint": "Episode is complete.", "email_id": None}
    # peek at current email id from env internals
    try:
        email_id = _env._email_ids[_env._step_idx]
        h = _HINTS.get(email_id)
        if not h:
            import random
            h = random.choice(_GENERIC_HINTS)
        return {"hint": h, "email_id": email_id}
    except Exception:
        return {"hint": "No hint available.", "email_id": None}

@app.post("/leaderboard")
def post_leaderboard(entry: LeaderboardEntry):
    lb = _load_lb()
    lb.append({
        "name": entry.name[:32],
        "task_id": entry.task_id,
        "score": round(entry.score, 4),
        "grade": entry.grade,
        "steps": entry.steps,
        "time_total": entry.time_total,
        "ts": int(time.time()),
    })
    # Keep top 100 by score
    lb.sort(key=lambda x: x["score"], reverse=True)
    lb = lb[:100]
    _save_lb(lb)
    rank = next((i+1 for i, e in enumerate(lb) if e["ts"] == lb[lb.index(next(x for x in lb if x["score"] == entry.score and x["name"] == entry.name[:32]), 0)]["ts"]), 1)
    return {"saved": True, "rank": rank, "total": len(lb)}

@app.get("/leaderboard")
def get_leaderboard(task_id: Optional[str] = None, limit: int = 20):
    lb = _load_lb()
    if task_id:
        lb = [e for e in lb if e["task_id"] == task_id]
    return {"entries": lb[:limit], "total": len(lb)}

@app.get("/analytics")
def analytics():
    cat_data = {
        cat: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": round(v["correct"] / v["total"], 3) if v["total"] > 0 else None,
        }
        for cat, v in _analytics.items()
    }
    pri_data = {
        pri: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": round(v["correct"] / v["total"], 3) if v["total"] > 0 else None,
        }
        for pri, v in _priority_analytics.items()
    }
    return {"categories": cat_data, "priorities": pri_data}
