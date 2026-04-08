"""
FastAPI server — Incident Response OpenEnv
Built by Team BitShift

Endpoints:
  GET  /                  frontend UI
  POST /reset             reset(task_id, session_id?) -> Observation
  POST /step              step(action, session_id?)   -> {observation, reward, done, info}
  GET  /state             state()        -> dict
  GET  /health            200 OK
  GET  /hint              hint for current alert
  GET  /explain           explain last action result
  GET  /streak            current/best streak
  GET  /tasks             list all tasks
  GET  /timeline          full episode replay with correct answers
  GET  /difficulty        per-alert difficulty scores
  GET  /session/export    export full session as JSON
  POST /benchmark         run all 3 tasks and return composite score
  POST /replay            replay an episode from a list of saved actions
  GET  /leaderboard       get scores
  POST /leaderboard       save score
  GET  /analytics         accuracy stats
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os, time, json, uuid
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from env import IncidentResponseEnv
from env.models import Action
from env.dataset import ALL_ALERT_MAP, TASK_CONFIGS

app = FastAPI(title="Incident Response OpenEnv — BitShift", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# IST = UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")

def now_ist_display() -> str:
    return datetime.now(IST).strftime("%d %b %Y, %I:%M:%S %p IST")

# ── Persistence ────────────────────────────────────────────────────────────
LEADERBOARD_FILE = Path("leaderboard.json")

def _load_lb() -> list:
    if LEADERBOARD_FILE.exists():
        try: return json.loads(LEADERBOARD_FILE.read_text())
        except: return []
    return []

def _save_lb(data: list):
    LEADERBOARD_FILE.write_text(json.dumps(data, indent=2))

# ── Multi-session state ────────────────────────────────────────────────────
# Each session_id gets its own env + state
_sessions: dict = {}          # session_id -> {env, analytics, sev_analytics, last_result, streak, best_streak, timeline, start, task_id}
_default_session = "default"  # backward-compat single-session

def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "env": IncidentResponseEnv(),
            "analytics": defaultdict(lambda: {"correct": 0, "total": 0}),
            "sev_analytics": defaultdict(lambda: {"correct": 0, "total": 0}),
            "last_result": {},
            "streak": 0,
            "best_streak": 0,
            "timeline": [],
            "start": time.time(),
            "task_id": "",
        }
    return _sessions[session_id]

# Keep a default env for backward-compat endpoints that don't pass session_id
_env = IncidentResponseEnv()

# ── Global analytics (across all sessions) ────────────────────────────────
_analytics: dict = defaultdict(lambda: {"correct": 0, "total": 0})
_sev_analytics: dict = defaultdict(lambda: {"correct": 0, "total": 0})
_last_result: dict = {}
_streak: int = 0
_best_streak: int = 0
_timeline: list = []
_session_start: float = time.time()
_current_task_id: str = ""

# ── Hints ──────────────────────────────────────────────────────────────────
_HINTS = {
    "a001": "Connection pool exhausted + replica lag = database overload. This is revenue-impacting.",
    "a002": "Planned maintenance with no customer impact = informational. Don't over-triage scheduled work.",
    "a003": "4,200 failed logins in 10 minutes from foreign IPs is a credential stuffing attack — security team.",
    "a004": "NullPointerException in payment processor + $1,200/min revenue loss = P1, backend team.",
    "a005": "14 days until cert expiry with failed auto-renewal = P3. Urgent but not on fire yet.",
    "a006": "Stale CDN cache affecting product listings = frontend issue, not backend.",
    "a007": "87% disk with 3 days runway = P3. Not urgent today but needs scheduling.",
    "a008": "p99 latency 8.4s + 34% error rate + 12k users = P1. N+1 query = backend.",
    "a009": "Backup completed with minor warnings = informational. No action needed urgently.",
    "a010": "Redis split-brain = database team. Session inconsistency but not total outage = P2.",
    "adv001": "Check if real user traffic is affected before escalating synthetic monitor alerts.",
    "adv002": "Cross-reference the destination IP against known scheduled jobs before flagging exfiltration.",
    "adv003": "Read the full alert — dedicated high-memory instance running a scheduled job is expected.",
    "adv004": "Analytics replica lag within SLA is not a production incident.",
}
_GENERIC_HINTS = [
    "Check if the alert source is a synthetic monitor — these can false-positive after restarts.",
    "Revenue impact and user-facing degradation are the key P1 criteria.",
    "Scheduled/planned events should never be P1 or P2.",
    "Cross-reference with recent deploys before escalating application errors.",
]

# ── Request models ─────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "easy"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    severity: str
    incident_type: str
    team: str
    status_update: Optional[str] = None
    is_false_positive: Optional[bool] = None
    confidence: Optional[float] = None
    time_taken: Optional[float] = None
    session_id: Optional[str] = None

class ReplayRequest(BaseModel):
    task_id: str
    actions: List[dict]  # list of {severity, incident_type, team, status_update?}

class LeaderboardEntry(BaseModel):
    name: str
    task_id: str
    score: float
    grade: str
    steps: int
    time_total: Optional[float] = None


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "env": "incident-response-env",
        "version": "2.0.0",
        "team": "BitShift",
        "time_ist": now_ist_display(),
        "active_sessions": len(_sessions),
    }

@app.get("/")
def index():
    return FileResponse(os.path.join("static", "index.html"))

@app.post("/reset")
def reset(req: ResetRequest):
    global _streak, _last_result, _timeline, _session_start, _current_task_id
    sid = req.session_id or _default_session

    # Per-session reset
    sess = _get_session(sid)
    sess["streak"] = 0
    sess["last_result"] = {}
    sess["timeline"] = []
    sess["start"] = time.time()
    sess["task_id"] = req.task_id
    sess["analytics"] = defaultdict(lambda: {"correct": 0, "total": 0})
    sess["sev_analytics"] = defaultdict(lambda: {"correct": 0, "total": 0})
    obs = sess["env"].reset(req.task_id)

    # Also reset default global state for backward compat
    if sid == _default_session:
        _streak = 0
        _last_result = {}
        _timeline = []
        _session_start = time.time()
        _current_task_id = req.task_id
        _env.reset(req.task_id)

    result = obs.model_dump()
    result["session_id"] = sid
    result["reset_at_ist"] = now_ist_display()
    return result

@app.post("/step")
def step(req: StepRequest):
    global _streak, _best_streak, _last_result, _timeline
    sid = req.session_id or _default_session
    sess = _get_session(sid)

    try:
        action = Action(
            severity=req.severity,
            incident_type=req.incident_type,
            team=req.team,
            status_update=req.status_update,
            is_false_positive=req.is_false_positive,
        )
        next_obs, reward, done, info = sess["env"].step(action)

        alert_id = info["alert_id"]
        alert = ALL_ALERT_MAP.get(alert_id, {})

        # Analytics (per-session + global)
        if alert:
            sev = alert.get("expected_severity", "")
            sess["sev_analytics"][sev]["total"] += 1
            _sev_analytics[sev]["total"] += 1
            if reward.breakdown.get("severity", 0) == 1.0:
                sess["sev_analytics"][sev]["correct"] += 1
                _sev_analytics[sev]["correct"] += 1
            team = alert.get("expected_team", "")
            sess["analytics"][team]["total"] += 1
            _analytics[team]["total"] += 1
            if reward.breakdown.get("team", 0) == 1.0:
                sess["analytics"][team]["correct"] += 1
                _analytics[team]["correct"] += 1

        # Streak
        if reward.value >= 0.99:
            sess["streak"] += 1
            sess["best_streak"] = max(sess["best_streak"], sess["streak"])
        else:
            sess["streak"] = 0

        if sid == _default_session:
            _streak = sess["streak"]
            _best_streak = sess["best_streak"]

        # Last result for /explain
        result_entry = {
            "alert_id": alert_id,
            "action": {"severity": req.severity, "incident_type": req.incident_type, "team": req.team, "status_update": req.status_update},
            "reward": reward.value,
            "breakdown": reward.breakdown,
            "expected_severity": alert.get("expected_severity"),
            "expected_team": alert.get("expected_team"),
            "expected_type": alert.get("expected_type"),
        }
        sess["last_result"] = result_entry
        if sid == _default_session:
            _last_result = result_entry

        # Timeline entry
        tl_entry = {
            "step": info["step"],
            "alert_id": alert_id,
            "alert_title": alert.get("title", ""),
            "alert_source": alert.get("source", ""),
            "action": {"severity": req.severity, "incident_type": req.incident_type, "team": req.team},
            "expected": {"severity": alert.get("expected_severity"), "team": alert.get("expected_team"), "incident_type": alert.get("expected_type")},
            "reward": reward.value,
            "breakdown": reward.breakdown,
            "ts": round(time.time() - sess["start"], 2),
            "ts_ist": now_ist_display(),
        }
        sess["timeline"].append(tl_entry)
        if sid == _default_session:
            _timeline.append(tl_entry)

        return {
            "observation": next_obs.model_dump() if next_obs else None,
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
            "streak": sess["streak"],
            "session_id": sid,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    return _env.state()

@app.get("/hint")
def hint():
    s = _env.state()
    if s["done"] or s["step"] >= s["total_steps"]:
        return {"hint": "Episode is complete.", "alert_id": None}
    try:
        alert_id = _env._alert_ids[_env._step_idx]
        h = _HINTS.get(alert_id)
        if not h:
            import random
            h = random.choice(_GENERIC_HINTS)
        return {"hint": h, "alert_id": alert_id}
    except Exception:
        return {"hint": "No hint available.", "alert_id": None}

@app.get("/explain")
def explain():
    if not _last_result:
        return {"message": "No action taken yet."}
    r = _last_result
    bd = r["breakdown"]
    lines = []

    if "severity" in bd:
        es = r["expected_severity"]
        ac = r["action"]["severity"]
        if bd["severity"] == 1.0:
            lines.append(f"Severity '{ac}' was correct.")
        elif bd["severity"] == 0.5:
            lines.append(f"Severity '{ac}' was close — expected '{es}' (off by one level, partial credit).")
        else:
            lines.append(f"Severity '{ac}' was wrong — expected '{es}'.")

    if "team" in bd:
        et = r["expected_team"]
        at = r["action"]["team"]
        if bd["team"] == 1.0:
            lines.append(f"Team routing to '{at}' was correct.")
        else:
            lines.append(f"Team routing to '{at}' was wrong — expected '{et}'.")

    if "incident_type" in bd:
        eit = r["expected_type"]
        ait = r["action"]["incident_type"]
        if bd["incident_type"] == 1.0:
            lines.append(f"Incident type '{ait}' was correct.")
        else:
            lines.append(f"Incident type '{ait}' was wrong — expected '{eit}'.")

    if "status_update" in bd:
        sc = bd["status_update"]
        if sc >= 0.8:
            lines.append(f"Status update quality was good (score {sc:.2f}).")
        elif sc > 0:
            lines.append(f"Status update was partial — missing some expected keywords (score {sc:.2f}).")
        else:
            lines.append("Status update was missing or too short.")

    if "false_positive" in bd:
        if bd["false_positive"] == 1.0:
            lines.append("False positive detection was correct.")
        else:
            lines.append("False positive detection was wrong.")

    return {"reward": r["reward"], "breakdown": bd, "explanation": lines}

@app.get("/streak")
def streak():
    return {"current_streak": _streak, "best_streak": _best_streak}

@app.get("/tasks")
def tasks():
    result = []
    for task_id, cfg in TASK_CONFIGS.items():
        result.append({
            "id": task_id,
            "alert_count": len(cfg["alert_ids"]),
            "grade_fields": cfg["grade_fields"],
            "description": cfg.get("description", ""),
            "persona": cfg.get("persona"),
            "noise_level": cfg.get("noise_level", 0.0),
        })
    return {"tasks": result}

@app.get("/timeline")
def timeline():
    """Full episode replay — every step with action taken vs correct answer."""
    state = _env.state()
    return {
        "task_id": _current_task_id,
        "steps": _timeline,
        "total_steps": len(_timeline),
        "mean_reward": round(sum(s["reward"] for s in _timeline) / len(_timeline), 4) if _timeline else 0.0,
        "done": state["done"],
    }


# Per-alert difficulty scores (hand-calibrated based on ambiguity)
_DIFFICULTY = {
    "a001": {"score": 0.2, "reason": "Obvious P1 — explicit revenue impact and connection pool exhausted"},
    "a002": {"score": 0.9, "reason": "Easy to over-triage — planned maintenance looks alarming"},
    "a003": {"score": 0.5, "reason": "Security alert with ambiguous severity — P2 not P1"},
    "a004": {"score": 0.3, "reason": "Clear P1 — payment failure with dollar impact stated"},
    "a005": {"score": 0.6, "reason": "P3 not P2 — 14 days runway, not on fire yet"},
    "a006": {"score": 0.7, "reason": "Tricky team routing — CDN/cache is frontend not backend"},
    "a007": {"score": 0.5, "reason": "P3 — 3 days runway, easy to over-escalate to P2"},
    "a008": {"score": 0.3, "reason": "Clear P1 — latency + error rate + user count all stated"},
    "a009": {"score": 0.8, "reason": "Easy to over-triage backup warnings — it's P4"},
    "a010": {"score": 0.6, "reason": "Redis split-brain sounds scary but it's P2 — analytics replica"},
    "adv001": {"score": 0.85, "reason": "False positive — synthetic monitor restart, easy to miss"},
    "adv002": {"score": 0.75, "reason": "False positive — scheduled export, requires cross-referencing"},
    "adv003": {"score": 0.7, "reason": "False positive — dedicated analytics host, easy to miss"},
    "adv004": {"score": 0.5, "reason": "Legit P4 — analytics replica lag within SLA"},
}

@app.get("/difficulty")
def difficulty():
    """Per-alert difficulty scores (0=trivial, 1=very hard) with reasoning."""
    return {
        "alerts": {
            aid: {**d, "alert_title": ALL_ALERT_MAP.get(aid, {}).get("title", "")}
            for aid, d in _DIFFICULTY.items()
        },
        "note": "Difficulty 0.0=trivial, 1.0=very hard to triage correctly"
    }


@app.get("/session/export")
def session_export():
    """Export the full session as JSON for reproducibility and research."""
    state = _env.state()
    rewards = [s["reward"] for s in _timeline]
    return JSONResponse({
        "env": "incident-response-env",
        "version": "2.0.0",
        "team": "BitShift",
        "task_id": _current_task_id,
        "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exported_at_ist": now_ist_display(),
        "session_duration_s": round(time.time() - _session_start, 2),
        "summary": {
            "total_steps": len(_timeline),
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "best_streak": _best_streak,
            "done": state["done"],
        },
        "timeline": _timeline,
        "analytics": {
            "teams": {t: v for t, v in _analytics.items()},
            "severities": {s: v for s, v in _sev_analytics.items()},
        }
    })


@app.post("/benchmark")
def benchmark():
    """
    Run all 3 core tasks (easy, medium, hard) back-to-back with a dummy agent
    and return a composite benchmark score. Used by evaluators to get a single
    quality number for the environment.
    """
    results = {}
    bench_env = IncidentResponseEnv()

    for task_id in ("easy", "medium", "hard"):
        obs = bench_env.reset(task_id)
        rewards = []
        done = False
        while not done:
            # Dummy agent: always picks P2 / application / backend
            action = Action(severity="P2", incident_type="application", team="backend",
                            status_update="Investigating the incident. Team has been notified.")
            next_obs, reward, done, info = bench_env.step(action)
            rewards.append(reward.value)
            if not done:
                obs = next_obs
        mean = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
        results[task_id] = {
            "steps": len(rewards),
            "mean_reward": mean,
            "rewards": rewards,
        }

    composite = round(sum(r["mean_reward"] for r in results.values()) / len(results), 4)
    return {
        "env": "incident-response-env",
        "team": "BitShift",
        "benchmark_at_ist": now_ist_display(),
        "composite_score": composite,
        "tasks": results,
        "note": "Dummy agent baseline. Replace with your model for real scores.",
    }


@app.post("/replay")
def replay(req: ReplayRequest):
    """
    Deterministic episode replay — given a saved list of actions, replay the
    full episode and return all rewards. Enables exact reproducibility from
    a session export.
    """
    if req.task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")

    replay_env = IncidentResponseEnv()
    obs = replay_env.reset(req.task_id)
    results = []
    done = False

    for i, act_dict in enumerate(req.actions):
        if done:
            break
        try:
            action = Action(
                severity=act_dict.get("severity", "P3"),
                incident_type=act_dict.get("incident_type", "application"),
                team=act_dict.get("team", "backend"),
                status_update=act_dict.get("status_update"),
                is_false_positive=act_dict.get("is_false_positive"),
            )
            next_obs, reward, done, info = replay_env.step(action)
            results.append({
                "step": i + 1,
                "alert_id": info["alert_id"],
                "action": act_dict,
                "reward": reward.value,
                "breakdown": reward.breakdown,
                "done": done,
            })
            if not done:
                obs = next_obs
        except Exception as e:
            results.append({"step": i + 1, "error": str(e)})
            break

    rewards = [r["reward"] for r in results if "reward" in r]
    return {
        "task_id": req.task_id,
        "replayed_at_ist": now_ist_display(),
        "total_steps": len(results),
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "steps": results,
    }


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
        "ts_ist": now_ist_display(),
    })
    lb.sort(key=lambda x: x["score"], reverse=True)
    lb = lb[:100]
    _save_lb(lb)
    return {"saved": True, "total": len(lb)}

@app.get("/leaderboard")
def get_leaderboard(task_id: Optional[str] = None, limit: int = 20):
    lb = _load_lb()
    if task_id:
        lb = [e for e in lb if e["task_id"] == task_id]
    return {"entries": lb[:limit], "total": len(lb)}

@app.get("/analytics")
def analytics():
    team_data = {
        t: {
            "correct": v["correct"], "total": v["total"],
            "accuracy": round(v["correct"] / v["total"], 3) if v["total"] > 0 else None,
        } for t, v in _analytics.items()
    }
    sev_data = {
        s: {
            "correct": v["correct"], "total": v["total"],
            "accuracy": round(v["correct"] / v["total"], 3) if v["total"] > 0 else None,
        } for s, v in _sev_analytics.items()
    }
    return {"teams": team_data, "severities": sev_data}
