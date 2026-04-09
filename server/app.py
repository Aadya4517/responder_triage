"""
FastAPI server — Incident Response OpenEnv
Built by Team BitShift

Endpoints:
  GET  /                  frontend UI
  POST /reset             reset(task_id, session_id?) -> Observation
  POST /step              step(action, session_id?)   -> {observation, reward, done, info}
  POST /step/confident    step with confidence score  -> calibration tracking
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
  GET  /autopsy           post-mortem: what went wrong and why (unique)
  GET  /heatmap           confusion matrix for severity + team routing (unique)
  GET  /compare           side-by-side session comparison (unique)
  GET  /skills            agent skill profile by source/severity/type (unique)
  GET  /calibration       confidence vs accuracy calibration report (unique)
  GET  /simulate/storm    cascading incident storm scenario (unique)
  GET  /feed              SSE live alert stream like PagerDuty (unique)
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

    model_config = {"extra": "ignore"}

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
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
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


# ── /autopsy ──────────────────────────────────────────────────────────────
@app.get("/autopsy")
def autopsy(session_id: Optional[str] = None):
    """Post-mortem: for every wrong step, explain what went wrong and how to fix it."""
    sid = session_id or _default_session
    tl = _get_session(sid)["timeline"] if session_id else _timeline
    if not tl:
        return {"message": "No episode data. Run an episode first."}
    mistakes, perfect = [], []
    for entry in tl:
        rv, act, exp, bd = entry["reward"], entry["action"], entry["expected"], entry["breakdown"]
        alert = ALL_ALERT_MAP.get(entry["alert_id"], {})
        diff = _DIFFICULTY.get(entry["alert_id"], {})
        if rv >= 0.99:
            perfect.append({"alert_id": entry["alert_id"], "title": entry["alert_title"]}); continue
        reasons, suggestions = [], []
        if bd.get("severity", 1.0) < 1.0:
            got, want = act["severity"], exp["severity"]
            reasons.append(f"Severity {'off by one' if bd['severity']==0.5 else 'wrong'}: chose {got}, expected {want}")
            sev_guide = {"P1":"revenue impact/total outage","P2":"major degradation","P3":"minor issue","P4":"informational"}
            suggestions.append(f"{want} = {sev_guide.get(want,'')}")
        if bd.get("team", 1.0) < 1.0:
            got, want = act["team"], exp["team"]
            reasons.append(f"Wrong team: routed to {got}, should be {want}")
            team_guide = {"database":"DB errors, pools, replication","backend":"app errors, payment, API",
                          "infra":"certs, disk, load balancers","security":"auth failures, login spikes","frontend":"CDN, cache, UI"}
            suggestions.append(f"{want} handles: {team_guide.get(want,'')}")
        if bd.get("incident_type", 1.0) < 1.0:
            reasons.append(f"Wrong type: chose {act['incident_type']}, expected {exp['incident_type']}")
        if bd.get("status_update", 1.0) < 0.8:
            kws = alert.get("status_keywords", [])
            reasons.append(f"Status update weak ({bd.get('status_update',0):.2f}) — missing: {kws}")
            suggestions.append(f"Include: {', '.join(kws)}")
        if bd.get("false_positive", 1.0) < 1.0:
            if alert.get("is_false_positive"):
                reasons.append("Missed false positive — look for 'scheduled','expected','authorized' in body")
            else:
                reasons.append("Incorrectly flagged as false positive — this was real")
        mistakes.append({"step": entry["step"], "alert_id": entry["alert_id"], "title": entry["alert_title"],
                         "reward": rv, "difficulty": diff.get("score"), "difficulty_reason": diff.get("reason"),
                         "what_went_wrong": reasons, "how_to_fix": suggestions})
    mean = round(sum(e["reward"] for e in tl) / len(tl), 4)
    grade_letter = "S" if mean>=0.95 else "A" if mean>=0.8 else "B" if mean>=0.65 else "C" if mean>=0.5 else "D"
    return {"grade": grade_letter, "mean_reward": mean, "total_steps": len(tl),
            "perfect_steps": len(perfect), "mistake_steps": len(mistakes),
            "perfect": perfect, "mistakes": mistakes,
            "verdict": {"S":"Excellent — near-perfect.","A":"Strong with minor gaps.","B":"Decent but missing signals.",
                        "C":"Needs improvement.","D":"Significant gaps — study alert patterns."}.get(grade_letter)}


# ── /heatmap — confusion matrix ───────────────────────────────────────────
@app.get("/heatmap")
def heatmap(session_id: Optional[str] = None):
    """Confusion matrix: severity and team routing mistakes visualized."""
    sid = session_id or _default_session
    tl = _get_session(sid)["timeline"] if session_id else _timeline
    if not tl:
        return {"message": "No episode data."}
    SEVS = ["P1","P2","P3","P4"]
    TEAMS = ["backend","infra","security","database","frontend"]
    sev_m = {s: {t: 0 for t in SEVS} for s in SEVS}
    team_m = {t: {u: 0 for u in TEAMS} for t in TEAMS}
    for e in tl:
        es, as_ = e["expected"].get("severity"), e["action"].get("severity")
        et, at = e["expected"].get("team"), e["action"].get("team")
        if es in sev_m and as_ in sev_m: sev_m[es][as_] += 1
        if et in team_m and at in team_m: team_m[et][at] += 1
    biases = sorted([{"expected":e,"predicted":p,"count":c} for e,row in sev_m.items()
                     for p,c in row.items() if c>0 and e!=p], key=lambda x: x["count"], reverse=True)
    return {"severity_confusion": sev_m, "team_confusion": team_m,
            "top_severity_mistakes": biases[:5], "note": "Rows=expected, Columns=predicted. Diagonal=correct."}


# ── /compare — side-by-side session comparison ────────────────────────────
@app.get("/compare")
def compare(session_a: str, session_b: str):
    """Compare two sessions side-by-side — human vs AI, or two model runs."""
    missing = [s for s in [session_a, session_b] if s not in _sessions]
    if missing:
        raise HTTPException(status_code=404, detail=f"Sessions not found: {missing}")
    tl_a = {e["alert_id"]: e for e in _sessions[session_a]["timeline"]}
    tl_b = {e["alert_id"]: e for e in _sessions[session_b]["timeline"]}
    common = set(tl_a) & set(tl_b)
    diffs = sorted([{"alert_id": aid, "title": tl_a[aid]["alert_title"],
                     "session_a": tl_a[aid]["reward"], "session_b": tl_b[aid]["reward"],
                     "delta": round(tl_a[aid]["reward"] - tl_b[aid]["reward"], 4),
                     "winner": "A" if tl_a[aid]["reward"] > tl_b[aid]["reward"] else
                               "B" if tl_b[aid]["reward"] > tl_a[aid]["reward"] else "tie"}
                    for aid in common], key=lambda x: abs(x["delta"]), reverse=True)
    ma = round(sum(e["reward"] for e in _sessions[session_a]["timeline"]) / len(_sessions[session_a]["timeline"]), 4)
    mb = round(sum(e["reward"] for e in _sessions[session_b]["timeline"]) / len(_sessions[session_b]["timeline"]), 4)
    return {"session_a": session_a, "session_b": session_b, "mean_a": ma, "mean_b": mb,
            "winner": "A" if ma > mb else "B" if mb > ma else "tie",
            "margin": round(abs(ma - mb), 4), "per_alert": diffs}


# ── /skills — agent skill profile ────────────────────────────────────────
@app.get("/skills")
def skills(session_id: Optional[str] = None):
    """Skill profile: strengths and weaknesses by alert source, severity, and type."""
    sid = session_id or _default_session
    tl = _get_session(sid)["timeline"] if session_id else _timeline
    if not tl:
        return {"message": "No episode data."}
    by_src, by_sev, by_type = {}, {}, {}
    for entry in tl:
        alert = ALL_ALERT_MAP.get(entry["alert_id"], {})
        for bucket, key in [(by_src, alert.get("source","?")),
                            (by_sev, alert.get("expected_severity","?")),
                            (by_type, alert.get("expected_type","?"))]:
            bucket.setdefault(key, []).append(entry["reward"])
    def summarize(d):
        return {k: {"count": len(v), "mean": round(sum(v)/len(v),3),
                    "rating": "strong" if sum(v)/len(v)>=0.85 else "weak" if sum(v)/len(v)<0.5 else "average"}
                for k, v in d.items()}
    all_s = {**summarize(by_src), **summarize(by_sev), **summarize(by_type)}
    return {"by_source": summarize(by_src), "by_severity": summarize(by_sev), "by_type": summarize(by_type),
            "strengths": [k for k,v in all_s.items() if v["rating"]=="strong"],
            "weaknesses": [k for k,v in all_s.items() if v["rating"]=="weak"]}


# ── /calibration — confidence vs accuracy ────────────────────────────────
_confidence_log: list = []

class ConfidentStepRequest(BaseModel):
    severity: str
    incident_type: str
    team: str
    confidence: float
    status_update: Optional[str] = None
    is_false_positive: Optional[bool] = None
    session_id: Optional[str] = None

@app.post("/step/confident")
def step_confident(req: ConfidentStepRequest):
    """Like /step but accepts a confidence score (0-1). Tracks calibration."""
    sid = req.session_id or _default_session
    sess = _get_session(sid)
    action = Action(severity=req.severity, incident_type=req.incident_type, team=req.team,
                    status_update=req.status_update, is_false_positive=req.is_false_positive)
    next_obs, reward, done, info = sess["env"].step(action)
    _confidence_log.append({"confidence": req.confidence, "reward": reward.value,
                             "correct": reward.value >= 0.99, "alert_id": info["alert_id"]})
    return {"observation": next_obs.model_dump() if next_obs else None,
            "reward": reward.model_dump(), "done": done, "info": info}

@app.get("/calibration")
def calibration():
    """Are high-confidence decisions actually more accurate? Calibration report."""
    if not _confidence_log:
        return {"message": "No confident steps yet. Use POST /step/confident"}
    buckets = {"low (0-0.4)": [], "medium (0.4-0.7)": [], "high (0.7-1.0)": []}
    for e in _confidence_log:
        b = "low (0-0.4)" if e["confidence"]<0.4 else "high (0.7-1.0)" if e["confidence"]>=0.7 else "medium (0.4-0.7)"
        buckets[b].append(e["reward"])
    result = {b: {"count": len(v), "mean_reward": round(sum(v)/len(v),3) if v else None} for b,v in buckets.items()}
    hi = result["high (0.7-1.0)"]["mean_reward"]
    overconfident = hi is not None and hi < 0.6 and result["high (0.7-1.0)"]["count"] >= 3
    return {"buckets": result, "total_steps": len(_confidence_log),
            "verdict": "Overconfident — high confidence but low accuracy." if overconfident else "Well calibrated."}


# ── /simulate/storm — cascading incident storm ────────────────────────────
@app.get("/simulate/storm")
def simulate_storm():
    """
    Returns a synthetic cascading incident storm scenario — 3 correlated alerts
    that fired simultaneously. Tests how well an agent handles correlated failures.
    """
    return {
        "description": "Cascading incident storm — 3 correlated alerts fired simultaneously",
        "context": "DB outage → payment failures → auth retry storm",
        "alerts": [
            {"id": "storm_001", "title": "CRITICAL: Primary DB connection pool exhausted",
             "source": "PagerDuty", "body": "prod-db-primary: all 500 connections used. API latency spiking. Revenue impact starting.",
             "expected_severity": "P1", "expected_type": "database", "expected_team": "database"},
            {"id": "storm_002", "title": "CRITICAL: Payment service cascading failure",
             "source": "Datadog", "body": "checkout-service 503s — downstream of DB outage. $2,400/min revenue loss.",
             "expected_severity": "P1", "expected_type": "application", "expected_team": "backend"},
            {"id": "storm_003", "title": "WARNING: Auth service CPU 98% — retry storm",
             "source": "CloudWatch", "body": "auth-service CPU 98%. Retry storms from payment failures flooding auth endpoints.",
             "expected_severity": "P2", "expected_type": "application", "expected_team": "backend"},
        ],
        "tip": "Root cause is storm_001. storm_002 and storm_003 are downstream effects.",
        "scoring": "Use POST /reset + POST /step to score your triage decisions.",
    }


# ── /feed — SSE live alert stream ─────────────────────────────────────────
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/feed")
async def alert_feed(task_id: str = "medium"):
    """SSE stream — delivers alerts one-by-one with 2s delays, like a real PagerDuty feed."""
    async def generate():
        cfg = TASK_CONFIGS.get(task_id, TASK_CONFIGS["medium"])
        ids = cfg["alert_ids"]
        yield f"data: {json.dumps({'type':'start','task':task_id,'total':len(ids)})}\n\n"
        for i, aid in enumerate(ids):
            await asyncio.sleep(2)
            a = ALL_ALERT_MAP.get(aid, {})
            yield f"data: {json.dumps({'type':'alert','index':i,'id':aid,'title':a.get('title',''),'source':a.get('source',''),'body':a.get('body',''),'timestamp':a.get('timestamp','')})}\n\n"
        yield f"data: {json.dumps({'type':'end','total_sent':len(ids)})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


def main():

if __name__ == "__main__":
    main()
