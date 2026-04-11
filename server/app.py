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
  GET  /explain/sre       "What would a senior SRE do?" mentor explainer (unique)
  GET  /explain/last      Explain last action with SRE reasoning (unique)
  POST /fingerprint       Alert DNA — auto-suggest severity+team from text (unique)
  GET  /daily/challenge   Daily challenge — same alerts for everyone today (unique)
  POST /daily/reset       Start a daily challenge session (unique)
  POST /daily/submit      Submit score to daily leaderboard (unique)
  GET  /daily/leaderboard Today's daily challenge rankings (unique)
  POST /step/timed        Timed triage — speed bonus for fast P1 decisions (unique)
  GET  /speed/leaderboard Fastest triage times leaderboard (unique)
  GET  /drift             Severity drift detector — are you systematically biased? (unique)
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


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1: AI EXPLAINER — "What would a senior SRE do?"
# After each step, explains why the correct answer was right using LLM-style
# rule-based reasoning (no external API needed — deterministic expert system)
# ══════════════════════════════════════════════════════════════════════════════

_SRE_WISDOM = {
    "severity": {
        "P1": [
            "P1 = revenue impact or total outage. Look for: dollar amounts, '100% failing', 'all users affected'.",
            "If the alert mentions payment failures, checkout errors, or complete service unavailability — that's P1.",
            "P1 requires immediate response. Revenue loss per minute is the clearest signal.",
        ],
        "P2": [
            "P2 = major degradation but not total outage. Partial failures, elevated error rates, security incidents.",
            "Security breaches (credential stuffing, login spikes) are typically P2 — serious but not revenue-stopping.",
            "P2 means the service is degraded but still partially functional. Escalate fast but don't panic.",
        ],
        "P3": [
            "P3 = minor issue with a workaround. Cert expiry with days of runway, disk at 80%, CDN cache issues.",
            "If there's a deadline but no immediate impact, it's P3. Schedule it, don't page at 3am.",
            "P3 alerts need attention within hours, not minutes. No user impact yet.",
        ],
        "P4": [
            "P4 = informational. Planned maintenance, successful backups with minor warnings, scheduled jobs.",
            "If the alert says 'planned', 'scheduled', 'expected', or 'no customer impact' — it's P4.",
            "P4 is noise you need to track but not act on urgently. Don't over-triage scheduled work.",
        ],
    },
    "team": {
        "database": "Database team owns: connection pools, replication lag, query performance, DB crashes.",
        "backend": "Backend team owns: application errors, payment processing, API failures, NullPointerExceptions.",
        "infra": "Infra team owns: SSL certs, disk space, load balancers, cloud infrastructure, scheduled maintenance.",
        "security": "Security team owns: login spikes, credential stuffing, data exfiltration, auth failures.",
        "frontend": "Frontend team owns: CDN issues, cache staleness, UI rendering, static asset delivery.",
    },
    "false_positive": {
        True: "False positive signals: 'scheduled', 'expected', 'authorized', 'planned', 'within SLA', 'dedicated host'.",
        False: "Real incident signals: revenue impact, user-facing errors, unexpected spikes, cascading failures.",
    }
}

@app.get("/explain/sre")
def sre_explainer(alert_id: str, session_id: Optional[str] = None):
    """
    'What would a senior SRE do?' — deterministic expert explanation for any alert.
    Returns the correct triage decision with reasoning, like a mentor reviewing your call.
    """
    alert = ALL_ALERT_MAP.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    exp_sev = alert.get("expected_severity", "P3")
    exp_team = alert.get("expected_team", "backend")
    exp_type = alert.get("expected_type", "application")
    is_fp = alert.get("is_false_positive", False)
    diff = _DIFFICULTY.get(alert_id, {})

    import random
    sev_tips = _SRE_WISDOM["severity"].get(exp_sev, [])
    sev_tip = random.choice(sev_tips) if sev_tips else ""
    team_tip = _SRE_WISDOM["team"].get(exp_team, "")
    fp_tip = _SRE_WISDOM["false_positive"].get(is_fp, "")

    # Extract key signals from alert body
    body = alert.get("body", "").lower()
    signals = []
    if any(w in body for w in ["revenue", "$/min", "dollar", "payment", "checkout"]):
        signals.append("💰 Revenue impact detected")
    if any(w in body for w in ["100%", "all ", "complete", "total"]):
        signals.append("🔴 Total/complete failure signal")
    if any(w in body for w in ["scheduled", "planned", "maintenance", "expected"]):
        signals.append("📅 Planned/scheduled event signal")
    if any(w in body for w in ["false", "authorized", "within sla", "dedicated"]):
        signals.append("✅ False positive signal")
    if any(w in body for w in ["login", "credential", "brute", "attack", "foreign"]):
        signals.append("🔒 Security threat signal")
    if any(w in body for w in ["replica", "lag", "connection pool", "too many connections"]):
        signals.append("🗄️ Database overload signal")

    return {
        "alert_id": alert_id,
        "alert_title": alert.get("title", ""),
        "correct_answer": {
            "severity": exp_sev,
            "team": exp_team,
            "incident_type": exp_type,
            "is_false_positive": is_fp,
        },
        "difficulty": diff.get("score"),
        "difficulty_reason": diff.get("reason"),
        "key_signals": signals,
        "senior_sre_says": {
            "severity_reasoning": sev_tip,
            "team_reasoning": team_tip,
            "false_positive_note": fp_tip if is_fp else None,
        },
        "one_liner": f"This is {exp_sev} → {exp_team} team. {diff.get('reason', '')}",
    }


@app.get("/explain/last")
def explain_last_sre(session_id: Optional[str] = None):
    """Explain the last action taken with full SRE mentor reasoning."""
    sid = session_id or _default_session
    sess = _get_session(sid)
    last = sess["last_result"] if session_id else _last_result
    if not last:
        return {"message": "No action taken yet."}
    alert_id = last.get("alert_id")
    if not alert_id:
        return {"message": "No alert_id in last result."}
    base = sre_explainer(alert_id, session_id)
    base["your_answer"] = last.get("action", {})
    base["your_reward"] = last.get("reward", 0)
    base["breakdown"] = last.get("breakdown", {})
    # Was it correct?
    correct = last.get("reward", 0) >= 0.99
    base["verdict"] = "✅ Correct call!" if correct else "❌ Missed this one — here's what a senior SRE would do:"
    return base


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2: ALERT DNA FINGERPRINTING — auto-suggest severity + team
# Analyzes alert body with keyword heuristics before user submits
# ══════════════════════════════════════════════════════════════════════════════

_SEVERITY_SIGNALS = {
    "P1": ["revenue", "$/min", "payment", "checkout", "100% fail", "all users", "complete outage",
           "nullpointerexception", "503", "down", "unreachable", "critical"],
    "P2": ["degraded", "elevated", "partial", "login spike", "credential", "brute force",
           "split-brain", "latency", "error rate", "p99", "security"],
    "P3": ["expir", "disk", "cdn", "cache", "runway", "warning", "notice", "renewal"],
    "P4": ["scheduled", "planned", "maintenance", "informational", "backup", "completed",
           "no customer impact", "expected", "authorized", "within sla"],
}
_TEAM_SIGNALS = {
    "database": ["database", "db", "connection pool", "replica", "replication", "postgres", "mysql", "redis"],
    "backend": ["payment", "checkout", "api", "503", "nullpointer", "application", "service"],
    "infra": ["ssl", "cert", "disk", "load balancer", "maintenance", "infrastructure", "cloud"],
    "security": ["login", "credential", "brute", "attack", "foreign ip", "exfiltration", "auth"],
    "frontend": ["cdn", "cache", "static", "ui", "rendering", "frontend"],
}

@app.post("/fingerprint")
def fingerprint_alert(req: dict):
    """
    Alert DNA Fingerprinting — analyze alert text and suggest severity + team
    with confidence scores. Like autocomplete for incident triage.
    """
    text = (req.get("title", "") + " " + req.get("body", "")).lower()

    sev_scores = {}
    for sev, keywords in _SEVERITY_SIGNALS.items():
        hits = sum(1 for kw in keywords if kw in text)
        sev_scores[sev] = hits

    team_scores = {}
    for team, keywords in _TEAM_SIGNALS.items():
        hits = sum(1 for kw in keywords if kw in text)
        team_scores[team] = hits

    best_sev = max(sev_scores, key=sev_scores.get)
    best_team = max(team_scores, key=team_scores.get)
    total_sev = sum(sev_scores.values()) or 1
    total_team = sum(team_scores.values()) or 1

    sev_conf = round(sev_scores[best_sev] / total_sev, 2)
    team_conf = round(team_scores[best_team] / total_team, 2)

    # Fallback if no signals
    if sev_scores[best_sev] == 0:
        best_sev = "P3"
        sev_conf = 0.1
    if team_scores[best_team] == 0:
        best_team = "backend"
        team_conf = 0.1

    return {
        "suggested_severity": best_sev,
        "severity_confidence": sev_conf,
        "suggested_team": best_team,
        "team_confidence": team_conf,
        "severity_breakdown": {k: round(v/total_sev, 2) for k, v in sev_scores.items()},
        "team_breakdown": {k: round(v/total_team, 2) for k, v in team_scores.items()},
        "note": "AI suggestion based on alert text analysis. Always verify before submitting.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 3: DAILY CHALLENGE — same alerts for everyone, seeded by date
# ══════════════════════════════════════════════════════════════════════════════

DAILY_LB_FILE = Path("daily_leaderboard.json")

def _load_daily_lb() -> dict:
    if DAILY_LB_FILE.exists():
        try: return json.loads(DAILY_LB_FILE.read_text())
        except: return {}
    return {}

def _save_daily_lb(data: dict):
    DAILY_LB_FILE.write_text(json.dumps(data, indent=2))

@app.get("/daily/challenge")
def daily_challenge():
    """
    Daily Challenge — same 5 alerts for everyone today, seeded by date.
    Like Wordle for SREs. Resets at midnight UTC.
    """
    import hashlib
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    seed = int(hashlib.md5(today.encode()).hexdigest(), 16) % 10000

    # Pick 5 alerts deterministically from the pool
    all_ids = list(ALL_ALERT_MAP.keys())
    import random as rnd
    rnd.seed(seed)
    daily_ids = rnd.sample(all_ids, min(5, len(all_ids)))

    alerts_preview = []
    for aid in daily_ids:
        a = ALL_ALERT_MAP[aid]
        alerts_preview.append({
            "id": aid,
            "source": a.get("source", ""),
            "title": a.get("title", ""),
            # Don't reveal expected answers
        })

    return {
        "date": today,
        "seed": seed,
        "alert_ids": daily_ids,
        "alerts": alerts_preview,
        "task_id": "daily_" + today,
        "note": "Same 5 alerts for everyone today. Resets at midnight UTC.",
        "how_to_play": "POST /daily/reset to start, then POST /step as normal.",
    }

@app.post("/daily/reset")
def daily_reset(req: Optional[ResetRequest] = None):
    """Start a daily challenge session."""
    import hashlib, random as rnd
    if req is None:
        req = ResetRequest()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    seed = int(hashlib.md5(today.encode()).hexdigest(), 16) % 10000
    all_ids = list(ALL_ALERT_MAP.keys())
    rnd.seed(seed)
    daily_ids = rnd.sample(all_ids, min(5, len(all_ids)))

    sid = req.session_id or ("daily_" + today + "_" + str(int(time.time())))
    sess = _get_session(sid)
    sess["streak"] = 0
    sess["last_result"] = {}
    sess["timeline"] = []
    sess["start"] = time.time()
    sess["task_id"] = "daily"
    sess["analytics"] = defaultdict(lambda: {"correct": 0, "total": 0})
    sess["sev_analytics"] = defaultdict(lambda: {"correct": 0, "total": 0})

    # Manually set up the env with daily alert IDs
    env = sess["env"]
    env._task_id = "daily"
    env._alert_ids = daily_ids
    env._grade_fields = ["severity", "team"]
    env._persona = None
    env._noise_level = 0.0
    env._step_idx = 0
    env._episode_rewards = []
    env._done = False
    env._alert_cache = {aid: ALL_ALERT_MAP[aid] for aid in daily_ids}

    obs = env._make_obs()
    result = obs.model_dump()
    result["session_id"] = sid
    result["daily_date"] = today
    result["total_steps"] = len(daily_ids)
    return result

@app.post("/daily/submit")
def daily_submit(entry: LeaderboardEntry):
    """Submit score to daily leaderboard."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lb = _load_daily_lb()
    if today not in lb:
        lb[today] = []
    lb[today].append({
        "name": entry.name[:32],
        "score": round(entry.score, 4),
        "grade": entry.grade,
        "steps": entry.steps,
        "ts": int(time.time()),
        "ts_ist": now_ist_display(),
    })
    lb[today].sort(key=lambda x: x["score"], reverse=True)
    lb[today] = lb[today][:100]
    _save_daily_lb(lb)
    rank = next((i+1 for i, e in enumerate(lb[today]) if e["ts"] == lb[today][-1]["ts"]), len(lb[today]))
    return {"saved": True, "date": today, "rank": rank, "total": len(lb[today])}

@app.get("/daily/leaderboard")
def daily_leaderboard(limit: int = 20):
    """Get today's daily challenge leaderboard."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lb = _load_daily_lb()
    entries = lb.get(today, [])
    return {
        "date": today,
        "entries": entries[:limit],
        "total": len(entries),
        "resets_in": _time_until_midnight(),
    }

def _time_until_midnight() -> str:
    now = datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    delta = midnight - now
    h, m = divmod(delta.seconds // 60, 60)
    return f"{h}h {m}m"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 4: TIME-TO-TRIAGE SCORING — speed bonus for fast P1 decisions
# ══════════════════════════════════════════════════════════════════════════════

class TimedStepRequest(BaseModel):
    severity: str
    incident_type: str
    team: str
    time_taken_ms: int  # milliseconds since alert was shown
    status_update: Optional[str] = None
    is_false_positive: Optional[bool] = None
    session_id: Optional[str] = None

_SPEED_THRESHOLDS = {
    "P1": {"fast": 8000, "ok": 20000},   # <8s = fast, <20s = ok, else slow
    "P2": {"fast": 15000, "ok": 35000},
    "P3": {"fast": 25000, "ok": 60000},
    "P4": {"fast": 30000, "ok": 90000},
}

@app.post("/step/timed")
def step_timed(req: TimedStepRequest):
    """
    Like /step but with time-to-triage scoring.
    P1s triaged in under 8 seconds get a speed bonus.
    Slow P1 triage gets a penalty — in real incidents, every second counts.
    """
    sid = req.session_id or _default_session
    sess = _get_session(sid)

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
    expected_sev = alert.get("expected_severity", "P3")
    thresholds = _SPEED_THRESHOLDS.get(expected_sev, _SPEED_THRESHOLDS["P3"])

    # Speed bonus/penalty
    t = req.time_taken_ms
    if t <= thresholds["fast"]:
        speed_label = "⚡ Lightning fast"
        speed_bonus = 0.1 if expected_sev == "P1" else 0.05
    elif t <= thresholds["ok"]:
        speed_label = "✅ Good response time"
        speed_bonus = 0.0
    else:
        speed_label = "🐢 Too slow for this severity"
        speed_bonus = -0.1 if expected_sev == "P1" else -0.05

    # Apply bonus to reward
    adjusted_reward = round(min(1.0, max(0.0, reward.value + speed_bonus)), 4)

    return {
        "observation": next_obs.model_dump() if next_obs else None,
        "reward": {"value": adjusted_reward, "base_reward": reward.value, "breakdown": reward.breakdown},
        "done": done,
        "info": info,
        "speed": {
            "time_ms": t,
            "label": speed_label,
            "bonus": speed_bonus,
            "threshold_fast_ms": thresholds["fast"],
            "threshold_ok_ms": thresholds["ok"],
        },
        "streak": sess["streak"],
        "session_id": sid,
    }

@app.get("/speed/leaderboard")
def speed_leaderboard():
    """Fastest average triage times from the global leaderboard."""
    lb = _load_lb()
    timed = [e for e in lb if e.get("time_total")]
    timed.sort(key=lambda x: x["time_total"])
    return {
        "fastest": timed[:10],
        "note": "Ranked by total episode time. Includes all tasks.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 5: SEVERITY DRIFT DETECTOR — are you systematically biased?
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/drift")
def severity_drift(session_id: Optional[str] = None):
    """
    Detects systematic severity bias — are you always over-escalating to P1
    or under-escalating to P3? Shows your drift score and corrective advice.
    """
    sid = session_id or _default_session
    tl = _get_session(sid)["timeline"] if session_id else _timeline
    if not tl:
        return {"message": "No episode data."}

    SEV_NUM = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    diffs = []
    over_escalations = 0
    under_escalations = 0

    for entry in tl:
        exp = entry["expected"].get("severity")
        act = entry["action"].get("severity")
        if exp and act and exp in SEV_NUM and act in SEV_NUM:
            diff = SEV_NUM[act] - SEV_NUM[exp]  # negative = over-escalated, positive = under
            diffs.append(diff)
            if diff < 0: over_escalations += 1
            elif diff > 0: under_escalations += 1

    if not diffs:
        return {"message": "No severity data yet."}

    mean_drift = round(sum(diffs) / len(diffs), 3)
    bias = "over-escalating" if mean_drift < -0.3 else "under-escalating" if mean_drift > 0.3 else "well-calibrated"

    advice = {
        "over-escalating": "You're treating too many alerts as more critical than they are. Look for 'planned', 'scheduled', 'no impact' signals before jumping to P1/P2.",
        "under-escalating": "You're not taking alerts seriously enough. Revenue impact, complete failures, and security breaches need P1/P2 — don't downplay them.",
        "well-calibrated": "Your severity judgement is well-calibrated. Keep it up.",
    }

    return {
        "mean_drift": mean_drift,
        "bias": bias,
        "over_escalations": over_escalations,
        "under_escalations": under_escalations,
        "correct_severity": len(diffs) - over_escalations - under_escalations,
        "total": len(diffs),
        "advice": advice[bias],
        "drift_scale": "negative = over-escalated (chose higher severity), positive = under-escalated",
    }


def main():
    """Entry point for multi-mode deployment."""
    import uvicorn, os
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

if __name__ == "__main__":
    main()
