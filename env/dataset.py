"""
Static incident/alert dataset with ground-truth labels.

Each entry:
  id, title, source, body, timestamp
  expected_severity, expected_type, expected_team
  status_keywords        — for grading status update drafts
  urgency_deadline       — step index by which P1s must be triaged (else decay)
  is_false_positive      — adversarial: looks real but is noise
  cascading_from         — id of alert that caused this one
  noise_level            — 0.0 clean, 0.5 partial garble, 1.0 heavily corrupted
"""
import random
import re

# ---------------------------------------------------------------------------
# Core alerts
# ---------------------------------------------------------------------------
ALERTS = [
    {
        "id": "a001",
        "title": "CRITICAL: Primary database cluster unreachable",
        "source": "PagerDuty",
        "body": (
            "All read/write operations to prod-db-primary are failing. "
            "Connection pool exhausted. Error: 'too many connections' (max 500). "
            "Replica lag: 47s and climbing. Affecting all API endpoints. "
            "Started 08:03 UTC. 100% of health checks failing."
        ),
        "timestamp": "2026-04-06T08:03:00Z",
        "expected_severity": "P1",
        "expected_type": "database",
        "expected_team": "database",
        "status_keywords": ["database", "investigating", "impact", "team"],
        "urgency_deadline": 1,
    },
    {
        "id": "a002",
        "title": "INFO: Scheduled maintenance window starting",
        "source": "OpsGenie",
        "body": (
            "Planned maintenance for eu-west-1 load balancer begins in 15 minutes. "
            "Duration: 30 min. Traffic will be rerouted to us-east-1. "
            "No customer impact expected. Ticket: MAINT-2041."
        ),
        "timestamp": "2026-04-06T08:10:00Z",
        "expected_severity": "P4",
        "expected_type": "infrastructure",
        "expected_team": "infra",
        "status_keywords": ["maintenance", "planned", "no impact"],
    },
    {
        "id": "a003",
        "title": "WARNING: Unusual login spike from foreign IPs",
        "source": "Splunk SIEM",
        "body": (
            "Detected 4,200 failed login attempts in 10 minutes from IPs in "
            "RU/CN/KP. Credential stuffing pattern confirmed. "
            "3 accounts temporarily locked. Auth service CPU at 94%. "
            "Possible brute-force or botnet attack."
        ),
        "timestamp": "2026-04-06T08:15:00Z",
        "expected_severity": "P2",
        "expected_type": "security",
        "expected_team": "security",
        "status_keywords": ["security", "login", "investigating", "accounts"],
    },
    {
        "id": "a004",
        "title": "ERROR: Payment service returning 503s",
        "source": "Datadog",
        "body": (
            "checkout-service is returning HTTP 503 for 18% of requests. "
            "Stripe webhook handler throwing NullPointerException at "
            "PaymentProcessor.java:142. Last successful transaction: 08:21 UTC. "
            "Revenue impact: ~$1,200/min."
        ),
        "timestamp": "2026-04-06T08:22:00Z",
        "expected_severity": "P1",
        "expected_type": "application",
        "expected_team": "backend",
        "status_keywords": ["payment", "503", "investigating", "impact"],
        "urgency_deadline": 2,
    },
    {
        "id": "a005",
        "title": "NOTICE: SSL certificate expiring in 14 days",
        "source": "Certbot Monitor",
        "body": (
            "SSL certificate for api.company.com expires 2026-04-20. "
            "Auto-renewal failed: ACME challenge DNS record not found. "
            "Manual renewal required. No current impact."
        ),
        "timestamp": "2026-04-06T08:30:00Z",
        "expected_severity": "P3",
        "expected_type": "infrastructure",
        "expected_team": "infra",
        "status_keywords": ["certificate", "renewal", "deadline"],
    },
    {
        "id": "a006",
        "title": "CRITICAL: CDN edge nodes returning stale content",
        "source": "CloudWatch",
        "body": (
            "All 12 CDN edge nodes in us-east-1 serving cached responses from "
            "2 hours ago. Cache invalidation job failed silently at 06:00 UTC. "
            "Users seeing outdated product listings and prices. "
            "Cache-Control headers misconfigured after last deploy."
        ),
        "timestamp": "2026-04-06T08:35:00Z",
        "expected_severity": "P2",
        "expected_type": "application",
        "expected_team": "frontend",
        "status_keywords": ["cdn", "cache", "stale", "investigating"],
    },
    {
        "id": "a007",
        "title": "WARNING: Disk usage at 87% on prod-worker-03",
        "source": "Prometheus",
        "body": (
            "prod-worker-03 /var/log partition at 87% (43GB / 50GB). "
            "Log rotation misconfigured. Growth rate: ~2GB/day. "
            "Estimated time to full: 3 days. No current service impact."
        ),
        "timestamp": "2026-04-06T08:40:00Z",
        "expected_severity": "P3",
        "expected_type": "infrastructure",
        "expected_team": "infra",
        "status_keywords": ["disk", "log", "rotation", "worker"],
    },
    {
        "id": "a008",
        "title": "CRITICAL: API gateway latency spike — p99 > 8s",
        "source": "New Relic",
        "body": (
            "API gateway p99 latency jumped from 120ms to 8,400ms at 08:44 UTC. "
            "Downstream: user-service timeout chain. "
            "Root cause suspected: N+1 query in /api/v2/users/feed endpoint. "
            "Error rate: 34%. Active users impacted: ~12,000."
        ),
        "timestamp": "2026-04-06T08:44:00Z",
        "expected_severity": "P1",
        "expected_type": "application",
        "expected_team": "backend",
        "status_keywords": ["latency", "api", "investigating", "users", "impact"],
        "urgency_deadline": 1,
    },
    {
        "id": "a009",
        "title": "INFO: Nightly backup completed with warnings",
        "source": "Bacula",
        "body": (
            "Nightly backup job completed at 03:15 UTC. "
            "2 files skipped (locked by process). Backup size: 1.2TB. "
            "Verification checksum: PASS. Retention policy applied."
        ),
        "timestamp": "2026-04-06T08:50:00Z",
        "expected_severity": "P4",
        "expected_type": "infrastructure",
        "expected_team": "infra",
        "status_keywords": ["backup", "completed", "warnings"],
    },
    {
        "id": "a010",
        "title": "ERROR: Redis cache cluster split-brain detected",
        "source": "PagerDuty",
        "body": (
            "Redis sentinel reports split-brain: 2 nodes claiming master role. "
            "Session data inconsistency detected. ~8% of users being logged out "
            "unexpectedly. Started after network blip at 08:55 UTC. "
            "Failover script did not complete cleanly."
        ),
        "timestamp": "2026-04-06T08:58:00Z",
        "expected_severity": "P2",
        "expected_type": "database",
        "expected_team": "database",
        "status_keywords": ["redis", "session", "split-brain", "investigating"],
    },
]

# ---------------------------------------------------------------------------
# Adversarial alerts — false positives disguised as real incidents
# ---------------------------------------------------------------------------
ADVERSARIAL_ALERTS = [
    {
        "id": "adv001",
        "title": "CRITICAL: All services down — 100% error rate",
        "source": "Synthetic Monitor",
        "body": (
            "Synthetic monitoring probe reports 100% failure rate across all endpoints. "
            "However: probe agent was restarted 5 minutes ago and is still warming up. "
            "Real user traffic metrics show 0% error rate. "
            "This is a known false positive pattern after probe restarts."
        ),
        "timestamp": "2026-04-06T09:00:00Z",
        "expected_severity": "P4",
        "expected_type": "infrastructure",
        "expected_team": "infra",
        "status_keywords": [],
        "is_false_positive": True,
    },
    {
        "id": "adv002",
        "title": "WARNING: Possible data exfiltration — 50GB outbound",
        "source": "Network Monitor",
        "body": (
            "50GB outbound transfer detected to external IP 203.0.113.42. "
            "Investigation: this is the scheduled nightly analytics export to "
            "the data warehouse (job ID: DW-EXPORT-2041). "
            "Transfer matches expected size and destination. Authorized."
        ),
        "timestamp": "2026-04-06T09:05:00Z",
        "expected_severity": "P4",
        "expected_type": "network",
        "expected_team": "infra",
        "status_keywords": [],
        "is_false_positive": True,
    },
    {
        "id": "adv003",
        "title": "CRITICAL: Memory leak — process consuming 98% RAM",
        "source": "Datadog",
        "body": (
            "Process 'analytics-worker' consuming 98% of available RAM on analytics-01. "
            "This is expected behavior: analytics-01 is a dedicated high-memory instance "
            "(256GB RAM) running a large in-memory aggregation job scheduled weekly. "
            "Job will complete in ~20 minutes. No other services on this host."
        ),
        "timestamp": "2026-04-06T09:10:00Z",
        "expected_severity": "P4",
        "expected_type": "application",
        "expected_team": "backend",
        "status_keywords": [],
        "is_false_positive": True,
    },
    {
        "id": "adv004",
        "title": "ERROR: Database replication lag 45 seconds",
        "source": "CloudWatch",
        "body": (
            "Read replica prod-db-replica-2 showing 45s replication lag. "
            "This replica is used exclusively for analytics queries, not user traffic. "
            "Lag is within acceptable SLA for analytics workloads (< 5 minutes). "
            "Primary database is healthy. No user impact."
        ),
        "timestamp": "2026-04-06T09:15:00Z",
        "expected_severity": "P4",
        "expected_type": "database",
        "expected_team": "database",
        "status_keywords": [],
        "is_false_positive": False,  # legit low-priority, not a false positive
    },
]

# ---------------------------------------------------------------------------
# Cascading incident context — shown when a later alert is caused by an earlier one
# ---------------------------------------------------------------------------
CASCADING_CONTEXT = {
    "a008": "NOTE: This may be related to a001 (DB cluster unreachable at 08:03). Timeout chain suspected.",
    "a010": "NOTE: Redis split-brain may be downstream effect of network blip that also caused a003 login spike.",
}

# ---------------------------------------------------------------------------
# Persona overrides
# ---------------------------------------------------------------------------
PERSONA_OVERRIDES = {
    # startup_oncall: small team, everything feels more urgent, less process
    "startup_oncall": {
        "a002": {"expected_severity": "P3"},   # maintenance still matters when you're small
        "a005": {"expected_severity": "P2"},   # cert expiry is existential for a startup
        "a007": {"expected_severity": "P2"},   # disk full = outage for startup
        "a009": {"expected_severity": "P3"},   # backup warnings matter more
    },
    # enterprise_sre: mature runbooks, higher tolerance, strict P1 criteria
    "enterprise_sre": {
        "a003": {"expected_severity": "P3"},   # login spike handled by WAF automatically
        "a006": {"expected_severity": "P3"},   # CDN stale content, not revenue-critical
        "a007": {"expected_severity": "P4"},   # 3 days runway, low urgency
        "a010": {"expected_severity": "P3"},   # analytics replica, not user-facing
    },
    # solo_dev: everything is urgent, no team to delegate to
    "solo_dev": {
        "a002": {"expected_severity": "P2"},
        "a005": {"expected_severity": "P1"},
        "a007": {"expected_severity": "P2"},
        "a009": {"expected_severity": "P3"},
    },
}

# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------
_NOISE_CHARS = ["###", "???", "...", "ERR", "NUL"]

def _inject_noise(text: str, level: float, seed: int = 42) -> str:
    if level == 0.0:
        return text
    rng = random.Random(seed)
    words = text.split()
    n_corrupt = max(1, int(len(words) * level))
    indices = rng.sample(range(len(words)), min(n_corrupt, len(words)))
    for i in indices:
        if level >= 0.8:
            words[i] = "???"
        elif level >= 0.5:
            words[i] = words[i][:max(1, len(words[i]) // 2)] + "#"
        else:
            words[i] = re.sub(r"[aeiou]", "@", words[i])
    if level >= 0.9:
        words = words[:max(3, len(words) // 3)]
        words.append("[TRUNCATED]")
    return " ".join(words)


def get_noisy_alert(alert: dict, noise_level: float) -> dict:
    a = dict(alert)
    seed_base = sum(ord(c) for c in alert["id"])
    a["title"] = _inject_noise(alert["title"], noise_level, seed=seed_base * 3)
    a["body"]  = _inject_noise(alert["body"],  noise_level, seed=seed_base * 7)
    a["noise_level"] = noise_level
    return a


# ---------------------------------------------------------------------------
# Lookup maps
# ---------------------------------------------------------------------------
ALERT_MAP     = {a["id"]: a for a in ALERTS}
ADV_ALERT_MAP = {a["id"]: a for a in ADVERSARIAL_ALERTS}
ALL_ALERT_MAP = {**ALERT_MAP, **ADV_ALERT_MAP}

# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------
TASK_CONFIGS = {
    "easy": {
        "alert_ids": ["a001", "a002", "a003", "a004", "a005"],
        "grade_fields": ["severity"],
        "description": "Classify incident severity (P1-P4) for 5 alerts.",
    },
    "medium": {
        "alert_ids": ["a001", "a002", "a003", "a004", "a005", "a006", "a007", "a008"],
        "grade_fields": ["severity", "team"],
        "description": "Classify severity and route to the correct on-call team for 8 alerts.",
    },
    "hard": {
        "alert_ids": ["a001", "a002", "a003", "a004", "a005", "a006", "a007", "a008", "a009", "a010"],
        "grade_fields": ["severity", "team", "status_update"],
        "description": "Full triage: severity + team routing + draft a status update for 10 alerts.",
    },
    "adversarial": {
        "alert_ids": ["adv001", "adv002", "adv003", "adv004"],
        "grade_fields": ["severity", "false_positive"],
        "description": "Identify false positive alerts disguised as real incidents.",
    },
    "persona_startup": {
        "alert_ids": ["a001", "a002", "a004", "a005", "a007", "a008", "a009"],
        "grade_fields": ["severity", "team"],
        "persona": "startup_oncall",
        "description": "Triage as a startup on-call engineer — everything feels more urgent.",
    },
    "persona_enterprise": {
        "alert_ids": ["a001", "a003", "a004", "a006", "a007", "a008", "a010"],
        "grade_fields": ["severity", "team"],
        "persona": "enterprise_sre",
        "description": "Triage as an enterprise SRE — strict P1 criteria, mature runbooks.",
    },
    "noisy_easy": {
        "alert_ids": ["a001", "a002", "a003", "a004", "a005"],
        "grade_fields": ["severity", "team"],
        "noise_level": 0.3,
        "description": "Alerts have 30% word corruption. Can you still triage correctly?",
    },
    "noisy_hard": {
        "alert_ids": ["a001", "a002", "a003", "a004", "a005", "a006", "a007", "a008", "a009", "a010"],
        "grade_fields": ["severity", "team"],
        "noise_level": 0.7,
        "description": "Heavily garbled log entries. Tests robustness under signal degradation.",
    },
}
