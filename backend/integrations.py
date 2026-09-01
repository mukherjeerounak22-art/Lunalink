"""SIH26166 - optional integrations: Supabase, Upstash Redis, Sentry, Gemini.

Every integration is a graceful no-op without its env keys - the demo runs
fully on localhost with zero keys (Tomorrow Plan, TIER 0). Keys come from
backend/.env (never committed) and are backend-only for anything privileged.
"""
import json
import os
import random
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent


def _load_dotenv():
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
SENTRY_DSN_BACKEND = os.environ.get("SENTRY_DSN_BACKEND", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
# Use the "-latest" alias: pinned names (gemini-2.0-flash, gemini-2.5-flash-lite)
# get RETIRED and start returning 404, while the alias always resolves to an
# available model. flash-lite = the highest free-tier quota class.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
NARRATE_RATE_LIMIT = int(os.environ.get("NARRATE_RATE_LIMIT", "20"))

# --------------------------------------------------------------------------
# Sentry (backend project) - breadcrumbs after each pipeline stage so a
# silent band-order / coordinate-transform bug surfaces as an event
# --------------------------------------------------------------------------
_sentry_enabled = False


def init_sentry():
    global _sentry_enabled
    if not SENTRY_DSN_BACKEND:
        return False
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=SENTRY_DSN_BACKEND, traces_sample_rate=0.1)
        _sentry_enabled = True
        return True
    except ImportError:
        return False


def breadcrumb(message, category="pipeline", data=None):
    if _sentry_enabled:
        import sentry_sdk
        sentry_sdk.add_breadcrumb(message=message, category=category,
                                  level="info", data=data or {})


def capture_exception(exc):
    if _sentry_enabled:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)


def capture_message(message, level="info"):
    if _sentry_enabled:
        import sentry_sdk
        sentry_sdk.capture_message(message, level=level)


# --------------------------------------------------------------------------
# Supabase (service role key - backend only, never frontend/git)
# --------------------------------------------------------------------------
def supabase_ok():
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def _sb_headers(extra=None):
    h = {"apikey": SUPABASE_SERVICE_KEY,
         "Authorization": "Bearer %s" % SUPABASE_SERVICE_KEY,
         "Content-Type": "application/json"}
    h.update(extra or {})
    return h


def supabase_insert(table, row, prefer=None):
    if not supabase_ok():
        return None
    import httpx
    extra = {"Prefer": prefer} if prefer else {}
    try:
        r = httpx.post("%s/rest/v1/%s" % (SUPABASE_URL, table),
                       headers=_sb_headers(extra), json=row, timeout=10)
        breadcrumb("supabase insert %s -> %s" % (table, r.status_code),
                   "supabase")
        return r.status_code
    except Exception as exc:                       # noqa: BLE001
        capture_exception(exc)
        return None


def supabase_upsert_scene(row):
    """Merge-duplicate on the scenes table (product_id is unique)."""
    return supabase_insert("scenes", row,
                           prefer="resolution=merge-duplicates")


def supabase_upload(bucket, path, data: bytes, content_type="application/octet-stream"):
    if not supabase_ok():
        return None
    import httpx
    try:
        r = httpx.post(
            "%s/storage/v1/object/%s/%s" % (SUPABASE_URL, bucket, path),
            headers={"Authorization": "Bearer %s" % SUPABASE_SERVICE_KEY,
                     "Content-Type": content_type},
            content=data, timeout=30)
        breadcrumb("supabase upload %s/%s -> %s" % (bucket, path, r.status_code),
                   "supabase")
        return r.status_code
    except Exception as exc:                       # noqa: BLE001
        capture_exception(exc)
        return None



# --------------------------------------------------------------------------
# Upstash Redis (REST) - job status keys + narration rate-limiting
# --------------------------------------------------------------------------
def redis_ok():
    return bool(UPSTASH_URL and UPSTASH_TOKEN)


def redis_set(key, value, ttl_seconds=3600):
    if not redis_ok():
        return None
    import httpx
    try:
        r = httpx.get("%s/set/%s/%s?EX=%d" % (UPSTASH_URL, key, value,
                                              ttl_seconds),
                      headers={"Authorization": "Bearer %s" % UPSTASH_TOKEN},
                      timeout=10)
        return r.json().get("result")
    except Exception as exc:                       # noqa: BLE001
        capture_exception(exc)
        return None


def redis_rate_limit(key, limit, window_seconds=60):
    """Sliding-window limiter per the Modern Web + AI Stack Guide (§5.1):
    a sorted set of timestamps, pruned per request. Returns (allowed, n)."""
    if not redis_ok():
        return True, 0                              # no limiter configured
    import random
    import time
    import httpx
    h = {"Authorization": "Bearer %s" % UPSTASH_TOKEN}
    now_ms = int(time.time() * 1000)
    window_ms = window_seconds * 1000
    member = "%d.%d" % (now_ms, random.randint(0, 10 ** 6))
    try:
        httpx.get("%s/zadd/%s/%d/%s" % (UPSTASH_URL, key, now_ms, member),
                  headers=h, timeout=10)
        httpx.get("%s/zremrangebyscore/%s/0/%d" % (UPSTASH_URL, key,
                                                   now_ms - window_ms),
                  headers=h, timeout=10)
        n = httpx.get("%s/zcard/%s" % (UPSTASH_URL, key), headers=h,
                      timeout=10).json().get("result", 0)
        httpx.get("%s/pexpire/%s/%d" % (UPSTASH_URL, key, window_ms),
                  headers=h, timeout=10)
        return n <= limit, n
    except Exception as exc:                       # noqa: BLE001
        capture_exception(exc)
        return True, 0


def redis_get_json(key):
    if not redis_ok():
        return None
    import httpx
    try:
        r = httpx.get("%s/get/%s" % (UPSTASH_URL, key),
                      headers={"Authorization": "Bearer %s" % UPSTASH_TOKEN},
                      timeout=15).json()
        v = r.get("result")
        return json.loads(v) if v else None
    except Exception as exc:                       # noqa: BLE001
        capture_exception(exc)
        return None


def redis_set_json(key, payload, ttl_seconds=300):
    """Cache-aside write (Guide §3.3). Upstash caps single values near 1 MB,
    so large match payloads are trimmed before caching - the authoritative
    full payload always lives in the backend's in-memory cache."""
    try:
        blob = json.dumps(payload)
        if len(blob) > 900_000:
            payload = dict(payload)
            payload["matches"] = payload.get("matches", [])[:500]
            payload["cached_trimmed"] = True
            blob = json.dumps(payload)
        return _redis_set_raw(key, blob, ttl_seconds)
    except Exception as exc:                       # noqa: BLE001
        capture_exception(exc)
        return None


def _redis_set_raw(key, value, ttl_seconds):
    import httpx
    try:
        r = httpx.post("%s/set/%s?EX=%d" % (UPSTASH_URL, key, ttl_seconds),
                       headers={"Authorization": "Bearer %s" % UPSTASH_TOKEN},
                       content=value, timeout=15)
        return r.status_code
    except Exception as exc:                       # noqa: BLE001
        capture_exception(exc)
        return None


# --------------------------------------------------------------------------
# Gemini - result narration ONLY, never metric generation (per the plan)
# --------------------------------------------------------------------------
def gemini_ok():
    return bool(GOOGLE_API_KEY)


_gemini_cooldown_until = 0.0     # 429 quota cooldown (seconds since epoch)


def gemini_narrate(prompt):
    """Gemini narration with 429-quota awareness: on a quota-exceeded
    response we log ONE warning-level event and stop calling the API for
    5 minutes (the local-template fallback takes over), so a dead quota
    cannot flood Sentry with an error per narrate click."""
    global _gemini_cooldown_until
    if not gemini_ok():
        return None
    if time.time() < _gemini_cooldown_until:
        return None                                  # stay on local fallback
    import httpx
    try:
        r = httpx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "%s:generateContent?key=%s" % (GEMINI_MODEL, GOOGLE_API_KEY),
            json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except httpx.HTTPStatusError as exc:              # noqa: BLE001
        if exc.response is not None and exc.response.status_code == 429:
            _gemini_cooldown_until = time.time() + 300
            capture_message(
                "Gemini quota exceeded (429) - narration falls back to the "
                "local template for 5 minutes; free-tier quota resets daily "
                "or upgrade the key's plan.", level="warning")
        else:
            capture_exception(exc)
        return None
    except Exception as exc:                          # noqa: BLE001
        capture_exception(exc)
        return None


def local_narration(m):
    """Fallback narration when no Gemini key - same honesty framing."""
    return (
        "This scene matched at %.1f%%: %d of %d SIFT candidate "
        "correspondences survived RANSAC homography verification, with a "
        "reprojection RMSE of %.2f pixels. The RANSAC iteration budget was "
        "derived from this scene's own inlier fraction (k >= log(1-p) / "
        "log(1-w^4), w=%.2f), not hard-coded. The source is %s. When RMSE "
        "stops improving, that is the image's own gradient-information "
        "ceiling (the Cramer-Rao bound), not a pipeline limitation."
    ) % (m.get("match_percentage", 0), m.get("inlier_count", 0),
         len(m.get("matches", [])), m.get("rmse_px") or 0,
         m.get("ransac", {}).get("inlier_fraction_w", 0),
         "real Chandrayaan-2 OHRC imagery" if m.get("scene_id") == "ohrc_20210401"
         else "the synthetic Tycho DEM stand-in")

