// SIH26166 frontend configuration.
//
// ⚠️ HOW THE VERCEL PAGE FINDS THE BACKEND (two ways):
//
// 1. Bake it in (Render-style deploys): set API_BASE below to the backend's
//    public HTTPS URL and `vercel --prod`, e.g.
//        window.API_BASE = "https://sih26166-backend.onrender.com";
//
// 2. Pass it at runtime — NO redeploy needed (laptop + tunnel demo):
//        https://<your-vercel-url>/?api=https://<tunnel-or-backend-url>
//    The ?api= query parameter overrides everything below for that page
//    load only (see DEMO_INSTRUCTIONS.md §14 Option B). Handy because
//    free tunnel URLs change every run.
//
// Empty string = same origin (correct ONLY when uvicorn serves the frontend
// itself on http://127.0.0.1:8000).
window.API_BASE =
  (new URLSearchParams(location.search).get("api") || "")
    .replace(/\/+$/, "") || "";

// fetchT: fetch with a HARD timeout. Without this a stalled backend/tunnel
// (or a stale browser DNS cache) leaves the status badge stuck on
// "connecting…" forever — every request in the app goes through here so the
// UI always reaches an error/retry state instead of hanging.
window.fetchT = function (url, opts, ms) {
  opts = opts || {};
  ms = ms || 20000;
  const ctrl = new AbortController();
  const timer = setTimeout(function () { ctrl.abort(); }, ms);
  return fetch(url, Object.assign({}, opts, { signal: ctrl.signal }))
    .finally(function () { clearTimeout(timer); })
    .catch(function (err) {
      if (err && (err.name === "AbortError" || /abort/i.test(err.message || ""))) {
        throw new Error("timed out after " + Math.round(ms / 1000) +
          " s - backend/tunnel not reachable (click the status badge to retry)");
      }
      throw err;
    });
};

// Sentry browser project (per Modern Web + AI Stack Guide §8: the browser DSN
// is safe to expose client-side). Leave empty to disable.
window.SENTRY_DSN_FRONTEND = "https://328764c4bcadc9cfe17ff28e231a438a@o4511983664103424.ingest.us.sentry.io/4512005888081920";

