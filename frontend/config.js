// SIH26166 frontend configuration.
// Empty = same origin (localhost demo: uvicorn serves frontend + API).
// After deploying: set this to your backend's public URL, e.g.
//   window.API_BASE = "https://sih26166-backend.onrender.com";
window.API_BASE = "";

// Sentry browser project (per Modern Web + AI Stack Guide §8: the browser DSN
// is safe to expose client-side). Leave empty to disable.
window.SENTRY_DSN_FRONTEND = "https://328764c4bcadc9cfe17ff28e231a438a@o4511983664103424.ingest.us.sentry.io/4512005888081920";

