// SIH26166 frontend configuration.
//
// ⚠️ VERCEL DEMO: this MUST hold your backend's public URL before you
// `vercel --prod`, or the page will show "backend offline". Deploy the
// backend with render.yaml (see DEMO_INSTRUCTIONS.md §14), then e.g.:
//   window.API_BASE = "https://sih26166-backend.onrender.com";
// Empty string = same origin (correct ONLY for the localhost demo, where
// uvicorn serves both the frontend and the API on :8000).
window.API_BASE = "";

// Sentry browser project (per Modern Web + AI Stack Guide §8: the browser DSN
// is safe to expose client-side). Leave empty to disable.
window.SENTRY_DSN_FRONTEND = "https://328764c4bcadc9cfe17ff28e231a438a@o4511983664103424.ingest.us.sentry.io/4512005888081920";

