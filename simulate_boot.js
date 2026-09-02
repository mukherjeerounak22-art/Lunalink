// Boot-wiring regression test: executes frontend scripts the way a browser
// does — config.js as a classic script, each type="module" block in its own
// function scope, all sharing one `window`. Catches cross-module
// ReferenceErrors and boot-order bugs that syntax checks cannot see.
// Usage: node simulate_boot.js [path/to/index.html] [path/to/config.js]
const fs = require("fs"), vm = require("vm"), path = require("path");
const htmlPath = process.argv[2] || path.join(__dirname, "frontend", "index.html");
const cfgPath = process.argv[3] || path.join(path.dirname(htmlPath), "config.js");
const html = fs.readFileSync(htmlPath, "utf8");
const cfg = fs.readFileSync(cfgPath, "utf8");

function magicEl() {
  const f = function () { return magicEl(); };
  return new Proxy(f, {
    get(t, p) {
      if (p === Symbol.toPrimitive) return () => "";
      if (p === "classList") return { add() {}, remove() {}, toggle() {}, contains() { return false; } };
      if (p in t) return t[p];
      return magicEl();
    },
    set(t, p, v) { t[p] = v; return true; },
    apply() { return magicEl(); }
  });
}
const documentStub = {
  getElementById: () => magicEl(),
  createElement: () => magicEl(),
  querySelectorAll: () => [],
  querySelector: () => magicEl(),
  head: magicEl(),
  addEventListener() {}
};
const sharedWindow = {
  API_BASE: "https://tunnel.example",
  SENTRY_DSN_FRONTEND: "",
  dispatchEvent() {},
  addEventListener() {},
  location: { search: "", origin: "https://x.example" }
};
const sandboxBase = () => ({
  window: sharedWindow,
  document: documentStub,
  console,
  URLSearchParams,
  location: { search: "", origin: "https://x.example" },
  setTimeout: () => 0,
  clearTimeout() {},
  fetch: () => Promise.resolve({ ok: true, status: 200, text: async () => "",
    json: async () => ({}) }),
  navigator: { userAgent: "sim" }
});
function runBody(body, label) {
  const sb = sandboxBase();
  sb.window = sharedWindow;
  vm.createContext(sb);
  const wrapped = "(function(window, document, console, URLSearchParams, setTimeout, clearTimeout){\n" +
    body + "\n})";
  const fn = new vm.Script(wrapped, { filename: label }).runInContext(sb);
  fn(sharedWindow, documentStub, console, URLSearchParams, () => 0, () => {});
}

{ // config.js first (classic script — defines window.fetchT etc.)
  const sb = sandboxBase();
  sb.window = sharedWindow;
  vm.createContext(sb);
  new vm.Script(cfg, { filename: "config.js" }).runInContext(sb);
}

const re = /<script type="module">([\s\S]*?)<\/script>/g;
let m; const modules = [];
while ((m = re.exec(html))) modules.push(m[1]);

const out2 = [];
modules.forEach((body, idx) => {
  if (/^\s*import\s/m.test(body)) {   // terrain module: real ES import — skip in vm
    out2.push("module#" + (idx + 1) + ": SKIPPED (ES import: three.js)");
    return;
  }
  try {
    runBody(body, "module#" + (idx + 1));
    out2.push("module#" + (idx + 1) + ": executed OK");
  } catch (e) {
    out2.push("module#" + (idx + 1) + ": THREW -> " + e.message);
  }
});

const need = ["__checkHealth", "__loadScenes", "__currentScene", "__runMatch",
  "__resetMatchCache", "__showTab", "__setSun", "__buildTerrain",
  "__loadTerrain", "__boot"];
need.forEach(n => out2.push(
  n + ": " + (typeof sharedWindow[n] === "function" ? "FUNCTION ok" : "MISSING")));
const critical = ["__checkHealth", "__loadScenes", "__currentScene", "__runMatch", "__boot"];
const passed = critical.every(n => typeof sharedWindow[n] === "function");
out2.push("SIMULATION " + (passed ? "PASSED" : "FAILED"));
console.log(out2.join("\n"));
process.exitCode = passed ? 0 : 1;