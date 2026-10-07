/* Recovery Console — vanilla JS, no build step, no third-party code.
 *
 * Security rule for this file: data from the API (file names, labels, agent
 * answers) is attacker-influenced — ransomware controls the disk we index.
 * It is only ever inserted with textContent / text nodes via el(); there is
 * no innerHTML anywhere. The page is also served with a strict CSP.
 */
"use strict";

const POLL_MS = 5000;
const MAX_ROWS = 200;

// ---------------------------------------------------------------- helpers
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Build an element. attrs: class, text, title, on<Event>, data-*, any attribute. */
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    // CSSOM, not setAttribute("style"): the CSP blocks inline style attributes.
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k === "checked" || k === "disabled" || k === "selected") node[k] = Boolean(v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}
const clear = (node) => { node.replaceChildren(); return node; };
/** Replace a node's children; accepts nested arrays (native append() would stringify them). */
function fill(node, ...kids) {
  node.replaceChildren();
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false || k === "") continue;
    node.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return node;
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const raw = await res.text();
  let data = null;
  try { data = raw ? JSON.parse(raw) : null; } catch { data = { detail: raw.slice(0, 300) }; }
  if (!res.ok) {
    const detail = data && data.detail
      ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))
      : res.statusText;
    const err = new Error(`${res.status}: ${detail}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

function toast(msg, kind = "") {
  const t = el("div", { class: `toast ${kind}`, role: "status", text: msg });
  $("#toasts").append(t);
  setTimeout(() => t.remove(), kind === "bad" ? 9000 : 5000);
}

/** Run an async action with a busy button and error toast. */
async function withBusy(btn, fn) {
  if (btn) { btn.disabled = true; btn.classList.add("busy"); }
  try { return await fn(); }
  catch (e) { toast(e.message, "bad"); return undefined; }
  finally { if (btn) { btn.disabled = false; btn.classList.remove("busy"); } }
}

const fmtTime = (iso) => iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—";
const fmtShort = (iso) => iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—";
function ago(iso) {
  if (!iso) return "never";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
function bytes(n) {
  if (n === null || n === undefined) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0; let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i ? 1 : 0)} ${u[i]}`;
}
const num = (x, d = 2) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));
const pct = (x) => (x === null || x === undefined ? "—" : `${(x * 100).toFixed(1)}%`);
const badge = (s, label) => el("span", { class: `badge ${s}`, text: label || s });

function entropyCell(e) {
  const bar = el("span", { class: `bar${e > 7.5 ? " alert" : ""}` }, el("span", { style: `width:${Math.min(100, (e / 8) * 100)}%` }));
  return el("span", { class: "entropy", title: "Shannon entropy, bits/byte (text ≈ 4–5, encrypted ≈ 8)" }, bar, num(e));
}
function scoreBar(score, threshold) {
  const alert = score >= threshold;
  return el("div", { class: `bar${alert ? " alert" : ""}`, title: `score ${num(score, 3)} / threshold ${num(threshold, 3)}` },
    el("span", { style: `width:${Math.min(100, score * 100)}%` }),
    el("span", { class: "thr", style: `left:${Math.min(100, threshold * 100)}%` }));
}
function table(headers, rows) {
  return el("table", {},
    el("thead", {}, el("tr", {}, headers.map((h) => el("th", { text: h })))),
    el("tbody", {}, rows));
}

/** Safe Markdown subset (headings, bullets, **bold**, `code`) rendered as DOM nodes. */
function renderMarkdown(text) {
  const root = el("div", { class: "md" });
  const inline = (s) => s.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part) => {
    if (part.startsWith("**") && part.endsWith("**")) return el("strong", { text: part.slice(2, -2) });
    if (part.startsWith("`") && part.endsWith("`")) return el("code", { text: part.slice(1, -1) });
    return document.createTextNode(part);
  });
  let list = null; let para = [];
  const flushPara = () => { if (para.length) { root.append(el("p", {}, inline(para.join(" ")))); para = []; } };
  for (const line of String(text || "").split(/\r?\n/)) {
    const t = line.trim();
    const h = t.match(/^#{1,4}\s+(.*)$/);
    const li = t.match(/^[-*]\s+(.*)$/) || t.match(/^\d+[.)]\s+(.*)$/);
    if (h) { flushPara(); list = null; root.append(el("h3", {}, inline(h[1]))); }
    else if (li) { flushPara(); if (!list) { list = el("ul"); root.append(list); } list.append(el("li", {}, inline(li[1]))); }
    else if (!t) { flushPara(); list = null; }
    else { list = null; para.push(t); }
  }
  flushPara();
  return root;
}

function confirmDialog(title, body, okLabel = "Confirm") {
  const dlg = $("#confirm-dialog");
  $("#confirm-title").textContent = title;
  clear($("#confirm-body")).append(...(Array.isArray(body) ? body : [body]));
  $("#confirm-ok").textContent = okLabel;
  dlg.returnValue = "cancel";
  dlg.showModal();
  return new Promise((resolve) => dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok"), { once: true }));
}

// ------------------------------------------------------------------ state
const state = {
  info: null, health: null, status: null, snapshots: [], incidents: [], candidates: null, jobs: [],
  selected: new Set(), restoreSel: null, restoreSelManual: false, lastReco: undefined,
  scoreHistory: [], labDone: new Set(),
};

function showTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${name}`));
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
}

// --------------------------------------------------------------- refresh
async function refresh() {
  const [health, status, snaps, incidents, candidates, jobs] = await Promise.allSettled([
    fetch("/health").then((r) => r.json()),
    api("GET", "/detection/status"),
    api("GET", "/backups?limit=200"),
    api("GET", "/detection/incidents?limit=100"),
    api("GET", "/restore/candidates"),
    api("GET", "/restore/jobs?limit=50"),
  ]);
  const val = (r, fallback) => (r.status === "fulfilled" ? r.value : fallback);
  state.health = val(health, null);
  state.status = val(status, null);
  state.snapshots = val(snaps, { items: [] }).items;
  state.incidents = val(incidents, []);
  state.candidates = val(candidates, null);
  state.jobs = val(jobs, []);
  if (state.status && state.status.monitor) {
    state.scoreHistory.push({ t: Date.now(), score: state.status.monitor.last_score || 0 });
    state.scoreHistory = state.scoreHistory.slice(-60);
  }
  renderPills(); renderOverview(); renderSnapshots(); renderDetection(); renderRestore(); renderLabLive();
}

function renderPills() {
  const h = state.health; const s = state.status;
  const pill = (kind, text, title) => el("span", { class: `pill ${kind}`, title }, el("span", { class: "dot" }), text);
  const pills = [
    pill(h && h.database ? "ok" : "bad", "database"),
    pill(h && h.storage ? "ok" : "bad", "storage", state.info ? `bucket ${state.info.bucket}` : ""),
    pill(s && s.model_loaded ? "ok" : "warn", s && s.model_loaded ? `model: ${s.model.model_name}` : "no model"),
    pill(s && s.monitor.alive ? "ok" : "warn", s && s.monitor.alive ? "watcher live" : "watcher offline",
      s && s.monitor.last_seen ? `last heartbeat ${ago(s.monitor.last_seen)}` : "no heartbeat"),
    pill(s && s.open_incidents ? "bad" : "ok", s && s.open_incidents ? `${s.open_incidents} open incident(s)` : "no open incidents"),
  ];
  clear($("#status-pills")).append(...pills);
  const detTab = $('.tabs button[data-tab="detection"]');
  clear(detTab).append("Detection", s && s.open_incidents ? el("span", { class: "count", text: s.open_incidents }) : "");
}

// --------------------------------------------------------------- overview
function renderOverview() {
  const snaps = state.snapshots; const incs = state.incidents; const jobs = state.jobs;
  const open = incs.filter((i) => i.status === "open");
  const lastInc = incs[0];
  const lastRestore = jobs.find((j) => !j.dry_run);
  const restoredAfter = lastInc && lastRestore && lastRestore.status === "succeeded"
    && new Date(lastRestore.created_at) > new Date(lastInc.started_at);
  const latest = snaps[0];
  const lastClean = state.candidates && state.candidates.last_clean_snapshot_id;
  const stages = [
    { t: "Protected", d: snaps.some((s) => s.state === "clean") ? `${snaps.filter((s) => s.state === "clean").length} clean snapshot(s)` : "take a first snapshot",
      c: snaps.some((s) => s.state === "clean") ? "done" : "next" },
    { t: "Attack detected", d: open.length ? `incident #${open[open.length - 1].id} open` : lastInc ? `last: incident #${lastInc.id}` : "nothing detected",
      c: open.length ? "active" : lastInc ? "done" : "next" },
    { t: "Restore point", d: lastInc ? (lastClean ? lastClean : "no clean snapshot!") : "—",
      c: lastInc && lastClean ? "done" : "next" },
    { t: "Restored & verified", d: restoredAfter ? `job #${lastRestore.id}: ${lastRestore.files_verified}/${lastRestore.files_total} verified` : open.length ? "restore needed" : "—",
      c: restoredAfter ? "done" : "next" },
    { t: "All clear", d: !open.length && latest && latest.state === "clean" ? "latest snapshot clean" : open.length ? "resolve the incident after restoring" : "—",
      c: !open.length && latest && latest.state === "clean" ? "done" : "next" },
  ];
  clear($("#flow")).append(...stages.map((s) => el("li", { class: s.c }, el("div", { class: "t", text: s.t }), el("div", { class: "d", text: s.d }))));

  const m = state.status ? state.status.monitor : null;
  const stat = (value, label, extra) => el("div", { class: "card stat" }, el("div", { class: "value" }, value), el("div", { class: "label", text: label }), extra || "");
  clear($("#stat-cards")).append(
    stat(String(snaps.length), "snapshots", latest ? el("div", {}, "latest ", badge(latest.state), " ", el("span", { class: "muted small", text: ago(latest.created_at) })) : null),
    stat(String(open.length), "open incidents", lastInc ? el("div", { class: "muted small", text: `${incs.length} total · last ${ago(lastInc.started_at)}` }) : null),
    stat(m && m.alive ? "live" : "offline", "watcher", m ? el("div", { class: "muted small", text: `${m.windows_scored} windows scored · ${m.alerts} alerts` }) : null),
    stat(lastRestore ? lastRestore.status : "—", "last restore", lastRestore ? el("div", { class: "muted small", text: `${lastRestore.files_verified}/${lastRestore.files_total} SHA-256 verified · ${ago(lastRestore.created_at)}` }) : null),
  );

  const events = [
    ...snaps.map((s) => ({ t: s.created_at, node: el("span", {}, "Snapshot ", el("code", { text: s.id }), " ", badge(s.state), s.label ? ` · ${s.label}` : "", ` · ${s.file_count} files`) })),
    ...incs.map((i) => ({ t: i.started_at, node: el("span", {}, badge(i.status === "open" ? "alert" : "resolved", "incident"), ` #${i.id} score ${num(i.score, 3)} · ${i.affected_file_count} files · via ${i.source}`) })),
    ...jobs.map((j) => ({ t: j.created_at, node: el("span", {}, badge(j.status), ` ${j.dry_run ? "dry-run" : "restore"} #${j.id} of `, el("code", { text: j.snapshot_id }), j.dry_run ? "" : ` · ${j.files_verified}/${j.files_total} verified`) })),
  ].sort((a, b) => new Date(b.t) - new Date(a.t)).slice(0, 30);
  const tl = clear($("#timeline"));
  if (!events.length) tl.append(el("li", { class: "empty" }, el("span"), "No activity yet — try the Lab tab."));
  for (const e of events) tl.append(el("li", {}, el("span", { class: "when", text: fmtTime(e.t) }), e.node));
}

// -------------------------------------------------------------- snapshots
function renderSnapshots() {
  const rows = state.snapshots.map((s) => {
    const cb = el("input", { type: "checkbox", checked: state.selected.has(s.id), "aria-label": `select ${s.id}`,
      onclick: (ev) => { ev.stopPropagation(); ev.target.checked ? state.selected.add(s.id) : state.selected.delete(s.id); updateCompare(); } });
    return el("tr", { class: "clickable", onclick: () => showSnapshot(s.id) },
      el("td", {}, cb), el("td", { class: "mono", text: s.id }), el("td", { text: fmtTime(s.created_at) }),
      el("td", {}, badge(s.state)), el("td", { text: s.label || "" }), el("td", { class: "num", text: s.file_count }),
      el("td", { class: "num", text: bytes(s.total_bytes) }),
      el("td", { class: "num", title: "new content uploaded (dedup skips unchanged files)", text: `${s.uploaded_files} · ${bytes(s.uploaded_bytes)}` }),
      el("td", {}, entropyCell(s.mean_entropy)));
  });
  const t = $("#snap-table");
  t.replaceWith(Object.assign(table(["", "Snapshot", "Taken", "State", "Label", "Files", "Size", "Uploaded", "Mean entropy"],
    rows.length ? rows : [el("tr", {}, el("td", { colspan: 9, class: "empty", text: "No snapshots yet." }))]), { id: "snap-table" }));
  for (const id of [...state.selected]) if (!state.snapshots.some((s) => s.id === id)) state.selected.delete(id);
  updateCompare();
}
function updateCompare() {
  const b = $("#snap-compare");
  b.disabled = state.selected.size !== 2;
  b.textContent = state.selected.size === 2 ? "Compare selected (2)" : `Compare selected (${state.selected.size}/2)`;
}

async function showSnapshot(id) {
  const card = $("#snap-detail");
  const snap = await withBusy(null, () => api("GET", `/backups/${encodeURIComponent(id)}`));
  if (!snap) return;
  const filter = el("input", { class: "input", placeholder: "filter files…" });
  const body = el("div", { class: "table-wrap" });
  const draw = () => {
    const q = filter.value.toLowerCase();
    const files = snap.files.filter((f) => f.path.toLowerCase().includes(q)).sort((a, b) => b.entropy - a.entropy);
    clear(body).append(table(["Path", "Size", "Entropy", "SHA-256", "Modified"],
      files.slice(0, MAX_ROWS).map((f) => el("tr", {}, el("td", { class: "path", text: f.path }), el("td", { class: "num", text: bytes(f.size) }),
        el("td", {}, entropyCell(f.entropy)), el("td", { class: "mono small", title: f.sha256, text: `${f.sha256.slice(0, 12)}…` }), el("td", { text: fmtTime(f.mtime) })))));
    if (files.length > MAX_ROWS) body.append(el("p", { class: "muted small", text: `…and ${files.length - MAX_ROWS} more` }));
  };
  filter.addEventListener("input", draw);
  fill(card,
    el("div", { class: "row between" }, el("div", {}, el("h2", {}, "Snapshot ", el("code", { text: snap.id }), " ", badge(snap.state)),
      el("p", { class: "muted", text: `${fmtTime(snap.created_at)} · ${snap.file_count} files · ${bytes(snap.total_bytes)} · manifest ${snap.manifest_key}` })),
      el("button", { class: "btn ghost small", text: "Close", onclick: () => card.classList.add("hidden") })),
    el("p", { class: "muted small", text: "Sorted by entropy: encrypted files cluster near 8 bits/byte." }), filter, body);
  draw();
  card.classList.remove("hidden");
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function compareSelected() {
  const [a, b] = [...state.selected].map((id) => state.snapshots.find((s) => s.id === id)).sort((x, y) => new Date(x.created_at) - new Date(y.created_at));
  const diff = await withBusy($("#snap-compare"), () => api("GET", `/backups/${encodeURIComponent(a.id)}/diff/${encodeURIComponent(b.id)}`));
  if (!diff) return;
  const sm = diff.summary;
  const chip = (label, n, cls = "") => el("span", { class: `chip static ${cls}` }, label, el("b", { text: n }));
  const section = (title, hint, headers, items, row) => items.length ? [
    el("h3", { text: `${title} (${items.length})` }), hint ? el("p", { class: "muted small", text: hint }) : null,
    el("div", { class: "table-wrap" }, table(headers, items.slice(0, MAX_ROWS).map(row))),
    items.length > MAX_ROWS ? el("p", { class: "muted small", text: `…and ${items.length - MAX_ROWS} more` }) : null] : [];
  const card = $("#diff-card");
  fill(card,
    el("div", { class: "row between" }, el("h2", {}, "Diff ", el("code", { text: a.id }), " → ", el("code", { text: b.id })),
      el("button", { class: "btn ghost small", text: "Close", onclick: () => card.classList.add("hidden") })),
    el("div", { class: "chips summary-chips" }, chip("added", sm.added), chip("removed", sm.removed), chip("modified", sm.modified),
      chip("renamed", sm.renamed), chip("extension changed", sm.extension_changed), chip("unchanged", sm.unchanged),
      chip("mean entropy Δ", (sm.mean_entropy_delta >= 0 ? "+" : "") + num(sm.mean_entropy_delta))),
    sm.extension_changed + sm.added + sm.removed + sm.modified + sm.renamed === 0 ? el("div", { class: "notice ok", text: "Identical: no file changed between these snapshots." }) : "",
    section("Extension changed", "The classic ransomware footprint: report.docx → report.docx.locked, usually with an entropy jump.",
      ["Before", "After", "Ext", "Entropy before → after", "Content"], diff.extension_changed,
      (c) => el("tr", {}, el("td", { class: "path", text: c.old_path }), el("td", { class: "path", text: c.new_path }),
        el("td", { text: `${c.old_extension || "∅"} → ${c.new_extension || "∅"}` }),
        el("td", {}, entropyCell(c.entropy_before), " → ", entropyCell(c.entropy_after)), el("td", { text: c.content_changed ? "rewritten" : "same" }))),
    section("Modified", "Sorted by entropy increase.", ["Path", "Size", "Entropy before → after", "Δ"],
      [...diff.modified].sort((x, y) => y.entropy_delta - x.entropy_delta),
      (m) => el("tr", {}, el("td", { class: "path", text: m.path }), el("td", { class: "num", text: `${bytes(m.size_before)} → ${bytes(m.size_after)}` }),
        el("td", {}, entropyCell(m.entropy_before), " → ", entropyCell(m.entropy_after)), el("td", { class: "num", text: (m.entropy_delta >= 0 ? "+" : "") + num(m.entropy_delta) }))),
    section("Added", null, ["Path", "Size", "Entropy"], diff.added,
      (f) => el("tr", {}, el("td", { class: "path", text: f.path }), el("td", { class: "num", text: bytes(f.size) }), el("td", {}, entropyCell(f.entropy)))),
    section("Removed", null, ["Path", "Size", "Entropy"], diff.removed,
      (f) => el("tr", {}, el("td", { class: "path", text: f.path }), el("td", { class: "num", text: bytes(f.size) }), el("td", {}, entropyCell(f.entropy)))),
    section("Renamed (same content)", null, ["From", "To"], diff.renamed,
      (r) => el("tr", {}, el("td", { class: "path", text: r.old_path }), el("td", { class: "path", text: r.new_path }))),
  );
  card.classList.remove("hidden");
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

// -------------------------------------------------------------- detection
function featureRows(feats) {
  if (!feats || !feats.length) return el("p", { class: "muted small", text: "no single feature stood out" });
  const maxZ = Math.max(...feats.map((f) => f.z_score), 1);
  return el("div", { class: "feat" }, feats.map((f) => [
    el("span", { title: `value ${f.value} vs normal ≈ ${f.benign_mean}` }, el("code", { text: f.feature }), el("span", { class: "muted small", text: ` ${num(f.value, 3)} (normal ≈ ${num(f.benign_mean, 3)}, z=${num(f.z_score, 1)})` })),
    el("div", { class: "bar alert" }, el("span", { style: `width:${Math.min(100, (f.z_score / maxZ) * 100)}%` }))]));
}

function renderDetection() {
  const s = state.status;
  const wc = clear($("#watcher-card"));
  wc.append(el("h2", { text: "Watcher (live monitor)" }));
  if (s) {
    const m = s.monitor; const thr = s.model ? s.model.threshold : 0.5;
    wc.append(el("div", { class: "row" }, badge(m.alive ? "ok" : "warn", m.alive ? "alive" : "offline"),
      el("span", { class: "muted small", text: m.last_seen ? `last heartbeat ${ago(m.last_seen)}` : "never reported — is the watcher running?" })),
      el("dl", { class: "kv" }, el("dt", { text: "watching" }), el("dd", { class: "mono small", text: m.watch_dir || (state.info && state.info.watch_dir) || "—" }),
        el("dt", { text: "windows scored" }), el("dd", { text: m.windows_scored }), el("dt", { text: "alerts" }), el("dd", { text: m.alerts }),
        el("dt", { text: "window" }), el("dd", { text: state.info ? `${state.info.detection_window_seconds}s` : "—" })),
      el("div", { class: "small muted", text: `last window score ${num(m.last_score, 3)} (threshold ${num(thr, 3)})` }), scoreBar(m.last_score || 0, thr));
  }
  const mc = clear($("#model-card"));
  mc.append(el("h2", { text: "Detection model" }));
  if (s && s.model) {
    const tm = s.model.test_metrics || {};
    mc.append(el("dl", { class: "kv" }, el("dt", { text: "model" }), el("dd", { text: s.model.model_name }),
      el("dt", { text: "threshold" }), el("dd", { text: num(s.model.threshold, 4) }), el("dt", { text: "trained" }), el("dd", { text: fmtTime(s.model.trained_at) })),
      el("h3", { text: "Held-out test metrics (synthetic data)" }),
      table(["Precision", "Recall", "F1", "ROC-AUC", "False-positive rate"], [el("tr", {},
        el("td", { text: num(tm.precision, 3) }), el("td", { text: num(tm.recall, 3) }), el("td", { text: num(tm.f1, 3) }),
        el("td", { text: num(tm.roc_auc, 3) }), el("td", { text: pct(tm.false_positive_rate) }))]));
  } else mc.append(el("div", { class: "notice", text: "No model loaded. Run `python -m ml.train` (the Docker image trains one at build time)." }));

  const list = clear($("#incident-list"));
  if (!state.incidents.length) list.append(el("p", { class: "empty", text: "No incidents. Simulate an attack in the Lab tab to see one." }));
  for (const i of state.incidents) {
    const resolveBtn = i.status === "open" ? el("button", { class: "btn ghost small", text: "Resolve", onclick: (ev) => resolveIncident(i, ev.target) }) : null;
    list.append(el("div", { class: `incident ${i.status}` },
      el("header", {}, el("strong", { text: `Incident #${i.id}` }), badge(i.status), badge("info", i.source),
        el("span", { class: "muted small", text: `${fmtTime(i.started_at)} → ${fmtShort(i.ended_at)} · ${i.windows} window(s) · ${i.affected_file_count} files` }),
        el("span", { style: "margin-left:auto" }, resolveBtn)),
      el("div", { class: "small", text: `score ${num(i.score, 3)} vs threshold ${num(i.threshold, 3)} (${i.model_name})` }), scoreBar(i.score, i.threshold),
      el("h3", { text: "Why it fired" }), featureRows(i.top_features),
      el("details", {}, el("summary", { class: "small", text: `Affected files (showing ${Math.min(i.affected_files.length, 40)} of ${i.affected_file_count})` }),
        el("ul", { class: "small mono" }, i.affected_files.slice(0, 40).map((f) => el("li", { text: f }))))));
  }
}

async function resolveIncident(i, btn) {
  const ok = await confirmDialog(`Resolve incident #${i.id}?`, [
    el("p", { text: "Do this after you have restored. Snapshots taken while an incident is open are marked suspect; once resolved, new snapshots are trusted as clean again." })], "Resolve");
  if (!ok) return;
  if (await withBusy(btn, () => api("POST", `/detection/incidents/${i.id}/resolve`))) { toast(`Incident #${i.id} resolved`, "ok"); refresh(); }
}

async function scanNow(btn) {
  const r = await withBusy(btn, () => api("POST", "/detection/scan", {}));
  const out = clear($("#scan-result"));
  if (!r) return;
  const ds = r.diff_summary;
  out.append(el("div", { class: `notice ${r.verdict === "alert" ? "bad" : "ok"}` },
    el("strong", { text: r.verdict === "alert" ? "ALERT — ransomware-like activity" : "Clean — nothing suspicious" }),
    ` · max window score ${num(r.max_score, 3)} (threshold ${num(r.threshold, 3)}) · compared `, el("code", { text: r.base_snapshot_id }), " with ", r.target),
    el("div", { class: "chips" }, Object.entries(ds).map(([k, v]) => el("span", { class: "chip static" }, `${k.replaceAll("_", " ")} `, el("b", { text: typeof v === "number" && !Number.isInteger(v) ? num(v) : v })))),
    r.windows.length ? el("div", { class: "table-wrap" }, table(["Window", "Events", "Score", "Top features"], r.windows.map((w) => el("tr", {},
      el("td", { text: `${fmtShort(w.start)} – ${fmtShort(w.end)}` }), el("td", { class: "num", text: w.event_count }),
      el("td", { style: "min-width:140px" }, scoreBar(w.score, r.threshold), el("span", { class: "small", text: num(w.score, 3) })),
      el("td", { class: "small" }, w.top_features.map((f) => el("div", {}, el("code", { text: f.feature }), ` ${num(f.value, 2)}`))))))) : el("p", { class: "muted", text: "No changes since the snapshot." }),
    r.incident ? el("p", {}, "Recorded as ", el("strong", { text: `incident #${r.incident.id}` }), ". ", el("a", { href: "#restore", text: "Go to Restore →" })) : "");
  refresh();
}

// ---------------------------------------------------------------- restore
function selectedTarget() {
  return $('input[name="target"]:checked').value === "inplace" && state.info ? state.info.watch_dir : null;
}

function renderRestore() {
  const c = state.candidates;
  const reco = clear($("#restore-reco"));
  reco.append(el("h2", { text: "Recommended restore point" }));
  if (!c) { reco.append(el("p", { class: "muted", text: "Loading…" })); return; }
  if (c.last_clean_snapshot_id) {
    reco.append(el("div", { class: "reco" }, el("span", { class: "big", text: c.last_clean_snapshot_id }), badge("clean"),
      el("button", { class: "btn small", text: "Select", onclick: () => { state.restoreSel = c.last_clean_snapshot_id; renderRestore(); } })),
      el("p", { class: "muted small", text: c.reference_incident_id
        ? `Newest clean snapshot taken before incident #${c.reference_incident_id} started (${fmtTime(c.incident_started_at)}).`
        : "No open incident: this is simply the newest clean snapshot." }));
  } else reco.append(el("div", { class: "notice bad", text: "No clean snapshot exists before the incident — nothing safe to restore." }));

  // Follow the recommendation whenever it changes (e.g. a new incident moves it),
  // unless the user deliberately picked another snapshot since it last changed.
  // Keeping a stale selection could restore an older point and lose good work.
  if (c.last_clean_snapshot_id !== state.lastReco) {
    state.lastReco = c.last_clean_snapshot_id;
    state.restoreSelManual = false;
  }
  if (!state.restoreSelManual || !c.candidates.some((s) => s.id === state.restoreSel)) {
    state.restoreSel = c.last_clean_snapshot_id || (c.candidates[0] && c.candidates[0].id);
  }
  const list = clear($("#restore-candidates"));
  if (!c.candidates.length) list.append(el("p", { class: "empty", text: "No snapshots yet." }));
  for (const s of c.candidates.slice(0, 50)) {
    list.append(el("label", { class: "cand" }, el("input", { type: "radio", name: "restore-snap", value: s.id, checked: s.id === state.restoreSel, onchange: () => { state.restoreSel = s.id; state.restoreSelManual = true; } }),
      el("span", { class: "id", text: s.id }), badge(s.state), s.recommended ? badge("info", "recommended") : "",
      el("span", { class: "muted small", text: `${fmtTime(s.created_at)} · ${s.file_count} files${s.before_incident === false ? " · after incident start" : ""}` })));
  }
  if (state.info) {
    $("#target-separate").textContent = `${state.info.restore_dir}/<snapshot id>`;
    $("#target-inplace").textContent = state.info.watch_dir;
  }
  const jt = $("#jobs-table");
  jt.replaceWith(Object.assign(table(["Job", "When", "Snapshot", "Target", "Kind", "Status", "Restored", "Verified", "Failed", "Extras"],
    state.jobs.length ? state.jobs.map((j) => el("tr", {}, el("td", { text: `#${j.id}` }), el("td", { text: fmtTime(j.created_at) }),
      el("td", { class: "mono small", text: j.snapshot_id }), el("td", { text: j.in_place ? "in place" : "separate folder", title: j.target_path }),
      el("td", { text: j.dry_run ? "dry run" : "restore" }), el("td", {}, badge(j.status)), el("td", { class: "num", text: j.files_restored }),
      el("td", { class: "num", text: `${j.files_verified}/${j.files_total}` }), el("td", { class: "num", text: j.files_failed }), el("td", { class: "num", text: j.extra_files })))
      : [el("tr", {}, el("td", { colspan: 10, class: "empty", text: "No restores yet." }))]), { id: "jobs-table" }));
}

function restoreBody(dryRun) {
  const target = selectedTarget();
  return { snapshot_id: state.restoreSel, target_path: target, dry_run: dryRun,
    force: $("#opt-force").checked, quarantine_extras: Boolean(target) && $("#opt-quarantine").checked };
}

function renderRestoreResult(r) {
  const card = clear($("#restore-result"));
  const j = r.job; const sm = r.summary;
  const verified = !j.dry_run && j.status === "succeeded" && j.files_verified === j.files_total;
  fill(card,
    el("div", { class: "row between" }, el("h2", {}, j.dry_run ? "Dry run — nothing was changed" : `Restore job #${j.id}`, " ", badge(j.status)),
      el("button", { class: "btn ghost small", text: "Close", onclick: () => card.classList.add("hidden") })),
    el("p", { class: "muted small" }, `snapshot ${j.snapshot_id} → `, el("code", { text: j.target_path }), j.in_place ? " (in place)" : ""),
    el("div", { class: "chips" }, ["create", "overwrite", "unchanged", "extra"].map((k) => el("span", { class: "chip static" }, `${k} `, el("b", { text: sm[k] })))),
    j.dry_run ? el("p", { text: `Would write ${sm.create + sm.overwrite} file(s); ${sm.unchanged} already identical; ${sm.extra} file(s) on disk are not in the snapshot${j.in_place ? " (e.g. *.locked copies — tick “quarantine” to move them aside)" : ""}.` })
      : el("p", { class: verified ? "verify-ok" : "verify-bad", text: verified
        ? `✓ ${j.files_verified}/${j.files_total} files SHA-256 verified after restore.`
        : `${j.files_failed} file(s) failed verification — see below.` }),
    r.quarantined.length ? el("p", { class: "small" }, `${r.quarantined.length} extra file(s) quarantined (moved, not deleted) to `, el("code", { text: r.quarantine_dir })) : "",
    r.failures.length ? el("div", { class: "notice bad" }, r.failures.slice(0, 20).map((f) => el("div", { class: "small" }, el("code", { text: f.path }), ` — ${f.error}`))) : "",
    !j.dry_run && verified && state.status && state.status.open_incidents && j.in_place
      ? el("div", { class: "notice ok" }, "Folder restored. Next: resolve the incident so new snapshots are trusted again. ", el("a", { href: "#detection", text: "Detection tab →" })) : "",
    el("details", {}, el("summary", { class: "small", text: `File actions (${r.files.length}${r.files_truncated ? "+" : ""})` }),
      el("div", { class: "table-wrap" }, table(["Path", "Action", "Size"], r.files.filter((f) => f.action !== "unchanged").slice(0, MAX_ROWS)
        .map((f) => el("tr", {}, el("td", { class: "path", text: f.path }), el("td", {}, badge(f.action === "create" ? "info" : "warn", f.action)), el("td", { class: "num", text: bytes(f.size) }))))),
      r.extra_files.length ? [el("h3", { text: "Extra files on disk (not in snapshot)" }), el("ul", { class: "small mono" }, r.extra_files.slice(0, 50).map((p) => el("li", { text: p })))] : ""));
  card.classList.remove("hidden");
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function dryRun(btn) {
  if (!state.restoreSel) return toast("Select a snapshot first", "bad");
  const r = await withBusy(btn, () => api("POST", "/restore", restoreBody(true)));
  if (r) { renderRestoreResult(r); refresh(); }
}

async function realRestore(btn) {
  if (!state.restoreSel) return toast("Select a snapshot first", "bad");
  const plan = await withBusy(btn, () => api("POST", "/restore", restoreBody(true)));
  if (!plan) return;
  const body = restoreBody(false);
  const reco = state.candidates && state.candidates.last_clean_snapshot_id;
  const ok = await confirmDialog(body.target_path ? "Restore IN PLACE?" : "Restore to a separate folder?", [
    el("p", {}, "Snapshot ", el("code", { text: body.snapshot_id }), " → ", el("code", { text: plan.job.target_path })),
    reco && body.snapshot_id !== reco
      ? el("div", { class: "notice bad" }, "This is NOT the recommended restore point (", el("code", { text: reco }), "). Restoring an older snapshot discards newer clean work.")
      : el("div", { class: "notice ok", text: "This is the recommended restore point (newest clean snapshot before the incident)." }),
    el("p", { text: `${plan.summary.create} file(s) created, ${plan.summary.overwrite} overwritten, ${plan.summary.unchanged} unchanged.` }),
    body.target_path ? el("div", { class: "notice" }, body.quarantine_extras ? `${plan.summary.extra} extra file(s) will be moved to quarantine.` : `${plan.summary.extra} extra file(s) will be left in place.`) : "",
    el("p", { class: "muted small", text: "Every file is downloaded to a temp file, SHA-256 checked, then atomically moved into place." })], "Restore");
  if (!ok) return;
  const r = await withBusy(btn, () => api("POST", "/restore", body));
  if (r) { renderRestoreResult(r); toast(`Restore job #${r.job.id}: ${r.job.status}`, r.job.status === "succeeded" ? "ok" : "bad"); refresh(); }
}

// ------------------------------------------------------------------ agent
const EXAMPLES = [
  "What changed today, and is it safe to restore?",
  "Was there a ransomware attack in the last 24 hours?",
  "Which snapshot should I restore and why?",
  "Has anything been restored already?",
];

async function ask(btn) {
  const q = $("#question").value.trim();
  if (q.length < 3) return toast("Type a question first", "bad");
  const wrap = $("#answer-wrap"); const ans = clear($("#answer")); const meta = clear($("#answer-meta"));
  wrap.classList.remove("hidden");
  ans.append(el("p", { class: "muted", text: "Thinking — the agent is searching backup metadata and calling tools…" }));
  try {
    btn.disabled = true; btn.classList.add("busy");
    const a = await api("POST", "/agent/ask", { question: q });
    clear(ans).append(renderMarkdown(a.answer));
    const idChips = (ids, cls, onClick) => ids.length ? el("div", { class: "chips" }, ids.map((id) => el("button", { class: `chip ${cls || ""}`, text: id, onclick: onClick ? () => onClick(id) : null }))) : el("span", { class: "muted small", text: "none" });
    fill(meta,
      el("h2", { text: "Recommendation" }),
      a.recommended_snapshot_id ? el("div", {}, el("code", { text: a.recommended_snapshot_id }), " ", badge("clean", "validated clean"),
        el("div", { class: "row", style: "margin-top:.5rem" }, el("button", { class: "btn small", text: "Prepare restore →", onclick: () => { state.restoreSel = a.recommended_snapshot_id; state.restoreSelManual = true; showTab("restore"); renderRestore(); } })))
        : el("p", { class: "muted", text: "No restore recommended." }),
      a.recommendation_warning ? el("div", { class: "notice", text: a.recommendation_warning }) : "",
      el("h3", { text: "Cited snapshots (verified to exist)" }), idChips(a.cited_snapshot_ids, "", (id) => { showTab("snapshots"); showSnapshot(id); }),
      el("h3", { text: "Cited incidents" }), idChips(a.cited_incident_ids.map((i) => `#${i}`), "", () => showTab("detection")),
      a.unverified_ids.length ? [el("h3", { text: "Unverified ids (possible hallucination)" }), idChips(a.unverified_ids, "")] : "",
      el("h3", { text: "How it answered" }),
      el("dl", { class: "kv small" }, el("dt", { text: "time range" }), el("dd", { text: a.time_range || "any time" }),
        el("dt", { text: "model" }), el("dd", { text: a.model }), el("dt", { text: "retrieved" }), el("dd", { text: `${a.retrieved_keys.length} chunks` })),
      el("ol", { class: "small" }, a.tool_calls.map((t) => el("li", {}, el("code", { text: t.name }), t.is_error ? " (error)" : "", el("span", { class: "muted", text: ` ${JSON.stringify(t.input)}` })))),
    );
  } catch (e) {
    clear(ans).append(el("div", { class: `notice ${e.status === 503 ? "" : "bad"}` },
      e.status === 503 ? "Claude is not configured (set ANTHROPIC_API_KEY in .env and restart). Showing what the agent would retrieve instead:" : e.message));
    try {
      const s = await api("GET", `/agent/search?q=${encodeURIComponent(q)}&k=6`);
      ans.append(el("p", { class: "muted small", text: s.time_range ? `Date filter: ${s.time_range}${s.widened ? " (nothing inside it — widened to all time)" : ""}` : "No date filter in the question." }),
        el("ol", {}, s.hits.map((h) => el("li", {}, badge("info", h.kind), el("span", { class: "muted small", text: ` similarity ${num(h.score, 2)} · ${fmtTime(h.timestamp)}` }), el("div", { class: "small", text: h.text })))));
      clear(meta).append(el("h2", { text: "Retrieval only" }), el("p", { class: "muted small", text: "This is the RAG step on its own: date-aware semantic search over snapshot, diff, incident and restore records. With an API key, Claude reads these and calls tools for exact evidence." }));
    } catch (e2) { ans.append(el("p", { class: "muted", text: e2.message })); }
  } finally { btn.disabled = false; btn.classList.remove("busy"); }
}

// -------------------------------------------------------------------- lab
const LAB_STEPS = [
  { key: "seed", title: "Seed test files", desc: "Creates ~120 sample documents, spreadsheets, code and photos in the watched sandbox folder.",
    run: async () => `seeded ${(await api("POST", "/lab/seed", { files: 120 })).seeded} files` },
  { key: "snap", title: "Take a clean baseline snapshot", desc: "Hashes every file and uploads new content to object storage.",
    run: async () => { const s = await api("POST", "/backups", { label: "lab-baseline" }); return `snapshot ${s.id} (${s.state}, ${s.file_count} files)`; } },
  { key: "benign", title: "Normal activity", desc: "Edits a few notes and builds a zip (high entropy, but expected). Watch the live score stay under the threshold.",
    run: async () => `touched ${(await api("POST", "/lab/benign")).touched.length} file(s) — no alert expected` },
  { key: "attack", title: "Simulate a ransomware attack", desc: "Overwrites the seeded copies with random bytes and renames them to *.locked. The watcher should alert within one 10-second window.",
    mode: true, run: async (mode) => { const r = await api("POST", "/lab/attack", { mode }); return `attack started (${r.mode}${r.limit ? `, ${r.limit} files` : ""}) — open the Detection tab`; } },
  { key: "restore", title: "Investigate & restore", desc: "Ask the agent what happened, then restore the recommended snapshot in place with quarantine.",
    nav: true, run: async () => { showTab("restore"); $('input[name="target"][value="inplace"]').checked = true; return "switched to Restore (in place selected)"; } },
  { key: "clean", title: "Clean up", desc: "Removes every file the simulator created (seeded files, .locked copies, ransom note).",
    run: async () => `removed ${(await api("POST", "/lab/clean")).removed} file(s)` },
];

function renderLab() {
  const ol = clear($("#lab-steps"));
  if (state.info && !state.info.lab_enabled) {
    ol.append(el("li", {}, el("div", {}, el("strong", { text: "Lab is disabled" }),
      el("div", { class: "muted small", text: "Set ENABLE_LAB=true (docker compose does this) to drive the simulator from here. Or run python -m simulator.fake_ransomware … from a terminal." }))));
    return;
  }
  for (const step of LAB_STEPS) {
    const result = el("div", { class: "result muted" });
    const mode = step.mode ? el("select", { class: "input", "aria-label": "attack mode" },
      ["fast", "partial", "inplace", "slow"].map((m) => el("option", { value: m, text: m }))) : null;
    const btn = el("button", { class: `btn${step.key === "attack" ? " danger" : step.nav ? " ghost" : ""}`, text: step.nav ? "Go" : "Run" });
    const li = el("li", { class: state.labDone.has(step.key) ? "done" : "" },
      el("div", {}, el("strong", { text: step.title }), el("div", { class: "muted small", text: step.desc })),
      el("div", { class: "row" }, mode, btn), result);
    btn.addEventListener("click", async () => {
      const msg = await withBusy(btn, () => step.run(mode ? mode.value : undefined));
      if (msg !== undefined) { result.textContent = msg; result.classList.remove("muted"); state.labDone.add(step.key); li.classList.add("done"); refresh(); }
    });
    ol.append(li);
  }
}

function renderLabLive() {
  const box = clear($("#live-score"));
  const s = state.status;
  if (!s) return;
  const thr = s.model ? s.model.threshold : 0.5;
  const hist = state.scoreHistory;
  const spark = el("div", { style: "display:flex;align-items:flex-end;gap:2px;height:70px;margin:.5rem 0;border-bottom:1px solid var(--line);position:relative" },
    el("div", { title: `threshold ${num(thr, 3)}`, style: `position:absolute;left:0;right:0;bottom:${thr * 100}%;border-top:2px dashed var(--bad);opacity:.6` }),
    hist.map((h) => el("div", { title: `${new Date(h.t).toLocaleTimeString()} · ${num(h.score, 3)}`,
      style: `flex:1;min-width:3px;height:${Math.max(2, h.score * 100)}%;background:${h.score >= thr ? "var(--bad)" : "var(--accent)"};border-radius:2px 2px 0 0` })));
  box.append(el("div", { class: "row" }, badge(s.monitor.alive ? "ok" : "warn", s.monitor.alive ? "watcher alive" : "watcher offline"),
    el("span", { class: "small", text: `last window ${num(s.monitor.last_score, 3)} · threshold ${num(thr, 3)} · ${s.monitor.alerts} alert(s)` })),
    spark, el("p", { class: "muted small", text: "Each bar is one dashboard poll of the watcher's most recent window score (≈5 s apart). Idle windows score 0." }));
}

// ------------------------------------------------------------------- boot
async function boot() {
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  window.addEventListener("hashchange", () => { const t = location.hash.slice(1); if ($(`#tab-${t}`)) showTab(t); });
  $("#refresh-btn").addEventListener("click", () => refresh());
  $("#snap-now").addEventListener("click", async (ev) => {
    const label = $("#snap-label").value.trim() || null;
    const s = await withBusy(ev.target, () => api("POST", "/backups", { label }));
    if (s) { toast(`Snapshot ${s.id} — ${s.state}, ${s.uploaded_files} new file(s) uploaded`, s.state === "clean" ? "ok" : ""); $("#snap-label").value = ""; refresh(); }
  });
  $("#snap-compare").addEventListener("click", compareSelected);
  $("#scan-now").addEventListener("click", (ev) => scanNow(ev.target));
  $("#restore-dry").addEventListener("click", (ev) => dryRun(ev.target));
  $("#restore-go").addEventListener("click", (ev) => realRestore(ev.target));
  $("#ask-btn").addEventListener("click", (ev) => ask(ev.target));
  $("#question").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) $("#ask-btn").click(); });
  $("#example-qs").append(...EXAMPLES.map((q) => el("button", { class: "chip", text: q, onclick: () => { $("#question").value = q; $("#question").focus(); } })));

  try { state.info = await api("GET", "/info"); } catch (e) { toast(`Cannot reach the API: ${e.message}`, "bad"); }
  renderLab();
  const initial = location.hash.slice(1);
  showTab($(`#tab-${initial}`) ? initial : "overview");
  await refresh();
  setInterval(() => { if (!document.hidden) refresh(); }, POLL_MS);
}

document.addEventListener("DOMContentLoaded", boot);
