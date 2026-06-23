"use strict";

// --- tiny helpers ------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return (s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// Highlight inline page citations like "(p.227)" the model produces.
function highlightCites(text) {
  return escapeHtml(text).replace(/\(p\.?\s?\d+[^)]*\)/gi, (m) => `<span class="cite">${m}</span>`);
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// --- initial load ------------------------------------------------------------
async function refreshAll() {
  await Promise.all([loadStatus(), loadReports(), loadFacts()]);
}

async function loadStatus() {
  try {
    const h = await api("/api/health");
    const n = h.chunks_indexed;
    $("index-status").textContent = h.openai_key_set
      ? `${n.toLocaleString()} chunks indexed`
      : "no API key set — add OPENAI_API_KEY to .env";
  } catch (e) {
    $("index-status").textContent = "backend unreachable";
  }
}

async function loadReports() {
  const reports = await api("/api/reports");
  const sel = $("company-filter");
  const current = sel.value;
  sel.innerHTML = '<option value="">All reports</option>';
  for (const r of reports) {
    const opt = document.createElement("option");
    opt.value = r.company;
    opt.textContent = r.company;
    sel.appendChild(opt);
  }
  sel.value = current; // preserve selection across refreshes
  updateEmptyState(reports.length);
}

// Guide a first-time user: with no reports, open the upload form and point the
// chat hint at uploading rather than dangling an example for missing data.
function updateEmptyState(reportCount) {
  const hint = $("chat-hint");
  if (reportCount === 0) {
    $("upload-form").classList.remove("hidden");
    if (hint) {
      hint.innerHTML = 'No reports yet — upload an annual report PDF with ' +
        '<strong>“+ Upload”</strong> to get started.';
    }
  } else if (hint) {
    hint.innerHTML = 'Ask a question about the ingested reports — e.g. ' +
      '<em>"What are the sustainability goals?"</em>';
  }
}

async function loadFacts() {
  const facts = await api("/api/facts");
  const box = $("facts");
  if (!facts.length) {
    box.innerHTML = '<div class="empty-hint">No reports ingested yet. Use “+ Upload”.</div>';
    return;
  }
  box.innerHTML = facts.map(renderFactCard).join("");
}

function renderFactCard(f) {
  const initials = f.company.slice(0, 2).toUpperCase();
  const ftePages = f.fte.pages?.length ? ` <span class="pg">p.${f.fte.pages.join(", ")}</span>` : "";
  const goals = (f.sustainability_goals || []).map((g) => {
    const gp = g.pages?.length ? ` <span class="pg">p.${g.pages.join(", ")}</span>` : "";
    return `<div class="goal"><span class="dot">&#9679;</span><span>${escapeHtml(g.goal)}${gp}</span></div>`;
  }).join("");

  return `
    <div class="fact-card">
      <div class="fact-head">
        <div class="avatar">${escapeHtml(initials)}</div>
        <div>
          <div class="fact-company">${escapeHtml(f.company)}</div>
          <div class="fact-sub">${escapeHtml(f.source)}</div>
        </div>
      </div>
      <div class="metric">
        <div class="metric-label">Employees (FTE)</div>
        <div class="metric-value">${escapeHtml(f.fte.value)}${ftePages}</div>
        ${f.fte.quote ? `<div class="metric-quote">“${escapeHtml(f.fte.quote)}”</div>` : ""}
      </div>
      <div class="goals-label">Sustainability goals (${f.sustainability_goals.length})</div>
      ${goals || '<div class="muted">none extracted</div>'}
    </div>`;
}

// --- chat --------------------------------------------------------------------
function addMessage(html, cls) {
  $("chat-hint")?.remove();
  const div = document.createElement("div");
  div.className = cls;
  div.innerHTML = html;
  $("messages").appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
  return div;
}

function renderSources(sources) {
  if (!sources?.length) return "";
  const items = sources.map((s) => {
    const text = s.text.replace(/\s+/g, " ").trim();
    return `<div class="source-item">
      <div class="src-head">${escapeHtml(s.company)} · page ${s.page}</div>
      ${escapeHtml(text.slice(0, 320))}${text.length > 320 ? "…" : ""}
    </div>`;
  }).join("");
  return `<details class="sources"><summary>&#9656; ${sources.length} source(s) · click to verify</summary>${items}</details>`;
}

async function onChatSubmit(e) {
  e.preventDefault();
  const q = $("question").value.trim();
  if (!q) return;
  const company = $("company-filter").value || null;

  addMessage(escapeHtml(q), "msg-user");
  $("question").value = "";
  $("send-btn").disabled = true;

  const thinking = addMessage('<div class="bubble muted">Thinking…</div>', "msg-bot");
  try {
    const r = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, company }),
    });
    thinking.innerHTML = `<div class="bubble">${highlightCites(r.answer)}</div>${renderSources(r.sources)}`;
  } catch (err) {
    thinking.innerHTML = `<div class="bubble">Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    $("send-btn").disabled = false;
    $("messages").scrollTop = $("messages").scrollHeight;
  }
}

// --- upload ------------------------------------------------------------------
async function onUploadSubmit(e) {
  e.preventDefault();
  const file = $("upload-file").files[0];
  const company = $("upload-company").value.trim();
  if (!file || !company) return;

  const fd = new FormData();
  fd.append("file", file);
  fd.append("company", company);

  $("upload-btn").disabled = true;
  $("upload-status").textContent = "ingesting… (parsing + embedding can take a minute)";
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    $("upload-status").textContent = `done: ${r.chunks} chunks, FTE ${r.facts.fte}`;
    $("upload-form").reset();
    await refreshAll();
  } catch (err) {
    $("upload-status").textContent = "failed: " + err.message;
  } finally {
    $("upload-btn").disabled = false;
  }
}

// --- wire up -----------------------------------------------------------------
$("chat-form").addEventListener("submit", onChatSubmit);
$("upload-form").addEventListener("submit", onUploadSubmit);
$("upload-toggle").addEventListener("click", () => $("upload-form").classList.toggle("hidden"));
refreshAll();
