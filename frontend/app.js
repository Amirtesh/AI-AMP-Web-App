/* AI-AMP console client v5. Same-origin API. GSAP enhances, never gates. */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
const AA = new Set("ACDEFGHIKLMNPQRSTVWY".split(""));
const HAS_GSAP = typeof window.gsap !== "undefined";
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- entrance choreography (GSAP, guarded) ---------- */
function entrance(scope) {
  if (!HAS_GSAP || REDUCED) return;
  const items = scope.querySelectorAll(".gs-in");
  if (!items.length) return;
  window.gsap.fromTo(
    items,
    { opacity: 0, y: 18 },
    { opacity: 1, y: 0, duration: 0.7, ease: "expo.out", stagger: 0.07, overwrite: true, clearProps: "all" }
  );
}
function animateView(view) {
  if (!HAS_GSAP || REDUCED) return;
  window.gsap.fromTo(
    view,
    { opacity: 0, y: 14 },
    { opacity: 1, y: 0, duration: 0.55, ease: "expo.out", overwrite: true, clearProps: "all" }
  );
}

/* ---------- magnetic buttons (rAF transform, no state) ---------- */
document.querySelectorAll(".magnet").forEach((btn) => {
  let raf = null;
  btn.addEventListener("pointermove", (e) => {
    if (REDUCED) return;
    const r = btn.getBoundingClientRect();
    const x = (e.clientX - r.left - r.width / 2) * 0.12;
    const y = (e.clientY - r.top - r.height / 2) * 0.18;
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      btn.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    });
  });
  btn.addEventListener("pointerleave", () => {
    if (raf) cancelAnimationFrame(raf);
    btn.style.transform = "";
  });
});

/* ---------- tabs ---------- */
const navLinks = [...document.querySelectorAll(".nav-link")];
navLinks.forEach((btn) =>
  btn.addEventListener("click", () => {
    navLinks.forEach((x) => x.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    const view = $("view-" + btn.dataset.tab);
    view.classList.add("active");
    window.scrollTo({ top: 0, behavior: REDUCED ? "auto" : "smooth" });
    animateView(view);
    if (btn.dataset.tab === "validation") loadStats();
  })
);

/* ---------- status ---------- */
const statusEl = $("api-status");
const statusLabel = statusEl.querySelector("span");
async function ping() {
  try {
    const r = await fetch("/");
    if (!r.ok) throw new Error(r.status);
    statusEl.className = "live-dot live";
    statusLabel.textContent = "api live";
  } catch {
    statusEl.className = "live-dot down";
    statusLabel.textContent = "api unreachable";
  }
}
ping();
setInterval(ping, 60000);

/* ---------- errors ---------- */
function setErrors(elId, items) {
  const el = $(elId);
  el.innerHTML = "";
  if (!items || !items.length) return;
  items.slice(0, 8).forEach((e) => {
    const d = document.createElement("div");
    d.className = "err";
    d.textContent = typeof e === "string" ? e : e.reason || e.message || JSON.stringify(e);
    el.appendChild(d);
  });
  if (items.length > 8) {
    const d = document.createElement("div");
    d.className = "err";
    d.textContent = `+${items.length - 8} more`;
    el.appendChild(d);
  }
}

/* ================= PREDICT ================= */
let predictMode = "combined";
document.querySelectorAll("#predict-mode .mode-btn").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#predict-mode .mode-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    predictMode = b.dataset.mode;
    $("seq-input").dispatchEvent(new Event("input"));
  })
);

const seqInput = $("seq-input");
seqInput.addEventListener("input", () => {
  const lines = seqInput.value.split("\n").map((s) => s.trim()).filter(Boolean);
  $("seq-count").textContent = `${lines.length} sequence${lines.length === 1 ? "" : "s"}`;
});

let lastCsv = "";
let lastCsvName = "aiamp_predictions.csv";

document.querySelectorAll(".toolbar-tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".toolbar-tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    $(t.dataset.target === "table-view" ? "table-view" : "csv-view").classList.add("active");
  })
);

$("download-csv").addEventListener("click", () => {
  if (!lastCsv) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lastCsv], { type: "text/csv" }));
  a.download = lastCsvName;
  a.click();
  URL.revokeObjectURL(a.href);
});

$("predict-btn").addEventListener("click", async () => {
  const btn = $("predict-btn");
  const sequences = seqInput.value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!sequences.length) {
    setErrors("predict-errors", ["Enter at least one sequence to score."]);
    return;
  }
  btn.disabled = true;
  $("predict-progress").hidden = false;
  $("predict-empty").hidden = true;
  $("predict-progress-text").textContent = `Scoring ${sequences.length} sequence${sequences.length === 1 ? "" : "s"}…`;
  setErrors("predict-errors", []);
  try {
    const r = await fetch(`/predict/${predictMode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequences }),
    });
    const data = await r.json();
    if (!r.ok) {
      setErrors("predict-errors", data.detail?.errors || [data.detail?.message || `Request failed (${r.status}). Check the input and try again.`]);
      $("predict-empty").hidden = false;
      return;
    }
    renderPredict(data);
  } catch (e) {
    setErrors("predict-errors", ["Could not reach the API. Confirm the server is running, then try again."]);
    $("predict-empty").hidden = false;
  } finally {
    btn.disabled = false;
    $("predict-progress").hidden = true;
  }
});

function verdict(prob, threshold) {
  const hit = prob >= threshold;
  return `<span class="verdict ${hit ? "hit" : "miss"}">${hit ? "positive" : "negative"}</span>`;
}

function renderPredict(data) {
  const rows = data.results;
  const m1 = rows[0]?.model1 || (predictMode !== "model2" ? rows[0] : null);
  const hasM1 = !!(m1 && m1.gram_positive_prob !== undefined);
  const m2 = rows[0]?.model2 || (predictMode !== "model1" ? rows[0] : null);
  const hasM2 = !!(m2 && m2.prediction !== undefined);
  const th = data.model1_thresholds || data.thresholds || {};
  const th2 = data.model2_threshold;

  const head = ["Sequence"];
  if (hasM2) head.push("AMP call", "AMP prob");
  if (hasM1) head.push("Gram+", "Gram−", "Fungal", "G+ p", "G− p", "F p");
  $("results-thead").innerHTML = "<tr>" + head.map((h) => `<th>${esc(h)}</th>`).join("") + "</tr>";

  $("results-tbody").innerHTML = rows
    .map((row) => {
      const r1 = row.model1 || (predictMode !== "model2" ? row : null);
      const r2 = row.model2 || (predictMode !== "model1" ? row : null);
      const seq = esc(row.sequence ?? r1?.sequence ?? "");
      let tds = [`<td title="${seq}">${seq}</td>`];
      if (hasM2) {
        tds.push(`<td>${verdict(r2.probability, th2 ?? 0.5)} ${esc(r2.prediction)}</td>`);
        tds.push(`<td>${r2.probability.toFixed(3)}</td>`);
      }
      if (hasM1) {
        tds.push(
          `<td>${verdict(r1.gram_positive_prob, th.gram_positive ?? 0.5)}</td>`,
          `<td>${verdict(r1.gram_negative_prob, th.gram_negative ?? 0.5)}</td>`,
          `<td>${verdict(r1.fungal_prob, th.fungal ?? 0.5)}</td>`,
          `<td>${r1.gram_positive_prob.toFixed(3)}</td>`,
          `<td>${r1.gram_negative_prob.toFixed(3)}</td>`,
          `<td>${r1.fungal_prob.toFixed(3)}</td>`
        );
      }
      return `<tr class="gs-in">${tds.join("")}</tr>`;
    })
    .join("");

  const csv = predictMode === "combined" ? data.model1_table : data.table;
  lastCsv = csv || "";
  lastCsvName = `aiamp_${predictMode}_predictions.csv`;
  $("csv-text").textContent = lastCsv;
  const chips = [];
  if (hasM1) Object.entries(th).forEach(([k, v]) => chips.push(`<span class="threshold-chip">${esc(k)} ≥ <b>${v}</b></span>`));
  if (hasM2 && th2 !== undefined) chips.push(`<span class="threshold-chip">amp ≥ <b>${th2}</b></span>`);
  if (predictMode === "combined") chips.push(`<span class="threshold-chip">one shared embedding</span>`);
  $("thresholds-strip").innerHTML = chips.join("");

  $("predict-output").hidden = false;
  entrance($("predict-output"));
}

/* ================= GENERATE ================= */
let genStage = "stage2";
document.querySelectorAll(".stage-card").forEach((c) =>
  c.addEventListener("click", () => {
    document.querySelectorAll(".stage-card").forEach((x) => {
      x.classList.remove("active");
      x.setAttribute("aria-pressed", "false");
    });
    c.classList.add("active");
    c.setAttribute("aria-pressed", "true");
    genStage = c.dataset.stage;
    $("stage1-warn").hidden = genStage !== "stage1";
  })
);

[["gen-n", "gen-n-val", (v) => v], ["gen-min", "gen-min-val", (v) => v], ["gen-max", "gen-max-val", (v) => v],
 ["gen-temp", "gen-temp-val", (v) => (+v).toFixed(1)], ["gen-topp", "gen-topp-val", (v) => (+v).toFixed(2)]
].forEach(([a, b, f]) => {
  const el = $(a);
  el.addEventListener("input", () => ($(b).textContent = f(el.value)));
});

let lastGenSeqs = [];
$("generate-btn").addEventListener("click", async () => {
  const btn = $("generate-btn");
  const body = {
    n_sequences: +$("gen-n").value,
    min_length: +$("gen-min").value,
    max_length: +$("gen-max").value,
    temperature: +$("gen-temp").value,
    top_p: +$("gen-topp").value,
  };
  if (body.min_length > body.max_length) {
    setErrors("gen-errors", [`Min length (${body.min_length}) is above max length (${body.max_length}). Lower the minimum first.`]);
    return;
  }
  btn.disabled = true;
  $("gen-progress").hidden = false;
  $("gen-progress-text").textContent = `Sampling ${body.n_sequences} peptides…`;
  setErrors("gen-errors", []);
  try {
    const r = await fetch(`/generate/${genStage}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) {
      setErrors("gen-errors", [data.detail?.message || `Request failed (${r.status}). Try fewer sequences or a shorter max length.`]);
      return;
    }
    lastGenSeqs = data.sequences;
    $("gen-result-label").textContent =
      `${data.sequences.length} sequences · ${data.model} · amp_validated=${data.amp_validated}`;
    const list = $("seq-list");
    list.innerHTML = "";
    data.sequences.forEach((s, i) => {
      const d = document.createElement("div");
      d.className = "seq-row gs-in";
      const n = document.createElement("span");
      n.className = "n";
      n.textContent = String(i + 1).padStart(2, "0");
      const span = document.createElement("span");
      span.className = "s";
      span.textContent = s;
      const len = document.createElement("span");
      len.className = "len";
      len.textContent = `${s.length} aa`;
      const copy = document.createElement("button");
      copy.className = "copy-btn";
      copy.textContent = "copy";
      copy.addEventListener("click", async () => {
        await navigator.clipboard.writeText(s);
        copy.textContent = "copied";
        setTimeout(() => (copy.textContent = "copy"), 1200);
      });
      d.append(n, span, len, copy);
      list.appendChild(d);
    });
    $("gen-output").hidden = false;
    entrance(list);
  } catch (e) {
    setErrors("gen-errors", ["Could not reach the API. Confirm the server is running, then try again."]);
  } finally {
    btn.disabled = false;
    $("gen-progress").hidden = true;
  }
});

$("copy-all-seq").addEventListener("click", async () => {
  if (!lastGenSeqs.length) return;
  await navigator.clipboard.writeText(lastGenSeqs.join("\n"));
  $("copy-all-seq").textContent = "Copied";
  setTimeout(() => ($("copy-all-seq").innerHTML = '<i class="ph ph-copy" aria-hidden="true"></i>Copy all'), 1200);
});

/* ================= VALIDATION (per model) ================= */
let statsLoaded = false;
let statsLoading = false;

function metricTable(rows) {
  if (!rows || !rows.length) return '<div class="empty-state"><h2>No rows</h2><p>Nothing recorded for this metric.</p></div>';
  const cols = Object.keys(rows[0]);
  return `<div class="table-wrap"><table class="data-table"><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${cols.map((c) => `<td>${esc(row[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function plotsGrid(images, bento) {
  if (!images.length) return "";
  return `<div class="${bento ? "plots-bento" : "plots-duo"}">${images.map((img) => `<figure class="plot gs-in" data-full="${img.url}" data-cap="${esc(img.description)}"><img loading="lazy" src="${img.url}" alt="${esc(img.description)}" /><figcaption>${esc(img.description)}</figcaption></figure>`).join("")}</div>`;
}

function skeleton(body) {
  body.innerHTML = '<div class="skel-table"><div class="skel-line" style="width:38%"></div><div class="skel-line" style="width:92%"></div><div class="skel-line" style="width:71%"></div><div class="skel-line" style="width:84%"></div></div>';
}

function bindLightbox(scope) {
  scope.querySelectorAll(".plot").forEach((p) =>
    p.addEventListener("click", () => {
      $("lb-img").src = p.dataset.full;
      $("lb-cap").textContent = p.dataset.cap;
      $("lightbox").hidden = false;
    })
  );
}
$("lightbox").addEventListener("click", () => {
  $("lightbox").hidden = true;
  $("lb-img").removeAttribute("src");
});

async function loadStats() {
  if (statsLoaded || statsLoading) return;
  statsLoading = true;
  const m1 = $("m1-val-body"), m2 = $("m2-val-body"), g = $("gen-val-body");
  [m1, m2, g].forEach(skeleton);
  try {
    const r = await fetch("/stats");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    const images = data.images || [];
    const m2imgs = images.filter((i) => i.name.startsWith("model2_"));
    const gimgs = images.filter((i) => !i.name.startsWith("model2_"));

    const threshRows = Object.entries(data.metrics.model1.thresholds).map(([k, v]) => ({ head: k, threshold: v }));
    m1.innerHTML =
      '<div class="val-sub">Decision thresholds</div>' + metricTable(threshRows) +
      '<div class="val-sub">External validation summary</div>' + metricTable(data.metrics.model1.external_validation_summary);

    m2.innerHTML =
      '<div class="val-sub">Ablation results with thresholds</div>' + metricTable(data.metrics.model2.ablation_results_with_thresholds) +
      '<div class="val-sub">External validation plots</div>' + plotsGrid(m2imgs, false);

    g.innerHTML =
      '<div class="val-sub">Configuration</div>' +
      metricTable([{ parameter: "max generation length", value: `${data.metrics.generative.max_generation_length} aa` }]) +
      '<div class="val-sub">Training diagnostics</div>' + plotsGrid(gimgs, true);

    const scope = $("view-validation");
    bindLightbox(scope);
    entrance(scope);
    statsLoaded = true;
  } catch (e) {
    [m1, m2, g].forEach((b) => {
      b.innerHTML = `<div class="empty-state"><h2>Validation data did not load</h2><p>${esc(e.message)}. Check the API status above, then reopen this tab.</p></div>`;
    });
  } finally {
    statsLoading = false;
  }
}
