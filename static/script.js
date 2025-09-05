// ---------- Elements
const screenLanding = document.getElementById("screen-landing");
const screenResult = document.getElementById("screen-result");

const previewImg = document.getElementById("previewImg");
const resultImg = document.getElementById("resultImg");

const fileInput = document.getElementById("fileInput");
const openPickerFromImage = document.getElementById("openPickerFromImage");
const checkImageBtn = document.getElementById("checkImageBtn");

const tryAnotherBtn = document.getElementById("tryAnotherBtn");
const checkAgainBtn = document.getElementById("checkAgainBtn");
const backToHome = document.getElementById("backToHome");

const toast = document.getElementById("toast");

const headlineVerdict = document.getElementById("headlineVerdict");
const scorePill = document.getElementById("scorePill");

const barGenAI = document.getElementById("bar-genai");
const valGenAI = document.getElementById("val-genai");
const barFace = document.getElementById("bar-facemanip"); // kita pakai utk HUMAN
const valFace = document.getElementById("val-facemanip");

const diffusionList = document.getElementById("diffusionList");
const ganList = document.getElementById("ganList");
const otherList = document.getElementById("otherList");

// ---------- Config
const API_ENDPOINT = "/api/predict";

// ---------- Helpers UI
function openPicker() {
  fileInput.click();
}
function showToast(msg) {
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.remove("hidden");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toast.classList.add("hidden"), 2200);
}
function clamp01(x) {
  const n = Number(x);
  return isNaN(n) ? 0 : Math.max(0, Math.min(1, n));
}
function toPct(num) {
  return Math.round(clamp01(num) * 100);
}
function setBar(elBar, elVal, pct) {
  pct = Math.max(0, Math.min(100, Math.round(pct)));
  elBar.style.width = pct + "%";
  elVal.textContent = pct + "%";
  elBar.classList.remove("ok", "warn", "bad");
  if (pct < 35) elBar.classList.add("ok");
  else if (pct < 65) elBar.classList.add("warn");
  else elBar.classList.add("bad");
}
function toggleSection(listEl, shouldHide) {
  // hide list + judul tepat di atasnya
  const title = listEl?.previousElementSibling;
  listEl?.classList.toggle("hidden", !!shouldHide);
  if (title && title.classList.contains("font-semibold")) {
    title.classList.toggle("hidden", !!shouldHide);
  }
  if (!shouldHide) listEl.innerHTML = ""; // bersihkan kalau nanti mau diisi
}
function renderBreakdown(listEl, obj) {
  const entries = Object.entries(obj || {});
  if (!entries.length) {
    toggleSection(listEl, true);
    return;
  }
  toggleSection(listEl, false);
  entries.forEach(([k, v]) => {
    const pct = typeof v === "number" ? toPct(v) : parseInt(v, 10) || 0;
    const row = document.createElement("div");
    row.className = "grid grid-cols-[auto_1fr_auto] items-center gap-3";
    // (perbaikan) gunakan template literal agar valid
    row.innerHTML = `
      <div class="text-sm text-slate-700">${k}</div>
      <div class="progress-rail w-full"><div class="progress-bar w-0"></div></div>
      <div class="text-sm text-slate-600 w-10 text-right">${pct}%</div>
    `;
    listEl.appendChild(row);
    requestAnimationFrame(() => {
      const bar = row.querySelector(".progress-bar");
      bar.style.width = pct + "%";
      bar.classList.add(pct < 35 ? "ok" : pct < 65 ? "warn" : "bad");
    });
  });
}
function verdictText(aiPct) {
  if (aiPct >= 50) return "Likely AI-generated";
  if (aiPct <= 50) return "Not likely to be AI-generated";
}
function updateScorePill(aiPct) {
  scorePill.textContent = aiPct + "%";
  scorePill.classList.remove("score-good", "score-mid", "score-bad");
  if (aiPct < 35) scorePill.classList.add("score-good");
  else if (aiPct < 65) scorePill.classList.add("score-mid");
  else scorePill.classList.add("score-bad");
}
function switchToResult() {
  screenLanding.classList.add("hidden");
  screenResult.classList.remove("hidden");
  screenResult.classList.add("fade-in");
}
function switchToLanding() {
  screenResult.classList.add("hidden");
  screenLanding.classList.remove("hidden");
  screenLanding.classList.add("fade-in");
}
function readAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(file);
  });
}

// ---------- Normalizer: ambil AI & HUMAN prob dari payload Flask
// Bentuk utama: { success:true, result:{ classes:[{label,prob},...], top:{label,prob} } }
function normalizeFromFlask(data) {
  const res = data?.result || data || {};
  const classes = Array.isArray(res.classes) ? res.classes : [];
  const top = res.top;

  let aiProb = null;
  let humanProb = null;

  // cari langsung dari classes
  const aiCls = classes.find((c) => /\bai\b/i.test(String(c?.label || "")));
  const humanCls = classes.find((c) =>
    /\bhuman\b/i.test(String(c?.label || ""))
  );

  if (aiCls && typeof aiCls.prob === "number") aiProb = aiCls.prob;
  if (humanCls && typeof humanCls.prob === "number") humanProb = humanCls.prob;

  // fallback pakai top
  if (
    aiProb === null &&
    top &&
    typeof top?.prob === "number" &&
    typeof top?.label === "string"
  ) {
    aiProb = /\bai\b/i.test(top.label) ? top.prob : 1 - top.prob;
  }
  // derive human jika belum ada
  if (humanProb === null && typeof aiProb === "number") {
    humanProb = 1 - aiProb;
  }

  aiProb = clamp01(aiProb);
  humanProb = clamp01(humanProb);

  return {
    aiPct: toPct(aiProb),
    humanPct: toPct(humanProb),

    // kalau suatu saat backend kirim detail ini, akan ditampilkan
    diffusion: res.diffusion || {},
    gan: res.gan || {},
    other: res.other || {},
  };
}

// ---------- Networking
function fakePredict() {
  return {
    success: true,
    result: {
      classes: [
        { label: "Human", prob: 0.02 },
        { label: "AI", prob: 0.98 },
      ],
      top: { label: "AI", prob: 0.98 },
    },
  };
}
async function analyzeSelectedFile(file) {
  showToast("Uploading & analyzing…");
  try {
    const form = new FormData();
    form.append("image", file);
    const res = await fetch(API_ENDPOINT, { method: "POST", body: form });
    if (!res.ok) throw new Error("Server returned " + res.status);
    const data = await res.json();
    if (data?.success !== true && !data?.result) {
      throw new Error(data?.error || "Unknown error");
    }
    return data;
  } catch (err) {
    console.warn("Falling back to demo data:", err);
    return fakePredict();
  }
}

// ---------- Render
function relabelAIHumanOnce() {
  // Ubah teks label bar pakai DOM yang ada (ID tidak diubah)
  const labelGen = barGenAI?.closest(".flex")?.previousElementSibling;
  if (labelGen && /genai/i.test(labelGen.textContent))
    labelGen.textContent = "AI";

  const labelFace = barFace?.closest(".flex")?.previousElementSibling;
  if (labelFace && /face/i.test(labelFace.textContent))
    labelFace.textContent = "Human";
}
let _relabeled = false;

async function handleFileSelected(file) {
  if (!file) return;
  window.__LAST_FILE__ = file;

  // Update previews
  const dataURL = await readAsDataURL(file);
  previewImg.src = dataURL;
  resultImg.src = dataURL;

  const raw = await analyzeSelectedFile(file);
  const n = normalizeFromFlask(raw);

  // Headline & pill
  headlineVerdict.textContent = verdictText(n.aiPct);
  updateScorePill(n.aiPct);

  // AI & HUMAN bars
  setBar(barGenAI, valGenAI, n.aiPct);
  setBar(barFace, valFace, n.humanPct);

  // Relabel sekali (tanpa ubah struktur elemen)
  if (!_relabeled) {
    relabelAIHumanOnce();
    _relabeled = true;
  }

  // Hide/Show breakdown only if ada data
  renderBreakdown(diffusionList, n.diffusion);
  renderBreakdown(ganList, n.gan);
  renderBreakdown(otherList, n.other);

  switchToResult();
}

// ---------- Events
openPickerFromImage.addEventListener("click", openPicker);
checkImageBtn.addEventListener("click", openPicker);
resultImg.addEventListener("click", openPicker);

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) handleFileSelected(file);
});

tryAnotherBtn.addEventListener("click", openPicker);
checkAgainBtn.addEventListener("click", async () => {
  if (fileInput.files?.[0]) handleFileSelected(fileInput.files[0]);
  else if (window.__LAST_FILE__) handleFileSelected(window.__LAST_FILE__);
  else openPicker();
});
backToHome.addEventListener("click", switchToLanding);

// Drag & drop (optional)
["dragenter", "dragover"].forEach((evt) => {
  document.addEventListener(evt, (e) => {
    e.preventDefault();
    e.dataTransfer && (e.dataTransfer.dropEffect = "copy");
  });
});
document.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) {
    fileInput.files = e.dataTransfer.files;
    handleFileSelected(f);
  }
});

// ---------- Hydrate from SSR (when /predict renders index.html with result)
(function hydrateFromSSR() {
  try {
    const raw = document.getElementById("SSR_DATA")?.textContent?.trim();
    if (!raw || raw === "null") return;
    const resultObj = JSON.parse(raw); // {classes, top, ...}
    const n = normalizeFromFlask({ success: true, result: resultObj });

    headlineVerdict.textContent = verdictText(n.aiPct);
    updateScorePill(n.aiPct);
    setBar(barGenAI, valGenAI, n.aiPct);
    setBar(barFace, valFace, n.humanPct);

    if (!_relabeled) {
      relabelAIHumanOnce();
      _relabeled = true;
    }

    renderBreakdown(diffusionList, n.diffusion);
    renderBreakdown(ganList, n.gan);
    renderBreakdown(otherList, n.other);

    switchToResult();
  } catch (e) {
    console.warn("SSR hydrate failed", e);
  }
})();

// cari elemen "Model insight summary" tanpa ID baru
function getInsightSubEl() {
  // struktur: <div><div id=headlineVerdict>...</div><div class="text-slate-500 ...">THIS</div></div>
  const parent = document.getElementById("headlineVerdict")?.parentElement;
  if (!parent) return null;
  const subs = parent.querySelectorAll(".text-slate-500");
  return subs[0] || null;
}

function fmtPct01(x) {
  return Math.round(Math.max(0, Math.min(1, Number(x) || 0)) * 100);
}

function fillInsightSummary(meta) {
  const el = getInsightSubEl();
  if (!el || !meta) return;
  const ai = fmtPct01(meta.ai_prob);
  const human = fmtPct01(meta.human_prob);
  const margin = Math.abs(ai - human); // dalam persen
  const t = meta.inference_ms != null ? `${meta.inference_ms}ms` : "—";
  const wh = meta.orig_w && meta.orig_h ? `${meta.orig_w}×${meta.orig_h}` : "—";
  const shape = Array.isArray(meta.input_shape)
    ? meta.input_shape.join("×")
    : "—";
  const thr = meta.threshold != null ? meta.threshold.toFixed(2) : "—";
  const model = meta.model_name || "model";

  // rule kecil: kalau margin kecil, kasih hint review
  const hint = margin < 15 ? " • Needs review (low margin)" : "";

  el.textContent =
    `Model: ${model} • AI=${ai}% • Human=${human}% • Δ=${margin}% • t=${t} • + ` +
    `Input=${shape} • Image=${wh} • Thresh=${thr}${hint}`;
}

// ---- panggil setelah dapat hasil
function applyFromApiPayload(payload) {
  // payload = { success:true, result:{ classes:[...], top:{...}, meta:{...} } }
  const res = payload?.result || {};
  const meta = res.meta || {};
  // ... (update bar AI/Human & verdict kamu di sini seperti biasa)
  fillInsightSummary(meta);
}

// contoh di handler kamu setelah fetch:
// const data = await res.json();
// applyFromApiPayload(data);

// contoh SSR hydrate:
(function hydrateSummaryOnly() {
  try {
    const raw = document.getElementById("SSR_DATA")?.textContent?.trim();
    if (!raw || raw === "null") return;
    const resultObj = JSON.parse(raw);
    const meta = resultObj?.meta;
    fillInsightSummary(meta);
  } catch (e) {}
})();
