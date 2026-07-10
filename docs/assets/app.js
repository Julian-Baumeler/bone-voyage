/**
 * Bone Voyage / OpenGEM — drop CT → optional paint → Generate → 3D
 *
 * When served from GitHub Pages (or any non-local host), API calls go to
 * the local backend at http://127.0.0.1:8742 (run `opengem serve`).
 * Override with ?api=http://host:port or window.OPENGEM_API.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const state = {
  ctArray: null,
  ctShape: null, // [z,y,x] voxels
  ctSpacing: [1, 1, 1], // [sx,sy,sz] mm — for correct aspect ratio
  paint: null,
  tool: "include",
  view: {},
  painting: false,
  hasMesh: false,
  bones: [], // from auto-split
  boneVisible: {}, // id -> bool
  boneMeshes: {}, // id -> THREE.Mesh
  boneLabels: null, // Uint16Array ZYX — label id per voxel (same as Bone N)
  boneLabelShape: null,
  boneColorById: {}, // id -> [r,g,b]
};

const $ = (s) => document.querySelector(s);

/** Local backend when UI is hosted statically (e.g. GitHub Pages). */
function resolveApiBase() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("api")) return params.get("api").replace(/\/$/, "");
  if (typeof window.OPENGEM_API === "string" && window.OPENGEM_API) {
    return window.OPENGEM_API.replace(/\/$/, "");
  }
  const host = window.location.hostname;
  const local =
    host === "localhost" ||
    host === "127.0.0.1" ||
    host === "" ||
    host === "[::1]";
  // Same-origin when FastAPI serves the UI; localhost backend when on Pages.
  return local ? "" : "http://127.0.0.1:8742";
}

/** Mutable so Advanced “API URL” can re-point without full reload. */
let API_BASE = resolveApiBase();

const REPO = "Julian-Baumeler/bone-voyage";
const ENGINE = {
  zip: `https://github.com/${REPO}/archive/refs/heads/main.zip`,
  repo: `https://github.com/${REPO}`,
  installRaw: `https://raw.githubusercontent.com/${REPO}/main/scripts/install-engine.sh`,
  pages: `https://julian-baumeler.github.io/bone-voyage/`,
  defaultApi: "http://127.0.0.1:8742",
};

let engineOnline = false;
let enginePollTimer = null;
let setupDismissed = false;

function apiUrl(path) {
  if (!path) return API_BASE || "/";
  if (/^https?:\/\//i.test(path)) return path;
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${p}`;
}

function setStatus(msg, kind = "") {
  const el = $("#status");
  el.textContent = msg || "";
  el.className = "status " + (kind || "muted");
}

function setEngineDot(mode) {
  const el = $("#engine-dot");
  if (!el) return;
  el.className = "engine-dot " + (mode || "offline");
  el.title =
    mode === "online"
      ? "Local engine online"
      : mode === "checking"
        ? "Checking local engine…"
        : "Local engine offline";
}

function oneLinerInstall() {
  return `curl -fsSL ${ENGINE.installRaw} | bash`;
}

function fillEngineSetupLinks() {
  const zip = $("#btn-download-zip");
  if (zip) {
    zip.href = ENGINE.zip;
    zip.setAttribute("download", "bone-voyage-engine.zip");
  }
  const repo = $("#link-repo");
  if (repo) repo.href = ENGINE.repo;
  const ol = $("#install-oneliner");
  if (ol) ol.textContent = oneLinerInstall();
  const local = $("#install-local");
  if (local) {
    local.textContent =
      "cd bone-voyage-main && bash scripts/install-engine.sh";
  }
  const start = $("#start-cmd");
  if (start) start.textContent = "~/bone-voyage/start-engine.sh";
  const apiIn = $("#api-url-input");
  if (apiIn && !apiIn.value) {
    apiIn.value = API_BASE || ENGINE.defaultApi;
  }
}

function showEngineSetup(show) {
  const el = $("#engine-setup");
  if (!el) return;
  if (show && !setupDismissed) el.classList.remove("hidden");
  else el.classList.add("hidden");
}

function setBackendBanner(online) {
  const el = $("#backend-banner");
  const chip = $("#btn-engine-setup");
  if (chip) {
    chip.textContent = online ? "Engine online" : "Get engine";
    chip.classList.toggle("online", !!online);
  }
  setEngineDot(online ? "online" : "offline");
  if (!el) return;
  if (online) {
    el.classList.add("hidden");
    showEngineSetup(false);
    return;
  }
  // Same-origin local serve: short hint only
  if (!API_BASE) {
    el.classList.add("hidden");
    return;
  }
  el.classList.remove("hidden");
  el.innerHTML =
    `<span><strong>Local engine offline.</strong> UI is static; compute runs on your machine at ` +
    `<code>${API_BASE}</code>.</span>` +
    `<button type="button" class="btn btn-inline primary" id="btn-banner-setup">Download &amp; start engine</button>`;
  $("#btn-banner-setup")?.addEventListener("click", () => {
    setupDismissed = false;
    showEngineSetup(true);
  });
}

function setPollStatus(msg, kind = "") {
  const el = $("#engine-poll-status");
  if (!el) return;
  el.textContent = msg;
  el.className = "engine-poll" + (kind ? " " + kind : "");
}

async function probeEngine() {
  setEngineDot("checking");
  try {
    const res = await fetch(apiUrl("/api/health"), { cache: "no-store" });
    if (!res.ok) throw new Error(res.statusText);
    const h = await res.json();
    return h;
  } catch (e) {
    setEngineDot("offline");
    throw e;
  }
}

async function onEngineOnline(h, { first = false } = {}) {
  engineOnline = true;
  setBackendBanner(true);
  setPollStatus(
    `Connected — ${h.product || "Bone Voyage"} ${h.version || ""}`.trim(),
    "ok"
  );
  const product = h.product || "Bone Voyage";
  setStatus(`${product} ${h.version || ""} · drop a CT to auto-split bones`, "ok");
  if (first) {
    setDebug(
      "Ready.\n\n1. Drop a CT — we analyze ALL slices\n" +
        "2. Bones appear as Bone 1, Bone 2, … with checkboxes\n" +
        "3. Hide/unhide in 3D · checked bones go to Generate\n" +
        (API_BASE ? `\nAPI: ${API_BASE}\n` : "")
    );
    try {
      const d = await refreshServerDebug();
      try {
        await loadBonesFromServer();
      } catch (_) {
        if (d?.has_preview_file || d?.has_surface_file) {
          await loadPreviewFromServer();
        }
      }
    } catch (_) {}
  }
  stopEnginePoll();
}

function onEngineOffline({ showSetup = true } = {}) {
  engineOnline = false;
  setBackendBanner(false);
  setStatus(
    API_BASE
      ? `Engine offline — download & start local compute`
      : "Server not reachable",
    "error"
  );
  setDebug(
    API_BASE
      ? `This page is the UI only.\n\n` +
          `1. Download the engine (button or zip from GitHub)\n` +
          `2. Install:  ${oneLinerInstall()}\n` +
          `3. Start:    ~/bone-voyage/start-engine.sh\n` +
          `4. Stay on this tab — it auto-connects to ${API_BASE}\n`
      : "Could not reach /api/health. Is opengem serve running?"
  );
  if (showSetup && API_BASE) showEngineSetup(true);
  startEnginePoll();
}

function startEnginePoll() {
  if (enginePollTimer || engineOnline) return;
  let n = 0;
  enginePollTimer = setInterval(async () => {
    n += 1;
    setPollStatus(`Listening for engine at ${API_BASE || "(same origin)"}… (#${n})`);
    try {
      const h = await probeEngine();
      await onEngineOnline(h, { first: true });
      setPollStatus("Engine online — you can drop a CT.", "ok");
    } catch (_) {
      /* keep waiting */
    }
  }, 2500);
}

function stopEnginePoll() {
  if (enginePollTimer) {
    clearInterval(enginePollTimer);
    enginePollTimer = null;
  }
}

function wireEngineSetupUi() {
  fillEngineSetupLinks();

  $("#btn-engine-setup")?.addEventListener("click", () => {
    if (engineOnline) {
      setStatus("Local engine is already online", "ok");
      return;
    }
    setupDismissed = false;
    showEngineSetup(true);
  });

  $("#btn-dismiss-setup")?.addEventListener("click", () => {
    setupDismissed = true;
    showEngineSetup(false);
  });

  $("#btn-check-engine")?.addEventListener("click", async () => {
    setPollStatus("Checking…");
    try {
      const h = await probeEngine();
      await onEngineOnline(h, { first: true });
    } catch (e) {
      setPollStatus(
        `Still offline (${e.message || "unreachable"}). Is start-engine.sh running?`,
        "err"
      );
      setEngineDot("offline");
    }
  });

  $("#btn-apply-api")?.addEventListener("click", () => {
    const v = ($("#api-url-input")?.value || "").trim().replace(/\/$/, "");
    if (!v) return;
    API_BASE = v;
    const url = new URL(window.location.href);
    url.searchParams.set("api", v);
    window.history.replaceState({}, "", url);
    setPollStatus(`API set to ${v} — checking…`);
    engineOnline = false;
    stopEnginePoll();
    startEnginePoll();
    probeEngine()
      .then((h) => onEngineOnline(h, { first: true }))
      .catch(() => onEngineOffline({ showSetup: true }));
  });

  document.querySelectorAll(".btn-copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-copy");
      const code = id ? document.getElementById(id) : null;
      const text = code?.textContent || "";
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = "Copy";
          btn.classList.remove("copied");
        }, 1500);
      } catch (_) {
        btn.textContent = "Select & copy";
      }
    });
  });
}

function setDebug(obj, kind = "") {
  const el = $("#debug");
  const text = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  el.textContent = text;
  el.className = "debug" + (kind ? " " + kind : "");
}

async function api(path, opts = {}) {
  const res = await fetch(apiUrl(path), opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || j.error || JSON.stringify(j);
    } catch (_) {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}

// --- CT load ---
async function loadCtFile(file) {
  if (!file) return;
  setStatus("Uploading CT…", "busy");
  setDebug("Uploading " + file.name + "…");
  clearBoneMeshes();
  state.bones = [];
  state.boneLabels = null;
  state.boneColorById = {};
  $("#bones-list").innerHTML = "";
  $("#bones-status").textContent = "Analyzing bones…";
  const fd = new FormData();
  fd.append("file", file);
  const info = await api("/api/session/upload-ct", { method: "POST", body: fd });
  $("#ct-info").textContent =
    `${info.filename}\n${info.shape.join(" × ")}\n` +
    `HU ${info.min.toFixed(0)} … ${info.max.toFixed(0)} (p99 ${info.p99.toFixed(0)})`;
  $("#drop-label").innerHTML = `<strong>${info.filename}</strong><br/><small>Click to replace</small>`;
  await loadCtVolume();
  setStatus("CT loaded — detecting separate bones…", "busy");
  setDebug({
    loaded: true,
    shape: info.shape,
    hu: [info.min, info.max],
    split_job: info.split_job_id,
    tip: "Auto bone split running on all slices…",
  });
  if (info.split_job_id) {
    await pollBoneSplitJob(info.split_job_id);
  }
}

async function pollBoneSplitJob(jobId) {
  showProgress(true);
  setProgress(2, "Bone analysis queued…", "load");
  let finished = false;
  while (!finished) {
    await new Promise((r) => setTimeout(r, 400));
    const job = await api(`/api/session/jobs/${jobId}`);
    setProgress(job.percent || 0, `${job.message} (${job.elapsed_s}s)`, job.step === "mesh" ? "surface" : job.step);
    if (job.log?.length) {
      setDebug(`Bone analysis (${job.elapsed_s}s)\n` + job.log.join("\n"));
    }
    if (job.state === "done" || job.state === "error") {
      finished = true;
      if (job.state === "error") {
        $("#bones-status").textContent = "Bone detect failed: " + (job.error || "?");
        setStatus(job.error || "Bone detect failed", "error");
      } else {
        await loadBonesFromServer();
        setProgress(100, `Found ${state.bones.length} bones`, "done");
        setStatus(`${state.bones.length} bones — toggle checkboxes, then Generate`, "ok");
      }
    }
  }
}

function hexToRgb(hex) {
  // accepts 0x7ec8ff or "#7ec8ff"
  let n = hex;
  if (typeof hex === "string") {
    n = parseInt(hex.replace("#", ""), 16);
  }
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

async function loadBonesFromServer() {
  const data = await api("/api/session/bones");
  state.bones = data.bones || [];
  state.boneVisible = {};
  state.boneColorById = {};
  state.bones.forEach((b) => {
    state.boneVisible[b.id] = true;
    state.boneColorById[b.id] = hexToRgb(b.color_hex ?? b.color ?? 0x7ec8ff);
  });
  renderBonesList();
  if (state.bones.length) {
    $("#bones-status").textContent = `${state.bones.length} bones detected — colored on 2D + 3D`;
    await loadBoneLabels();
    drawSlice(); // refresh 2D with bone colors
    await loadAllBoneMeshes();
  } else {
    $("#bones-status").textContent = "No bones found — lower Bone HU min and re-detect";
    state.boneLabels = null;
  }
}

async function loadBoneLabels() {
  try {
    const res = await fetch(apiUrl("/api/session/bones-labels"));
    if (!res.ok) throw new Error("labels " + res.status);
    const shape = res.headers.get("X-Shape").split(",").map(Number);
    const buf = await res.arrayBuffer();
    state.boneLabels = new Uint16Array(buf);
    state.boneLabelShape = shape;
    const n = shape[0] * shape[1] * shape[2];
    if (state.boneLabels.length !== n) {
      console.warn("label size mismatch", state.boneLabels.length, n);
    }
  } catch (e) {
    console.warn("bone labels load failed", e);
    state.boneLabels = null;
  }
}

function renderBonesList() {
  const box = $("#bones-list");
  box.innerHTML = "";
  state.bones.forEach((b) => {
    const row = document.createElement("label");
    row.className = "bone-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.boneVisible[b.id] !== false;
    cb.dataset.boneId = String(b.id);
    cb.addEventListener("change", () => {
      state.boneVisible[b.id] = cb.checked;
      if (state.boneMeshes[b.id]) {
        state.boneMeshes[b.id].visible = cb.checked;
      }
      drawSlice(); // 2D colors follow visibility
    });
    const sw = document.createElement("span");
    sw.className = "bone-swatch";
    sw.style.background = b.color || "#7ec8ff";
    const name = document.createElement("span");
    name.textContent = b.name || `Bone ${b.id}`;
    const meta = document.createElement("span");
    meta.className = "bone-meta";
    meta.textContent = `${(b.voxels / 1000).toFixed(1)}k vx`;
    row.appendChild(cb);
    row.appendChild(sw);
    row.appendChild(name);
    row.appendChild(meta);
    box.appendChild(row);
  });
}

function clearBoneMeshes() {
  if (!three) {
    state.boneMeshes = {};
    return;
  }
  Object.values(state.boneMeshes).forEach((m) => {
    three.scene.remove(m);
    m.geometry?.dispose();
    m.material?.dispose();
  });
  state.boneMeshes = {};
  if (three.mesh) {
    three.scene.remove(three.mesh);
    three.mesh.geometry?.dispose();
    three.mesh.material?.dispose();
    three.mesh = null;
  }
}

async function loadAllBoneMeshes() {
  showMeshTab();
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const t = ensureThree();
  clearBoneMeshes();
  $("#mesh-empty").classList.add("hidden");

  let any = false;
  const box = new THREE.Box3();
  for (const b of state.bones) {
    if (!b.preview_url) continue;
    try {
      const preview = await api(b.preview_url);
      if (!preview?.points?.length) continue;
      const mesh = previewToMesh(preview, b.color_hex || 0x7ec8ff);
      mesh.visible = state.boneVisible[b.id] !== false;
      mesh.name = b.name;
      t.scene.add(mesh);
      state.boneMeshes[b.id] = mesh;
      box.expandByObject(mesh);
      any = true;
    } catch (e) {
      console.warn("bone mesh fail", b.id, e);
    }
  }
  if (!any) {
    $("#mesh-empty").classList.remove("hidden");
    $("#mesh-empty").innerHTML = "Bones listed but no mesh previews yet.";
    return;
  }
  state.hasMesh = true;
  const c = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(c);
  box.getSize(size);
  const r = Math.max(size.length() * 0.5, 1);
  const dist = r * 2.6;
  t.camera.near = Math.max(dist / 200, 0.01);
  t.camera.far = dist * 100;
  t.camera.position.set(c.x + dist * 0.7, c.y + dist * 0.45, c.z + dist * 0.7);
  t.camera.updateProjectionMatrix();
  t.controls.target.copy(c);
  t.controls.update();
  resizeThree();
  setStatus(`${Object.keys(state.boneMeshes).length} bone meshes loaded — toggle checkboxes`, "ok");
}

function previewToMesh(preview, colorHex) {
  const n = preview.points.length;
  const flat = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    flat[i * 3] = preview.points[i][0];
    flat[i * 3 + 1] = preview.points[i][1];
    flat[i * 3 + 2] = preview.points[i][2];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(flat, 3));
  if (preview.faces?.length) {
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(preview.faces), 1));
  }
  geo.computeVertexNormals();
  return new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({
      color: colorHex,
      metalness: 0.05,
      roughness: 0.45,
      side: THREE.DoubleSide,
    })
  );
}

async function loadCtVolume() {
  setStatus("Loading volume…", "busy");
  const res = await fetch(apiUrl("/api/session/ct-volume"));
  if (!res.ok) throw new Error("Could not load CT volume");
  const shape = res.headers.get("X-Shape").split(",").map(Number);
  const spacingHdr = res.headers.get("X-Spacing");
  state.ctSpacing = spacingHdr
    ? spacingHdr.split(",").map(Number)
    : [1, 1, 1];
  const buf = await res.arrayBuffer();
  state.ctArray = new Float32Array(buf);
  state.ctShape = shape;
  const n = shape[0] * shape[1] * shape[2];
  if (state.ctArray.length !== n) throw new Error(`Size mismatch ${state.ctArray.length} vs ${n}`);
  state.paint = new Uint8Array(n);
  const zmax = shape[0] - 1;
  $("#slice-z").max = String(zmax);
  $("#slice-z").value = String(Math.floor(zmax / 2));
  // auto window from data
  const p1 = percentile(state.ctArray, 1);
  const p99 = percentile(state.ctArray, 99);
  $("#wl").value = String(Math.round((p1 + p99) / 2));
  $("#ww").value = String(Math.max(200, Math.round(p99 - p1)));
  drawSlice();
  setStatus(`Viewer ready ${shape.join("×")}`, "ok");
}

function percentile(arr, p) {
  // sample for speed
  const step = Math.max(1, Math.floor(arr.length / 50000));
  const s = [];
  for (let i = 0; i < arr.length; i += step) s.push(arr[i]);
  s.sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.floor((p / 100) * s.length));
  return s[i];
}

// --- draw / paint ---
/**
 * Fit entire CT slice in the wrap: scale up/down uniformly (contain).
 * Never crop, never stretch. Black bars if needed.
 * Canvas element is sized to the image (not the full wrap).
 */
function drawSlice() {
  const canvas = $("#slice-canvas");
  if (!state.ctArray || !state.ctShape) return;
  const [nz, ny, nx] = state.ctShape;
  const z = Math.min(nz - 1, Math.max(0, parseInt($("#slice-z").value, 10) || 0));
  $("#slice-label").textContent = `slice ${z} / ${nz - 1}  ·  scroll wheel = next slice`;
  const ww = parseFloat($("#ww").value) || 2000;
  const wl = parseFloat($("#wl").value) || 300;
  const lo = wl - ww / 2;
  const hi = wl + ww / 2;

  const wrap = canvas.parentElement;
  const wrapRect = wrap.getBoundingClientRect();
  const availW = Math.max(1, Math.floor(wrapRect.width));
  const availH = Math.max(1, Math.floor(wrapRect.height));

  // Physical aspect (mm) — same scale X and Y
  const sp = state.ctSpacing || [1, 1, 1];
  const physW = nx * (sp[0] || 1);
  const physH = ny * (sp[1] || 1);
  // CONTAIN: largest scale where BOTH sides fit → full image visible
  const fit = Math.min(availW / physW, availH / physH);
  const dw = Math.max(1, Math.floor(physW * fit));
  const dh = Math.max(1, Math.floor(physH * fit));
  const left = Math.floor((availW - dw) / 2);
  const top = Math.floor((availH - dh) / 2);

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  // Canvas = exactly the image box (not the full pane) → cannot crop content
  canvas.width = Math.floor(dw * dpr);
  canvas.height = Math.floor(dh * dpr);
  canvas.style.position = "absolute";
  canvas.style.left = left + "px";
  canvas.style.top = top + "px";
  canvas.style.width = dw + "px";
  canvas.style.height = dh + "px";
  canvas.style.right = "auto";
  canvas.style.bottom = "auto";

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Rasterize slice at native voxel size then scale into dw×dh
  const img = ctx.createImageData(nx, ny);
  const base = z * ny * nx;
  const labels = state.boneLabels;
  const labelOk =
    labels &&
    state.boneLabelShape &&
    state.boneLabelShape[0] === nz &&
    state.boneLabelShape[1] === ny &&
    state.boneLabelShape[2] === nx;

  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const srcY = ny - 1 - y;
      const idx = base + srcY * nx + x;
      const v = state.ctArray[idx];
      let g = ((v - lo) / (hi - lo)) * 255;
      g = Math.max(0, Math.min(255, g));
      const i = (y * nx + x) * 4;
      const p = state.paint ? state.paint[idx] : 0;

      // Default greyscale CT
      let r = g;
      let gg = g;
      let b = g;

      // Bone label overlay — same colors as 3D meshes
      if (labelOk) {
        const lab = labels[idx];
        if (lab > 0 && state.boneVisible[lab] !== false) {
          const col = state.boneColorById[lab];
          if (col) {
            // Blend CT grey with bone color (~55% color, keep some anatomy)
            const t = 0.55;
            r = g * (1 - t) + col[0] * t;
            gg = g * (1 - t) + col[1] * t;
            b = g * (1 - t) + col[2] * t;
          }
        }
      }

      // Paint brush on top (include/exclude)
      if (p === 1) {
        r = Math.min(255, g * 0.35 + 40);
        gg = Math.min(255, g * 0.35 + 200);
        b = Math.min(255, g * 0.35 + 90);
      } else if (p === 2) {
        r = Math.min(255, g * 0.35 + 210);
        gg = Math.min(255, g * 0.35 + 50);
        b = Math.min(255, g * 0.35 + 50);
      }

      img.data[i] = r;
      img.data[i + 1] = gg;
      img.data[i + 2] = b;
      img.data[i + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = nx;
  off.height = ny;
  off.getContext("2d").putImageData(img, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  // Fill the whole canvas with the FULL image (uniform scale already baked into dw/dh)
  ctx.drawImage(off, 0, 0, nx, ny, 0, 0, dw, dh);

  const scaleX = dw / nx;
  const scaleY = dh / ny;
  state.view = {
    scale: (scaleX + scaleY) / 2,
    scaleX,
    scaleY,
    ox: 0,
    oy: 0,
    nx,
    ny,
    dw,
    dh,
    left,
    top,
  };
}

function canvasToVoxel(clientX, clientY) {
  const canvas = $("#slice-canvas");
  const rect = canvas.getBoundingClientRect();
  // Canvas is exactly the image — coords map directly
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const { scaleX, scaleY, nx, ny } = state.view;
  if (!scaleX || !scaleY) return null;
  // CSS size may differ slightly from dw/dh used when drawing
  const scaleXcss = rect.width / nx;
  const scaleYcss = rect.height / ny;
  const vx = Math.floor(x / scaleXcss);
  const vyDisp = Math.floor(y / scaleYcss);
  if (vx < 0 || vyDisp < 0 || vx >= nx || vyDisp >= ny) return null;
  return {
    x: vx,
    y: ny - 1 - vyDisp,
    z: parseInt($("#slice-z").value, 10) || 0,
  };
}

function paintAt(cx, cy) {
  if (!state.paint || !state.ctShape) return;
  const vox = canvasToVoxel(cx, cy);
  if (!vox) return;
  const [, ny, nx] = state.ctShape;
  const r = Math.max(1, Math.round((parseInt($("#brush-size").value, 10) || 16) / (state.view.scale || 1)));
  const val = state.tool === "include" ? 1 : state.tool === "exclude" ? 2 : 0;
  const r2 = r * r;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy > r2) continue;
      const x = vox.x + dx;
      const y = vox.y + dy;
      if (x < 0 || y < 0 || x >= nx || y >= ny) continue;
      state.paint[vox.z * ny * nx + y * nx + x] = val;
    }
  }
  drawSlice();
}

// --- Three.js ---
// BUG we hit: canvas was sized while #view-mesh had display:none → 0×0 WebGL → blank 3D.
let three = null;

function showMeshTab() {
  document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
  const tab = document.querySelector('.tab[data-view="mesh"]');
  if (tab) tab.classList.add("active");
  $("#view-slice").classList.add("hidden");
  $("#view-mesh").classList.remove("hidden");
}

function resizeThree() {
  if (!three) return;
  const canvas = $("#mesh-canvas");
  const p = canvas.parentElement;
  // Fit exactly in the visible canvas-wrap (full remaining screen)
  const w = Math.max(p.clientWidth || 0, 2);
  const h = Math.max(p.clientHeight || 0, 2);
  three.renderer.setPixelRatio(window.devicePixelRatio || 1);
  three.renderer.setSize(w, h, false);
  three.camera.aspect = w / Math.max(h, 1);
  three.camera.updateProjectionMatrix();
  three.renderer.render(three.scene, three.camera);
}

function ensureThree() {
  if (three) {
    resizeThree();
    return three;
  }
  const canvas = $("#mesh-canvas");
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x070a0e, 1);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070a0e);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1e7);
  camera.position.set(80, 60, 80);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  // Lights
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(1, 1.4, 0.9);
  scene.add(dir);
  const dir2 = new THREE.DirectionalLight(0xaaccff, 0.35);
  dir2.position.set(-1, -0.5, -1);
  scene.add(dir2);
  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  three = { renderer, scene, camera, controls, mesh: null };
  window.addEventListener("resize", resizeThree);
  resizeThree();
  (function tick() {
    requestAnimationFrame(tick);
    if (three) {
      three.controls.update();
      three.renderer.render(three.scene, three.camera);
    }
  })();
  return three;
}

function showSurface(preview) {
  // 1) Make mesh pane visible BEFORE measuring canvas
  showMeshTab();
  $("#mesh-empty").classList.add("hidden");

  if (!preview?.points?.length) {
    $("#mesh-empty").classList.remove("hidden");
    $("#mesh-empty").innerHTML =
      "No surface points returned.<br/>See Debug log for why.";
    state.hasMesh = false;
    setDebug("showSurface: empty points", "err");
    return;
  }

  // 2) Wait a frame so layout is non-zero, then build mesh
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      try {
        const t = ensureThree();
        resizeThree();

        if (t.mesh) {
          t.scene.remove(t.mesh);
          t.mesh.geometry.dispose();
          if (Array.isArray(t.mesh.material)) t.mesh.material.forEach((m) => m.dispose());
          else t.mesh.material.dispose();
          t.mesh = null;
        }

        const n = preview.points.length;
        const flat = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {
          const p = preview.points[i];
          flat[i * 3] = p[0];
          flat[i * 3 + 1] = p[1];
          flat[i * 3 + 2] = p[2];
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(flat, 3));
        if (preview.faces?.length) {
          // Three r152+ prefers typed array indices
          const idx = preview.faces.length > 65535
            ? new Uint32Array(preview.faces)
            : new Uint32Array(preview.faces);
          geo.setIndex(new THREE.BufferAttribute(idx, 1));
        }
        geo.computeVertexNormals();
        geo.computeBoundingSphere();
        geo.computeBoundingBox();

        const mat = new THREE.MeshStandardMaterial({
          color: 0x7ec8ff,
          metalness: 0.05,
          roughness: 0.45,
          side: THREE.DoubleSide,
          flatShading: false,
        });
        t.mesh = new THREE.Mesh(geo, mat);
        t.scene.add(t.mesh);

        const sphere = geo.boundingSphere;
        const c = sphere ? sphere.center.clone() : new THREE.Vector3();
        const r = Math.max(sphere?.radius || 1, 1);
        // Fit camera to bounds (works for large DICOM Z offsets like z≈1600)
        const dist = r * 2.6;
        t.camera.near = Math.max(dist / 200, 0.01);
        t.camera.far = dist * 100;
        t.camera.position.set(c.x + dist * 0.7, c.y + dist * 0.45, c.z + dist * 0.7);
        t.camera.updateProjectionMatrix();
        t.controls.target.copy(c);
        t.controls.update();
        resizeThree();
        t.renderer.render(t.scene, t.camera);

        state.hasMesh = true;
        setStatus(
          `3D OK: ${preview.n_points || n} pts, ${preview.n_faces || (preview.faces.length / 3) | 0} faces · drag to orbit`,
          "ok"
        );
        setDebug(
          (document.getElementById("debug")?.textContent || "") +
            `\n\n=== 3D VIEWER ===\npoints=${n}\nfaces=${preview.faces?.length || 0}\n` +
            `center=[${c.x.toFixed(1)},${c.y.toFixed(1)},${c.z.toFixed(1)}]\n` +
            `radius=${r.toFixed(2)}\ncanvas=${$("#mesh-canvas").width}x${$("#mesh-canvas").height}`,
          "ok"
        );
      } catch (e) {
        console.error(e);
        state.hasMesh = false;
        $("#mesh-empty").classList.remove("hidden");
        $("#mesh-empty").innerHTML =
          `<div style="color:#f07178"><strong>3D render error</strong><br/>${escapeHtml(e.message)}</div>`;
        setStatus("3D render error: " + e.message, "error");
        setDebug("3D render exception:\n" + e.stack, "err");
      }
    });
  });
}

// --- progress UI ---
function showProgress(show) {
  $("#progress-wrap").classList.toggle("hidden", !show);
}
function setProgress(pct, message, step) {
  $("#progress-fill").style.width = `${Math.max(0, Math.min(100, pct))}%`;
  $("#progress-label").textContent = `${pct}% · ${message || ""}`;
  const order = ["load", "mask", "surface", "volume", "material", "done"];
  const idx = order.indexOf(step);
  document.querySelectorAll("#progress-steps li").forEach((li) => {
    const s = li.dataset.step;
    li.classList.remove("active", "done", "err");
    const si = order.indexOf(s);
    if (step === "error") {
      if (s === "done") li.classList.add("err");
    } else if (si >= 0 && idx >= 0) {
      if (si < idx || step === "done") li.classList.add("done");
      if (si === idx && step !== "done") li.classList.add("active");
    }
  });
}

async function loadPreviewFromServer() {
  try {
    setStatus("Loading 3D preview…", "busy");
    const preview = await api("/api/session/preview");
    if (!preview?.points?.length) {
      throw new Error("Preview has 0 points");
    }
    showSurface(preview);
    setStatus(`3D loaded (${preview.n_points} pts, ${preview.n_faces} faces)`, "ok");
    return true;
  } catch (e) {
    setStatus("No 3D preview: " + e.message, "error");
    $("#mesh-empty").classList.remove("hidden");
    $("#mesh-empty").innerHTML =
      `<div style="color:#f07178;max-width:28rem"><strong>Could not load 3D</strong><br/>${escapeHtml(e.message)}<br/><br/>Click <em>Refresh debug</em>.</div>`;
    return false;
  }
}

async function refreshServerDebug() {
  try {
    const d = await api("/api/session/debug");
    const lines = [
      "=== SERVER DEBUG ===",
      `session: ${d.session}`,
      `status: ${JSON.stringify(d.status)}`,
      `last_error: ${d.last_error}`,
      `files: ${Object.keys(d.files || {}).join(", ") || "(none)"}`,
      "",
      "--- outputs on disk ---",
      ...Object.entries(d.outputs || {}).map(([k, v]) => `  ${k}: ${v.mb} MB`),
      "",
      `mask_fg_voxels: ${d.mask_fg_voxels ?? "n/a"} (${d.mask_fg_pct ?? "?"}%)`,
      `mask_shape: ${JSON.stringify(d.mask_shape)}`,
      `preview: ${JSON.stringify(d.preview)}`,
      `has_surface_file: ${d.has_surface_file}`,
      `has_volume_file: ${d.has_volume_file}`,
      `has_preview_file: ${d.has_preview_file}`,
    ];
    if (d.mask_error) lines.push("mask_error: " + d.mask_error);
    if (d.preview_error) lines.push("preview_error: " + d.preview_error);
    setDebug(lines.join("\n"), d.has_preview_file ? "ok" : "err");
    return d;
  } catch (e) {
    setDebug("debug fetch failed: " + e.message, "err");
    return null;
  }
}

function finishWithResult(result) {
  const lines = [];
  lines.push(result.ok ? "✓ Pipeline finished" : "✗ Pipeline failed");
  if (result.error) lines.push("ERROR: " + result.error);
  if (result.partial) lines.push("(partial — surface may still be available)");
  if (result.log) lines.push("", "--- log ---", ...result.log);
  if (result.steps?.mask) lines.push("", "--- mask ---", JSON.stringify(result.steps.mask, null, 2));
  if (result.steps?.surface) lines.push("", "--- surface ---", JSON.stringify(result.steps.surface, null, 2));
  if (result.steps?.volume) lines.push("", "--- volume ---", JSON.stringify(result.steps.volume, null, 2));
  if (result.steps?.material) lines.push("", "--- material ---", JSON.stringify(result.steps.material, null, 2));
  lines.push("", "Fetching server debug + 3D preview…");
  setDebug(lines.join("\n"), result.ok ? "ok" : "err");

  const box = $("#downloads");
  box.innerHTML = "";
  for (const [k, rel] of Object.entries(result.downloads || {})) {
    const a = document.createElement("a");
    a.href = apiUrl(`/api/session/files/${rel}`);
    a.textContent = `↓ ${k}`;
    a.download = "";
    box.appendChild(a);
  }

  // Always try separate preview endpoint (not embedded in job JSON)
  const try3d = result.ok || result.has_surface || result.partial;
  if (try3d) {
    loadPreviewFromServer().then(async (ok) => {
      await refreshServerDebug();
      if (!ok && !result.ok) {
        setStatus(result.error || "Failed", "error");
      } else if (ok) {
        setStatus(
          result.has_volume ? "3D model ready — drag to rotate" : "Surface ready",
          "ok"
        );
      }
    });
  } else {
    refreshServerDebug();
    $("#mesh-empty").classList.remove("hidden");
    $("#mesh-empty").innerHTML =
      `<div style="color:#f07178;max-width:28rem">` +
      `<strong>No 3D model generated</strong><br/><br/>` +
      `${escapeHtml(result.error || "Unknown error")}<br/><br/>` +
      `<span style="color:#8b9bb0;font-size:0.9em">See Debug log. Try grow radius 10–15mm.</span>` +
      `</div>`;
    document.querySelector('.tab[data-view="mesh"]').click();
    setStatus(result.error || "Failed", "error");
  }
}

// --- generate ---
async function generate() {
  if (!state.ctShape) {
    setStatus("Drop a CT first", "error");
    setDebug({ error: "No CT loaded" }, "err");
    return;
  }
  const btn = $("#btn-generate");
  btn.disabled = true;
  showProgress(true);
  setProgress(1, "Starting job…", "load");
  setStatus("Generating… watch progress below", "busy");
  setDebug("Job started — progress updates every 0.5s.\nVolume meshing is often the slow part (1–2 min).");
  $("#mesh-empty").classList.add("hidden");

  try {
    const [z, y, x] = state.ctShape;
    const fd = new FormData();
    const nPaint = state.paint ? state.paint.reduce((a, v) => a + (v === 1 ? 1 : 0), 0) : 0;
    if (state.paint && nPaint > 0) {
      fd.append("paint", new Blob([state.paint]), "paint.raw");
      fd.append("shape_z", String(z));
      fd.append("shape_y", String(y));
      fd.append("shape_x", String(x));
    }
    fd.append("bone_hu_min", $("#hu-min").value || "200");
    fd.append("cell_size", $("#cell-size").value || "4");
    fd.append("quadratic", "true");
    fd.append("grow_mode", $("#grow-mode")?.value || "local");
    fd.append("grow_radius_mm", $("#grow-radius")?.value || "6");
    // Selected bones from checkboxes
    const selected = state.bones
      .filter((b) => state.boneVisible[b.id] !== false)
      .map((b) => b.id);
    if (selected.length) {
      fd.append("selected_bones", selected.join(","));
    }

    const start = await api("/api/session/generate", { method: "POST", body: fd });
    if (!start.job_id) throw new Error(start.error || "No job_id returned — hard-refresh the page");
    const jobId = start.job_id;
    setProgress(3, `Job ${jobId} running…`, "load");

    // poll until done
    let finished = false;
    while (!finished) {
      await new Promise((r) => setTimeout(r, 500));
      const job = await api(`/api/session/jobs/${jobId}`);
      setProgress(job.percent || 0, `${job.message} (${job.elapsed_s}s)`, job.step);
      if (job.log?.length) {
        setDebug(
          `Progress (${job.elapsed_s}s)\n` + job.log.join("\n"),
          job.state === "error" ? "err" : ""
        );
      }
      if (job.state === "done" || job.state === "error") {
        finished = true;
        if (job.result) {
          finishWithResult(job.result);
        } else {
          setStatus(job.error || "Failed", "error");
          setDebug(job.error || "No result", "err");
        }
        if (job.state === "done") setProgress(100, "Complete", "done");
        else setProgress(100, job.error || "Failed", "error");
      }
    }
  } catch (e) {
    setStatus(e.message, "error");
    setDebug("Exception:\n" + e.message, "err");
    setProgress(100, e.message, "error");
    $("#mesh-empty").classList.remove("hidden");
    $("#mesh-empty").innerHTML =
      `<div style="color:#f07178"><strong>Request failed</strong><br/>${escapeHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// --- wire UI ---
const fileInput = $("#file-ct");
const drop = $("#dropzone");

fileInput.addEventListener("change", (e) => {
  loadCtFile(e.target.files?.[0]).catch((err) => {
    setStatus(err.message, "error");
    setDebug(err.message, "err");
  });
});

["dragenter", "dragover"].forEach((ev) => {
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((ev) => {
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove("drag");
  });
});
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) loadCtFile(f).catch((err) => setStatus(err.message, "error"));
});

document.querySelectorAll(".brush-tools .tool").forEach((b) => {
  b.addEventListener("click", () => {
    state.tool = b.dataset.tool;
    document.querySelectorAll(".brush-tools .tool").forEach((t) => t.classList.remove("active"));
    b.classList.add("active");
  });
});

$("#btn-clear-paint").addEventListener("click", () => {
  if (state.paint) state.paint.fill(0);
  drawSlice();
});

$("#btn-generate").addEventListener("click", () => generate());

$("#slice-z").addEventListener("input", drawSlice);
$("#ww").addEventListener("change", drawSlice);
$("#wl").addEventListener("change", drawSlice);
window.addEventListener("resize", drawSlice);

const canvas = $("#slice-canvas");
canvas.addEventListener("pointerdown", (e) => {
  state.painting = true;
  canvas.setPointerCapture(e.pointerId);
  paintAt(e.clientX, e.clientY);
});
canvas.addEventListener("pointermove", (e) => {
  if (state.painting) paintAt(e.clientX, e.clientY);
});
canvas.addEventListener("pointerup", () => {
  state.painting = false;
});
canvas.addEventListener(
  "wheel",
  (e) => {
    if (!state.ctShape) return;
    e.preventDefault();
    const sl = $("#slice-z");
    let z = parseInt(sl.value, 10) || 0;
    z += e.deltaY > 0 ? 1 : -1;
    z = Math.max(0, Math.min(state.ctShape[0] - 1, z));
    sl.value = String(z);
    drawSlice();
  },
  { passive: false }
);

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const v = tab.dataset.view;
    $("#view-slice").classList.toggle("hidden", v !== "slice");
    $("#view-mesh").classList.toggle("hidden", v !== "mesh");
    if (v === "mesh") {
      // Critical: resize AFTER un-hiding, else canvas is 0×0
      requestAnimationFrame(() => {
        ensureThree();
        resizeThree();
        if (!state.hasMesh) {
          $("#mesh-empty").classList.remove("hidden");
          // Auto-try loading last preview so user isn't stuck
          loadPreviewFromServer().catch(() => {});
        }
      });
    }
    if (v === "slice") drawSlice();
  });
});

$("#btn-refresh-debug")?.addEventListener("click", () => {
  refreshServerDebug().catch((e) => setStatus(e.message, "error"));
});
$("#btn-load-3d")?.addEventListener("click", () => {
  if (state.bones.length) {
    loadAllBoneMeshes().catch((e) => setStatus(e.message, "error"));
  } else {
    loadPreviewFromServer().catch((e) => setStatus(e.message, "error"));
  }
});
$("#btn-redetect")?.addEventListener("click", async () => {
  try {
    const fd = new FormData();
    fd.append("bone_hu_min", $("#hu-min").value || "200");
    fd.append("min_voxels", "1000");
    fd.append("open_radius", "1");
    setStatus("Re-detecting bones…", "busy");
    const r = await api("/api/session/bones/redetect", { method: "POST", body: fd });
    await pollBoneSplitJob(r.job_id);
  } catch (e) {
    setStatus(e.message, "error");
  }
});

wireEngineSetupUi();

probeEngine()
  .then((h) => onEngineOnline(h, { first: true }))
  .catch(() => onEngineOffline({ showSetup: !!API_BASE }));
