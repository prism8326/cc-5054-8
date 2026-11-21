// 实验台前端：报告加载、三态帧切换、Canvas 图表与分段时间线。
let REPORT = null;
let playTimer = null;

const $ = (id) => document.getElementById(id);
const els = ["group", "route", "layer", "frame", "frameLabel", "play",
  "frameImg", "frameTag"];
const E = {};
els.forEach((k) => (E[k] = $(k)));

async function init() {
  const res = await fetch("/api/report");
  if (!res.ok) {            // 仅有 grainlab 数据时隐藏主实验面板
    document.body.classList.add("no-report");
    return;
  }
  REPORT = await res.json();
  const n = REPORT.config.frames;
  E.frame.max = n - 1;
  ["group", "route", "layer", "frame"].forEach((k) =>
    E[k].addEventListener("change", renderAll));
  E.play.addEventListener("click", togglePlay);
  renderAll();
}

function cur() {
  const group = E.group.value;
  const route = E.route.value;
  return {
    group, route,
    metrics: REPORT.results[group].metrics,
    separator: REPORT.separator[group],
    frame: parseInt(E.frame.value, 10),
  };
}

function togglePlay() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    E.play.textContent = "播放";
    return;
  }
  E.play.textContent = "暂停";
  playTimer = setInterval(() => {
    const max = parseInt(E.frame.max, 10);
    E.frame.value = (parseInt(E.frame.value, 10) + 1) % (max + 1);
    renderAll();
  }, 250);
}

function renderAll() {
  const c = cur();
  E.frameLabel.textContent = c.frame;
  renderFrame(c);
  renderSummary(c);
  renderTrail(c);
  renderBands(c);
  renderContrast(c);
  renderSeparator(c);
  renderTimeline(c);
}

function renderFrame(c) {
  const f = REPORT.sample_frames.reduce((a, b) =>
    Math.abs(b - c.frame) < Math.abs(a - c.frame) ? b : a);
  const layer = E.layer.value;
  let prefix, tag;
  if (layer === "source") {
    prefix = "source"; tag = "基准原片（含全部颗粒/微纹）";
  } else if (layer === "clean") {
    prefix = "clean"; tag = "无颗粒底图（aomdec --skip-film-grain 之外的参照真值）";
  } else {
    prefix = c.route;
    tag = `解码器合成输出 · ${c.route}` +
      (c.route === "model" ? "（AV1 film grain 元数据合成）" : "");
  }
  // clean/source 两组三态共用生成器产物；解码帧按路由各自存放在同名前缀。
  E.frameImg.src = `/frames/${c.group}/${prefix}_f${String(f).padStart(3, "0")}.png`;
  E.frameTag.textContent = `${tag} · 帧 ${f}`;
}

function renderSummary(c) {
  const tbody = document.querySelector("#summaryTable tbody");
  const rows = ["direct", "model", "hybrid"].map((r) => {
    const m = c.metrics[r];
    const g = m.with_grain;
    return `<tr><td>${r}</td><td>${m.bytes}</td>
      <td>${g.mean_psnr.toFixed(2)}</td>
      <td>${g.mean_high_band.toFixed(3)}</td>
      <td>${g.mean_texture_contrast.toFixed(2)}</td>
      <td>${g.trail_object_corr.toFixed(3)}</td>
      <td>${g.trail_screen_corr.toFixed(3)}</td></tr>`;
  });
  tbody.innerHTML = rows.join("");

  const parity = REPORT.luma_parity;
  const meanDelta = avg(parity.map((p) => p.mean_delta));
  const aMean = avg(parity.map((p) => p.mean_adherent));
  const sMean = avg(parity.map((p) => p.mean_screen));
  const aStd = avg(parity.map((p) => p.std_adherent));
  const sStd = avg(parity.map((p) => p.std_screen));
  $("parity").innerHTML =
    `<span class="pill">两组逐帧亮度均值差: ${meanDelta.toFixed(3)} 级</span>` +
    `<span class="pill">adherent 均值/标准差: ${aMean.toFixed(1)} / ${aStd.toFixed(1)}</span>` +
    `<span class="pill">screen 均值/标准差: ${sMean.toFixed(1)} / ${sStd.toFixed(1)}</span>` +
    `<span class="pill">去噪底图对真值 PSNR: ${c.metrics.reference.denoiser_psnr_to_true_clean.toFixed(2)} dB</span>`;
}

function renderTrail(c) {
  const cv = $("trailChart"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const ref = c.metrics.reference;
  const series = [
    { label: "原片 · 物体坐标", v: ref.source_trail_object_corr, color: "#6db1ff" },
    { label: "原片 · 屏幕坐标", v: ref.source_trail_screen_corr, color: "#93a0b0" },
  ];
  ["direct", "model", "hybrid"].forEach((r) => {
    const g = c.metrics[r].with_grain;
    series.push({ label: `${r} 物体`, v: g.trail_object_corr, color: routeColor(r, 1) });
    series.push({ label: `${r} 屏幕`, v: g.trail_screen_corr, color: routeColor(r, 0.6) });
  });
  drawBarChart(ctx, cv, series, [-0.6, 1.0]);
}

function renderBands(c) {
  const cv = $("bandChart"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const series = [];
  ["direct", "model", "hybrid"].forEach((r) => {
    const g = c.metrics[r].with_grain;
    series.push({ label: `${r} 低频`, v: g.mean_low_band, color: routeColor(r, 1) });
    series.push({ label: `${r} 高频`, v: g.mean_high_band, color: routeColor(r, 0.55) });
  });
  drawBarChart(ctx, cv, series, [0, 1]);
}

function renderContrast(c) {
  const cv = $("contrastChart"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const colors = { direct: "#ef7d7d", model: "#5fd38d", hybrid: "#f0b35e" };
  Object.entries(colors).forEach(([r, color], i) => {
    const data = c.metrics[r].with_grain.per_frame_contrast;
    drawLine(ctx, cv, data, color, i);
  });
  drawAxes(ctx, cv, "帧", "对比度");
  ctx.fillStyle = "#93a0b0";
  ctx.font = "11px sans-serif";
  Object.entries(colors).forEach(([r, color], i) => {
    ctx.fillStyle = color;
    ctx.fillText(r, 12 + i * 70, 16);
  });
}

function renderSeparator(c) {
  const box = $("separator");
  box.innerHTML = c.separator.map((seg, i) => {
    const maxStr = Math.max(...seg.luma_bins.map((b) => b.strength), 1);
    const bins = seg.luma_bins.map((b) => `
      <div class="bin-row">
        <span>Y ${b.lo}-${b.hi}</span>
        <div class="bar" style="width:${(b.strength / maxStr * 100).toFixed(0)}%"></div>
        <span>σ=${b.strength.toFixed(1)}</span>
        <span>ρt=${b.temporal_corr.toFixed(2)}</span>
      </div>`).join("");
    return `<div style="margin-bottom:10px">
      <strong>段 ${i} · 帧 ${seg.start}-${seg.end - 1}</strong>
      <span class="pill" style="color:${seg.model_feasible ? "#5fd38d" : "#ef7d7d"}">
        ${seg.model_feasible ? "模型可行" : "需直接编码"}</span>
      <span class="pill">时间相干 ${seg.temporal_coherence.toFixed(2)}</span>
      <span class="pill">拟合误差 ${seg.model_rms_error.toFixed(2)}</span>
      ${bins}</div>`;
  }).join("");
}

function renderTimeline(c) {
  const map = c.metrics.route_change_map;
  const tl = $("timeline");
  tl.innerHTML = map.segments.map((s) => {
    const d = c.metrics.route_change_map;
    const decision = REPORT.results[c.group].manifest.decisions[s.index];
    return `<div class="seg ${decision.route}" data-i="${s.index}">
      段${s.index}<br>帧${s.frames[0]}-${s.frames[1] - 1}<br>${decision.route}</div>`;
  }).join("");
  tl.querySelectorAll(".seg").forEach((node) => {
    node.addEventListener("click", () => {
      const i = parseInt(node.dataset.i, 10);
      const d = REPORT.results[c.group].manifest.decisions[i];
      $("decisionDetail").textContent =
        `段 ${i}（帧 ${d.start}-${d.end - 1}）→ ${d.route}\n` +
        `依据: ${d.reason}\n` +
        `模型误差=${d.model_error.toFixed(3)}  ` +
        `直接=${d.direct_bytes}B 模型=${d.model_bytes}B 预算=${d.budget_bytes}B` +
        (d.error_bound !== null ? `\n带界近似误差上界: ${d.error_bound.toFixed(3)}` : "");
    });
  });
}

function drawBarChart(ctx, cv, series, [ymin, ymax]) {
  const padL = 44, padB = 60, padT = 10, padR = 10;
  const w = cv.width - padL - padR, h = cv.height - padB - padT;
  ctx.strokeStyle = "#2c343f";
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + h);
  ctx.lineTo(padL + w, padT + h); ctx.stroke();
  const bw = w / series.length * 0.72;
  series.forEach((s, i) => {
    const x = padL + (i + 0.5) * (w / series.length) - bw / 2;
    const norm = (s.v - ymin) / (ymax - ymin);
    const bh = Math.max(2, norm * h);
    ctx.fillStyle = s.color;
    ctx.fillRect(x, padT + h - bh, bw, bh);
    ctx.save();
    ctx.translate(x + bw / 2, padT + h + 8);
    ctx.rotate(-Math.PI / 5);
    ctx.fillStyle = "#93a0b0"; ctx.font = "10px sans-serif";
    ctx.fillText(s.label, 0, 0);
    ctx.restore();
    ctx.fillStyle = "#e6e9ee"; ctx.font = "10px sans-serif";
    ctx.fillText(s.v.toFixed(2), x, padT + h - bh - 3);
  });
  ctx.fillStyle = "#93a0b0"; ctx.font = "10px sans-serif";
  [0, 0.5, 1].forEach((t) => {
    const y = padT + h - t * h;
    ctx.fillText((ymin + t * (ymax - ymin)).toFixed(1), 6, y + 3);
  });
}

function drawLine(ctx, cv, data, color, offset) {
  const padL = 40, padB = 28, padT = 10, padR = 10;
  const w = cv.width - padL - padR, h = cv.height - padB - padT;
  const max = Math.max(...data, 1);
  ctx.strokeStyle = color; ctx.lineWidth = 1.6;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = padL + (i / (data.length - 1)) * w;
    const y = padT + h - (v / max) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawAxes(ctx, cv, xl, yl) {
  ctx.strokeStyle = "#2c343f";
  ctx.beginPath();
  ctx.moveTo(40, 10); ctx.lineTo(40, cv.height - 28);
  ctx.lineTo(cv.width - 10, cv.height - 28); ctx.stroke();
  ctx.fillStyle = "#93a0b0"; ctx.font = "10px sans-serif";
  ctx.fillText(xl, cv.width - 24, cv.height - 12);
}

function routeColor(r, alpha) {
  const map = { direct: [239, 125, 125], model: [95, 211, 141], hybrid: [240, 179, 94] };
  const [rr, gg, bb] = map[r];
  return `rgba(${rr},${gg},${bb},${alpha})`;
}
function avg(a) { return a.reduce((x, y) => x + y, 0) / a.length; }

init();
