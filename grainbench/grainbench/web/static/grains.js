// 颗粒状态实验界面：逐帧状态浏览、两种访问模型对照、任意帧独立重演。
let GL = null;
let glTimer = null;
const G = {};
["glStream", "glLayer", "glFrame", "glFrameLabel", "glPlay", "glImg",
 "glTag", "glState", "glReplay", "glTimeline", "glAccess", "glSwitch",
 "glTwin", "glDouble", "glScope"].forEach((k) => (G[k] = $(k)));

async function initGrainlab() {
  let res;
  try { res = await fetch("/api/grainlab"); } catch (e) { return; }
  if (!res.ok) return;
  GL = await res.json();
  $("grainlab").hidden = false;
  const n = GL.frame_counts.frames;
  G.glFrame.max = n - 1;
  G.glScope.textContent = GL.determinism_scope;
  ["glStream", "glLayer", "glFrame"].forEach((k) =>
    G[k].addEventListener("change", renderGrainlab));
  G.glPlay.addEventListener("click", toggleGlPlay);
  renderGrainlab();
}

function glState() {
  const stream = G.glStream.value;
  const info = GL.streams[stream];
  const frame = parseInt(G.glFrame.value, 10);
  return { stream, info, frame, layer: G.glLayer.value };
}

function toggleGlPlay() {
  if (glTimer) {
    clearInterval(glTimer); glTimer = null; G.glPlay.textContent = "播放"; return;
  }
  G.glPlay.textContent = "暂停";
  glTimer = setInterval(() => {
    const max = parseInt(G.glFrame.max, 10);
    G.glFrame.value = (parseInt(G.glFrame.value, 10) + 1) % (max + 1);
    renderGrainlab();
  }, 320);
}

function renderGrainlab() {
  const c = glState();
  G.glFrameLabel.textContent = c.frame;
  renderGlFrame(c);
  renderGlState(c);
  renderGlReplay(c);
  renderGlTimeline(c);
  renderGlAccess(c);
  renderGlSwitch(c);
  renderGlTwin(c);
  renderGlDouble(c);
}

function renderGlFrame(c) {
  const f = String(c.frame).padStart(3, "0");
  G.glImg.src = `/gl/${c.stream}/f${f}_${c.layer}.png`;
  const tags = { replay: "状态缓存重演", grain: "解码器含颗粒", base: "无颗粒底图" };
  G.glTag.textContent = `${c.stream} · ${tags[c.layer]} · 帧 ${c.frame}`;
}

async function renderGlState(c) {
  // 逐帧状态来自码流解析，随重演结论一并保存在摘要中。
  const replayRow = (c.info.replay || []).find((r) => r.frame === c.frame);
  let html = "";
  if (replayRow) {
    html += `<span class="pill">来源 ${replayRow.source}</span>`;
    html += `<span class="pill">种子 ${replayRow.seed ?? "—"}</span>`;
    html += `<span class="pill">${replayRow.exact ? "<span class='ok'>重演逐像素一致</span>" : "<span class='bad'>重演有差异</span>"}</span>`;
  }
  const cut = GL.scene_cut_display;
  if (c.frame === cut) html += `<span class="pill">切镜点</span>`;
  if ((c.info.keyframes || []).includes(c.frame)) html += `<span class="pill">关键帧</span>`;
  const se = (c.info.show_existing || []).find((s) => s.frame === c.frame);
  if (se) html += `<span class="pill">show_existing → 槽 ${se.slot}</span>`;
  G.glState.innerHTML = html +
    `<div class="hint">该帧状态存于 grainlab/states/${c.stream}.json；` +
    `点击时间线任意帧可独立重演。</div>`;
}

function renderGlReplay(c) {
  const rows = c.info.replay || [];
  if (!rows.length) {
    G.glReplay.innerHTML = `<div class="hint">该流由同一状态缓存重演；见 grain_on。</div>`;
    return;
  }
  const exact = rows.filter((r) => r.exact).length;
  const worst = rows.reduce((a, b) => (b.maxdiff > a.maxdiff ? b : a), rows[0]);
  G.glReplay.innerHTML =
    `<span class="pill">${exact}/${rows.length} 帧逐像素一致</span>` +
    `<span class="pill">最大差 ${worst.maxdiff}（帧 ${worst.frame}）</span>` +
    `<div class="hint">用码流解析出的逐帧状态（种子+参数）对干净底图重演，和 aomdec 含颗粒输出逐像素比对。</div>`;
}

function renderGlTimeline(c) {
  const key = new Set(c.info.keyframes || []);
  const se = new Set((c.info.show_existing || []).map((s) => s.frame));
  const drift = new Set(GL.access.per_frame
    .filter((r) => r.cache_vs_restart_maxdiff > 0).map((r) => r.frame));
  const n = GL.frame_counts.frames;
  const cut = GL.scene_cut_display;
  G.glTimeline.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const cell = document.createElement("div");
    cell.className = "gl-cell";
    if (key.has(i)) cell.classList.add("key");
    if (se.has(i)) cell.classList.add("se");
    if (drift.has(i)) cell.classList.add("drift");
    if (i === cut) cell.classList.add("cut");
    cell.title = `帧 ${i}` + (drift.has(i) ? " · 序号推算会分叉" : "");
    cell.addEventListener("click", () => {
      G.glFrame.value = i; renderGrainlab();
    });
    G.glTimeline.appendChild(cell);
  }
}

function renderGlAccess() {
  const a = GL.access.summary;
  G.glAccess.innerHTML =
    `<span class="pill">缓存对齐解码器 ${a.cache_matches_decoder[0]}/${a.cache_matches_decoder[1]}</span>` +
    `<span class="pill bad">序号推算分叉 ${a.cache_vs_naive_diverged}/${a.frames} 帧</span>` +
    `<span class="pill">最大差 ${a.max_cache_vs_naive} 灰阶</span>` +
    `<div class="hint">逐帧缓存 = 从码流解析的真实种子/参数（含隐藏帧推进、show_existing 复用）；` +
    `序号推算 = 7391 + 3381×显示序号，隐藏帧处开始错位。点击时间线上的红格查看。</div>`;
}

function renderGlSwitch() {
  const s = GL.grain_switch;
  const bo = s.base_identical_on_vs_off;
  G.glSwitch.innerHTML =
    `<span class="pill ${bo ? "ok" : "bad"}">颗粒开/关底图${bo ? "完全相同" : "不同"}</span>` +
    `<span class="pill">空间方差 开 ${avgN(s.spatial_variance_on).toFixed(2)} / 关 ${avgN(s.spatial_variance_off).toFixed(2)}</span>` +
    `<span class="pill">帧间相关 开 ${avgN(s.interframe_corr_on).toFixed(2)} / 关 ${avgN(s.interframe_corr_off).toFixed(2)}</span>` +
    `<div class="hint">平场段空间方差为 0；颗粒开启后帧间相关下降，因颗粒逐帧独立。</div>`;
}

function renderGlTwin() {
  const t = GL.twin;
  G.glTwin.innerHTML =
    `<span class="pill ${t.base_identical ? "ok" : "bad"}">底图一致</span>` +
    `<span class="pill">切镜后强度 P=${fmt(t.grain_std_segB_P)} Q=${fmt(t.grain_std_segB_Q)}</span>` +
    `<span class="pill">切镜后像素差均值 ${fmt(t.pixel_diff_at_cut_mean)}</span>` +
    `<div class="hint">两片视觉强度相同（约 2.37 灰阶），切镜后参数不同：P 水平 AR、无 overlap；Q 垂直 AR、有 overlap。` +
    `底图完全相同，故像素差全部来自颗粒参数。</div>`;
}

function renderGlDouble() {
  const d = GL.double_apply;
  G.glDouble.innerHTML =
    `<span class="pill">正确施加 σ=${fmt(d.grain_std_correct)}</span>` +
    `<span class="pill bad">重复施加 σ=${fmt(d.grain_std_doubled)}</span>` +
    `<span class="pill">对单次最大差 ${d.double_vs_single_maxdiff}</span>` +
    `<div class="hint">颗粒必须恰好一次作用在干净底图上；对已含颗粒的画面再施加一次会使其幅度翻倍（帧 ${d.frame}）。</div>`;
}

function avgN(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
function fmt(v) { return v == null ? "—" : Number(v).toFixed(2); }

initGrainlab();
