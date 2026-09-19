const $ = (id) => document.getElementById(id);
let appState = null;
let playing = false;
let stepping = false;
let plotPoints = [];
let plotScreenPoints = [];
let tokenSource = "card";
let tokenPlotPoints = [];
let tokenScreenPoints = [];
let distributionBars = [];
let miniMapScreenNodes = [];
let uploadManuallyOpen = false;
let watchSeedDirty = false;
let watchSimulationsDirty = false;
let watchDepthDirty = false;
let watchLeafBatchSizeDirty = false;
let watchPriorTemperatureDirty = false;
let watchSearchModeDirty = false;
let watchGumbelParamsDirty = false;
let watchPaperModeDirty = false;
let watchPolicyModeDirty = false;
let watchUpperStrategyDirty = false;
let watchCombatStrategyDirty = false;
let watchPrecomputeDirty = false;
let evalSimulationsDirty = false;
let evalDepthDirty = false;
let evalLeafBatchSizeDirty = false;
let evalPriorTemperatureDirty = false;
let evalSearchModeDirty = false;
let evalGumbelParamsDirty = false;
let evalPaperModeDirty = false;
let evalUpperStrategyDirty = false;
let evalCombatStrategyDirty = false;
let seeking = false;
let selectedMapAct = null;
let mapPreviewing = false;
let previewedMapKeys = new Set();
let rewardChoicesCollapsed = false;
let searchTreeCollapsed = false;
let searchTreeKey = "";
let searchTreeCache = new Map();
let searchTreeOpen = new Set();
let searchTreeLoading = new Set();
let searchTreeSearchMode = "puct";

const zhCards = {
  Strike_R: "打击", Defend_R: "防御", Bash: "痛击", Anger: "愤怒",
  Armaments: "武装", "Body Slam": "全身撞击", Clash: "交锋", Cleave: "顺劈斩",
  Clothesline: "金刚臂", Flex: "活动肌肉", Havoc: "破灭", Headbutt: "头槌",
  "Heavy Blade": "重刃", "Iron Wave": "铁斩波", "Perfected Strike": "完美打击",
  "Pommel Strike": "剑柄打击", "Shrug It Off": "耸肩无视",
  "Sword Boomerang": "飞剑回旋镖", Thunderclap: "闪电霹雳", "True Grit": "坚毅",
  "Twin Strike": "双重打击", Warcry: "战吼", "Wild Strike": "狂野打击"
};
const phaseNames = { combat: "战斗", map: "选择路线", combat_reward: "战斗奖励", card_select: "选择卡牌", rest: "篝火", event: "事件", chest: "宝箱", boss_relic: "Boss 遗物", shop: "商店", neow: "涅奥" };
const neowBonusZh = [
  "选择一张牌加入牌组", "获得一张随机稀有牌", "移除一张牌", "升级一张牌", "变化一张牌", "选择一张无色牌加入牌组",
  "获得三瓶药水", "获得一个随机普通遗物", "最大生命提高 10%", "获得涅奥的悲恸", "获得 100 金币",
  "选择一张稀有无色牌加入牌组", "移除两张牌", "获得一个随机稀有遗物", "选择一张稀有牌加入牌组", "获得 250 金币",
  "变化两张牌", "最大生命提高 20%", "获得一个随机 Boss 遗物"
];
const neowDrawbackZh = ["", "", "最大生命降低 10%", "失去所有金币", "获得一张诅咒牌", "失去当前生命的 30%", "失去初始遗物"];

function neowOptionText(option) {
  return {
    bonus: neowBonusZh[Number(option.bonus_id)] || option.bonus || "未知奖励",
    drawback: neowDrawbackZh[Number(option.drawback_id)] || (option.drawback && option.drawback !== "INVALID" ? option.drawback : "")
  };
}

function renderNeowOptions(options, decision) {
  const rows = options.map((option, index) => {
    const text = neowOptionText(option);
    const action = decision?.top_actions?.find(item => item.action === 124 + index);
    const selected = decision?.action === 124 + index;
    const probability = action?.prob == null ? "" : `${(action.prob * 100).toFixed(1)}%`;
    return `<div class="neow-option ${selected ? "selected" : ""}">
      <span class="neow-index">${index + 1}</span>
      <div><strong>${escapeHtml(text.bonus)}</strong>${text.drawback ? `<small>代价：${escapeHtml(text.drawback)}</small>` : `<small class="free-cost">无代价</small>`}</div>
      ${probability ? `<b>${probability}</b>` : ""}
    </div>`;
  }).join("");
  return `<div class="neow-screen"><p>选择涅奥的祝福</p><div class="neow-options">${rows}</div></div>`;
}

function renderEvent(event, decision) {
  const rows = (event.options || []).map(option => {
    const action = decision?.top_actions?.find(item => item.action === option.action);
    const selected = decision?.action === option.action;
    const probability = action?.prob == null ? "" : `${(action.prob * 100).toFixed(1)}%`;
    return `<div class="event-option ${selected ? "selected" : ""}">
      <span class="event-index">${option.index + 1}</span>
      <div><strong>${escapeHtml(option.label_zh)}</strong><small>${escapeHtml(option.description_zh)}</small></div>
      ${probability ? `<b>${probability}</b>` : ""}
    </div>`;
  }).join("");
  const empty = rows ? "" : `<div class="event-empty">当前事件没有可由模型选择的普通选项</div>`;
  return `<section class="event-screen">
    <header class="event-heading"><span>随机事件</span><h2>${escapeHtml(event.title_zh)}</h2><p>${escapeHtml(event.title_en)}</p></header>
    <p class="event-description">${escapeHtml(event.description_zh)}</p>
    <div class="event-options">${rows || empty}</div>
  </section>`;
}

function decisionActionText(action, event) {
  const option = event?.options?.find(item => item.action === action?.action);
  return option
    ? `${option.label_zh}：${option.description_zh}`
    : action?.label;
}

function actionDisplayText(action, event) {
  const text = decisionActionText(action, event);
  const target = action?.target_slot;
  return target == null || Number(target) < 0
    ? text
    : `${text} -> target monster #${Number(target) + 1}`;
}

function treePathKey(path) { return (path || []).join("/"); }

function treeEdgeHtml(edge, path) {
  const childPath = [...path, Number(edge.edge_index)];
  const key = treePathKey(childPath);
  const expandable = Boolean(edge.expandable);
  const open = expandable && searchTreeOpen.has(key);
  const cached = searchTreeCache.get(key);
  const childStates = Number(edge.child_states || 0);
  const childVisits = Number(edge.child_visits || 0);
  const randomSplit = Boolean(edge.random_split) || childStates > 1;
  const childSummary = childStates > 1
    ? ` · ${childStates} public states · child N ${childVisits}`
    : "";
  const gumbelMetrics = searchTreeSearchMode !== "full_gumbel" || edge.completed_q == null
    ? ""
    : ` · CQ ${Number(edge.completed_q || 0).toFixed(3)} · score ${Number(edge.completed_score || 0).toFixed(3)}`;
  const visits = Number(edge.visits || 0);
  const valueLabel = visits > 0 ? "Q(up)" : "FPU";
  const transitionLabel = randomSplit ? " · random-world average" : "";
  const metrics = `N ${visits} · ${valueLabel} ${Number(edge.value || 0).toFixed(3)} · P ${(Number(edge.prior || 0) * 100).toFixed(1)}%${gumbelMetrics}${childSummary}${transitionLabel}`;
  const branch = open
    ? `<div class="search-tree-branch${searchTreeLoading.has(key) ? " loading" : ""}">${cached ? treeNodeHtml(cached, childPath) : "loading..."}</div>`
    : "";
  return `<div class="search-tree-edge-wrap">
    <button type="button" class="search-tree-edge${edge.selected ? " selected" : ""}" data-tree-expand="${escapeHtml(key)}" data-expandable="${expandable ? "true" : "false"}" ${expandable ? "" : "disabled"}>
      <span class="search-tree-caret">${expandable ? (open ? "▾" : "▸") : "·"}</span>
      <span class="search-tree-edge-label">${escapeHtml(edge.label || "native action")}</span>
      <span class="search-tree-edge-metrics">${metrics}</span>
    </button>${branch}
  </div>`;
}

function treeNodeHtml(snapshot, path) {
  const label = path.length ? `node ${path.join("/")}` : "root";
  const children = snapshot?.children || [];
  return `<div class="search-tree-node">
    <div class="search-tree-node-heading">${label} · visits ${Number(snapshot?.visits || 0)} · V(up) ${Number(snapshot?.value || 0).toFixed(3)} · V(sample) ${Number(snapshot?.sample_value ?? snapshot?.value ?? 0).toFixed(3)}</div>
    <div class="search-tree-children">${children.length ? children.map(edge => treeEdgeHtml(edge, path)).join("") : `<div class="search-tree-empty">no expanded children</div>`}</div>
  </div>`;
}

function renderSearchTree(search) {
  const panel = $("searchTreePanel");
  if (!search || search.method !== "nosl_is_az_cpp" || !search.tree?.valid) {
    panel.classList.add("hidden");
    searchTreeSearchMode = "puct";
    searchTreeCache.clear(); searchTreeOpen.clear(); searchTreeKey = "";
    return;
  }
  searchTreeSearchMode = search.search_mode || "puct";
  panel.classList.remove("hidden");
  const key = String(search.tree_id || `${search.simulations}:${search.native_action_bits}`);
  if (key !== searchTreeKey) {
    searchTreeKey = key;
    searchTreeCache = new Map([["", search.tree]]);
    searchTreeOpen.clear(); searchTreeLoading.clear();
  } else if (!searchTreeCache.has("")) {
    searchTreeCache.set("", search.tree);
  }
  const root = searchTreeCache.get("") || search.tree;
  $("searchTreeSummary").textContent = `root visits ${Number(root.visits || 0)} · ${root.children?.length || 0} immediate edges · click an expandable edge to inspect descendants`;
  $("searchTreeRoot").innerHTML = searchTreeCollapsed ? "" : treeNodeHtml(root, []);
  $("searchTreeRoot").style.display = searchTreeCollapsed ? "none" : "grid";
  const toggle = $("searchTreeToggle");
  toggle.textContent = searchTreeCollapsed ? "展开" : "隐藏";
  toggle.setAttribute("aria-expanded", String(!searchTreeCollapsed));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
}
function evaluationWinSeeds(e) {
  if (Array.isArray(e?.win_seeds)) return e.win_seeds.map(Number).filter(Number.isFinite);
  const results = e?.results || {};
  if (!results.win || results.dead_screen || results.other_truncation) return [];
  const start = Number(e?.seed), total = Number(e?.completed || e?.episodes);
  if (!Number.isFinite(start) || !Number.isFinite(total) || total > 20000) return [];
  const death = new Set(Object.values(e?.death_seeds || {}).flat().map(Number));
  const seeds = [];
  for (let seed = start; seed < start + total; seed += 1) {
    if (!death.has(seed)) seeds.push(seed);
  }
  return seeds;
}
function toast(message) {
  $("toast").textContent = message; $("toast").classList.add("show");
  setTimeout(() => $("toast").classList.remove("show"), 2200);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === button));
  document.querySelectorAll(".view").forEach(x => x.classList.remove("active"));
  $(`${button.dataset.tab}View`).classList.add("active");
  if (button.dataset.tab === "embedding") drawEmbedding();
  if (button.dataset.tab === "tokens") drawTokens();
  if (button.dataset.tab === "evaluation") drawDistribution();
}));

const customDefaultState = {
  encounter: "Cultist",
  player: {
    hp: 80, max_hp: 80, energy: 3, block: 0,
    powers: [], counters: {}
  },
  monsters: [
    {hp: 48, max_hp: 48, block: 0, move: "INCANTATION", powers: []}
  ],
  hand: [["STRIKE_RED", 0], ["DEFEND_RED", 0], ["BASH", 0]],
  draw_pile: [["STRIKE_RED", 0], ["DEFEND_RED", 0], ["STRIKE_RED", 0]],
  discard_pile: [],
  exhaust_pile: [],
  relics: []
};
const zhEncounterNames = {
  "Cultist": "邪教徒", "Jaw Worm": "颚虫", "Two Louse": "双虱虫",
  "Three Louse": "三虱虫", "Gremlin Nob": "地精首领", "Lagavulin": "拉格沃林",
  "The Guardian": "守护者", "Hexaghost": "六火亡魂", "Slime Boss": "史莱姆首领",
  "Champ": "冠军", "The Heart": "心脏", "Time Eater": "时间吞噬者",
  "Awakened One": "觉醒者", "Donu and Deca": "多纳与迪卡"
};
function setCustomStateText(value) {
  $("customStateJson").value = JSON.stringify(value, null, 2);
}
async function loadCustomEncounters() {
  try {
    const data = await api("/api/custom-combat/encounters");
    const select = $("customEncounter");
    select.innerHTML = (data.encounters || []).map(name =>
      `<option value="${escapeHtml(name)}">${escapeHtml(zhEncounterNames[name] || name)}</option>`).join("");
    if ([...select.options].some(x => x.value === customDefaultState.encounter)) {
      select.value = customDefaultState.encounter;
    }
  } catch (error) {
    $("customStateError").textContent = `无法读取敌人列表：${error.message}`;
  }
}
if ($("customStateJson")) {
  setCustomStateText(customDefaultState);
  loadCustomEncounters();
  $("customTemplateButton").addEventListener("click", () => {
    setCustomStateText({...customDefaultState, encounter: $("customEncounter").value || "Cultist"});
    $("customStateError").textContent = "";
  });
  $("customApplyButton").addEventListener("click", async () => {
    const button = $("customApplyButton");
    try {
      const spec = JSON.parse($("customStateJson").value);
      spec.encounter = $("customEncounter").value || spec.encounter || "Cultist";
      button.disabled = true;
      $("customStateError").textContent = "正在构造战斗状态…";
      appState.watch = await api("/api/custom-combat/apply", {
        method: "POST", body: JSON.stringify(spec)
      });
      renderWatch();
      document.querySelector('[data-tab="watch"]').click();
      $("customStateError").textContent = "已应用。当前状态已进入 Watch，可直接使用 MCTS 搜索。";
    } catch (error) {
      $("customStateError").textContent = `应用失败：${error.message}`;
    } finally {
      button.disabled = false;
    }
  });
}

const dropzone = $("dropzone");
["dragenter", "dragover"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", event => {
  const files = [...event.dataTransfer.files];
  if (files.length !== 1) {
    $("uploadStatus").classList.add("error");
    $("uploadStatus").textContent = "请只拖入一个 .zip checkpoint 文件";
    return;
  }
  uploadFile(files[0]);
});
$("fileInput").addEventListener("change", event => uploadFile(event.target.files[0]));
$("replaceButton").addEventListener("click", () => {
  uploadManuallyOpen = true;
  $("uploadOverlay").classList.remove("hidden");
});
$("closeUploadButton").addEventListener("click", () => {
  uploadManuallyOpen = false;
  $("uploadOverlay").classList.add("hidden");
});

async function uploadFile(file) {
  if (!file) return;
  const lowerName = file.name.toLowerCase();
  const isZip = lowerName.endsWith(".zip");
  if (!isZip) {
    $("uploadStatus").classList.add("error");
    $("uploadStatus").textContent = "请选择 SB3 .zip";
    return;
  }
  const signature = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  const zipSignature = signature.length === 4 && signature[0] === 0x50 && signature[1] === 0x4b && (
    (signature[2] === 0x03 && signature[3] === 0x04) ||
    (signature[2] === 0x05 && signature[3] === 0x06) ||
    (signature[2] === 0x07 && signature[3] === 0x08)
  );
  if (isZip && !zipSignature) {
    $("uploadStatus").classList.add("error");
    $("uploadStatus").textContent = "文件扩展名是 .zip，但内容不是 ZIP；请重新下载 checkpoint";
    return;
  }
  $("uploadStatus").classList.remove("error");
  $("uploadStatus").textContent = `正在上传并加载 ${file.name}…`;
  dropzone.style.pointerEvents = "none";
  try {
    const data = await api("/api/upload", {
      method: "POST", body: file,
      headers: {"X-Filename": file.name, "X-Eval-Episodes": $("uploadEpisodes").value}
    });
    appState = data;
    selectedMapAct = null;
    previewedMapKeys.clear();
    uploadManuallyOpen = false;
    $("uploadOverlay").classList.add("hidden");
    renderAll();
    toast("参数已载入，四项测试已启动");
  } catch (error) {
    $("uploadStatus").classList.add("error"); $("uploadStatus").textContent = error.message;
  } finally { dropzone.style.pointerEvents = ""; }
}

function gumbelSearchParams(prefix) {
  return {
    search_mode: $(`${prefix}SearchMode`).value,
    paper_full_gumbel: $(`${prefix}PaperFullGumbel`).checked,
    gumbel_candidates: Number($(`${prefix}GumbelCandidates`).value),
    gumbel_scale: Number($(`${prefix}GumbelScale`).value),
    gumbel_value_scale: Number($(`${prefix}GumbelValueScale`).value),
    gumbel_max_visit_init: Number($(`${prefix}GumbelMaxVisitInit`).value),
    gumbel_rescale_q: $(`${prefix}GumbelRescaleQ`).checked,
    gumbel_use_mixed_value: $(`${prefix}GumbelMixedValue`).checked,
    gumbel_interior_uniformity: Number($(`${prefix}GumbelInteriorUniformity`).value),
    gumbel_interior_coverage_fraction: Number($(`${prefix}GumbelInteriorCoverageFraction`).value),
    gumbel_interior_value_mode: Number($(`${prefix}GumbelInteriorValueMode`)?.value ?? 2),
    end_turn_as_leaf: $(`${prefix}EndTurnAsLeaf`).checked,
  };
}

const strictPaperControlValues = {
  GumbelCandidates: 16,
  GumbelScale: 1,
  GumbelValueScale: 0.1,
  GumbelMaxVisitInit: 50,
  GumbelInteriorUniformity: 0,
  GumbelInteriorCoverageFraction: 0,
  GumbelInteriorValueMode: 0,
};

function applyStrictPaperControls(prefix) {
  $(`${prefix}SearchMode`).value = "full_gumbel";
  $(`${prefix}LeafBatchSize`).value = 1;
  $(`${prefix}PriorTemperature`).value = 1;
  Object.entries(strictPaperControlValues).forEach(([suffix, value]) => {
    const node = $(`${prefix}${suffix}`);
    if (node) node.value = value;
  });
  $(`${prefix}GumbelRescaleQ`).checked = true;
  $(`${prefix}GumbelMixedValue`).checked = true;
  if (prefix === "watch") setWatchPolicyMode(false, false);
  const simulations = $(`${prefix}Simulations`);
  simulations.min = "1";
  if (Number(simulations.value) < 1) simulations.value = 512;
}

function setGumbelControlsVisible(prefix) {
  const mode = $(`${prefix}SearchMode`).value;
  const visible = mode === "gumbel" || mode === "full_gumbel";
  const paperControl = $(`${prefix}PaperFullGumbel`);
  const strict = mode === "full_gumbel" && paperControl.checked;
  paperControl.closest(".paper-control").classList.toggle("unavailable", mode !== "full_gumbel");
  document.querySelectorAll(".gumbel-control").forEach(control => {
    const input = control.querySelector("[id]");
    if (input?.id.startsWith(prefix)) {
      control.classList.toggle("unavailable", !visible);
      input.disabled = !visible || strict;
    }
  });
  $(`${prefix}SearchMode`).disabled = strict;
  $(`${prefix}LeafBatchSize`).disabled = strict;
  $(`${prefix}PriorTemperature`).disabled = strict;
  if (prefix === "watch") {
    $("watchPolicyMode").querySelectorAll("button").forEach(button => {
      button.disabled = strict;
    });
  }
  $(`${prefix}Simulations`).min = strict ? "1" : "0";
}

function restoreGumbelControls(prefix, source, modeDirty, paramsDirty, paperModeDirty) {
  const fieldPrefix = prefix === "watch" ? "mcts_" : "";
  if (!modeDirty) {
    const mode = source?.[`${fieldPrefix}search_mode`];
    if (mode != null) $(`${prefix}SearchMode`).value = mode;
  }
  if (!paramsDirty) {
    const values = {
      GumbelCandidates: source?.[`${fieldPrefix}gumbel_candidates`],
      GumbelScale: source?.[`${fieldPrefix}gumbel_scale`],
      GumbelValueScale: source?.[`${fieldPrefix}gumbel_value_scale`],
      GumbelMaxVisitInit: source?.[`${fieldPrefix}gumbel_max_visit_init`],
      GumbelInteriorUniformity: source?.[`${fieldPrefix}gumbel_interior_uniformity`],
      GumbelInteriorCoverageFraction: source?.[`${fieldPrefix}gumbel_interior_coverage_fraction`],
      GumbelInteriorValueMode: source?.[`${fieldPrefix}gumbel_interior_value_mode`],
    };
    Object.entries(values).forEach(([suffix, value]) => {
      const node = $(`${prefix}${suffix}`);
      if (value != null && document.activeElement !== node) node.value = value;
    });
    const rescale = source?.[`${fieldPrefix}gumbel_rescale_q`];
    const mixed = source?.[`${fieldPrefix}gumbel_use_mixed_value`];
    if (rescale != null) $(`${prefix}GumbelRescaleQ`).checked = Boolean(rescale);
    if (mixed != null) $(`${prefix}GumbelMixedValue`).checked = Boolean(mixed);
    const endTurnLeaf = source?.[`${fieldPrefix}end_turn_as_leaf`];
    if (endTurnLeaf != null) $(`${prefix}EndTurnAsLeaf`).checked = Boolean(endTurnLeaf);
  }
  if (!paperModeDirty) {
    const paper = source?.[`${fieldPrefix}paper_full_gumbel`];
    if (paper != null) $(`${prefix}PaperFullGumbel`).checked = Boolean(paper);
  }
  if ($(`${prefix}PaperFullGumbel`).checked) applyStrictPaperControls(prefix);
  setGumbelControlsVisible(prefix);
}

$("resetButton").addEventListener("click", async () => {
  pause();
  const button = $("resetButton");
  button.disabled = true;
  const precompute = $("watchPrecompute").checked;
  button.textContent = precompute ? "Precomputing..." : "Resetting...";
  $("watchMeta").textContent = precompute ? "Precomputing route..." : "Resetting...";
  try {
    appState.watch = await api("/api/watch/reset", {method:"POST", body:JSON.stringify({
      seed: Number($("watchSeed").value),
      simulations: Number($("watchSimulations").value),
      depth: Number($("watchDepth").value),
      leaf_batch_size: Number($("watchLeafBatchSize").value),
      prior_temperature: Number($("watchPriorTemperature").value),
      ...gumbelSearchParams("watch"),
      stochastic: watchStochasticValue(),
      upper_strategy: watchUpperStrategyValue(),
      combat_strategy: watchCombatStrategyValue(),
      precompute,
    }), headers:{"Content-Type":"application/json"}});
    watchSeedDirty = false;
    watchSimulationsDirty = false;
    watchDepthDirty = false;
    watchLeafBatchSizeDirty = false;
    watchPriorTemperatureDirty = false;
    watchSearchModeDirty = false;
    watchGumbelParamsDirty = false;
    watchPaperModeDirty = false;
    watchPolicyModeDirty = false;
    watchUpperStrategyDirty = false;
    watchCombatStrategyDirty = false;
    watchPrecomputeDirty = false;
    selectedMapAct = null;
    previewedMapKeys.clear();
    renderWatch();
  }
  catch (e) { toast(e.message); }
  finally { button.disabled = false; button.textContent = "重开"; }
});
$("watchSeed").addEventListener("input", () => { watchSeedDirty = true; });
$("watchSimulations").addEventListener("input", () => { watchSimulationsDirty = true; });
$("watchDepth").addEventListener("input", () => { watchDepthDirty = true; });
$("watchLeafBatchSize").addEventListener("input", () => { watchLeafBatchSizeDirty = true; });
$("watchPriorTemperature").addEventListener("input", () => { watchPriorTemperatureDirty = true; });
$("watchSearchMode").addEventListener("change", () => {
  watchSearchModeDirty = true;
  if ($("watchSearchMode").value !== "full_gumbel") {
    $("watchPaperFullGumbel").checked = false;
    watchPaperModeDirty = true;
  }
  setGumbelControlsVisible("watch");
});
$("watchPaperFullGumbel").addEventListener("change", () => {
  watchPaperModeDirty = true;
  if ($("watchPaperFullGumbel").checked) applyStrictPaperControls("watch");
  setGumbelControlsVisible("watch");
});
for (const id of ["watchGumbelCandidates", "watchGumbelScale", "watchGumbelValueScale", "watchGumbelMaxVisitInit", "watchGumbelInteriorUniformity"]) {
  $(id).addEventListener("input", () => { watchGumbelParamsDirty = true; });
}
document.querySelectorAll("#watchGumbelInteriorCoverageFraction,#watchGumbelInteriorValueMode").forEach(node => node.addEventListener("input", () => { watchGumbelParamsDirty = true; }));
document.querySelectorAll("#watchGumbelInteriorValueMode").forEach(node => node.addEventListener("change", () => { watchGumbelParamsDirty = true; }));
for (const id of ["watchGumbelRescaleQ", "watchGumbelMixedValue"]) {
  $(id).addEventListener("change", () => { watchGumbelParamsDirty = true; });
}
$("watchEndTurnAsLeaf").addEventListener("change", () => {
  watchGumbelParamsDirty = true;
});
$("watchPrecompute").addEventListener("change", () => { watchPrecomputeDirty = true; });
$("watchBossButton").addEventListener("click", async () => {
  if (!appState?.watch) return;
  pause();
  const button = $("watchBossButton");
  const targetAct = Number($("watchBossAct").value);
  button.disabled = true;
  button.textContent = "快进中...";
  $("watchMeta").textContent = `正在推进到 Act ${targetAct} Boss...`;
  try {
    appState.watch = await api("/api/watch/reset", {method:"POST", body:JSON.stringify({
      seed: Number($("watchSeed").value),
      simulations: Number($("watchSimulations").value),
      depth: Number($("watchDepth").value),
      leaf_batch_size: Number($("watchLeafBatchSize").value),
      prior_temperature: Number($("watchPriorTemperature").value),
      ...gumbelSearchParams("watch"),
      stochastic: watchStochasticValue(),
      upper_strategy: watchUpperStrategyValue(),
      combat_strategy: watchCombatStrategyValue(),
      precompute: false,
      target_boss_act: targetAct,
    }), headers:{"Content-Type":"application/json"}});
    watchSeedDirty = false;
    watchSimulationsDirty = false;
    watchDepthDirty = false;
    watchLeafBatchSizeDirty = false;
    watchPriorTemperatureDirty = false;
    watchSearchModeDirty = false;
    watchGumbelParamsDirty = false;
    watchPaperModeDirty = false;
    watchPolicyModeDirty = false;
    watchUpperStrategyDirty = false;
    watchCombatStrategyDirty = false;
    watchPrecomputeDirty = false;
    selectedMapAct = targetAct;
    previewedMapKeys.clear();
    renderWatch();
  } catch (e) {
    toast(e.message);
  } finally {
    button.disabled = false;
    button.textContent = "快进 Boss";
  }
});
function watchStochasticValue() {
  return $("watchPolicyMode").querySelector("button.active")?.dataset.stochastic !== "false";
}
function setWatchPolicyMode(stochastic, dirty = true) {
  const value = Boolean(stochastic);
  $("watchPolicyMode").querySelectorAll("button").forEach(button => {
    button.classList.toggle("active", button.dataset.stochastic === String(value));
  });
  if (dirty) watchPolicyModeDirty = true;
}
$("watchPolicyMode").querySelectorAll("button").forEach(button => {
  button.addEventListener("click", () => setWatchPolicyMode(button.dataset.stochastic === "true"));
});
function watchUpperStrategyValue() {
  return $("watchUpperStrategy").querySelector("button.active")?.dataset.upperStrategy || "rules";
}
function setWatchUpperStrategy(strategy, dirty = true) {
  const value = strategy === "model" ? "model" : "rules";
  $("watchUpperStrategy").querySelectorAll("button").forEach(button => {
    button.classList.toggle("active", button.dataset.upperStrategy === value);
  });
  if (dirty) watchUpperStrategyDirty = true;
}
$("watchUpperStrategy").querySelectorAll("button").forEach(button => {
  button.addEventListener("click", () => setWatchUpperStrategy(button.dataset.upperStrategy));
});
function watchCombatStrategyValue() {
  return $("watchCombatStrategy").querySelector("button.active")?.dataset.combatStrategy || "model";
}
function setWatchCombatStrategy(strategy, dirty = true) {
  const value = strategy === "planner" ? "planner" : "model";
  $("watchCombatStrategy").querySelectorAll("button").forEach(button => {
    button.classList.toggle("active", button.dataset.combatStrategy === value);
  });
  if (dirty) watchCombatStrategyDirty = true;
}
$("watchCombatStrategy").querySelectorAll("button").forEach(button => {
  button.addEventListener("click", () => setWatchCombatStrategy(button.dataset.combatStrategy));
});
$("watchSeed").addEventListener("keydown", event => {
  if (event.key === "Enter") $("resetButton").click();
});
for (const id of ["watchSimulations", "watchDepth", "watchLeafBatchSize", "watchPriorTemperature", "watchGumbelCandidates", "watchGumbelScale", "watchGumbelValueScale", "watchGumbelMaxVisitInit"]) {
  $(id).addEventListener("keydown", event => {
    if (event.key === "Enter") $("watchSearchApply").click();
  });
}
$("watchSearchApply").addEventListener("click", async () => {
  if (!appState?.watch) return;
  const button = $("watchSearchApply");
  pause();
  button.disabled = true;
  try {
    appState.watch = await api("/api/watch/apply", {
      method: "POST",
      body: JSON.stringify({
        simulations: Number($("watchSimulations").value),
        depth: Number($("watchDepth").value),
        leaf_batch_size: Number($("watchLeafBatchSize").value),
        prior_temperature: Number($("watchPriorTemperature").value),
        ...gumbelSearchParams("watch"),
        stochastic: watchStochasticValue(),
        combat_strategy: watchCombatStrategyValue(),
      }),
      headers: {"Content-Type": "application/json"},
    });
    watchSimulationsDirty = false;
    watchDepthDirty = false;
    watchLeafBatchSizeDirty = false;
    watchPriorTemperatureDirty = false;
    watchSearchModeDirty = false;
    watchGumbelParamsDirty = false;
    watchPaperModeDirty = false;
    watchPolicyModeDirty = false;
    watchCombatStrategyDirty = false;
    renderWatch();
    toast(appState.watch.combat_strategy === "planner" ? "回合规划下层已应用" : "Neural NOSL search settings applied");
  } catch (e) {
    toast(e.message);
  } finally {
    button.disabled = false;
  }
});
$("searchTreeToggle").addEventListener("click", () => {
  searchTreeCollapsed = !searchTreeCollapsed;
  renderSearchTree(appState?.watch?.decision?.search);
});
$("searchTreeRoot").addEventListener("click", async event => {
  const button = event.target.closest("[data-tree-expand]");
  if (!button || button.dataset.expandable !== "true") return;
  const key = button.dataset.treeExpand || "";
  const path = key ? key.split("/").map(Number).filter(Number.isInteger) : [];
  if (!path.length || searchTreeLoading.has(key)) return;
  if (searchTreeOpen.has(key)) {
    searchTreeOpen.delete(key);
    renderSearchTree(appState?.watch?.decision?.search);
    return;
  }
  searchTreeOpen.add(key);
  searchTreeLoading.add(key);
  renderSearchTree(appState?.watch?.decision?.search);
  try {
    const tree = await api("/api/watch/search-tree", {
      method: "POST",
      body: JSON.stringify({path, max_children: 64}),
      headers: {"Content-Type": "application/json"},
    });
    searchTreeCache.set(key, tree);
  } catch (error) {
    searchTreeOpen.delete(key);
    toast(error.message);
  } finally {
    searchTreeLoading.delete(key);
    renderSearchTree(appState?.watch?.decision?.search);
  }
});
$("undoButton").addEventListener("click", async () => {
  pause();
  try { appState.watch = await api("/api/watch/undo", {method:"POST"}); renderWatch(); }
  catch (e) { toast(e.message); }
});
$("mapPrevAct").addEventListener("click", () => switchMapAct(-1));
$("mapNextAct").addEventListener("click", () => switchMapAct(1));
function switchMapAct(delta) {
  selectedMapAct = Math.max(1, Math.min(3, (selectedMapAct || 1) + delta));
  renderWatch();
}

async function previewMapAct(act) {
  const w = appState?.watch;
  const key = `${w?.seed}:${act}`;
  if (!w || mapPreviewing || previewedMapKeys.has(key) || w.timeline_exhausted) return;
  previewedMapKeys.add(key);
  mapPreviewing = true;
  $("miniMapPanel").classList.add("loading");
  try {
    appState.watch = await api("/api/watch/map-preview", {
      method: "POST", body: JSON.stringify({act}),
      headers: {"Content-Type": "application/json"}
    });
    renderWatch();
  } catch (e) {
    previewedMapKeys.delete(key);
    toast(e.message);
  } finally {
    mapPreviewing = false;
    $("miniMapPanel").classList.remove("loading");
  }
}
$("stepButton").addEventListener("click", stepOnce);
$("playButton").addEventListener("click", () => { playing ? pause() : play(); });
async function stepOnce() {
  if (stepping || !appState?.watch || appState.watch.done) return;
  stepping = true; $("stepButton").disabled = true;
  const transition = captureCombatTransition(appState.watch);
  try {
    const [nextWatch] = await Promise.all([
      api("/api/watch/step", {method:"POST"}),
      animateSelectedCard(transition.animation)
    ]);
    appState.watch = nextWatch;
    renderWatch();
    playCombatOutcome(transition, nextWatch);
  }
  catch (e) { pause(); toast(e.message); }
  finally { stepping = false; $("stepButton").disabled = false; }
}
function play() { playing = true; $("playIcon").textContent = "Ⅱ"; $("playText").textContent = "暂停"; autoplay(); }
function pause() { playing = false; $("playIcon").textContent = "▶"; $("playText").textContent = "自动"; }
async function autoplay() {
  if (!playing) return;
  await stepOnce();
  if (appState?.watch?.done) { pause(); return; }
  setTimeout(autoplay, Number($("speedSelect").value));
}

function evalUpperStrategyValue() {
  return $("evalUpperStrategy").querySelector("button.active")?.dataset.evalUpperStrategy || "model";
}
function setEvalUpperStrategy(strategy, dirty = true) {
  const value = strategy === "rules" ? "rules" : "model";
  $("evalUpperStrategy").querySelectorAll("button").forEach(button => {
    button.classList.toggle("active", button.dataset.evalUpperStrategy === value);
  });
  if (dirty) evalUpperStrategyDirty = true;
}
function evalCombatStrategyValue() {
  return $("evalCombatStrategy").querySelector("button.active")?.dataset.evalCombatStrategy || "model";
}
const evalSearchControlIds = [
  "evalSimulations", "evalDepth", "evalLeafBatchSize", "evalPriorTemperature",
  "evalEndTurnAsLeaf", "evalSearchMode", "evalPaperFullGumbel",
  "evalGumbelCandidates", "evalGumbelScale", "evalGumbelValueScale",
  "evalGumbelMaxVisitInit", "evalGumbelRescaleQ", "evalGumbelMixedValue",
  "evalGumbelInteriorUniformity", "evalGumbelInteriorCoverageFraction",
  "evalGumbelInteriorValueMode",
];
function syncEvalSearchControls() {
  const disabled = evalCombatStrategyValue() === "planner";
  evalSearchControlIds.forEach(id => { $(id).disabled = disabled; });
}
function setEvalCombatStrategy(strategy, dirty = true) {
  const value = strategy === "planner" ? "planner" : "model";
  $("evalCombatStrategy").querySelectorAll("button").forEach(button => {
    button.classList.toggle("active", button.dataset.evalCombatStrategy === value);
  });
  if (dirty) evalCombatStrategyDirty = true;
  syncEvalSearchControls();
}
$("evalUpperStrategy").querySelectorAll("button").forEach(button => {
  button.addEventListener("click", () => setEvalUpperStrategy(button.dataset.evalUpperStrategy));
});
$("evalCombatStrategy").querySelectorAll("button").forEach(button => {
  button.addEventListener("click", () => setEvalCombatStrategy(button.dataset.evalCombatStrategy));
});

$("evalButton").addEventListener("click", async () => {
  try {
    appState.evaluation = await api("/api/eval/start", {method:"POST", body:JSON.stringify({
      episodes: Number($("evalEpisodes").value),
      seed: Number($("evalSeed").value),
      workers: Number($("evalWorkers").value),
      simulations: Number($("evalSimulations").value),
      depth: Number($("evalDepth").value),
      leaf_batch_size: Number($("evalLeafBatchSize").value),
      prior_temperature: Number($("evalPriorTemperature").value),
      upper_strategy: evalUpperStrategyValue(),
      combat_strategy: evalCombatStrategyValue(),
      ...gumbelSearchParams("eval"),
    }), headers:{"Content-Type":"application/json"}});
    evalSimulationsDirty = false;
    evalDepthDirty = false;
    evalLeafBatchSizeDirty = false;
    evalPriorTemperatureDirty = false;
    evalSearchModeDirty = false;
    evalGumbelParamsDirty = false;
    evalPaperModeDirty = false;
    evalUpperStrategyDirty = false;
    evalCombatStrategyDirty = false;
    renderEvaluation();
  } catch (e) { toast(e.message); }
});
$("evalSimulations").addEventListener("input", () => { evalSimulationsDirty = true; });
$("evalDepth").addEventListener("input", () => { evalDepthDirty = true; });
$("evalLeafBatchSize").addEventListener("input", () => { evalLeafBatchSizeDirty = true; });
$("evalPriorTemperature").addEventListener("input", () => { evalPriorTemperatureDirty = true; });
$("evalSearchMode").addEventListener("change", () => {
  evalSearchModeDirty = true;
  if ($("evalSearchMode").value !== "full_gumbel") {
    $("evalPaperFullGumbel").checked = false;
    evalPaperModeDirty = true;
  }
  setGumbelControlsVisible("eval");
});
$("evalPaperFullGumbel").addEventListener("change", () => {
  evalPaperModeDirty = true;
  if ($("evalPaperFullGumbel").checked) applyStrictPaperControls("eval");
  setGumbelControlsVisible("eval");
});
for (const id of ["evalGumbelCandidates", "evalGumbelScale", "evalGumbelValueScale", "evalGumbelMaxVisitInit", "evalGumbelInteriorUniformity"]) {
  $(id).addEventListener("input", () => { evalGumbelParamsDirty = true; });
}
document.querySelectorAll("#evalGumbelInteriorCoverageFraction,#evalGumbelInteriorValueMode").forEach(node => node.addEventListener("input", () => { evalGumbelParamsDirty = true; }));
$("evalGumbelInteriorValueMode").addEventListener("change", () => { evalGumbelParamsDirty = true; });
for (const id of ["evalGumbelRescaleQ", "evalGumbelMixedValue"]) {
  $(id).addEventListener("change", () => { evalGumbelParamsDirty = true; });
}
$("evalEndTurnAsLeaf").addEventListener("change", () => {
  evalGumbelParamsDirty = true;
});
for (const id of ["evalSimulations", "evalDepth", "evalLeafBatchSize", "evalPriorTemperature", "evalGumbelCandidates", "evalGumbelScale", "evalGumbelValueScale", "evalGumbelMaxVisitInit"]) {
  $(id).addEventListener("keydown", event => {
    if (event.key === "Enter") $("evalButton").click();
  });
}
$("winSeedList").addEventListener("click", async event => {
  const button = event.target.closest(".seed-pill");
  if (!button) return;
  $("watchSeed").value = button.dataset.seed;
  watchSeedDirty = false;
  document.querySelector('.tab[data-tab="watch"]').click();
  try {
    appState.watch = await api("/api/watch/reset", {method:"POST", body:JSON.stringify({
      seed: Number(button.dataset.seed),
      simulations: Number($("watchSimulations").value),
      depth: Number($("watchDepth").value),
      leaf_batch_size: Number($("watchLeafBatchSize").value),
      prior_temperature: Number($("watchPriorTemperature").value),
      ...gumbelSearchParams("watch"),
      stochastic: watchStochasticValue(),
      upper_strategy: watchUpperStrategyValue(),
      combat_strategy: watchCombatStrategyValue(),
      precompute: $("watchPrecompute").checked,
    }), headers:{"Content-Type":"application/json"}});
    watchSimulationsDirty = false;
    watchDepthDirty = false;
    watchLeafBatchSizeDirty = false;
    watchPriorTemperatureDirty = false;
    watchSearchModeDirty = false;
    watchGumbelParamsDirty = false;
    watchPaperModeDirty = false;
    watchPolicyModeDirty = false;
    watchUpperStrategyDirty = false;
    watchCombatStrategyDirty = false;
    watchPrecomputeDirty = false;
    renderWatch();
  } catch (e) { toast(e.message); }
});

function renderAll() {
  const format = appState?.checkpoint_format === "sb3" ? "SB3" : "";
  const paperMode = appState?.watch?.mcts_paper_full_gumbel ? " · Strict paper Gumbel" : "";
  $("checkpointPill").textContent = appState?.checkpoint ? `${format}${paperMode} · ${appState.checkpoint}` : "未载入参数";
  if (appState?.checkpoint && !uploadManuallyOpen) $("uploadOverlay").classList.add("hidden");
  renderWatch(); renderEmbedding(); renderTokens(); renderEvaluation();
}
function powersText(powers) { return Object.entries(powers || {}).filter(([,v]) => v !== 0).map(([k,v]) => `${k} ${v}`).join(" · "); }
function cardName(card) { const id = card.id ?? card[0]; const up = card.up ?? card[1] ?? 0; return `${appState?.card_catalog?.[id]?.name_zh || zhCards[id] || id}${up ? "+" : ""}`; }
const skillCards = new Set(["Defend_R", "Armaments", "Battle Trance", "Bloodletting", "Burning Pact", "Disarm", "Dual Wield", "Entrench", "Exhume", "Feel No Pain", "Flame Barrier", "Flex", "Ghostly Armor", "Havoc", "Impervious", "Infernal Blade", "Intimidate", "Limit Break", "Offering", "Power Through", "Rage", "Second Wind", "Seeing Red", "Sentinel", "Sever Soul", "Shockwave", "Shrug It Off", "Spot Weakness", "True Grit", "Warcry"]);
const powerCards = new Set(["Barricade", "Berserk", "Brutality", "Combust", "Corruption", "Dark Embrace", "Demon Form", "Evolve", "Fire Breathing", "Inflame", "Juggernaut", "Metallicize", "Rupture"]);

function deckCardClass(id) {
  const catalogKind = appState?.card_catalog?.[id]?.kind;
  if (["attack", "skill", "power", "status"].includes(catalogKind)) return catalogKind;
  if (skillCards.has(id)) return "skill";
  if (powerCards.has(id)) return "power";
  if (["AscendersBane", "Clumsy", "Curse of the Bell", "Decay", "Doubt", "Injury", "Necronomicurse", "Normality", "Pain", "Parasite", "Pride", "Regret", "Shame", "Writhe"].includes(id)) return "curse";
  if (["Burn", "Dazed", "Slimed", "Void", "Wound"].includes(id)) return "status";
  return "attack";
}

function cardMeta(card) {
  const id = card?.id ?? card?.[0] ?? String(card ?? "");
  const up = Number(card?.up ?? card?.[1] ?? 0) || 0;
  const catalog = appState?.card_catalog?.[id] || {};
  const baseEffect = catalog.effect_zh || catalog.effect_en || id;
  return {
    id,
    up,
    name: `${catalog.name_zh || zhCards[id] || id}${up ? "+" : ""}`,
    kind: deckCardClass(id),
    rawKind: catalog.kind || deckCardClass(id),
    cost: card?.cost ?? catalog.cost?.[Math.min(up, 1)] ?? "?",
    damage: catalog.damage?.[Math.min(up, 1)] ?? 0,
    block: catalog.block?.[Math.min(up, 1)] ?? 0,
    effect: upgradeEffectText(baseEffect, up > 0, id),
  };
}

function upgradeEffectText(text, upgraded, cardId) {
  if (cardId === "Burn") {
    return upgraded
      ? "无法打出。回合结束时受到4点伤害。"
      : "无法打出。回合结束时受到2点伤害。";
  }
  let out = String(text || "").trim();
  if (!out) return "";
  out = out.replace(/未升级时消耗；升级后不消耗。/g, upgraded ? "不消耗。" : "消耗。");
  out = out.replace(/升级后固有。/g, upgraded ? "固有。" : "");
  out = out.replace(/一张\/所有牌/g, upgraded ? "所有牌" : "一张牌");
  out = out.replace(/随机\/选择/g, upgraded ? "选择" : "随机");
  out = out.replace(/one\/all/gi, upgraded ? "all" : "one");
  out = out.replace(/random\/chosen/gi, upgraded ? "chosen" : "random");
  out = out.replace(/([0-9X]+)\/([0-9X]+)/g, (_m, left, right) => (
    upgraded ? right : left
  ));
  out = out.replace(/\s+/g, " ").replace(/\s*([。！？；，,])/g, "$1").trim();
  return out;
}

function splitCardText(text, maxChars, maxLines) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return [""];
  const lines = [];
  let rest = clean;
  while (rest && lines.length < maxLines) {
    if (rest.length <= maxChars || lines.length === maxLines - 1) {
      lines.push(rest);
      break;
    }
    let cut = rest.lastIndexOf(" ", maxChars);
    if (cut < Math.floor(maxChars * 0.45)) cut = maxChars;
    lines.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trimStart();
  }
  if (rest && lines.length >= maxLines) {
    lines[maxLines - 1] = `${lines[maxLines - 1].slice(0, Math.max(0, maxChars - 1))}…`;
  }
  return lines.filter(Boolean);
}

function svgTextLines(text, x, y, lineHeight, maxChars, maxLines) {
  return splitCardText(text, maxChars, maxLines).map((line, index) => (
    index === 0
      ? `<tspan x="${x}" dy="0">${escapeHtml(line)}</tspan>`
      : `<tspan x="${x}" dy="${lineHeight}">${escapeHtml(line)}</tspan>`
  )).join("");
}

function cardArtSvg(meta) {
  return `<svg class="card-svg" viewBox="0 0 100 140" role="img" aria-label="${escapeHtml(meta.name)}">
    <rect class="card-frame" x="3" y="3" width="94" height="134" rx="8"/>
    <rect class="card-name-band" x="10" y="9" width="80" height="30" rx="5"/>
    <text class="card-name-text" x="50" y="19">${svgTextLines(meta.name, 50, 19, 9.6, 13, 2)}</text>
    <rect class="card-effect-box" x="11" y="45" width="78" height="81" rx="6"/>
    <text class="card-effect-text" x="50" y="60">${svgTextLines(meta.effect, 50, 60, 9.5, 12, 5)}</text>
  </svg>`;
}

function renderVectorCard(card, {index = null, slot = null, className = "", compact = false, showOrder = index != null} = {}) {
  const meta = cardMeta(card);
  const attrs = [
    `class="vector-card ${meta.kind} ${compact ? "compact" : ""} ${className}"`,
    `title="${escapeHtml(`${index != null ? `第 ${index + 1} 张 · ` : ""}${meta.name} · ${meta.id}${meta.up ? "+" : ""}\n${meta.effect}`)}"`,
  ];
  if (slot != null) attrs.push(`data-slot="${slot}"`);
  return `<div ${attrs.join(" ")}>
    ${showOrder && index != null ? `<span class="card-order">${index + 1}</span>` : ""}
    ${cardArtSvg(meta)}
  </div>`;
}

function renderDeck(deck) {
  const cards = deck || [];
  $("deckButtonCount").textContent = cards.length;
  $("deckCards").innerHTML = cards.map((card, index) => {
    return renderVectorCard(card, {index, compact: true});
  }).join("");
}

function relicName(id) {
  return appState?.relic_catalog?.[id]?.name_zh || id;
}

function renderRelics(relics) {
  const items = (relics || []).filter(id => id && id !== "PAD");
  $("relicButtonCount").textContent = items.length;
  $("relicCards").innerHTML = items.length
    ? items.map((id, index) => `<div class="relic-chip" title="${escapeHtml(id)}">
        <span>${index + 1}</span>
        <strong>${escapeHtml(relicName(id))}</strong>
        <small>${escapeHtml(id)}</small>
      </div>`).join("")
    : `<div class="relic-empty">暂无遗物</div>`;
}

function setRewardChoicesCollapsed(collapsed) {
  rewardChoicesCollapsed = Boolean(collapsed);
  const panel = $("rewardChoicesPanel");
  if (panel) panel.classList.toggle("collapsed", rewardChoicesCollapsed);
  const button = $("rewardChoicesToggle");
  if (button) {
    button.setAttribute("aria-expanded", String(!rewardChoicesCollapsed));
    button.textContent = rewardChoicesCollapsed ? "显示" : "隐藏";
  }
}

function renderRewardChoices() {
  const list = appState?.watch?.card_reward_choices || [];
  const container = $("rewardChoicesList");
  const panel = $("rewardChoicesPanel");
  if (!container || !panel) return;
  panel.classList.toggle("empty", !list.length);
  if (!list.length) {
    container.innerHTML = `<div class="reward-choice-empty">暂无选牌记录</div>`;
    return;
  }
  const recent = list.slice(-12);
  container.innerHTML = recent.map(choice => {
    const cards = (choice.cards || []).slice(0, 3);
    const selected = choice.selected_index;
    const relics = new Set([
      ...(choice.relics || []),
      ...(appState?.watch?.canon?.global?.relics || []),
    ]);
    const missing = Math.max(0, 3 - cards.length);
    const limitedReason = missing
      ? relics.has("Busted Crown")
        ? "破碎王冠：仅 1 张候选"
        : `仅 ${cards.length} 张候选`
      : "";
    const note = choice.skipped
      ? "跳过"
      : selected != null && cards[selected]
        ? `模型选中：${cardName(cards[selected])}`
        : "未记录";
    const slots = [0, 1, 2].map(index => {
      const card = cards[index];
      if (!card) {
        return `<div class="reward-choice-card reward-choice-empty-card">
          <span>无候选</span>
        </div>`;
      }
      return renderVectorCard(card, {
        index,
        compact: true,
        showOrder: false,
        className: `reward-choice-card${selected === index ? " selected" : ""}`,
      });
    }).join("");
    return `<article class="reward-choice-row ${choice.skipped ? "skipped" : ""}">
      <div class="reward-choice-head">
        <strong>第 ${choice.floor || "?"} 层</strong>
        <span>${escapeHtml([note, limitedReason].filter(Boolean).join(" · "))}</span>
      </div>
      <div class="reward-choice-cards">
        ${slots}
      </div>
    </article>`;
  }).join("");
}

$("deckButton").addEventListener("click", () => {
  const open = $("deckViewer").classList.toggle("open");
  $("deckButton").setAttribute("aria-expanded", String(open));
});
$("relicButton").addEventListener("click", () => {
  const open = $("relicViewer").classList.toggle("open");
  $("relicButton").setAttribute("aria-expanded", String(open));
});
if ($("rewardChoicesToggle")) {
  $("rewardChoicesToggle").addEventListener("click", () => {
    setRewardChoicesCollapsed(!rewardChoicesCollapsed);
  });
  setRewardChoicesCollapsed(false);
}
document.addEventListener("click", event => {
  if (!$("deckViewer").contains(event.target)) {
    $("deckViewer").classList.remove("open");
    $("deckButton").setAttribute("aria-expanded", "false");
  }
  if (!$("relicViewer").contains(event.target)) {
    $("relicViewer").classList.remove("open");
    $("relicButton").setAttribute("aria-expanded", "false");
  }
});
function intentLabel(monster) {
  if (monster.base_damage > 0) return `攻击 ${monster.adjusted_damage ?? monster.base_damage}${monster.hits > 1 ? ` × ${monster.hits}` : ""}`;
  const move = String(monster.move || "").toUpperCase();
  if (move.includes("DEFEND") || move.includes("PROTECT")) return "防御意图";
  if (move.includes("DEBUFF") || move.includes("WEAK") || move.includes("LICK")) return "弱化意图";
  if (move.includes("SLEEP") || move.includes("STUN")) return "休眠意图";
  if (move.includes("BUFF") || move.includes("GROW") || move.includes("RITUAL")) return "强化意图";
  return "技能意图";
}

const mapRoomNames = {MONSTER:"怪物", ELITE:"精英", EVENT:"未知", SHOP:"商店", REST:"篝火", TREASURE:"宝箱", BOSS:"Boss"};
function mapNodeKey(x,y){return `${x},${y}`;}
function drawMapRoomIcon(ctx, room, x, y) {
  ctx.save();ctx.translate(x,y);ctx.strokeStyle="#30291d";ctx.fillStyle="#30291d";ctx.lineWidth=1.5;ctx.lineCap="round";
  if(room==="MONSTER"){ctx.beginPath();ctx.moveTo(-4,-4);ctx.lineTo(4,4);ctx.moveTo(4,-4);ctx.lineTo(-4,4);ctx.stroke();}
  else if(room==="ELITE"){ctx.beginPath();ctx.moveTo(0,-5);ctx.lineTo(5,0);ctx.lineTo(0,5);ctx.lineTo(-5,0);ctx.closePath();ctx.stroke();ctx.beginPath();ctx.arc(0,0,1.6,0,Math.PI*2);ctx.fill();}
  else if(room==="EVENT"){ctx.font='700 12px Georgia';ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("?",0,.5);}
  else if(room==="SHOP"){ctx.font='700 11px Georgia';ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("$",0,.5);}
  else if(room==="REST"){ctx.beginPath();ctx.moveTo(-5,4);ctx.lineTo(5,4);ctx.moveTo(-3,5);ctx.lineTo(3,-4);ctx.moveTo(3,5);ctx.lineTo(-2,-1);ctx.stroke();ctx.fillStyle="#a84b32";ctx.beginPath();ctx.moveTo(0,-5);ctx.lineTo(3,1);ctx.lineTo(0,3);ctx.lineTo(-3,1);ctx.closePath();ctx.fill();}
  else if(room==="TREASURE"){ctx.strokeRect(-5,-3,10,7);ctx.beginPath();ctx.moveTo(-5,-3);ctx.quadraticCurveTo(0,-7,5,-3);ctx.moveTo(0,-3);ctx.lineTo(0,4);ctx.stroke();}
  else {ctx.beginPath();ctx.arc(0,0,2.5,0,Math.PI*2);ctx.fill();}
  ctx.restore();
}
function renderMiniMap(map, watch) {
  const panel=$("miniMapPanel"),canvas=$("miniMapCanvas");
  const nodes=map?.nodes||[];panel.classList.toggle("empty",!nodes.length);
  const dagEnabled = Boolean(appState?.capabilities?.map_dag);
  $("miniMapAct").textContent=map?.act?`ACT ${map.act}${dagEnabled ? " · DAG" : ""}`:"ACT —";
  $("miniMapBoss").textContent=map?.boss||"";
  $("mapPrevAct").disabled=(selectedMapAct||1)<=1;
  $("mapNextAct").disabled=(selectedMapAct||1)>=3;
  if(!canvas.offsetParent)return;
  const {ctx,w,h}=canvasSetup(canvas);ctx.clearRect(0,0,w,h);miniMapScreenNodes=[];
  if(!nodes.length){ctx.fillStyle="#655c47";ctx.font='12px "Microsoft YaHei"';ctx.textAlign="center";ctx.fillText(watch?.timeline_exhausted?"本局未到达该 Act":"正在推演该 Act 路线…",w/2,h/2);return;}
  const px=x=>20+x/6*(w-40),py=y=>h-22-y/14*(h-72),nodeKeys=new Set(nodes.map(n=>mapNodeKey(n.x,n.y)));
  const route=(watch?.timeline||[]).filter(n=>n.act===map.act&&n.y>=0&&n.y<=14);
  const routeByKey=new Map(route.map(n=>[mapNodeKey(n.x,n.y),n]));
  ctx.save();ctx.strokeStyle="rgba(65,54,36,.62)";ctx.lineWidth=1.2;ctx.setLineDash([3,4]);
  nodes.forEach(n=>(n.children||[]).forEach(child=>{if(!nodeKeys.has(mapNodeKey(child,n.y+1)))return;ctx.beginPath();ctx.moveTo(px(n.x),py(n.y)-7);ctx.lineTo(px(child),py(n.y+1)+7);ctx.stroke();}));
  const topNodes=nodes.filter(n=>n.y===14),boss={x:w/2,y:19};topNodes.forEach(n=>{ctx.beginPath();ctx.moveTo(px(n.x),py(14)-7);ctx.lineTo(boss.x,boss.y+9);ctx.stroke();});ctx.restore();
  ctx.save();ctx.lineCap="round";ctx.lineWidth=4;ctx.setLineDash([]);
  route.slice(1).forEach((node,index)=>{const prev=route[index];if(node.y!==prev.y+1)return;ctx.strokeStyle=node.step<=watch.step?"#9d392d":"#39758a";ctx.beginPath();ctx.moveTo(px(prev.x),py(prev.y));ctx.lineTo(px(node.x),py(node.y));ctx.stroke();});ctx.restore();
  ctx.save();ctx.fillStyle="#8e2f29";ctx.strokeStyle="#3e2118";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(boss.x,boss.y-8);ctx.lineTo(boss.x+8,boss.y);ctx.lineTo(boss.x,boss.y+8);ctx.lineTo(boss.x-8,boss.y);ctx.closePath();ctx.fill();ctx.stroke();ctx.fillStyle="#f1d889";ctx.font='700 9px Georgia';ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("B",boss.x,boss.y+.5);ctx.restore();
  nodes.forEach(n=>{const x=px(n.x),y=py(n.y),key=mapNodeKey(n.x,n.y),pathNode=routeByKey.get(key),isReached=Boolean(pathNode&&pathNode.step<=watch.step),isCurrent=map.act===(watch.stats?.act||watch.canon?.global?.act)&&map.current&&n.x===map.current[0]&&n.y===map.current[1];ctx.save();ctx.fillStyle=pathNode?(isReached?"#d7b56a":"#a9ccd2"):n.choice?"#b7cfd0":n.reachable?"#d8cda8":"#aaa184";ctx.strokeStyle=pathNode?(isReached?"#9d392d":"#39758a"):n.choice?"#467b91":"#514836";ctx.lineWidth=pathNode?2.6:1.2;ctx.beginPath();ctx.arc(x,y,pathNode?9:7.5,0,Math.PI*2);ctx.fill();ctx.stroke();if(isCurrent){ctx.strokeStyle="#f0c653";ctx.shadowColor="#f0c653";ctx.shadowBlur=8;ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,y,13.5,0,Math.PI*2);ctx.stroke();}ctx.restore();drawMapRoomIcon(ctx,n.room,x,y);miniMapScreenNodes.push({...n,sx:x,sy:y,reached:isReached,current:isCurrent,pathStep:pathNode?.step});});
}

function captureCombatTransition(w) {
  const canon = w?.canon || {}, animation = w?.decision?.animation || null;
  const targetSlot = animation?.target_slot ?? (animation?.kind === "target" ? animation.slot : null);
  const target = targetSlot != null
    ? document.querySelector(`.monster[data-slot="${targetSlot}"]`)
    : null;
  const targetRect = target?.getBoundingClientRect();
  return {
    animation,
    playerHp: canon.player?.hp,
    enemyAttackExpected: (canon.monsters || []).reduce((total, monster) => total + (
      monster.gone ? 0 : Math.max(0, monster.adjusted_damage ?? monster.base_damage ?? 0) * Math.max(1, monster.hits || 1)
    ), 0),
    monsters: (canon.monsters || []).map(monster => ({hp: monster.hp, block: monster.block, gone: monster.gone})),
    targetPoint: targetRect ? {x: targetRect.left + targetRect.width / 2, y: targetRect.top + targetRect.height / 2} : null
  };
}

function animateSelectedCard(animation) {
  if (animation?.kind !== "play_card") return Promise.resolve();
  const card = document.querySelector(`.game-card[data-slot="${animation.slot}"]`);
  if (!card) return Promise.resolve();
  const rect = card.getBoundingClientRect(), clone = card.cloneNode(true);
  clone.classList.add("card-flight");
  Object.assign(clone.style, {left:`${rect.left}px`, top:`${rect.top}px`, width:`${rect.width}px`, height:`${rect.height}px`});
  document.body.appendChild(clone);
  card.classList.add("card-playing");
  requestAnimationFrame(() => clone.classList.add("active"));
  return new Promise(resolve => setTimeout(() => { clone.remove(); resolve(); }, 340));
}

function combatText(text, point, className = "") {
  if (!point) return;
  const arena = $("arena"), rect = arena.getBoundingClientRect(), node = document.createElement("div");
  node.className = `combat-text ${className}`;
  node.textContent = text;
  node.style.left = `${point.x - rect.left}px`;
  node.style.top = `${point.y - rect.top}px`;
  arena.appendChild(node);
  setTimeout(() => node.remove(), 850);
}

function playCombatOutcome(before, after) {
  const canon = after?.canon || {}, nextMonsters = canon.monsters || [];
  if (before.playerHp != null && canon.player?.hp != null) {
    const lost = Math.max(0, before.playerHp - canon.player.hp);
    const attacked = before.animation?.kind === "end_turn" && before.enemyAttackExpected > 0;
    if (lost > 0 || attacked) {
      const player = document.querySelector(".player-avatar"), figure = document.querySelector(".player-figure");
      const hitClass = lost > 0 ? "player-hit" : "player-blocked";
      player?.classList.add(hitClass);
      setTimeout(() => player?.classList.remove(hitClass), 430);
      const rect = figure?.getBoundingClientRect();
      if (lost > 0) combatText(`-${lost}`, rect ? {x:rect.left + rect.width/2, y:rect.top} : null, "player-damage");
    }
  }
  before.monsters.forEach((monster, slot) => {
    const next = nextMonsters[slot];
    if (!next) return;
    const hpLost = Math.max(0, monster.hp - next.hp);
    const blockLost = Math.max(0, monster.block - next.block);
    const wasTargeted = before.animation?.target_slot === slot
      || (before.animation?.kind === "target" && before.animation.slot === slot);
    if (!hpLost && !blockLost && !wasTargeted) return;
    const target = document.querySelector(`.monster[data-slot="${slot}"]`);
    target?.classList.add("monster-hit");
    setTimeout(() => target?.classList.remove("monster-hit"), 430);
    const rect = target?.getBoundingClientRect();
    const point = rect
      ? {x:rect.left + rect.width/2, y:rect.top + rect.height/2}
      : (wasTargeted ? before.targetPoint : null);
    if (hpLost > 0) combatText(`-${hpLost}`, point, "monster-damage");
  });
}

function renderWatch() {
  const w = appState?.watch;
  if (!w) return;
  const capabilities = appState?.capabilities || {};
  $("miniMapPanel").classList.toggle("dag-enabled", Boolean(capabilities.map_dag));
  if (!watchSeedDirty && document.activeElement !== $("watchSeed")) {
    $("watchSeed").value = w.seed;
  }
  if (!watchSimulationsDirty && document.activeElement !== $("watchSimulations")) {
    $("watchSimulations").value = w.mcts_simulations ?? 0;
  }
  if (!watchDepthDirty && document.activeElement !== $("watchDepth")) {
    $("watchDepth").value = w.mcts_depth ?? 0;
  }
  if (!watchLeafBatchSizeDirty && document.activeElement !== $("watchLeafBatchSize")) {
    $("watchLeafBatchSize").value = w.mcts_leaf_batch_size ?? 16;
  }
  if (!watchPriorTemperatureDirty && document.activeElement !== $("watchPriorTemperature")) {
    $("watchPriorTemperature").value = w.mcts_prior_temperature ?? 1;
  }
  restoreGumbelControls(
    "watch", w, watchSearchModeDirty, watchGumbelParamsDirty, watchPaperModeDirty
  );
  if (!watchPolicyModeDirty) setWatchPolicyMode(w.stochastic !== false, false);
  if (!watchUpperStrategyDirty) setWatchUpperStrategy(w.upper_strategy, false);
  if (!watchCombatStrategyDirty) setWatchCombatStrategy(w.combat_strategy, false);
  if (!watchPrecomputeDirty && document.activeElement !== $("watchPrecompute")) {
    $("watchPrecompute").checked = Boolean(w.precompute_enabled);
  }
  const stepProgress = w.precomputed_steps ? `${w.step} / ${w.precomputed_steps}` : `${w.step}`;
  const route = w.route_preprocess || {};
  const routeStatus = route.enabled
    ? ` · Act ${route.target_act} Boss${route.reached ? "就绪" : ""} · 忽略 HP/药水`
    : "";
  const upperStatus = w.upper_strategy === "rules" ? "规则" : "模型";
  $("watchMeta").textContent = `step ${stepProgress} · return ${Number(w.total_reward).toFixed(2)} · 上层 ${upperStatus}${routeStatus}`;
  $("undoButton").disabled = !w.can_undo;
  $("stepButton").disabled = w.done;
  const stats = w.stats || {}, canon = w.canon || {}, global = canon.global || {};
  const currentAct = Math.max(1, Math.min(3, Number(stats.act ?? global.act ?? 1)));
  if (selectedMapAct == null) selectedMapAct = currentAct;
  $("floorStat").textContent = `第 ${stats.floor ?? global.floor ?? "—"} 层`;
  $("actStat").textContent = `Act ${stats.act ?? global.act ?? "—"}`;
  $("goldStat").textContent = `${global.gold ?? canon.gold ?? 0} 金币`;
  $("deckStat").textContent = `${(global.deck || canon.deck || []).length} 张牌`;
  renderDeck(global.deck || canon.deck || []);
  renderRelics(global.relics || canon.relics || []);
  $("phaseBanner").textContent = w.done ? (w.won ? "胜利" : "本局结束") : (phaseNames[w.phase] || w.phase || "等待");
  const combat = w.phase === "combat" && canon.player;
  $("monsters").innerHTML = combat ? (canon.monsters || []).map((m, slot) => ({m, slot})).filter(({m}) => !m.gone).map(({m, slot}) => {
    const hp = Math.max(0, Math.min(100, 100 * m.hp / Math.max(1, m.max_hp)));
    const intent = intentLabel(m);
    return `<div class="monster" data-slot="${slot}"><div class="intent">${escapeHtml(intent)}</div><div class="monster-figure">${slot + 1}</div><div class="monster-name">${escapeHtml(m.id)}</div><div class="hpbar"><i style="width:${hp}%"></i></div><div class="monster-stats">${m.hp}/${m.max_hp} HP · ${m.block} 格挡</div><div class="power-row">${escapeHtml(powersText(m.powers))}</div></div>`;
  }).join("") : "";
  if (combat) {
    const hpPct = Math.max(0, Math.min(100, 100 * canon.player.hp / Math.max(1, canon.player.max_hp)));
    const blockPct = Math.max(0, Math.min(100, 100 * canon.player.block / 20));
    $("player").innerHTML = `<div class="player-avatar"><div class="energy-orb" title="当前能量"><strong>${canon.player.energy}</strong><span>能量</span></div><div class="player-core"><div class="player-figure">战</div><div class="resource-block hp-resource"><div><span>生命</span><b>${canon.player.hp}/${canon.player.max_hp}</b></div><div class="resource-track"><i style="width:${hpPct}%"></i></div></div><div class="resource-block block-resource"><div><span>格挡</span><b>${canon.player.block}</b></div><div class="resource-track"><i style="width:${blockPct}%"></i></div></div></div><div class="player-copy"><strong>铁甲战士</strong><div>回合 ${canon.turn}</div><div class="power-row">${escapeHtml(powersText(canon.player.powers))}</div></div></div>`;
  } else $("player").innerHTML = "";
  $("hand").innerHTML = combat ? (canon.hand || []).map((card, slot) => {
    const committed = w.pending_card_slot === slot ? "card-committed" : "";
    return renderVectorCard(card, {slot, className: `game-card ${committed}`});
  }).join("") : "";
  $("piles").textContent = combat ? `抽牌 ${canon.draw?.length || 0}　弃牌 ${canon.discard?.length || 0}　消耗 ${canon.exhaust?.length || 0}` : "";
  $("overworld").classList.toggle("visible", !combat);
  const eventVisible = !combat && w.phase === "event" && Boolean(w.event);
  $("overworld").classList.toggle("event-visible", eventVisible);
  $("overworld").parentElement.classList.toggle("event-active", eventVisible);
  if (!combat) {
    const opts = canon.options || {};
    if (w.phase === "event" && w.event) {
      $("overworld").innerHTML = renderEvent(w.event, w.decision);
    } else if ((w.phase === "event" || w.phase === "neow") && opts.neow?.length) {
      $("overworld").innerHTML = renderNeowOptions(opts.neow, w.decision);
    } else {
    let detail = `HP ${canon.hp ?? "—"}/${canon.max_hp ?? "—"}　金币 ${canon.gold ?? global.gold ?? 0}\n`;
    if (opts.nodes) detail += `可达路线：${opts.nodes.map(n => `${n[0]}:${n[1]}`).join("　")}\n`;
    const optionCards = opts.cards
      ? `<div class="option-card-grid">${opts.cards.map((card, index) => renderVectorCard(card, {index, compact: true})).join("")}</div>`
      : "";
    if (opts.cards) detail += `卡牌：${opts.cards.map(cardName).join("　")}\n`;
    if (opts.event) detail += `事件：${opts.event}`;
    $("overworld").innerHTML = `<div class="overworld-box"><h2>${escapeHtml(phaseNames[w.phase] || w.phase)}</h2><pre>${escapeHtml(detail)}</pre>${optionCards}</div>`;
    }
  }
  const d = w.decision;
  const turnPlannerActive = w.phase === "combat" && w.combat_strategy === "planner";
  $("decisionHeading").textContent = turnPlannerActive ? "回合规划决策" : "模型决策";
  $("chosenAction").textContent = actionDisplayText(d, w.event) || (w.done ? "没有下一步动作" : "等待模型");
  $("valueEstimate").textContent = d?.value == null ? "V(s) —" : `V(s) ${Number(d.value).toFixed(3)}`;
  const search = d?.search;
  const actions = search?.root_actions?.length ? search.root_actions : (d?.top_actions || []);
  $("actionList").innerHTML = actions.map(a => {
    const selected = a.selected || (!search && a.action === d?.action);
    // For native search, show the acting distribution (root visits).  Full
    // Gumbel also returns an improved policy target for training; keep that
    // target visible as a secondary annotation instead of presenting it as
    // the probability of the action that will actually be executed.
    const actingProb = search && a.visit_prob != null ? a.visit_prob : a.prob;
    const probability = actingProb == null ? "—" : `${(actingProb * 100).toFixed(1)}%`;
    const detail = search
      ? `${selected ? "实际执行 · " : ""}N ${Number(a.visits || 0)}${a.improved_prob != null ? ` · 策略目标 ${(Number(a.improved_prob) * 100).toFixed(1)}%` : ""}`
      : "";
    return `<div class="action-row${selected ? " selected" : ""}"><b>${probability}</b><span>${escapeHtml(actionDisplayText(a, w.event))}${detail ? `<small>${detail}</small>` : ""}</span><div class="prob-track"><i style="width:${actingProb == null ? 0 : actingProb*100}%"></i></div></div>`;
  }).join("");
  const planner = d?.combat_planner;
  $("searchStatus").textContent = turnPlannerActive
    ? `回合规划下层 · ${w.combat_strategy_version || "ironclad-turn-v8-awakened-phase"}${planner ? ` · ${planner.worlds_completed || 0} worlds · ${planner.native_transitions || 0} transitions` : ""} · 下方为 checkpoint prior 对照`
    : search
    ? (search.method === "nosl_is_az_cpp"
      ? `${search.paper_full_gumbel ? "Strict paper Full Gumbel" : search.search_mode_label || "Neural PUCT"} · ${search.simulations} simulations · depth ${search.depth === 0 ? "auto" : search.depth} · prior T ${Number(search.prior_temperature ?? 1).toFixed(2)}${search.search_mode === "gumbel" || search.search_mode === "full_gumbel" ? ` · candidates ${Number(search.gumbel_candidates || 0) || "all"} · G ${Number(search.gumbel_scale ?? 1).toFixed(2)} · Q scale ${Number(search.gumbel_value_scale ?? 0.1).toFixed(2)}` : ""} · ${search.stochastic ? "stochastic" : "deterministic"} · NOSL${search.root_value == null ? "" : ` · root V ${Number(search.root_value).toFixed(3)}`}`
      : `Direct policy fallback: ${search.error || "neural NOSL unavailable"}`)
    : `${w.stochastic ? "stochastic" : "deterministic"} direct policy${w.mcts_simulations > 0 && w.phase !== "combat" ? " (search is combat-only)" : ""}`;
  renderSearchTree(search);
  $("lastAction").textContent = w.last_action?.label || "—";
  $("lastReward").textContent = `${Number(w.last_reward).toFixed(3)}`;
  const selectedMap = (w.maps || []).find(map => map.act === selectedMapAct) || {act:selectedMapAct, nodes:[], visited:[], current:null, boss:""};
  renderMiniMap(selectedMap, w);
  renderRewardChoices();
  const routeCount = (w.timeline || []).filter(node => node.act === selectedMapAct).length;
  if (w.precompute_enabled && routeCount < 15 && !w.timeline_exhausted) setTimeout(() => previewMapAct(selectedMapAct), 0);
}

$("miniMapCanvas").addEventListener("mousemove",event=>{const canvas=event.currentTarget,rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let best=null,dist=Infinity;miniMapScreenNodes.forEach(n=>{const d=Math.hypot(n.sx-x,n.sy-y);if(d<dist){dist=d;best=n;}});const tip=$("miniMapTooltip");if(!best||dist>15){tip.style.display="none";canvas.style.cursor="default";return;}canvas.style.cursor=best.pathStep==null?"default":"pointer";tip.innerHTML=`<strong>${escapeHtml(mapRoomNames[best.room]||best.room)}</strong><br>地图节点 ${best.y+1} · 列 ${best.x+1}${best.current?"<br>当前位置":best.pathStep!=null?(best.reached?"<br>已走路线 · 点击跳转":"<br>未来路线 · 点击跳转"):best.choice?"<br>下一步可选":""}`;tip.style.display="block";tip.style.left=`${Math.max(6,Math.min(x+12,canvas.clientWidth-125))}px`;tip.style.top=`${Math.max(32,y-48)}px`;});
$("miniMapCanvas").addEventListener("mouseleave",()=>$("miniMapTooltip").style.display="none");
$("miniMapCanvas").addEventListener("click",async event=>{if(seeking)return;const rect=event.currentTarget.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let best=null,dist=Infinity;miniMapScreenNodes.forEach(n=>{const d=Math.hypot(n.sx-x,n.sy-y);if(d<dist){dist=d;best=n;}});if(!best||dist>15||best.pathStep==null)return;pause();seeking=true;$("miniMapPanel").classList.add("loading");try{appState.watch=await api("/api/watch/seek",{method:"POST",body:JSON.stringify({step:best.pathStep}),headers:{"Content-Type":"application/json"}});renderWatch();}catch(e){toast(e.message);}finally{seeking=false;$("miniMapPanel").classList.remove("loading");}});

function renderEmbedding() {
  plotPoints = appState?.embedding?.points || [];
  const ratio = appState?.embedding?.explained_ratio || [];
  const cardCount = appState?.embedding?.card_count || Math.floor(plotPoints.length / 2);
  $("variance").textContent = ratio.length ? `全战士卡 ${cardCount} 张 · PC1 ${(ratio[0]*100).toFixed(1)}% · PC2 ${(ratio[1]*100).toFixed(1)}%` : "等待参数";
  drawEmbedding();
}
function canvasSetup(canvas) {
  const rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width*dpr)); canvas.height = Math.max(1, Math.round(rect.height*dpr));
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); return {ctx,w:rect.width,h:rect.height};
}
function drawEmbedding() {
  const canvas = $("embeddingCanvas"); if (!canvas.offsetParent) return;
  const {ctx,w,h} = canvasSetup(canvas); ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#37423c"; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(55,h-45); ctx.lineTo(w-25,h-45); ctx.moveTo(55,25); ctx.lineTo(55,h-45); ctx.stroke();
  if (!plotPoints.length) { ctx.fillStyle="#8f9993"; ctx.fillText("上传 checkpoint 后生成二维向量", 75, 70); return; }
  const xs=plotPoints.map(p=>p.x), ys=plotPoints.map(p=>p.y), minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
  const sx=x=>55+(x-minX)/(maxX-minX||1)*(w-90), sy=y=>h-45-(y-minY)/(maxY-minY||1)*(h-80);
  plotScreenPoints=plotPoints.map(p=>({...p,sx:sx(p.x),sy:sy(p.y)}));
  const cardCount = appState?.embedding?.card_count || Math.floor(plotScreenPoints.length / 2);
  ctx.strokeStyle="rgba(145,154,148,.22)"; ctx.lineWidth=1;
  for (let i=0;i<cardCount;i++){ const a=plotScreenPoints[i], b=plotScreenPoints[i+cardCount]; if(!a||!b)continue; ctx.beginPath();ctx.moveTo(a.sx,a.sy);ctx.lineTo(b.sx,b.sy);ctx.stroke(); }
  ctx.font='12px "Microsoft YaHei", sans-serif'; ctx.textAlign="center";
  const kindColor = {attack:"#e46b5f", skill:"#5d83a8", power:"#d0a94f", status:"#84918a", curse:"#9b7bb4"};
  plotScreenPoints.forEach(p=>{
    ctx.fillStyle = kindColor[p.kind] || "#5d83a8";
    if(p.upgrade){ctx.save();ctx.translate(p.sx,p.sy);ctx.rotate(Math.PI/4);ctx.fillRect(-4,-4,8,8);ctx.restore();}
    else{ctx.beginPath();ctx.arc(p.sx,p.sy,6,0,Math.PI*2);ctx.fill(); if(cardCount <= 35){ctx.fillStyle="#cdd2ce";ctx.fillText(p.name,p.sx,p.sy-12);}}
  });
}
$("embeddingCanvas").addEventListener("mousemove", event => {
  const rect=event.target.getBoundingClientRect(), x=event.clientX-rect.left, y=event.clientY-rect.top;
  let best=null, dist=Infinity; plotScreenPoints.forEach(p=>{const d=Math.hypot(p.sx-x,p.sy-y);if(d<dist){dist=d;best=p;}});
  const tip=$("plotTooltip"); if(!best||dist>18){tip.style.display="none";return;}
  tip.innerHTML=`<strong>${escapeHtml(best.name)}</strong><br>${escapeHtml(best.card_id)}<br>费用 ${best.cost} · 伤害 ${best.damage} · 格挡 ${best.block}`;
  tip.style.display="block"; tip.style.left=`${Math.min(x+14,rect.width-160)}px`; tip.style.top=`${Math.max(8,y-60)}px`;
});
$("embeddingCanvas").addEventListener("mouseleave",()=>$("plotTooltip").style.display="none");

const tokenColors = {structure:"#84918a", operator:"#e46b5f", metadata:"#d0a94f", number:"#5d83a8", semantic:"#5a9a72"};
function renderTokens() {
  const sources = appState?.token_embeddings || {}, keys = ["card", "combat"].filter(key => sources[key]);
  if (!keys.includes(tokenSource)) tokenSource = keys[0] || "card";
  $("tokenSourceSwitch").innerHTML = keys.map(key => `<button data-source="${key}" class="${key===tokenSource?"active":""}">${escapeHtml(sources[key].label)} · ${sources[key].points.length}</button>`).join("");
  $("tokenSourceSwitch").querySelectorAll("button").forEach(button => button.addEventListener("click", () => {tokenSource=button.dataset.source;renderTokens();}));
  const data = sources[tokenSource];
  tokenPlotPoints = data?.points || [];
  const ratio = data?.explained_ratio || [];
  $("tokenPlotMeta").textContent = data ? `${data.label} · ${tokenPlotPoints.length} tokens` : "当前 checkpoint 没有语言 token embedding";
  $("tokenVariance").textContent = ratio.length ? `PC1 ${(ratio[0]*100).toFixed(1)}% · PC2 ${(ratio[1]*100).toFixed(1)}%` : "等待参数";
  drawTokens();
}
function drawTokens() {
  const canvas=$("tokenCanvas"); if(!canvas.offsetParent)return;
  const {ctx,w,h}=canvasSetup(canvas);ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#37423c";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(55,h-45);ctx.lineTo(w-25,h-45);ctx.moveTo(55,25);ctx.lineTo(55,h-45);ctx.stroke();
  if(!tokenPlotPoints.length){ctx.fillStyle="#8f9993";ctx.font='13px "Microsoft YaHei", sans-serif';ctx.fillText("载入带语言层的 checkpoint 后显示 Token 向量",75,70);tokenScreenPoints=[];return;}
  const xs=tokenPlotPoints.map(p=>p.x),ys=tokenPlotPoints.map(p=>p.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const sx=x=>55+(x-minX)/(maxX-minX||1)*(w-90),sy=y=>h-45-(y-minY)/(maxY-minY||1)*(h-80),query=$("tokenSearch").value.trim().toLowerCase();
  tokenScreenPoints=tokenPlotPoints.map(p=>({...p,sx:sx(p.x),sy:sy(p.y),match:!query||p.token.toLowerCase().includes(query)}));
  ctx.font='11px "Segoe UI", sans-serif';ctx.textAlign="center";
  tokenScreenPoints.forEach(p=>{ctx.globalAlpha=p.match?1:(query ? .16 : 1);ctx.fillStyle=tokenColors[p.group]||tokenColors.semantic;ctx.beginPath();ctx.arc(p.sx,p.sy,p.match&&query?6:4,0,Math.PI*2);ctx.fill();if(query&&p.match){ctx.fillStyle="#e9e7df";ctx.fillText(p.token,p.sx,p.sy-10);}});ctx.globalAlpha=1;
}
$("tokenSearch").addEventListener("input",drawTokens);
$("tokenCanvas").addEventListener("mousemove",event=>{const rect=event.target.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let best=null,dist=Infinity;tokenScreenPoints.forEach(p=>{const d=Math.hypot(p.sx-x,p.sy-y);if(d<dist){dist=d;best=p;}});const tip=$("tokenTooltip");if(!best||dist>14){tip.style.display="none";return;}tip.innerHTML=`<strong>${escapeHtml(best.token)}</strong> <span>#${best.index}</span><br>${escapeHtml(best.group)} · norm ${Number(best.norm).toFixed(4)}${best.description_en?`<br>${escapeHtml(best.description_en)}`:""}${best.description_zh?`<br>${escapeHtml(best.description_zh)}`:""}`;tip.style.display="block";tip.style.left=`${Math.max(8,Math.min(x+14,rect.width-340))}px`;tip.style.top=`${Math.max(8,y-70)}px`;});
$("tokenCanvas").addEventListener("mouseleave",()=>$("tokenTooltip").style.display="none");

function renderEvaluation() {
  const e=appState?.evaluation || {}, done=e.completed||0, total=e.episodes||0, results=e.results||{};
  if (!evalUpperStrategyDirty && e.upper_strategy) setEvalUpperStrategy(e.upper_strategy, false);
  if (!evalCombatStrategyDirty && e.combat_strategy) setEvalCombatStrategy(e.combat_strategy, false);
  if ($("evalWorkers") && e.workers != null && document.activeElement !== $("evalWorkers")) $("evalWorkers").value = e.workers;
  if (!evalSimulationsDirty && e.simulations != null) $("evalSimulations").value = e.simulations;
  if (!evalDepthDirty && e.depth != null) $("evalDepth").value = e.depth;
  if (!evalLeafBatchSizeDirty && e.leaf_batch_size != null) $("evalLeafBatchSize").value = e.leaf_batch_size;
  if (!evalPriorTemperatureDirty && e.prior_temperature != null) $("evalPriorTemperature").value = e.prior_temperature;
  restoreGumbelControls(
    "eval", e, evalSearchModeDirty, evalGumbelParamsDirty, evalPaperModeDirty
  );
  $("evalProgress").textContent=`${done} / ${total}`; $("meanFloor").textContent=e.floor_mean==null?"—":Number(e.floor_mean).toFixed(2);
  $("winRate").textContent=done?`${((results.win||0)/done*100).toFixed(1)}%`:"—"; $("meanSteps").textContent=e.steps_mean==null?"—":Number(e.steps_mean).toFixed(0);
  $("evalEta").textContent=e.eta_seconds==null?"—":`${Math.ceil(e.eta_seconds/60)} 分钟`; $("evalProgressBar").style.width=total?`${done/total*100}%`:"0%";
  const stateLabel=e.state==="running"?"运行中":e.state==="complete"?"已完成":e.state==="error"?`错误：${e.error}`:"等待参数";
  const searchLabel=e.search_method==="turn_planner"
    ? `回合规划下层 · ${e.combat_strategy_version || "ironclad-turn-v8-awakened-phase"}`
    : e.search_method==="nosl_is_az_cpp"
    ? `${e.paper_full_gumbel ? "Strict paper Full Gumbel" : e.search_mode_label || "NOSL MCTS"} · ${e.simulations} sims · leaf batch ${e.leaf_batch_size ?? 16} · depth ${Number(e.depth)===0?"auto":e.depth} · prior T ${Number(e.prior_temperature ?? 1).toFixed(2)}${e.search_mode === "gumbel" || e.search_mode === "full_gumbel" ? ` · candidates ${Number(e.gumbel_candidates || 0) || "all"} · G ${Number(e.gumbel_scale ?? 1).toFixed(2)} · Q scale ${Number(e.gumbel_value_scale ?? 0.1).toFixed(2)}` : ""} · ${e.search_decisions||0} decisions`
    : e.search_method==="direct_policy"?"Direct policy":"";
  const upperLabel=e.upper_strategy==="rules"
    ? `规则上层 · ${e.upper_strategy_version || "ironclad-overworld-v1"}`
    : e.upper_strategy==="model"?"模型上层":"";
  const labels=[stateLabel, upperLabel, searchLabel].filter(Boolean);
  $("evalStatus").textContent=labels.join(" · ");
  const winSeeds = evaluationWinSeeds(e);
  $("winSeedCount").textContent = String(winSeeds.length);
  $("winSeedList").innerHTML = winSeeds.length
    ? winSeeds.map(seed => `<button class="seed-pill" type="button" data-seed="${seed}">${seed}</button>`).join("")
    : "—";
  drawDistribution();
}
function setDefaultInput(id, value) {
  const node = $(id);
  if (node && !node.value && value != null) node.value = value;
}
function metricText(value, percent = false) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return percent ? `${(Number(value) * 100).toFixed(1)}%` : Number(value).toFixed(2);
}
function deltaText(value, percent = false) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  const sign = number > 0 ? "+" : "";
  return percent ? `${sign}${(number * 100).toFixed(1)}pp` : `${sign}${number.toFixed(2)}`;
}

async function poll() {
  try { const fresh=await api("/api/state"); if(!appState)appState=fresh; else {appState.checkpoint=fresh.checkpoint;appState.checkpoint_format=fresh.checkpoint_format;appState.embedding=fresh.embedding;appState.token_embeddings=fresh.token_embeddings;appState.card_catalog=fresh.card_catalog;appState.relic_catalog=fresh.relic_catalog;appState.capabilities=fresh.capabilities;appState.evaluation=fresh.evaluation;} renderAll(); }
  catch (_) {}
  const fastPoll = appState?.evaluation?.state==="running";
  setTimeout(poll, fastPoll?1000:3000);
}
window.addEventListener("resize",()=>{drawEmbedding();drawTokens();drawDistribution();});
const initialParams = new URLSearchParams(location.search);
const initialView = initialParams.get("view");
if (["watch", "embedding", "tokens", "evaluation"].includes(initialView)) {
  document.querySelector(`.tab[data-tab="${initialView}"]`).click();
}
if (initialParams.get("upload") === "1") {
  uploadManuallyOpen = true;
  $("uploadOverlay").classList.remove("hidden");
}
setGumbelControlsVisible("watch");
setGumbelControlsVisible("eval");
poll();
