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
let seeking = false;
let selectedMapAct = null;
let mapPreviewing = false;
let previewedMapKeys = new Set();

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

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
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
  if (!appState?.checkpoint) return;
  uploadManuallyOpen = false;
  $("uploadOverlay").classList.add("hidden");
});

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".zip")) {
    $("uploadStatus").classList.add("error");
    $("uploadStatus").textContent = "请选择 .zip checkpoint";
    return;
  }
  const signature = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  const zipSignature = signature.length === 4 && signature[0] === 0x50 && signature[1] === 0x4b && (
    (signature[2] === 0x03 && signature[3] === 0x04) ||
    (signature[2] === 0x05 && signature[3] === 0x06) ||
    (signature[2] === 0x07 && signature[3] === 0x08)
  );
  if (!zipSignature) {
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

$("resetButton").addEventListener("click", async () => {
  pause();
  const button = $("resetButton");
  button.disabled = true;
  button.textContent = "预处理中…";
  $("watchMeta").textContent = "正在预处理完整路线…";
  try {
    appState.watch = await api("/api/watch/reset", {method:"POST", body:JSON.stringify({seed: Number($("watchSeed").value), precompute: true}), headers:{"Content-Type":"application/json"}});
    watchSeedDirty = false;
    selectedMapAct = null;
    previewedMapKeys.clear();
    renderWatch();
  }
  catch (e) { toast(e.message); }
  finally { button.disabled = false; button.textContent = "重开"; }
});
$("watchSeed").addEventListener("input", () => { watchSeedDirty = true; });
$("watchSeed").addEventListener("keydown", event => {
  if (event.key === "Enter") $("resetButton").click();
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

$("evalButton").addEventListener("click", async () => {
  try {
    appState.evaluation = await api("/api/eval/start", {method:"POST", body:JSON.stringify({episodes:Number($("evalEpisodes").value), seed:Number($("evalSeed").value)}), headers:{"Content-Type":"application/json"}});
    renderEvaluation();
  } catch (e) { toast(e.message); }
});

function renderAll() {
  $("checkpointPill").textContent = appState?.checkpoint || "未载入参数";
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
  return {
    id,
    up,
    name: `${catalog.name_zh || zhCards[id] || id}${up ? "+" : ""}`,
    kind: deckCardClass(id),
    rawKind: catalog.kind || deckCardClass(id),
    cost: card?.cost ?? catalog.cost?.[Math.min(up, 1)] ?? "?",
    damage: catalog.damage?.[Math.min(up, 1)] ?? 0,
    block: catalog.block?.[Math.min(up, 1)] ?? 0,
    effect: catalog.effect_zh || catalog.effect_en || id,
  };
}

function cardCostText(cost) {
  if (cost === -1 || cost === "X") return "X";
  if (cost == null || cost === "") return "?";
  return String(cost);
}

function shortEffect(text, max = 48) {
  const clean = String(text || "").replace(/\s+/g, "");
  return clean.length > max ? `${clean.slice(0, max)}…` : clean;
}

function hashString(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function cardArtSvg(meta) {
  const h = hashString(meta.id);
  const r1 = 18 + (h % 58), r2 = 18 + ((h >> 6) % 58), r3 = 18 + ((h >> 12) % 58);
  const initials = escapeHtml(String(meta.name || meta.id).replace("+", "").slice(0, 2));
  const typeGlyph = meta.kind === "skill" ? "盾" : meta.kind === "power" ? "焰" : meta.kind === "status" ? "晕" : meta.kind === "curse" ? "咒" : "斩";
  const motif = meta.kind === "skill"
    ? `<path class="card-art-main" d="M50 19 L78 30 Q74 70 50 82 Q26 70 22 30 Z"/><path d="M50 30 V68 M35 45 H65"/>`
    : meta.kind === "power"
      ? `<path class="card-art-main" d="M49 82 C28 65 35 48 47 35 C48 49 63 39 55 18 C78 39 78 68 49 82 Z"/><path d="M45 70 C53 60 56 53 52 43"/>`
      : meta.kind === "status" || meta.kind === "curse"
        ? `<path class="card-art-main" d="M23 52 C31 25 70 25 77 52 C67 80 34 80 23 52 Z"/><circle cx="50" cy="52" r="12"/><path d="M37 40 Q50 29 63 40"/>`
        : `<path class="card-art-main" d="M25 74 L68 20 L76 28 L33 82 Z"/><path d="M63 22 L78 37 M23 76 L38 91"/>`;
  return `<svg class="card-svg" viewBox="0 0 100 140" role="img" aria-label="${escapeHtml(meta.name)}">
    <defs>
      <radialGradient id="g${h}" cx="50%" cy="36%" r="70%">
        <stop offset="0%" stop-color="rgb(${r1 + 130},${r2 + 116},${r3 + 92})"/>
        <stop offset="100%" stop-color="rgb(${r1 + 38},${r2 + 32},${r3 + 26})"/>
      </radialGradient>
    </defs>
    <rect class="card-frame" x="3" y="3" width="94" height="134" rx="8"/>
    <rect class="card-title-band" x="10" y="9" width="80" height="18" rx="5"/>
    <circle class="card-cost-orb" cx="17" cy="18" r="11"/>
    <text class="card-cost-text" x="17" y="22">${escapeHtml(cardCostText(meta.cost))}</text>
    <text class="card-name-text" x="53" y="22">${escapeHtml(meta.name)}</text>
    <rect class="card-art-box" x="11" y="31" width="78" height="55" rx="6" fill="url(#g${h})"/>
    <g class="card-art-lines">${motif}</g>
    <text class="card-type-glyph" x="50" y="66">${typeGlyph}</text>
    <rect class="card-text-box" x="11" y="91" width="78" height="33" rx="5"/>
    <text class="card-effect-text" x="50" y="104">${escapeHtml(shortEffect(meta.effect, 18))}</text>
    <text class="card-id-text" x="50" y="118">${escapeHtml(meta.id)}</text>
    ${meta.up ? `<path class="card-upgrade-star" d="M82 12 l2.5 5 5.5 .8 -4 3.9 .9 5.5 -4.9-2.6 -4.9 2.6 .9-5.5 -4-3.9 5.5-.8z"/>` : ""}
  </svg>`;
}

function renderVectorCard(card, {index = null, slot = null, className = "", compact = false} = {}) {
  const meta = cardMeta(card);
  const attrs = [
    `class="vector-card ${meta.kind} ${compact ? "compact" : ""} ${className}"`,
    `title="${escapeHtml(`${index != null ? `第 ${index + 1} 张 · ` : ""}${meta.name} · ${meta.id}${meta.up ? "+" : ""}\n${meta.effect}`)}"`,
  ];
  if (slot != null) attrs.push(`data-slot="${slot}"`);
  return `<div ${attrs.join(" ")}>
    ${index != null ? `<span class="card-order">${index + 1}</span>` : ""}
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

$("deckButton").addEventListener("click", () => {
  const open = $("deckViewer").classList.toggle("open");
  $("deckButton").setAttribute("aria-expanded", String(open));
});
document.addEventListener("click", event => {
  if (!$("deckViewer").contains(event.target)) {
    $("deckViewer").classList.remove("open");
    $("deckButton").setAttribute("aria-expanded", "false");
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
  const target = animation?.kind === "target"
    ? document.querySelector(`.monster[data-slot="${animation.slot}"]`)
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
    const wasTargeted = before.animation?.kind === "target" && before.animation.slot === slot;
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
  const stepProgress = w.precomputed_steps ? `${w.step} / ${w.precomputed_steps}` : `${w.step}`;
  $("watchMeta").textContent = `step ${stepProgress} · return ${Number(w.total_reward).toFixed(2)}`;
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
  if (!combat) {
    const opts = canon.options || {};
    if ((w.phase === "event" || w.phase === "neow") && opts.neow?.length) {
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
  $("chosenAction").textContent = d?.label || (w.done ? "没有下一步动作" : "等待模型");
  $("valueEstimate").textContent = d?.value == null ? "V(s) —" : `V(s) ${Number(d.value).toFixed(3)}`;
  $("actionList").innerHTML = (d?.top_actions || []).map(a => `<div class="action-row"><b>${a.prob == null ? "—" : (a.prob*100).toFixed(1)+"%"}</b><span>${escapeHtml(a.label)}</span><div class="prob-track"><i style="width:${a.prob == null ? 0 : a.prob*100}%"></i></div></div>`).join("");
  $("lastAction").textContent = w.last_action?.label || "—";
  $("lastReward").textContent = `${Number(w.last_reward).toFixed(3)}`;
  const selectedMap = (w.maps || []).find(map => map.act === selectedMapAct) || {act:selectedMapAct, nodes:[], visited:[], current:null, boss:""};
  renderMiniMap(selectedMap, w);
  const routeCount = (w.timeline || []).filter(node => node.act === selectedMapAct).length;
  if (routeCount < 15 && !w.timeline_exhausted) setTimeout(() => previewMapAct(selectedMapAct), 0);
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
  $("evalProgress").textContent=`${done} / ${total}`; $("meanFloor").textContent=e.floor_mean==null?"—":Number(e.floor_mean).toFixed(2);
  $("winRate").textContent=done?`${((results.win||0)/done*100).toFixed(1)}%`:"—"; $("meanSteps").textContent=e.steps_mean==null?"—":Number(e.steps_mean).toFixed(0);
  $("evalEta").textContent=e.eta_seconds==null?"—":`${Math.ceil(e.eta_seconds/60)} 分钟`; $("evalProgressBar").style.width=total?`${done/total*100}%`:"0%";
  $("evalStatus").textContent=e.state==="running"?"运行中":e.state==="complete"?"已完成":e.state==="error"?`错误：${e.error}`:"等待参数";
  drawDistribution();
}
function drawDistribution() {
  const canvas=$("distributionCanvas"); if(!canvas.offsetParent)return; const {ctx,w,h}=canvasSetup(canvas);ctx.clearRect(0,0,w,h); distributionBars=[];
  const data=appState?.evaluation?.death_floors||{}, floors=Object.keys(data).map(Number).sort((a,b)=>a-b); if(!floors.length){ctx.fillStyle="#8f9993";ctx.fillText("评测完成后显示死亡楼层",30,45);$("distributionTooltip").style.display="none";return;}
  const max=Math.max(...floors.map(f=>data[f])), left=42,bottom=35,top=20,barW=(w-left-20)/floors.length;
  ctx.strokeStyle="#46514b";ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,h-bottom);ctx.lineTo(w-15,h-bottom);ctx.stroke();ctx.textAlign="center";ctx.font="11px Segoe UI";
  floors.forEach((f,i)=>{const bh=(h-bottom-top)*data[f]/max,x=left+i*barW+2,bw=Math.max(3,barW-4),y=h-bottom-bh;distributionBars.push({floor:f,count:data[f],x,y,w:bw,h:bh});ctx.fillStyle=f===16?"#d0a94f":"#be4c43";ctx.fillRect(x,y,bw,bh);ctx.fillStyle="#aeb5b0";if(floors.length<25||i%2===0)ctx.fillText(f,x+bw/2,h-16);if(barW>24)ctx.fillText(data[f],x+bw/2,y-5);});
}
$("distributionCanvas").addEventListener("mousemove", event => {
  const canvas=event.currentTarget,rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
  const bar=distributionBars.find(item=>x>=item.x&&x<=item.x+item.w&&y>=item.y&&y<=item.y+item.h),tip=$("distributionTooltip");
  if(!bar){tip.style.display="none";return;}
  const seeds=appState?.evaluation?.death_seeds?.[bar.floor]||[],shown=seeds.slice(0,12),more=Math.max(0,seeds.length-shown.length);
  tip.innerHTML=`<strong>第 ${bar.floor} 层 · ${bar.count} 次死亡</strong><br>Seeds：${shown.map(escapeHtml).join("、")}${more?`<br>另有 ${more} 个 seed`:""}`;
  tip.style.display="block";tip.style.left=`${Math.min(canvas.offsetLeft+x+14,canvas.parentElement.clientWidth-260)}px`;tip.style.top=`${Math.max(48,canvas.offsetTop+y-58)}px`;
});
$("distributionCanvas").addEventListener("mouseleave",()=>$("distributionTooltip").style.display="none");

async function poll() {
  try { const fresh=await api("/api/state"); if(!appState)appState=fresh; else {appState.checkpoint=fresh.checkpoint;appState.embedding=fresh.embedding;appState.token_embeddings=fresh.token_embeddings;appState.card_catalog=fresh.card_catalog;appState.capabilities=fresh.capabilities;appState.evaluation=fresh.evaluation;} renderAll(); }
  catch (_) {}
  setTimeout(poll, appState?.evaluation?.state==="running"?1000:3000);
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
poll();
