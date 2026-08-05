const $ = (id) => document.getElementById(id);
let appState = null;
let playing = false;
let stepping = false;
let plotPoints = [];
let plotScreenPoints = [];
let uploadManuallyOpen = false;

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
  if (button.dataset.tab === "evaluation") drawDistribution();
}));

const dropzone = $("dropzone");
["dragenter", "dragover"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", event => uploadFile(event.dataTransfer.files[0]));
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
  if (!file.name.toLowerCase().endsWith(".zip")) { $("uploadStatus").textContent = "请选择 .zip checkpoint"; return; }
  $("uploadStatus").classList.remove("error");
  $("uploadStatus").textContent = `正在上传并加载 ${file.name}…`;
  dropzone.style.pointerEvents = "none";
  try {
    const data = await api("/api/upload", {
      method: "POST", body: file,
      headers: {"X-Filename": file.name, "X-Eval-Episodes": $("uploadEpisodes").value}
    });
    appState = data;
    uploadManuallyOpen = false;
    $("uploadOverlay").classList.add("hidden");
    renderAll();
    toast("参数已载入，三项测试已启动");
  } catch (error) {
    $("uploadStatus").classList.add("error"); $("uploadStatus").textContent = error.message;
  } finally { dropzone.style.pointerEvents = ""; }
}

$("resetButton").addEventListener("click", async () => {
  pause();
  try { appState.watch = await api("/api/watch/reset", {method:"POST", body:JSON.stringify({seed: Number($("watchSeed").value)}), headers:{"Content-Type":"application/json"}}); renderWatch(); }
  catch (e) { toast(e.message); }
});
$("undoButton").addEventListener("click", async () => {
  pause();
  try { appState.watch = await api("/api/watch/undo", {method:"POST"}); renderWatch(); }
  catch (e) { toast(e.message); }
});
$("stepButton").addEventListener("click", stepOnce);
$("playButton").addEventListener("click", () => { playing ? pause() : play(); });
async function stepOnce() {
  if (stepping || !appState?.watch || appState.watch.done) return;
  stepping = true; $("stepButton").disabled = true;
  try { appState.watch = await api("/api/watch/step", {method:"POST"}); renderWatch(); }
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
  renderWatch(); renderEmbedding(); renderEvaluation();
}
function powersText(powers) { return Object.entries(powers || {}).filter(([,v]) => v !== 0).map(([k,v]) => `${k} ${v}`).join(" · "); }
function cardName(card) { const id = card.id ?? card[0]; const up = card.up ?? card[1] ?? 0; return `${zhCards[id] || id}${up ? "+" : ""}`; }
function intentLabel(monster) {
  if (monster.base_damage > 0) return `攻击 ${monster.base_damage}${monster.hits > 1 ? ` × ${monster.hits}` : ""}`;
  const move = String(monster.move || "").toUpperCase();
  if (move.includes("DEFEND") || move.includes("PROTECT")) return "防御意图";
  if (move.includes("DEBUFF") || move.includes("WEAK") || move.includes("LICK")) return "弱化意图";
  if (move.includes("SLEEP") || move.includes("STUN")) return "休眠意图";
  if (move.includes("BUFF") || move.includes("GROW") || move.includes("RITUAL")) return "强化意图";
  return "技能意图";
}

function renderWatch() {
  const w = appState?.watch;
  if (!w) return;
  $("watchSeed").value = w.seed;
  $("watchMeta").textContent = `step ${w.step} · return ${Number(w.total_reward).toFixed(2)}`;
  $("undoButton").disabled = !w.can_undo;
  $("stepButton").disabled = w.done;
  const stats = w.stats || {}, canon = w.canon || {}, global = canon.global || {};
  $("floorStat").textContent = `第 ${stats.floor ?? global.floor ?? "—"} 层`;
  $("actStat").textContent = `Act ${stats.act ?? global.act ?? "—"}`;
  $("goldStat").textContent = `${global.gold ?? canon.gold ?? 0} 金币`;
  $("deckStat").textContent = `${(global.deck || canon.deck || []).length} 张牌`;
  $("phaseBanner").textContent = w.done ? (w.won ? "胜利" : "本局结束") : (phaseNames[w.phase] || w.phase || "等待");
  const combat = w.phase === "combat" && canon.player;
  $("monsters").innerHTML = combat ? (canon.monsters || []).filter(m => !m.gone).map((m, i) => {
    const hp = Math.max(0, Math.min(100, 100 * m.hp / Math.max(1, m.max_hp)));
    const intent = intentLabel(m);
    return `<div class="monster"><div class="intent">${escapeHtml(intent)}</div><div class="monster-figure">${i + 1}</div><div class="monster-name">${escapeHtml(m.id)}</div><div class="hpbar"><i style="width:${hp}%"></i></div><div class="monster-stats">${m.hp}/${m.max_hp} HP · ${m.block} 格挡</div><div class="power-row">${escapeHtml(powersText(m.powers))}</div></div>`;
  }).join("") : "";
  $("player").innerHTML = combat ? `<div class="player-avatar"><div class="player-figure">战</div><div class="player-copy"><strong>铁甲战士</strong><div>${canon.player.hp}/${canon.player.max_hp} HP · ${canon.player.block} 格挡</div><div class="energy">${canon.player.energy} 能量 · 回合 ${canon.turn}</div><div class="power-row">${escapeHtml(powersText(canon.player.powers))}</div></div></div>` : "";
  $("hand").innerHTML = combat ? (canon.hand || []).map(card => {
    const cost = card.cost ?? "?"; const id = card.id; const cls = ["Defend_R","Armaments","Flex","Havoc","Shrug It Off","True Grit","Warcry"].includes(id) ? "skill" : "";
    return `<div class="game-card ${cls}"><span class="cost">${cost}</span><strong>${escapeHtml(cardName(card))}</strong><small>${escapeHtml(id)}</small></div>`;
  }).join("") : "";
  $("piles").textContent = combat ? `抽牌 ${canon.draw?.length || 0}　弃牌 ${canon.discard?.length || 0}　消耗 ${canon.exhaust?.length || 0}` : "";
  $("overworld").classList.toggle("visible", !combat);
  if (!combat) {
    const opts = canon.options || {};
    let detail = `HP ${canon.hp ?? "—"}/${canon.max_hp ?? "—"}　金币 ${canon.gold ?? global.gold ?? 0}\n`;
    if (opts.nodes) detail += `可达路线：${opts.nodes.map(n => `${n[0]}:${n[1]}`).join("　")}\n`;
    if (opts.cards) detail += `卡牌：${opts.cards.map(cardName).join("　")}\n`;
    if (opts.event) detail += `事件：${opts.event}`;
    $("overworld").innerHTML = `<div class="overworld-box"><h2>${escapeHtml(phaseNames[w.phase] || w.phase)}</h2><pre>${escapeHtml(detail)}</pre></div>`;
  }
  const d = w.decision;
  $("chosenAction").textContent = d?.label || (w.done ? "没有下一步动作" : "等待模型");
  $("valueEstimate").textContent = d?.value == null ? "V(s) —" : `V(s) ${Number(d.value).toFixed(3)}`;
  $("actionList").innerHTML = (d?.top_actions || []).map(a => `<div class="action-row"><b>${a.prob == null ? "—" : (a.prob*100).toFixed(1)+"%"}</b><span>${escapeHtml(a.label)}</span><div class="prob-track"><i style="width:${a.prob == null ? 0 : a.prob*100}%"></i></div></div>`).join("");
  $("lastAction").textContent = w.last_action?.label || "—";
  $("lastReward").textContent = `${Number(w.last_reward).toFixed(3)}`;
}

function renderEmbedding() {
  plotPoints = appState?.embedding?.points || [];
  const ratio = appState?.embedding?.explained_ratio || [];
  $("variance").textContent = ratio.length ? `PC1 ${(ratio[0]*100).toFixed(1)}% · PC2 ${(ratio[1]*100).toFixed(1)}%` : "等待参数";
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
  ctx.strokeStyle="rgba(145,154,148,.3)"; ctx.lineWidth=1;
  for (let i=0;i<20;i++){ const a=plotScreenPoints[i], b=plotScreenPoints[i+20]; ctx.beginPath();ctx.moveTo(a.sx,a.sy);ctx.lineTo(b.sx,b.sy);ctx.stroke(); }
  ctx.font='12px "Microsoft YaHei", sans-serif'; ctx.textAlign="center";
  plotScreenPoints.forEach(p=>{
    if(p.upgrade){ctx.fillStyle="#d0a94f";ctx.save();ctx.translate(p.sx,p.sy);ctx.rotate(Math.PI/4);ctx.fillRect(-4,-4,8,8);ctx.restore();}
    else{ctx.fillStyle="#5d83a8";ctx.beginPath();ctx.arc(p.sx,p.sy,7,0,Math.PI*2);ctx.fill();ctx.fillStyle="#cdd2ce";ctx.fillText(p.name,p.sx,p.sy-12);}
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

function renderEvaluation() {
  const e=appState?.evaluation || {}, done=e.completed||0, total=e.episodes||0, results=e.results||{};
  $("evalProgress").textContent=`${done} / ${total}`; $("meanFloor").textContent=e.floor_mean==null?"—":Number(e.floor_mean).toFixed(2);
  $("winRate").textContent=done?`${((results.win||0)/done*100).toFixed(1)}%`:"—"; $("meanSteps").textContent=e.steps_mean==null?"—":Number(e.steps_mean).toFixed(0);
  $("evalEta").textContent=e.eta_seconds==null?"—":`${Math.ceil(e.eta_seconds/60)} 分钟`; $("evalProgressBar").style.width=total?`${done/total*100}%`:"0%";
  $("evalStatus").textContent=e.state==="running"?"运行中":e.state==="complete"?"已完成":e.state==="error"?`错误：${e.error}`:"等待参数";
  drawDistribution();
}
function drawDistribution() {
  const canvas=$("distributionCanvas"); if(!canvas.offsetParent)return; const {ctx,w,h}=canvasSetup(canvas);ctx.clearRect(0,0,w,h);
  const data=appState?.evaluation?.death_floors||{}, floors=Object.keys(data).map(Number).sort((a,b)=>a-b); if(!floors.length){ctx.fillStyle="#8f9993";ctx.fillText("评测完成后显示死亡楼层",30,45);return;}
  const max=Math.max(...floors.map(f=>data[f])), left=42,bottom=35,top=20,barW=(w-left-20)/floors.length;
  ctx.strokeStyle="#46514b";ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,h-bottom);ctx.lineTo(w-15,h-bottom);ctx.stroke();ctx.textAlign="center";ctx.font="11px Segoe UI";
  floors.forEach((f,i)=>{const bh=(h-bottom-top)*data[f]/max,x=left+i*barW+2;ctx.fillStyle=f===16?"#d0a94f":"#be4c43";ctx.fillRect(x,h-bottom-bh,Math.max(3,barW-4),bh);ctx.fillStyle="#aeb5b0";if(floors.length<25||i%2===0)ctx.fillText(f,x+(barW-4)/2,h-16);if(barW>24)ctx.fillText(data[f],x+(barW-4)/2,h-bottom-bh-5);});
}

async function poll() {
  try { const fresh=await api("/api/state"); if(!appState)appState=fresh; else {appState.checkpoint=fresh.checkpoint;appState.embedding=fresh.embedding;appState.evaluation=fresh.evaluation;} renderAll(); }
  catch (_) {}
  setTimeout(poll, appState?.evaluation?.state==="running"?1000:3000);
}
window.addEventListener("resize",()=>{drawEmbedding();drawDistribution();});
const initialParams = new URLSearchParams(location.search);
const initialView = initialParams.get("view");
if (["watch", "embedding", "evaluation"].includes(initialView)) {
  document.querySelector(`.tab[data-tab="${initialView}"]`).click();
}
if (initialParams.get("upload") === "1") {
  uploadManuallyOpen = true;
  $("uploadOverlay").classList.remove("hidden");
}
poll();
