const el = id => document.getElementById(id);
const SKIP = "__skip__";
const kinds = {attack: "攻击", skill: "技能", power: "能力"};
let catalog = new Map(), order = [], savedOrder = [], revision = null;
let history = [], saving = false, loaded = false, dragged = null, previewVersion = 0;
const escape = text => String(text).replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
const payload = () => ({schema_version: 1, character: "IRONCLAD", upgrade_policy: "shared", order: [...order]});
const dirty = () => loaded && order.join("\n") !== savedOrder.join("\n");

async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function error(message = "") {
  el("rankingError").textContent = message;
  el("rankingError").hidden = !message;
}
function updateStatus() {
  el("rankingStatus").textContent = saving ? "正在保存…" : !loaded ? "尚未读取排名" : dirty() ? "草稿有改动 · 尚未保存" : revision ? "已保存到本机" : "初始目录顺序 · 请按偏好排序";
  el("rankingSave").disabled = saving || !loaded || (!dirty() && revision !== null);
  el("rankingSaveSide").disabled = el("rankingSave").disabled;
  el("rankingUndo").disabled = !history.length;
}
function name(id) { return id === SKIP ? "跳过，不拿牌" : catalog.get(id).name; }
function render() {
  const query = el("rankingSearch").value.trim().toLowerCase(), kind = el("rankingKind").value;
  const skipIndex = order.indexOf(SKIP);
  const visible = order.filter(id => id === SKIP || ((!kind || catalog.get(id).kind === kind) &&
    `${id} ${name(id)} ${catalog.get(id).effect}`.toLowerCase().includes(query)));
  el("rankingCount").textContent = `${visible.length - 1} / ${catalog.size} 张`;
  el("rankingList").innerHTML = visible.map(id => {
    const i = order.indexOf(id), card = catalog.get(id), skip = id === SKIP;
    const cost = skip ? "" : card.cost.map(n => n < 0 ? "X" : n).join(" / ");
    return `<li class="ranking-row ${skip ? "skip" : card.kind} ${i > skipIndex ? "below-skip" : ""}" data-id="${escape(id)}">
      <button type="button" class="ranking-handle" draggable="true" title="拖动排序，也可用右侧名次和箭头" aria-label="拖动${escape(name(id))}">⠿</button>
      <div><div class="ranking-card-title">${escape(name(id))}</div><div class="ranking-card-id">${skip ? "拿牌 / 放弃的分界" : escape(id)}</div><div class="ranking-meta">${skip ? `${i} 张在前 · ${order.length - i - 1} 张在后` : `${kinds[card.kind]} · 费用 ${cost}`}</div></div>
      <div class="ranking-effect">${skip ? "没有排在我前面的候选牌，就跳过。" : escape(card.effect)}${!skip && card.effect.includes("/") ? '<span class="ranking-card-id">（普通 / 升级）</span>' : ""}</div>
      <div class="ranking-move"><input type="number" min="1" max="${order.length}" value="${i + 1}" aria-label="${escape(name(id))}的名次"><button type="button" data-move="-1" ${i === 0 ? "disabled" : ""} aria-label="上移${escape(name(id))}">↑</button><button type="button" data-move="1" ${i === order.length - 1 ? "disabled" : ""} aria-label="下移${escape(name(id))}">↓</button></div>
    </li>`;
  }).join("");
  updateStatus();
}
function replaceOrder(next) {
  if (next.join("\n") === order.join("\n")) return;
  history.push([...order]);
  if (history.length > 100) history.shift();
  order = next;
  error(); render(); preview();
}
function move(id, index) {
  const from = order.indexOf(id);
  if (from < 0 || !Number.isInteger(index)) return;
  const next = order.filter(x => x !== id);
  next.splice(Math.max(0, Math.min(next.length, index)), 0, id);
  replaceOrder(next);
}
function rowFor(id) { return [...el("rankingList").children].find(row => row.dataset.id === id); }
el("rankingList").addEventListener("change", event => {
  if (!event.target.matches('input[type="number"]')) return;
  const id = event.target.closest("li").dataset.id;
  const number = Number(event.target.value);
  if (!event.target.value || !Number.isInteger(number) || number < 1 || number > order.length) {
    event.target.value = order.indexOf(id) + 1;
    error(`名次应为 1 到 ${order.length} 的整数`); return;
  }
  move(id, number - 1);
});
el("rankingList").addEventListener("click", event => {
  const button = event.target.closest("[data-move]");
  if (!button) return;
  const id = button.closest("li").dataset.id;
  move(id, order.indexOf(id) + Number(button.dataset.move));
  rowFor(id)?.querySelector(`[data-move="${button.dataset.move}"]`)?.focus({preventScroll: true});
});
function clearDrop() {
  el("rankingList").querySelectorAll(".drop-before, .drop-after").forEach(row => row.classList.remove("drop-before", "drop-after"));
}
el("rankingList").addEventListener("dragstart", event => {
  if (!event.target.classList.contains("ranking-handle")) return;
  const row = event.target.closest("li");
  dragged = row.dataset.id;
  event.dataTransfer.setData("text/plain", dragged);
  event.dataTransfer.effectAllowed = "move";
  row.classList.add("dragging");
});
el("rankingList").addEventListener("dragover", event => {
  const row = event.target.closest("li");
  if (!dragged || !row || row.dataset.id === dragged) return;
  event.preventDefault(); clearDrop();
  event.dataTransfer.dropEffect = "move";
  const rect = row.getBoundingClientRect();
  row.classList.add(event.clientY < rect.top + rect.height / 2 ? "drop-before" : "drop-after");
  if (event.clientY < 150) window.scrollBy(0, -20);
  if (event.clientY > window.innerHeight - 90) window.scrollBy(0, 20);
});
el("rankingList").addEventListener("drop", event => {
  event.preventDefault();
  const row = event.target.closest("li"), id = dragged;
  if (id && row && row.dataset.id !== id) {
    const remaining = order.filter(x => x !== id), rect = row.getBoundingClientRect();
    move(id, remaining.indexOf(row.dataset.id) + (event.clientY >= rect.top + rect.height / 2 ? 1 : 0));
  }
  dragged = null; clearDrop();
});
el("rankingList").addEventListener("dragend", () => { dragged = null; clearDrop(); el("rankingList").querySelectorAll(".dragging").forEach(row => row.classList.remove("dragging")); });
el("rankingSearch").addEventListener("input", render);
el("rankingKind").addEventListener("change", render);
el("rankingUndo").addEventListener("click", () => { if (history.length) { order = history.pop(); render(); preview(); } });
el("rankingFindSkip").addEventListener("click", () => rowFor(SKIP)?.scrollIntoView({behavior: "smooth", block: "center"}));

async function preview() {
  if (!loaded) return;
  const version = ++previewVersion;
  el("rankingChoice").textContent = "试选中…";
  try {
    const result = await request("/api/card-ranking/preview", {ranking: payload(), cards: [0, 1, 2].map(i => el(`rankingOffer${i}`).value), can_skip: el("rankingCanSkip").checked});
    if (version === previewVersion) el("rankingChoice").textContent = result.index === null ? "→ 跳过，不拿牌" : `→ ${name(result.card)}（候选 ${result.index + 1}）`;
  } catch (e) { if (version === previewVersion) el("rankingChoice").textContent = `试选失败：${e.message}`; }
}
document.querySelectorAll(".ranking-offer, #rankingCanSkip").forEach(input => input.addEventListener("change", preview));

async function load() {
  try {
    const state = await request("/api/card-ranking");
    catalog = new Map(state.cards.map(card => [card.id, card]));
    order = [...state.ranking.order]; savedOrder = [...order]; revision = state.revision;
    history = []; loaded = true;
    const options = state.cards.map(card => `<option value="${escape(card.id)}">${escape(card.name)} · ${escape(card.id)}</option>`).join("");
    ["Demon Form", "Sword Boomerang", "Clash"].forEach((id, i) => { el(`rankingOffer${i}`).innerHTML = options; el(`rankingOffer${i}`).value = id; });
    el("rankingPath").textContent = `本机文件：${state.path}`;
    error(); render(); preview();
  } catch (e) { error(`读取失败：${e.message}`); }
}
el("rankingSave").addEventListener("click", async () => {
  if (!loaded || saving) return;
  const snapshot = payload(); saving = true; updateStatus();
  try {
    const state = await request("/api/card-ranking", {ranking: snapshot, base_revision: revision});
    savedOrder = [...state.ranking.order]; revision = state.revision; error();
  } catch (e) { error(`保存失败，草稿仍保留在页面：${e.message}`); }
  finally { saving = false; updateStatus(); }
});
el("rankingSaveSide").addEventListener("click", () => el("rankingSave").click());
el("rankingReload").addEventListener("click", () => { if (!saving && (!dirty() || confirm("放弃当前未保存的改动，读取本机文件？"))) load(); });
el("rankingExport").addEventListener("click", () => {
  if (!loaded) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload(), null, 2) + "\n"], {type: "application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = "ironclad-card-ranking.json";
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
el("rankingImport").addEventListener("click", () => { if (loaded) el("rankingFile").click(); });
el("rankingFile").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    if (file.size > 65536) throw new Error("文件不能超过 64 KiB");
    const result = await request("/api/card-ranking/validate", JSON.parse(await file.text()));
    replaceOrder(result.order);
  } catch (e) { error(`导入失败，当前排名未变：${e.message}`); }
  finally { event.target.value = ""; }
});
window.addEventListener("beforeunload", event => { if (dirty() || saving) { event.preventDefault(); event.returnValue = ""; } });
load();
