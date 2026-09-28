(function () {
  "use strict";

  const Game = window.CausalRoomsEngine;
  const playerScale = Math.max(...Game.CHAPTER_HP);
  const enemyScale = Math.max(...Game.STAGES.flatMap((stage) => stage.enemyHps));
  const route = document.getElementById("route");
  const chapterMark = document.getElementById("chapterMark");
  const player = document.getElementById("player");
  const enemies = document.getElementById("enemies");
  const history = document.getElementById("history");
  const attackButton = document.getElementById("attack");
  const chargeButton = document.getElementById("charge");
  const endButton = document.getElementById("endTurn");
  const outcome = document.getElementById("outcome");
  const resultSymbol = document.getElementById("resultSymbol");
  const continueButton = document.getElementById("continue");
  const restartButton = document.getElementById("restart");

  const state = { run: Game.createRun(), selected: 0 };
  const buffIcon = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 27V5m-7 8 7-8 7 8M8 23h16"/></svg>';
  const shieldIcon = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 3 27 8v9c0 7-5 10-11 13C10 27 5 24 5 17V8Z"/></svg>';
  const perfectIcon = '<svg viewBox="0 0 120 120"><path d="M60 8 70 48 111 60 70 72 60 112 49 72 9 60 49 48Z"/><circle cx="60" cy="60" r="14"/></svg>';
  const clearIcon = '<svg viewBox="0 0 120 120"><path d="M60 13a47 47 0 1 1-34 80"/><path d="M60 31a29 29 0 1 1-21 49"/></svg>';
  const deadIcon = '<svg viewBox="0 0 120 120"><path d="M21 20 50 49 40 77 62 106M98 17 69 45 79 66 53 92M12 72 46 64 57 41 82 24"/></svg>';

  function light(value, max) {
    return `${Math.round(18 + 61 * Math.max(0, Math.min(1, value / max)))}%`;
  }

  function pulse(value) {
    return `${Math.max(420, 1820 - Math.min(value, 10) * 145)}ms`;
  }

  function powerLight(value) {
    return `${Math.round(36 + 43 * Math.max(0, Math.min(1, value / 9)))}%`;
  }

  function unit(hp, maxHp, hue, kind, sigils = "", struck = false) {
    return `<div class="unit ${kind} ${struck ? "struck" : ""}" style="--hue:${hue};--light:${light(hp, maxHp)}">${sigils}</div>`;
  }

  function renderRoute() {
    route.innerHTML = Game.STAGES.map((stage, index) => {
      const status = index < state.run.completed.length
        ? state.run.completed[index].perfect ? "perfect" : "cleared"
        : index === state.run.stageIndex ? "current" : "";
      const start = index > 0 && stage.chapter !== Game.STAGES[index - 1].chapter
        ? "chapter-start" : "";
      return `<i class="route-dot ${status} ${start}" aria-hidden="true"></i>`;
    }).join("");
    const chapter = state.run.status === "complete" ? 3 : Game.currentStage(state.run).chapter;
    chapterMark.className = `chapter-mark c${chapter}`;
  }

  function renderBattle() {
    const run = state.run;
    const battle = run.battle;
    const stage = Game.currentStage(run) || Game.STAGES[Game.STAGES.length - 1];
    const preview = 1 + battle.playerBonus + (battle.chargeReady ? 2 : 0);
    const playerBuff = battle.playerBonus > 0 || battle.chargeReady
      ? `<span class="sigil" aria-hidden="true" style="--power-light:${powerLight(preview)};--power-speed:${pulse(preview)}">${buffIcon}</span>` : "";
    const playerStruck = battle.events.some((event) => event.type === "enemy-hit");
    player.innerHTML = unit(run.hp, playerScale, 204, "player-unit", playerBuff, playerStruck);
    player.setAttribute("aria-label", `玩家生命 ${run.hp}，当前攻击 ${preview}`);

    if (state.selected >= battle.enemies.length || battle.enemies[state.selected].hp <= 0) {
      state.selected = battle.enemies.findIndex((enemy) => enemy.hp > 0);
    }
    const cards = battle.enemies.map((enemy, index) => {
      const buff = enemy.bonus + enemy.pendingBonus > 0
        ? `<span class="sigil" aria-hidden="true" style="--power-light:55%;--power-speed:1200ms">${buffIcon}</span>` : "";
      const shield = enemy.shield > 0 || enemy.pendingShield > 0
        ? `<span class="sigil shield-sigil" aria-hidden="true">${shieldIcon}</span>` : "";
      const struck = battle.events.some((event) =>
        ["hit", "blocked"].includes(event.type) && event.target === index);
      return `<button type="button" class="enemy-button ${state.selected === index ? "selected" : ""}" data-enemy="${index}" aria-label="选择怪物 ${index + 1}，生命 ${enemy.hp}" ${enemy.hp <= 0 || run.status !== "active" ? "disabled" : ""}>
        ${unit(enemy.hp, enemyScale, 340 + index * 38, "enemy-unit", buff + shield, struck)}
      </button>`;
    });
    enemies.innerHTML = cards.join(stage.linked && cards.length === 2
      ? '<span class="bond-wire" aria-hidden="true"></span>' : "");
    history.innerHTML = battle.history.slice(-6).map((damage) =>
      `<i class="history-bead" style="--light:${powerLight(damage)};--speed:${pulse(damage)}" aria-hidden="true"></i>`).join("");
    history.setAttribute("aria-label", `已承受伤害：${battle.history.join("，")}`);

    const available = run.status === "active" && !battle.usedAction;
    attackButton.disabled = !available || state.selected < 0;
    attackButton.style.setProperty("--power-light", powerLight(preview));
    attackButton.style.setProperty("--power-speed", pulse(preview));
    chargeButton.hidden = stage.kind !== "charge";
    chargeButton.disabled = !available;
    endButton.disabled = run.status !== "active";
  }

  function renderOutcome() {
    const run = state.run;
    outcome.hidden = run.status === "active";
    if (outcome.hidden) return;
    const allPerfect = run.completed.every((entry) => entry.perfect);
    const variant = run.status === "won" ? run.battle.perfect ? "perfect" : "cleared"
      : run.status === "dead" ? "dead" : allPerfect ? "perfect" : "complete";
    outcome.className = `outcome ${variant}`;
    outcome.setAttribute("aria-label", variant === "perfect" ? "完美通关"
      : variant === "cleared" || variant === "complete" ? "通关，未全部完美" : "失败，可重试");
    resultSymbol.innerHTML = variant === "perfect" ? perfectIcon
      : variant === "dead" ? deadIcon : clearIcon;
    continueButton.setAttribute("aria-label", run.status === "dead" ? "重试本关"
      : run.status === "complete" ? "重新开始整局" : "进入下一关");
  }

  function render() {
    renderRoute();
    renderBattle();
    renderOutcome();
  }

  function apply(result) {
    if (!result.accepted) return;
    state.run = result.run;
    render();
  }

  enemies.addEventListener("click", (event) => {
    const target = event.target.closest(".enemy-button");
    if (!target || target.disabled) return;
    state.selected = Number(target.dataset.enemy);
    render();
  });
  attackButton.addEventListener("click", () => apply(Game.attack(state.run, state.selected)));
  chargeButton.addEventListener("click", () => apply(Game.charge(state.run)));
  endButton.addEventListener("click", () => apply(Game.endTurn(state.run)));
  continueButton.addEventListener("click", () => {
    if (state.run.status === "won") state.run = Game.continueRun(state.run);
    else if (state.run.status === "dead") state.run = Game.retry(state.run);
    else state.run = Game.createRun();
    state.selected = 0;
    render();
  });
  restartButton.addEventListener("click", () => {
    state.run = Game.createRun();
    state.selected = 0;
    render();
  });

  render();
})();
