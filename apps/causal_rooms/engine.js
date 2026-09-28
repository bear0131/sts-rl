(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.CausalRoomsEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const STAGE_HP = 20;
  const STAGES = [
    { kind: "odd", chapter: 0, enemyHps: [4], minLoss: 9 },
    { kind: "reactive", chapter: 0, enemyHps: [4], minLoss: 9 },
    { kind: "mirror", chapter: 0, enemyHps: [5], minLoss: 2 },
    ...Array.from({ length: 6 }, (_, index) => ({
      kind: "charge", chapter: 1, enemyHps: [1], minLoss: 0, lesson: true, variant: index,
    })),
    { kind: "charge", chapter: 1, enemyHps: [3], minLoss: 2 },
    ...[[1, 2], [1, 3], [2, 3], [2, 1], [3, 1], [3, 2]].map((hps, index) => ({
      kind: "bond", chapter: 2, enemyHps: hps, minLoss: Math.min(...hps) === 1
        ? Math.max(...hps) === 2 ? 2 : 3 : 5,
      lesson: true, variant: index,
    })),
    { kind: "bond", chapter: 2, enemyHps: [1, 3], minLoss: 5, linked: true },
    ...Array.from({ length: 6 }, (_, index) => ({
      kind: "shield", chapter: 3, enemyHps: [2, 2], minLoss: 4, lesson: true, variant: index,
    })),
    { kind: "shield", chapter: 3, enemyHps: [2, 2], minLoss: 5, guarded: true },
  ];

  function startBattle(stage, hp, retried = false) {
    return {
      turn: 1,
      startHp: hp,
      loss: 0,
      usedAction: false,
      firstAction: null,
      playerBonus: 0,
      chargeReady: false,
      enemies: stage.enemyHps.map((value, index) => ({
        id: index, hp: value, maxHp: value, bonus: 0, pendingBonus: 0,
        shield: 0, pendingShield: 0,
      })),
      history: [],
      events: [],
      retried,
      perfect: false,
    };
  }

  function createRun(savedCompletions = []) {
    const completed = [];
    if (Array.isArray(savedCompletions)) {
      for (const entry of savedCompletions.slice(0, STAGES.length)) {
        if (!entry || typeof entry.perfect !== "boolean") break;
        completed.push({ perfect: entry.perfect,
          firstAction: ["attack", "charge", "wait"].includes(entry.firstAction)
            ? entry.firstAction : null,
          loss: Number.isFinite(entry.loss) && entry.loss >= 0 ? entry.loss : null });
      }
    }
    const stageIndex = Math.min(completed.length, STAGES.length - 1);
    return {
      stageIndex,
      hp: STAGE_HP,
      maxHp: STAGE_HP,
      status: "active",
      completed,
      battle: startBattle(STAGES[stageIndex], STAGE_HP),
    };
  }

  function copyRun(run) {
    return {
      ...run,
      completed: run.completed.slice(),
      battle: {
        ...run.battle,
        enemies: run.battle.enemies.map((enemy) => ({ ...enemy })),
        history: run.battle.history.slice(),
        events: [],
      },
    };
  }

  function currentStage(run) {
    return STAGES[run.stageIndex];
  }

  function win(next) {
    next.status = "won";
    next.battle.perfect = !next.battle.retried && next.battle.loss === currentStage(next).minLoss;
    const result = { perfect: next.battle.perfect, firstAction: next.battle.firstAction,
      loss: next.battle.loss };
    const previous = next.completed[next.stageIndex];
    if (!previous || result.perfect || !previous.perfect) next.completed[next.stageIndex] = result;
    next.battle.events.push({ type: "win", perfect: next.battle.perfect });
    return next;
  }

  function attack(run, targetIndex) {
    if (run.status !== "active" || run.battle.usedAction) return { run, accepted: false };
    const enemy = run.battle.enemies[targetIndex];
    if (!enemy || enemy.hp <= 0) return { run, accepted: false };
    const next = copyRun(run);
    const battle = next.battle;
    const target = battle.enemies[targetIndex];
    const amount = 1 + battle.playerBonus + (battle.chargeReady ? 2 : 0);
    const blocked = Math.min(target.shield, amount);
    target.shield -= blocked;
    const damage = Math.min(target.hp, amount - blocked);
    target.hp -= damage;
    battle.usedAction = true;
    battle.chargeReady = false;
    if (battle.firstAction === null) battle.firstAction = "attack";
    if (["reactive", "mirror"].includes(currentStage(next).kind)) target.pendingBonus += 2 * damage;
    if (currentStage(next).guarded && target.hp > 0) target.pendingShield += damage;
    battle.events.push({ type: blocked ? "blocked" : "hit", target: targetIndex, amount: damage });
    if (target.hp === 0 && currentStage(next).linked) {
      battle.enemies.forEach((other) => {
        if (other.hp > 0) other.pendingBonus += 3;
      });
      battle.events.push({ type: "bond", target: targetIndex });
    }
    if (battle.enemies.every((other) => other.hp === 0)) win(next);
    return { run: next, accepted: true };
  }

  function charge(run) {
    if (run.status !== "active" || currentStage(run).kind !== "charge" || run.battle.usedAction) {
      return { run, accepted: false };
    }
    const next = copyRun(run);
    next.battle.usedAction = true;
    next.battle.chargeReady = true;
    if (next.battle.firstAction === null) next.battle.firstAction = "charge";
    next.battle.events.push({ type: "charge" });
    return { run: next, accepted: true };
  }

  function endTurn(run) {
    if (run.status !== "active") return { run, accepted: false };
    const next = copyRun(run);
    const battle = next.battle;
    const stage = currentStage(next);
    if (battle.firstAction === null) battle.firstAction = "wait";
    let received = 0;
    battle.enemies.forEach((enemy) => {
      if (enemy.hp <= 0) return;
      const damage = stage.kind === "odd" ? 2 * battle.turn - 1 :
        stage.kind === "charge" ? 2 : 1 + enemy.bonus;
      received += damage;
      battle.events.push({ type: "enemy-hit", target: enemy.id, amount: damage });
    });
    const lost = Math.min(next.hp, received);
    next.hp -= lost;
    battle.loss += lost;
    battle.history.push(lost);
    if (next.hp === 0) {
      next.status = "dead";
      return { run: next, accepted: true };
    }
    if (stage.kind === "mirror") battle.playerBonus += 2 * received;
    battle.enemies.forEach((enemy) => {
      enemy.bonus += enemy.pendingBonus;
      enemy.pendingBonus = 0;
      enemy.shield = enemy.pendingShield;
      enemy.pendingShield = 0;
    });
    battle.usedAction = false;
    battle.turn += 1;
    return { run: next, accepted: true };
  }

  function continueRun(run) {
    if (run.status !== "won") return run;
    const next = copyRun(run);
    next.stageIndex += 1;
    if (next.stageIndex >= STAGES.length) {
      next.status = "complete";
      return next;
    }
    next.hp = STAGE_HP;
    next.maxHp = STAGE_HP;
    next.battle = startBattle(currentStage(next), STAGE_HP);
    next.status = "active";
    return next;
  }

  function selectStage(run, index) {
    if (!Number.isInteger(index) || index < 0 || index >= STAGES.length ||
      index > run.completed.length) return run;
    const next = copyRun(run);
    const retried = index === run.stageIndex && ["active", "dead"].includes(run.status);
    next.stageIndex = index;
    next.hp = STAGE_HP;
    next.maxHp = STAGE_HP;
    next.battle = startBattle(STAGES[index], STAGE_HP, retried);
    next.status = "active";
    return next;
  }

  function retry(run) {
    if (run.status !== "dead") return run;
    const next = copyRun(run);
    next.hp = STAGE_HP;
    next.maxHp = STAGE_HP;
    next.battle = startBattle(currentStage(next), STAGE_HP, true);
    next.status = "active";
    return next;
  }

  return { STAGES, STAGE_HP, createRun, currentStage, attack, charge, endTurn,
    continueRun, retry, selectStage };
});
