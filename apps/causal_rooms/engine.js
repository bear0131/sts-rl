(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.CausalRoomsEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const CHAPTER_HP = [32, 20, 42, 46];
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

  function createRun() {
    return {
      stageIndex: 0,
      hp: CHAPTER_HP[0],
      maxHp: CHAPTER_HP[0],
      status: "active",
      completed: [],
      battle: startBattle(STAGES[0], CHAPTER_HP[0]),
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
    next.hp = Math.max(0, next.hp - received);
    battle.loss += received;
    battle.history.push(received);
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
    next.completed.push({ perfect: next.battle.perfect, firstAction: next.battle.firstAction,
      loss: next.battle.loss });
    next.stageIndex += 1;
    if (next.stageIndex >= STAGES.length) {
      next.status = "complete";
      return next;
    }
    const stage = currentStage(next);
    if (stage.chapter !== currentStage(run).chapter) {
      next.maxHp = CHAPTER_HP[stage.chapter];
      next.hp = next.maxHp;
    }
    next.battle = startBattle(stage, next.hp);
    next.status = "active";
    return next;
  }

  function retry(run) {
    if (run.status !== "dead") return run;
    const next = copyRun(run);
    next.hp = Math.min(next.maxHp, Math.max(next.battle.startHp, Math.ceil(next.maxHp * 0.6)));
    next.battle = startBattle(currentStage(next), next.hp, true);
    next.status = "active";
    return next;
  }

  return { STAGES, CHAPTER_HP, createRun, currentStage, attack, charge, endTurn,
    continueRun, retry };
});
