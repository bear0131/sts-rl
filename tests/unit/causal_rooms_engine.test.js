"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const Game = require("../../apps/causal_rooms/engine.js");

function turn(run, action, target = 0) {
  if (action === "attack") run = Game.attack(run, target).run;
  if (action === "charge") run = Game.charge(run).run;
  return run.status === "active" ? Game.endTurn(run).run : run;
}

function finish(run, actions) {
  for (const [action, target] of actions) run = turn(run, action, target);
  assert.equal(run.status, "won");
  return run;
}

test("the three original encounters preserve the intended contrast", () => {
  let run = Game.createRun();
  for (const expected of [9, 9]) {
    run = finish(run, Array.from({ length: 4 }, () => ["attack", 0]));
    assert.equal(run.battle.loss, expected);
    assert.equal(run.battle.perfect, true);
    assert.equal(run.completed.length, run.stageIndex + 1);
    run = Game.continueRun(run);
    assert.equal(run.hp, Game.STAGE_HP);
  }
  const conventional = finish(run, [["attack", 0], ["attack", 0], ["attack", 0]]);
  assert.equal(conventional.battle.loss, 4);
  assert.equal(conventional.battle.perfect, false);
  assert.equal(conventional.hp, 16);
  const continued = Game.continueRun(conventional);
  assert.equal(continued.stageIndex, 3);
  assert.equal(continued.hp, Game.STAGE_HP);
  assert.equal(continued.completed[2].perfect, false);
  const patient = finish(run, [["wait"], ["wait"], ["attack", 0]]);
  assert.equal(patient.battle.loss, 2);
  assert.equal(patient.battle.perfect, true);
  assert.deepEqual(patient.battle.history, [1, 1]);
});

test("a lethal experiment records actual HP lost and retries locally", () => {
  let run = Game.createRun();
  for (const _ of [0, 1]) {
    run = finish(run, Array.from({ length: 4 }, () => ["attack", 0]));
    run = Game.continueRun(run);
  }
  for (let index = 0; index < 30 && run.status === "active"; index += 1) {
    run = turn(run, "wait");
  }
  assert.equal(run.status, "dead");
  assert.equal(run.battle.loss, Game.STAGE_HP);
  assert.equal(run.battle.history.reduce((sum, value) => sum + value, 0), Game.STAGE_HP);
  const retried = Game.retry(run);
  assert.equal(retried.status, "active");
  assert.equal(retried.stageIndex, 2);
  assert.equal(retried.completed.length, 2);
  assert.ok(retried.hp > 0);
  assert.equal(retried.hp, Game.STAGE_HP);
  assert.equal(retried.battle.retried, true);
});

test("stages unlock in order, can be selected, and preserve best completion", () => {
  let run = Game.createRun();
  assert.equal(Game.selectStage(run, 1), run);
  run = finish(run, Array.from({ length: 4 }, () => ["attack", 0]));
  assert.equal(run.completed.length, 1);
  const saved = JSON.parse(JSON.stringify(run.completed));
  const resumed = Game.createRun(saved);
  assert.equal(resumed.stageIndex, 1);
  assert.equal(resumed.hp, Game.STAGE_HP);
  assert.equal(Game.selectStage(resumed, 2), resumed);
  const replay = Game.selectStage(resumed, 0);
  assert.equal(replay.stageIndex, 0);
  assert.equal(replay.hp, Game.STAGE_HP);
  const worse = finish(replay, [["wait"], ["attack", 0], ["attack", 0],
    ["attack", 0], ["attack", 0]]);
  assert.equal(worse.battle.perfect, false);
  assert.equal(worse.completed[0].perfect, true);
  assert.equal(Game.selectStage(worse, 1).stageIndex, 1);
});

test("a full low-loss route can finish without requiring zero damage", () => {
  let run = Game.createRun();
  while (run.status !== "complete") {
    const stage = Game.currentStage(run);
    if (stage.kind === "odd" || stage.kind === "reactive") {
      run = finish(run, Array.from({ length: 4 }, () => ["attack", 0]));
    } else if (stage.kind === "mirror") {
      run = finish(run, [["wait"], ["wait"], ["attack", 0]]);
    } else if (stage.kind === "charge") {
      run = finish(run, stage.lesson ? [["attack", 0]] : [["charge"], ["attack", 0]]);
    } else if (stage.kind === "bond") {
      const first = stage.linked ? stage.enemyHps.indexOf(3) :
        stage.enemyHps.indexOf(Math.min(...stage.enemyHps));
      const second = 1 - first;
      run = finish(run, [
        ...Array.from({ length: stage.enemyHps[first] }, () => ["attack", first]),
        ...Array.from({ length: stage.enemyHps[second] }, () => ["attack", second]),
      ]);
    } else {
      run = finish(run, stage.guarded
        ? [["attack", 0], ["attack", 1], ["attack", 0], ["attack", 1]]
        : [["attack", 0], ["attack", 0], ["attack", 1], ["attack", 1]]);
    }
    assert.equal(run.battle.perfect, true, `stage ${run.stageIndex}`);
    run = Game.continueRun(run);
  }
  assert.equal(run.completed.length, Game.STAGES.length);
  assert.equal(run.completed.every((entry) => entry.perfect), true);
});

test("counterexamples penalize the rehearsed action but allow recovery", () => {
  const go = (index, actions) => {
    let run = Game.createRun();
    run.stageIndex = index;
    run.hp = run.maxHp = Game.STAGE_HP;
    run.battle = {
      ...run.battle,
      startHp: Game.STAGE_HP,
      enemies: Game.STAGES[index].enemyHps.map((hp, id) => ({ id, hp, maxHp: hp,
        bonus: 0, pendingBonus: 0, shield: 0, pendingShield: 0 })),
    };
    return finish(run, actions);
  };
  assert.equal(go(9, [["attack", 0], ["attack", 0], ["attack", 0]]).battle.loss, 4);
  assert.equal(go(16, [["attack", 0], ["attack", 1], ["attack", 1], ["attack", 1]]).battle.loss, 9);
  assert.equal(go(23, [["attack", 0], ["attack", 0], ["attack", 0],
    ["attack", 1], ["attack", 1], ["attack", 1]]).battle.loss, 7);
});
