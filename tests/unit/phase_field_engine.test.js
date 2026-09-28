"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const Engine = require("../../apps/phase_field/engine.js");

test("a seed fixes the anonymous visual vocabulary", () => {
  const first = Engine.createRun(73);
  const second = Engine.createRun(73);
  const other = Engine.createRun(74);
  assert.deepEqual(first.glyphOrder, second.glyphOrder);
  assert.deepEqual(first.glyphMap, second.glyphMap);
  assert.equal(first.paletteOffset, second.paletteOffset);
  assert.notDeepEqual(
    [first.glyphOrder, first.glyphMap, first.paletteOffset],
    [other.glyphOrder, other.glyphMap, other.paletteOffset],
  );
});

test("the first two rooms admit a one-pulse shallow strategy", () => {
  for (const roomIndex of [0, 1]) {
    const result = Engine.advanceWorld(Engine.createRoom(roomIndex));
    assert.equal(result.accepted, true);
    assert.equal(result.room.outcome, "success");
    assert.equal(result.room.pulse, 1);
  }
});

test("the third room admits one local intervention followed by one pulse", () => {
  const local = Engine.applyLocalAction(Engine.createRoom(2), "tilt", 2);
  assert.equal(local.accepted, true);
  const result = Engine.advanceWorld(local.room);
  assert.equal(result.room.outcome, "success");
  assert.equal(result.room.pulse, 1);
});

test("the boss rewards withholding learned local interventions", () => {
  const direct = Engine.advanceWorld(Engine.createRoom(3));
  assert.equal(direct.room.outcome, "success");

  const overControlled = Engine.applyLocalAction(Engine.createRoom(3), "tilt", 0);
  assert.equal(overControlled.accepted, true);
  const deviated = Engine.advanceWorld(overControlled.room);
  assert.equal(deviated.room.outcome, "active");
});

test("local actions lock once while global flow remains repeatable", () => {
  const room = Engine.createRoom(2);
  const first = Engine.applyLocalAction(room, "tilt", 2);
  assert.equal(first.accepted, true);
  assert.equal(Engine.isActionAvailable(first.room, "tilt"), false);
  assert.equal(Engine.isActionAvailable(first.room, "flow"), true);

  const repeated = Engine.applyLocalAction(first.room, "tilt", 2);
  assert.equal(repeated.accepted, false);
  const evolved = Engine.advanceWorld(first.room);
  assert.equal(Engine.isActionAvailable(evolved.room, "tilt"), false);
  assert.equal(evolved.room.outcome, "success");
});

test("an echo is history-bound and fires only through global evolution", () => {
  const seeded = Engine.applyLocalAction(Engine.createRoom(2), "echo", 0);
  assert.equal(seeded.room.echoes.length, 1);
  assert.equal(seeded.room.pulse, 0);

  const once = Engine.advanceWorld(seeded.room).room;
  assert.equal(once.echoes.length, 1);
  assert.equal(once.pulse, 1);

  const phaseBefore = once.nodes[0].phase;
  const twice = Engine.advanceWorld(once).room;
  assert.equal(twice.echoes.length, 0);
  assert.equal(twice.pulse, 2);
  assert.equal(
    twice.nodes[0].phase,
    Engine.modulo(phaseBefore + once.nodes[0].spin + 2),
  );
});
