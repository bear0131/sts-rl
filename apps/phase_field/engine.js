(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.PhaseFieldEngine = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const PHASE_COUNT = 6;
  const ACTIONS = ["tilt", "veer", "exchange", "echo", "flow"];
  const LOCAL_ACTIONS = ACTIONS.filter((action) => action !== "flow");

  const ROOM_DEFINITIONS = [
    {
      id: "opposition",
      nodePhases: [5, 2, 5, 2],
      spins: [1, 1, 1, 1],
      edges: [
        [0, 1, 3, 1],
        [1, 2, 4, 1],
        [2, 3, 5, -1],
        [3, 0, 0, 1],
      ],
      objective: "opposition",
      target: [0, 3, 0, 3],
      stableRequired: 1,
      minimumPulse: 1,
      maxPulses: 6,
      layoutShift: 0,
    },
    {
      id: "traveling-wave",
      nodePhases: [5, 0, 1, 2, 3],
      spins: [1, 1, 1, 1, 1],
      edges: [
        [0, 1, 3, 1],
        [1, 2, 4, -1],
        [2, 3, 5, 1],
        [3, 4, 0, 1],
        [4, 0, 2, -1],
      ],
      objective: "wave",
      target: [0, 1, 2, 3, 4],
      stableRequired: 1,
      minimumPulse: 1,
      maxPulses: 7,
      layoutShift: 1,
    },
    {
      id: "delayed-mirror",
      nodePhases: [2, 4, 1, 4, 2],
      spins: [1, 1, -1, 1, 1],
      edges: [
        [0, 1, 1, 1],
        [4, 3, 1, 1],
        [1, 2, 2, 1],
        [3, 2, 2, 1],
      ],
      objective: "mirror-lock",
      target: [3, 0, 1, 0, 3],
      stableRequired: 1,
      minimumPulse: 1,
      maxPulses: 7,
      layoutShift: -1,
    },
    {
      id: "grammar-boss",
      boss: true,
      nodePhases: [5, 5, 0, 1, 2, 3],
      spins: [1, 1, 1, 1, 1, 1],
      edges: [
        [0, 1, 2, 1],
        [1, 2, 3, 1],
        [2, 3, 4, -1],
        [3, 4, 5, 1],
        [4, 5, 0, -1],
        [5, 0, 1, 1],
      ],
      objective: "boss-orbit",
      target: [5, 0, 1, 2, 3, 4],
      stableRequired: 1,
      minimumPulse: 1,
      maxPulses: 9,
      layoutShift: 1,
    },
  ];

  function modulo(value, base = PHASE_COUNT) {
    return ((value % base) + base) % base;
  }

  function normalizeSeed(seed) {
    const parsed = Number(seed);
    return Number.isFinite(parsed) ? parsed >>> 0 : 20260928;
  }

  function mulberry32(seed) {
    let state = normalizeSeed(seed);
    return function random() {
      state += 0x6d2b79f5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffled(values, random) {
    const result = values.slice();
    for (let index = result.length - 1; index > 0; index -= 1) {
      const other = Math.floor(random() * (index + 1));
      [result[index], result[other]] = [result[other], result[index]];
    }
    return result;
  }

  function copyRoom(room) {
    return {
      ...room,
      nodes: room.nodes.map((node) => ({ ...node })),
      edges: room.edges.map((edge) => ({ ...edge })),
      echoes: room.echoes.map((echo) => ({ ...echo })),
      target: room.target.slice(),
      lockedActions: room.lockedActions.slice(),
      lastEvents: room.lastEvents.map((event) => ({ ...event })),
      trace: room.trace.map((frame) => ({
        ...frame,
        nodes: frame.nodes.map((node) => ({ ...node })),
        echoes: frame.echoes.map((echo) => ({ ...echo })),
        activeEdgeIds: frame.activeEdgeIds.slice(),
        echoNodeIds: frame.echoNodeIds.slice(),
      })),
    };
  }

  function createRoom(roomIndex) {
    const definition = ROOM_DEFINITIONS[roomIndex];
    if (!definition) {
      throw new RangeError(`Unknown room index ${roomIndex}`);
    }
    return {
      id: definition.id,
      roomIndex,
      boss: Boolean(definition.boss),
      pulse: 0,
      globalPhase: 0,
      layoutOffset: 0,
      maxPulses: definition.maxPulses,
      minimumPulse: definition.minimumPulse,
      stableRequired: definition.stableRequired,
      objective: definition.objective,
      target: definition.target.slice(),
      layoutShift: definition.layoutShift,
      stableCount: 0,
      outcome: "active",
      nodes: definition.nodePhases.map((phase, index) => ({
        id: index,
        phase: modulo(phase),
        spin: definition.spins[index],
      })),
      edges: definition.edges.map(([from, to, gate, twist], index) => ({
        id: index,
        from,
        to,
        gate,
        twist,
      })),
      echoes: [],
      lockedActions: [],
      lastEvents: [],
      trace: [],
    };
  }

  function createRun(seed = 20260928) {
    const normalizedSeed = normalizeSeed(seed);
    const random = mulberry32(normalizedSeed);
    const glyphIds = shuffled([0, 1, 2, 3, 4], random);
    return {
      seed: normalizedSeed,
      paletteOffset: Math.floor(random() * 360),
      glyphOrder: shuffled(ACTIONS, random),
      glyphMap: Object.fromEntries(ACTIONS.map((action, index) => [action, glyphIds[index]])),
      roomIndex: 0,
      completedRooms: [],
      outcome: "active",
      room: createRoom(0),
    };
  }

  function outgoingEdge(room, nodeIndex) {
    return room.edges.find((edge) => edge.from === nodeIndex) || null;
  }

  function isActionAvailable(room, action) {
    if (room.outcome !== "active" || !ACTIONS.includes(action)) {
      return false;
    }
    return action === "flow" || !room.lockedActions.includes(action);
  }

  function applyLocalAction(room, action, nodeIndex) {
    const next = copyRoom(room);
    next.lastEvents = [];
    if (!LOCAL_ACTIONS.includes(action) || !isActionAvailable(next, action)) {
      return { room: next, accepted: false, reason: "unavailable" };
    }
    if (!Number.isInteger(nodeIndex) || nodeIndex < 0 || nodeIndex >= next.nodes.length) {
      return { room: next, accepted: false, reason: "selection" };
    }

    const node = next.nodes[nodeIndex];
    if (action === "tilt") {
      node.phase = modulo(node.phase + 1);
      next.lastEvents.push({ type: "phase", node: nodeIndex, delta: 1 });
    } else if (action === "veer") {
      node.spin *= -1;
      next.lastEvents.push({ type: "spin", node: nodeIndex, spin: node.spin });
    } else if (action === "exchange") {
      const edge = outgoingEdge(next, nodeIndex);
      if (!edge) {
        return { room: next, accepted: false, reason: "connection" };
      }
      const target = next.nodes[edge.to];
      [node.phase, target.phase] = [target.phase, node.phase];
      next.lastEvents.push({ type: "exchange", from: nodeIndex, to: edge.to });
    } else if (action === "echo") {
      next.echoes.push({
        id: `${next.pulse}:${nodeIndex}:${next.echoes.length}`,
        node: nodeIndex,
        due: next.pulse + 2,
        delta: 2,
      });
      next.lastEvents.push({ type: "echo-seeded", node: nodeIndex, due: next.pulse + 2 });
    }

    next.lockedActions.push(action);
    return { room: next, accepted: true, reason: null };
  }

  function objectiveSatisfied(room) {
    if (room.pulse < room.minimumPulse) {
      return false;
    }
    return room.nodes.every((node, index) => node.phase === room.target[index]);
  }

  function traceFrame(room, type, activeEdgeIds = [], echoNodeIds = []) {
    return {
      type,
      pulse: room.pulse,
      globalPhase: room.globalPhase,
      layoutOffset: room.layoutOffset,
      nodes: room.nodes.map((node) => ({ phase: node.phase, spin: node.spin })),
      echoes: room.echoes.map((echo) => ({ ...echo })),
      activeEdgeIds: activeEdgeIds.slice(),
      echoNodeIds: echoNodeIds.slice(),
    };
  }

  function advanceWorld(room) {
    const next = copyRoom(room);
    next.lastEvents = [];
    if (!isActionAvailable(next, "flow")) {
      return { room: next, accepted: false, reason: "unavailable" };
    }
    next.trace = [traceFrame(next, "start")];

    next.pulse += 1;
    next.globalPhase = modulo(next.globalPhase + 1);
    next.nodes.forEach((node) => {
      node.phase = modulo(node.phase + node.spin);
    });
    next.trace.push(traceFrame(next, "drift"));

    const propagation = new Array(next.nodes.length).fill(0);
    const grammarSign = next.boss && next.pulse % 2 === 1 ? -1 : 1;
    const activeEdgeIds = [];
    next.edges.forEach((edge) => {
      if (edge.gate === next.globalPhase) {
        activeEdgeIds.push(edge.id);
        const delta = edge.twist * grammarSign;
        propagation[edge.to] += delta;
        next.lastEvents.push({
          type: "propagation",
          from: edge.from,
          to: edge.to,
          delta,
        });
      }
    });
    propagation.forEach((delta, nodeIndex) => {
      if (delta !== 0) {
        next.nodes[nodeIndex].phase = modulo(next.nodes[nodeIndex].phase + delta);
      }
    });
    next.trace.push(traceFrame(next, "propagate", activeEdgeIds));

    const pendingEchoes = [];
    next.echoes.forEach((echo) => {
      if (echo.due <= next.pulse) {
        next.nodes[echo.node].phase = modulo(next.nodes[echo.node].phase + echo.delta);
        next.lastEvents.push({ type: "echo-fired", node: echo.node, delta: echo.delta });
      } else {
        pendingEchoes.push(echo);
      }
    });
    next.echoes = pendingEchoes;
    next.trace.push(traceFrame(next, "echo", [],
      next.lastEvents.filter((event) => event.type === "echo-fired").map((event) => event.node)));
    next.layoutOffset = modulo(
      next.layoutOffset + next.layoutShift,
      next.nodes.length,
    );
    next.lockedActions = [];
    next.trace.push(traceFrame(next, "reposition"));

    next.stableCount = objectiveSatisfied(next) ? next.stableCount + 1 : 0;
    if (next.stableCount >= next.stableRequired) {
      next.outcome = "success";
    } else if (next.pulse >= next.maxPulses) {
      next.outcome = "fracture";
    }
    return { room: next, accepted: true, reason: null };
  }

  function performAction(room, action, nodeIndex) {
    return action === "flow"
      ? advanceWorld(room)
      : applyLocalAction(room, action, nodeIndex);
  }

  function enterNextRoom(run) {
    if (run.room.outcome !== "success") {
      return run;
    }
    const nextIndex = run.roomIndex + 1;
    const completedRooms = run.completedRooms.concat(run.roomIndex);
    if (nextIndex >= ROOM_DEFINITIONS.length) {
      return {
        ...run,
        completedRooms,
        outcome: "success",
      };
    }
    return {
      ...run,
      roomIndex: nextIndex,
      completedRooms,
      room: createRoom(nextIndex),
    };
  }

  function restartRoom(run) {
    return {
      ...run,
      room: createRoom(run.roomIndex),
      outcome: "active",
    };
  }

  function targetPhases(room) {
    return room.target.slice();
  }

  return {
    ACTIONS,
    LOCAL_ACTIONS,
    PHASE_COUNT,
    ROOM_DEFINITIONS,
    advanceWorld,
    applyLocalAction,
    createRoom,
    createRun,
    enterNextRoom,
    isActionAvailable,
    modulo,
    objectiveSatisfied,
    performAction,
    restartRoom,
    targetPhases,
  };
});
