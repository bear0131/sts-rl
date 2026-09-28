(function () {
  "use strict";

  const Engine = window.PhaseFieldEngine;
  const fieldCanvas = document.getElementById("fieldCanvas");
  const targetCanvas = document.getElementById("targetCanvas");
  const pulseCanvas = document.getElementById("pulseCanvas");
  const actionsElement = document.getElementById("actions");
  const mapElement = document.getElementById("map");
  const outcomeElement = document.getElementById("outcome");
  const outcomeAction = document.getElementById("outcomeAction");
  const resetRun = document.getElementById("resetRun");

  const state = {
    run: Engine.createRun(readSeed()),
    selectedNode: null,
    hoverNode: null,
    particles: [],
    lastFrame: performance.now(),
    flash: 0,
    replay: null,
  };
  const REPLAY_STEP_MS = 420;

  function visibleRoom() {
    const room = state.run.room;
    if (!state.replay) return room;
    return { ...room, ...state.replay.frames[state.replay.index] };
  }

  const actionPaths = [
    '<path d="M19 31 31 19 43 31 31 43Z"/><path d="M31 13v12M31 37v12M13 31h12M37 31h12"/>',
    '<path d="M14 39 31 12 47 39 31 34Z"/><path d="M31 12v22"/>',
    '<path d="M16 20h30M16 32h30"/><path d="m26 14-6 6 6 6M36 26l6 6-6 6"/>',
    '<path d="M31 13c7 0 13 6 13 13s-6 13-13 13-13-6-13-13 6-13 13-13Z"/><path d="m31 18 4 8-4 8-4-8Z"/>',
    '<path d="M15 31c8-15 24-15 32 0-8 15-24 15-32 0Z"/><path d="M31 22v18M22 31h18"/>',
  ];

  function readSeed() {
    const search = new URLSearchParams(window.location.search);
    const value = Number(search.get("seed"));
    return Number.isFinite(value) ? value : 20260928;
  }

  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width: rect.width, height: rect.height };
  }

  function nodePositions(room, width, height) {
    const centerX = width / 2;
    const centerY = height / 2 + 1.5;
    const radius = Math.min(width, height) * (room.nodes.length > 4 ? 0.29 : 0.25);
    const count = room.nodes.length;
    return room.nodes.map((_, index) => {
      const visualSlot = Engine.modulo(index + room.layoutOffset, count);
      const angle = -Math.PI / 2 + (visualSlot * Math.PI * 2) / count;
      return {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      };
    });
  }

  function phaseColor(phase, alpha = 1, offset = 0) {
    const hue = (offset + phase * 60 + 220) % 360;
    return `hsla(${hue}, 88%, 73%, ${alpha})`;
  }

  function drawPhaseScale(context, x, y, phase, radius, paletteOffset, compact = false) {
    context.save();
    context.translate(x, y);
    context.setLineDash([]);
    for (let index = 0; index < Engine.PHASE_COUNT; index += 1) {
      const angle = -Math.PI / 2 + index * Math.PI / 3;
      const inner = radius + (compact ? 2 : 5);
      const outer = inner + (compact ? 3 : 7);
      context.beginPath();
      context.moveTo(Math.cos(angle) * inner, Math.sin(angle) * inner);
      context.lineTo(Math.cos(angle) * outer, Math.sin(angle) * outer);
      context.lineWidth = index === phase ? (compact ? 2 : 3) : 1;
      context.strokeStyle = phaseColor(index, index === phase ? 1 : 0.35, paletteOffset);
      context.shadowBlur = index === phase ? (compact ? 5 : 10) : 0;
      context.shadowColor = context.strokeStyle;
      context.stroke();
    }
    context.restore();
  }

  function drawBackground(context, width, height, time) {
    const glow = context.createRadialGradient(
      width / 2,
      height / 2,
      0,
      width / 2,
      height / 2,
      Math.max(width, height) * 0.58,
    );
    glow.addColorStop(0, "rgba(105, 121, 239, .08)");
    glow.addColorStop(0.5, "rgba(69, 77, 145, .025)");
    glow.addColorStop(1, "rgba(4, 5, 13, 0)");
    context.fillStyle = glow;
    context.fillRect(0, 0, width, height);

    context.save();
    context.translate(width / 2, height / 2);
    context.rotate(time * 0.00002);
    context.strokeStyle = "rgba(168, 181, 255, .035)";
    context.lineWidth = 1;
    for (let radius = 62; radius < Math.max(width, height); radius += 62) {
      context.beginPath();
      context.arc(0, 0, radius, 0, Math.PI * 2);
      context.stroke();
    }
    context.restore();
  }

  function drawEdge(context, edge, positions, room, selectedNode) {
    const from = positions[edge.from];
    const to = positions[edge.to];
    const active = room.activeEdgeIds
      ? room.activeEdgeIds.includes(edge.id)
      : edge.gate === room.globalPhase;
    const target = Engine.targetPhases(room);
    const aligned = room.nodes[edge.from].phase === target[edge.from]
      && room.nodes[edge.to].phase === target[edge.to];
    const selected = selectedNode === edge.from || selectedNode === edge.to;
    context.save();
    context.beginPath();
    context.moveTo(from.x, from.y);
    const bend = (edge.id % 2 ? 1 : -1) * Math.min(38, Math.abs(to.x - from.x) * 0.18 + 18);
    const middleX = (from.x + to.x) / 2 + bend;
    const middleY = (from.y + to.y) / 2 - bend;
    context.quadraticCurveTo(middleX, middleY, to.x, to.y);
    context.lineWidth = active ? 2.2 : selected ? 1.8 : 1;
    context.setLineDash(active ? [] : [2, 8]);
    context.strokeStyle = active
      ? phaseColor(room.globalPhase, 0.7, 25)
      : aligned
        ? "rgba(124, 246, 211, .58)"
      : selected
        ? "rgba(202, 213, 255, .42)"
        : "rgba(163, 174, 229, .13)";
    context.shadowBlur = active || aligned ? 13 : 0;
    context.shadowColor = context.strokeStyle;
    context.stroke();
    context.restore();
  }

  function drawNode(context, node, position, room, nodeIndex, time) {
    const active = state.selectedNode === nodeIndex;
    const hovered = state.hoverNode === nodeIndex;
    const pulse = 1 + Math.sin(time * 0.002 + nodeIndex) * 0.04;
    const radius = (active ? 29 : 24) * pulse;
    const color = phaseColor(node.phase, 0.9, state.run.paletteOffset);
    const target = Engine.targetPhases(room);
    const aligned = node.phase === target[nodeIndex];

    context.save();
    context.translate(position.x, position.y);
    context.rotate((node.phase * Math.PI) / 3 + (node.spin < 0 ? 0.35 : 0));
    context.beginPath();
    for (let side = 0; side < 7; side += 1) {
      const angle = -Math.PI / 2 + (side * Math.PI * 2) / 7;
      const distance = radius * (side % 2 ? 0.86 : 1.04);
      const x = Math.cos(angle) * distance;
      const y = Math.sin(angle) * distance;
      if (side === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.closePath();
    context.fillStyle = color.replace("0.9", "0.11");
    context.strokeStyle = color;
    context.lineWidth = active ? 2.5 : hovered ? 2 : 1.4;
    context.shadowBlur = active ? 30 : hovered ? 18 : 12;
    context.shadowColor = color;
    context.fill();
    context.stroke();

    if (aligned) {
      context.beginPath();
      context.arc(0, 0, radius + 7, 0, Math.PI * 2);
      context.setLineDash([2, 4]);
      context.lineWidth = 1;
      context.strokeStyle = "rgba(124, 246, 211, .7)";
      context.shadowBlur = 12;
      context.shadowColor = "rgba(124, 246, 211, .8)";
      context.stroke();
    }

    context.rotate(-((node.phase * Math.PI) / 3 + (node.spin < 0 ? 0.35 : 0)));
    context.beginPath();
    context.arc(0, 0, radius * 0.35, 0, Math.PI * 2);
    context.fillStyle = phaseColor(node.phase + (room.globalPhase % 2), 0.7, state.run.paletteOffset + 30);
    context.fill();
    context.restore();
    drawPhaseScale(context, position.x, position.y, node.phase, radius + (aligned ? 7 : 0), state.run.paletteOffset);
    if (room.echoNodeIds?.includes(nodeIndex)) {
      context.save();
      context.beginPath();
      context.arc(position.x, position.y, radius + 18, 0, Math.PI * 2);
      context.strokeStyle = "rgba(255, 180, 236, .9)";
      context.lineWidth = 3;
      context.shadowBlur = 18;
      context.shadowColor = context.strokeStyle;
      context.stroke();
      context.restore();
    }
  }

  function drawField(time) {
    const { context, width, height } = resizeCanvas(fieldCanvas);
    const room = visibleRoom();
    context.clearRect(0, 0, width, height);
    drawBackground(context, width, height, time);
    const positions = nodePositions(room, width, height);
    room.edges.forEach((edge) => drawEdge(context, edge, positions, room, state.selectedNode));

    if (room.echoes.length) {
      room.echoes.forEach((echo) => {
        const position = positions[echo.node];
        context.save();
        context.beginPath();
        context.arc(position.x, position.y, 37 + Math.sin(time * 0.003) * 3, 0, Math.PI * 2);
        context.setLineDash([3, 8]);
        context.lineWidth = 1;
        context.strokeStyle = "rgba(255, 180, 236, .54)";
        context.stroke();
        context.restore();
      });
    }
    room.nodes.forEach((node, index) => drawNode(context, node, positions[index], room, index, time));

    const center = { x: width / 2, y: height / 2 };
    context.save();
    context.beginPath();
    context.arc(center.x, center.y, 4 + room.stableCount * 2, 0, Math.PI * 2);
    context.fillStyle = room.outcome === "fracture" ? "#ff90ad" : "#e1e5ff";
    context.shadowBlur = 24;
    context.shadowColor = context.fillStyle;
    context.fill();
    context.restore();
  }

  function drawTarget() {
    const { context, width, height } = resizeCanvas(targetCanvas);
    const room = visibleRoom();
    context.clearRect(0, 0, width, height);
    const target = Engine.targetPhases(room);
    const centerX = width / 2;
    const centerY = height / 2;
    const positions = nodePositions(room, width, height);
    const radius = Math.min(width, height) * 0.28;
    context.save();
    context.strokeStyle = "rgba(190, 201, 255, .14)";
    context.lineWidth = 0.9;
    context.beginPath();
    context.arc(centerX, centerY, radius * 1.8, 0, Math.PI * 2);
    context.stroke();

    room.edges.forEach((edge) => {
      const from = positions[edge.from];
      const to = positions[edge.to];
      const bend = (edge.id % 2 ? 1 : -1) * Math.min(12, Math.abs(to.x - from.x) * 0.18 + 6);
      context.beginPath();
      context.moveTo(from.x, from.y);
      context.quadraticCurveTo(
        (from.x + to.x) / 2 + bend,
        (from.y + to.y) / 2 - bend,
        to.x,
        to.y,
      );
      context.setLineDash([2, 3]);
      context.lineWidth = 1.1;
      context.strokeStyle = phaseColor(edge.gate, 0.34, 25);
      context.stroke();
    });

    target.forEach((phase, index) => {
      const { x, y } = positions[index];
      context.beginPath();
      context.arc(x, y, 5, 0, Math.PI * 2);
      context.fillStyle = phaseColor(phase, 0.85, state.run.paletteOffset);
      context.shadowBlur = 10;
      context.shadowColor = context.fillStyle;
      context.fill();
      context.beginPath();
      context.arc(x, y, 9, 0, Math.PI * 2);
      context.setLineDash([2, 3]);
      context.lineWidth = 1;
      context.strokeStyle = "rgba(225, 231, 255, .35)";
      context.stroke();
      drawPhaseScale(context, x, y, phase, 9, state.run.paletteOffset, true);
    });
    context.restore();
  }

  function drawPulse() {
    const { context, width, height } = resizeCanvas(pulseCanvas);
    const room = visibleRoom();
    context.clearRect(0, 0, width, height);
    const centerX = width / 2;
    const centerY = height / 2;
    context.save();
    for (let index = 0; index < Engine.PHASE_COUNT; index += 1) {
      const angle = -Math.PI / 2 + index * (Math.PI * 2 / Engine.PHASE_COUNT);
      const x = centerX + Math.cos(angle) * width * 0.28;
      const y = centerY + Math.sin(angle) * height * 0.28;
      context.beginPath();
      context.arc(x, y, index === room.globalPhase ? 5 : 2.5, 0, Math.PI * 2);
      context.fillStyle = index === room.globalPhase
        ? phaseColor(index, 0.95, state.run.paletteOffset)
        : "rgba(181, 192, 245, .22)";
      context.shadowBlur = index === room.globalPhase ? 16 : 0;
      context.shadowColor = context.fillStyle;
      context.fill();
    }
    context.beginPath();
    context.arc(centerX, centerY, 5, 0, Math.PI * 2);
    context.fillStyle = "rgba(222, 227, 255, .72)";
    context.fill();
    context.restore();
  }

  function renderMap() {
    mapElement.replaceChildren();
    Engine.ROOM_DEFINITIONS.forEach((definition, index) => {
      const node = document.createElement("div");
      node.className = "map-node";
      if (index === state.run.roomIndex) node.classList.add("current");
      if (state.run.completedRooms.includes(index)) node.classList.add("completed");
      if (definition.boss) node.classList.add("boss");
      const shape = document.createElement("span");
      shape.className = "map-node-shape";
      shape.setAttribute("aria-hidden", "true");
      node.append(shape);
      mapElement.append(node);
    });
  }

  function createActionButton(action) {
    const button = document.createElement("button");
    button.className = "action";
    button.type = "button";
    button.dataset.action = action;
    button.setAttribute("aria-label", "抽象操作");
    button.innerHTML = `<svg viewBox="0 0 62 62" aria-hidden="true">${actionPaths[state.run.glyphMap[action]]}</svg>`;
    button.addEventListener("click", () => {
      const result = Engine.performAction(state.run.room, action, state.selectedNode);
      if (!result.accepted) {
        button.classList.remove("denied");
        void button.offsetWidth;
        button.classList.add("denied");
        return;
      }
      state.run = { ...state.run, room: result.room };
      state.replay = action === "flow"
        ? { frames: result.room.trace, index: 0, started: performance.now() }
        : null;
      state.selectedNode = null;
      state.flash = 1;
      update();
    });
    actionsElement.append(button);
  }

  function renderActions() {
    actionsElement.replaceChildren();
    state.run.glyphOrder.forEach((action) => createActionButton(action));
    Array.from(actionsElement.children).forEach((button) => {
      const action = button.dataset.action;
      button.disabled = Boolean(state.replay) || !Engine.isActionAvailable(state.run.room, action);
    });
  }

  function updateOutcome() {
    const outcome = state.run.room.outcome;
    const finished = outcome !== "active";
    outcomeElement.hidden = !finished || Boolean(state.replay);
    outcomeElement.classList.toggle("success", outcome === "success");
    outcomeElement.classList.toggle("fracture", outcome === "fracture");
  }

  function update() {
    renderMap();
    renderActions();
    updateOutcome();
    drawTarget();
    drawPulse();
  }

  function selectNodeAt(event) {
    const rect = fieldCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const positions = nodePositions(state.run.room, rect.width, rect.height);
    let closest = null;
    let distance = Infinity;
    positions.forEach((position, index) => {
      const candidate = Math.hypot(x - position.x, y - position.y);
      if (candidate < distance) {
        distance = candidate;
        closest = index;
      }
    });
    if (distance < 46 && !state.replay && state.run.room.outcome === "active") {
      state.selectedNode = closest;
      state.flash = 1;
      update();
    }
  }

  function hoverNodeAt(event) {
    const rect = fieldCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const positions = nodePositions(state.run.room, rect.width, rect.height);
    state.hoverNode = null;
    positions.forEach((position, index) => {
      if (Math.hypot(x - position.x, y - position.y) < 43) state.hoverNode = index;
    });
  }

  fieldCanvas.addEventListener("pointerdown", selectNodeAt);
  fieldCanvas.addEventListener("pointermove", hoverNodeAt);
  fieldCanvas.addEventListener("pointerleave", () => { state.hoverNode = null; });
  resetRun.addEventListener("click", () => {
    state.replay = null;
    state.run = Engine.restartRoom(state.run);
    state.selectedNode = null;
    update();
  });
  outcomeAction.addEventListener("click", () => {
    state.replay = null;
    if (state.run.room.outcome === "success") {
      state.run = state.run.roomIndex === Engine.ROOM_DEFINITIONS.length - 1
        ? Engine.createRun(readSeed())
        : Engine.enterNextRoom(state.run);
    } else {
      state.run = Engine.restartRoom(state.run);
    }
    state.selectedNode = null;
    update();
  });
  window.addEventListener("resize", update);

  function frame(time) {
    const elapsed = time - state.lastFrame;
    state.lastFrame = time;
    state.flash = Math.max(0, state.flash - elapsed / 600);
    if (state.replay) {
      const index = Math.floor((time - state.replay.started) / REPLAY_STEP_MS);
      if (index >= state.replay.frames.length) {
        state.replay = null;
        update();
      } else if (index !== state.replay.index) {
        state.replay.index = index;
        update();
      }
    }
    drawField(time);
    requestAnimationFrame(frame);
  }

  update();
  requestAnimationFrame(frame);
})();
