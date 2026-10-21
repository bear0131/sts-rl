"""Human-controlled real-game demo gate; never a simulator/policy fallback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

from bridge.campaign import seed_string_to_long
from bridge.schemas.entity_ids import DEMO_ACTIONS, DEMO_CONTROL_VERSION, DEMO_STATUSES


class InteractiveDemo:
    def __init__(self, args):
        self.args = args
        self.request = (0, "NONE", "")
        self.status = None
        self.expected_seed = None
        self.started = None
        self.entered = False
        self.active = False
        self.faulted = False
        self.reset_requested = False
        self.output = None

    def receipt(self, status, detail=""):
        if status not in DEMO_STATUSES:
            raise ValueError("Unknown demo status")
        self.status = status
        return f"DEMO {status}" + (f" {detail}" if detail else "")

    def fail(self, reason):
        self.active = False
        self.faulted = True
        return self.receipt("ERROR", reason)

    @staticmethod
    def controls(frame):
        data = frame.get("demo_controls")
        if (not isinstance(data, dict)
                or set(data) != {"version", "request_id", "action", "seed", "enabled"}
                or type(data["version"]) is not int or data["version"] != DEMO_CONTROL_VERSION
                or type(data["request_id"]) is not int or data["request_id"] < 0
                or data["action"] not in DEMO_ACTIONS
                or not isinstance(data["seed"], str) or type(data["enabled"]) is not bool):
            raise ValueError("Missing or invalid interactive-demo control receipt")
        return data

    def gate(self, frame, controls):
        """Return a transport command, or None to allow ordinary policy execution."""
        self.reset_requested = False
        if self.status is None:
            return self.receipt("READY", Path(self.args.checkpoint).parent.name)
        request = (controls["request_id"], controls["action"], controls["seed"])
        if request[0] < self.request[0] or (request[0] == self.request[0] and request != self.request):
            raise ValueError("Regressed or mutated demo request")
        if request[0] > self.request[0]:
            self.request = request  # Consume once, including a request rejected below.
            if not controls["enabled"]:
                raise ValueError("New demo request must explicitly enable AUTO")
            gs = frame.get("game_state") or {}
            start = request[1] == "START"
            if start:
                if (frame.get("in_game") is not False or frame.get("ready_for_command") is not True
                        or "START" not in {v.upper() for v in frame["available_commands"]}
                        or not (self.args.auto_start and self.args.continuous_test)):
                    raise ValueError("Random demo requires an explicit idle-menu request and launch authorization")
                self.expected_seed = seed_string_to_long(request[2])
            elif request[1] == "ATTACH":
                if (frame.get("in_game") is not True or type(gs.get("seed")) is not int
                        or gs.get("class") != "IRONCLAD" or gs.get("ascension_level") != 0
                        or type(gs.get("act")) is not int or not 1 <= gs["act"] <= 3
                        or gs.get("screen_type") == "GAME_OVER"):
                    raise ValueError("AUTO takeover requires a live Ironclad A0 Act1-3 state")
                self.expected_seed = gs["seed"]
            else:
                raise ValueError("New demo request has no supported action")
            output = Path(self.args.demo_output_dir) / str(time.time_ns())
            output.mkdir(parents=True, exist_ok=False)
            if start:
                # Preserve every save slot, not just a guessed currently selected slot.
                save_dir = Path(self.args.demo_save_dir)
                if not save_dir.is_dir():
                    raise ValueError("Configured game save directory does not exist")
                hashes = {}
                for source in save_dir.iterdir():
                    if source.is_file() and source.name.lower().endswith((".autosave", ".autosave.backup", ".autosave.bak")):
                        digest = hashlib.sha256(source.read_bytes()).hexdigest()
                        shutil.copy2(source, output / source.name)
                        if (hashlib.sha256((output / source.name).read_bytes()).hexdigest() != digest
                                or hashlib.sha256(source.read_bytes()).hexdigest() != digest):
                            raise ValueError("Save changed while preserving it; no START sent")
                        hashes[source.name] = digest
                (output / "original-save-hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
            (output / "request.json").write_text(json.dumps({
                "kind": "interactive-demo-not-formal-evaluation", "request": request,
                "expected_seed": self.expected_seed, "checkpoint": self.args.checkpoint,
                "source": "explicit game UI button", "timeout_seconds": 3600,
            }, indent=2), encoding="utf-8")
            self.output = output
            self.started = time.monotonic()
            self.entered = not start
            self.active, self.faulted, self.reset_requested = True, False, True
            self.status = "RUNNING" if start else None
            return f"START IRONCLAD 0 {request[2]}" if start else self.receipt("RUNNING", "current-screen-takeover")
        if not controls["enabled"]:
            if self.active and self.status != "PAUSED":
                self.active = False
                return self.receipt("PAUSED")
            return "STATE"
        if self.faulted or not self.active:
            return "STATE"
        if time.monotonic() - self.started > 3600:
            return self.fail("demo-time-budget-exceeded")
        if frame.get("in_game") is False:
            if self.entered:
                self.active = False
                return self.receipt("STOPPED", "returned-to-menu")
            if time.monotonic() - self.started > 30:
                return self.fail("start-not-acknowledged")
            return "STATE"
        gs = frame["game_state"]
        if gs.get("seed") != self.expected_seed:
            return self.fail("seed-changed-requires-explicit-AUTO")
        self.entered = True
        if type(gs.get("act")) is not int or not 1 <= gs["act"] <= 3:
            self.active = False
            return self.receipt("STOPPED", "outside-Act1-3-contract")
        if gs.get("screen_type") == "GAME_OVER":
            if frame.get("ready_for_command") is not True:
                return "STATE"
            ss = gs.get("screen_state")
            if (not isinstance(ss, dict) or type(ss.get("victory")) is not bool
                    or type(ss.get("score")) is not int or type(gs.get("floor")) is not int):
                raise ValueError("Demo terminal lacks actual typed outcome facts")
            (self.output / "terminal.json").write_text(json.dumps(frame), encoding="utf-8")
            self.active = False
            return self.receipt("FINISHED", "victory" if ss["victory"] else "defeat")
        if self.status != "RUNNING":
            return self.receipt("RUNNING")
        return None
