"""Phase-0 human-in-the-loop orchestrator for CommunicationMod.

CommunicationMod launches this script as a subprocess and talks to it over the
script's stdin/stdout: game state JSON comes in on stdin, a single command line
goes out on stdout. Those streams belong to the mod, so the human plays through
a separate tkinter window instead.

Protocol: print "ready" once, then loop {read state line -> show -> human types
a command -> write command line}. Never print anything else to stdout.

Everything (raw frames in, commands out, tracebacks) is mirrored to
orchestrator.log next to this file, because the GUI runs under pythonw with no
console and we cannot debug by eyeballing the window.
"""

import json
import logging
import queue
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import scrolledtext

# The mod writes/reads with the JVM's platform default charset (GBK on this
# Chinese Windows + Java 8), NOT UTF-8: DataWriter does message.getBytes() with
# no charset. So decode incoming state as GBK (an ASCII superset, safe even for
# an English game); errors="replace" guarantees the reader never dies on a
# stray byte. Our outgoing commands are pure ASCII, identical under any codec.
sys.stdin.reconfigure(encoding="gbk", errors="replace")
sys.stdout.reconfigure(encoding="gbk")

LOG_PATH = Path(__file__).with_name("orchestrator.log")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("orchestrator")

# Background reader thread -> GUI, via this queue. Each item is (kind, payload).
inbox = queue.Queue()


def reader_loop():
    try:
        for line in sys.stdin:
            line = line.strip()
            if line:
                log.debug("RECV %s", line)
                inbox.put(("line", line))
        log.info("stdin EOF")
        inbox.put(("eof", None))
    except Exception:
        log.exception("reader_loop crashed")
        inbox.put(("eof", None))


def send(command):
    log.info("SEND %s", command)
    sys.stdout.write(command + "\n")
    sys.stdout.flush()


def render(state):
    """Turn one game-state JSON dict into a compact human-readable block."""
    out = []
    cmds = state.get("available_commands", [])
    out.append("available: " + " ".join(cmds))
    out.append(f"ready_for_command: {state.get('ready_for_command')}")
    if not state.get("in_game"):
        out.append("\n[not in a run] use: START IRONCLAD [ascension] [seed]")
        return "\n".join(out)

    gs = state.get("game_state", {})
    out.append(
        f"\nAct {gs.get('act')}  Floor {gs.get('floor')}  "
        f"HP {gs.get('current_hp')}/{gs.get('max_hp')}  Gold {gs.get('gold')}"
    )
    out.append(
        f"screen={gs.get('screen_type')}  room={gs.get('room_type')}  "
        f"phase={gs.get('room_phase')}"
    )

    combat = gs.get("combat_state")
    if combat:
        p = combat.get("player", {})
        out.append(
            f"\n-- PLAYER  HP {p.get('current_hp')}/{p.get('max_hp')}  "
            f"block {p.get('block')}  energy {p.get('energy')}"
        )
        if p.get("powers"):
            out.append("   powers: " + powers_str(p["powers"]))

        out.append("-- MONSTERS")
        for i, m in enumerate(combat.get("monsters", [])):
            if m.get("is_gone"):
                continue
            dmg = m.get("move_adjusted_damage", -1)
            hits = m.get("move_hits", 1)
            intent = m.get("intent", "?")
            atk = f" {dmg}x{hits}" if dmg and dmg > 0 else ""
            pw = "  [" + powers_str(m["powers"]) + "]" if m.get("powers") else ""
            out.append(
                f"  [{i}] {m.get('name')}  HP {m.get('current_hp')}/{m.get('max_hp')}"
                f"  block {m.get('block')}  intent {intent}{atk}{pw}"
            )

        out.append("-- HAND (1-indexed)")
        for i, c in enumerate(combat.get("hand", []), start=1):
            out.append(f"  {i}. {card_str(c)}")
        out.append(
            f"draw {len(combat.get('draw_pile', []))}  "
            f"discard {len(combat.get('discard_pile', []))}  "
            f"exhaust {len(combat.get('exhaust_pile', []))}"
        )

    choices = gs.get("choice_list")
    if choices:
        out.append("\n-- CHOICES (CHOOSE <index|name>)")
        for i, ch in enumerate(choices):
            out.append(f"  [{i}] {ch}")

    ss = gs.get("screen_state")
    if ss and not combat:
        out.append("\n-- screen_state")
        out.append(json.dumps(ss, ensure_ascii=False, indent=2))

    pots = [p for p in gs.get("potions", []) if p.get("id") != "Potion Slot"]
    if pots:
        out.append("\npotions: " + ", ".join(p.get("name", "?") for p in pots))

    return "\n".join(out)


def card_str(c):
    flags = []
    if c.get("ethereal"):
        flags.append("ethereal")
    if c.get("exhausts"):
        flags.append("exhausts")
    if not c.get("is_playable", True):
        flags.append("unplayable")
    tag = (" {" + ",".join(flags) + "}") if flags else ""
    target = "  ->target" if c.get("has_target") else ""
    return f"{c.get('name')} (cost {c.get('cost')}){target}{tag}"


def powers_str(powers):
    return ", ".join(f"{p.get('name')}:{p.get('amount')}" for p in powers)


class App:
    def __init__(self, root):
        self.root = root
        root.title("STS orchestrator (human-in-the-loop)")
        root.geometry("760x820")

        self.state_view = scrolledtext.ScrolledText(
            root, height=30, font=("Consolas", 10), wrap="word"
        )
        self.state_view.pack(fill="both", expand=True, padx=6, pady=(6, 3))
        self.state_view.configure(state="disabled")

        bar = tk.Frame(root)
        bar.pack(fill="x", padx=6)
        for label, cmd in [
            ("END", "END"), ("PROCEED", "PROCEED"), ("RETURN", "RETURN"),
            ("STATE", "STATE"), ("Raw", "__RAW__"),
        ]:
            tk.Button(bar, text=label, command=lambda c=cmd: self.quick(c)).pack(
                side="left", padx=2, pady=3
            )

        entry_row = tk.Frame(root)
        entry_row.pack(fill="x", padx=6, pady=3)
        self.entry = tk.Entry(entry_row, font=("Consolas", 11))
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.submit())
        self.entry.focus_set()
        tk.Button(entry_row, text="Send", command=self.submit).pack(side="left", padx=4)

        self.log_view = scrolledtext.ScrolledText(
            root, height=8, font=("Consolas", 9), wrap="word"
        )
        self.log_view.pack(fill="both", padx=6, pady=(3, 6))
        self.log_view.configure(state="disabled")

        self.last_raw = "(no state yet)"
        self.logline(f"started; log file: {LOG_PATH}")
        self.logline("sent 'ready', waiting for game state...")
        self.root.after(50, self.poll)

    def poll(self):
        try:
            while True:
                kind, payload = inbox.get_nowait()
                if kind == "eof":
                    self.logline("!! stdin closed - game/mod disconnected")
                    continue
                self.handle(payload)
        except queue.Empty:
            pass
        except Exception:
            # A bug in handling one frame must never kill the refresh loop.
            log.exception("poll loop error")
        self.root.after(50, self.poll)

    def handle(self, line):
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            log.error("bad JSON: %s", line[:500])
            self.logline("!! could not parse a line (see log)")
            return
        if "error" in state:
            log.warning("mod error: %s", state["error"])
            self.logline("ERROR from mod: " + str(state["error"]))
            return
        self.last_raw = json.dumps(state, ensure_ascii=False, indent=2)
        try:
            text = render(state)
        except Exception:
            log.exception("render() failed; raw frame follows:\n%s", self.last_raw)
            text = (
                "!! render() crashed - showing raw JSON instead "
                "(traceback in log)\n\n" + self.last_raw
            )
        self.show(text)

    def show(self, text):
        self.state_view.configure(state="normal")
        self.state_view.delete("1.0", "end")
        self.state_view.insert("1.0", text)
        self.state_view.configure(state="disabled")

    def quick(self, cmd):
        if cmd == "__RAW__":
            self.logline("--- raw state ---\n" + self.last_raw)
            return
        send(cmd)
        self.logline("> " + cmd)

    def submit(self):
        cmd = self.entry.get().strip()
        if not cmd:
            return
        self.entry.delete(0, "end")
        send(cmd)
        self.logline("> " + cmd)

    def logline(self, msg):
        self.log_view.configure(state="normal")
        self.log_view.insert("end", msg + "\n")
        self.log_view.see("end")
        self.log_view.configure(state="disabled")


def main():
    log.info("=== session start ===")
    log.info("executable=%s argv=%s cwd=%s", sys.executable, sys.argv, Path.cwd())
    try:
        send("ready")
        threading.Thread(target=reader_loop, daemon=True).start()
        root = tk.Tk()
        App(root)
        root.mainloop()
        log.info("mainloop exited")
    except Exception:
        log.exception("fatal error in main")
        raise


if __name__ == "__main__":
    main()
