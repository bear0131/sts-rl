# CommunicationMod Agent

With `--combat-search`, `--gumbel-scale 0` disables search Gumbel noise.
Omitting it preserves the checkpoint's setting. Upper decisions remain
deterministic unless `--stochastic` is explicitly requested.

Automatic real-game runner:

```powershell
uv run --no-sync python -m bridge.client.agent_orchestrator path\to\final.zip --ascension 0
```

For CommunicationMod, point the mod to:

```text
D:\sts-rl\bridge\client\agent_orchestrator.bat
```

If the mod cannot pass arguments, set environment variables before launching the
game:

```powershell
$env:STS_RL_CHECKPOINT = "D:\sts-rl\runs\whole-run\...\final.zip"
$env:STS_RL_ASCENSION = "0"
```

Optional start-new-run behavior:

```powershell
$env:STS_RL_AUTO_START = "1"
$env:STS_RL_START_SEED = "108"
```

Checkpoints that were saved on another machine may refer to a frozen card-vector
file by an absolute path. Relocate that dependency explicitly:

```powershell
$env:STS_RL_CARD_VECTOR_CHECKPOINT = "D:\path\to\card-vectors.pt"
```

Simulator-backed Full-Gumbel combat can be enabled explicitly. The bridge keeps
a persistent native battle mirror, checks public state and legal actions against
each settled CommunicationMod frame, and refuses to fall back to the direct
combat policy on a mismatch:

```powershell
$env:STS_RL_COMBAT_SEARCH = "1"
$env:STS_RL_NATIVE_SEARCH_THREADS = "4"
```

The native extension must come from the same source/configuration as the
checkpoint. The direct combat policy remains available only as an explicit
transport diagnostic through `--allow-direct-combat-policy`; it does not
reproduce Full-Gumbel search or exact `equivalent-plays-v1` grouping. The
overworld policy is deterministic unless `--stochastic` is passed.

For bounded real-game campaigns, provide distinct seeds and a JSONL receipt.
Only the first idle frame sends each `START`, and every natural game-over row
records the requested and actual seed plus search decision/alignment counts:

```powershell
$env:STS_RL_START_SEEDS = "310001,310002,310003,310004,310005"
$env:STS_RL_CAMPAIGN_RESULTS = "D:\sts-rl\runs\local\campaign.jsonl"
```

CommunicationMod initially represents a post-combat card reward as a `card`
placeholder. Before asking the upper policy for an action, the bridge opens
that reversible sub-screen and merges its actual candidates with the cached
gold, relic, and potion rewards. The policy therefore sees the same combined
`combat_reward` decision as it does in the simulator. If it chooses an outer
reward or `PROCEED`, the bridge returns from the card page and completes that
semantic action without another policy call; a policy `SKIP` masks the still
physical card placeholder for the remainder of that reward screen.

Event choices and overworld reward/card-selection commands wait 0.5 seconds
before being sent. Combat card generation (such as Nilry's Codex) and ordinary
combat commands keep their existing timing.

Shop potion offers remain visible in the observation, but buying them is masked
when the player has Sozu or no empty potion slot, matching both the simulator's
legal-action contract and the real game's accepted commands.

Logs are written to `bridge/client/agent_orchestrator.log`. The script writes
only CommunicationMod commands to stdout.

### Simulator-parity state exporter

Neural simulation exposes the current native monster move even with Runic Dome.
For the same current-state information in real-game evaluation, use
`ExposeCurrentIntent.java` to patch a backed-up CommunicationMod jar offline.
It patches seven `GameStateConverter` serializers and both actual JSON writers
(`getCommunicationState` and `CommunicationMod.receivePreUpdate`) to use the
bundled shaded Gson's `serializeNulls()`. An intermediate Map containing null
is insufficient: Gson's default would remove that field from the wire.
The Dome-specific export mask is bypassed, and monsters receive
`current_intent_export: simulator-parity-v13`.
Monster serialization includes the actual `is_dying`, `is_dead`, `is_escaping`,
`escape_next` flags and Awakened One's room-level first-life protection
(`unawakened`). Zero HP and `is_gone` must not substitute for these fields.
The bridge preserves corpse/revival powers and move history; incomplete entity
mapping or private-state replacement fails explicitly instead of dropping a body.
The exporter also supplies all three run keys, recurring energy, combat/relic
counters, private power values, and explicit card-selection source/context.
The v11 exporter identifies DrugDealer's two-card transform grid by its actual
event class: all three generic grid operation flags are false in the game.
All GRID candidates and selected cards require complete instance records and
unique UUIDs, including outside combat; names and upgrades cannot replace an
instance identity. This does not by itself enable an unaudited event receipt.
Map candidates retain all seven possible columns and their original UI indices.
The frozen three-token local preview is only redundant context: when more than
three choices exist, the 105-node map input must contain every choice with its
exact room and choice flag. Missing/contradictory graphs are rejected; older
checkpoints without the full-map channel cannot handle these wide map screens.
Neither the action list nor the complete graph is truncated to fit the preview.
Card serialization always includes the actual `misc`, including zero, and a
versioned `simulator_card_state` object with base cost/damage/block, free-once,
retention/innate flags and all three bottle flags. Every monster also exports
`simulator_stasis_card`: the full held card from live StasisPower, or explicit
null when none is held. Master-deck restore consumes current persistent values
and physical bottle slots. Settled combat roots restore complete native card
instances with UUID mapping; in-progress selection and independent dynamic
attributes still require their own alignment audit.
The v13 wire also always includes `current_action` (explicit null when none)
and `choice_list` (explicit [] when CHOOSE is unavailable). Normal frames and
error replies have separate presence contracts; missing/null screen payloads,
inventories, command lists or contradictory aliases fail before menu actions,
ACK handling or policy inference. See `json_presence_contracts` in the complete
word tables. Do not mix v13 Python with an older jar. Model tensor layouts and
checkpoint parameters are unchanged; this is a transport change.
Two read-only game hooks track the last monster attack damage and reset it at
combat entry. Gameplay, UI and RNG are untouched.
The bridge fails on a live Dome monster whose current move is still missing;
it no longer substitutes a shadow-world guess. Future search outcomes remain
hypothetical, not the real game's hidden future.

Compile with Java 8 target and ModTheSpire.jar on the classpath; run the class
with arguments `original.jar patched.jar desktop-1.0.jar BaseMod.jar ModTheSpire.jar`.
The output must not exist. The patcher checks exactly one Dome call and one Gson
constructor per wire writer, and checks that other methods in both modified
classes retain identical bytecode. Keep originals and
SHA256s outside Git, stop the game before deployment, and verify that only the
converter and CommunicationMod classes differ among existing classes; the exporter and its two hook
classes are added. This is an explicit evaluation protocol
change, not a claim of complete game/simulator equivalence.

When the checkpoint expects `--global-map-view`, the agent reads
CommunicationMod's `game_state.map` plus the map screen's
`screen_state.current_node`, `screen_state.next_nodes`, and
`screen_state.boss_available` fields to fill the same 15x7 DAG observation used
by simulated training.
