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

### Simulator-parity current intents

Neural simulation exposes the current native monster move even with Runic Dome.
For the same current-state information in real-game evaluation, use
`ExposeCurrentIntent.java` to patch a backed-up CommunicationMod jar offline.
It changes only `GameStateConverter.convertMonsterToJson`: the Dome-specific
export mask is bypassed, and each monster receives
`current_intent_export: simulator-parity-v1`. Gameplay, UI and RNG are untouched.
The bridge fails on a live Dome monster whose current move is still missing;
it no longer substitutes a shadow-world guess. Future search outcomes remain
hypothetical, not the real game's hidden future.

Compile with Java 8 target and ModTheSpire.jar on the classpath; run the class
with arguments `original.jar patched.jar desktop-1.0.jar BaseMod.jar ModTheSpire.jar`.
The output must not exist. The patcher checks exactly one target call and checks
that other converter methods retain identical bytecode. Keep originals and
SHA256s outside Git, stop the game before deployment, and verify that only the
converter class differs between jars. This is an explicit evaluation protocol
change, not a claim of complete game/simulator equivalence.

When the checkpoint expects `--global-map-view`, the agent reads
CommunicationMod's `game_state.map` plus the map screen's
`screen_state.current_node`, `screen_state.next_nodes`, and
`screen_state.boss_available` fields to fill the same 15x7 DAG observation used
by simulated training.
