# CommunicationMod Agent

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
placeholder. After the other claimable rewards are handled, the bridge opens
that placeholder automatically. The upper policy then receives the actual
three candidates on `CARD_REWARD` and chooses a card or `SKIP` there, matching
the simulator's candidate-bearing decision point.

Event choices and overworld reward/card-selection commands wait 0.5 seconds
before being sent. Combat card generation (such as Nilry's Codex) and ordinary
combat commands keep their existing timing.

Logs are written to `bridge/client/agent_orchestrator.log`. The script writes
only CommunicationMod commands to stdout.

When the checkpoint expects `--global-map-view`, the agent reads
CommunicationMod's `game_state.map` plus the map screen's
`screen_state.current_node`, `screen_state.next_nodes`, and
`screen_state.boss_available` fields to fill the same 15x7 DAG observation used
by simulated training.
