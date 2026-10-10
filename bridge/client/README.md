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

CommunicationMod can execute the checkpoint's direct combat policy, but it
cannot reconstruct the simulator state required by Full-Gumbel search or exact
`equivalent-plays-v1` probability grouping. For transport/alignment diagnostics
only, opt in with `--allow-direct-combat-policy` (or set
`STS_RL_ALLOW_DIRECT_COMBAT_POLICY=1`). The overworld policy remains
deterministic unless `--stochastic` is passed.

CommunicationMod initially represents a post-combat card reward as a `card`
placeholder. After the other claimable rewards are handled, the bridge opens
that placeholder automatically. The upper policy then receives the actual
three candidates on `CARD_REWARD` and chooses a card or `SKIP` there, matching
the simulator's candidate-bearing decision point.

Logs are written to `bridge/client/agent_orchestrator.log`. The script writes
only CommunicationMod commands to stdout.

When the checkpoint expects `--global-map-view`, the agent reads
CommunicationMod's `game_state.map` plus the map screen's
`screen_state.current_node`, `screen_state.next_nodes`, and
`screen_state.boss_available` fields to fill the same 15x7 DAG observation used
by simulated training.
