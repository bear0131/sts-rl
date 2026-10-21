# CommunicationMod Agent

### 游戏内手动演示（Windows 本机部署）

主菜单右侧显示“随机种子演示”；局内左上方、遗物栏下显示 **AUTO：关**，
旁边状态说明点击接管/暂停，避开右上角种子/牌堆/设置区域。
打开游戏、手动点 Continue 或进入一场战斗都不会自动接管。点击 AUTO 才从当前真实画面
尝试建立严格模型状态，再次点击关闭。关闭后可能仍有一个已发出的动作正在结算；手动
操作后再开 AUTO 会丢弃旧推测根/未发命令并重新接入，不复用过时动作或候选。
若暂停首帧与重新开启时的完整 game_state 完全相同，则保留同一搜索根/逻辑索引/待确认事务，
不把普通暂停变成从索引 0 重新搜索；任何状态变化都不能用该捷径。快速开后又关，若关闭在
请求被处理前已经可见，则取消请求而非误报错误或执行开局。
不支持/信息不完整的当前状态会明确停止并显示错误，详情在 agent 日志；没有 PAD 或直接策略降级。
这不保证任意历史中途档的隐藏 RNG 与从开局连续运行相同，不算固定版本胜率样本，也不是 SL 修复。

`experiments/launchers/train47-demo.properties` 是当前本机最终 TRAIN47 的部署配置，
使用 `--demo-controls`，与配套带 `InteractiveDemoControls` 的 Java jar 一起部署。
菜单新开仍需两个明确授权参数 `--continuous-test --auto-start`，但这两个参数在 demo 模式
**不会自行开局**：还必须有游戏内按钮的单次新请求。没有请求只发 STATE。
每局新开前在 `--demo-output-dir` 的独立子目录保存所有原始 autosave/bak 与 SHA256，
终局原始帧也写在该目录。最终/报错/返回菜单/3600 秒预算结束后停止，不自动再开。
现有 campaign 和手动续档 CLI 在不传 `--demo-controls` 时行为不变。

UI 控制合同 `demo_controls.version=1` 独立于观测和动作 codec：
完整字段 `version/request_id/action/seed/enabled`，动作 `NONE/START/ATTACH`；
单调请求号去重，开关关闭只刷新状态。`DEMO READY/RUNNING/PAUSED/STOPPED/FINISHED/ERROR`
只是界面回执，由 Java 在原 gameplay dispatcher 前单独拦截，不是模型动作，不进入合法动作表、
模型张量、RNG 或搜索预算。模型成功加载并完成搜索初始化后才发 READY，进程退出显示错误。
导出器的非目标方法字节码保护包含新增的三个 UI 钩子；不要把旧诊断 supervisor 的 original.zip
整体恢复到本次永久修复的生产 jar/config 上，避免重新启用已废弃入口。

With `--combat-search`, `--gumbel-scale 0` disables search Gumbel noise.
Omitting it preserves the checkpoint's setting. Upper decisions remain
deterministic unless `--stochastic` is explicitly requested.

For a checkpoint predating `run_keys`/`reward_keys`, the default `native` reward
contract refuses unsupported key decisions. After explicitly agreeing to a
historical Act 1-3 evaluation, pass
`--reward-policy-contract checkpoint-v7-act3`. This preserves the old policy's
decision set by excluding only the added `CLAIM_KEY` action from its policy
mask. Real keys, all rewards, world legality and raw frames remain complete;
the agent does not auto-claim keys or change the old campfire recall action.
The protocol and excluded actions are logged, terminal receipts name the
protocol, and actions outside Acts 1-3 are rejected. This is not an Act 4
evaluation or evidence that the old network learned key features.

Campaigns record only ready `GAME_OVER` frames with actual typed seed, victory,
score, floor and act fields; missing results are errors, not losses/zero scores.
Metadata cannot overwrite terminal facts. The agent waits for an available
`PROCEED`, and only a ready explicit menu advances the seed. An interrupted run
does not authorize restarting it; numerically duplicate configured seeds are
rejected even if their case or leading-zero spelling differs.

`search_simulations` sums explicit strategy diagnostics, including genuine zero
for forced END roots; `search_simulation_budget` retains decisions × configured
budget. `search_simulation_accounting=strategy-diagnostics-v1` identifies this
reporting contract. Older records without it used the budget product and must
not be presented as measured work. Leaf evaluations are a different quantity.
When a dispatched native-search command receives one of the exact pre-dispatch
overlay availability rejections in the shared codec table, the production CLI
rolls back the whole speculative transaction and its logical index, clears that
rejected command's waits, and asks for fresh state. It does not replay the old
command. Recovery is bounded to three consecutive rejections and cannot apply
to an unsent prepared command, another verb, an argument error or an unknown
exception. Such errors remain failures; a normal STATE response alone does not
reset the rejection budget (BRIDGE170).

The shared raw-screen table also suspends ACK/settlement time while the user
inspects settings, the master deck, or the draw/discard/exhaust pile. The draw
viewer is actually `GAME_DECK_VIEW`, not `DRAW_PILE_VIEW`. Both the exact raw
screen name and `is_screen_up=true` are required. Pending transactions remain
intact; only STATE is requested until the viewer closes. Real map, event and
card-selection screens are not blanket-paused (BRIDGE171).

Simulator evaluation can select the same contract through
`PolicyStrategy(..., reward_policy_contract="checkpoint-v7-act3")`; both paths
use `rl.policies.model.action_mask_for_policy` without altering the environment.

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
CommunicationMod still advertises affordable potion buttons in these cases.
The adapter matches their real shelf identities for transport coverage before
filtering policy actions; it does not treat those buttons as unknown or buy them.

Logs are written to `bridge/client/agent_orchestrator.log`. The script writes
only CommunicationMod commands to stdout.

### Diagnostic continuation from saves

For ongoing bridge debugging, resume the latest applicable fault/room-entry
save instead of replaying a run from floor 1. Preserve the user's current save,
preferences, mod and configuration before installing an archived diagnostic
save; preserve the new result before restoring those originals on exit.
Inspect the archived save's actual seed, floor and pre/post-combat state first.
A game autosave resumes a room boundary, not necessarily the exact failing turn.

Use `--attach-existing-seed <internal-seed> --attach-existing-floor <floor>`
without `--auto-start` or `--start-seeds`, then use the game's **Continue**
button. If the corresponding original log establishes the logical search index
at that saved boundary, also pass `--attach-search-index <index>`; do not infer
it merely by counting logged commands (cancellations and multi-pick prefixes
can differ). The first attached frame must match the requested seed/floor.
Record this as an SL diagnostic, not a new complete run or formal win-rate
sample. A new full run requires a reason such as an unusable save or an explicit
whole-run evaluation; do not silently fall back to START if attachment fails.

Bounded diagnostic supervisors share the no-CLI process-identity helpers in
`experiments/launchers/real_game_processes.py`. Capture each actual owner and
child with `capture_process_identity` while live, persist its exact PID and
creation time, and retain separate records for different incarnations of a
reused PID. Parent PID, executable, name and observation time are provenance;
denied metadata is explicit null with a reason, never a made-up value.
Before checking or stopping an owner, use `matching_process(record)`; signal
only the returned psutil Process, not the bare PID. Access denial is not proof
of exit. Numeric `child_pids` may remain for display but cannot authorize cleanup.

Old records without birth identities require read-only provenance review.
`inspect_legacy_pid` can prove absence or reuse after a documented observation
bound; otherwise it returns `unverifiable`, not "still our process". In
particular, final status mtime cannot resolve PID reuse during the game. Do not
backfill guessed birth times, stop an unknown current owner, or rewrite sealed
scripts/protocols from completed runs to pretend they recorded identities.

### Simulator-parity state exporter

Neural simulation exposes the current native monster move even with Runic Dome.
For the same current-state information in real-game evaluation, use
`ExposeCurrentIntent.java` to patch a backed-up CommunicationMod jar offline.
It patches eight `GameStateConverter` serializers and both actual JSON writers
(`getCommunicationState` and `CommunicationMod.receivePreUpdate`) to use the
bundled shaded Gson's `serializeNulls()`. An intermediate Map containing null
is insufficient: Gson's default would remove that field from the wire.
The Dome-specific export mask is bypassed, and monsters receive
`current_intent_export: simulator-parity-v41`.
Monster serialization includes the actual `is_dying`, `is_dead`, `is_escaping`,
`escape_next` flags and Awakened One's room-level first-life protection
(`unawakened`). Zero HP and `is_gone` must not substitute for these fields.
The bridge preserves corpse/revival powers and move history; incomplete entity
mapping or private-state replacement fails explicitly instead of dropping a body.
The exporter also supplies all three run keys, recurring energy, combat/relic
counters, private power values, and explicit card-selection source/context.
Since v40, combat counters also preserve the actual `bloodied` and
`red_skull_active` latches. Increasing maximum HP changes the half-HP threshold
before healing; deriving the old state from the new ratio can incorrectly
remove Red Skull Strength. Both booleans are required by the paired Python and
native strict roots, copied/read back and distinguished by search keys.
They are transition metadata, not added frozen-model counter columns. Deploy
the exporter, Python and native extension together; older roots are not accepted.
Since v41, `cards_played` counts queue admission, while `attacks_played` and
`skills_played` count actual non-`dontTriggerOnUseCard` use events. The read-only
hook observes only the two-argument `UseCardAction` constructor (the one-argument
overload delegates to it), including purge copies but excluding failed autoplay.
The constructor hook declares the patched `UseCardAction` instance followed by
both constructor arguments, as required by the installed ModTheSpire binding.
`preBattlePrep` initializes the observer, and the actual action-manager turn
resets its per-turn counts. Unobserved ownership or a regressed turn fails;
the exporter never guesses a use count from a relic's modulo-three display.
An admitted copy skipped for a half-dead/null enemy still counts for Choker,
Normality and Pocketwatch, but not Kunai/Shuriken/Fan/Letter Opener/Art of War.
The native fields and frozen 24 counter columns retain their original meanings
and widths; deploy v41 with the BRIDGE158 queue fix, not the v40 exporter.
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
The wire always includes `current_action` (explicit null when none)
and `choice_list` (explicit [] when CHOOSE is unavailable). Normal frames and
error replies have separate presence contracts; missing/null screen payloads,
inventories, command lists or contradictory aliases fail before menu actions,
ACK handling or policy inference. See `json_presence_contracts` in the complete
word tables. Since v14, a present current action is its Java binary class name
(`Class.getName()`), not its simple name: anonymous actions have no simple name
but are still executing actions. The package and `$` suffix are preserved;
they are not replaced with null, a base-class name or a placeholder. Selection
semantics still come from the separately validated selection source/context.
`current_action_is_anonymous` is the actual reflection result, not a name
heuristic. `anonymous_action_contracts` inventories all eight stock and three
installed BaseMod anonymous actions by exact binary identity, source, effects,
native counterpart and audit status. Unknown identities, mismatched flags,
deprecated/custom unsupported semantics and stock wrappers claiming a selection
are rejected explicitly. A `settle` entry only permits waiting for its real
completion; it does not certify pending native effect/root-restore gaps.
Since v15 every monster also has `simulator_thief_state`: Looter/Mugger export
their actual `stolenGold`, `slashCount`, and `goldAmt`, including dead/escaping
entities; other species export explicit null. Missing state, invented counters,
or a non-thief with thief state are rejected. Canonical state, strict native
root restore, full dict comparison and future search keys retain these fields.
Attack count is not the room turn number; theft amount is not the removable
ThieveryPower. The compact leaf observation remains the frozen model projection,
not a complete restorable combat state.
Since v16 `simulator_lagavulin_state` carries whether damage still wakes the
monster and its completed idle-action count. This restores gameplay sleep after
the power list and distinguishes sleeping from the shared sleep/stun model token.
It does not reproduce animation flags or change that frozen observation token.
Since v17 CARD_REWARD exports `reward_item_present` from the actual `rItem`
reference. Noncombat standalone offers (including Neow) contain one card group,
not a missing outer reward inventory. Parent-owned offers require an observed
outer reward snapshot. Combat Discovery/Codex remain combat selections.
Since v18 the shop serializer preserves physical shelf slots and actual sold
status. Two read-only ShopScreen hooks retain initial card slots and Courier
replacement identities; StoreRelic/StorePotion supply their real slot/purchased
fields. Missing slots require explicit sale evidence, never inferred padding.
Since v19, explicitly audited events export actual execution phases and delayed
card-obtain status, including zero-gain results and forced multi-page dialogue.
Wheel spin is the stock CommunicationMod transport handle, not a new policy
choice. Hidden random outcomes are not exported. Bonfire's rarity-reward
selection context takes precedence over the generic purge UI flag.
Since v20, summon leaders export their actual minion-reference slots. Retired
replaced corpses remain in full raw records and an explicit historical snapshot;
only confirmed inert deaths leave the bounded search projection. Target commands
still use actual current UI indices. Wizard charge is restored explicitly, never
inferred from a new random template. The matching native extension must be rebuilt
for this full-state field and strict audited-minion replacement support.
Since v21 the last-monster-attack counter tracks only NORMAL monster damage,
not Constricted/other THORNS. Both sides cap it at HP before damage/revival,
matching Java lastDamageTaken and Shelled Parasite's actual life-steal amount.
The matching native build also fixes HP_LOSS Buffer protection, self-damage
sources and minimum Lizard Tail revival HP. No policy tensor or action IDs change.
Since v22 GRID exports `confirmation_card_uuid` separately from `selected_cards`:
single-card upgrade/transform/purge previews temporarily clear that list. The
UUID comes from the actual original hovered card, never the animated replacement.
The adapter confirms that preview once; staged events also match the requested
physical UUID before allowing confirmation. Missing fields are rejected.

Since v23 each monster also exports `simulator_writhing_mass_used_implant`:
the actual Writhing Mass `usedMegaDebuff` boolean, or explicit null for other
species. It is restored and compared, not inferred from recent moves or Parasite
ownership. A settled real root already loads the actual master deck, so its
used flag must not cause another curse/Omamori charge at native combat exit.
The native miscInfo states distinguish a newly simulated pending Implant from
an already-settled real one; both prohibit a second Implant. Rebuild the native
extension together with the Java helper.

Since v24 `simulator_move_state` contains the exact audited AI fields for
Time Eater, Red Slaver, Bronze Orb, Spiker, Champ and Bronze Automaton. V25 adds
Book of Stabbing's actual `stabCount`, including during its single-stab intent.
V26 adds both Louse species' `biteDamage`, Darkling's `nipDmg` and Guardian's
`dmgThreshold`, including during non-attacking/defensive intents. V27 adds
Giant Head's actual `count` (-6..5), including its A18 pre-battle decrement.
V28 adds Hexaghost's actual `activated`, `burnUpgraded`, `orbActiveCount` and
`damage[2].base` (Divider damage, -1 before activation). Other species have
explicit null. `monster_move_state_fields` lists each actual Java
field, type and native domain. Canonical state, strict native restoration and
diff retain all of them. One-shot flags are set at the actual intent-selection
or execution point; Champ and Automaton retain their own move counters instead
of guessing from global turns, buffs or recent move history. The native search
key already includes their miscInfo storage. This is not full pending-action
restoration or a claim that all monster-private fields have been audited.

Restore the Book's actual counter before checking visible intent hits: a single
stab is still one hit but must not reset its accumulated multi-stab count. A
multi-stab hit count conflicting with the actual counter is an error.

Random base damage is restored before checking visible attack damage; it must
not be guessed from a non-attacking intent. Darkling's constructor already uses
the ascension-specific range, so execution must not add another ascension bonus.
Guardian's next-cycle threshold increments on entering Defensive Mode and is
distinct from the remaining Mode Shift power amount.

Giant Head's large-attack intent, damage and next move all use its actual
countdown, not the global turn number. The decrement on A18+ happens after
the initial move selection, matching the game's pre-battle callback.

Hexaghost's cycle and flags use named miscInfo bits, separate from power storage;
its Divider damage has a dedicated integer included in the native search key.
Inferno upgrades Burn only in draw/discard and adds three Burn+ to discard;
Sear uses the actual upgrade flag. Activate does not roll again. Activating the
sixth orb sets Inferno before the queued RollMove sets it again, retaining both
real history entries. This does not claim arbitrary pending-action restoration.

Since v29 combat state also exports `simulator_bombs`, an explicit ordered list
of each TheBomb power's actual ID, countdown and private damage (or `[]`).
The bridge checks every instance against the actual power list and the three
aggregate counters. Native roots retain each independent hit; combining two
explosions would incorrectly deal only one damage to an Intangible enemy.
The old three damage-sum inputs remain unchanged for existing checkpoints.
Counts and per-bomb damage are never inferred from those totals. Native copies
and search keys retain the instances; rebuild the native extension with v29.

Since v30 Toolbox's actual `ChooseOneColorless` is a supported generated-card
selection source. It shares the checkpoint's existing `DISCOVERY` choice token,
but not DiscoveryAction's zero-cost effect: `generated_count` and
`generated_free_for_turn` explicitly preserve one normal-cost Toolbox card
versus one/two temporarily free Discovery cards. Other reward sources export
null for these parameters. The bridge checks source/parameters and pending
native metadata before choosing; native copies and search keys retain the cost
mode. The model tensor layout and action vocabulary do not gain a new token.

Parity-v31 also exports nullable `simulator_match_state`. For Match and Keep,
the human-authorized fixed transport always selects the first two remaining
public board slots. It maps each physical slot to stock CommunicationMod's
current CHOOSE index; it waits for the first flip, the pair's exact attempt
decrement and removal/flip-back, asynchronous card obtains, and the final
COMPLETE dialogue. It reads no hidden card identities or remembered faces and
does not invoke or modify the model for this minigame. Other events export null.
Parity-v32 additionally exports the actual `AbstractPlayer.gameHandSize` and
nullable `DrawReductionPower.justApplied`. They are transition metadata, not new
model channels. Canonical state, strict native root restore and read-back checks
retain both fields; a shadow/default draw count cannot stand in for real state.
Draw Reduction's numeric amount retains its duration instead of becoming 1.
The paired native build stores/ticks that duration and restores hand size only
on final removal. Ordinary one-layer inputs and tensor layouts are unchanged.

Parity-v33 corrects Collector's actual one-based `enemySlots` keys: native
left/near/leader slots reference Java keys 2/1/the leader. An empty map before
the first summon is valid; a nonempty partial/unknown map is rejected. Living
unreferenced minions are still errors, never dropped or padded. Gremlin Leader
and Reptomancer retain their separately audited reference permutations.

Parity-v34 exports actual `GridCardSelectScreen.isJustForConfirming`, distinct
from the stock merged `confirm_up`. Its `OBTAIN_CONFIRM` context receives the
entire actual group (Calling Bell, Pandora's Box and Neow), without fabricating
selected cards or a hovered-card UUID. The adapter confirms once and waits for
the same source/group to close; existing obtain-effect queue settling remains.
Normal upgrade/transform/purge previews keep their separate original-card rule.

Parity-v35 additionally exports WeMeetAgain's actual public offer objects: a
full master-card UUID, a physical potion slot, and the offered gold amount.
Typed kinds map the four frozen semantic choices to stock's compressed CHOOSE
indices, even when earlier buttons are disabled. Enrichment runs only when the
actual screen enum is EVENT, the mod is waiting for a command, and CHOOSE is
available; room-loading snapshots can legitimately have no initialized buttons.
Once actionable, missing or inconsistent targets remain errors. The INTRO/COMPLETE phase
and completed visible relic identity are transport receipts; INTRO never exposes
the future prize. Trade confirmation waits for the selected cost, received relic
and its screenless on-equip effects. No localized-name matching or forced leave
stands in for the model's decision. Other dynamic events still require their own
typed exporter; declaring their vocabulary does not prove bridge coverage.

Do not mix v35 Python with an older jar/native extension. Model tensor layouts
and checkpoint parameters are unchanged; these private fields affect simulation
and search, not the dimensions of the policy input. Source changes are not a
claim of a rebuilt or behavior-verified runtime.

Parity-v36 adds Designer's actual service variants, prices, counts, and eligible
master UUIDs on the public MAIN page. INTRO never exposes its private service
rolls. Its current pending service, exact GRID flags/source/candidates, physical
picks and confirmation are acknowledged before the final DONE dialogue can leave.
Random upgrade/transform targets are not exported in advance. This retains the
existing EVENT_OPT0..5 meanings and model tensors, and requires a matching rebuilt
Java exporter. It is source/build checked, not proof of naturally covered services.
Parity-v37 adds NoteForYourself's actual public card instance and current
acquire/store progress. The offered card stays null during INTRO. CHOOSE uses
the same complete card serializer as master/GRID cards, not a local default.
After acquisition its flag-free GRID is REMOVE (not OBTAIN), including the new
card among eligible candidates. The exact selected `saveCard` and final master
deck acknowledge storage even if the newly acquired card itself is stored back.
Native Note insertion shares the normal obtain core but bypasses Omamori's
effect-constructor interception; other obtain sources keep their original guard.
This requires the rebuilt native extension and matching Java/Python version;
source/build checks do not claim natural coverage of every Note or service branch.

Parity-v38 adds Falling's actual public target objects. INTRO does not expose
its preselected future cards. CHOICE carries full master-card snapshots from
the event's `skillCard`/`powerCard`/`attackCard`, not a lookup by translated name
or `(id, upgrades)`. Three raw button slots retain disabled positions; frozen
skill0/power1/attack2 map to the compressed CHOOSE index. When every type is
absent or bottled, the real one-button choice maps to native leave3. The RESULT
receipt checks the exact requested UUID removal and every surviving instance,
so same-printed cards with different accumulated values cannot be interchanged.
Model tensors, rule scores and native transition logic are unchanged. Deploy
matching rebuilt Java/Python together, only at an isolated run boundary.

Parity-v39 adds N'loth's actual owned relic offers and current result phase.
Its receipt accounts for bottle removal and Necronomicon's direct first-curse
removal; an already-owned N'loth's Gift instead grants Circlet without consuming
the donor. Native losses share `GameContext.loseRelic`. Du-Vu Doll now preserves
its actual cached counter through export/restore and battle initialization:
ordinary deck mutations refresh it, but Necronomicon's raw group removal does
not. It is not safe to reconstruct this counter from the current curse count.
This requires the rebuilt native extension plus matching Java/Python. Existing
checkpoint tensors and action meanings are unchanged; a build alone does not
prove natural coverage of these combinations.

A fresh combat attachment already inside the opening Gambling Chip or Toolbox
selection must initialize its complete actual public root before comparing
physical candidates. The native `initialize_selection_root` entry retains the
existing pending task, queued continuation and RNG; it only accepts the untouched
first-turn opener, with no played/discarded cards or selected instances. All
subsequent selections retain strict UUID/order checks. An older native extension
without this entry is rejected at agent initialization, not silently substituted.

When freezing an isolated runtime, archive all tracked application/source roots
at the exact commit, excluding `runs/` (historical outputs/checkpoints, not runtime
source). Record that exclusion and verify the relevant Git blobs after extraction.
Keep the selected model, vectors, native extension and mod jar as separately
hashed inputs; do not duplicate unrelated historical checkpoints in each runtime.
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
