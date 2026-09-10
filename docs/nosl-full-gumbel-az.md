# NOSL Full Gumbel AlphaZero 方案

状态：初版已实现并贯通训练、评测和前端（2026-09-10）。
`rootSearchMode=0/1` 分别保留 PUCT 和 legacy Gumbel 作为对照，
`rootSearchMode=2` 为 Full Gumbel AlphaZero。

## 1. 目标和边界

局内唯一目标是最大化战斗结束时的玩家血量。统一使用

```text
z = clamp(final_player_hp / 200, 0, 1)
```

搜索中的叶值和训练 value target 都使用这个标尺；不要在搜索时混用
`HP/maxHP`、伤害负值或 PPO 的另一套尺度。

NOSL（no fixed world line）是硬约束：一次 simulation 从根状态 clone 后，
用独立 nonce 对隐藏 RNG 重新确定一次，之后在该 branch 上推进；不能读取、
复用当前真实世界线的抽牌 RNG。树节点的 key 只包含可观察且会影响后续
转移的状态字段。抽牌堆顺序、隐藏 RNG 和 hand slot 的排列不进入 key；
手牌/抽牌堆/弃牌堆的无序组成及其所有 transition-relevant card fields 要
进入 key。

搜索的 native action 必须是完整动作（卡牌加目标）。只有向 whole-run
网络或环境提交时才聚合为 thin action；不同目标不能在打分前合并，否则会
把目标选择的 value 混在一起。

## 2. 与现有三种模式的关系

| 模式 | 根节点 | 非根节点 | 用途 |
|---|---|---|---|
| 0 | PUCT | PUCT | 现有基线 |
| 1 | Gumbel-Top-k + 简化 Sequential Halving | PUCT | legacy 对照；不能称为完整 Gumbel |
| 2 | Gumbel-Top-k + completed-score Sequential Halving | Full-Gumbel 确定性选择 | 本方案 |

模式 2 不需要 Dirichlet root noise。Dirichlet 可以作为单独消融，但默认
`alpha=epsilon=0`，不能把它和 Gumbel 噪声混成一个“标准”算法。

## 3. 根节点算法

设合法的完整动作集合为 \(A\)，网络给出未归一化 policy logits
\(\mu_a\)。如果网络只有 thin card head，一个卡牌的多个目标变体应令

```text
mu_(card,target) = mu_card - log(number_of_legal_targets_for_card)
```

这样所有目标变体的 prior 之和仍等于该卡牌的 prior。优先使用原始 logits；
用 `log(prior)` 也可以，但必须在相同的 mask、temperature 和动作拆分下进行。

1. 对每个合法完整动作只采样一次

   ```text
   g_a ~ Gumbel(0, 1)
   r_a = mu_a / prior_temperature + gumbel_scale * g_a
   ```

   同一组 `g_a` 必须同时用于候选筛选和最后的 score，不能每轮重新采样。

2. Gumbel-Top-k 候选集

   ```text
   m = min(gumbel_candidates if > 0 else |A|, |A|, simulations)
   C = argtop_m(r_a)
   ```

   `m <= simulations` 保证每个候选至少能获得一次新访问。`m=2` 适合严格
   Bash/Clash 顺序 fixture，正式战斗建议从 16 开始并做 `m=all` 消融。

3. 每个候选维护 `N[a]`、`W[a]`、`Q[a]=W[a]/N[a]`。未访问边不使用一个
   固定的“差一点”的 FPU 值，而使用第 4 节的 completed-Q 估计。

4. 对候选执行 Sequential Halving。设当前候选数为 `k`，剩余总预算为 `B`，
   阶段数为 `L=ceil(log2(k))`。每一阶段给当前候选尽量相等的新增 quota：

   ```text
   quota = max(1, floor(B / (k * remaining_phases)))
   ```

   余数按 round-robin 补齐，保证总 simulation 数不超过 `simulations`，并
   保证每个活跃候选在淘汰前至少得到一次新访问。阶段结束后按
   `completed_score` 保留 `ceil(k/2)` 个候选；剩余预算和下一阶段重新计算。

5. 预算耗尽后，只在“访问次数达到最大值”的候选中取

   ```text
   argmax_a (mu_a / prior_temperature + gumbel_scale*g_a
             + sigma(completedQ_a))
   ```

   这一步仍使用同一组 Gumbel。若有完全相同的 score，使用完整 action bits
   作为稳定 tie-break。评估时可固定 nonce 以复现；训练时 nonce 不同会产生
   按 prior 分布的根探索，而不需要额外的根 Dirichlet。

## 4. completed Q、混合 value 和 Q 归一化

对一个节点，已访问边的经验均值为 \(\hat Q_a\)，网络 value 为
\(\hat v_\pi\)，网络 prior 为 \(\pi_a\)。先计算未访问边的混合估计：

\[
v_{mix} = \frac{\hat v_\pi +
 {\sum_b N_b\over \sum_{b:N_b>0}\pi_b}
 \sum_{a:N_a>0}\pi_a\hat Q_a}
 {1+\sum_bN_b}.
\]

当没有已访问边、prior 分母为零或数值非有限时，退化为
`v_mix = v_hat_pi`。随后

```text
completedQ[a] = Q[a]                 if N[a] > 0
                v_mix                otherwise
```

建议按 MuZero 的方式做 bounded Q normalization。令 `observed` 为
`v_hat_pi` 与所有已访问 raw Q 的集合，

```text
lo = min(observed)
hi = max(observed)
q_norm[a] = (completedQ[a] - lo) / (hi - lo)  if hi > lo
            0.5                                otherwise
```

对于本项目的 `[0,1]` HP value，也可以在日志中同时保留 raw Q；不能因为
做了归一化就丢掉原始 HP 价值。若将来改为零和视角，需显式处理玩家换手
和符号，不能隐式复用当前 HP 标尺。

根节点和 Full-Gumbel 非根节点都使用

```text
sigma(Q[a]) = (gumbel_max_visit_init + max_b N[b])
              * gumbel_value_scale * q_norm[a]
```

推荐 `gumbel_max_visit_init=50`、`gumbel_value_scale=1.0`。当前代码默认的
`0.1` 可作为 Atari 风格的小尺度消融，但对已压缩到 `[0,1]` 的 HP value
会明显让 prior 占优，应单独记录。

## 5. Full-Gumbel 的非根节点选择

非根节点不再采样新的独立 Gumbel，也不使用根节点的 Gumbel。对该节点先
构造改进 policy：

```text
pi_improved[a] = softmax(mu_a / prior_temperature + sigma(completedQ[a]))
```

然后使用论文 Eq. 5.14 的确定性动作：

```text
a* = argmax_a (
        pi_improved[a] - N[a] / (1 + sum_b N[b])
      )
```

它等价于选择一个动作，使“执行下一次 simulation 后的归一化 visit count”
最接近 `pi_improved`，不会在每个深层节点额外引入采样方差。模式 0/1
仍可使用 PUCT，便于比较“根部 Gumbel 的收益”和“完整内部选择”的收益。

## 6. 一次 NOSL 搜索的伪代码

```text
function search(root_public_state, root_logits, nonce, S, m, D):
    A = enumerate_complete_legal_actions(root_public_state)
    root = Node(root_public_state)
    expand(root, root_logits, A)

    for a in A:
        g[a] = one_gumbel(nonce, a)
        root_score[a] = logit(a)/T + c_g*g[a]
    active = top_m(root_score, min(m, |A|, S))

    while simulations_done < S:
        if root_round_finished(active):
            active = keep_top_half(active, completed_score(root, active))
            update_equal_quota(active, remaining_budget)

        # Each pending path gets its own NOSL world.  A batch may contain
        # several paths; virtual visits are installed before requesting leaves.
        world = clone(root_public_state)
        redeterminize_hidden_rng(world, nonce, simulation_index)
        node = root
        path = [root]
        depth = 0

        while not terminal(world) and (D == 0 or depth < D):
            if node is root:
                a = select_root_candidate(active, root, g)
            elif full_gumbel_mode:
                q = completed_Q(node)
                pi = softmax(node.logits/T + sigma(q))
                a = argmax(pi[a] - N[a]/(1 + sum(N)))
            else:
                a = puct(node)
            install_virtual_visit(path, a)
            execute_complete_action(world, a)
            node = get_or_create_public_child(node, a, public_state_key(world))
            path.append(node)
            depth += 1

            if node is unexpanded:
                break

        if terminal(world):
            v = clamp(projected_exit_hp(world) / 200, 0, 1)
            backup(path, v)
        else:
            queue_leaf_for_one_batched_network_call(path, public_observation(world))

        when_batch_ready:
            (values, policies) = network_forward(all_queued_observations)
            for leaf in batch:
                if leaf was truncated:
                    expand = false
                else:
                    expand_leaf(leaf, policies[leaf])
                remove_virtual_visits(leaf.path)
                backup(leaf.path, values[leaf])

    q = completed_Q(root)
    score = root_logits/T + c_g*g + sigma(q)
    eligible = active edges with maximal N
    action = argmax(score over eligible)
    target_complete = softmax(root_logits/T + sigma(q))
    target_thin[i] = sum(target_complete[a] for a with thin(a)=i)
    return action, target_thin, root_statistics
```

`D=0` 表示不设人为深度截断，但必须有 simulator terminal 检查和最大
decision/cycle guard。截断叶只使用 value network bootstrap，不应把截断状态
误记成战斗终局。

## 7. 训练接口

每个战斗决策保存：

```text
observation, legal_mask, complete/root policy target, z=final_hp/200,
root visits, raw Q, completed Q, normalized Q, gumbel scores,
NOSL nonce, model version, depth/truncation/terminal flags
```

训练时对 target `stop_gradient`：

```text
L_policy = KL(target_complete || masked_softmax(policy_logits / T))
L_value  = Huber(value_pred, z)       # MSE 也可作为消融
L_total  = w_policy*L_policy + w_value*L_value + w_entropy*L_entropy
```

一次战斗的最终 HP `z` 可以复制给该战斗的所有决策；它是客观训练 target。
搜索中的叶 value 只负责当前 simulation 的 bootstrap，不能替代最终 HP
target，也不能把某次 root value 当作另一条世界线的事实结果。

建议保留 replay buffer，但按完整信息集样本存储，不保存隐藏 deck order。
重放时重新生成 Gumbel 只适用于重新搜索；若直接重放已生成的 target，必须
保留当时的 target 和模型版本，不能把新的随机 Gumbel 当成旧 target 的一部分。
外层 whole-run policy 冻结时，root 和 leaf 必须使用同一份冻结外层模型，
只有 combat specialist 的参数参与 optimizer。

## 8. 并行和效率形态

C++ worker 负责 clone、NOSL redeterminization、动作执行、树统计和 virtual
visits；learner 将多个 worker 的 leaf observation 拼成一次 GPU forward，再
按 request id 回填 value/policy。一个批次内的叶子在选择时尚未有真实 value，
所以必须先临时增加路径上的 virtual visits；回填时撤销并做真实 backup。

令 `A` 为平均合法动作数，`H` 为实际展开长度（有限 depth 时
`H <= D`，无限 depth 时为到 terminal 的平均 horizon），`S` 为 simulations，
`B` 为 leaf batch：

```text
root Gumbel/top-k       O(A log A)   # 可用 partial-sort 降为 O(A log m)
simulator/tree work     O(S * H * A)
network forwards        ceil(nonterminal_leaves / B)
tree memory             O(S * H * A) worst case, usually O(S * H)
```

NOSL 的随机 clone/redeterminization 是每条 simulation 一次；不能为了提速而
把同一个 hidden RNG branch 共享给 sibling。主要可调的吞吐旋钮是 `B`、worker
批大小和候选数 `m`，而不是删除 NOSL 重定向。

## 9. 建议参数

| 参数（C++ 名称） | 建议起点 | 作用和注意事项 |
|---|---:|---|
| `rootSearchMode` | `2` | 0 PUCT，1 legacy，2 Full Gumbel |
| `simulations` | smoke `128`；正式 `512` | 根预算；必须计入终局 simulation |
| `gumbelCandidates` | `16`；严格二选一 `2` | `0` 表示全部合法动作；不得大于 `simulations` |
| `maxDepth` | `3`，再测 `0` | 0 为 unlimited；需要 cycle/decision guard |
| `leafBatchSize` | `32` 或 `64` | 越大 IPC/GPU 越高效，但 virtual visit 偏差也更明显 |
| `priorTemperature` | `1.0` | 只改变 prior logits；高温不是替代候选覆盖 |
| `gumbelScale` | `1.0` | Gumbel 噪声尺度；训练保留，评估可设 0 做消融 |
| `gumbelMaxVisitInit` (`c_visit`) | `50` | 控制初期 value 对 logits 的相对影响 |
| `gumbelValueScale` (`c_scale`) | `1.0` | HP/200 已 bounded；`0.1` 会更信 prior |
| `gumbelRescaleQ` | `true` | 建议开启；同时记录 raw Q |
| `gumbelUseMixedValue` | `true` | 未访问边用 `v_mix`，避免 sparse root 全部相同 FPU |
| `cPuct` | `1.5` | 仅 mode 0/兼容 fallback 使用 |
| `fpuReduction` | `0.01` | 仅 PUCT/legacy fallback；不作为 Full-Gumbel completed Q |
| `dirichletAlpha/Epsilon` | `0/0` | Full Gumbel 默认关闭，另做消融 |
| `rootActionNoise` | Gumbel | 训练每根一次；不要每 simulation 重采样 |
| `rootActionTemperature` | train `1`；eval `0`/score | 不要把 target temperature 和 prior temperature 混为一项 |
| `valueScale` | `200` | 所有终局/叶值统一 HP 标尺 |
| `policy/value loss weight` | `1/1` 起步 | 记录各项 loss 和梯度范数，避免 value 爆大掩盖 policy |
| `learning rate` | `1e-4` | 与当前冻结外层实验一致，之后再做 schedule 消融 |

推荐首轮配置：

```text
mode=2, simulations=128, candidates=16, depth=3, leaf_batch=64,
prior_temperature=1, gumbel_scale=1, c_visit=50, c_scale=1,
mixed_value=true, q_normalize=true, dirichlet=(0,0)
```

通过严格 Bash/Clash fixture 后，再使用 `simulations=512` 和 `depth=0` 做
长战斗；训练和评估都应同时跑 PUCT、legacy mode 1，才能区分“Gumbel 根部
覆盖”与“完整内部选择”的贡献。

## 10. 验证矩阵和日志

### 数学/实现单测

1. Gumbel-Top-k 多 nonce 的选择频率接近 masked prior；同一次搜索中筛选和
   最终 score 使用完全相同的 `g`。
2. `v_mix` 与公式逐项相等；无 visited edge、零 prior、`hi=lo` 时不产生
   NaN/Inf。
3. 每个 Sequential Halving 阶段预算和总 `N` 精确等于请求的 simulations，
   每个存活候选至少有一次访问。
4. Eq. 5.14 在手算例子上给出预期边；非根不产生额外 Gumbel。
5. 完整目标动作先打分再聚合；Bash 的多个目标不会覆盖彼此的统计量。
6. 相同 public state、不同 hidden nonce 可以产生不同 child outcome，但不会
   因为固定当前世界线 RNG 而完全相同；hidden pile order 不改变 state key。
7. virtual visits 在 batch 回填后完全撤销，`root.N` 与子边访问总和的差异
   只来自显式终局/截断处理。

### 行为矩阵

固定 normal draw pile 的 energy=2、唯一可打牌为 Bash/Clash 的 root，运行：

```text
simulations = 32, 64, 128, 256, 512
candidates  = 2, 3, all
depth       = 1, 2, 3, 0
mode        = PUCT, legacy Gumbel, Full Gumbel
prior       = model, uniform, disabled
```

记录 Bash->Clash、Clash->Bash、End Turn、root candidate coverage，以及每条
边的 `N/raw Q/completed Q/normalized Q/gumbel/score`。特别加入 depth 恰好
卡在 End Turn 前后的状态和多个 NOSL future nonce。

### 长战斗/楼层评估

到 Act 2 为止按楼层分桶，并单独报告普通战斗、精英和 boss：平均最终 HP、
胜率、死亡层、每决策 leaf 数、树节点数、截断率、fallback 率、
`env_steps/s` 和 GPU forward 次数。比较时固定 checkpoint、seed、nonce
协议和 outer policy，不能只比较一次 Bash/Clash 结果。

## 11. 当前实现状态

当前 mode 2 已完成以下链路：

- C++ 根节点使用一次性 Gumbel Top-k、considered-visit schedule 和
  completed-score；后继节点使用 Full Gumbel 的确定性 improved-policy 选择，
  不再使用 PUCT。
- 每个 simulation 从根状态 clone，并使用独立 nonce 做 NOSL
  redeterminization；公开状态按可观察且 transition-relevant 的字段合并。
- C++ 分别返回 raw visit policy 和不含一次性 Gumbel noise 的 completed
  policy target；Python learner 在 mode 2 使用 completed target。
- 训练 CLI、SB3 checkpoint 字段、完整爬塔评测和交互式 Watch 使用相同的
  mode 及 Gumbel 参数。旧 checkpoint 缺少这些字段时默认回退到 PUCT。
- 前端可切换 Direct、PUCT、Simplified Gumbel 和 Full Gumbel AZ，并可调整
  candidates、Gumbel scale、Q scale、visit init、Q normalization 和 mixed Q。
- 搜索树按完整 native action 展示 visits、raw Q、completed-Q transform、
  Gumbel score、候选状态和随机公开后继节点。

本地验收包括 native `simulations=8/64`、`depth=1/2/0`、
`leaf_batch=1/16`，以及 7.68M checkpoint 的严格 Bash/Clash fixture。
行为质量仍需使用固定 checkpoint、seed 和 nonce 的多战斗矩阵评估，不能由
单个牌序 case 推断。
