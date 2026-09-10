# Card representation comparison

Checkpoint: epoch 7000 (`last-epoch7000-resume.pt`, SHA-256
`69541bffe81c90e80738b2d8c2599480c93e8ca706132c833fe0a79fde53f3a`).

The encoder representation for one sample is the mean hidden state over its
`ACTION_RULE_BEGIN ... ACTION_RULE_END` span. `CARD_REF_*` and `WORD_CARD`
are excluded from the pooling positions, so the anonymous card identity token
does not directly enter the pooled positions. This remains a contextual
representation: the reference token can still affect neighboring hidden states
through attention. The same card is sampled from random combat contexts and
its context-level vectors are averaged. Four data shards
were merged; the merge is a sample-count weighted average of each shard's
normalized card mean, followed by normalization.

## Coverage

| Group | Versions | Card names |
|---|---:|---:|
| Contains damage | 74 | 36 |
| Contains block | 28 | 14 |
| Damage only | 70 | 34 |
| Block only | 24 | 12 |
| Both | 4 | 2 (`Flame Barrier`, `Iron Wave`) |
| Neither | 55 | 28 |

The vectors have dimension 256. Most versions use 128 contexts after merging;
rare `Searing Blow` upgrade levels have fewer samples (42, 5, and 3).

## Comparison

| Metric | Value |
|---|---:|
| Damage-only vs block-only centroid cosine | 0.896615 |
| Same centroid distance (unit-vector Euclidean) | 0.454719 |
| Mean cosine within damage-only group | 0.804143 |
| Mean cosine within block-only group | 0.895666 |
| Mean cosine across damage-only/block-only | 0.764102 |
| Separation gap: mean within minus cross | 0.085802 |
| Leave-one-out nearest-centroid balanced accuracy | 95.714% |

The cosine values are high because the encoder has a large shared direction.
The cross-group value is nevertheless lower than both within-group values, and
the 0.0858 gap plus 95.7% balanced nearest-centroid accuracy show that the
damage/block distinction is present in the representation. The groups are not
orthogonal; this is a directional separation on top of a shared state/template
representation.

The mean cosine between base and upgraded versions of the same card is 0.969486
(80 base/upgrade pairs). This is higher than the damage/block group separation,
so card identity, upgrade, and rule-template information remain stronger than
the coarse effect label. That is expected for a contextual encoder and means
the vector should not be interpreted as a pure damage-vs-block axis.

## Extreme cross-group pairs

Lowest similarities:

- `Immolate@up0` vs `Metallicize@up1`: 0.549549
- `Immolate@up0` vs `Metallicize@up0`: 0.552395
- `Bludgeon@up0` vs `Shrug It Off@up1`: 0.554045

Highest similarities:

- `Combust@up1` vs `Metallicize@up1`: 0.969576
- `Combust@up0` vs `Metallicize@up0`: 0.969327
- `Fire Breathing@up0` vs `Metallicize@up0`: 0.967711

The high-similarity exceptions are trigger/status templates: `Combust` and
`Fire Breathing` deal damage through ongoing effects, while `Metallicize` and
`Feel No Pain` provide block through ongoing effects. They are close because
their rule structure and trigger language are similar despite opposite effect
flags. The low-similarity pairs are large direct attacks versus compact block
or status programs.

Generated JSON, including all 256-dimensional vectors and centroids:
`docs/card-representations-damage-vs-block-epoch7000.json`.

PCA projection (PC1 explains 24.1%, PC2 explains 13.6%):

`docs/card-representations-damage-vs-block-epoch7000-pca.png`
