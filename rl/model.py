from __future__ import annotations

from collections.abc import Mapping
from math import prod

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from env.core import encode
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env.core.card_language import (
    MAX_PROGRAM_TOKENS,
    TOKEN_VOCAB,
    encoded_card_programs,
    encoded_card_version_programs,
)
from env.core.combat_language import TOKEN_VOCAB as COMBAT_TOKEN_VOCAB


def uses_global_map_observation(model_or_space) -> bool:
    """Whether a model/space expects the optional full-map observation fields."""
    space = getattr(model_or_space, "observation_space", model_or_space)
    return "map_full_rooms" in getattr(space, "spaces", {})


def uses_map_dag(model) -> bool:
    """Whether a loaded policy uses the topology-aware full-map encoder."""
    policy = getattr(model, "policy", model)
    core = getattr(policy, "core", None)
    if core is not None and hasattr(core, "map_dag_enabled"):
        return bool(core.map_dag_enabled)
    cfg = getattr(policy, "_core_cfg", None)
    if isinstance(cfg, Mapping):
        return bool(cfg.get("map_dag", False))
    kwargs = getattr(policy, "policy_kwargs", None)
    return bool(isinstance(kwargs, Mapping) and kwargs.get("map_dag", False))


def uses_deck_size_observation(model_or_space) -> bool:
    """Whether a model/space expects the explicit normalized deck size."""
    space = getattr(model_or_space, "observation_space", model_or_space)
    return "deck_size" in getattr(space, "spaces", {})


def uses_neow_observation(model_or_space) -> bool:
    """Whether a model/space carries per-option Neow blessing features."""
    space = getattr(model_or_space, "observation_space", model_or_space)
    return "neow_bonus_ids" in getattr(space, "spaces", {})


def uses_event_option_observation(model_or_space) -> bool:
    """Whether a model/space carries per-option normal-event features."""
    space = getattr(model_or_space, "observation_space", model_or_space)
    return "event_option_ids" in getattr(space, "spaces", {})


def uses_combat_language(model_or_space) -> bool:
    """Whether a model/space carries tokenized intent and status programs."""
    space = getattr(model_or_space, "observation_space", model_or_space)
    return "player_status_language" in getattr(space, "spaces", {})


def uses_card_language(model) -> bool:
    """Whether a loaded policy uses semantic card-effect programs."""
    space = getattr(model, "observation_space", model)
    if "draw_card_ids" in getattr(space, "spaces", {}):
        return True
    policy = getattr(model, "policy", model)
    core = getattr(policy, "core", None)
    encoder = getattr(core, "encoder", None)
    if encoder is None:
        encoder = getattr(policy, "features_extractor", None)
    shared = getattr(encoder, "shared", None)
    if shared is None or "card" not in shared:
        return False
    return isinstance(shared["card"], CardLanguageEmbedding)


def _embedding_dim_for(vocab_size: int) -> int:
    """Heuristic embedding width from vocabulary size (fasttext-style rule of
    thumb, ~1.6 * vocab**0.56, clamped). Used when the caller doesn't pin one."""
    raw = round(1.6 * vocab_size ** 0.56)
    return max(4, min(64, int(raw)))


def derive_categorical_fields(
    observation_space: spaces.Dict,
) -> dict[str, dict[str, int]]:
    """Auto-detect categorical groups from the obs space itself.

    encode.obs_space() builds every group as a Box; the categorical id groups
    use an integer dtype (`cat(...)`, np.int64) and the scalar groups use float
    (`scal(...)`, np.float32). So "integer dtype == categorical" is the single
    source of truth — no hand-maintained key list to drift out of sync with
    encode.py. vocab_size = high + 1 (encode builds cat() as Box(0, n-1)), and
    embedding_dim is chosen by `_embedding_dim_for`.
    """
    fields: dict[str, dict[str, int]] = {}
    for key, subspace in observation_space.spaces.items():
        if not isinstance(subspace, spaces.Box):
            continue
        if subspace.dtype.kind not in ("i", "u"):
            continue
        vocab_size = int(np.max(subspace.high)) + 1
        fields[key] = {
            "vocab_size": vocab_size,
            "embedding_dim": _embedding_dim_for(vocab_size),
        }
    return fields


class StSMLPExtractor(BaseFeaturesExtractor):
    """Encode Dict observations with embeddings for categorical ID fields.

    By default the categorical groups are auto-detected from `observation_space`
    (integer-dtype Boxes; see `derive_categorical_fields`), so the extractor stays
    in sync with encode.obs_space() with no key list to maintain. Pass an explicit
    `categorical_fields` only to override vocab/embedding sizes, e.g.

        {"hand_ids": {"vocab_size": 371, "embedding_dim": 32}}

    Every Dict key not treated as categorical is flattened and fed as numeric.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        categorical_fields: Mapping[str, Mapping[str, int]] | None = None,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        if categorical_fields is None:
            categorical_fields = derive_categorical_fields(observation_space)

        obs_keys = set(observation_space.spaces)
        unknown_keys = set(categorical_fields) - obs_keys
        if unknown_keys:
            raise ValueError(
                f"categorical_fields contains unknown observation keys: "
                f"{sorted(unknown_keys)}"
            )

        self.field_order = list(observation_space.spaces.keys())
        self.categorical_keys = set(categorical_fields)

        self.embeddings = nn.ModuleDict()
        input_dim = 0

        for key in self.field_order:
            subspace = observation_space.spaces[key]

            if not isinstance(subspace, spaces.Box):
                raise TypeError(
                    f"{key!r} must be a Box space for this extractor; "
                    f"got {type(subspace).__name__}"
                )

            field_size = int(prod(subspace.shape)) if subspace.shape else 1

            if key in self.categorical_keys:
                spec = categorical_fields[key]
                vocab_size = int(spec["vocab_size"])
                embedding_dim = int(spec["embedding_dim"])

                if vocab_size < 2:
                    raise ValueError(
                        f"{key!r}: vocab_size must be at least 2, got {vocab_size}"
                    )

                self.embeddings[key] = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    padding_idx=0,
                )
                input_dim += field_size * embedding_dim
            else:
                input_dim += field_size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, features_dim),
            nn.GELU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded: list[torch.Tensor] = []

        for key in self.field_order:
            x = observations[key]

            if key in self.categorical_keys:
                # Expected ID range: [0, vocab_size - 1].
                # Keep this strict: out-of-range IDs should crash early.
                x = x.long()
                x = self.embeddings[key](x)
                x = x.reshape(x.shape[0], -1)
            else:
                x = x.float().reshape(x.shape[0], -1)

            encoded.append(x)

        return self.encoder(torch.cat(encoded, dim=1))


# ============================ transformer extractor ============================
#
# The flat MLP extractor above pours all 42 obs groups into one Linear(7401->512),
# which throws away the structure that matters in StS: the 30 relic slots, 64
# card-select slots, 10 hand slots etc. are permutation-equivalent, share no
# parameters, and their PAD entries are fed in as noise. StSTransformerExtractor
# instead tokenizes the obs — one token per *entity instance* (each card, each
# monster, each relic) plus a few summary/global tokens — and runs a small
# Transformer encoder so entities can attend to each other. A learnable [CLS]
# token reads the sequence out to a fixed `features_dim` for the SB3 heads.
#
# Embedding tables are shared by *semantic vocabulary*, not by obs key: the one
# card table serves hand/offer/select/shop, and a per-family "role" embedding
# tells the model whether a card token is in hand or on a shop shelf.

# Shared-embedding widths. Card / move / relic are pinned wider (richest
# semantics, drive most decisions); the rest fall back to the heuristic.
_SHARED_EMB_DIMS = {
    "card": 64,
    "move": 64,
    "relic": 48,
}


class CardLanguageEmbedding(nn.Embedding):
    """Card-id embedding enriched by the card's tokenized effect program.

    A small 1-D convolution preserves local operator/argument order, masked
    mean-pooling yields one semantic vector, and that vector is added to the
    normal learned card-id row. Non-Ironclad cards have an all-PAD program and
    therefore retain the plain ID embedding behavior.
    """

    LANGUAGE_VERSION = 4

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx=0)
        programs = torch.as_tensor(encoded_card_programs(), dtype=torch.long)
        if programs.shape[0] != num_embeddings:
            raise ValueError(
                f"card language has {programs.shape[0]} rows, expected "
                f"{num_embeddings}"
            )
        self.register_buffer("program_ids", programs, persistent=False)
        version_programs = torch.as_tensor(
            encoded_card_version_programs(), dtype=torch.long
        )
        self.register_buffer(
            "version_program_ids", version_programs, persistent=False
        )
        token_dim = 32
        self.token_embedding = nn.Embedding(
            len(TOKEN_VOCAB), token_dim, padding_idx=0
        )
        self.local_encoder = nn.Conv1d(
            token_dim, embedding_dim, kernel_size=3, padding=1
        )
        self.program_position = nn.Parameter(
            torch.empty(MAX_PROGRAM_TOKENS, token_dim)
        )
        nn.init.normal_(self.program_position, std=0.02)
        self.effect_scale = nn.Parameter(torch.tensor(0.1))

    def _encode_programs(self, programs: torch.Tensor) -> torch.Tensor:
        shape = programs.shape
        flat = programs.reshape(-1, shape[-1])
        mask = flat != 0
        positions = self.program_position[:shape[-1]].unsqueeze(0)
        tokens = self.token_embedding(flat) + positions
        tokens = (tokens * mask.unsqueeze(-1)).transpose(1, 2)
        encoded = torch.nn.functional.gelu(self.local_encoder(tokens)).transpose(1, 2)
        encoded = encoded * mask.unsqueeze(-1)
        pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled.reshape(*shape[:-1], self.embedding_dim)

    def effect_vectors(self, ids: torch.Tensor) -> torch.Tensor:
        return self._encode_programs(self.program_ids[ids.long()])

    def version_effect_vectors(
        self, ids: torch.Tensor, upgrades: torch.Tensor
    ) -> torch.Tensor:
        versions = upgrades.long().clamp(0, 1)
        programs = self.version_program_ids[ids.long(), versions]
        return self._encode_programs(programs)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return super().forward(ids) + self.effect_scale * self.effect_vectors(ids)

    def effective_weight(self) -> torch.Tensor:
        ids = torch.arange(self.num_embeddings, device=self.weight.device)
        return self(ids)


class PileCardLanguageEncoder(nn.Module):
    """Connect per-card DSL vectors into an ordered pile-level representation."""

    def __init__(self, card_dim: int, output_dim: int, max_cards: int) -> None:
        super().__init__()
        self.max_cards = max_cards
        self.card_position = nn.Parameter(torch.empty(max_cards, card_dim))
        nn.init.normal_(self.card_position, std=0.02)
        self.upgrade_projection = nn.Linear(1, card_dim)
        self.count_projection = nn.Linear(1, card_dim)
        self.connect = nn.Sequential(
            nn.Conv1d(card_dim, card_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(card_dim, card_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Linear(card_dim, output_dim)

    def forward(
        self,
        card_ids: torch.Tensor,
        upgrade_sums: torch.Tensor,
        counts: torch.Tensor,
        upgraded_counts: torch.Tensor,
        card_embedding: CardLanguageEmbedding,
    ) -> torch.Tensor:
        mask = card_ids.long() != 0
        base_vectors = card_embedding.version_effect_vectors(
            card_ids, torch.zeros_like(card_ids)
        )
        upgraded_vectors = card_embedding.version_effect_vectors(
            card_ids, torch.ones_like(card_ids)
        )
        base_counts = (counts - upgraded_counts).clamp_min(0)
        version_vectors = (
            base_vectors * base_counts.unsqueeze(-1)
            + upgraded_vectors * upgraded_counts.unsqueeze(-1)
        ) / counts.unsqueeze(-1).clamp_min(1)
        x = (
            version_vectors
            + self.upgrade_projection(upgrade_sums.float().unsqueeze(-1))
            + self.count_projection(torch.log1p(counts.float()).unsqueeze(-1))
            + self.card_position[:card_ids.shape[-1]].unsqueeze(0)
        )
        x = x * mask.unsqueeze(-1)
        x = self.connect(x.transpose(1, 2)).transpose(1, 2)
        x = x * mask.unsqueeze(-1)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.output(pooled)

    def singleton_vectors(
        self, card_embedding: CardLanguageEmbedding, upgrades: int = 0
    ) -> torch.Tensor:
        ids = torch.arange(
            card_embedding.num_embeddings, device=card_embedding.weight.device
        ).unsqueeze(1)
        upgrade_sums = torch.full_like(ids, upgrades / 5, dtype=torch.float32)
        counts = torch.ones_like(ids, dtype=torch.float32)
        upgraded_counts = torch.full_like(ids, int(upgrades > 0), dtype=torch.float32)
        return self(
            ids, upgrade_sums, counts, upgraded_counts, card_embedding
        )


class CombatProgramEncoder(nn.Module):
    """Encode variable-length intent/status DSL programs into entity vectors."""

    LANGUAGE_VERSION = 1

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        token_dim = 32
        self.token_embedding = nn.Embedding(
            len(COMBAT_TOKEN_VOCAB), token_dim, padding_idx=0
        )
        self.local_encoder = nn.Conv1d(
            token_dim, token_dim, kernel_size=3, padding=1
        )
        self.output = nn.Linear(token_dim, output_dim)

    def forward(self, programs: torch.Tensor) -> torch.Tensor:
        shape = programs.shape
        flat = programs.long().reshape(-1, shape[-1])
        mask = flat != 0
        x = self.token_embedding(flat).transpose(1, 2)
        x = torch.nn.functional.gelu(self.local_encoder(x)).transpose(1, 2)
        x = x * mask.unsqueeze(-1)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        pooled = self.output(pooled)
        return pooled.reshape(*shape[:-1], self.output.out_features)


class MapDAGEncoder(nn.Module):
    """Encode the full map as a directed acyclic graph.

    ``future`` is propagated from the Boss row toward the current row, so every
    node representation contains a learned summary of its descendants.  A
    second light pass propagates context from the current node down its edges;
    this preserves route identity when two branches later merge.
    """

    def __init__(self, room_dim: int, d_model: int, *, rows: int, cols: int):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.node = nn.Sequential(
            nn.Linear(room_dim + 5, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.child_message = nn.Linear(d_model, d_model, bias=False)
        self.future_update = nn.GRUCell(d_model, d_model)
        self.down_update = nn.GRUCell(d_model, d_model)
        self.output = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, room_embeddings: torch.Tensor, features: torch.Tensor):
        batch = room_embeddings.shape[0]
        node_features = features[..., :5]
        base = self.node(torch.cat([room_embeddings, node_features], dim=-1))
        grid = base.reshape(batch, self.rows, self.cols, -1)
        feature_grid = features.reshape(batch, self.rows, self.cols, -1)
        edges = feature_grid[..., 5:]

        # Reverse topological DP: each node receives a message from all of its
        # children in the next row.
        future_rows: list[torch.Tensor | None] = [None] * self.rows
        for row in range(self.rows - 1, -1, -1):
            if row == self.rows - 1:
                future_rows[row] = grid[:, row]
                continue
            degree = edges[:, row].sum(dim=-1, keepdim=True).clamp_min(1.0)
            child = torch.bmm(edges[:, row], future_rows[row + 1]) / degree
            shape = child.shape
            future_rows[row] = self.future_update(
                self.child_message(child).reshape(-1, shape[-1]),
                grid[:, row].reshape(-1, shape[-1]),
            ).reshape(shape)
        future = torch.stack(future_rows, dim=1)

        # Forward route context: only nodes marked current seed this pass;
        # subsequent messages follow the map edges toward descendants.
        down_rows: list[torch.Tensor] = [grid[:, 0] * feature_grid[:, 0, :, 2:3]]
        active_rows: list[torch.Tensor] = [feature_grid[:, 0, :, 2:3]]
        for row in range(self.rows - 1):
            incoming = torch.bmm(
                edges[:, row].transpose(1, 2), down_rows[row]
            )
            incoming_active = torch.bmm(
                edges[:, row].transpose(1, 2), active_rows[row]
            ).clamp_max(1.0)
            shape = incoming.shape
            updated = self.down_update(
                incoming.reshape(-1, shape[-1]),
                grid[:, row + 1].reshape(-1, shape[-1]),
            ).reshape(shape)
            seed = feature_grid[:, row + 1, :, 2:3]
            down_rows.append(updated * incoming_active + grid[:, row + 1] * seed)
            active_rows.append(torch.maximum(incoming_active, seed))
        down = torch.stack(down_rows, dim=1)

        encoded = self.output(torch.cat([grid, future, down], dim=-1))
        return encoded.reshape(batch, self.rows * self.cols, -1)


class StSTransformerExtractor(BaseFeaturesExtractor):
    """Tokenize Dict observations and encode them with a Transformer.

    Each entity instance becomes one token (shared embedding by vocabulary +
    per-slot scalars, projected to d_model). PAD slots (id == 0) are masked out
    of attention. A prepended [CLS] token is read out to `features_dim`.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        features_dim: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        card_language: bool = False,
        combat_language: bool = False,
        map_dag: bool = False,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        self.d_model = d_model
        self._obs = observation_space
        self.map_dag_enabled = bool(map_dag and "map_full_rooms" in observation_space.spaces)
        if combat_language != ("player_status_language" in observation_space.spaces):
            raise ValueError(
                "combat_language must match the environment observation layout"
            )
        pile_language_fields = all(
            f"{pile}_card_ids" in observation_space.spaces
            for pile in ("draw", "discard", "deck")
        )
        if card_language != pile_language_fields:
            raise ValueError(
                "card_language must match the environment observation layout"
            )
        self.combat_language = (
            CombatProgramEncoder(d_model) if combat_language else None
        )
        self.combat_language_scale = (
            nn.Parameter(torch.tensor(0.1)) if combat_language else None
        )
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # --- shared embedding tables (by semantic vocabulary) ---
        cap = self._vocab_caps(observation_space)
        self.shared = nn.ModuleDict()

        def make_emb(name: str, vocab_size: int) -> int:
            dim = _SHARED_EMB_DIMS.get(name, _embedding_dim_for(vocab_size))
            if name == "card" and card_language:
                self.shared[name] = CardLanguageEmbedding(vocab_size, dim)
            else:
                self.shared[name] = nn.Embedding(vocab_size, dim, padding_idx=0)
            return dim

        d_card = make_emb("card", cap["card"])
        d_relic = make_emb("relic", cap["relic"])
        d_potion = make_emb("potion", cap["potion"])
        d_move = make_emb("move", cap["move"])
        d_monster = make_emb("monster", cap["monster"])
        d_intent = make_emb("intent", cap["intent"])
        d_room = make_emb("room", cap["room"])
        self.neow_enabled = "neow_bonus_ids" in observation_space.spaces
        if self.neow_enabled:
            d_neow_bonus = make_emb(
                "neow_bonus",
                int(observation_space["neow_bonus_ids"].high.max()) + 1,
            )
            d_neow_drawback = make_emb(
                "neow_drawback",
                int(observation_space["neow_drawback_ids"].high.max()) + 1,
            )

        # small categorical embeddings that all live on the single global token
        self.global_cat_keys = [
            "phase", "arg_step", "character",
            "select_context", "event_id", "next_boss",
        ]
        d_global_cat = 0
        for key in self.global_cat_keys:
            v = int(observation_space[key].high.max()) + 1
            d_global_cat += make_emb(key, v)
        self.event_option_enabled = "event_option_ids" in observation_space.spaces
        if self.event_option_enabled:
            d_event_id = self.shared["event_id"].embedding_dim
            d_event_option = make_emb(
                "event_option",
                int(observation_space["event_option_ids"].high.max()) + 1,
            )

        # --- generic "id + per-slot scalars" families ---
        # (role, n_slots, id_key, shared_emb, [(scalar_key, width), ...])
        self.generic = [
            ("hand",        self._n("hand_ids"),        "hand_ids",        "card",
                [("hand_upgrades", 1), ("hand_costs", 1),
                 ("hand_damage", 1), ("hand_block", 1),
                 ("select_hand_mask", 1)]),
            ("offer",       self._n("offer_card_ids"),  "offer_card_ids",  "card", []),
            ("select",      self._n("select_card_ids"), "select_card_ids", "card", []),
            ("selected",    self._n("selected_card_ids"),"selected_card_ids","card", []),
            ("shop_card",   self._n("shop_card_ids"),   "shop_card_ids",   "card",
                [("shop_card_prices", 1)]),
            ("relic_held",  self._n("relics"),          "relics",          "relic", []),
            ("shop_relic",  self._n("shop_relic_ids"),  "shop_relic_ids",  "relic",
                [("shop_relic_prices", 1)]),
            ("boss_relic",  self._n("boss_relic_ids"),  "boss_relic_ids",  "relic", []),
            ("potion_held", self._n("potions"),         "potions",         "potion", []),
            ("shop_potion", self._n("shop_potion_ids"), "shop_potion_ids", "potion",
                [("shop_potion_prices", 1)]),
            ("map",         self._n("map_node_rooms"),  "map_node_rooms",  "room",
                [("map_node_x", 1)]),
        ]
        if "map_full_rooms" in observation_space.spaces:
            map_width = (
                self._n("map_full_features", flat=True)
                // self._n("map_full_rooms")
            )
            self.generic.append(
                ("map_full", self._n("map_full_rooms"), "map_full_rooms", "room",
                 [("map_full_features", map_width)])
            )
        emb_dim_of = {
            "card": d_card, "relic": d_relic, "potion": d_potion, "room": d_room,
        }

        # roles: 0=cls, 1=global, 2=monster, 3=pile, then one per generic
        # family. New singleton roles are appended so old checkpoints keep the
        # same role ids for existing token families.
        self.role_ids = {"cls": 0, "global": 1, "monster": 2, "pile": 3}
        for i, (role, *_rest) in enumerate(self.generic):
            self.role_ids[role] = 4 + i
        if self.neow_enabled:
            self.role_ids["neow"] = 4 + len(self.generic)
        if self.event_option_enabled:
            self.role_ids["event_option"] = len(self.role_ids)
        n_roles = len(self.role_ids)

        # --- per-family tokenizers (project to d_model) + position embeddings ---
        self.tok = nn.ModuleDict()
        self.pos = nn.ParameterDict()

        for role, n_slots, _id_key, emb_name, scalar_keys in self.generic:
            if role == "map_full" and self.map_dag_enabled:
                continue
            in_dim = emb_dim_of[emb_name] + sum(w for _, w in scalar_keys)
            self.tok[role] = nn.Linear(in_dim, d_model)
            self.pos[role] = nn.Parameter(torch.zeros(n_slots, d_model))

        if self.map_dag_enabled:
            self.map_dag = MapDAGEncoder(
                emb_dim_of["room"], d_model,
                rows=encode.MAP_ROWS, cols=encode.MAP_COLS,
            )

        # monster token: id + move + intent embeddings + scalars + buffs
        self.n_monsters = self._n("monster_id")
        mon_in = (
            d_monster + d_move + d_intent
            + self._n("monster_scalars", flat=True) // self.n_monsters
            + self._n("monster_buffs", flat=True) // self.n_monsters
        )
        self.tok["monster"] = nn.Linear(mon_in, d_model)
        self.pos["monster"] = nn.Parameter(torch.zeros(self.n_monsters, d_model))

        if self.neow_enabled:
            neow_in = d_neow_bonus + d_neow_drawback + self._n(
                "neow_features", flat=True
            ) // self._n("neow_bonus_ids")
            self.tok["neow"] = nn.Linear(neow_in, d_model)
            self.pos["neow"] = nn.Parameter(
                torch.zeros(self._n("neow_bonus_ids"), d_model)
            )

        if self.event_option_enabled:
            event_option_in = d_event_id + d_event_option + self._n(
                "event_option_features", flat=True
            ) // self._n("event_option_ids")
            self.tok["event_option"] = nn.Linear(event_option_in, d_model)
            self.pos["event_option"] = nn.Parameter(
                torch.zeros(self._n("event_option_ids"), d_model)
            )

        # pile tokens: draw / discard / exhaust / deck count vectors
        self.pile_keys = ["draw_counts", "discard_counts", "exhaust_counts", "deck_counts"]
        self.tok["pile"] = nn.Linear(self._n("draw_counts"), d_model)
        self.pos["pile"] = nn.Parameter(torch.zeros(len(self.pile_keys), d_model))
        self.pile_card_language = (
            PileCardLanguageEncoder(
                d_card, d_model,
                observation_space["draw_card_ids"].shape[0],
            )
            if card_language else None
        )
        self.exhaust_card_language = (
            nn.Linear(d_card, d_model, bias=False) if card_language else None
        )

        # global token: assorted run/player scalars + small categorical embeddings
        self.global_scalar_keys = [
            "player_scalars", "player_buffs", "ascension", "act", "floor",
            "gold", "pick_remaining", "can_pick_zero", "can_pick_any",
            "shop_remove_cost",
        ]
        if "deck_size" in observation_space.spaces:
            self.global_scalar_keys.append("deck_size")
        g_in = sum(self._n(k, flat=True) for k in self.global_scalar_keys) + d_global_cat
        self.tok["global"] = nn.Linear(g_in, d_model)

        self.role_emb = nn.Embedding(n_roles, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # token_pos: role -> (start, count) in the concatenated sequence. MUST
        # mirror the add() order in forward()/encode_tokens() exactly. The pointer
        # policy (pointer_policy.py) reads per-entity token slices by this map.
        self.token_pos: dict[str, tuple[int, int]] = {}
        _p = 0
        for _role, _cnt in (
            [("cls", 1), ("global", 1), ("pile", len(self.pile_keys)),
             ("monster", self.n_monsters)]
            + [(role, n_slots) for role, n_slots, *_ in self.generic]
            + (
                [("neow", self._n("neow_bonus_ids"))]
                if self.neow_enabled else []
            )
            + (
                [("event_option", self._n("event_option_ids"))]
                if self.event_option_enabled else []
            )
        ):
            self.token_pos[_role] = (_p, _cnt)
            _p += _cnt
        self.n_tokens = _p

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, features_dim),
            nn.GELU(),
        )

    # ---- helpers -----------------------------------------------------------
    def _n(self, key: str, *, flat: bool = False) -> int:
        """Slot count (first dim) of obs key, or flat product if `flat`."""
        shape = self._obs[key].shape
        if flat:
            return int(prod(shape)) if shape else 1
        return int(shape[0]) if shape else 1

    @staticmethod
    def _vocab_caps(obs: spaces.Dict) -> dict[str, int]:
        def hi(key: str) -> int:
            return int(obs[key].high.max()) + 1
        return {
            "card": hi("hand_ids"),
            "relic": hi("relics"),
            "potion": hi("potions"),
            "move": hi("monster_move"),
            "monster": hi("monster_id"),
            "intent": hi("monster_intent_cat"),
            "room": hi("map_node_rooms"),
        }

    def _role(self, x: torch.Tensor, role: str) -> torch.Tensor:
        """Add the role embedding (broadcast over slots) to a (B, n, d) block."""
        rid = torch.full(
            (1,), self.role_ids[role], dtype=torch.long, device=x.device
        )
        return x + self.role_emb(rid).view(1, 1, self.d_model)

    # ---- forward -----------------------------------------------------------
    def encode_tokens(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the tokenizer + Transformer and return the FULL per-token sequence.

        Returns (out, pad_mask): out is (B, n_tokens, d_model) contextualized token
        embeddings (token 0 is [CLS]; see token_pos for the rest), pad_mask is
        (B, n_tokens) bool with True == PAD. `forward` reads out[:, 0]; the pointer
        policy reads per-entity slices via token_pos.
        """
        B = obs["phase"].shape[0]
        device = self.cls.device

        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []  # True == PAD (ignored by attention)

        def add(tok: torch.Tensor, pad: torch.Tensor) -> None:
            tokens.append(tok)
            masks.append(pad)

        # [CLS] — always present
        add(self.cls.expand(B, -1, -1),
            torch.zeros(B, 1, dtype=torch.bool, device=device))

        # global token — always present
        gparts = [obs[k].float().reshape(B, -1) for k in self.global_scalar_keys]
        for k in self.global_cat_keys:
            gparts.append(self.shared[k](obs[k].long()).reshape(B, -1))
        g = self.tok["global"](torch.cat(gparts, dim=1)).unsqueeze(1)
        if self.combat_language is not None:
            status = self.combat_language(obs["player_status_language"])
            g = g + self.combat_language_scale * status.unsqueeze(1)
        add(self._role(g, "global"),
            torch.zeros(B, 1, dtype=torch.bool, device=device))

        # pile tokens — always present
        pile_toks = []
        for i, k in enumerate(self.pile_keys):
            t = self.tok["pile"](obs[k].float()) + self.pos["pile"][i]
            if self.pile_card_language is not None:
                pile = k.removesuffix("_counts")
                ids_key = f"{pile}_card_ids"
                if ids_key in obs:
                    t = t + self.pile_card_language(
                        obs[ids_key].long(),
                        obs[f"{pile}_card_upgrades"].float(),
                        obs[f"{pile}_card_counts"].float(),
                        obs[f"{pile}_card_upgraded_counts"].float(),
                        self.shared["card"],
                    )
                else:
                    card_table = self.shared["card"].effective_weight()
                    semantic_counts = obs[k].float() @ card_table
                    t = t + self.exhaust_card_language(semantic_counts)
            pile_toks.append(t)
        pile = torch.stack(pile_toks, dim=1)
        add(self._role(pile, "pile"),
            torch.zeros(B, len(self.pile_keys), dtype=torch.bool, device=device))

        # monster tokens
        mon = torch.cat([
            self.shared["monster"](obs["monster_id"].long()),
            self.shared["move"](obs["monster_move"].long()),
            self.shared["intent"](obs["monster_intent_cat"].long()),
            obs["monster_scalars"].float(),
            obs["monster_buffs"].float(),
        ], dim=-1)
        mon = self.tok["monster"](mon) + self.pos["monster"].unsqueeze(0)
        if self.combat_language is not None:
            mon_language = (
                self.combat_language(obs["monster_status_language"])
                + self.combat_language(obs["monster_intent_language"])
            )
            mon = mon + self.combat_language_scale * mon_language
        add(self._role(mon, "monster"), obs["monster_id"].long() == 0)

        # generic id+scalar families
        for role, _n_slots, id_key, emb_name, scalar_keys in self.generic:
            ids = obs[id_key].long()
            if role == "map_full" and self.map_dag_enabled:
                x = self.map_dag(self.shared[emb_name](ids), obs["map_full_features"].float())
                add(self._role(x, role), ids == 0)
                continue
            parts = [self.shared[emb_name](ids)]
            for skey, w in scalar_keys:
                parts.append(obs[skey].float().reshape(B, ids.shape[1], w))
            x = self.tok[role](torch.cat(parts, dim=-1)) + self.pos[role].unsqueeze(0)
            add(self._role(x, role), ids == 0)

        if self.neow_enabled:
            neow = torch.cat([
                self.shared["neow_bonus"](obs["neow_bonus_ids"].long()),
                self.shared["neow_drawback"](obs["neow_drawback_ids"].long()),
                obs["neow_features"].float(),
            ], dim=-1)
            neow = self.tok["neow"](neow) + self.pos["neow"].unsqueeze(0)
            add(self._role(neow, "neow"), obs["neow_bonus_ids"].long() == 0)

        if self.event_option_enabled:
            event_option = torch.cat([
                self.shared["event_id"](obs["event_option_event_ids"].long()),
                self.shared["event_option"](obs["event_option_ids"].long()),
                obs["event_option_features"].float(),
            ], dim=-1)
            event_option = (
                self.tok["event_option"](event_option)
                + self.pos["event_option"].unsqueeze(0)
            )
            add(
                self._role(event_option, "event_option"),
                obs["event_option_ids"].long() == 0,
            )

        src = torch.cat(tokens, dim=1)
        pad_mask = torch.cat(masks, dim=1)
        out = self.transformer(src, src_key_padding_mask=pad_mask)
        return out, pad_mask

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        out, _ = self.encode_tokens(obs)
        return self.readout(out[:, 0])  # [CLS]


# Back-compat alias: StSMLPExtractor was named StSFeatureExtractor before the
# Transformer extractor landed. cloudpickle stores a checkpoint's extractor by
# qualified name, so keeping this alias lets MaskablePPO.load() still resolve
# pre-rename MLP checkpoints. New runs save under the current class names.
StSFeatureExtractor = StSMLPExtractor
