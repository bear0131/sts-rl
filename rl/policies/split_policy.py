"""Phase-specialized actor-critic policy for complete Slay the Spire runs.

Combat and overworld observations are routed into independent Transformer,
policy-head, and value-head parameters. Card, relic, and potion semantics are
tokenized once by a shared module before either specialist sees them.
"""

from __future__ import annotations

from math import prod

import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from torch import nn

from env.core import action_space as A
from env.core import encode
from env.core import vocab as V
from rl.policies.model import (
    CardLanguageEmbedding,
    CombatProgramEncoder,
    MapDAGEncoder,
    PileCardLanguageEncoder,
    _embedding_dim_for,
)
from rl.policies.pointer_policy import _NullExtractor
from rl.rules.rule_vector_model import load_frozen_card_rule_encoder


def _space_width(observation_space: spaces.Dict, key: str) -> int:
    return int(prod(observation_space[key].shape))


def _vocab_size(observation_space: spaces.Dict, key: str) -> int:
    return int(observation_space[key].high.max()) + 1


def _active_prefix_width(ids: torch.Tensor) -> int:
    """Keep every populated positional slot while dropping trailing PAD."""
    active_columns = ids.long().ne(0).any(dim=0).nonzero(as_tuple=False)
    return int(active_columns[-1, 0].item()) + 1 if active_columns.numel() else 1


# Runtime routing is deliberately stricter than the unified Gym Dict space.
# A specialist cannot accidentally consume fields owned by the other phase.
COMBAT_OBSERVATION_KEYS = frozenset({
    "combat_is_boss",
    "combat_legal",
    "arg_step",
    "ascension",
    "can_pick_any",
    "can_pick_zero",
    "character",
    "combat_counters",
    "post_enemy_turn_boundary",
    "discard_card_counts",
    "discard_card_ids",
    "discard_card_upgraded_counts",
    "discard_card_upgrades",
    "discard_counts",
    "draw_card_counts",
    "draw_card_ids",
    "draw_card_upgraded_counts",
    "draw_card_upgrades",
    "draw_counts",
    "exhaust_counts",
    "gold",
    "hand_block",
    "hand_costs",
    "hand_damage",
    "hand_ids",
    "hand_upgrades",
    "monster_buffs",
    "monster_id",
    "monster_intent_cat",
    "monster_intent_language",
    "monster_move",
    "monster_scalars",
    "monster_status_language",
    "pick_remaining",
    "player_buffs",
    "player_scalars",
    "player_status_language",
    "potions",
    "relic_counters",
    "relics",
    "select_context",
    "select_hand_mask",
    "select_card_ids",
    "select_card_upgrades",
    "select_card_costs",
    "select_native_indices",
    "select_zone",
    "pending_card_slot",
    "pending_potion_slot",
})

OVERWORLD_OBSERVATION_KEYS = frozenset({
    "act",
    "arg_step",
    "ascension",
    "can_pick_any",
    "can_pick_zero",
    "character",
    "deck_ids",
    "deck_size",
    "deck_upgrades",
    "event_id",
    "floor",
    "gold",
    "next_boss",
    "phase",
    "pick_remaining",
    "potions",
    "relic_counters",
    "relics",
    "run_hp",
    "select_context",
    "shop_remove_cost",
})

OVERWORLD_PHASE_OBSERVATION_KEYS = {
    V.PHASE_VOCAB_INDEX["combat_reward"]: frozenset({
        "offer_card_ids",
        "offer_card_upgrades",
        "reward_gold",
        "reward_potion_ids",
        "reward_relic_ids",
    }),
    V.PHASE_VOCAB_INDEX["card_select"]: frozenset({
        "select_card_ids",
        "select_card_upgrades",
        "selected_card_ids",
        "selected_card_upgrades",
    }),
    V.PHASE_VOCAB_INDEX["map"]: frozenset({
        "map_full_features",
        "map_full_rooms",
        "map_node_rooms",
        "map_node_x",
    }),
    V.PHASE_VOCAB_INDEX["shop"]: frozenset({
        "shop_card_ids",
        "shop_card_prices",
        "shop_card_upgrades",
        "shop_potion_ids",
        "shop_potion_prices",
        "shop_relic_ids",
        "shop_relic_prices",
    }),
    V.PHASE_VOCAB_INDEX["boss_relic"]: frozenset({"boss_relic_ids"}),
    V.PHASE_VOCAB_INDEX["event"]: frozenset({
        "event_option_event_ids",
        "event_option_features",
        "event_option_ids",
        "neow_bonus_ids",
        "neow_drawback_ids",
        "neow_features",
    }),
}


class SharedSemanticTokenizer(nn.Module):
    """Shared physical-entity token design used by both specialists."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        d_model: int,
        *,
        card_language: bool,
        card_dim: int = 64,
        card_embedding: nn.Module | None = None,
    ) -> None:
        super().__init__()
        n_cards = _vocab_size(observation_space, "deck_ids")
        n_relics = _vocab_size(observation_space, "relics")
        n_potions = _vocab_size(observation_space, "potions")
        if card_dim < 1:
            raise ValueError("card_dim must be positive")
        relic_dim = 48
        potion_dim = _embedding_dim_for(n_potions)

        if card_embedding is not None:
            embedding_width = getattr(card_embedding, "embedding_dim", None)
            if embedding_width is None or int(embedding_width) != card_dim:
                raise ValueError(
                    "card_embedding width must match card_dim "
                    f"({card_dim}), got {embedding_width}"
                )
            if getattr(card_embedding, "num_embeddings", n_cards) != n_cards:
                raise ValueError(
                    "card_embedding must cover the environment card vocabulary "
                    f"({n_cards} rows)"
                )
            self.card = card_embedding
        else:
            self.card = (
                CardLanguageEmbedding(n_cards, card_dim)
                if card_language
                else nn.Embedding(n_cards, card_dim, padding_idx=0)
            )
        # A frozen rule encoder is independent of the current observation and
        # never receives gradients in wholerun.  Materialize both card
        # versions once so every policy forward only performs a table lookup;
        # the trainable projection below is intentionally kept outside this
        # cache and is still evaluated each step.
        card_cache = None
        self.card_vector_cache_enabled = False
        card_parameters = tuple(self.card.parameters())
        if (
            hasattr(self.card, "versioned_forward")
            and card_parameters
            and all(not parameter.requires_grad for parameter in card_parameters)
        ):
            card_ids = torch.arange(n_cards, dtype=torch.long)
            versions = torch.tensor([0, 1], dtype=torch.long).view(1, 2)
            card_ids = card_ids.view(-1, 1).expand(-1, 2)
            with torch.no_grad():
                card_cache = self.card.versioned_forward(card_ids, versions.expand_as(card_ids))
            card_cache = card_cache.detach()
            self.card_vector_cache_enabled = True
        self.register_buffer(
            "card_vector_cache", card_cache, persistent=False
        )
        self.relic = nn.Embedding(n_relics, relic_dim, padding_idx=0)
        self.potion = nn.Embedding(n_potions, potion_dim, padding_idx=0)
        self.card_projection = nn.Linear(card_dim + 1, d_model)
        self.relic_projection = nn.Linear(relic_dim + 1, d_model)
        self.potion_projection = nn.Linear(potion_dim, d_model)

    def card_rule_vectors(
        self, ids: torch.Tensor, upgrades: torch.Tensor
    ) -> torch.Tensor:
        """Return frozen or live rule vectors for physical card instances."""

        ids = ids.long()
        versions = upgrades.long().gt(0).long().clamp(0, 1)
        if self.card_vector_cache is not None:
            return self.card_vector_cache[ids, versions]
        if hasattr(self.card, "versioned_forward"):
            return self.card.versioned_forward(ids, upgrades)
        return self.card(ids)

    def card_effect_vectors(
        self, ids: torch.Tensor, upgrades: torch.Tensor
    ) -> torch.Tensor:
        """Return vectors used by pile-level semantic summaries.

        ``CardLanguageEmbedding`` historically exposed effect-only vectors for
        piles, while the v7 frozen encoder's entire vector is its rule effect.
        The cache is valid for the latter and avoids re-running its Transformer.
        """

        ids = ids.long()
        versions = upgrades.long().gt(0).long().clamp(0, 1)
        if self.card_vector_cache is not None:
            return self.card_vector_cache[ids, versions]
        if hasattr(self.card, "version_effect_vectors"):
            return self.card.version_effect_vectors(ids, upgrades)
        return self.card_rule_vectors(ids, upgrades)

    def effective_card_weight(self) -> torch.Tensor:
        """Return base-version vectors for count-weighted pile summaries."""

        if self.card_vector_cache is not None:
            return self.card_vector_cache[:, 0]
        if hasattr(self.card, "effective_weight"):
            return self.card.effective_weight()
        return self.card_rule_vectors(
            torch.arange(self.card.num_embeddings, device=self.card.weight.device),
            torch.zeros(self.card.num_embeddings, device=self.card.weight.device),
        )

    def cards(self, ids: torch.Tensor, upgrades: torch.Tensor) -> torch.Tensor:
        upgrades = upgrades.float()
        embedding = self.card_rule_vectors(ids, upgrades)
        return self.card_projection(
            torch.cat([embedding, upgrades.unsqueeze(-1)], dim=-1)
        )

    def relics(self, ids: torch.Tensor, counters: torch.Tensor) -> torch.Tensor:
        return self.relic_projection(
            torch.cat(
                [self.relic(ids.long()), counters.float().unsqueeze(-1)], dim=-1
            )
        )

    def potions(self, ids: torch.Tensor) -> torch.Tensor:
        return self.potion_projection(self.potion(ids.long()))


class _SpecialistEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        features_dim: int,
        roles: tuple[str, ...],
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.role_index = {role: i for i, role in enumerate(roles)}
        self.role_embedding = nn.Embedding(len(roles), d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=0.0,
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

    def _role(self, token: torch.Tensor, role: str) -> torch.Tensor:
        # A role is a model constant: avoid a CPU→CUDA scalar copy per block.
        return token + self.role_embedding.weight[self.role_index[role]].view(1, 1, -1)

    def _run(
        self,
        batch_size: int,
        blocks: list[tuple[str, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = self.cls.device
        tokens = [self._role(self.cls.expand(batch_size, -1, -1), "cls")]
        masks = [torch.zeros(batch_size, 1, dtype=torch.bool, device=device)]
        spans: dict[str, tuple[int, int]] = {}
        offset = 1
        for role, token, pad in blocks:
            token = self._role(token, role)
            tokens.append(token)
            masks.append(pad)
            spans[role] = (offset, offset + token.shape[1])
            offset += token.shape[1]
        source = torch.cat(tokens, dim=1)
        pad_mask = torch.cat(masks, dim=1)
        out = self.transformer(source, src_key_padding_mask=pad_mask)
        contextual = {role: out[:, start:end] for role, (start, end) in spans.items()}
        return out[:, 0], contextual


class CombatEncoder(_SpecialistEncoder):
    """Consumes only live combat state plus combat-relevant run resources."""

    _ROLES = (
        "cls", "global", "pile", "monster", "hand", "relic", "potion",
    )

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        features_dim: int,
        combat_language: bool,
        card_dim: int = 64,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            features_dim=features_dim,
            roles=self._ROLES,
        )
        self.observation_space = observation_space
        self.global_categories = nn.ModuleDict()
        global_category_keys = ("arg_step", "character", "select_context")
        if "combat_legal" in observation_space.spaces:
            # A board representation shared by root and pending target views.
            global_category_keys = ("character", "select_context")
        self.public_choices = "select_native_indices" in observation_space.spaces
        if self.public_choices:
            global_category_keys += ("select_zone",)
            self.choice_context = nn.Linear(1, d_model, bias=False)
            self.choice_position = nn.Parameter(torch.zeros(A.G_MAX, d_model))
        category_width = 0
        for key in global_category_keys:
            width = _embedding_dim_for(_vocab_size(observation_space, key))
            self.global_categories[key] = nn.Embedding(
                _vocab_size(observation_space, key), width, padding_idx=0
            )
            category_width += width
        self.global_category_keys = global_category_keys
        # ``post_enemy_turn_boundary`` was added after the first generation of
        # whole-run checkpoints.  Build this list from the actual observation
        # space so those checkpoints remain loadable with the new encoder.  A
        # legacy combat observation simply omits the feature and therefore
        # retains its original input geometry.
        scalar_keys = (
            "player_scalars", "player_buffs", "combat_counters", "ascension",
            "post_enemy_turn_boundary", "gold", "pick_remaining",
            "can_pick_zero", "can_pick_any",
        )
        self.global_scalar_keys = tuple(
            key for key in scalar_keys if key in observation_space.spaces
        )
        self.global_projection = nn.Linear(
            sum(_space_width(observation_space, key) for key in self.global_scalar_keys)
            + category_width,
            d_model,
        )

        monster_dims = []
        self.monster_embeddings = nn.ModuleDict()
        for key in ("monster_id", "monster_move", "monster_intent_cat"):
            width = 64 if key == "monster_move" else _embedding_dim_for(
                _vocab_size(observation_space, key)
            )
            self.monster_embeddings[key] = nn.Embedding(
                _vocab_size(observation_space, key), width, padding_idx=0
            )
            monster_dims.append(width)
        n_monsters = observation_space["monster_id"].shape[0]
        monster_scalar_width = (
            _space_width(observation_space, "monster_scalars")
            + _space_width(observation_space, "monster_buffs")
        ) // n_monsters
        self.monster_projection = nn.Linear(
            sum(monster_dims) + monster_scalar_width, d_model
        )

        self.pile_keys = ("draw_counts", "discard_counts", "exhaust_counts")
        self.pile_projection = nn.Linear(
            _space_width(observation_space, "draw_counts"), d_model
        )
        self.pile_card_language = (
            PileCardLanguageEncoder(
                card_dim,
                d_model,
                observation_space["draw_card_ids"].shape[0],
            )
            if "draw_card_ids" in observation_space.spaces
            else None
        )
        self.exhaust_card_language = (
            nn.Linear(card_dim, d_model, bias=False)
            if self.pile_card_language is not None
            else None
        )
        # SemanticReasoningPolicyCore can disable pile semantics while keeping
        # these modules present for checkpoint/state-dict compatibility. Hand
        # cards remain the only card programs consumed by the combat encoder.
        self.pile_card_language_enabled = True
        self.exhaust_card_language_enabled = True
        self.hand_context = nn.Linear(4, d_model, bias=False)
        self.position = nn.ParameterDict({
            "pile": nn.Parameter(torch.zeros(len(self.pile_keys), d_model)),
            "monster": nn.Parameter(torch.zeros(n_monsters, d_model)),
            "hand": nn.Parameter(
                torch.zeros(observation_space["hand_ids"].shape[0], d_model)
            ),
            "relic": nn.Parameter(
                torch.zeros(observation_space["relics"].shape[0], d_model)
            ),
            "potion": nn.Parameter(
                torch.zeros(observation_space["potions"].shape[0], d_model)
            ),
        })
        self.combat_language = (
            CombatProgramEncoder(d_model) if combat_language else None
        )
        self.combat_language_scale = (
            nn.Parameter(torch.tensor(0.1)) if combat_language else None
        )

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        shared: SharedSemanticTokenizer,
        *,
        active_relic_width: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = obs["player_scalars"].shape[0]
        global_parts = [
            obs[key].float().reshape(batch_size, -1)
            for key in self.global_scalar_keys
        ]
        global_parts.extend(
            self.global_categories[key](obs[key].long()).reshape(batch_size, -1)
            for key in self.global_category_keys
        )
        global_token = self.global_projection(torch.cat(global_parts, dim=1)).unsqueeze(1)
        choice_tokens = None
        if self.public_choices:
            choice_tokens = shared.cards(obs["select_card_ids"], obs["select_card_upgrades"])
            choice_tokens = choice_tokens + self.choice_context(obs["select_card_costs"].float().unsqueeze(-1))
            # Only presence is public: the native handle's magnitude would leak
            # draw-pile order. Skip is an explicit candidate with a PAD card.
            present = obs["select_native_indices"].ne(0).unsqueeze(-1)
            pooled = (choice_tokens * present).sum(dim=1, keepdim=True) / present.sum(dim=1, keepdim=True).clamp_min(1)
            global_token = global_token + pooled
            choice_tokens = choice_tokens + self.choice_position
        if self.combat_language is not None:
            global_token = global_token + self.combat_language_scale * self.combat_language(
                obs["player_status_language"]
            ).unsqueeze(1)

        pile_tokens = []
        for i, key in enumerate(self.pile_keys):
            token = self.pile_projection(obs[key].float())
            pile_name = key.removesuffix("_counts")
            if (
                self.pile_card_language_enabled
                and self.pile_card_language is not None
                and pile_name != "exhaust"
            ):
                token = token + self.pile_card_language(
                    obs[f"{pile_name}_card_ids"].long(),
                    obs[f"{pile_name}_card_upgrades"].float(),
                    obs[f"{pile_name}_card_counts"].float(),
                    obs[f"{pile_name}_card_upgraded_counts"].float(),
                    shared.card,
                    card_vector_provider=shared.card_effect_vectors,
                )
            elif (
                self.exhaust_card_language_enabled
                and self.exhaust_card_language is not None
            ):
                semantic_counts = obs[key].float() @ shared.effective_card_weight()
                token = token + self.exhaust_card_language(semantic_counts)
            pile_tokens.append(token + self.position["pile"][i])
        pile = torch.stack(pile_tokens, dim=1)
        monster = torch.cat(
            [
                self.monster_embeddings["monster_id"](obs["monster_id"].long()),
                self.monster_embeddings["monster_move"](obs["monster_move"].long()),
                self.monster_embeddings["monster_intent_cat"](
                    obs["monster_intent_cat"].long()
                ),
                obs["monster_scalars"].float(),
                obs["monster_buffs"].float(),
            ],
            dim=-1,
        )
        monster = self.monster_projection(monster) + self.position["monster"]
        if self.combat_language is not None:
            monster = monster + self.combat_language_scale * (
                self.combat_language(obs["monster_status_language"])
                + self.combat_language(obs["monster_intent_language"])
            )

        hand = shared.cards(obs["hand_ids"], obs["hand_upgrades"])
        hand_extra = torch.stack(
            [
                obs["hand_costs"], obs["hand_damage"], obs["hand_block"],
                obs["select_hand_mask"],
            ],
            dim=-1,
        )
        hand = hand + self.hand_context(hand_extra.float()) + self.position["hand"]
        relic_width = (
            _active_prefix_width(obs["relics"])
            if active_relic_width is None
            else max(1, min(int(active_relic_width), obs["relics"].shape[1]))
        )
        relic_ids = obs["relics"][:, :relic_width]
        relic = shared.relics(
            relic_ids, obs["relic_counters"][:, :relic_width]
        ) + self.position["relic"][:relic_width]
        potion = shared.potions(obs["potions"]) + self.position["potion"]
        device = global_token.device
        blocks = [
            (
                "global", global_token,
                torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            ),
            (
                "pile", pile,
                torch.zeros(batch_size, len(self.pile_keys), dtype=torch.bool, device=device),
            ),
            ("monster", monster, obs["monster_id"].long() == 0),
            ("hand", hand, obs["hand_ids"].long() == 0),
            ("relic", relic, relic_ids.long() == 0),
            ("potion", potion, obs["potions"].long() == 0),
        ]
        global_out, encoded = self._run(batch_size, blocks)
        if choice_tokens is not None:
            # Score candidates against the contextual global token without
            # padding every combat attention sequence with 96 empty tokens.
            encoded["select"] = choice_tokens
        return global_out, encoded


class OverworldEncoder(_SpecialistEncoder):
    """Consumes persistent run state and only the active overworld screen."""

    _ROLES = (
        "cls", "global", "deck", "relic", "potion", "offer", "reward_gold",
        "reward_relic", "reward_potion", "select", "selected", "map",
        "map_full", "shop_card", "shop_relic", "shop_potion", "boss_relic",
        "neow", "event_option",
    )

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        features_dim: int,
        map_dag: bool,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            features_dim=features_dim,
            roles=self._ROLES,
        )
        self.observation_space = observation_space
        self.map_dag_enabled = bool(
            map_dag and "map_full_rooms" in observation_space.spaces
        )
        global_category_keys = (
            "phase", "arg_step", "character", "select_context", "event_id",
            "next_boss",
        )
        self.global_categories = nn.ModuleDict()
        category_width = 0
        for key in global_category_keys:
            width = _embedding_dim_for(_vocab_size(observation_space, key))
            self.global_categories[key] = nn.Embedding(
                _vocab_size(observation_space, key), width, padding_idx=0
            )
            category_width += width
        self.global_category_keys = global_category_keys
        scalar_keys = [
            "run_hp", "ascension", "act", "floor", "gold", "pick_remaining",
            "can_pick_zero", "can_pick_any", "shop_remove_cost",
        ]
        if "deck_size" in observation_space.spaces:
            scalar_keys.append("deck_size")
        self.global_scalar_keys = tuple(scalar_keys)
        self.global_projection = nn.Linear(
            sum(_space_width(observation_space, key) for key in scalar_keys)
            + category_width,
            d_model,
        )

        self.card_price = nn.Linear(1, d_model, bias=False)
        self.relic_price = nn.Linear(1, d_model, bias=False)
        self.potion_price = nn.Linear(1, d_model, bias=False)
        self.reward_gold_projection = nn.Linear(1, d_model)

        room_dim = _embedding_dim_for(_vocab_size(observation_space, "map_node_rooms"))
        self.room_embedding = nn.Embedding(
            _vocab_size(observation_space, "map_node_rooms"), room_dim, padding_idx=0
        )
        self.map_projection = nn.Linear(room_dim + 1, d_model)
        self.map_dag = (
            MapDAGEncoder(
                room_dim, d_model, rows=encode.MAP_ROWS, cols=encode.MAP_COLS
            )
            if self.map_dag_enabled else None
        )

        self.neow_enabled = "neow_bonus_ids" in observation_space.spaces
        if self.neow_enabled:
            bonus_dim = _embedding_dim_for(
                _vocab_size(observation_space, "neow_bonus_ids")
            )
            drawback_dim = _embedding_dim_for(
                _vocab_size(observation_space, "neow_drawback_ids")
            )
            self.neow_bonus = nn.Embedding(
                _vocab_size(observation_space, "neow_bonus_ids"), bonus_dim,
                padding_idx=0,
            )
            self.neow_drawback = nn.Embedding(
                _vocab_size(observation_space, "neow_drawback_ids"), drawback_dim,
                padding_idx=0,
            )
            self.neow_projection = nn.Linear(bonus_dim + drawback_dim + 2, d_model)

        self.event_option_enabled = "event_option_ids" in observation_space.spaces
        if self.event_option_enabled:
            event_dim = _embedding_dim_for(_vocab_size(observation_space, "event_id"))
            option_dim = _embedding_dim_for(
                _vocab_size(observation_space, "event_option_ids")
            )
            self.event_for_option = nn.Embedding(
                _vocab_size(observation_space, "event_id"), event_dim, padding_idx=0
            )
            self.event_option_embedding = nn.Embedding(
                _vocab_size(observation_space, "event_option_ids"), option_dim,
                padding_idx=0,
            )
            self.event_option_projection = nn.Linear(
                event_dim + option_dim + 2, d_model
            )

        position_widths = {
            "deck": observation_space["deck_ids"].shape[0],
            "relic": observation_space["relics"].shape[0],
            "potion": observation_space["potions"].shape[0],
            "offer": observation_space["offer_card_ids"].shape[0],
            "reward_gold": observation_space["reward_gold"].shape[0],
            "reward_relic": observation_space["reward_relic_ids"].shape[0],
            "reward_potion": observation_space["reward_potion_ids"].shape[0],
            "select": observation_space["select_card_ids"].shape[0],
            "selected": observation_space["selected_card_ids"].shape[0],
            "map": observation_space["map_node_rooms"].shape[0],
            "shop_card": observation_space["shop_card_ids"].shape[0],
            "shop_relic": observation_space["shop_relic_ids"].shape[0],
            "shop_potion": observation_space["shop_potion_ids"].shape[0],
            "boss_relic": observation_space["boss_relic_ids"].shape[0],
        }
        if "map_full_rooms" in observation_space.spaces:
            position_widths["map_full"] = observation_space["map_full_rooms"].shape[0]
        if self.neow_enabled:
            position_widths["neow"] = observation_space["neow_bonus_ids"].shape[0]
        if self.event_option_enabled:
            position_widths["event_option"] = observation_space["event_option_ids"].shape[0]
        self.position = nn.ParameterDict({
            role: nn.Parameter(torch.zeros(width, d_model))
            for role, width in position_widths.items()
        })

    def _card_block(
        self, obs, shared, role, ids_key, upgrades_key, extra=None,
        trim_trailing_padding=False,
    ):
        ids = obs[ids_key]
        upgrades = obs[upgrades_key]
        position = self.position[role]
        if trim_trailing_padding:
            width = _active_prefix_width(ids)
            ids = ids[:, :width]
            upgrades = upgrades[:, :width]
            position = position[:width]
            if extra is not None:
                extra = extra[:, :width]
        token = shared.cards(ids, upgrades)
        if extra is not None:
            token = token + extra
        return role, token + position, ids.long() == 0

    def _relic_block(
        self, obs, shared, role, ids_key, counters=None, extra=None,
        trim_trailing_padding=False,
    ):
        ids = obs[ids_key]
        if counters is None:
            counters = torch.zeros_like(ids, dtype=torch.float32)
        position = self.position[role]
        if trim_trailing_padding:
            width = _active_prefix_width(ids)
            ids = ids[:, :width]
            counters = counters[:, :width]
            position = position[:width]
            if extra is not None:
                extra = extra[:, :width]
        token = shared.relics(ids, counters)
        if extra is not None:
            token = token + extra
        return role, token + position, ids.long() == 0

    def _potion_block(self, obs, shared, role, ids_key, extra=None):
        ids = obs[ids_key]
        token = shared.potions(ids)
        if extra is not None:
            token = token + extra
        return role, token + self.position[role], ids.long() == 0

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        shared: SharedSemanticTokenizer,
        phase_id: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = obs["phase"].shape[0]
        global_parts = [
            obs[key].float().reshape(batch_size, -1)
            for key in self.global_scalar_keys
        ]
        global_parts.extend(
            self.global_categories[key](obs[key].long()).reshape(batch_size, -1)
            for key in self.global_category_keys
        )
        global_token = self.global_projection(torch.cat(global_parts, dim=1)).unsqueeze(1)
        device = global_token.device
        blocks: list[tuple[str, torch.Tensor, torch.Tensor]] = [
            (
                "global", global_token,
                torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            ),
            self._card_block(
                obs, shared, "deck", "deck_ids", "deck_upgrades",
                trim_trailing_padding=True,
            ),
            self._relic_block(
                obs, shared, "relic", "relics", obs["relic_counters"],
                trim_trailing_padding=True,
            ),
            self._potion_block(obs, shared, "potion", "potions"),
        ]

        if phase_id == V.PHASE_VOCAB_INDEX["combat_reward"]:
            blocks.extend([
                self._card_block(
                    obs, shared, "offer", "offer_card_ids", "offer_card_upgrades"
                ),
                (
                    "reward_gold",
                    self.reward_gold_projection(obs["reward_gold"].float().unsqueeze(-1))
                    + self.position["reward_gold"],
                    obs["reward_gold"] <= 0,
                ),
                self._relic_block(obs, shared, "reward_relic", "reward_relic_ids"),
                self._potion_block(obs, shared, "reward_potion", "reward_potion_ids"),
            ])
        elif phase_id == V.PHASE_VOCAB_INDEX["card_select"]:
            blocks.extend([
                self._card_block(
                    obs, shared, "select", "select_card_ids", "select_card_upgrades"
                ),
                self._card_block(
                    obs, shared, "selected", "selected_card_ids",
                    "selected_card_upgrades",
                ),
            ])
        elif phase_id == V.PHASE_VOCAB_INDEX["map"]:
            map_token = self.map_projection(torch.cat([
                self.room_embedding(obs["map_node_rooms"].long()),
                obs["map_node_x"].float().unsqueeze(-1),
            ], dim=-1)) + self.position["map"]
            blocks.append(("map", map_token, obs["map_node_rooms"].long() == 0))
            if "map_full_rooms" in obs:
                ids = obs["map_full_rooms"].long()
                if self.map_dag is not None:
                    full = self.map_dag(
                        self.room_embedding(ids), obs["map_full_features"].float()
                    )
                else:
                    full = self.map_projection(torch.cat([
                        self.room_embedding(ids),
                        obs["map_full_features"][..., :1].float(),
                    ], dim=-1))
                blocks.append(
                    ("map_full", full + self.position["map_full"], ids == 0)
                )
        elif phase_id == V.PHASE_VOCAB_INDEX["shop"]:
            blocks.extend([
                self._card_block(
                    obs, shared, "shop_card", "shop_card_ids",
                    "shop_card_upgrades",
                    self.card_price(obs["shop_card_prices"].float().unsqueeze(-1)),
                ),
                self._relic_block(
                    obs, shared, "shop_relic", "shop_relic_ids", extra=
                    self.relic_price(obs["shop_relic_prices"].float().unsqueeze(-1)),
                ),
                self._potion_block(
                    obs, shared, "shop_potion", "shop_potion_ids",
                    self.potion_price(obs["shop_potion_prices"].float().unsqueeze(-1)),
                ),
            ])
        elif phase_id == V.PHASE_VOCAB_INDEX["boss_relic"]:
            blocks.append(
                self._relic_block(obs, shared, "boss_relic", "boss_relic_ids")
            )
        elif phase_id == V.PHASE_VOCAB_INDEX["event"]:
            if self.neow_enabled:
                neow = self.neow_projection(torch.cat([
                    self.neow_bonus(obs["neow_bonus_ids"].long()),
                    self.neow_drawback(obs["neow_drawback_ids"].long()),
                    obs["neow_features"].float(),
                ], dim=-1)) + self.position["neow"]
                blocks.append(("neow", neow, obs["neow_bonus_ids"].long() == 0))
            if self.event_option_enabled:
                event_option = self.event_option_projection(torch.cat([
                    self.event_for_option(obs["event_option_event_ids"].long()),
                    self.event_option_embedding(obs["event_option_ids"].long()),
                    obs["event_option_features"].float(),
                ], dim=-1)) + self.position["event_option"]
                blocks.append(
                    ("event_option", event_option, obs["event_option_ids"].long() == 0)
                )
        return self._run(batch_size, blocks)


class _SpecialistActorCritic(nn.Module):
    def __init__(
        self,
        encoder: _SpecialistEncoder,
        *,
        d_model: int,
        features_dim: int,
        vf_arch: tuple[int, ...],
        kind: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.kind = kind
        self.joint_actions = "combat_legal" in encoder.observation_space.spaces
        if self.joint_actions and kind == "combat":
            from .complete_combat import CompleteCombatHead
            self.complete_head = CompleteCombatHead(d_model)
        self.ptr_k = nn.Linear(d_model, d_model)
        self.ptr_q = nn.Linear(d_model, d_model, bias=False)
        self.ptr_v = nn.Linear(d_model, 1)
        self.fixed_head = nn.Linear(d_model, A.N_ACTIONS)
        layers: list[nn.Module] = []
        previous = features_dim
        for width in vf_arch:
            layers.extend([nn.Linear(previous, width), nn.GELU()])
            previous = width
        self.value_mlp = nn.Sequential(*layers)
        self.value_head = nn.Linear(previous, 1)
        self.ptr_v.weight.data.mul_(0.01)
        nn.init.zeros_(self.ptr_v.bias)
        self.fixed_head.weight.data.mul_(0.01)
        nn.init.zeros_(self.fixed_head.bias)

    def _score(self, token: torch.Tensor, global_token: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(
            self.ptr_k(token) + self.ptr_q(global_token).unsqueeze(1)
        )
        return self.ptr_v(hidden).squeeze(-1)

    def _assign(self, logits, segment, blocks, role):
        if role not in blocks:
            return
        score = self._score(blocks[role], blocks["__global__"])
        base = A.seg_base(segment)
        logits[:, base:base + score.shape[1]] = score

    def _map_logits(self, logits, obs, blocks):
        global_token = blocks["__global__"]
        if "map_full" in blocks and self.encoder.map_dag_enabled:
            scores = self._score(blocks["map_full"], global_token)
            features = obs["map_full_features"]
            choices = features[..., 3] > 0.5
            x_coordinates = (features[..., 0] * encode.MAP_X_SCALE).round().long()
        elif "map" in blocks:
            scores = self._score(blocks["map"], global_token)
            choices = obs["map_node_rooms"].long() != 0
            x_coordinates = (obs["map_node_x"] * encode.MAP_X_SCALE).round().long()
        else:
            return
        map_scores = torch.full(
            (scores.shape[0], A.N_COLS), -1e9,
            dtype=scores.dtype, device=scores.device,
        )
        for x in range(A.N_COLS):
            selected = torch.where(
                choices & (x_coordinates == x),
                scores,
                torch.full_like(scores, -1e9),
            )
            map_scores[:, x] = selected.max(dim=1).values
        base = A.seg_base("MAP_NODE")
        logits[:, base:base + A.N_COLS] = map_scores

    def forward(
        self, obs, shared, phase_id=None, *, active_relic_width=None, evaluate_subsets=True
    ):
        if self.kind == "combat":
            global_token, blocks = self.encoder(
                obs, shared, active_relic_width=active_relic_width
            )
        else:
            global_token, blocks = self.encoder(obs, shared, phase_id)
        blocks["__global__"] = global_token
        logits = self.fixed_head(global_token).clone()
        if self.kind == "combat":
            for segment, role in (
                ("PLAY_CARD", "hand"),
                ("PICK_TARGET", "monster"),
                ("USE_POTION", "potion"),
                ("PICK_CARD", "select" if self.encoder.public_choices else "hand"),
            ):
                self._assign(logits, segment, blocks, role)
        else:
            for segment, role in (
                ("USE_POTION", "potion"),
                ("PICK_CARD", "select"),
                ("REWARD_GOLD", "reward_gold"),
                ("REWARD_RELIC", "reward_relic"),
                ("REWARD_POTION", "reward_potion"),
                ("REWARD_CARD", "offer"),
                ("SHOP_CARD", "shop_card"),
                ("SHOP_RELIC", "shop_relic"),
                ("SHOP_POTION", "shop_potion"),
                ("BOSS_RELIC", "boss_relic"),
                ("NEOW", "neow"),
                ("EVENT_OPT", "event_option"),
            ):
                self._assign(logits, segment, blocks, role)
            self._map_logits(logits, obs, blocks)
        value_features = self.encoder.readout(global_token)
        values = self.value_head(self.value_mlp(value_features))
        if self.joint_actions:
            from env.core import native_actions as C
            if self.kind == "combat":
                sources = torch.cat((blocks['hand'], blocks['potion']), dim=1)
                targets = self.complete_head.targets(global_token, sources, blocks['monster'])
                logits = self.complete_head(logits, global_token, sources, blocks['monster'],
                                            obs['combat_legal'], evaluate_subsets=evaluate_subsets,
                                            target_logits=targets)
                # Legacy bridge pending views use exactly the root's target
                # factor. Pending slots never change the shared board encoder.
                pending_card = obs['pending_card_slot'].long().flatten() - 1
                pending_potion = obs['pending_potion_slot'].long().flatten() - 1
                selected = torch.where(pending_card >= 0, pending_card, 10 + pending_potion).clamp(0, 14)
                rows = torch.arange(logits.shape[0], device=logits.device)
                logits[:, A._TARGET:A._TARGET + 5] = targets[rows, selected, :5].float()
            else:
                logits = torch.nn.functional.pad(logits, (0, C.JOINT_WIDTH), value=-1e9)
        return logits, values


class SplitStSPolicyCore(nn.Module):
    """Routes each sample to exactly one independently parameterized model."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        card_language: bool = False,
        combat_language: bool = False,
        map_dag: bool = False,
        card_dim: int = 64,
        card_embedding: nn.Module | None = None,
        card_vector_checkpoint: str | None = None,
        features_dim: int = 256,
        vf_arch: tuple[int, ...] = (256,),
        combat_arch: str = "transformer",
    ) -> None:
        super().__init__()
        if card_vector_checkpoint is not None:
            if card_embedding is not None:
                raise ValueError(
                    "pass card_vector_checkpoint or card_embedding, not both"
                )
            card_embedding = load_frozen_card_rule_encoder(
                card_vector_checkpoint,
                expected_num_cards=_vocab_size(observation_space, "deck_ids"),
            )
            card_dim = card_embedding.embedding_dim
        if n_layers < 2:
            raise ValueError(
                "split policy requires --n-layers >= 2 so each specialist "
                "has at least one Transformer layer"
            )
        pile_language_fields = all(
            f"{pile}_card_ids" in observation_space.spaces
            for pile in ("draw", "discard", "deck")
        )
        if card_language != pile_language_fields:
            raise ValueError(
                "card_language must match the environment observation layout"
            )
        has_combat_language = "player_status_language" in observation_space.spaces
        if combat_language != has_combat_language:
            raise ValueError(
                "combat_language must match the environment observation layout"
            )
        combat_layers = max(1, (n_layers + 1) // 2)
        overworld_layers = max(1, n_layers // 2)
        self.specialist_layers = (combat_layers, overworld_layers)
        self.shared = SharedSemanticTokenizer(
            observation_space,
            d_model,
            card_language=card_language,
            card_dim=card_dim,
            card_embedding=card_embedding,
        )
        self.joint_actions = "combat_legal" in observation_space.spaces
        from env.core import native_actions as C
        if self.joint_actions and observation_space['combat_legal'].shape[-1] != C.JOINT_WIDTH:
            raise ValueError('autoregressive-v2 requires EntityCombatPolicy')
        if self.joint_actions and combat_arch != "transformer":
            raise ValueError("joint-v2 currently requires the Split Transformer combat policy")
        if combat_arch not in {"transformer", "mlp"}:
            raise ValueError("combat_arch must be transformer or mlp")
        if combat_arch == "transformer":
            combat_encoder = CombatEncoder(
                observation_space,
                d_model=d_model,
                n_layers=combat_layers,
                n_heads=n_heads,
                features_dim=features_dim,
                combat_language=combat_language,
                card_dim=card_dim,
            )
        overworld_encoder = OverworldEncoder(
            observation_space,
            d_model=d_model,
            n_layers=overworld_layers,
            n_heads=n_heads,
            features_dim=features_dim,
            map_dag=map_dag,
        )
        if combat_arch == "mlp":
            from rl.policies.combat_mlp import CombatMLPActorCritic

            self.combat = CombatMLPActorCritic(observation_space, COMBAT_OBSERVATION_KEYS)
        else:
            self.combat = _SpecialistActorCritic(
                combat_encoder,
                d_model=d_model,
                features_dim=features_dim,
                vf_arch=vf_arch,
                kind="combat",
            )
        self.overworld = _SpecialistActorCritic(
            overworld_encoder,
            d_model=d_model,
            features_dim=features_dim,
            vf_arch=vf_arch,
            kind="overworld",
        )
        self._combat_phase = V.PHASE_VOCAB_INDEX["combat"]
        self.map_dag_enabled = overworld_encoder.map_dag_enabled

    # Search-leaf observations can use this public, model-owned contract
    # without importing a policy implementation into the environment.
    combat_observation_keys = COMBAT_OBSERVATION_KEYS

    @staticmethod
    def _subset(obs, indices, allowed_keys):
        return {
            key: obs[key].index_select(0, indices)
            for key in allowed_keys
            if key in obs
        }

    @staticmethod
    def _keys_only(obs, allowed_keys):
        return {key: obs[key] for key in allowed_keys if key in obs}

    def combat_actor_critic(self, obs, *, active_relic_width=None, evaluate_subsets=True):
        """Evaluate a known all-combat batch without phase routing copies."""
        extra = {'evaluate_subsets': evaluate_subsets} if self.joint_actions else {}
        return self.combat(
            self._keys_only(obs, COMBAT_OBSERVATION_KEYS),
            self.shared,
            active_relic_width=active_relic_width,
            **extra,
        )

    def actor_critic(self, obs, *, combat_relic_width=None):
        phase = obs["phase"].long().reshape(-1)
        # The old router scanned every entry in the phase vocabulary and then
        # copied even homogeneous batches with index_select.  Search leaves are
        # overwhelmingly all combat, while ordinary rollout batches contain
        # only a handful of distinct phases.  Synchronize the small unique-id
        # list once and route only phases that are actually present.
        phase_ids = [int(value) for value in torch.unique(phase).tolist()]
        if len(phase_ids) == 1:
            phase_id = phase_ids[0]
            if phase_id == self._combat_phase:
                return self.combat_actor_critic(
                    obs, active_relic_width=combat_relic_width
                )
            allowed_keys = (
                OVERWORLD_OBSERVATION_KEYS
                | OVERWORLD_PHASE_OBSERVATION_KEYS.get(phase_id, frozenset())
            )
            return self.overworld(
                self._keys_only(obs, allowed_keys), self.shared, phase_id
            )

        indices_parts = []
        logits_parts = []
        value_parts = []

        for phase_id in phase_ids:
            indices = torch.nonzero(phase == phase_id, as_tuple=False).flatten()
            if phase_id == self._combat_phase:
                logits, values = self.combat(
                    self._subset(obs, indices, COMBAT_OBSERVATION_KEYS),
                    self.shared,
                    active_relic_width=combat_relic_width,
                )
                indices_parts.append(indices)
                logits_parts.append(logits)
                value_parts.append(values)
                continue
            allowed_keys = (
                OVERWORLD_OBSERVATION_KEYS
                | OVERWORLD_PHASE_OBSERVATION_KEYS.get(phase_id, frozenset())
            )
            logits, values = self.overworld(
                self._subset(obs, indices, allowed_keys), self.shared, phase_id
            )
            indices_parts.append(indices)
            logits_parts.append(logits)
            value_parts.append(values)

        if not indices_parts:
            raise ValueError("cannot route an empty observation batch")
        order = torch.cat(indices_parts)
        restore = torch.argsort(order)
        return torch.cat(logits_parts)[restore], torch.cat(value_parts)[restore]

    def action_logits(self, obs):
        return self.actor_critic(obs)[0]

    def value(self, obs):
        return self.actor_critic(obs)[1]


class CombatStSPolicyCore(nn.Module):
    """Standalone combat specialist used by combat-only pretraining."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        card_language: bool = False,
        combat_language: bool = False,
        card_dim: int = 64,
        card_embedding: nn.Module | None = None,
        card_vector_checkpoint: str | None = None,
        features_dim: int = 256,
        vf_arch: tuple[int, ...] = (256,),
        **_unused,
    ) -> None:
        super().__init__()
        if _unused.get("combat_arch", "transformer") != "transformer":
            raise ValueError("combat-only pretraining currently supports transformer combat_arch only")
        if card_vector_checkpoint is not None:
            if card_embedding is not None:
                raise ValueError(
                    "pass card_vector_checkpoint or card_embedding, not both"
                )
            card_embedding = load_frozen_card_rule_encoder(
                card_vector_checkpoint,
                expected_num_cards=_vocab_size(observation_space, "deck_ids"),
            )
            card_dim = card_embedding.embedding_dim
        pile_language_fields = all(
            f"{pile}_card_ids" in observation_space.spaces
            for pile in ("draw", "discard", "deck")
        )
        if card_language != pile_language_fields:
            raise ValueError(
                "card_language must match the environment observation layout"
            )
        has_combat_language = "player_status_language" in observation_space.spaces
        if combat_language != has_combat_language:
            raise ValueError(
                "combat_language must match the environment observation layout"
            )
        self.shared = SharedSemanticTokenizer(
            observation_space,
            d_model,
            card_language=card_language,
            card_dim=card_dim,
            card_embedding=card_embedding,
        )
        encoder = CombatEncoder(
            observation_space,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            features_dim=features_dim,
            combat_language=combat_language,
            card_dim=card_dim,
        )
        self.combat = _SpecialistActorCritic(
            encoder,
            d_model=d_model,
            features_dim=features_dim,
            vf_arch=vf_arch,
            kind="combat",
        )
        self.map_dag_enabled = False

    def actor_critic(self, obs):
        combat_obs = {
            key: obs[key]
            for key in COMBAT_OBSERVATION_KEYS
            if key in obs
        }
        return self.combat(combat_obs, self.shared)

    def action_logits(self, obs):
        return self.actor_critic(obs)[0]

    def value(self, obs):
        return self.actor_critic(obs)[1]


class SplitStSPointerPolicy(MaskableMultiInputActorCriticPolicy):
    """SB3 adapter for :class:`SplitStSPolicyCore`."""

    supports_bf16_rollout = True

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        card_language: bool = False,
        combat_language: bool = False,
        map_dag: bool = False,
        card_vector_checkpoint: str | None = None,
        combat_arch: str = "transformer",
        **kwargs,
    ) -> None:
        self._core_cfg = dict(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            card_language=card_language,
            combat_language=combat_language,
            map_dag=map_dag,
            card_vector_checkpoint=card_vector_checkpoint,
            combat_arch=combat_arch,
        )
        kwargs["features_extractor_class"] = _NullExtractor
        kwargs.pop("features_extractor_kwargs", None)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        self.core = SplitStSPolicyCore(self.observation_space, **self._core_cfg)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(self, obs, deterministic: bool = False, action_masks=None):
        logits, values = self.core.actor_critic(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values.float(), log_prob.float()

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self.core.actor_critic(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return (
            values.float(),
            distribution.log_prob(actions).float(),
            distribution.entropy().float(),
        )

    def get_distribution(self, obs, action_masks=None):
        distribution = self.action_dist.proba_distribution(
            action_logits=self.core.action_logits(obs)
        )
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs):
        return self.core.value(obs).float()


class CombatStSPointerPolicy(SplitStSPointerPolicy):
    """Combat-only adapter with no overworld model parameters."""

    def _build(self, lr_schedule) -> None:
        self.core = CombatStSPolicyCore(self.observation_space, **self._core_cfg)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
