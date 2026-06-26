"""Structured candidate (pointer) policy for the whole-run action space.

THE EXPERIMENT (controlled A/B vs the flat baseline): everything else — env,
reward, rollout, GAE, PPO hyperparameters, action mask, training steps — is held
fixed. The ONLY thing that changes is how the 149 action logits are produced:

    transformer-flat (baseline):  Linear(features_256 -> 149) over a pooled [CLS]
    transformer-pointer (this):   per-entity POINTER scoring

Why pointer: the action ids are entity-indexed. `PLAY_CARD[h]` means "play the
card in hand slot h", `SHOP_CARD[s]` means "buy shelf slot s", etc. The card in a
slot changes every state, so a fixed per-slot output weight (the flat head) has to
reconstruct "what is in slot h" from a pooled vector that already smeared the
per-token structure. A pointer head instead scores each candidate from ITS OWN
token: logit(PLAY_CARD[h]) = f(hand_token_h). The token already carries the card
embedding + a role + a slot-position embedding (see StSTransformerExtractor), so a
single shared scorer handles every entity family.

Segments with NO entity token in obs (END_TURN, MAP_NODE, CAMPFIRE, EVENT_OPT,
NEOW, CHEST, the reward gold/relic/potion claims, SHOP_REMOVE, PROCEED,
CONFIRM_SELECT) keep a flat head over [CLS] — that is the only thing available for
them, and it matches what the baseline did there anyway.

`StSPolicyCore` is framework-agnostic (no SB3 import). `StSPointerPolicy` is a
thin SB3 adapter; if we later hand-write a PPO loop, the core moves over verbatim.
"""

from __future__ import annotations

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from env.core import action_space as A
from env.core import vocab as V
from rl.model import StSTransformerExtractor


# action segments whose ids line up 1:1 with a per-entity token family.
# (segment_name, extractor token_pos role). Widths come from the token slot count,
# which equals the segment width for these (verified against encode.obs_space()).
_ENTITY_SEGMENTS = [
    ("PLAY_CARD", "hand"),          # play hand slot h        <- hand token h
    ("PICK_TARGET", "monster"),     # target monster t        <- monster token t
    ("USE_POTION", "potion_held"),  # use held potion p       <- potion token p (reserved)
    ("SHOP_CARD", "shop_card"),     # buy shelf card s        <- shop card token s
    ("SHOP_RELIC", "shop_relic"),   # buy shelf relic s       <- shop relic token s
    ("SHOP_POTION", "shop_potion"), # buy shelf potion s      <- shop potion token s
    ("BOSS_RELIC", "boss_relic"),   # take boss relic i       <- boss relic token i
    ("REWARD_CARD", "offer"),       # take card reward i      <- offer token i (first 3;
                                    #   obs tokenizes only OFFER_SLOTS of up-to-5)
]


class _NullExtractor(BaseFeaturesExtractor):
    """No-op features extractor. StSPolicyCore owns the whole network, but
    MaskableActorCriticPolicy.__init__ still builds a features extractor — make it
    a parameter-free stub that is never actually called on the hot path."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)

    def forward(self, obs):
        ref = next(iter(obs.values()))
        return ref.new_zeros((ref.shape[0], 1), dtype=torch.float32)


class StSPolicyCore(nn.Module):
    """Framework-agnostic actor-critic core.

    encode(obs)        -> (token_seq, pad_mask)   from the shared Transformer
    action_logits(obs) -> (B, N_ACTIONS)          pointer + flat head
    value(obs)         -> (B, 1)                   identical-shape baseline vf path
    actor_critic(obs)  -> (logits, value)          one Transformer pass (hot path)
    """

    def __init__(
        self,
        observation_space,
        *,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        features_dim: int = 256,
        vf_arch: tuple[int, ...] = (512, 256),
    ) -> None:
        super().__init__()
        self.encoder = StSTransformerExtractor(
            observation_space,
            features_dim=features_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        d = d_model

        # Pointer scorer: Bahdanau additive attention, shared across every entity
        # family. score(token, cls) = v . tanh(W_k token + W_q cls). The token's
        # role + slot-position embeddings let one shared scorer specialize.
        self.ptr_k = nn.Linear(d, d)
        self.ptr_q = nn.Linear(d, d, bias=False)
        self.ptr_v = nn.Linear(d, 1)

        # Flat head over [CLS] for phase-fixed (non-entity) segments. Sized to the
        # full table for index convenience; entity columns are overwritten by
        # pointer scores and simply never train.
        self.fixed_head = nn.Linear(d, A.N_ACTIONS)

        # Value head — same architecture as the baseline vf path (CLS readout ->
        # MLP[512,256] -> Linear 1), so an A/B isolates the action head.
        layers, prev = [], features_dim
        for w in vf_arch:
            layers += [nn.Linear(prev, w), nn.GELU()]
            prev = w
        self.vf_mlp = nn.Sequential(*layers)
        self.value_net = nn.Linear(prev, 1)

        # Small output init (SB3 uses a 0.01 action gain): near-uniform policy at
        # start so masking, not a confident wrong prior, drives early steps.
        self.ptr_v.weight.data.mul_(0.01)
        nn.init.zeros_(self.ptr_v.bias)
        self.fixed_head.weight.data.mul_(0.01)
        nn.init.zeros_(self.fixed_head.bias)

        # PICK_CARD reads hand tokens in combat (the id is a hand index there) and
        # grid (select) tokens in the overworld card_select screen.
        self._combat_phase = V.PHASE_VOCAB_INDEX["combat"]

    # ---- encoding ----------------------------------------------------------
    def encode(self, obs):
        return self.encoder.encode_tokens(obs)

    def _block(self, out, role):
        s, n = self.encoder.token_pos[role]
        return out[:, s:s + n]

    def _score(self, tokens, g):
        # tokens (B, n, d), g (B, d) -> (B, n)
        h = torch.tanh(self.ptr_k(tokens) + self.ptr_q(g).unsqueeze(1))
        return self.ptr_v(h).squeeze(-1)

    # ---- heads -------------------------------------------------------------
    def _logits(self, out, combat):
        g = out[:, 0]                       # [CLS]
        logits = self.fixed_head(g).clone()  # fixed segments; entity ones overwritten

        for seg, role in _ENTITY_SEGMENTS:
            scores = self._score(self._block(out, role), g)
            b = A.seg_base(seg)
            logits[:, b:b + scores.shape[1]] = scores

        # PICK_CARD dual source: grid tokens by default, hand tokens in combat.
        # The mask makes only one context legal at a time (combat in-hand select
        # vs overworld grid), so a per-phase pick is sufficient.
        sel = self._score(self._block(out, "select"), g)    # (B, G_MAX)
        hand = self._score(self._block(out, "hand"), g)     # (B, HAND_SLOTS)
        k = hand.shape[1]
        pick = sel.clone()
        pick[:, :k] = torch.where(combat.unsqueeze(1), hand, sel[:, :k])
        b = A.seg_base("PICK_CARD")
        logits[:, b:b + pick.shape[1]] = pick
        return logits

    def _value(self, out):
        feat = self.encoder.readout(out[:, 0])
        return self.value_net(self.vf_mlp(feat))

    def _combat(self, obs):
        return obs["phase"].long().reshape(-1) == self._combat_phase

    # ---- public API --------------------------------------------------------
    def action_logits(self, obs):
        out, _ = self.encode(obs)
        return self._logits(out, self._combat(obs))

    def value(self, obs):
        out, _ = self.encode(obs)
        return self._value(out)

    def actor_critic(self, obs):
        out, _ = self.encode(obs)
        return self._logits(out, self._combat(obs)), self._value(out)


class StSPointerPolicy(MaskableMultiInputActorCriticPolicy):
    """Thin SB3 adapter around StSPolicyCore.

    SB3's MaskableDistribution, rollout collection, GAE, PPO update, and action
    masking are all left untouched. Only the network is replaced: the four methods
    that turn observations into a distribution/value delegate to the core, and
    _build swaps the core in for SB3's mlp_extractor/action_net/value_net.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        **kwargs,
    ) -> None:
        self._core_cfg = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        # The core owns the whole net; force SB3's features extractor to a stub.
        kwargs["features_extractor_class"] = _NullExtractor
        kwargs.pop("features_extractor_kwargs", None)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        self.core = StSPolicyCore(self.observation_space, **self._core_cfg)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(self, obs, deterministic: bool = False, action_masks=None):
        logits, values = self.core.actor_critic(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self.core.actor_critic(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return values, dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs, action_masks=None):
        dist = self.action_dist.proba_distribution(
            action_logits=self.core.action_logits(obs)
        )
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    def predict_values(self, obs):
        return self.core.value(obs)
