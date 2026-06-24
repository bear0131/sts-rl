from __future__ import annotations

from collections.abc import Mapping
from math import prod

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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


class StSFeatureExtractor(BaseFeaturesExtractor):
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
