"""
aggregation.py — Token aggregation and geometric feature extraction.

Strategy
--------
* The hallucination signal lives in the *response* tokens, which appear at
  the tail of ``prompt + response``.  We therefore restrict pooling to the
  last ``RESPONSE_WINDOW`` real (non-padding) tokens, falling back to all
  real tokens for very short sequences.

* Multiple transformer layers carry complementary signal: middle layers
  encode factual content, late layers encode generation confidence.  We
  concatenate mean-pooled response vectors from a small set of evenly
  spaced layers plus the final-layer last-token vector.

* Geometric features capture representation drift through the network —
  norms and inter-layer cosine similarities of the pooled response vector.
"""

from __future__ import annotations

import torch

# Layers of the hidden_states tuple to use for aggregation.
# index 0 = token embeddings, index k = transformer layer k.
# Qwen2.5-0.5B has 24 transformer layers => indices 1..24 are valid.
# We pick four evenly spaced layers spanning middle->late representations,
# which empirically carry the strongest factuality signal.  Combined with
# the final-layer last-token vector this gives 5 × 896 = 4480 features.
SELECTED_LAYERS: tuple[int, ...] = (6, 12, 18, 24)

# Number of trailing real tokens to mean-pool over.  Covers the response;
# we clamp to the number of real tokens when sequences are short.
RESPONSE_WINDOW: int = 32


def _response_slice(attention_mask: torch.Tensor) -> tuple[int, int]:
    """Return ``(start, end)`` indices of the trailing response window.

    The response sits at the tail of the (left- or right-padded) sequence.
    We use the last ``RESPONSE_WINDOW`` real tokens, but clamp to the
    available real-token count.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    if real_positions.numel() == 0:
        return 0, 1
    last = int(real_positions[-1].item()) + 1
    first_real = int(real_positions[0].item())
    n_real = last - first_real
    window = min(RESPONSE_WINDOW, n_real)
    return last - window, last


def _mean_pool(layer: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Mean-pool a single layer over a token slice."""
    return layer[start:end].mean(dim=0)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    For each selected layer we compute two complementary statistics of
    the hidden state over the response window:
        * mean-pool — average representation of the response;
        * max-pool  — peak activations, which often spike on uncertain
                       or unusual tokens characteristic of hallucinations.
    Plus a single global feature:
        * last real-token vector at the final transformer layer — the
          model's last-step "summary" representation that traditional
          probes (SAPLMA) use on its own.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is
                        the final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of length
            (2 * len(SELECTED_LAYERS) + 1) * hidden_dim
        — for 4 layers and 896 hidden dim this is 9 * 896 = 8064 floats.
    """
    start, end = _response_slice(attention_mask)

    n_layers = hidden_states.size(0)

    parts: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        # Map possibly out-of-range indices to the last available layer.
        idx = layer_idx if layer_idx < n_layers else n_layers - 1
        layer = hidden_states[idx]
        window = layer[start:end]            # (window_len, hidden_dim)
        parts.append(window.mean(dim=0))     # mean-pool
        parts.append(window.amax(dim=0))     # max-pool over response tokens

    # Last real token from the final transformer layer — kept alongside
    # the pooled vectors as a classic strong probe feature.
    last_pos = end - 1
    parts.append(hidden_states[-1][last_pos])

    return torch.cat(parts, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hand-crafted geometric features describing representation dynamics.

    Computed on the mean-pooled response vector of every layer:
        * L2 norm per layer                                   -> n_layers
        * Cosine similarity between consecutive-layer vectors -> n_layers - 1
        * Mean / std of inter-layer cosines                   -> 2
        * Norm ratio (last_layer / first_layer)               -> 1
        * Real-token count (normalised by 512)                -> 1
        * Response-window length (normalised by 512)          -> 1

    Args:
        hidden_states:  ``(n_layers, seq_len, hidden_dim)``
        attention_mask: ``(seq_len,)``

    Returns:
        1-D float tensor of fixed length.
    """
    start, end = _response_slice(attention_mask)
    n_layers = hidden_states.size(0)

    # Per-layer mean-pooled response vectors -> (n_layers, hidden_dim)
    pooled = hidden_states[:, start:end, :].mean(dim=1)

    norms = pooled.norm(dim=-1)  # (n_layers,)

    # Inter-layer cosine similarity (consecutive layers).
    a = pooled[:-1]
    b = pooled[1:]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # (n_layers - 1,)

    cos_mean = cos.mean().unsqueeze(0)
    cos_std = cos.std(unbiased=False).unsqueeze(0)

    eps = torch.tensor(1e-6, dtype=norms.dtype, device=norms.device)
    norm_ratio = (norms[-1] / torch.maximum(norms[0], eps)).unsqueeze(0)

    real_count = (
        attention_mask.to(norms.device).float().sum().unsqueeze(0) / 512.0
    )
    window_len = torch.tensor(
        [(end - start) / 512.0], dtype=norms.dtype, device=norms.device
    )

    return torch.cat([norms, cos, cos_mean, cos_std, norm_ratio, real_count, window_len])


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
