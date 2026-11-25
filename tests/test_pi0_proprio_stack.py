#!/usr/bin/env python
"""Conservation tests for F6/S2's three-stream (vlm / proprio / action) fusion.

The refactor has to be *inert* when the proprio stack is off and *mathematically
equal to the two-stream path* when the proprio block reuses the action expert's
weights. Both properties are checked here with freshly-initialised weights, so the
suite runs on CPU in seconds and needs no π0* checkpoint.

π0* ties the proprio mixture to the action mixture (`pizero.py:274-276`), so the
three-stream model differs from the two-stream one by exactly one tensor pair:
`proprio_in_proj.{weight,bias}`. Given equal weights a three-stream model must
reproduce the two-stream model token for token. That can only happen if

* the suffix is re-interleaved back to the mask's block order
  (``[pro_n, act_n(0..P-1)]`` per block -- see ``PI0Attention._pack_streams``),
* each stream's ``o_proj`` gets its own slice of the joint attention output,
* and the state token is embedded identically (``proprio_in_proj`` <- the first
  ``proprio_dim`` columns of ``state_proj``, with the padding zeroed).

A plain ``cat([vlm, action, proprio])`` violates the first condition, which is
exactly the bug this test exists to catch.

Run::

    <env>/bin/python -m pytest tests/test_pi0_proprio_stack.py -q
    <env>/bin/python tests/test_pi0_proprio_stack.py
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402

from flashvla.policies.pi0.modeling_pi0_flashvla import PI0FlashVLAPolicy  # noqa: E402

CKPT = REPO_ROOT / "ckpt" / "pi0star_fractal_flashvla_proprio"

# Two layers keep the suite fast; width/heads/kv-heads/MLP stay production-sized
# and depth plays no part in the fusion arithmetic.
TEST_LAYERS = 2


def _config(*, slots: int, chunk: int, use_proprio_stack: bool):
    # `config.json` carries lerobot's `type` discriminator, which
    # `PI0FlashVLAConfig.from_pretrained` rejects; use the generic parser.
    config = copy.deepcopy(
        PreTrainedConfig.from_pretrained(pretrained_name_or_path=str(CKPT))
    )
    config.num_buffer_slots = slots
    config.chunk_size = chunk
    config.use_proprio_stack = use_proprio_stack
    config.proprio_dim = config.real_state_dim
    config.compile_model = False
    config.vlm_config.text_config.num_hidden_layers = TEST_LAYERS
    config.action_expert_config.num_hidden_layers = TEST_LAYERS
    return config


_PAIR_CACHE: dict = {}


def _build_pair(*, slots: int, chunk: int):
    """Three-stream policy plus an identically-weighted two-stream twin."""
    key = (slots, chunk)
    if key in _PAIR_CACHE:
        return _PAIR_CACHE[key]
    config3 = _config(slots=slots, chunk=chunk, use_proprio_stack=True)
    policy3 = PI0FlashVLAPolicy(config3).eval()

    config2 = _config(slots=slots, chunk=chunk, use_proprio_stack=False)
    policy2 = PI0FlashVLAPolicy(config2).eval()

    sd3 = policy3.state_dict()
    sd2 = policy2.state_dict()

    # π0* ties proprio to action, so the two models must differ by exactly one
    # weight (the input encoder). A third weight set -- or a duplicated decoder
    # key -- is a regression: it inflates the checkpoint and would hand the
    # proprio block parameters π0* never had.
    extra = sorted(set(sd3) - set(sd2))
    assert extra == [
        "model.suffix_embedder.proprio_in_proj.bias",
        "model.suffix_embedder.proprio_in_proj.weight",
    ], f"three-stream model added unexpected tensors: {extra}"
    missing = sorted(set(sd2) - set(sd3))
    assert not missing, f"two-stream tensors absent from the three-stream model: {missing}"

    policy2.load_state_dict({k: v for k, v in sd3.items() if k in sd2}, strict=False)

    with torch.no_grad():
        # state token: state_proj(state[..., :d]) == proprio_in_proj(state[..., :d])
        d = config3.proprio_input_dim
        policy3.model.suffix_embedder.proprio_in_proj.weight.copy_(
            policy3.model.suffix_embedder.state_proj.weight[:, :d]
        )
        policy3.model.suffix_embedder.proprio_in_proj.bias.copy_(
            policy3.model.suffix_embedder.state_proj.bias
        )
    _PAIR_CACHE[key] = (policy3, policy2)
    return policy3, policy2


def _make_inputs(policy, *, text_len: int, seed: int, pad: bool):
    config = policy.config
    model = policy.model
    num_views = len(config.image_features)
    image_size = (3, *config.image_resolution)
    gen = torch.Generator().manual_seed(seed)
    bsz = 1

    images = [
        (torch.rand(bsz, *image_size, generator=gen) * 2.0 - 1.0) for _ in range(num_views)
    ]
    img_masks = [torch.ones(bsz, dtype=torch.bool) for _ in range(num_views)]
    tokens = torch.randint(0, 256, (bsz, text_len), generator=gen)
    masks = torch.ones(bsz, text_len, dtype=torch.bool)

    N, C = model.N, model.C
    real_state = config.real_state_dim
    states = torch.zeros(bsz, config.max_state_dim)
    states[:, :real_state] = torch.randn(bsz, real_state, generator=gen) * 0.1

    total = N * N * C
    actions = torch.zeros(bsz, total, config.max_action_dim)
    actions[..., : config.real_action_dim] = (
        torch.randn(bsz, total, config.real_action_dim, generator=gen) * 0.05
    )
    noise = torch.randn(bsz, total, config.max_action_dim, generator=gen)

    action_is_pad = None
    if pad:
        # Cold-start schedule: config k only has its first k+1 slots real.
        idx = torch.arange(total)
        slot_of_token = idx % (N * C) // C
        config_of_token = idx // (N * C)
        pad_flat = slot_of_token > config_of_token
        action_is_pad = pad_flat.unsqueeze(0).expand(bsz, -1).contiguous()

    return {
        "images": images,
        "img_masks": img_masks,
        "tokens": tokens,
        "masks": masks,
        "states": states,
        "actions": actions,
        "action_is_pad": action_is_pad,
        "noise": noise,
    }


def _forward(policy, batch, *, seed: int):
    torch.manual_seed(seed)  # per-slot Beta time sampling draws from the global RNG
    return policy.model.forward_shared_observation(
        batch["images"],
        batch["img_masks"],
        batch["tokens"],
        batch["masks"],
        batch["states"],
        batch["actions"],
        batch["action_is_pad"],
        noise=batch["noise"],
    )


class ThreeStreamEquivalenceTest(unittest.TestCase):
    def _check(self, *, slots: int, chunk: int, pad: bool, atol: float = 1e-5):
        policy3, policy2 = _build_pair(slots=slots, chunk=chunk)
        batch = _make_inputs(policy2, text_len=8, seed=7, pad=pad)
        with torch.no_grad():
            losses3 = _forward(policy3, batch, seed=11)
            losses2 = _forward(policy2, batch, seed=11)
        self.assertEqual(tuple(losses3.shape), tuple(losses2.shape))
        self.assertTrue(torch.isfinite(losses3).all(), "three-stream loss is not finite")
        diff = (losses3 - losses2).abs().max().item()
        self.assertLessEqual(
            diff,
            atol,
            f"three-stream forward drifted from two-stream by {diff:.3e} "
            f"(slots={slots}, chunk={chunk}, pad={pad})",
        )

    def test_small_multiblock(self):
        # N=2 blocks of [pro, act x C]: the smallest layout with more than one
        # block, which is what a mis-interleaved cat() would break.
        self._check(slots=2, chunk=2, pad=False)

    def test_production_shape(self):
        self._check(slots=4, chunk=4, pad=False)

    def test_padded_actions(self):
        self._check(slots=4, chunk=4, pad=True)


class ProprioGradientCoverageTest(unittest.TestCase):
    """The state path must be trainable, and the shared weights must stay shared."""

    def test_proprio_stack_is_trained(self):
        policy3, policy2 = _build_pair(slots=2, chunk=2)
        policy3.train()
        policy3.zero_grad(set_to_none=True)
        batch = _make_inputs(policy3, text_len=8, seed=3, pad=False)
        losses = _forward(policy3, batch, seed=5)
        losses.mean().backward()

        # `remove_duplicate=False`: the action layer is reachable both as
        # `model.action_expert.model.layers.*` and `model.layers.*.self_attn.action_expert_attention.*`.
        graded = {
            name
            for name, p in policy3.named_parameters(remove_duplicate=False)
            if p.grad is not None
        }
        self.assertIn("model.suffix_embedder.proprio_in_proj.weight", graded)
        self.assertIn("model.suffix_embedder.proprio_in_proj.bias", graded)
        # The action stack now serves *both* suffix streams, so it must be trained.
        for proj in ("q", "k", "v", "o"):
            self.assertIn(
                f"model.layers.0.self_attn.action_expert_attention.{proj}_proj.weight", graded
            )
        self.assertIsNone(
            policy3.model.suffix_embedder.state_proj.weight.grad,
            "state_proj must stay frozen on the proprio-stack path",
        )

    def test_streams_one_and_two_share_weights(self):
        policy3, policy2 = _build_pair(slots=2, chunk=2)
        layer = policy3.model.layers[0]
        attns = layer.self_attn._stream_attentions()
        mlps = layer.mlp._stream_mlps()
        self.assertIs(attns[1], attns[2], "proprio stream must reuse the action attention")
        self.assertIs(mlps[1], mlps[2], "proprio stream must reuse the action MLP")
        self.assertEqual(len(layer.input_layernorm), 2, "one norm per weight set, not per stream")
        self.assertEqual(
            len(list(policy3.named_parameters())),
            len(list(policy2.named_parameters())) + 2,
            "the third stream must cost exactly proprio_in_proj.{weight,bias}",
        )


class StreamCountGuardTest(unittest.TestCase):
    """A length mismatch must raise instead of being swallowed by zip()."""

    def test_two_stream_model_rejects_three_streams(self):
        _, policy2 = _build_pair(slots=2, chunk=2)
        model = policy2.model
        width = policy2.config.action_expert_config.hidden_size
        hidden = [torch.zeros(1, 4, width) for _ in range(3)]
        with self.assertRaises(ValueError):
            model.layers[0](
                hidden,
                torch.zeros(1, 1, 12, 12, dtype=torch.bool),
                torch.zeros(1, 12, dtype=torch.long),
                [None, None, None],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
