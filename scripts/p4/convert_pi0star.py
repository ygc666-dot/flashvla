#!/usr/bin/env python
"""Convert an open-pi-zero / π0* checkpoint into a FlashVLA `pi0-flashvla` directory.

π0* ships as a single ``torch.save`` blob (``model`` + optimizer states) whose
key space is the open-pi-zero "joint mixture" layout::

    embed_tokens.weight                      (257216, 2048)
    vision_tower.vision_model.*              (SigLIP so400m, 437 tensors)
    multi_modal_projector.linear.*           (2048 -> 1152)
    joint_model.mixtures.vlm.layers.{i}.*    (2048 hidden, 18 layers)
    joint_model.mixtures.action.layers.{i}.* (1024 hidden, 18 layers)
    joint_model.mixtures.action.norm.weight  (1024)
    joint_model.mixtures.proprio.layers.{i}.*(1024 hidden, 18 layers)  -> dropped
                                       (π0* ties them to the action mixture)
    proprio_encoder.*                        (1024, 7 or 1024, 8)  -> proprio_in_proj
                                       (only with `--emit-proprio-stack`)
    action_encoder.linear_1/2/3.*            action_in_proj / action_time_mlp_in / _out
    action_decoder.*                         (7, 1024)               -> action_out_proj

FlashVLA's ``pi0-flashvla`` fuses the VLM and action-expert decoder stacks into
one joint layer per depth (``model.layers.{i}.self_attn.vlm_attention`` etc.), so
the converter writes the **alias-free** names. Every parameter FlashVLA expects
must be present: the tensors π0* does not carry (two unused lm_heads, the action
expert embedding, the VLM final RMSNorm, the state projection) are synthesized
here and reported, so ``assert_checkpoint_covers_parameters`` stays meaningful.

Usage (verification only, writes nothing)::

    python scripts/p4/convert_pi0star.py \
        --src third_party/SDN/pretrained/open-pi-zero/bridge_beta_step19296_2024-12-26_22-30_42.pt \
        --data /data/zjr/simpler_lerobot/bridge/widowx_stack_cube/success \
        --json logs/p4/convert_bridge.json

Add ``--write --dst ckpt/pi0star_bridge_flashvla`` to materialize
``model.safetensors`` + ``config.json``. The destination can then be used as
``policy.pretrained_path`` in a training yaml.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VLM = "vlm"
ACTION = "action"
PROPRIO = "proprio"

LAYER_RE = re.compile(r"^joint_model\.mixtures\.(vlm|action|proprio)\.layers\.(\d+)\.(.+)$")

# target key space of `pi0-flashvla`
VLM_EMBED = "model.vlm.model.language_model.embed_tokens.weight"
VLM_HEAD = "model.vlm.lm_head.weight"
VLM_NORM = "model.vlm.model.language_model.norm.weight"
EXPERT_EMBED = "model.action_expert.model.embed_tokens.weight"
EXPERT_HEAD = "model.action_expert.lm_head.weight"
EXPERT_NORM = "model.action_expert.model.norm.weight"
ACTION_IN = "model.suffix_embedder.action_in_proj"
TIME_IN = "model.suffix_embedder.action_time_mlp_in"
TIME_OUT = "model.suffix_embedder.action_time_mlp_out"
STATE_PROJ = "model.suffix_embedder.state_proj"
ACTION_OUT = "model.action_out_proj"

# F6 target name for π0*'s proprio pathway. π0* ties the proprio mixture to the
# action mixture (`pizero.py:274-276`), so the *only* proprio-specific tensor is
# the input encoder; the block itself rides on the action expert's weights.
PROPRIO_IN = "model.suffix_embedder.proprio_in_proj"

# π0* fractal ships its proprio in the EDR adapter's layout, which is *not* the
# FlashVLA dataset layout even though both are 8-D:
#   π0*  (simpler_adapter.py:222-235, EDRSimplerAdapter.preprocess_proprio)
#        [x, y, z, qx, qy, qz, qw, 1 - grip]      (np.roll(wxyz, -1), closedness)
#   ours (sim_eval/simpler/convert_simpler_to_lerobot.py:55 STATE_NAMES)
#        [x, y, z, qw, qx, qy, qz, grip]          (raw eef_pos, openness)
# Feeding the pretrained encoder the raw state therefore permutes the quaternion
# and flips the gripper. ``PROPRIO_TO_STATE_PERM`` maps our index -> π0* index.
PROPRIO_TO_STATE_PERM = (0, 1, 2, 6, 3, 4, 5, 7)
# π0* grips are ``1 - openness``: absorbing the constant into the bias turns the
# last column's sign and adds the removed weight to the bias.
PROPRIO_TO_STATE_SIGN = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0)


def _same(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous()


def _pad_cols(tensor: torch.Tensor, width: int) -> torch.Tensor:
    """Right-pad a 2-D linear weight from ``[out, in]`` to ``[out, width]`` with zeros."""
    out_dim, in_dim = tensor.shape
    if in_dim == width:
        return _same(tensor)
    if in_dim > width:
        raise ValueError(f"cannot pad {tuple(tensor.shape)} down to width {width}")
    padded = torch.zeros(out_dim, width, dtype=tensor.dtype)
    padded[:, :in_dim] = tensor
    return padded


def _pad_rows(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Right-pad the leading dimension (action output projection) with zeros."""
    if tensor.ndim == 1:
        if tensor.shape[0] == size:
            return _same(tensor)
        padded = torch.zeros(size, dtype=tensor.dtype)
        padded[: tensor.shape[0]] = tensor
        return padded
    lead, rest = tensor.shape
    if lead == size:
        return _same(tensor)
    if lead > size:
        raise ValueError(f"cannot pad {tuple(tensor.shape)} down to {size} rows")
    padded = torch.zeros(size, rest, dtype=tensor.dtype)
    padded[:lead] = tensor
    return padded


def _align_proprio_encoder(
    weight: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rewrite ``proprio_encoder`` so it consumes FlashVLA's state layout.

    ``y = W @ p + b`` with ``p`` in π0* order becomes ``y = W' @ s + b'`` with
    ``s`` in FlashVLA order, where ``W' = W[:, perm] * sign`` and
    ``b' = b + W[:, 7]`` (the ``1 - grip`` constant folded into the bias).
    """
    if weight.shape[1] != len(PROPRIO_TO_STATE_PERM):
        raise ValueError(
            f"proprio encoder width {weight.shape[1]} does not match the 8-D "
            f"π0*/FlashVLA state layout this alignment assumes"
        )
    perm = torch.tensor(PROPRIO_TO_STATE_PERM, device=weight.device)
    sign = torch.tensor(PROPRIO_TO_STATE_SIGN, dtype=weight.dtype, device=weight.device)
    aligned = weight[:, perm] * sign
    aligned_bias = bias + weight[:, PROPRIO_TO_STATE_PERM[-1]].to(bias.dtype)
    return aligned.contiguous(), aligned_bias.contiguous()


def _layer_tail_target(stream: str, index: int, tail: str) -> str | None:
    """Map one tensor inside a π0* mixture layer to its FlashVLA joint-layer name.

    ``PROPRIO`` layers map to ``None`` on purpose: π0* ties them to the action
    mixture, so the file's copies are duplicates of tensors already emitted and
    keeping them would both duplicate the state dict and force the model to own a
    third weight set it must not have.
    """
    if stream == VLM:
        attention, mlp, norm_index = "vlm_attention", "vlm_mlp", 0
    elif stream == ACTION:
        attention, mlp, norm_index = "action_expert_attention", "action_expert_mlp", 1
    else:
        return None

    if tail.startswith("self_attn."):
        return f"model.layers.{index}.self_attn.{attention}.{tail[len('self_attn.'):]}"
    if tail.startswith("mlp."):
        return f"model.layers.{index}.mlp.{mlp}.{tail[len('mlp.'):]}"
    if tail == "input_layernorm.weight":
        return f"model.layers.{index}.input_layernorm.{norm_index}.weight"
    if tail == "post_attention_layernorm.weight":
        return f"model.layers.{index}.post_attention_layernorm.{norm_index}.weight"
    return None


def convert_state_dict(
    source: dict[str, torch.Tensor],
    *,
    vocab_size: int,
    max_action_dim: int,
    max_state_dim: int,
    map_proprio_to_state: bool,
    dtype: torch.dtype,
    state_proj_init_std: float = 0.0,
    proprio_layout: str = "raw",
    emit_proprio_stack: bool = False,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return ``(alias-free state dict, report)`` for a π0* ``model`` state dict."""
    out: dict[str, torch.Tensor] = {}
    report: dict[str, object] = {
        "source_tensors": len(source),
        "mapped": 0,
        "dropped_proprio_tensors": 0,
        "dropped_proprio_params": 0,
        "unmapped_tensors": [],
        "synthesized": {},
        "padding": {},
    }

    def put(target: str, tensor: torch.Tensor) -> None:
        if target in out:
            raise ValueError(f"duplicate target key {target!r}")
        out[target] = tensor
        report["mapped"] += 1

    # 1) token embedding (π0* has 257216 rows; FlashVLA/PaliGemma uses 257152)
    embed = source["embed_tokens.weight"]
    if embed.shape[0] < vocab_size:
        raise ValueError(f"checkpoint vocabulary {embed.shape[0]} < model {vocab_size}")
    cut = embed[:vocab_size].to(dtype)
    put(VLM_EMBED, _same(cut))
    # open-pi-zero ties the language head to the embedding: materialize it.
    # ``clone`` so the two entries do not share storage (safetensors rejects it).
    put(VLM_HEAD, _same(cut).clone())
    report["padding"]["embed_tokens"] = {
        "source": list(embed.shape),
        "target": list(cut.shape),
        "note": "truncated extra rows",
    }

    # 2) vision tower + projector
    for key, value in source.items():
        if key.startswith("vision_tower.") or key.startswith("multi_modal_projector."):
            put(f"model.vlm.model.{key}", _same(value.to(dtype)))

    # 3) decoder mixtures -> fused joint layers
    for key, value in source.items():
        # π0* carries a third (proprio) block whose weights are *tied* to the
        # action mixture (`pizero.py:274-276`): every tensor under
        # `mixtures.proprio.*` is a duplicate of one already emitted for
        # `mixtures.action.*`, so they are counted and dropped either way. The
        # only proprio-specific tensor is `proprio_encoder`, handled in step 5b
        # when `--emit-proprio-stack` is set.
        is_proprio = key.startswith("joint_model.mixtures.proprio.") or key.startswith(
            "proprio_encoder."
        )
        if is_proprio and not emit_proprio_stack:
            report["dropped_proprio_tensors"] += 1
            report["dropped_proprio_params"] += value.numel()
            continue
        if key.startswith("proprio_encoder."):
            continue  # handled in step 5b (strict 1:1, no padding)
        match = LAYER_RE.match(key)
        if match is None:
            continue
        stream, index, tail = match.group(1), int(match.group(2)), match.group(3)
        target = _layer_tail_target(stream, index, tail)
        if target is None:
            report["unmapped_tensors"].append(key)
            continue
        put(target, _same(value.to(dtype)))

    # 4) action-expert final norm
    put(EXPERT_NORM, _same(source[f"joint_model.mixtures.{ACTION}.norm.weight"].to(dtype)))

    # 5) suffix embedder: action encoder MLP == action_in_proj + time MLP
    action_in = source["action_encoder.linear_1.weight"]
    put(f"{ACTION_IN}.weight", _pad_cols(action_in.to(dtype), max_action_dim))
    put(f"{ACTION_IN}.bias", _same(source["action_encoder.linear_1.bias"].to(dtype)))
    put(f"{TIME_IN}.weight", _same(source["action_encoder.linear_2.weight"].to(dtype)))
    put(f"{TIME_IN}.bias", _same(source["action_encoder.linear_2.bias"].to(dtype)))
    put(f"{TIME_OUT}.weight", _same(source["action_encoder.linear_3.weight"].to(dtype)))
    put(f"{TIME_OUT}.bias", _same(source["action_encoder.linear_3.bias"].to(dtype)))
    report["padding"]["action_in_proj"] = {
        "source": list(action_in.shape),
        "target": [action_in.shape[0], max_action_dim],
        "note": "zero columns for action dimensions >= source width",
    }

    # 6) action head
    action_out = source["action_decoder.weight"]
    put(f"{ACTION_OUT}.weight", _pad_rows(action_out.to(dtype), max_action_dim))
    put(f"{ACTION_OUT}.bias", _pad_rows(source["action_decoder.bias"].to(dtype), max_action_dim))
    report["padding"]["action_out_proj"] = {
        "source": list(action_out.shape),
        "target": [max_action_dim, action_out.shape[1]],
        "note": "zero rows beyond the source action width",
    }

    # 5b) F6: the proprio stack's input encoder. π0*'s `proprio_encoder` is a
    #     single linear (`(1024, 8)` on fractal, `(1024, 7)` on bridge) that feeds
    #     the proprio block; `proprio_in_proj` is the same thing. Strict 1:1 --
    #     no padding -- so a width mismatch must fail loudly rather than silently
    #     truncate. The bridge 7-vs-8 gap is S4's problem.
    if emit_proprio_stack:
        p_in = source["proprio_encoder.weight"].to(dtype)
        p_bias = source["proprio_encoder.bias"].to(dtype)
        if proprio_layout == "flashvla_quat":
            p_in, p_bias = _align_proprio_encoder(p_in, p_bias)
        elif proprio_layout != "raw":
            raise ValueError(f"unknown proprio layout {proprio_layout!r}")
        put(f"{PROPRIO_IN}.weight", p_in)
        put(f"{PROPRIO_IN}.bias", p_bias)
        report["padding"]["proprio_in_proj"] = {
            "source": list(p_in.shape),
            "target": list(p_in.shape),
            "layout": proprio_layout,
            "note": "strict 1:1, no padding",
        }

    # 7) state projection: π0* feeds proprio through a whole extra stack, which
    #    FlashVLA pi0 does not have. Copying the encoder is only shape-compatible
    #    (7 proprio dims vs 8 quaternion state dims), so it is opt-in.
    if emit_proprio_stack and not map_proprio_to_state:
        # Two state paths would double-count the state token; the proprio stack
        # owns it. Keep `state_proj` present (rollback path) but inert.
        width = source["proprio_encoder.weight"].shape[0]
        out[f"{STATE_PROJ}.weight"] = torch.zeros(width, max_state_dim, dtype=dtype)
        out[f"{STATE_PROJ}.bias"] = torch.zeros(width, dtype=dtype)
        report["synthesized"][f"{STATE_PROJ}.weight"] = (
            "zeros (inert: use_proprio_stack=true routes the state through proprio_in_proj)"
        )
        report["synthesized"][f"{STATE_PROJ}.bias"] = "zeros (inert, see weight)"
    elif map_proprio_to_state:
        proprio = source["proprio_encoder.weight"].to(dtype)
        proprio_bias = source["proprio_encoder.bias"].to(dtype)
        if proprio_layout == "flashvla_quat":
            proprio, proprio_bias = _align_proprio_encoder(proprio, proprio_bias)
        elif proprio_layout != "raw":
            raise ValueError(f"unknown proprio layout {proprio_layout!r}")
        put(f"{STATE_PROJ}.weight", _pad_cols(proprio, max_state_dim))
        put(f"{STATE_PROJ}.bias", _same(proprio_bias))
        report["padding"]["state_proj"] = {
            "source": list(source["proprio_encoder.weight"].shape),
            "target": [proprio.shape[0], max_state_dim],
            "layout": proprio_layout,
            "note": (
                "columns permuted to FlashVLA state order "
                "[x,y,z,qw,qx,qy,qz,grip]; 1-grip folded into the bias"
                if proprio_layout == "flashvla_quat"
                else "raw π0* proprio order; semantics assume the layouts match"
            ),
        }
    else:
        width = source["proprio_encoder.weight"].shape[0]
        state_w = torch.zeros(width, max_state_dim, dtype=dtype)
        if state_proj_init_std > 0:
            # Zero init keeps the state branch invisible: measured on bridge it
            # only reached rms 1.7e-5 after 4000 steps, i.e. the model stayed
            # state-blind. A small normal init puts it on the time embedding's
            # scale from step 0 without drowning the action head.
            generator = torch.Generator().manual_seed(0)
            state_w = torch.normal(
                mean=0.0,
                std=float(state_proj_init_std),
                size=(width, max_state_dim),
                generator=generator,
            ).to(dtype)
        out[f"{STATE_PROJ}.weight"] = state_w
        out[f"{STATE_PROJ}.bias"] = torch.zeros(width, dtype=dtype)
        report["synthesized"][f"{STATE_PROJ}.weight"] = (
            f"N(0, {state_proj_init_std}) (state branch trained from scratch)"
            if state_proj_init_std > 0
            else "zeros (state branch trained from scratch)"
        )
        report["synthesized"][f"{STATE_PROJ}.bias"] = "zeros (state branch trained from scratch)"

    # 8) parameters π0* does not carry but every FlashVLA tensor must be covered
    expert_embed_width = out[EXPERT_NORM].shape[0]
    out[EXPERT_EMBED] = torch.zeros(vocab_size, expert_embed_width, dtype=dtype)
    out[EXPERT_HEAD] = torch.zeros(vocab_size, expert_embed_width, dtype=dtype)
    report["synthesized"][EXPERT_EMBED] = "zeros (unused: expert tokens come from the suffix embedder)"
    report["synthesized"][EXPERT_HEAD] = "zeros (frozen head; expert never decodes text)"

    vlm_width = out[VLM_EMBED].shape[1]
    out[VLM_NORM] = torch.ones(vlm_width, dtype=dtype)
    report["synthesized"][VLM_NORM] = "ones (π0* ships no VLM final norm; RMSNorm neutral)"

    report["output_tensors"] = len(out)
    return out, report


def load_source_state_dict(path: Path) -> dict[str, torch.Tensor]:
    # ``cpu`` + ``mmap`` keeps the 11.8 GB blob lazily paged; only the tensors
    # this script copies are touched. ``meta`` would carry no data at all.
    blob = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(blob, dict) or "model" not in blob:
        raise ValueError(f"{path} is not an open-pi-zero training blob (no 'model' entry)")
    state = blob["model"]
    if not isinstance(state, dict):
        raise ValueError(f"'model' in {path} is {type(state)}, expected a state dict")
    return state


def build_config(args: argparse.Namespace):
    """Build the `pi0-flashvla` config matching the target dataset."""
    from lerobot.configs.types import FeatureType, PolicyFeature

    from flashvla.policies.pi0.configuration_pi0 import PI0FlashVLAConfig

    info = json.loads((Path(args.data).expanduser() / "meta" / "info.json").read_text())
    images: dict[str, PolicyFeature] = {}
    state_shape = None
    action_shape = None
    for key, feature in info["features"].items():
        dtype = feature["dtype"]
        shape = feature.get("shape")
        if dtype == "video" or dtype == "image":
            height, width = shape[0], shape[1]
            images[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
        elif key == "observation.state":
            state_shape = tuple(shape)
        elif key == "action":
            action_shape = tuple(shape)
    if not images or state_shape is None or action_shape is None:
        raise ValueError(f"could not derive features from {args.data}/meta/info.json")

    cfg = PI0FlashVLAConfig()
    cfg.device = "meta"  # instantiation only; the written config restores "cuda"
    cfg.num_buffer_slots = args.num_buffer_slots
    cfg.chunk_size = args.chunk_size
    cfg.n_action_steps = args.n_action_steps
    cfg.empty_cameras = args.empty_cameras
    cfg.use_adarms_time_cond = bool(args.use_adarms_time_cond)
    # The meta-device instance must mirror the target architecture, otherwise
    # every emitted proprio tensor shows up as an unknown key.
    cfg.use_proprio_stack = bool(args.emit_proprio_stack)
    cfg.proprio_dim = int(args.proprio_dim)
    # F6/S4: record the *data-side* layout the encoder expects, so a bare reload of
    # this directory builds the same preprocessor. The converter never rewrites the
    # state here -- `bridge_euler7` is applied by `ProprioLayoutProcessorStep`.
    cfg.state_layout = str(args.state_layout)
    cfg.state_stats_path = str(args.state_stats_path) if args.state_stats_path else None
    cfg.input_features = {
        **images,
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape)}
    cfg.validate_features()
    return cfg, info


def verify_against_model(mapped: dict[str, torch.Tensor], cfg) -> dict:
    """Shape/coverage check against a meta-device `pi0-flashvla` instance."""
    import torch as _torch

    from flashvla.policies.loading import find_unfilled_parameters
    from flashvla.policies.pi0.modeling_pi0_flashvla import PI0FlashVLAPolicy

    with _torch.device("meta"):
        policy = PI0FlashVLAPolicy(cfg)
    target = policy.state_dict()

    unknown = sorted(set(mapped) - set(target))
    bad_shape = [
        f"{key}: file {tuple(mapped[key].shape)} vs model {tuple(target[key].shape)}"
        for key in sorted(set(mapped) & set(target))
        if tuple(mapped[key].shape) != tuple(target[key].shape)
    ]
    unfilled = find_unfilled_parameters(policy, mapped)
    return {
        "ok": not unknown and not bad_shape and not unfilled,
        "unknown_keys": unknown[:10],
        "shape_mismatches": bad_shape[:10],
        "unfilled_parameters": unfilled[:10],
        "unfilled_count": len(unfilled),
        "model_tensors": len(target),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="π0* .pt blob")
    parser.add_argument("--data", type=Path, required=True, help="LeRobot dataset dir whose meta/info.json drives the features")
    parser.add_argument("--dst", type=Path, default=None, help="output dir (required with --write)")
    parser.add_argument("--write", action="store_true", help="materialize model.safetensors + config.json")
    parser.add_argument("--json", type=Path, default=None, help="write the conversion report here")
    parser.add_argument("--num-buffer-slots", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--empty-cameras", type=int, default=2)
    parser.add_argument("--use-adarms-time-cond", action="store_true")
    parser.add_argument(
        "--map-proprio-to-state",
        action="store_true",
        help="copy π0* proprio_encoder into state_proj (shape-compatible, semantics differ)",
    )
    parser.add_argument(
        "--state-proj-init-std",
        type=float,
        default=0.0,
        help="std of the N(0, std) init for the synthesized state_proj (0 = zeros)",
    )
    parser.add_argument(
        "--proprio-layout",
        choices=("raw", "flashvla_quat"),
        default="raw",
        help=(
            "how to reconcile π0* proprio with our 8-D state when "
            "--map-proprio-to-state / --emit-proprio-stack is set: 'raw' copies "
            "the encoder verbatim (only valid if the layouts already match); "
            "'flashvla_quat' permutes "
            "[x,y,z,qx,qy,qz,qw,1-grip] -> [x,y,z,qw,qx,qy,qz,grip]"
        ),
    )
    parser.add_argument(
        "--emit-proprio-stack",
        action="store_true",
        help=(
            "F6: enable π0*'s proprio pathway. Emits `proprio_in_proj` from "
            "`proprio_encoder` and zeroes `state_proj` (the proprio token owns "
            "the state). The proprio block itself reuses the action expert's "
            "weights -- π0* ties the two mixtures (pizero.py:274-276) -- so no "
            "decoder tensor is added. Requires use_proprio_stack=true."
        ),
    )
    parser.add_argument(
        "--proprio-dim",
        type=int,
        default=0,
        help="width of proprio_in_proj input (0 = the dataset's observation.state dim)",
    )
    parser.add_argument(
        "--state-layout",
        default="raw",
        choices=("raw", "bridge_euler7"),
        help=(
            "F6/S4: layout `observation.state` is fed to the proprio encoder in. "
            "`bridge_euler7` records that the preprocessor must convert the 8-D "
            "quaternion state to π0*'s 7-D [x,y,z,rpy,grip]; pair it with "
            "--proprio-dim 7 --proprio-layout raw."
        ),
    )
    parser.add_argument(
        "--state-stats-path",
        type=Path,
        default=None,
        help=(
            "F6/S4: 7-D q01/q99 stat JSON for the converted state "
            "(scripts/p4/f6_s4_bridge_proprio.py); written into config.json so the "
            "directory is self-describing."
        ),
    )
    args = parser.parse_args()

    # S4: the bridge encoder is (1024, 7) and consumes already-converted angles, so
    # the weight-space permutation (`flashvla_quat`) must not be applied on top.
    if args.state_layout == "bridge_euler7":
        if args.proprio_dim != 7:
            parser.error("--state-layout bridge_euler7 needs --proprio-dim 7")
        if args.proprio_layout != "raw":
            parser.error(
                "--state-layout bridge_euler7 must pair with --proprio-layout raw: "
                "the 8->7 conversion happens in the preprocessor, not in the weights"
            )
        if not args.emit_proprio_stack:
            parser.error("--state-layout bridge_euler7 needs --emit-proprio-stack")

    from safetensors.torch import save_file

    cfg, info = build_config(args)
    source = load_source_state_dict(args.src)
    dtype = torch.bfloat16 if str(getattr(cfg, "dtype", "bfloat16")) == "bfloat16" else torch.float32

    mapped, report = convert_state_dict(
        source,
        vocab_size=cfg.vlm_config.text_config.vocab_size,
        max_action_dim=cfg.max_action_dim,
        max_state_dim=cfg.max_state_dim,
        map_proprio_to_state=args.map_proprio_to_state,
        dtype=dtype,
        state_proj_init_std=float(args.state_proj_init_std),
        proprio_layout=args.proprio_layout,
        emit_proprio_stack=bool(args.emit_proprio_stack),
    )
    report.update(
        {
            "src": str(args.src),
            "data": str(args.data),
            "dataset": {"fps": info.get("fps"), "frames": info.get("total_frames"), "episodes": info.get("total_episodes")},
            "config": {
                "num_buffer_slots": cfg.num_buffer_slots,
                "chunk_size": cfg.chunk_size,
                "n_action_steps": cfg.n_action_steps,
                "empty_cameras": cfg.empty_cameras,
                "use_adarms_time_cond": cfg.use_adarms_time_cond,
            },
            "verify": verify_against_model(mapped, cfg),
        }
    )

    print(json.dumps({k: v for k, v in report.items() if k != "synthesized"}, indent=2, ensure_ascii=False))
    print("\nsynthesized tensors:", len(report["synthesized"]))
    for key, why in report["synthesized"].items():
        print(f"  {key}: {why}")

    if not report["verify"]["ok"]:
        print("\nVERIFY FAILED — nothing written", file=sys.stderr)
        return 1

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nreport -> {args.json}")

    if not args.write:
        print("\ndry run: weights not written (pass --write --dst <dir>)")
        return 0

    from flashvla.policies.pi0.configuration_pi0 import PI0FlashVLAConfig  # noqa: F401  (registry)

    assert args.dst is not None
    args.dst.mkdir(parents=True, exist_ok=True)
    save_file(mapped, str(args.dst / "model.safetensors"), metadata={"format": "pt"})
    cfg.device = "cuda"
    cfg.save_pretrained(args.dst)
    print(f"\nwrote {args.dst}/model.safetensors ({len(mapped)} tensors) and config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
