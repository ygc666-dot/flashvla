#!/usr/bin/env python
# Copyright 2025 FlashVLA team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from flashvla.policies.pi0.configuration_pi0 import PI0Config
from flashvla.policies.pi0.utils import bridge_proprio_from_state


@ProcessorStepRegistry.register(name="flashvla_pi0_prepare_language_processor_step")
@dataclass
class FlashVLAPi0PrepareLanguageProcessorStep(ProcessorStep):
    """Prepare language prompt for PI0 policies.

    PI0 conditions the action expert on state via a dedicated suffix token,
    so the language prompt is just the task description with a trailing
    newline (the format lerobot/pi0_base was trained with).
    """

    task_key: str = "task"

    def get_config(self) -> dict[str, Any]:
        # Same latent trap as the pi05 step: the base returns {} and a customized
        # `task_key` would silently reload as "task".
        return {"task_key": self.task_key}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        if isinstance(tasks, str):
            tasks = [tasks]

        full_prompts = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


# Number of state dimensions each supported `state_layout` produces. `raw` is the
# datasets' own `observation.state`, everything else is derived from it.
STATE_LAYOUT_DIMS: dict[str, int] = {
    "raw": 8,
    "bridge_euler7": 7,
}


def state_stats_from_json(path: str | Path, layout: str) -> dict[str, torch.Tensor]:
    """Load a 7-D state stat set in the shape `NormalizerProcessorStep` expects.

    The normalizer reads ``stats[key]["q01"/"q99"/...]`` as tensors, so the JSON
    keys are passed through verbatim (``scripts/p4/f6_s4_bridge_proprio.py``
    writes ``q01``/``q99`` either from our own dataset percentiles or from π0*'s
    published ``p01``/``p99``).
    """
    expected = STATE_LAYOUT_DIMS.get(layout)
    payload = json.loads(Path(path).read_text())
    stats: dict[str, torch.Tensor] = {}
    for name, values in payload.items():
        if name.startswith("_"):
            continue
        tensor = torch.tensor(values, dtype=torch.float32)
        if expected is not None and tensor.shape[0] != expected:
            raise ValueError(
                f"state stats {path!r} has {tensor.shape[0]} dims for {name!r}, "
                f"layout {layout!r} needs {expected}"
            )
        stats[name] = tensor
    if "q01" not in stats or "q99" not in stats:
        raise ValueError(f"state stats {path!r} must carry q01/q99")
    return stats


@ProcessorStepRegistry.register(name="flashvla_pi0_proprio_layout_processor_step")
@dataclass
class ProprioLayoutProcessorStep(ProcessorStep):
    """Rewrite `observation.state` into π0*'s proprio layout (F6/S4).

    Only the bridge encoder needs this: its `proprio_encoder` is ``(1024, 7)`` and
    consumes ``[x, y, z, roll, pitch, yaw, grip]`` (radians, bridge's top-down EEF
    frame), while our datasets store the raw 8-D ``eef_pos``
    ``[x, y, z, qw, qx, qy, qz, grip]``. The conversion is non-linear, so it has
    to run **before** normalization -- that is why the pipeline places this step
    ahead of `NormalizerProcessorStep` and swaps in a 7-D stat set.

    Fractal does not use it: its 8-D mismatch is a pure permutation + gripper sign
    flip, which `convert_pi0star.py --proprio-layout flashvla_quat` folds into the
    encoder's weights instead of touching the data.
    """

    layout: str = "bridge_euler7"
    state_key: str = OBS_STATE

    def get_config(self) -> dict[str, Any]:
        # Must be complete: the base returns {} and a reloaded pipeline would
        # silently fall back to the default layout.
        return {"layout": self.layout, "state_key": self.state_key}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        state = observation.get(self.state_key)
        if state is None:
            raise ValueError(f"no {self.state_key!r} in the observation")
        if not isinstance(state, torch.Tensor):
            raise TypeError(
                f"{self.state_key!r} must be a tensor by now (AddBatchDimensionProcessorStep "
                f"runs first), got {type(state)}"
            )

        observation[self.state_key] = self.convert(state)
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def convert(self, state: torch.Tensor) -> torch.Tensor:
        if self.layout == "bridge_euler7":
            return bridge_proprio_from_state(state)
        if self.layout == "raw":
            return state
        raise ValueError(f"unknown proprio layout {self.layout!r}")

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        dim = STATE_LAYOUT_DIMS.get(self.layout)
        if dim is None:
            return features
        out = {kind: dict(entries) for kind, entries in features.items()}
        for entries in out.values():
            if self.state_key in entries:
                entries[self.state_key] = PolicyFeature(type=FeatureType.STATE, shape=(dim,))
        return out


def make_flashvla_pi0_pre_post_processors(
    config: PI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre/post-processor pipelines for FlashVLA PI0 policy.

    Preprocessor pipeline:
    1. Rename observations (identity by default)
    2. Add batch dimension
    3. Normalize inputs and outputs using dataset statistics
    4. Prepare language prompt (append newline)
    5. Tokenize prompt with PaliGemma tokenizer
    6. Move to device

    Postprocessor pipeline:
    1. Unnormalize outputs
    2. Move to CPU
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
    ]

    # F6/S4: a non-raw state layout has to be applied *before* normalization (the
    # quaternion -> euler map is non-linear) and needs its own stat set, because
    # the dataset's `observation.state` stats describe the 8-D raw layout.
    state_layout = str(getattr(config, "state_layout", "raw") or "raw")
    if state_layout != "raw":
        input_steps.append(ProprioLayoutProcessorStep(layout=state_layout))
        stats_path = getattr(config, "state_stats_path", None)
        if not stats_path:
            raise ValueError(
                f"state_layout={state_layout!r} needs `state_stats_path` (the {STATE_LAYOUT_DIMS[state_layout]}-D "
                "q01/q99 stat JSON produced by scripts/p4/f6_s4_bridge_proprio.py)"
            )
        state_stats = state_stats_from_json(stats_path, state_layout)
        dataset_stats = {**(dataset_stats or {}), OBS_STATE: state_stats}

    input_steps += [
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        FlashVLAPi0PrepareLanguageProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
