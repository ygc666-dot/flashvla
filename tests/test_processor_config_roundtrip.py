"""A processor step must serialize enough config to reload itself.

`ProcessorStep.get_config()` returns `{}`, so a dataclass step that forgets to
override it is written to `policy_preprocessor.json` as `"config": {}` and
silently reloads with its *field defaults*. For
`FlashVLAPi05PrepareLanguageProcessorStep` that flipped `state_cond=True` to
`False`, which rewrote the prompt from the trained

    "Task: {task};\\nAction: "                    (12 tokens)

to

    "Task: {task}, State: 130 131 248 ...;\\nAction: "   (46 tokens)

On the bridge checkpoint that cost 0.31 vs 0.16 training loss and 0/10 vs 5/10
on SIMPLER `widowx_stack_cube` (see
`docs/sdn_flashvla/flashvla_simpler/3_5_p3_execution_report.md` §11.2).

The invariant tested here is deliberately generic: for every `ProcessorStep`
authored in this package, `type(step)(**step.get_config()) == step` when the
step is constructed with non-default field values. That guards the whole class
of bug rather than just the one instance of it.
"""
from __future__ import annotations

import dataclasses
import importlib
import unittest

import numpy as np

import torch

from lerobot.processor.pipeline import ProcessorStep

from flashvla.policies.pi05.processor import FlashVLAPi05PrepareLanguageProcessorStep

STEP_MODULES = (
    "flashvla.policies.pi05.processor",
    "flashvla.policies.pi0.processor",
    "flashvla.policies.smolvla.processor_smolvla",
    "flashvla.policies.lingbot.processor",
)


def _alternative(field: dataclasses.Field):
    """A value that differs from the field default, or MISSING if unsupported."""
    default = field.default
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 7
    if isinstance(default, float):
        return default + 0.5
    if isinstance(default, str):
        return default + "_x"
    return dataclasses.MISSING


def _flashvla_step_classes():
    for module_name in STEP_MODULES:
        module = importlib.import_module(module_name)
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, ProcessorStep)
                and obj.__module__ == module_name
            ):
                yield module_name, obj


class ProcessorStepConfigRoundTripTest(unittest.TestCase):
    def test_non_default_config_is_preserved(self):
        checked = 0
        for module_name, cls in _flashvla_step_classes():
            if not dataclasses.is_dataclass(cls):
                continue  # no configurable fields -> nothing can be lost
            overrides = {}
            for field in dataclasses.fields(cls):
                value = _alternative(field)
                if value is dataclasses.MISSING:
                    overrides = None  # unsupported field type, skip this class
                    break
                overrides[field.name] = value
            if not overrides:
                continue

            with self.subTest(f"{module_name}.{cls.__name__}"):
                step = cls(**overrides)
                config = step.get_config()
                self.assertTrue(
                    config,
                    f"{cls.__name__}.get_config() returned {{}} but the class has "
                    f"fields {sorted(overrides)}; they will be lost on reload",
                )
                self.assertEqual(
                    cls(**config),
                    step,
                    f"{cls.__name__} does not round-trip through get_config()",
                )
                checked += 1

        self.assertGreater(checked, 0, "no flashvla processor step was checked")

    def test_language_step_keeps_state_cond(self):
        """The concrete regression: `state_cond` must survive serialization."""
        step = FlashVLAPi05PrepareLanguageProcessorStep(
            state_cond=True, max_state_dim=16, task_key="instruction"
        )
        self.assertEqual(
            step.get_config(),
            {"state_cond": True, "max_state_dim": 16, "task_key": "instruction"},
        )

    def test_prompt_differs_with_state_cond(self):
        """Documents *why* losing `state_cond` is fatal, not cosmetic."""
        tasks = ["put carrot on plate"]
        state = torch.zeros(1, 8)

        def prompt(step):
            transition = {
                "observation": {"observation.state": state},
                "complementary_data": {"task": tasks},
            }
            out = step(transition)
            return out["complementary_data"]["task"][0]

        free = prompt(FlashVLAPi05PrepareLanguageProcessorStep(state_cond=True))
        in_prompt = prompt(FlashVLAPi05PrepareLanguageProcessorStep(state_cond=False))

        # A zero state lands in the middle of the 256 uniform [-1, 1) bins.
        digit = str(int(np.digitize(np.zeros(8), bins=np.linspace(-1, 1, 256 + 1)[:-1])[0]) - 1)

        self.assertEqual(free, "Task: put carrot on plate;\nAction: ")
        self.assertEqual(
            in_prompt,
            "Task: put carrot on plate, State: " + " ".join([digit] * 8) + ";\nAction: ",
        )
        self.assertNotEqual(free, in_prompt)


if __name__ == "__main__":
    unittest.main()
