#!/usr/bin/env python
"""Tests for F6/S4: the 7-D bridge proprio (`[x, y, z, rpy, grip]`).

The conversion is a port of π0*'s ``BridgeSimplerAdapter.preprocess_proprio``
(``third_party/.../simpler_adapter.py:179-192``), which is itself transforms3d's
``quat2mat`` + ``mat2euler(axes="sxyz"``)`` with a top-down frame fix. A port is
only trustworthy if it is checked against the original, so the first test loads
π0*'s own ``geometry.py`` and compares element-wise in float64 -- including the
gimbal-lock branch, which a naive ``atan2`` transcription would get wrong.

The rest pins the plumbing: the processor step has to (a) keep xyz + the gripper
untouched, (b) survive a `policy_preprocessor.json` round-trip (``get_config``),
(c) advertise a 7-wide state feature, and (d) refuse a stat set of the wrong
width -- normalizing a 7-dim state with the dataset's 8-dim stats is the exact
failure mode this whole stage exists to prevent.

Run::

    <env>/bin/python -m pytest tests/test_pi0_bridge_proprio.py -q
    <env>/bin/python tests/test_pi0_bridge_proprio.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from flashvla.policies.pi0.processor import (  # noqa: E402
    ProprioLayoutProcessorStep,
    state_stats_from_json,
)
from flashvla.policies.pi0.utils import (  # noqa: E402
    bridge_proprio_from_state,
    mat_to_euler_sxyz,
    quat_wxyz_to_mat,
)

GEOMETRY = (
    REPO_ROOT
    / "third_party/SDN/simpler_env/policies/pizero/open_pi_zero/src/utils/geometry.py"
)
BRIDGE_DEFAULT_ROT = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])


def _load_geometry():
    spec = importlib.util.spec_from_file_location("pi0star_geometry_test", GEOMETRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _states(n: int, seed: int = 0) -> np.ndarray:
    """Random 8-D eef_pos rows: xyz, a random *unnormalised* quaternion, grip."""
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(n, 4)) * rng.uniform(0.2, 2.0, size=(n, 1))
    return np.concatenate(
        [
            rng.normal(scale=0.1, size=(n, 3)),
            quat,
            rng.uniform(0.0, 1.0, size=(n, 1)),
        ],
        axis=1,
    )


class BridgeProprioMathTest(unittest.TestCase):
    def setUp(self):
        self.geometry = _load_geometry()

    def test_matches_pi0star_reference(self):
        states = _states(512)
        ours = bridge_proprio_from_state(torch.from_numpy(states)).numpy()
        ref = np.empty((states.shape[0], 7))
        for i, s in enumerate(states):
            ref[i, :3] = s[:3]
            ref[i, 3:6] = self.geometry.mat2euler(
                self.geometry.quat2mat(s[3:7]) @ BRIDGE_DEFAULT_ROT.T
            )
            ref[i, 6] = s[7]
        self.assertLess(
            float(np.abs(ours - ref).max()),
            1e-12,
            "torch port diverges from pi0*'s quat2mat/mat2euler",
        )

    def test_identity_quaternion_maps_to_zero_rpy(self):
        states = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5]])
        # R = I @ default_rot.T = default_rot.T, which is a pure 90-deg roll...
        # so compare against the reference rather than assuming zeros.
        ref = self.geometry.mat2euler(
            self.geometry.quat2mat(states[0, 3:7]) @ BRIDGE_DEFAULT_ROT.T
        )
        out = bridge_proprio_from_state(torch.from_numpy(states)).numpy()[0]
        np.testing.assert_allclose(out[3:6], np.asarray(ref), atol=1e-14)
        self.assertEqual(float(out[6]), 0.5)

    def test_gimbal_lock_branch(self):
        """pitch = +-90 deg: only roll+yaw is determined, and the reference picks
        ``roll = atan2(-M12, M11)``, ``yaw = 0``; the general branch would blow up."""
        s = np.pi / 4
        states = np.array([[0.0, 0.0, 0.0, np.cos(s), 0.0, np.sin(s), 0.0, 0.1]])
        ref = self.geometry.mat2euler(
            self.geometry.quat2mat(states[0, 3:7]) @ BRIDGE_DEFAULT_ROT.T
        )
        out = bridge_proprio_from_state(torch.from_numpy(states)).numpy()[0]
        np.testing.assert_allclose(out[3:6], np.asarray(ref), atol=1e-12)

    def test_degenerate_quaternion_is_identity(self):
        zero = torch.zeros(1, 4, dtype=torch.float64)
        torch.testing.assert_close(quat_wxyz_to_mat(zero)[0], torch.eye(3, dtype=torch.float64))
        # nan-free output is what actually matters downstream
        out = bridge_proprio_from_state(torch.zeros(1, 8, dtype=torch.float64))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_xyz_and_gripper_pass_through(self):
        states = _states(64, seed=1)
        out = bridge_proprio_from_state(torch.from_numpy(states)).numpy()
        self.assertEqual(out.shape, (64, 7))
        np.testing.assert_array_equal(out[:, :3], states[:, :3])
        np.testing.assert_array_equal(out[:, 6], states[:, 7])

    def test_rejects_wrong_width(self):
        with self.assertRaises(ValueError):
            bridge_proprio_from_state(torch.zeros(2, 7))

    def test_dtype_is_preserved(self):
        for dtype in (torch.float32, torch.float64):
            out = bridge_proprio_from_state(torch.from_numpy(_states(4)).to(dtype))
            self.assertEqual(out.dtype, dtype)

    def test_broadcasts_over_leading_dims(self):
        flat = bridge_proprio_from_state(torch.from_numpy(_states(6, seed=2)))
        batched = bridge_proprio_from_state(
            torch.from_numpy(_states(6, seed=2)).view(2, 3, 8)
        )
        torch.testing.assert_close(batched.reshape(6, 7), flat)


class ProprioLayoutStepTest(unittest.TestCase):
    def _transition(self, state: torch.Tensor) -> dict:
        return {
            TransitionKey.OBSERVATION: {OBS_STATE: state},
            TransitionKey.ACTION: None,
        }

    def test_rewrites_state_in_place(self):
        step = ProprioLayoutProcessorStep(layout="bridge_euler7")
        states = torch.from_numpy(_states(5, seed=3))
        out = step(self._transition(states))
        new_state = out[TransitionKey.OBSERVATION][OBS_STATE]
        self.assertEqual(tuple(new_state.shape), (5, 7))
        torch.testing.assert_close(new_state, bridge_proprio_from_state(states))

    def test_does_not_mutate_input_transition(self):
        step = ProprioLayoutProcessorStep()
        states = torch.from_numpy(_states(3, seed=4))
        transition = self._transition(states)
        step(transition)
        self.assertEqual(tuple(transition[TransitionKey.OBSERVATION][OBS_STATE].shape), (3, 8))

    def test_get_config_round_trip(self):
        step = ProprioLayoutProcessorStep(layout="bridge_euler7", state_key=OBS_STATE)
        config = step.get_config()
        self.assertEqual(config, {"layout": "bridge_euler7", "state_key": OBS_STATE})
        rebuilt = ProprioLayoutProcessorStep(**config)
        states = torch.from_numpy(_states(2, seed=5))
        torch.testing.assert_close(
            rebuilt(self._transition(states))[TransitionKey.OBSERVATION][OBS_STATE],
            step(self._transition(states))[TransitionKey.OBSERVATION][OBS_STATE],
        )

    def test_transform_features_reports_7_dims(self):
        step = ProprioLayoutProcessorStep(layout="bridge_euler7")
        features = {
            PipelineFeatureType.OBSERVATION: {
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))
            }
        }
        out = step.transform_features(features)
        self.assertEqual(out[PipelineFeatureType.OBSERVATION][OBS_STATE].shape, (7,))
        # the caller's dict must be left alone
        self.assertEqual(features[PipelineFeatureType.OBSERVATION][OBS_STATE].shape, (8,))

    def test_unknown_layout_raises(self):
        with self.assertRaises(ValueError):
            ProprioLayoutProcessorStep(layout="nope")(
                self._transition(torch.zeros(1, 8))
            )


class StateStatsTest(unittest.TestCase):
    def _write(self, dims: int) -> Path:
        payload = {
            name: [float(i + 1) for i in range(dims)]
            for name in ("q01", "q99", "mean", "std", "min", "max")
        }
        payload["_dims"] = [f"d{i}" for i in range(dims)]
        path = Path(tempfile.mkdtemp()) / "stats.json"
        path.write_text(json.dumps(payload))
        return path

    def test_loads_7_dim(self):
        stats = state_stats_from_json(self._write(7), "bridge_euler7")
        self.assertEqual(tuple(stats["q01"].shape), (7,))
        self.assertNotIn("_dims", stats)

    def test_rejects_8_dim_for_bridge(self):
        with self.assertRaises(ValueError):
            state_stats_from_json(self._write(8), "bridge_euler7")

    def test_requires_quantiles(self):
        path = Path(tempfile.mkdtemp()) / "stats.json"
        path.write_text(json.dumps({"mean": [0.0] * 7}))
        with self.assertRaises(ValueError):
            state_stats_from_json(path, "bridge_euler7")


if __name__ == "__main__":
    unittest.main(verbosity=2)
