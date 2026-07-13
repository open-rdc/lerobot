# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey

# Physical scale (meters) used to bring the rebased [dx, dy] waypoint offsets into
# roughly unit-order magnitude before they reach the model. Chosen from the measured
# single-step advance in the old [dx_body, dyaw] action (mean ~0.195 m / 0.2 s tick).
# NOTE: this is a fixed constant, not a learned/dataset statistic -- see the docstring
# of WaypointRebaseProcessorStep for why a pooled mean/std would be the wrong tool here.
WAYPOINT_POS_SCALE = 0.2


@ProcessorStepRegistry.register("smolvla_waypoint_rebase_processor")
@dataclass
class WaypointRebaseProcessorStep(ProcessorStep):
    """Rebases a chunk of absolute poses onto a single shared origin.

    The dataset stores each frame's own absolute pose ``[x, y, cos(yaw), sin(yaw)]``.
    lerobot's stock chunk assembly (``action_delta_indices``) fetches ``chunk_size``
    consecutive frames' own stored action verbatim -- entry ``k`` is frame ``t+k``'s
    own absolute pose, not "waypoint k as seen from frame t". This step converts the
    chunk into NavVLA-style waypoints that all share ONE origin: frame ``t`` (chunk
    offset 0, which ``action_delta_indices`` always includes). Reconstructing a path
    from the result needs no integration/chaining -- each entry is already relative
    to "now".

    Also divides ``dx, dy`` by a fixed physical scale (``pos_scale``). This step sits
    upstream of ``NormalizerProcessorStep``, but ``normalization_mapping["ACTION"]``
    is IDENTITY for this policy (see ``configuration_smolvla.py``) -- the dataset's
    automatically-computed action mean/std reflects the *raw stored absolute pose*
    (unbounded, can span many meters across episodes), which would be meaningless
    applied to the small rebased offsets this step produces. A single pooled mean/std
    would also be the wrong statistical model here regardless: waypoint magnitude
    grows systematically with chunk offset k (k=1 is close, k=49 is far), so a fixed
    constant (mirroring NavVLA's own ``metric_waypoint_spacing`` convention) is used
    instead of a per-channel learned statistic.

    No-ops whenever there is no ground-truth action in the transition, which is
    always the case at inference (a live observation batch has no "action" key) --
    so this step only ever fires during training.
    """

    enabled: bool = True
    pos_scale: float = WAYPOINT_POS_SCALE

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        origin = action[..., 0:1, :]  # (..., 1, 4) -- frame t's own absolute pose
        x0, y0, c0, s0 = origin.unbind(-1)
        x, y, c, s = action.unbind(-1)

        dxg, dyg = x - x0, y - y0
        # Same rotation convention as the old to_body_frame() helper
        # (lerobot_dataset.py): rotmat = [[c, -s], [s, c]], row-vector .dot(rotmat).
        dx = c0 * dxg + s0 * dyg
        dy = -s0 * dxg + c0 * dyg
        cos_d = c0 * c + s0 * s  # cos(theta_k - theta_0)
        sin_d = c0 * s - s0 * c  # sin(theta_k - theta_0)

        new_action = torch.stack(
            [dx / self.pos_scale, dy / self.pos_scale, cos_d, sin_d], dim=-1
        )
        new_transition[TransitionKey.ACTION] = new_action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "pos_scale": self.pos_scale}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("smolvla_waypoint_unscale_processor")
@dataclass
class WaypointUnscaleProcessorStep(ProcessorStep):
    """Undoes WaypointRebaseProcessorStep's ``pos_scale`` division on model output.

    Does NOT undo the rotation/rebasing -- the model's predicted chunk is already
    exactly the representation the navigation controller wants (waypoints relative
    to "now"), so there is nothing else to invert. Unlike ``AbsoluteActionsProcessorStep``,
    no cached origin/state is needed: the origin at inference time is implicitly
    "wherever the robot physically is right now", which software never actuates.

    Applying this step is required, not optional: without it, every predicted
    distance would be off by a constant factor of ``pos_scale`` (silent, no crash).
    """

    enabled: bool = True
    pos_scale: float = WAYPOINT_POS_SCALE

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        action = action.clone()
        action[..., 0] *= self.pos_scale
        action[..., 1] *= self.pos_scale
        new_transition[TransitionKey.ACTION] = action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "pos_scale": self.pos_scale}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
