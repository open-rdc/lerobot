#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_smolvla import SmolVLAConfig
from .paraphrase_processor import ParaphraseAugmentProcessorStep
from .waypoint_action_processor import WaypointRebaseProcessorStep, WaypointUnscaleProcessorStep


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Adding a batch dimension.
    3.  Optionally swapping the task description for a random paraphrase (training only;
        no-op unless config.paraphrase_augment_path is set).
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.
    7.  Normalizing input and output features based on dataset statistics.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # 各waypointが現在姿勢を共通原点とした絶対オフセット([x,y,cos,sin]->[dx,dy,hx,hy])を
    # 使うaction_dim=4のデータセットでのみ有効化する。既存のaction_dim=2チェックポイントは
    # 自身のシリアライズ済みprocessor configをロードするため影響を受けない。
    use_waypoint_actions = config.action_feature is not None and config.action_feature.shape[0] == 4
    waypoint_rebase = WaypointRebaseProcessorStep(enabled=use_waypoint_actions)
    waypoint_unscale = WaypointUnscaleProcessorStep(
        enabled=use_waypoint_actions, pos_scale=waypoint_rebase.pos_scale
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # 学習時: task文字列をパラフレーズ集合からランダムに差し替える(map_path未設定ならno-op)。
        # NewLineTask/Tokenizerより前に置き、差し替え後の文字列がトークナイズされるようにする。
        ParaphraseAugmentProcessorStep(map_path=config.paraphrase_augment_path),
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        waypoint_rebase,  # 学習時: [x,y,cos,sin]の絶対姿勢chunkをt=0原点のwaypointに変換
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        waypoint_unscale,  # pos_scaleの除算を戻す(回転/再原点化はモデル出力がそのまま欲しい形なので戻さない)
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
