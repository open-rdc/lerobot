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

import json
import random
from dataclasses import dataclass, field

from lerobot.processor import ComplementaryDataProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("smolvla_paraphrase_augment_processor")
@dataclass
class ParaphraseAugmentProcessorStep(ComplementaryDataProcessorStep):
    """Randomly swaps a dataset's fixed per-frame instruction text for one of a
    precomputed set of paraphrases (same landmark/direction words, different wording).

    ``map_path`` points to a JSON file of ``{original_task: [paraphrase, ...]}``. Whichever
    string ``complementary_data["task"]`` holds is looked up verbatim; a uniformly random
    pick from ``[original, *paraphrases]`` replaces it. A sampler visits each dataset frame
    once per epoch, so re-rolling on every call means a frame gets a different phrasing on
    different epochs rather than being pinned to one string for the whole run.

    Any task string not present in the map (unseen at inference, a dataset that never went
    through this map, or ``map_path`` unset/missing/unreadable) passes through unchanged --
    this makes the step a safe no-op by default, including at deployment.

    Must run upstream of TokenizerProcessorStep so the swap lands before tokenization.
    """

    map_path: str | None = None
    enabled: bool = True
    seed: int | None = None
    _map: dict = field(default=None, init=False, repr=False, compare=False)
    _rng: random.Random = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._map = {}
        if self.enabled and self.map_path:
            try:
                with open(self.map_path, encoding="utf-8") as f:
                    self._map = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._map = {}

    def _swap(self, task: str) -> str:
        variants = self._map.get(task)
        if not variants:
            return task
        return self._rng.choice([task, *variants])

    def complementary_data(self, complementary_data: dict) -> dict:
        if not self._map or "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)
        if isinstance(task, str):
            new_complementary_data["task"] = self._swap(task)
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            new_complementary_data["task"] = [self._swap(t) for t in task]
        return new_complementary_data

    def get_config(self) -> dict:
        return {"map_path": self.map_path, "enabled": self.enabled, "seed": self.seed}

    def transform_features(self, features):
        return features
