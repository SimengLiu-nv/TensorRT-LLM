# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Which speculative configs may run with the mooncake-store-m3 connector.

A disaggregated prefill server with a one-model drafter (MTP or EAGLE3) only
serves context requests, so the connector stays valid there. Every other
combination, including every other connector, must keep raising at executor
creation.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    _connector_runs_speculative_prefill_only,
)
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    Eagle3DecodingConfig,
    KvCacheConnectorConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
)

MOONCAKE = KvCacheConnectorConfig(connector="mooncake-store-m3")
DISAGG = CacheTransceiverConfig(backend="DEFAULT")


def eagle3() -> Eagle3DecodingConfig:
    # One-model is the only EAGLE3 mode on this branch (two-model speculation was retired), so the config has no
    # eagle3_one_model field.
    return Eagle3DecodingConfig(max_draft_len=3, speculative_model="draft")


@pytest.mark.parametrize(
    "spec_config",
    [MTPDecodingConfig(num_nextn_predict_layers=1), eagle3()],
    ids=["mtp", "eagle3_one_model"],
)
def test_one_model_drafter_runs_prefill_only_on_disagg_mooncake(spec_config):
    assert _connector_runs_speculative_prefill_only(spec_config, MOONCAKE, DISAGG)


@pytest.mark.parametrize(
    "spec_config, connector, transceiver",
    [
        (None, MOONCAKE, DISAGG),
        (NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2), MOONCAKE, DISAGG),
        (eagle3(), KvCacheConnectorConfig(connector="kvbm"), DISAGG),
        (eagle3(), KvCacheConnectorConfig(connector="mooncake-store"), DISAGG),
        (eagle3(), None, DISAGG),
        (eagle3(), MOONCAKE, None),
        (eagle3(), MOONCAKE, CacheTransceiverConfig()),
        (MTPDecodingConfig(num_nextn_predict_layers=1), MOONCAKE, None),
    ],
    ids=[
        "no_spec",
        "ngram",
        "another_connector",
        "upstream_mooncake_connector",
        "no_connector",
        "aggregated",
        "transceiver_without_backend",
        "mtp_aggregated",
    ],
)
def test_other_combinations_keep_the_connector_guard(spec_config, connector, transceiver):
    assert not _connector_runs_speculative_prefill_only(spec_config, connector, transceiver)
