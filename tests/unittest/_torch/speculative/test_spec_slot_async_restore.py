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
"""Speculative slots across an asynchronous KV connector restore.

A connector restore parks a context request after resource preparation, and
the request comes back on the same first chunk. The drafter must hand it the
slot it already owns instead of reserving a second one.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.eagle3 import Eagle3ResourceManager
from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def make_manager(max_num_requests: int = 4) -> Eagle3ResourceManager:
    config = Eagle3DecodingConfig(max_draft_len=3, speculative_model="draft")
    return Eagle3ResourceManager(
        config,
        torch.bfloat16,
        hidden_size=8,
        max_num_requests=max_num_requests,
        max_seq_len=16,
        max_num_tokens=64,
    )


def first_chunk(request_id: int) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, is_first_context_chunk=True)


def test_eagle3_resumed_request_keeps_its_slot():
    manager = make_manager()
    parked, other = first_chunk(7), first_chunk(8)
    manager.prepare_resources(SimpleNamespace(context_requests=[parked]))
    owned = manager.slot_manager.get_slot(parked.request_id)
    free_before = len(manager.slot_manager.free_slots)

    # The restore completes; the request resumes on its first chunk next to a new one.
    manager.prepare_resources(SimpleNamespace(context_requests=[parked, other]))

    assert manager.slot_ids[0] == owned
    assert manager.slot_manager.get_slot(other.request_id) == manager.slot_ids[1]
    assert len(manager.slot_manager.free_slots) == free_before - 1


def test_eagle3_freed_request_returns_its_slot_after_resume():
    manager = make_manager()
    req = first_chunk(7)
    free_at_start = len(manager.slot_manager.free_slots)
    manager.prepare_resources(SimpleNamespace(context_requests=[req]))
    manager.prepare_resources(SimpleNamespace(context_requests=[req]))

    manager.free_resources(req)

    assert manager.slot_manager.get_slot(req.request_id) is None
    assert len(manager.slot_manager.free_slots) == free_at_start
