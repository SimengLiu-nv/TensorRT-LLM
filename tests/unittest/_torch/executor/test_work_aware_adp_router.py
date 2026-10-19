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
"""Work-aware attention-DP routing: balance by estimated prefill work, not prompt length."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    ADPRouter,
    DefaultADPRouter,
    RankState,
    WorkAwareADPRouter,
)


def _dist(tp_rank=0):
    dist = MagicMock()
    dist.tp_rank = tp_rank
    dist.has_cp_helix = False
    return dist


def _item(request_id, prompt_len, conversation_id=None):
    request = SimpleNamespace(
        num_input_tokens=prompt_len,
        py_scheduling_params=None,
        py_conversation_params=(
            None if conversation_id is None else SimpleNamespace(conversation_id=conversation_id)
        ),
    )
    return RequestQueueItem(request_id, request)


def _states(*loads):
    return [
        RankState(rank=r, num_active_requests=0, num_active_tokens=load)
        for r, load in enumerate(loads)
    ]


def test_factory_selects_work_aware_router_from_env(monkeypatch):
    monkeypatch.setenv("TRTLLM_ADP_WORK_AWARE_ROUTING", "1")
    assert isinstance(ADPRouter.create(_dist()), WorkAwareADPRouter)
    monkeypatch.setenv("TRTLLM_ADP_WORK_AWARE_ROUTING", "0")
    router = ADPRouter.create(_dist())
    assert isinstance(router, DefaultADPRouter) and not isinstance(router, WorkAwareADPRouter)


def test_estimate_is_prompt_growth_within_a_conversation():
    router = WorkAwareADPRouter(_dist())
    assert router.estimate_work(_item(1, 100_000, "conv")) == 100_000
    # The next turn reuses the previous prompt; only the growth is prefilled.
    assert router.estimate_work(_item(2, 102_000, "conv")) == 2_000
    # A shrinking prompt (non-monotonic turn) has no usable predecessor.
    assert router.estimate_work(_item(3, 50_000, "conv")) == 50_000
    # No conversation id: the whole prompt.
    assert router.estimate_work(_item(4, 7_000)) == 7_000


def test_large_prefill_and_small_requests_land_on_different_ranks():
    router = WorkAwareADPRouter(_dist())
    for conv in ("a", "b", "c", "whale-seed"):
        router.estimate_work(_item(hash(conv) % 1000 + 1000, 130_000, conv))
    # Same prompt length everywhere, but "whale" brings 100K new tokens and the
    # three others about 2K each. Prompt-length balancing would split 2/2.
    items = [
        _item(10, 230_000, "whale-seed"),
        _item(11, 132_000, "a"),
        _item(12, 132_000, "b"),
        _item(13, 132_000, "c"),
    ]
    assigned, expected = router.route_requests(_states(0, 0), items, max_num_active_requests=8)
    ranks = {item.id: rank for rank, items_ in assigned.items() for item in items_}
    assert ranks[10] != ranks[11] == ranks[12] == ranks[13]
    assert expected == 3


def test_new_requests_go_to_the_rank_with_less_outstanding_work():
    router = WorkAwareADPRouter(_dist())
    assigned, _ = router.route_requests(
        _states(50_000, 1_000), [_item(20, 3_000)], max_num_active_requests=8
    )
    assert [item.id for item in assigned[1]] == [20] and assigned[0] == []


def test_expected_count_covers_every_rank_and_respects_capacity():
    router = WorkAwareADPRouter(_dist())
    states = [
        RankState(rank=0, num_active_requests=2, num_active_tokens=0),
        RankState(rank=1, num_active_requests=0, num_active_tokens=10**9),
    ]
    items = [_item(30 + i, 1_000) for i in range(3)]
    assigned, expected = router.route_requests(states, items, max_num_active_requests=3)
    # Rank 0 is least loaded but full after one more request; the rest spill over.
    assert len(assigned[0]) == 1 and len(assigned[1]) == 2
    assert expected == 3


def test_rank_state_counts_only_requests_still_in_context():
    router = WorkAwareADPRouter(_dist(tp_rank=1))
    router.estimate_work(_item(40, 9_000))
    router.estimate_work(_item(41, 5_000))
    prefilling = SimpleNamespace(
        py_request_id=40, is_context_init_state=True, py_orig_prompt_len=9_000
    )
    transferring = SimpleNamespace(
        py_request_id=41, is_context_init_state=False, py_orig_prompt_len=5_000
    )
    state = router.create_rank_state([prefilling, transferring], [])
    assert (state.rank, state.num_active_requests, state.num_active_tokens) == (1, 2, 9_000)
