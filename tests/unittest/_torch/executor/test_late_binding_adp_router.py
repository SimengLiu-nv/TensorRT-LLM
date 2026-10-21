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
"""Late-binding attention-DP routing: bind a request to a rank only once it has room."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    LATE_BINDING_BUDGET_ENV,
    LATE_BINDING_ENV,
    ADPRouter,
    DefaultADPRouter,
    LateBindingADPRouter,
    RankState,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY

BUDGET = 16384


def _dist(tp_rank=0):
    dist = MagicMock()
    dist.tp_rank = tp_rank
    dist.has_cp_helix = False
    return dist


def _item(request_id, prompt_len, conversation_id=None, dp_rank=None, relax=None):
    scheduling_params = (
        None
        if dp_rank is None
        else SimpleNamespace(attention_dp_rank=dp_rank, attention_dp_relax=relax)
    )
    request = SimpleNamespace(
        num_input_tokens=prompt_len,
        priority=DEFAULT_REQUEST_PRIORITY,
        py_scheduling_params=scheduling_params,
        py_conversation_params=(
            None if conversation_id is None else SimpleNamespace(conversation_id=conversation_id)
        ),
    )
    return RequestQueueItem(request_id, request)


def _states(*loads, counts=None):
    counts = counts or [0] * len(loads)
    return [
        RankState(rank=r, num_active_requests=c, num_active_tokens=load)
        for r, (load, c) in enumerate(zip(loads, counts))
    ]


def _ids(assigned):
    return {rank: [item.id for item in items] for rank, items in assigned.items()}


def _request(
    request_id,
    prompt_len,
    position=0,
    context=True,
    context_only=True,
    gen_transmission=False,
):
    return SimpleNamespace(
        py_request_id=request_id,
        py_orig_prompt_len=prompt_len,
        context_current_position=position,
        is_context_init_state=context,
        is_context_only_request=context_only,
        is_disagg_generation_transmission_in_progress=gen_transmission,
    )


# ---- factory ----


def test_factory_selects_late_binding_from_env(monkeypatch):
    monkeypatch.setenv(LATE_BINDING_ENV, "1")
    monkeypatch.setenv(LATE_BINDING_BUDGET_ENV, "8192")
    router = ADPRouter.create(_dist())
    assert isinstance(router, LateBindingADPRouter)
    assert router._budget == 8192

    monkeypatch.delenv(LATE_BINDING_BUDGET_ENV)
    assert ADPRouter.create(_dist())._budget == LateBindingADPRouter.DEFAULT_BUDGET

    monkeypatch.setenv(LATE_BINDING_ENV, "0")
    router = ADPRouter.create(_dist())
    assert isinstance(router, DefaultADPRouter)
    assert not isinstance(router, LateBindingADPRouter)


def test_budget_must_be_positive():
    with pytest.raises(ValueError, match="pending_token_budget"):
        LateBindingADPRouter(_dist(), pending_token_budget=0)


def test_other_routers_never_defer():
    router = DefaultADPRouter(_dist())
    router.route_requests(_states(0, 0), [_item(1, 10)], max_num_active_requests=4)
    assert router.take_deferred_requests() == []


# ---- routing ----


def test_busy_rank_gets_nothing_and_idle_rank_fills_up_to_the_budget():
    router = LateBindingADPRouter(_dist(), pending_token_budget=BUDGET)
    # Rank 0 is deep in a long prefill; rank 1 is idle.
    items = [_item(1, 6000), _item(2, 6000), _item(3, 6000), _item(4, 6000)]
    assigned, expected = router.route_requests(
        _states(200_000, 0), items, max_num_active_requests=8
    )
    # Rank 1 takes requests while its pending prefill is under the budget:
    # 0 -> 6000 -> 12000 -> 18000, so three of them.
    assert _ids(assigned) == {0: [], 1: [1, 2, 3]}
    assert [item.id for item in router.take_deferred_requests()] == [4]
    assert expected == 3
    # Taking the deferred list empties it.
    assert router.take_deferred_requests() == []


def test_idle_ranks_share_new_requests_by_pending_work():
    router = LateBindingADPRouter(_dist(), pending_token_budget=BUDGET)
    items = [_item(1, 9000), _item(2, 3000), _item(3, 3000), _item(4, 3000)]
    assigned, _ = router.route_requests(_states(0, 0), items, max_num_active_requests=8)
    # 1 -> rank 0 (9000); 2 -> rank 1 (3000); 3 -> rank 1 (6000); 4 -> rank 1 (9000)
    assert _ids(assigned) == {0: [1], 1: [2, 3, 4]}
    assert router.take_deferred_requests() == []


def test_the_first_request_without_room_stops_routing_so_order_is_kept():
    router = LateBindingADPRouter(_dist(), pending_token_budget=BUDGET)
    # Request 1 is huge; after it lands, every rank is over the budget, and the
    # small request 2 must not overtake request 3 or jump the queue.
    items = [_item(1, 100_000), _item(2, 10), _item(3, 10)]
    assigned, _ = router.route_requests(_states(20_000, 0), items, max_num_active_requests=8)
    assert _ids(assigned) == {0: [], 1: [1]}
    assert [item.id for item in router.take_deferred_requests()] == [2, 3]


def test_a_rank_at_the_request_cap_takes_nothing():
    router = LateBindingADPRouter(_dist(), pending_token_budget=BUDGET)
    assigned, expected = router.route_requests(
        _states(0, 0, counts=[2, 0]), [_item(1, 10), _item(2, 10)], max_num_active_requests=2
    )
    assert _ids(assigned) == {0: [], 1: [1, 2]}
    assert expected == 2


def test_a_deferred_request_keeps_its_estimate_when_routed_again():
    router = LateBindingADPRouter(_dist(), pending_token_budget=BUDGET)
    router.estimate_work(_item(100, 120_000, "conv"))  # an earlier turn
    turn = _item(101, 125_000, "conv")  # grows by 5000
    router.route_requests(_states(BUDGET, BUDGET), [turn], max_num_active_requests=8)
    assert [item.id for item in router.take_deferred_requests()] == [101]
    # Routed again later: still 5000, not re-estimated against its own prompt.
    assigned, _ = router.route_requests(_states(0, BUDGET), [turn], max_num_active_requests=8)
    assert _ids(assigned) == {0: [101], 1: []}
    assert router._estimated_work[101] == 5000


def test_a_strict_rank_request_goes_to_its_rank_whatever_its_load():
    router = LateBindingADPRouter(_dist(), pending_token_budget=BUDGET)
    strict = _item(1, 10, dp_rank=0, relax=False)
    assigned, _ = router.route_requests(_states(500_000, 0), [strict], max_num_active_requests=8)
    assert _ids(assigned) == {0: [1], 1: []}


def test_a_relaxed_rank_request_falls_back_to_a_rank_with_room():
    router = LateBindingADPRouter(_dist(), pending_token_budget=BUDGET)
    relaxed = _item(1, 10, dp_rank=0, relax=True)
    assigned, _ = router.route_requests(_states(500_000, 0), [relaxed], max_num_active_requests=8)
    assert _ids(assigned) == {0: [], 1: [1]}
    # With room on its rank it stays there.
    assigned, _ = router.route_requests(
        _states(0, 0), [_item(2, 10, dp_rank=1, relax=True)], max_num_active_requests=8
    )
    assert _ids(assigned) == {0: [], 1: [2]}


# ---- pending prefill ----


def test_pending_prefill_is_exact_once_a_request_has_started():
    router = LateBindingADPRouter(_dist())
    router.estimate_work(_item(7, 50_000))
    # Not scheduled yet: the estimate.
    assert router.pending_prefill(_request(7, 50_000)) == 50_000
    # Scheduled with a 45K-token reused prefix: what is left to compute.
    assert router.pending_prefill(_request(7, 50_000, position=45_000)) == 5_000
    # Unknown request: its whole prompt.
    assert router.pending_prefill(_request(8, 9_000)) == 9_000


def test_parked_context_requests_count_and_finished_ones_do_not():
    router = LateBindingADPRouter(_dist())
    parked = _request(1, 40_000, position=36_000, context=False, gen_transmission=True)
    assert router.pending_prefill(parked) == 4_000
    sending = _request(2, 40_000, position=40_000, context=False)
    assert router.pending_prefill(sending) == 0
    generation = _request(3, 40_000, context=False, context_only=False, gen_transmission=True)
    assert router.pending_prefill(generation) == 0


def test_rank_state_sums_pending_prefill():
    router = LateBindingADPRouter(_dist(tp_rank=1))
    active = [
        _request(1, 30_000, position=28_000),
        _request(2, 40_000, position=40_000, context=False),
        _request(3, 9_000),
    ]
    state = router.create_rank_state(active, [])
    assert state.rank == 1
    assert state.num_active_requests == 3
    assert state.num_active_tokens == 2_000 + 9_000


# ---- with the waiting queue ----


def test_deferred_requests_return_to_the_front_of_the_queue_in_order():
    """What the executor does each iteration: pop, route, put back the rest."""
    router = LateBindingADPRouter(_dist(), pending_token_budget=BUDGET)
    queue = FCFSWaitingQueue()
    queue.add_requests([_item(i, 10_000) for i in range(1, 6)])

    def iteration(loads):
        popped = [queue.pop_request() for _ in range(len(queue))]
        assigned, _ = router.route_requests(_states(*loads), popped, max_num_active_requests=8)
        queue.prepend_requests(router.take_deferred_requests())
        return _ids(assigned)

    # Both ranks idle: one 10K request each would leave room for a second.
    assert iteration((0, 0)) == {0: [1, 3], 1: [2, 4]}
    assert [item.id for item in queue] == [5]
    queue.add_request(_item(6, 10))
    # Both still over budget: nothing moves and the order holds.
    assert iteration((BUDGET, BUDGET)) == {0: [], 1: []}
    assert [item.id for item in queue] == [5, 6]
    # Rank 1 frees up first and takes both.
    assert iteration((BUDGET, 0)) == {0: [], 1: [5, 6]}
    assert len(queue) == 0
