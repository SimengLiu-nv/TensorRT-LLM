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
"""Disaggregated completion reservation in KVCacheManagerV2.

With no eviction tier, a full-attention disaggregated server reserves the pages an admitted
request still needs to finish: a decode request its output growth, a connector prefill its
remaining prompt. A new request is admitted only if the reservation of every admitted request
still fits.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "available, capacities, limits, candidate_limit, expected",
    [
        ([7], [8, 0], {1: 6}, 3, True),
        ([6], [8, 0], {1: 6}, 3, False),
        ([4], [20, 0], {1: 6}, 3, True),
        ([3], [8, 8], {1: 6, 2: 3}, 3, False),
        ([5], [8, 8], {1: 6, 2: 3}, 3, True),
        ([100, 6], [8, 0], {1: 6}, 3, False),
    ],
)
def test_completion_reservation_counts_unallocated_pages_in_every_pool(
    available: list[int],
    capacities: list[int],
    limits: dict[int, int],
    candidate_limit: int,
    expected: bool,
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._disagg_completion_limits = dict(limits)
    manager._disagg_completion_pool_counts = {i: 1 for i in range(len(available))}
    manager.kv_cache_map = {
        i + 1: Mock(capacity=capacity, num_blocks=(capacity + 3) // 4)
        for i, capacity in enumerate(capacities)
    }
    manager._get_storage_statistics = Mock(
        return_value=[SimpleNamespace(available=count) for count in available]
    )

    assert manager._can_reserve_disagg_completion(2, candidate_limit) is expected
    # Checking admission neither consumes a reservation nor touches a cache.
    assert manager._disagg_completion_limits == limits
    for cache in manager.kv_cache_map.values():
        cache.resize.assert_not_called()


@pytest.mark.cpu_only
def test_completion_reservation_skips_a_bound_whose_cache_is_gone() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._disagg_completion_limits = {1: 100, 2: 3}
    manager._disagg_completion_pool_counts = {0: 1}
    manager.kv_cache_map = {2: Mock(num_blocks=0)}
    manager._get_storage_statistics = Mock(return_value=[SimpleNamespace(available=3)])

    assert manager._can_reserve_disagg_completion(2, 3)


@pytest.mark.cpu_only
@pytest.mark.parametrize("enable_block_reuse", [False, True])
@pytest.mark.parametrize("ledger_tokens_per_block, expected_blocks", [(4, 10), (16, 3)])
def test_generation_limit_counts_pool_variants_and_speculative_padding(
    enable_block_reuse: bool, ledger_tokens_per_block: int, expected_blocks: int
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._ledger_tokens_per_block = ledger_tokens_per_block
    manager.max_seq_len = 32
    manager.num_extra_kv_tokens = 1
    manager._kv_reserve_draft_tokens = 5
    manager.enable_block_reuse = enable_block_reuse
    request = Mock(max_new_tokens=20)

    # Clamp to the model sequence length, then add extra KV, draft and base-token padding.
    limit = manager._disagg_generation_limit(request, 24)
    assert limit == expected_blocks + int(enable_block_reuse)

    manager._disagg_completion_limits = {}
    manager._disagg_completion_pool_counts = {0: 2}
    manager.kv_cache_map = {1: Mock(capacity=0, num_blocks=0)}
    stats = SimpleNamespace(available=2 * limit - 1)
    manager._get_storage_statistics = Mock(return_value=[stats])
    assert not manager._can_reserve_disagg_completion(1, limit)
    stats.available = 2 * limit
    assert manager._can_reserve_disagg_completion(1, limit)


def _context_reservation_manager(available: int, **connector_overrides) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager._ledger_tokens_per_block = 4
    manager.num_extra_kv_tokens = 0
    manager._kv_reserve_draft_tokens = 0
    manager.enable_block_reuse = False
    manager.is_draft = False
    manager._disagg_completion_limits = {}
    manager._disagg_completion_pool_counts = {0: 1}
    manager.kv_cache_map = {7: Mock(num_blocks=0)}
    manager._get_storage_statistics = Mock(return_value=[SimpleNamespace(available=available)])
    connector = dict(prefix_reservations_enabled=False, capacity_only=False)
    connector.update(connector_overrides)
    manager.kv_connector_manager = SimpleNamespace(**connector)
    return manager


def _first_chunk(**overrides) -> SimpleNamespace:
    fields = dict(
        py_request_id=7,
        prompt_len=16,
        is_first_context_chunk=True,
        is_dummy=False,
        is_generation_only_request=False,
        is_disagg_generation_init_state=False,
        py_connector_allocation_reported=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.cpu_only
def test_context_reservation_records_a_bound_once_it_fits() -> None:
    # 16 prompt tokens + 1 base token in 4-token blocks: 5 blocks.
    manager = _context_reservation_manager(available=5)
    request = _first_chunk()

    assert manager.reserve_context_completion(request)
    assert manager._disagg_completion_limits == {7: 5}
    # A recorded bound is not checked again on later passes.
    manager._get_storage_statistics.return_value = [SimpleNamespace(available=0)]
    assert manager.reserve_context_completion(request)


@pytest.mark.cpu_only
def test_context_reservation_defers_without_recording() -> None:
    manager = _context_reservation_manager(available=4)

    assert not manager.reserve_context_completion(_first_chunk())
    assert manager._disagg_completion_limits == {}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "request_overrides, connector_overrides, has_pool_counts",
    [
        (dict(is_first_context_chunk=False), {}, True),
        ({}, {}, False),
        ({}, dict(prefix_reservations_enabled=True), True),
        ({}, dict(capacity_only=True), True),
        (dict(is_dummy=True), {}, True),
    ],
    ids=["later_chunk", "eviction_tier", "source_reservations", "capacity_only", "dummy"],
)
def test_context_reservation_is_a_noop_outside_conservative_connector_prefill(
    request_overrides: dict, connector_overrides: dict, has_pool_counts: bool
) -> None:
    manager = _context_reservation_manager(available=0, **connector_overrides)
    if not has_pool_counts:
        manager._disagg_completion_pool_counts = {}

    assert manager.reserve_context_completion(_first_chunk(**request_overrides))
    assert manager._disagg_completion_limits == {}
    manager._get_storage_statistics.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "first_chunk, reported, released",
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_unconsumed_context_reservation_is_returned(
    first_chunk: bool, reported: bool, released: bool
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._disagg_completion_limits = {7: 5}
    request = _first_chunk(
        is_first_context_chunk=first_chunk, py_connector_allocation_reported=reported
    )

    manager._release_unconsumed_context_reservation(request)

    assert (7 not in manager._disagg_completion_limits) is released


@pytest.mark.parametrize("reserve_growth", [False, True])
@pytest.mark.parametrize("enable_block_reuse", [False, True])
def test_disagg_generation_progress_under_real_gpu_page_pressure(
    reserve_growth: bool, enable_block_reuse: bool
) -> None:
    """Prompt-only admission can fill the pool before either request finishes.

    Use the real allocator and admission path; the control disables only the reservation.
    The guarded path must defer the second long prompt, admit a short request alongside the
    first, and release reservations on free.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    manager = KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
            host_cache_size=0,
            max_gpu_total_bytes=128 << 10,
            max_util_for_resume=1.0,
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=8,
        head_dim=64,
        tokens_per_block=4,
        max_seq_len=65536,
        max_batch_size=4,
        max_num_tokens=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=4096,
        is_disagg=True,
    )
    try:
        assert manager._disagg_completion_pool_counts == {0: 1}
        if not reserve_growth:
            manager._disagg_completion_pool_counts.clear()
        total_blocks = manager.get_kv_cache_stats().max_num_blocks
        prompt_len = total_blocks // 3 * manager.tokens_per_block
        assert prompt_len >= 8
        # Allocation granularity can make the physical pool larger than the tiny
        # requested byte quota. Keep this pressure probe within the actual table.
        assert 2 * prompt_len < manager.max_seq_len

        def make_request(request_id: int, prompt: int, output: int) -> LlmRequest:
            request = LlmRequest(
                request_id=request_id,
                max_new_tokens=output,
                input_tokens=[request_id] * prompt,
                sampling_config=SamplingConfig(1),
                is_streaming=False,
            )
            request.state = LlmRequestState.DISAGG_GENERATION_INIT
            return request

        first = make_request(1, prompt_len, prompt_len)
        second = make_request(2, prompt_len, prompt_len)
        short = make_request(3, 4, 1)

        assert manager.prepare_disagg_gen_init(first)
        assert manager.kv_cache_map[1].capacity == prompt_len
        assert manager.prepare_disagg_gen_init(second) is not reserve_growth
        first.state = LlmRequestState.GENERATION_IN_PROGRESS
        if reserve_growth:
            assert 2 not in manager.kv_cache_map
            assert 2 not in manager._disagg_completion_limits
            assert manager.prepare_disagg_gen_init(short)
            for _ in range(prompt_len):
                assert manager.try_allocate_generation(first)
            manager.free_resources(first)
            assert 1 not in manager._disagg_completion_limits
            assert manager.prepare_disagg_gen_init(second)
            # Cancellation must return both the physical pages and the logical bound.
            manager.free_resources(second)
            manager.free_resources(short)
            assert not manager._disagg_completion_limits
            assert manager.get_kv_cache_stats().free_num_blocks == total_blocks
        else:
            second.state = LlmRequestState.GENERATION_IN_PROGRESS
            for _ in range(prompt_len):
                first_fits = manager.try_allocate_generation(first)
                second_fits = manager.try_allocate_generation(second)
                if not first_fits and not second_fits:
                    break
            else:
                pytest.fail("The unguarded control did not exhaust its GPU KV pool")
    finally:
        manager.shutdown()
