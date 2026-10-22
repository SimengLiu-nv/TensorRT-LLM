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
"""The KV connector manager's attention-DP and speculative-prefill modes.

Under attention DP every rank owns its requests outright, so the manager runs
its scheduler adapter locally and completes transfers without collectives.
Speculative prefill admits one-model drafting by refusing everything that is
not context-only work. Both modes are off unless the executor creator asks.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    AsyncRequests,
    KvCacheConnectorManager,
    KvCacheConnectorScheduler,
    KvCacheConnectorSchedulerOutputManager,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only
CONNECTOR_MODULE = "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"


class ADPAdapterMock(MagicMock):
    supports_attention_dp = True


def _request(request_id: int, *, dummy: bool = False) -> MagicMock:
    req = MagicMock(
        is_dummy_request=dummy,
        is_attention_dp_dummy=dummy,
        is_generation_only_request=False,
        is_context_only_request=True,
    )
    req.request_id = request_id
    req.state = LlmRequestState.CONTEXT_INIT
    req.context_current_position = 0
    req.context_remaining_length = 8
    req.context_chunk_size = 8
    req.get_tokens.return_value = list(range(8))
    req.py_draft_tokens = []
    req.kv_cache_retention_config = None
    req.cache_salt = None
    return req


def _manager(rank: int = 1) -> KvCacheConnectorManager:
    worker = ADPAdapterMock(capacity_only=False, reports_failed_loads=True)
    worker.get_finished.return_value = ([], [])
    worker.take_failed_load_requests.return_value = set()
    scheduler = ADPAdapterMock()
    scheduler.get_num_new_matched_tokens.return_value = (4, True)
    scheduler.request_finished.return_value = True
    with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=rank):
        return KvCacheConnectorManager(worker, scheduler, enable_attention_dp=True)


@pytest.fixture
def forbid_connector_collectives() -> Iterator[None]:
    with (
        patch(f"{CONNECTOR_MODULE}.mpi_allgather", side_effect=AssertionError("ADP allgather")),
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=AssertionError("ADP broadcast")),
    ):
        yield


@pytest.mark.parametrize("unsupported", [KvCacheConnectorWorker, KvCacheConnectorScheduler])
def test_adp_requires_explicit_backend_support(unsupported: type) -> None:
    with pytest.raises(NotImplementedError, match="supports_attention_dp=True"):
        KvCacheConnectorManager.validate_attention_dp(ADPAdapterMock, unsupported)


def test_adp_is_validated_when_the_manager_is_built() -> None:
    with pytest.raises(NotImplementedError, match="supports_attention_dp=True"):
        KvCacheConnectorManager(MagicMock(), ADPAdapterMock(), enable_attention_dp=True)


@pytest.mark.parametrize("rank", [0, 1])
def test_adp_requires_scheduler_on_every_rank(rank: int) -> None:
    with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=rank):
        with pytest.raises(AssertionError, match="every attention-DP rank"):
            KvCacheConnectorManager(ADPAdapterMock(), None, enable_attention_dp=True)


def test_modes_default_off() -> None:
    with (
        patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=0),
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=lambda result, root: result),
    ):
        manager = KvCacheConnectorManager(MagicMock(), MagicMock())
    assert not manager.enable_attention_dp
    assert not manager.speculative_prefill_only


@pytest.mark.parametrize("request_counts", [(2, 2), (3, 0), (0, 4)])
def test_disjoint_owner_callbacks_and_metadata(
    forbid_connector_collectives: None, request_counts: tuple[int, int]
) -> None:
    for rank, count in enumerate(request_counts):
        manager = _manager(rank)
        cache = MagicMock()
        cache.get_cache_indices.return_value = [17 + rank]
        cache.commit_and_get_block_hashes.return_value = [123]
        batch = ScheduledRequests()
        manager.scheduler.get_num_new_matched_tokens.return_value = (4, False)
        for index in range(count):
            request = _request(rank * 100 + index)
            assert manager.get_num_new_matched_tokens(request, 0) == 4
            manager.update_state_after_alloc(request, [17 + rank])
            request.context_current_position = 4
            batch.context_requests_last_chunk.append(request)

        with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=rank):
            manager.build_scheduler_output(batch, cache)
        manager.handle_metadata()
        output = manager.scheduler.build_connector_meta.call_args.args[0]
        assert output.attention_dp_rank == rank
        assert [req.request_id for req in output.new_requests] == [
            rank * 100 + index for index in range(count)
        ]
        assert all(req.new_block_ids == [17 + rank] for req in output.new_requests)
        assert all(req.computed_position == 0 for req in output.new_requests)
        manager.worker.bind_connector_meta.assert_called_once_with(
            manager.scheduler.build_connector_meta.return_value
        )
        assert manager.get_finished() == []


def test_owner_completion_does_not_wait_for_idle_peer(forbid_connector_collectives: None) -> None:
    manager = _manager()
    loading, saving = _request(10), _request(20)
    assert manager.get_num_new_matched_tokens(loading, 0) == 4
    manager.update_state_after_alloc(loading, [7, 8])
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [loading]
    manager.take_scheduled_requests_pending_load(batch)
    assert batch.batch_size == 0
    assert loading.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert manager.request_finished(saving, [9])
    assert manager.has_pending_load(loading)
    assert manager.has_outstanding_save(20)

    assert manager.get_finished() == []
    manager.worker.get_finished.assert_called_once_with([20], [10])

    # The worker may report load/save completion on different iterations.
    manager.worker.get_finished.return_value = ([], [10])
    assert manager.get_finished() == []
    assert loading.state == LlmRequestState.CONTEXT_INIT
    assert not manager.should_add_sequence(loading)
    assert not manager.has_pending_load(loading)
    assert manager.has_outstanding_save(20)
    manager.worker.get_finished.assert_called_with([], [])

    manager.worker.get_finished.return_value = ([20], [])
    assert manager.get_finished() == [saving]
    assert not manager.has_outstanding_save(20)
    assert not manager.local_finished_async_requests.saving
    assert not manager.pending_async_requests.loading


def test_dummy_requests_never_reach_storage(forbid_connector_collectives: None) -> None:
    manager = _manager()
    dummy = _request(999, dummy=True)
    assert manager.get_num_new_matched_tokens(dummy, 0) == 0
    manager.update_state_after_alloc(dummy, [5])
    assert not manager.request_finished(dummy, [5])
    manager.scheduler.get_num_new_matched_tokens.assert_not_called()
    manager.scheduler.update_state_after_alloc.assert_not_called()
    manager.scheduler.request_finished.assert_not_called()
    assert not manager.has_pending_load(dummy)
    assert not manager.has_outstanding_save(999)

    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [dummy]
    batch.generation_requests = [dummy]
    cache = MagicMock()
    output_manager = KvCacheConnectorSchedulerOutputManager()
    output = output_manager.build_scheduler_output(batch, AsyncRequests({}, {}), cache)
    assert not output.new_requests and not output.cached_requests
    assert not output_manager.requests
    cache.get_cache_indices.assert_not_called()
    cache.commit_and_get_block_hashes.assert_not_called()


def test_tp_still_waits_for_all_shards() -> None:
    worker, scheduler = MagicMock(capacity_only=False), MagicMock()
    scheduler.request_finished.return_value = True
    with (
        patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=0),
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=lambda result, root: result),
        patch(f"{CONNECTOR_MODULE}.mpi_allgather") as gather,
    ):
        manager = KvCacheConnectorManager(worker, scheduler)
        req = _request(42)
        manager.request_finished(req, [5])
        worker.get_finished.return_value = ([42], [])
        gather.return_value = [([42], []), ([], [])]
        assert manager.get_finished() == []
        assert manager.has_outstanding_save(42)
        worker.get_finished.return_value = ([], [])
        gather.return_value = [([42], []), ([42], [])]
        assert manager.get_finished() == [req]


def test_adp_v2_query_commit_and_grouped_callbacks(forbid_connector_collectives: None) -> None:
    manager = _manager(rank=3)
    request = _request(17)
    manager.scheduler.get_num_new_matched_tokens.return_value = (8, False)
    assert manager.query_num_new_matched_tokens(request, 0) == (8, False)
    assert not manager.scheduler_output_manager.external_loads
    manager.commit_new_matched_tokens(request, 4, False)
    assert manager.scheduler_output_manager.external_loads[17] == 4
    groups = [[2, 4], [9, 3]]
    manager.update_state_after_alloc(request, [], groups)
    manager.scheduler.update_state_after_alloc_by_layer_group.assert_called_once_with(
        request, groups
    )
    manager.scheduler.update_state_after_alloc.assert_not_called()
    manager.scheduler.request_finished_by_layer_group.return_value = True
    assert manager.request_finished(request, [], groups)
    manager.scheduler.request_finished_by_layer_group.assert_called_once_with(request, groups)
    manager.scheduler.request_finished.assert_not_called()
    manager.worker.get_finished.return_value = ([17], [])
    assert manager.get_finished() == [request]


def test_dummy_v2_callbacks_do_not_publish_transfer_state(
    forbid_connector_collectives: None,
) -> None:
    manager = _manager()
    dummy = _request(999, dummy=True)
    assert manager.query_num_new_matched_tokens(dummy, 0) == (0, False)
    manager.commit_new_matched_tokens(dummy, 4, True)
    manager.update_state_after_alloc(dummy, [], [[3], [5]])
    assert not manager.request_finished(dummy, [], [[3], [5]])
    manager.scheduler.get_num_new_matched_tokens.assert_not_called()
    manager.scheduler.update_state_after_alloc_by_layer_group.assert_not_called()
    manager.scheduler.request_finished_by_layer_group.assert_not_called()
    assert not manager.scheduler_output_manager.external_loads
    assert not manager.has_pending_load(dummy)


def test_failed_loads_and_restarts_reach_the_worker(forbid_connector_collectives: None) -> None:
    manager = _manager()
    manager.worker.take_failed_load_requests.return_value = {7}
    manager.worker.take_failed_async_load_requests.return_value = [8]
    assert manager.take_failed_load_requests() == {7}
    assert manager.take_failed_async_load_requests() == {8}

    request = _request(7)
    manager.scheduler_output_manager.external_loads[7] = 4
    manager.forget_request(request)
    manager.worker.drop_bound_saves.assert_called_once_with([7])
    manager.scheduler.request_reset.assert_called_once_with(request)
    assert 7 not in manager.scheduler_output_manager.external_loads


def test_failed_loads_are_reported_only_by_a_connector_that_declares_it() -> None:
    manager = _manager()
    assert manager.reports_failed_loads
    with (
        patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=0),
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=lambda result, root: result),
    ):
        # A mock answers every attribute truthily; only a declared True counts.
        assert not KvCacheConnectorManager(MagicMock(), MagicMock()).reports_failed_loads


def test_mock_requests_are_not_mistaken_for_dummies(forbid_connector_collectives: None) -> None:
    manager = _manager()
    request = MagicMock()
    request.request_id = 5
    manager.scheduler.request_finished.return_value = True
    assert manager.request_finished(request, [1])
    manager.scheduler.request_finished.assert_called_once_with(request, [1])


def test_failed_load_sets_are_empty_for_a_connector_without_them() -> None:
    class PlainWorker:
        capacity_only = False

    with (
        patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=0),
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=lambda result, root: result),
    ):
        manager = KvCacheConnectorManager(PlainWorker(), MagicMock())
    assert manager.take_failed_load_requests() == set()
    assert manager.take_failed_async_load_requests() == set()


@pytest.mark.parametrize("context_only,dummy", [(True, False), (False, True)])
def test_speculative_prefill_allows_context_work_and_adp_dummies(
    context_only: bool, dummy: bool
) -> None:
    manager = _manager()
    manager.speculative_prefill_only = True
    request = _request(7, dummy=dummy)
    request.is_context_only_request = context_only

    manager.query_num_new_matched_tokens(request, 0)

    assert manager.scheduler.get_num_new_matched_tokens.call_count == (0 if dummy else 1)


@pytest.mark.parametrize("generation_only", [False, True])
def test_speculative_prefill_rejects_generation_before_lookup(generation_only: bool) -> None:
    manager = _manager()
    manager.speculative_prefill_only = True
    request = _request(7)
    request.is_context_only_request = False
    request.is_generation_only_request = generation_only

    with pytest.raises(NotImplementedError, match="context-only"):
        manager.query_num_new_matched_tokens(request, 0)

    manager.scheduler.get_num_new_matched_tokens.assert_not_called()


def test_speculative_prefill_rejects_generation_before_transfer_scheduling() -> None:
    manager = _manager()
    manager.speculative_prefill_only = True
    request = _request(7)
    request.is_context_only_request = False
    batch = ScheduledRequests()
    batch.generation_requests.append(request)

    with pytest.raises(NotImplementedError, match="context-only"):
        manager.build_scheduler_output(batch, MagicMock())

    assert manager.scheduler_output_manager.requests == {}


@pytest.mark.parametrize("prefill_only", [False, True])
@pytest.mark.parametrize("context_only", [False, True])
def test_speculative_prefill_refuses_other_requests_at_admission(
    prefill_only: bool, context_only: bool
) -> None:
    """Only the refused request fails; the executor never reaches resource preparation."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    manager = _manager()
    manager.speculative_prefill_only = prefill_only
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    request = _request(7)
    request.is_context_only_request = context_only

    if prefill_only and not context_only:
        with pytest.raises(ValueError, match="context-only"):
            executor._validate_request(request)
    else:
        executor._validate_connector_request(request)


def test_reset_drops_scheduler_output_state(forbid_connector_collectives: None) -> None:
    """A released allocation is reported to the adapter as a new request when it reruns."""
    manager = _manager()
    request = _request(42)
    manager.scheduler_output_manager.requests[42]
    manager.scheduler_output_manager.external_loads[42] = 4
    manager.reset_request_state(request)
    assert 42 not in manager.scheduler_output_manager.requests
    assert 42 not in manager.scheduler_output_manager.external_loads
    manager.scheduler.request_reset.assert_called_once_with(request)


def test_async_padding_preserves_connector_plan() -> None:
    """Padding a batch the connector emptied must not re-report it to the connector."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType

    manager = _manager()
    req = _request(7)
    manager.get_num_new_matched_tokens(req, 0)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [req]
    plan = SchedulerOutput()
    manager.set_scheduler_output(plan)
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    executor.enable_attention_dp = True
    cache = MagicMock()
    cache.report_batch_to_connector.side_effect = lambda batch: manager.set_scheduler_output(
        SchedulerOutput()
    )
    executor.resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: cache})
    executor.kv_cache_manager = cache
    dummy = _request(999, dummy=True)

    def pad_batch(scheduled_batch: ScheduledRequests) -> None:
        assert scheduled_batch.batch_size == 0
        scheduled_batch.generation_requests.append(dummy)

    executor._pad_empty_attention_dp_batch = pad_batch
    with patch("torch.cuda.current_stream"):
        executor._kv_connector_start_batch(batch)
    assert batch.generation_requests == [dummy]
    cache.prepare_resources.assert_called_once_with(batch)
    cache.report_batch_to_connector.assert_not_called()
    manager.scheduler.build_connector_meta.assert_called_once_with(plan)
    manager.worker.start_load_kv.assert_called_once()


def test_failed_sync_load_restarts_the_request_and_pads_the_batch() -> None:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    manager = _manager()
    manager.scheduler.get_num_new_matched_tokens.return_value = (4, False)
    req = _request(7)
    manager.get_num_new_matched_tokens(req, 0)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [req]
    manager.set_scheduler_output(SchedulerOutput())
    manager.worker.take_failed_load_requests.return_value = {7}

    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    executor.enable_attention_dp = True
    executor.kv_cache_manager = MagicMock()
    executor.resource_manager = MagicMock()
    executor.resource_manager.get_resource_manager.return_value = None
    padded = []
    executor._pad_empty_attention_dp_batch = lambda scheduled: padded.append(scheduled.batch_size)

    with patch("torch.cuda.current_stream"):
        executor._kv_connector_start_batch(batch)

    # The request left the batch, its allocation was dropped, and the connector forgot it.
    assert batch.context_requests == []
    executor.kv_cache_manager.drop_context_allocation.assert_called_once_with(req)
    manager.worker.drop_bound_saves.assert_called_with([7])
    manager.scheduler.request_reset.assert_called_with(req)
    # The emptied batch was padded so the other ranks keep running.
    assert padded == [0]


def test_failed_async_load_restarts_the_parked_request() -> None:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    manager = _manager()
    req = _request(8)
    manager.finished_async_loading_requests[8] = req
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    executor.kv_cache_manager = MagicMock()
    spec_manager = MagicMock()
    executor.resource_manager = MagicMock()
    executor.resource_manager.get_resource_manager.return_value = spec_manager

    executor._recover_failed_async_loads({8, 9})

    assert 8 not in manager.finished_async_loading_requests
    assert req.state == LlmRequestState.CONTEXT_INIT
    executor.kv_cache_manager.drop_context_allocation.assert_called_once_with(req)
    spec_manager.free_resources.assert_called_once_with(req)
    # It is asked about again rather than skipped as an already-loaded request.
    assert manager.should_add_sequence(req)


def test_failed_async_load_of_a_finished_request_restarts_nothing_silently() -> None:
    """A failure reported for a request that already left is not a restart."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm.logger import logger

    manager = _manager()
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    executor.kv_cache_manager = MagicMock()

    with patch.object(logger, "warning") as warning:
        executor._recover_failed_async_loads({9})

    executor.kv_cache_manager.drop_context_allocation.assert_not_called()
    warning.assert_not_called()


def test_cancel_waits_for_connector_load(forbid_connector_collectives: None) -> None:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    manager = _manager()
    req = _request(8)
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    executor.kv_cache_transceiver = None
    manager.get_num_new_matched_tokens(req, 0)
    assert not executor._try_cancel_request(req)
    manager.worker.get_finished.return_value = ([], [8])
    manager.get_finished()
    assert executor._try_cancel_request(req)
