# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allocation handoff and retry contracts for Mooncake prefix loading."""

from contextlib import nullcontext
from queue import Queue
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    PageTransfer,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import (
    MooncakeStoreConnectorWorker,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType, SlotManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.mtp import MTPHiddenStatesManager

pytestmark = pytest.mark.cpu_only


def _request(request_id: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        py_request_id=request_id,
        is_first_context_chunk=True,
        py_connector_allocation_reported=False,
        state=LlmRequestState.CONTEXT_INIT,
    )


def test_async_resume_preserves_the_existing_mtp_slot() -> None:
    """A successful restore re-enters its first chunk after resource preparation already ran."""
    req = _request()
    batch = ScheduledRequests()
    batch.context_requests_chunking = [req]
    manager = SimpleNamespace(
        slot_manager=SlotManager(1), use_relaxed_acceptance_for_thinking=False
    )
    MTPHiddenStatesManager.prepare_resources(manager, batch)
    original_slot = manager.slot_manager.get_slot(req.request_id)
    MTPHiddenStatesManager.prepare_resources(manager, batch)
    assert manager.slot_manager.get_slot(req.request_id) == original_slot
    assert manager.slot_manager.free_slots == set()


@pytest.mark.parametrize("draft_groups", [[], [[21, 22]], [[21, 22], [31, 32]]])
def test_allocation_handoff_waits_for_all_kv_managers(
    draft_groups: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An async loader must receive the same group space as saves and the registered layout."""
    req = _request()
    batch = ScheduledRequests()
    batch.context_requests_chunking = [req]
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = False
    manager._stream = Mock()
    stream_context = Mock(side_effect=lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    manager.kv_connector_manager = Mock()
    manager.kv_connector_manager.should_add_sequence.return_value = True
    manager._connector_budgets_prefix = lambda: False
    manager._apply_connector_matched_prefix = lambda request: False
    manager.get_page_indices_by_layer_group = lambda request: [[11, 12]]
    draft = Mock(spec=KVCacheManagerV2) if draft_groups else None
    if draft is not None:
        draft._stream = Mock()
        draft.get_page_indices_by_layer_group.return_value = draft_groups

    manager.prepare_resources(batch)
    # The draft manager has not prepared its pages yet at this point.
    manager.kv_connector_manager.update_state_after_alloc.assert_not_called()
    assert not req.py_connector_allocation_reported

    manager.report_batch_to_connector(batch, draft)
    groups = [[11, 12], *draft_groups]
    flat = groups[0] if len(groups) == 1 else []
    manager.kv_connector_manager.update_state_after_alloc.assert_called_once_with(req, flat, groups)
    assert req.py_connector_allocation_reported
    stream_context.assert_called_once_with(manager._stream)
    if draft is not None:
        manager._stream.wait_stream.assert_called_once_with(draft._stream)

    # A deferred first chunk must not start a second DMA into the same allocation.
    manager.prepare_resources(batch)
    manager.report_batch_to_connector(batch, draft)
    assert manager.kv_connector_manager.update_state_after_alloc.call_count == 1


def test_async_store_waits_for_allocation_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither CPU-issued RDMA nor staging may race page recycling/COW on the allocator stream."""
    worker = object.__new__(MooncakeStoreConnectorWorker)
    worker._async_load = True
    worker._reraise_background_error = Mock()
    worker._load_lock = Lock()
    worker._load_queue = Queue()
    worker._async_pending = set()
    worker._async_active = set()
    worker._async_done = {}
    worker._async_stats = Mock()
    worker._load_streams = []
    worker._async_stagings = []
    worker._device_index = None
    worker._async_batch_size = 1
    worker._rank = 0
    worker._load_error = None
    transfers = RequestTransfers(7, [PageTransfer(b"prefix", 0, 1)])
    sequence = Mock()
    ready = Mock()
    ready.synchronize.side_effect = lambda: sequence.ready()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", lambda: ready)
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    worker._resolve = Mock(return_value=(["key"], [[123]], [[64]], 1))

    def load_pages(*args: object) -> tuple[set[int], int, None, int]:
        sequence.transfer()
        assert ready.synchronize.call_count == 1
        return set(), 0, None, 64

    worker._load_pages = load_pages
    worker.start_async_load(7, transfers)
    ready.record.assert_called_once_with(stream)
    worker._load_queue.put(None)
    worker._drain_loads(0)
    assert [entry[0] for entry in sequence.mock_calls] == ["ready", "transfer"]
    assert worker._async_done == {7: False}


@pytest.mark.parametrize("async_load", [False, True])
def test_failed_load_releases_speculative_slot_before_retry(async_load: bool) -> None:
    """A failed first chunk already owns an MTP slot; the retry must acquire it afresh."""
    req = _request()
    survivor = _request(8)
    slots = SlotManager(2)
    slots.add_slot(req.request_id)
    slots.add_slot(survivor.request_id)
    spec_manager = SimpleNamespace(
        free_resources=lambda request: slots.remove_slot(request.request_id)
    )
    executor = object.__new__(PyExecutor)
    executor.kv_cache_manager = Mock(spec=KVCacheManagerV2)
    executor.draft_kv_cache_manager = Mock(spec=KVCacheManagerV2)
    executor.kv_connector_manager = Mock()
    executor.resource_manager = Mock()
    executor.resource_manager.get_resource_manager.return_value = spec_manager
    executor.enable_attention_dp = False
    batch = ScheduledRequests()
    batch.context_requests_chunking = [req, survivor]

    if async_load:
        executor.kv_connector_manager.finished_async_loading_requests = {req.request_id: req}
        executor._recover_failed_async_loads({req.request_id})
    else:
        executor._recover_failed_connector_loads(batch, {req.request_id})
        assert batch.context_requests == [survivor]

    executor.resource_manager.get_resource_manager.assert_called_with(
        ResourceManagerType.SPEC_RESOURCE_MANAGER
    )
    assert slots.get_slot(req.request_id) is None
    assert slots.get_slot(survivor.request_id) is not None
    # Run the same slot admission that MTPHiddenStatesManager.prepare_resources uses.
    slots.add_slot(req.request_id)
    executor.kv_cache_manager.drop_context_allocation.assert_called_once_with(req)
    executor.draft_kv_cache_manager.free_resources.assert_called_once_with(req)
    executor.kv_connector_manager.forget_request.assert_called_once_with(req)


@pytest.mark.parametrize("v2", [False, True])
def test_sync_restore_uses_the_allocator_stream(v2: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.take_failed_load_requests.return_value = set()
    executor.enable_attention_dp = False
    executor.kv_cache_manager = Mock(spec=KVCacheManagerV2) if v2 else object()
    executor.execution_stream = Mock()
    default_stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: default_stream)
    executor._kv_connector_start_batch(ScheduledRequests())
    expected = executor.execution_stream if v2 else default_stream
    executor.kv_connector_manager.worker.start_load_kv.assert_called_once_with(expected)


def test_sync_gpu_rdma_waits_for_allocation_readiness() -> None:
    worker = object.__new__(MooncakeStoreConnectorWorker)
    transfers = RequestTransfers(7, [PageTransfer(b"prefix", 0, 1)])
    worker.get_connector_meta = Mock(return_value=SimpleNamespace(loads=[transfers]))
    worker._reraise_background_error = Mock()
    worker._resolve = Mock(return_value=(["key"], [[123]], [[64]], 1))
    worker._load_staging = None
    worker._rank = 0
    sequence = Mock()
    stream = Mock()
    stream.synchronize.side_effect = lambda: sequence.ready()

    def load_pages(*args: object) -> tuple[set[int], int, None, int]:
        sequence.transfer()
        assert stream.synchronize.call_count == 1
        return set(), 0, None, 64

    worker._load_pages = load_pages
    worker.start_load_kv(stream)
    assert [entry[0] for entry in sequence.mock_calls] == ["ready", "transfer"]
