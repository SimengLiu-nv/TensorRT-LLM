# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real C++ allocator, CUDA pages, and installed Mooncake lifecycle qualification."""

import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    RequestData,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import build_kv_cache_layout_v2
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    MooncakeStoreConnectorConfig,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import BlockHashChain
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import running_master
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    MooncakeStoreMetadata,
    PageTransfer,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.ownership import OffloadDirectory
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.ownership_server import (
    OwnershipServer,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import (
    MooncakeStoreConnectorScheduler,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import (
    MooncakeStoreConnectorWorker,
    _open_store,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MooncakeStoreConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import ReuseScope

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA KV pools")


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_native_allocator_offloads_and_restores_without_cpu_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mooncake.store")
    pool = MooncakeStoreConfig(launch_master=True, master_port=_port(), master_metrics_port=_port())
    with running_master(pool, str(tmp_path / "master")) as master:
        control = _open_store(
            MooncakeStoreConnectorConfig(
                master_server_address=master.address,
                protocol="tcp",
                global_segment_size=0,
                local_buffer_size=16 * 1024 * 1024,
            )
        )
        directory = OffloadDirectory(control, 16 * 1024 * 1024)
        with OwnershipServer(("127.0.0.1", 0), directory) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            config_path = tmp_path / "mooncake.json"
            config_path.write_text(
                json.dumps(
                    {
                        "master_server_address": master.address,
                        "protocol": "tcp",
                        "global_segment_size": "64MiB",
                        "local_buffer_size": "16MiB",
                        "write_policy": "offload",
                        "stage_through_host": True,
                        "staging_buffer_bytes": "1MiB",
                        "offload_coordinator_address": f"127.0.0.1:{server.server_address[1]}",
                    }
                )
            )
            monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
            manager = KVCacheManagerV2(
                kv_cache_config=KvCacheConfig(
                    max_tokens=2048, enable_block_reuse=True, host_cache_size=0
                ),
                kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
                num_layers=2,
                num_kv_heads=4,
                head_dim=64,
                tokens_per_block=32,
                max_seq_len=4096,
                max_batch_size=4,
                mapping=Mapping(world_size=1, tp_size=1, rank=0),
                dtype=tensorrt_llm.bindings.DataType.HALF,
                vocab_size=32000,
            )
            worker = MooncakeStoreConnectorWorker(
                SimpleNamespace(
                    model="ownership-native",
                    kv_cache_config=SimpleNamespace(tokens_per_block=32),
                    tensor_parallel_size=1,
                    enable_attention_dp=False,
                    pipeline_parallel_size=1,
                    context_parallel_size=1,
                    sparse_attention_config=None,
                    speculative_config=None,
                )
            )
            seed = manager.impl.create_kv_cache()
            pressure = None
            try:
                layout = build_kv_cache_layout_v2(manager)
                assert len(layout.groups) == 1
                worker.register_kv_cache_layout(layout)
                worker.register_kv_cache_manager(manager.impl)
                available = min(stat.free for stat in manager.impl.get_storage_statistics(0))
                tokens = list(range(available * 32))
                assert seed.resume(manager._stream.cuda_stream)
                assert seed.resize(len(tokens), len(tokens))
                chain = BlockHashChain(32)
                hashes = chain.extend(tokens)
                indices = list(seed.get_aggregated_page_indices(0))
                assert len(indices) == len(hashes)
                group = layout.groups[0]
                pages = [
                    PageTransfer(block_hash, 0, index) for block_hash, index in zip(hashes, indices)
                ]
                for ordinal, index in enumerate(indices):
                    for region in group.regions:
                        region.slot_tensor(index).fill_((ordinal % 251) + 1)
                worker.bind_connector_meta(
                    MooncakeStoreMetadata(saves=[RequestTransfers(1, pages)])
                )
                worker.wait_for_save(torch.cuda.current_stream())
                seed.commit(tokens, is_end=True)
                seed.close()
                assert directory.statistics()["cpu_keys"] == 0
                pressure = manager.impl.create_kv_cache()
                assert pressure.resume(manager._stream.cuda_stream)
                assert pressure.resize(32, 32)
                stats = directory.statistics()
                assert stats["cpu_keys"] >= 1
                assert stats["transitions"] == stats["settled_overlap_keys"] == 0
                keys = [worker._page_key(page) for page in pages]
                present = control.batch_is_exist(keys)
                ordinal = present.index(1)
                target = list(pressure.get_aggregated_page_indices(0))[0]
                for region in group.regions:
                    region.slot_tensor(target).zero_()
                lease, count = worker.reserve_prefix(2, [hashes[ordinal]])
                assert count == 1
                loaded = PageTransfer(hashes[ordinal], 0, target)
                worker.bind_connector_meta(
                    MooncakeStoreMetadata(loads=[RequestTransfers(2, [loaded], read_lease=lease)])
                )
                worker.start_load_kv(torch.cuda.current_stream())
                for region in group.regions:
                    assert torch.all(region.slot_tensor(target) == (ordinal % 251) + 1)
                assert control.batch_is_exist([keys[ordinal]]) == [0]
                assert directory.statistics()["settled_overlap_keys"] == 0
                assert directory.statistics()["read_leases"] == 0
                print("NATIVE_EXCLUSIVE_OFFLOAD_PASS", json.dumps(directory.statistics()))
            finally:
                seed.close()
                if pressure is not None:
                    pressure.close()
                worker.shutdown()
                manager.shutdown()
                server.shutdown()
                thread.join()
                control.close()


def test_native_rebased_prefix_is_not_registered_as_new_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps(
            {
                "master_server_address": "127.0.0.1:50051",
                "role": "producer",
            }
        )
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("TRTLLM_MOONCAKE_STORE_ROLE", raising=False)
    spec_config = MTPDecodingConfig(max_draft_len=1)
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(max_tokens=2048, enable_block_reuse=True, host_cache_size=0),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=4096,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
        vocab_size=32000,
        spec_config=spec_config,
    )
    scheduler = MooncakeStoreConnectorScheduler(
        SimpleNamespace(
            model="ownership-native",
            kv_cache_config=SimpleNamespace(tokens_per_block=32),
            tensor_parallel_size=1,
            enable_attention_dp=False,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            sparse_attention_config=None,
            speculative_config=spec_config,
        )
    )
    released: list[tuple[int, int]] = []

    def record_release(group: int, slot: int) -> None:
        released.append((group, slot))

    manager.impl.set_gpu_eviction_callbacks(None, record_release)
    original_tokens = list(range(97))
    other_tokens = list(original_tokens)
    other_tokens[32] += 1000
    first_slot = None
    try:
        for tokens in (original_tokens, other_tokens):
            cache = manager.impl.create_kv_cache()
            try:
                assert cache.resume(manager._stream.cuda_stream)
                assert cache.resize(len(tokens), len(tokens))
                cache.commit(tokens)
                slot = list(cache.get_aggregated_page_indices(0))[0]
                if first_slot is None:
                    first_slot = slot
                else:
                    assert slot == first_slot
            finally:
                cache.close()
        assert (0, first_slot) not in released
        assert (
            BlockHashChain(32, prompt_lookahead=1).extend(original_tokens)[0]
            != BlockHashChain(32, prompt_lookahead=1).extend(other_tokens)[0]
        )

        cache = manager.impl.create_kv_cache(ReuseScope(), other_tokens)
        try:
            local_tokens = cache.history_length
            assert local_tokens >= 64
            assert cache.resume(manager._stream.cuda_stream)
            assert cache.resize(len(other_tokens), len(other_tokens))
            indices = list(cache.get_aggregated_page_indices(0))
            assert indices[0] == first_slot
            request = SimpleNamespace(
                request_id=3, cache_salt=None, get_tokens=lambda beam: other_tokens
            )
            assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (0, False)
            metadata = scheduler.build_connector_meta(
                SchedulerOutput(
                    new_requests=[
                        RequestData(
                            request_id=3,
                            new_tokens=other_tokens,
                            new_block_ids=indices,
                            new_block_ids_by_layer_group=[indices],
                            computed_position=local_tokens,
                            num_scheduled_tokens=len(other_tokens) - local_tokens,
                        )
                    ]
                )
            )
            assert first_slot not in [
                page.page_index for transfer in metadata.saves for page in transfer.pages
            ]
        finally:
            cache.close()
    finally:
        manager.impl.set_gpu_eviction_callbacks(None, None)
        manager.shutdown()
