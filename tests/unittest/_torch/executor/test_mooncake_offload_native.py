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
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import (
    MooncakeStoreConnectorWorker,
    _open_store,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MooncakeStoreConfig
from tensorrt_llm.mapping import Mapping

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
