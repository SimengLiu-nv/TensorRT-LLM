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

"""GLM-5.2 NVFP4 heterogeneous-topology KV-transfer coverage."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import kv_transfer_harness as transfer_harness
import pytest
import torch

from tensorrt_llm import Mapping
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool, get_pool_bytes
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MTPDecodingConfig,
)

NUM_TARGET_LAYERS = 6
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
INDEX_HEAD_DIM = 128
TOKENS_PER_BLOCK = 64
EXPECTED_INDEXER_MASK = [True, True, True, False, False, False, True]


@dataclass(frozen=True)
class _LayerRegion:
    base_address: int
    slot_bytes: int
    offset: int
    size: int


def _create_manager(mapping: Mapping) -> DSACacheManagerV2:
    max_num_tokens = transfer_harness.MAX_SEQ_LEN * transfer_harness.MAX_BATCH_SIZE
    pretrained_config = SimpleNamespace(
        num_hidden_layers=NUM_TARGET_LAYERS,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_share_for_mtp_iteration=True,
    )
    sparse_attention_config = DeepSeekSparseAttentionConfig(
        index_head_dim=INDEX_HEAD_DIM,
        index_n_heads=32,
        index_topk=2048,
        indexer_k_dtype="fp8",
        index_share_for_mtp_iteration=True,
    )
    return DSACacheManagerV2(
        kv_cache_config=KvCacheConfig(
            dtype="nvfp4",
            enable_block_reuse=False,
            max_tokens=max_num_tokens,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=NUM_TARGET_LAYERS,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=transfer_harness.MAX_SEQ_LEN,
        max_batch_size=transfer_harness.MAX_BATCH_SIZE,
        mapping=mapping,
        dtype=DataType.NVFP4,
        spec_config=MTPDecodingConfig(max_draft_len=5),
        max_num_tokens=max_num_tokens,
        vocab_size=transfer_harness.VOCAB_SIZE,
        sparse_attention_config=sparse_attention_config,
        pretrained_config=pretrained_config,
    )


def _create_managers(tp: int, pp: int, enable_dp: bool) -> list[DSACacheManagerV2]:
    assert pp == 1
    return [
        _create_manager(
            Mapping(
                world_size=tp,
                rank=rank,
                tp_size=tp,
                pp_size=1,
                enable_attention_dp=enable_dp,
            )
        )
        for rank in range(tp)
    ]


def _unique_physical_pools(manager: DSACacheManagerV2) -> dict[int, int]:
    page_table = KVRegionExtractorV1(manager).page_table
    pools: dict[int, int] = {}
    for pool_group in page_table.pool_groups:
        for pool in pool_group.pools:
            pools[pool.base_address] = max(
                pools.get(pool.base_address, 0), get_pool_bytes(pool)
            )
    return pools


def _initialize_cache(
    managers: Sequence[DSACacheManagerV2],
    _tp: int,
    *,
    seed_base: int = 0,
    fill_random: bool = True,
) -> None:
    for rank, manager in enumerate(managers):
        for pool_index, (base_address, size) in enumerate(
            sorted(_unique_physical_pools(manager).items())
        ):
            pool = convert_to_torch_tensor(
                TensorWrapper(base_address, DataType.INT8, [size])
            )
            if fill_random:
                generator = torch.Generator(device=pool.device).manual_seed(
                    seed_base + 97 * rank + pool_index
                )
                pool.copy_(
                    torch.randint(
                        -128,
                        128,
                        pool.shape,
                        dtype=torch.int8,
                        device=pool.device,
                        generator=generator,
                    )
                )
            else:
                pool.zero_()


def _layer_regions(
    manager: DSACacheManagerV2, layer_idx: int
) -> dict[frozenset[str], _LayerRegion]:
    page_table = KVRegionExtractorV1(manager).page_table
    local_layer_idx = manager.layer_offsets[layer_idx]
    regions: dict[frozenset[str], _LayerRegion] = {}
    for layer_group_idx, layer_group in enumerate(page_table.layer_groups):
        if not any(
            local_layer.global_layer_id == layer_idx
            for local_layer in layer_group.local_layers
        ):
            continue
        for pool_view in layer_group.pool_views:
            entries = [
                entry
                for entry in pool_view.buffer_entries
                if int(entry["local_layer_id"]) == local_layer_idx
            ]
            if not entries:
                continue
            assert len(entries) == 1
            entry = entries[0]
            pool = get_physical_pool(page_table, layer_group_idx, pool_view.pool_idx)
            regions[pool_view.pool_role] = _LayerRegion(
                base_address=pool.base_address,
                slot_bytes=pool.slot_bytes,
                offset=int(entry["offset"]),
                size=int(entry["size"]),
            )
    return regions


def _read_region(
    manager: DSACacheManagerV2,
    region: _LayerRegion,
    request_id: int,
    num_blocks: int,
) -> torch.Tensor:
    slots = manager.get_pool_block_indices(
        1,
        request_ids=[request_id],
    ).flatten()[:num_blocks].to(torch.long)
    pool_bytes = _unique_physical_pools(manager)[region.base_address]
    assert pool_bytes % region.slot_bytes == 0
    pool = convert_to_torch_tensor(
        TensorWrapper(
            region.base_address,
            DataType.INT8,
            [pool_bytes // region.slot_bytes, region.slot_bytes],
        )
    )
    return pool[slots, region.offset : region.offset + region.size]


def _verify_cache(
    request_lengths: list[int],
    ctx_managers: Sequence[DSACacheManagerV2],
    gen_managers: Sequence[DSACacheManagerV2],
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    ctx_request_ids: list[int],
    gen_request_ids: list[int],
) -> None:
    assert (ctx_tp, ctx_pp, ctx_enable_dp) == (4, 1, True)
    assert (gen_tp, gen_pp, gen_enable_dp) == (8, 1, False)

    for req_idx, request_length in enumerate(request_lengths):
        ctx_manager = ctx_managers[req_idx % ctx_tp]
        num_blocks = math.ceil(request_length / TOKENS_PER_BLOCK)
        for gen_manager in gen_managers:
            assert gen_manager.indexer_k_cache_local_layer_mask == EXPECTED_INDEXER_MASK
            for layer_idx in gen_manager.pp_layers:
                ctx_regions = _layer_regions(ctx_manager, layer_idx)
                gen_regions = _layer_regions(gen_manager, layer_idx)
                assert gen_regions.keys() == ctx_regions.keys()
                for role, gen_region in gen_regions.items():
                    ctx_region = ctx_regions[role]
                    assert gen_region.size == ctx_region.size
                    actual = _read_region(
                        gen_manager,
                        gen_region,
                        gen_request_ids[req_idx],
                        num_blocks,
                    )
                    expected = _read_region(
                        ctx_manager,
                        ctx_region,
                        ctx_request_ids[req_idx],
                        num_blocks,
                    )
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=0,
                        atol=0,
                        msg=lambda message: (
                            f"GLM-5.2 NVFP4 transfer mismatch for request {req_idx}, "
                            f"layer {layer_idx}, role {sorted(role)}: {message}"
                        ),
                    )


def test_glm52_nvfp4_page_table_geometry() -> None:
    manager = _create_manager(
        Mapping(
            world_size=4,
            rank=0,
            tp_size=4,
            pp_size=1,
            enable_attention_dp=True,
        )
    )
    try:
        assert manager.mla_kv_cache_residual_dim == QK_ROPE_HEAD_DIM
        assert manager.indexer_k_cache_local_layer_mask == EXPECTED_INDEXER_MASK
        assert manager.get_disagg_role_mapper_kinds() == {
            Role.ALL: MapperKind.INDEXED,
            Role.INDEX_KEY: MapperKind.REPLICATED,
        }
        full_regions = _layer_regions(manager, 0)
        shared_regions = _layer_regions(manager, 3)
        assert {next(iter(role)) for role in full_regions} == {
            "key",
            "key_block_scale",
            "index_key",
        }
        assert {next(iter(role)) for role in shared_regions} == {
            "key",
            "key_block_scale",
        }
        assert (
            next(
                region.size for role, region in full_regions.items() if role == {"key"}
            )
            == (HEAD_DIM + QK_ROPE_HEAD_DIM) // 2 * TOKENS_PER_BLOCK
        )
        assert (
            next(
                region.size
                for role, region in full_regions.items()
                if role == {"key_block_scale"}
            )
            == (HEAD_DIM + QK_ROPE_HEAD_DIM) // 16 * TOKENS_PER_BLOCK
        )
    finally:
        manager.shutdown()


@pytest.mark.cuda
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "update_before_transfer",
    [True, False],
    ids=["update_before", "update_after"],
)
def test_glm52_nvfp4_kv_transfer(update_before_transfer: bool) -> None:
    transfer_harness.run_kv_transfer_test(
        ctx_tp=4,
        ctx_pp=1,
        gen_tp=8,
        gen_pp=1,
        ctx_enable_dp=True,
        gen_enable_dp=False,
        update_before_transfer=update_before_transfer,
        manager_factory=_create_managers,
        init_fn=_initialize_cache,
        verify_fn=_verify_cache,
    )
