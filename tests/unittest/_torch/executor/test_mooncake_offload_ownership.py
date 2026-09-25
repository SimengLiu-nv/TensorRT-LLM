# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exclusive-tier invariants, reservations, and worker lifetime fencing."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.ownership import OffloadDirectory
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.ownership_server import (
    OwnershipClient,
    OwnershipServer,
)


class MemoryStore:
    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.removed: list[str] = []
        self.fail_remove = False

    def batch_remove(self, keys: list[str], force: bool = False) -> list[int]:
        assert force
        if self.fail_remove:
            return [-1] * len(keys)
        assert set(keys) <= self.keys
        self.keys.difference_update(keys)
        self.removed.extend(keys)
        return [0] * len(keys)


def make_directory(capacity: int = 4) -> tuple[MemoryStore, OffloadDirectory]:
    store = MemoryStore()
    directory = OffloadDirectory(store, capacity)
    for owner in ("a", "b", "c"):
        directory.register(owner)
    return store, directory


def demote(store: MemoryStore, directory: OffloadDirectory, owner: str, keys: list[str]) -> None:
    token, writes = directory.prepare_eviction(owner, keys, [1] * len(keys))
    store.keys.update(writes)
    directory.finish_eviction(owner, token, True)


def test_restore_moves_cpu_copy_after_last_reader_finishes() -> None:
    store, directory = make_directory()
    assert directory.claim("a", ["k"])
    demote(store, directory, "a", ["k"])
    lease_a, count_a = directory.reserve_prefix("a", [["k"]])
    lease_b, count_b = directory.reserve_prefix("b", [["k"]])
    assert count_a == count_b == 1
    directory.start_read("a", lease_a, ["k"])
    assert directory.claim("a", ["k"], lease_a)
    assert "k" in store.keys
    assert directory.statistics()["settled_overlap_keys"] == 0
    assert directory.reserve_prefix("c", [["k"]]) == ("", 0)
    directory.start_read("b", lease_b, ["k"])
    assert directory.claim("b", ["k"], lease_b)
    assert "k" not in store.keys
    assert directory.statistics()["overlap_keys"] == 0
    # Only the final GPU owner publishes; the first cannot create a duplicate.
    demote(store, directory, "a", ["k"])
    assert "k" not in store.keys
    demote(store, directory, "b", ["k"])
    assert store.keys == {"k"}


def test_reserved_prefix_survives_pressure_and_partial_cancellation() -> None:
    store, directory = make_directory(capacity=2)
    directory.claim("a", ["first", "tail", "new"])
    demote(store, directory, "a", ["first", "tail"])
    lease, count = directory.reserve_prefix("b", [["first"], ["tail"], ["missing"]])
    assert count == 2
    with pytest.raises(RuntimeError, match="full of reserved"):
        directory.prepare_eviction("a", ["new"], [1])
    assert store.keys == {"first", "tail"}
    directory.release_read("b", lease, ["tail"])
    demote(store, directory, "a", ["new"])
    assert store.keys == {"first", "new"}
    directory.start_read("b", lease, ["first"])
    directory.claim("b", ["first"], lease)
    assert store.keys == {"new"}
    assert directory.statistics()["read_leases"] == 0


def test_cpu_plus_gpu_working_set_exceeds_cpu_capacity_without_overlap() -> None:
    store, directory = make_directory(capacity=2)
    directory.claim("a", ["a", "b", "c", "d"], sizes=[1, 1, 1, 1])
    demote(store, directory, "a", ["a", "b"])
    stats = directory.statistics()
    assert stats["cpu_keys"] + stats["gpu_keys"] == 4
    assert stats["cpu_bytes"] == 2
    assert stats["unique_bytes"] == 4
    assert stats["overlap_bytes"] == 0
    assert stats["overlap_keys"] == 0
    lease, count = directory.reserve_prefix("b", [["a"]])
    assert count == 1
    directory.start_read("b", lease, ["a"])
    directory.claim("b", ["a"], lease)
    demote(store, directory, "a", ["c"])
    stats = directory.statistics()
    assert stats["cpu_keys"] + stats["gpu_keys"] == 4
    assert stats["evicted_keys"] == 0
    assert stats["settled_overlap_keys"] == 0


def test_publication_blocks_other_ownership_changes_until_commit_or_abort() -> None:
    store, directory = make_directory()
    directory.claim("a", ["k"])
    token, writes = directory.prepare_eviction("a", ["k"], [1])
    assert writes == ["k"]
    assert not directory.claim("b", ["k"])
    assert directory.reserve_prefix("b", [["k"]]) == ("", 0)
    assert directory.prepare_eviction("a", ["k"], [1]) == ("", [])
    # Worker revokes any partial writes before aborting this transaction.
    directory.finish_eviction("a", token, False)
    assert directory.statistics()["gpu_keys"] == 1
    assert directory.statistics()["reserved_bytes"] == 0
    assert directory.claim("b", ["k"])
    demote(store, directory, "a", ["k"])
    assert not store.keys
    demote(store, directory, "b", ["k"])
    assert store.keys == {"k"}


def test_computed_copy_retires_cpu_after_existing_reader_cancels() -> None:
    store, directory = make_directory()
    directory.claim("a", ["k"])
    demote(store, directory, "a", ["k"])
    lease, _ = directory.reserve_prefix("b", [["k"]])
    directory.claim("c", ["k"])
    assert store.keys == {"k"}
    directory.release_read("b", lease)
    assert not store.keys
    assert directory.statistics()["overlap_keys"] == 0


def test_read_lease_covers_all_shards_and_layer_groups() -> None:
    store, directory = make_directory(capacity=8)
    keys = ["r0g0", "r0g1", "r1g0", "r1g1"]
    directory.claim("a", keys)
    demote(store, directory, "a", keys)
    lease, count = directory.reserve_prefix("a", [keys])
    assert count == 1
    directory.start_read("a", lease, keys[:2])
    directory.claim("a", keys[:2], lease)
    assert store.keys == set(keys[2:])
    assert directory.statistics()["read_leases"] == 1
    directory.start_read("b", lease, keys[2:])
    directory.claim("b", keys[2:], lease)
    assert not store.keys
    assert directory.statistics()["read_leases"] == 0


def test_failed_removal_fences_the_pool_instead_of_claiming_exclusivity() -> None:
    store, directory = make_directory()
    directory.claim("a", ["k"])
    demote(store, directory, "a", ["k"])
    store.fail_remove = True
    with pytest.raises(RuntimeError, match="removal failed"):
        directory.claim("b", ["k"])
    with pytest.raises(RuntimeError, match="ownership lost"):
        directory.reserve_prefix("c", [["k"]])


def test_worker_disconnect_and_epoch_change_fail_closed() -> None:
    _, directory = make_directory()
    with pytest.raises(RuntimeError, match="ownership lost"):
        directory.check("old-epoch")
    directory.disconnected("a")
    with pytest.raises(RuntimeError, match="ownership lost"):
        directory.claim("b", ["k"])


def test_graceful_close_releases_ownership_and_unused_reads() -> None:
    store, directory = make_directory()
    directory.claim("a", ["cpu", "gpu"])
    demote(store, directory, "a", ["cpu"])
    directory.reserve_prefix("a", [["cpu"]])
    directory.unregister("a")
    directory.disconnected("a")
    assert directory.statistics()["fenced"] == 0
    assert directory.statistics()["gpu_keys"] == 0
    assert directory.reserve_prefix("b", [["cpu"]])[1] == 1


def test_service_serializes_concurrent_owners() -> None:
    store = MemoryStore()
    directory = OffloadDirectory(store, 4)
    with OwnershipServer(("127.0.0.1", 0), directory) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        address = f"127.0.0.1:{server.server_address[1]}"
        clients = [OwnershipClient(address) for _ in range(2)]
        try:
            with ThreadPoolExecutor(2) as pool:
                list(pool.map(lambda client: client.claim(["k"]), clients))
            first, writes = clients[0].prepare_eviction(["k"], [1])
            assert not writes
            clients[0].finish_eviction(first, True)
            last, writes = clients[1].prepare_eviction(["k"], [1])
            assert writes == ["k"]
            store.keys.update(writes)
            clients[1].finish_eviction(last, True)
            lease, count = clients[0].reserve_prefix([["k"]])
            assert count == 1
            clients[0].start_read(lease, ["k"])
            clients[0].claim(["k"], lease)
            assert not store.keys
            assert clients[1].statistics()["settled_overlap_keys"] == 0
        finally:
            for client in clients:
                client.close()
            server.shutdown()
            thread.join()


def test_cancellation_cannot_retire_an_active_dma_read() -> None:
    store, directory = make_directory()
    directory.claim("a", ["active", "unused"])
    demote(store, directory, "a", ["active", "unused"])
    lease, _ = directory.reserve_prefix("a", [["active"], ["unused"]])
    directory.start_read("b", lease, ["active"])
    directory.claim("c", ["active", "unused"])
    directory.release_read("a", lease)
    assert store.keys == {"active"}
    with pytest.raises(RuntimeError, match="cancelled before loading"):
        directory.start_read("b", lease, ["unused"])
    directory.claim("b", ["active"], lease)
    assert not store.keys
    assert directory.statistics()["read_leases"] == 0
