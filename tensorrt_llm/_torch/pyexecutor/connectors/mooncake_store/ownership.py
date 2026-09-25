# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exclusive residency and read reservations for a single Mooncake offload pool.

The caller serializes operations. Store values are hard pinned: only this
controller may remove them, after all admitted reads and publications finish.
GPU copies are identified by worker lifetime, not by reusable rank numbers.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

__all__ = ["OffloadDirectory", "RemovalStore"]


class RemovalStore(Protocol):
    def batch_remove(self, keys: list[str], force: bool = False) -> list[int]: ...


@dataclass
class _Page:
    size: int = 0
    owners: set[str] = field(default_factory=set)
    readers: set[str] = field(default_factory=set)
    cpu: bool = False
    transition: str = ""


@dataclass
class _Eviction:
    owner: str
    keys: list[str]
    writes: list[str]


class OffloadDirectory:
    """Track an exclusive cache; unsuccessful transitions retain GPU ownership."""

    def __init__(self, store: RemovalStore, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("offload capacity must be positive")
        self.epoch = uuid4().hex
        self._store = store
        self._capacity = capacity_bytes
        self._pages: dict[str, _Page] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._leases: dict[str, tuple[str, set[str]]] = {}
        self._evictions: dict[str, _Eviction] = {}
        self._active_reads: dict[tuple[str, str], str] = {}
        self._clients: set[str] = set()
        self._used = 0
        self._reserved = 0
        self._poisoned = False
        self._evicted = 0
        self._promoted = 0

    def check(self, epoch: str) -> None:
        if self._poisoned or epoch != self.epoch:
            raise RuntimeError("offload ownership lost; restart the pool and its workers")

    def register(self, owner: str) -> str:
        if self._poisoned or not owner or owner in self._clients:
            raise RuntimeError("offload client registration rejected")
        self._clients.add(owner)
        return self.epoch

    def disconnected(self, owner: str) -> None:
        # A disconnected owner may still be serving GPU data. Expiring it would
        # permit a CPU publication underneath that live GPU copy.
        if owner in self._clients:
            self._poisoned = True

    def _client(self, owner: str) -> None:
        self.check(self.epoch)
        if owner not in self._clients:
            raise RuntimeError("unknown offload worker lifetime")

    def _delete_cpu(self, keys: list[str]) -> None:
        if not keys:
            return
        # All connector readers have completed before this call. Native GET
        # leases can outlive the synchronous copies, so force bypasses their
        # residual TTL, not an in-flight connector transfer.
        results = self._store.batch_remove(keys, force=True)
        if len(results) != len(keys) or any(result != 0 for result in results):
            self._poisoned = True
            raise RuntimeError("offload CPU removal failed; ownership is fenced")
        for key in keys:
            page = self._pages[key]
            self._used -= page.size
            page.cpu = False
            self._lru.pop(key, None)

    def _refresh_many(self, keys: list[str]) -> None:
        present = list(dict.fromkeys(key for key in keys if key in self._pages))
        retire = [
            key
            for key in present
            if (
                self._pages[key].cpu
                and self._pages[key].owners
                and not self._pages[key].readers
                and not self._pages[key].transition
            )
        ]
        self._delete_cpu(retire)
        for key in present:
            page = self._pages[key]
            if page.cpu and not page.owners and not page.readers and not page.transition:
                self._lru[key] = None
            else:
                self._lru.pop(key, None)
            if not page.cpu and not page.owners and not page.readers and not page.transition:
                del self._pages[key]

    def reserve_prefix(self, owner: str, blocks: list[list[str]]) -> tuple[str, int]:
        """Atomically reserve a fully available prefix before scheduler admission."""
        self._client(owner)
        selected: set[str] = set()
        count = 0
        for block in blocks:
            if not block or any(
                key not in self._pages
                or not self._pages[key].cpu
                or self._pages[key].owners
                or self._pages[key].transition
                for key in block
            ):
                break
            selected.update(block)
            count += 1
        if not selected:
            return "", 0
        lease = uuid4().hex
        self._leases[lease] = (owner, selected)
        for key in selected:
            self._pages[key].readers.add(lease)
            self._lru.pop(key, None)
        return lease, count

    def release_read(self, owner: str, lease: str, keys: list[str] | None = None) -> None:
        """Release an unused offer; loaded shards consume their own keys in claim."""
        self._client(owner)
        if lease not in self._leases:
            return
        lease_owner, held = self._leases[lease]
        if lease_owner != owner:
            raise RuntimeError("only the scheduling owner may cancel a read")
        candidates = set(held) if keys is None else set(keys)
        self._consume(lease, {key for key in candidates if (lease, key) not in self._active_reads})

    def start_read(self, owner: str, lease: str, keys: list[str]) -> None:
        """Make selected shards non-cancellable before starting their DMA."""
        self._client(owner)
        held = self._leases.get(lease)
        if held is None or not set(keys) <= held[1]:
            raise RuntimeError("read reservation was cancelled before loading")
        if any((lease, key) in self._active_reads for key in keys):
            raise RuntimeError("read reservation is already being consumed")
        for key in keys:
            self._active_reads[(lease, key)] = owner

    def _consume(self, lease: str, keys: set[str]) -> None:
        _, held = self._leases[lease]
        consumed = keys & held
        for key in consumed:
            self._active_reads.pop((lease, key), None)
            self._pages[key].readers.remove(lease)
        held.difference_update(keys)
        self._refresh_many(list(consumed))
        if not held:
            del self._leases[lease]

    def claim(
        self, owner: str, keys: list[str], lease: str = "", sizes: list[int] | None = None
    ) -> bool:
        """Publish GPU ownership after bytes are ready; retire unneeded CPU copies."""
        self._client(owner)
        if sizes is not None and (len(sizes) != len(keys) or any(size <= 0 for size in sizes)):
            raise ValueError("GPU claims require one positive size per key")
        if sizes is not None:
            for key, size in zip(keys, sizes):
                page = self._pages.get(key)
                if page is not None and page.size not in (0, size):
                    raise ValueError("page size changed within one content namespace")
        if any(self._pages.get(key, _Page()).transition for key in keys):
            return False
        if lease:
            held = self._leases.get(lease)
            if (
                held is None
                or not set(keys) <= held[1]
                or any(self._active_reads.get((lease, key)) != owner for key in keys)
            ):
                raise RuntimeError("promotion has no matching active read reservation")
        for index, key in enumerate(keys):
            page = self._pages.setdefault(key, _Page())
            if sizes is not None:
                page.size = sizes[index]
            page.owners.add(owner)
            self._lru.pop(key, None)
        if lease:
            self._promoted += len(set(keys))
            self._consume(lease, set(keys))
        else:
            self._refresh_many(keys)
        return True

    def prepare_eviction(
        self, owner: str, keys: list[str], sizes: list[int]
    ) -> tuple[str, list[str]]:
        """Reserve CPU space and serialize publications against promotions/evictions."""
        self._client(owner)
        if len(keys) != len(sizes) or len(set(keys)) != len(keys) or any(s <= 0 for s in sizes):
            raise ValueError("eviction requires distinct keys and positive page sizes")
        for key in keys:
            page = self._pages.get(key)
            if page is None or owner not in page.owners:
                raise RuntimeError("evicting a page without GPU ownership")
            if page.transition:
                return "", []
        writes = []
        needed = 0
        for key, size in zip(keys, sizes):
            page = self._pages[key]
            if page.size and page.size != size:
                raise ValueError("page size changed within one content namespace")
            page.size = size
            if page.owners == {owner} and not page.cpu:
                writes.append(key)
                needed += size
        victims: list[str] = []
        reclaimed = 0
        for key in self._lru:
            if self._used + self._reserved + needed - reclaimed <= self._capacity:
                break
            victims.append(key)
            reclaimed += self._pages[key].size
        if self._used + self._reserved + needed - reclaimed > self._capacity:
            raise RuntimeError("offload pool is full of reserved pages; retaining GPU slots")
        self._delete_cpu(victims)
        self._evicted += len(victims)
        self._refresh_many(victims)
        token = uuid4().hex
        self._evictions[token] = _Eviction(owner, keys, writes)
        self._reserved += needed
        for key in keys:
            self._pages[key].transition = token
            self._lru.pop(key, None)
        return token, writes

    def finish_eviction(self, owner: str, token: str, success: bool) -> None:
        """Commit only at slot release; abort after revoking partial publications."""
        self._client(owner)
        transition = self._evictions[token]
        if transition.owner != owner:
            raise RuntimeError("eviction belongs to a different worker")
        if success:
            for key in transition.writes:
                page = self._pages[key]
                page.cpu = True
                self._used += page.size
        self._reserved -= sum(self._pages[key].size for key in transition.writes)
        for key in transition.keys:
            page = self._pages[key]
            page.transition = ""
            if success:
                page.owners.remove(owner)
        self._refresh_many(transition.keys)
        del self._evictions[token]

    def forget(self, owner: str, keys: list[str]) -> None:
        """Discard uncommitted/cancelled GPU pages without publishing them."""
        self._client(owner)
        for key in keys:
            page = self._pages.get(key)
            if page is None:
                continue
            if page.transition:
                raise RuntimeError("cannot forget a page during an eviction transaction")
            page.owners.discard(owner)
        self._refresh_many(keys)

    def unregister(self, owner: str) -> None:
        self._client(owner)
        if any(t.owner == owner for t in self._evictions.values()):
            raise RuntimeError("cannot close a worker with unfinished publications")
        # Graceful worker close follows completion/failure of its synchronous
        # GETs, so its own active DMA reservations can now be retired.
        for lease in list(self._leases):
            keys = {
                key
                for (active_lease, key), reader in self._active_reads.items()
                if active_lease == lease and reader == owner
            }
            if keys:
                self._consume(lease, keys)
        for lease, (lease_owner, _) in list(self._leases.items()):
            if lease_owner == owner:
                self.release_read(owner, lease)
        self.forget(owner, [key for key, page in self._pages.items() if owner in page.owners])
        self._clients.remove(owner)

    def statistics(self) -> dict[str, int]:
        """Expose settled overlap separately from protected in-flight duplication."""
        return {
            "cpu_bytes": self._used,
            "reserved_bytes": self._reserved,
            "capacity_bytes": self._capacity,
            "gpu_keys": sum(bool(page.owners) for page in self._pages.values()),
            "cpu_keys": sum(page.cpu for page in self._pages.values()),
            "gpu_unique_bytes": sum(page.size for page in self._pages.values() if page.owners),
            "unique_bytes": sum(
                page.size for page in self._pages.values() if page.cpu or page.owners
            ),
            "overlap_bytes": sum(
                page.size for page in self._pages.values() if page.cpu and page.owners
            ),
            "overlap_keys": sum(page.cpu and bool(page.owners) for page in self._pages.values()),
            "settled_overlap_keys": sum(
                page.cpu and bool(page.owners) and not page.readers and not page.transition
                for page in self._pages.values()
            ),
            "read_leases": len(self._leases),
            "active_reads": len(self._active_reads),
            "transitions": len(self._evictions),
            "evicted_keys": self._evicted,
            "promoted_keys": self._promoted,
            "fenced": int(self._poisoned),
        }
