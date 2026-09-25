# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool-wide ownership service for exclusive Mooncake offloading.

Run one service per fresh offload namespace. An ungraceful worker disconnect
fences the service: losing GPU ownership must not silently enable CPU writes.
The protocol carries metadata only; KV transfers still use Mooncake RDMA.
"""

import argparse
import json
import socket
import socketserver
import threading
import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .ownership import OffloadDirectory

__all__ = ["OwnershipClient", "OwnershipServer"]
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal[
        "register",
        "reserve",
        "release",
        "start_read",
        "claim",
        "prepare",
        "finish",
        "forget",
        "close",
        "stats",
    ]
    owner: str = ""
    epoch: str = ""
    keys: list[str] = Field(default_factory=list)
    blocks: list[list[str]] = Field(default_factory=list)
    sizes: list[int] = Field(default_factory=list)
    lease: str = ""
    token: str = ""
    success: bool = False


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    error: str = ""
    epoch: str = ""
    lease: str = ""
    count: int = 0
    accepted: bool = True
    token: str = ""
    keys: list[str] = Field(default_factory=list)
    statistics: dict[str, int] = Field(default_factory=dict)


class OwnershipServer(socketserver.ThreadingTCPServer):
    """Serialize metadata transactions while workers transfer data independently."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], directory: OffloadDirectory) -> None:
        self.directory = directory
        self.lock = threading.Lock()
        super().__init__(address, _Handler)

    def dispatch(self, owner: str, request: _Request) -> _Reply:
        directory = self.directory
        directory.check(request.epoch)
        operation = request.operation
        if operation == "reserve":
            lease, count = directory.reserve_prefix(owner, request.blocks)
            return _Reply(lease=lease, count=count)
        if operation == "release":
            directory.release_read(owner, request.lease, request.keys or None)
        elif operation == "start_read":
            directory.start_read(owner, request.lease, request.keys)
        elif operation == "claim":
            return _Reply(
                accepted=directory.claim(owner, request.keys, request.lease, request.sizes or None)
            )
        elif operation == "prepare":
            token, keys = directory.prepare_eviction(owner, request.keys, request.sizes)
            return _Reply(token=token, keys=keys, accepted=bool(token))
        elif operation == "finish":
            directory.finish_eviction(owner, request.token, request.success)
        elif operation == "forget":
            return _Reply(accepted=directory.forget(owner, request.keys))
        elif operation == "close":
            return _Reply(accepted=directory.unregister(owner))
        elif operation == "stats":
            return _Reply(statistics=directory.statistics())
        else:
            raise ValueError("operation requires a new connection")
        return _Reply()


class _Handler(socketserver.StreamRequestHandler):
    server: OwnershipServer

    def handle(self) -> None:
        owner = ""
        try:
            while True:
                raw = self.rfile.readline(_MAX_MESSAGE_BYTES + 1)
                if not raw:
                    return
                if len(raw) > _MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
                    raise ValueError("oversized ownership message")
                closing = False
                try:
                    request = _Request.model_validate_json(raw)
                    with self.server.lock:
                        if not owner:
                            if request.operation != "register":
                                raise ValueError("register before sending ownership operations")
                            epoch = self.server.directory.register(request.owner)
                            owner = request.owner
                            reply = _Reply(epoch=epoch)
                        else:
                            reply = self.server.dispatch(owner, request)
                    if request.operation == "close" and reply.accepted:
                        owner = ""
                        closing = True
                except (ValidationError, ValueError, RuntimeError, KeyError) as exc:
                    reply = _Reply(error=str(exc))
                self.wfile.write(reply.model_dump_json().encode() + b"\n")
                self.wfile.flush()
                if closing:
                    return
        except (OSError, ValueError):
            return
        finally:
            with self.server.lock:
                self.server.directory.disconnected(owner)


class OwnershipClient:
    """One persistent, fail-closed connection per worker lifetime."""

    def __init__(self, address: str, timeout: float = 60.0) -> None:
        host, _, port = address.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError("offload coordinator address must be host:port")
        self._timeout = timeout
        self._socket = socket.create_connection((host.strip("[]"), int(port)), timeout)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._stream = self._socket.makefile("rwb")
        self._lock = threading.Lock()
        self._closed = False
        self._epoch = ""
        self._epoch = self._call(_Request(operation="register", owner=uuid4().hex)).epoch

    def _call(self, request: _Request) -> _Reply:
        with self._lock:
            if self._closed:
                raise RuntimeError("offload coordinator connection is closed")
            request.epoch = self._epoch
            self._stream.write(request.model_dump_json().encode() + b"\n")
            self._stream.flush()
            raw = self._stream.readline(_MAX_MESSAGE_BYTES + 1)
            if not raw or len(raw) > _MAX_MESSAGE_BYTES:
                raise RuntimeError("offload coordinator connection lost")
            reply = _Reply.model_validate_json(raw)
            if reply.error:
                raise RuntimeError(reply.error)
            return reply

    def _until_accepted(self, request: _Request) -> _Reply:
        deadline = time.monotonic() + self._timeout
        while True:
            reply = self._call(request)
            if reply.accepted:
                return reply
            if time.monotonic() >= deadline:
                raise TimeoutError("offload ownership transition did not complete")
            time.sleep(0.001)

    def reserve_prefix(self, blocks: list[list[str]]) -> tuple[str, int]:
        reply = self._call(_Request(operation="reserve", blocks=blocks))
        return reply.lease, reply.count

    def release_read(self, lease: str, keys: list[str] | None = None) -> None:
        if lease:
            self._call(_Request(operation="release", lease=lease, keys=keys or []))

    @property
    def epoch(self) -> str:
        return self._epoch

    def start_read(self, lease: str, keys: list[str]) -> None:
        self._call(_Request(operation="start_read", lease=lease, keys=keys))

    def claim(self, keys: list[str], lease: str = "", sizes: list[int] | None = None) -> None:
        if keys:
            self._until_accepted(
                _Request(operation="claim", keys=keys, lease=lease, sizes=sizes or [])
            )

    def prepare_eviction(self, keys: list[str], sizes: list[int]) -> tuple[str, list[str]]:
        reply = self._until_accepted(_Request(operation="prepare", keys=keys, sizes=sizes))
        return reply.token, reply.keys

    def finish_eviction(self, token: str, success: bool) -> None:
        self._call(_Request(operation="finish", token=token, success=success))

    def forget(self, keys: list[str]) -> None:
        if keys:
            self._until_accepted(_Request(operation="forget", keys=keys))

    def statistics(self) -> dict[str, int]:
        return self._call(_Request(operation="stats")).statistics

    def close(self) -> None:
        if not self._closed:
            try:
                self._until_accepted(_Request(operation="close"))
                if self._stream.read(1):
                    raise RuntimeError("offload coordinator did not close its worker session")
            finally:
                self._closed = True
                self._stream.close()
                self._socket.close()


def main() -> None:
    """Start an ownership service with an explicitly bounded, fresh CPU pool."""
    from .config import MooncakeStoreConnectorConfig, parse_size
    from .worker import _open_store

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Mooncake connection JSON")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50072)
    parser.add_argument("--capacity", required=True, help="CPU cache byte budget, e.g. 5TiB")
    args = parser.parse_args()
    from dataclasses import replace

    config = replace(
        MooncakeStoreConnectorConfig.from_file(args.config),
        global_segment_size=0,
        local_buffer_size=16 * 1024 * 1024,
    )
    store = _open_store(config)
    directory = OffloadDirectory(store, parse_size(args.capacity))
    print(json.dumps({"ownership_epoch": directory.epoch, "capacity": args.capacity}), flush=True)
    with OwnershipServer((args.host, args.port), directory) as server:
        try:
            server.serve_forever()
        finally:
            print(json.dumps(directory.statistics()), flush=True)
            store.close()


if __name__ == "__main__":
    main()
