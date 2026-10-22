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
"""Unit tests for the mooncake_master_m3 and mooncake_donor serve commands.

Both commands hold a resource, a master process or a mounted segment, until a
signal ends them. The resource is a context manager that records what it was
given and whether it was released, and the signal arrives from the command's
first idle wait, the point at which everything it holds is held.

Runs without a Mooncake installation and without a GPU.
"""

import contextlib
import json
import os
import signal
import time
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors import mooncake_store_m3
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store_m3.config import CONFIG_PATH_ENV
from tensorrt_llm.commands import _telemetry
from tensorrt_llm.commands import mooncake_m3 as m3_commands
from tensorrt_llm.usage.config import UsageContext

pytestmark = pytest.mark.cpu_only

HANDLED_SIGNALS = [signal.SIGINT, signal.SIGTERM]
GIB = 1024**3


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No ambient pool config: these tests set what they mean to test."""
    for name in (CONFIG_PATH_ENV, "TRTLLM_MOONCAKE_RUN_DIR", "TRTLLM_MOONCAKE_MASTER_BINARY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """The commands install handlers process-wide, so put pytest's back."""
    installed = {number: signal.getsignal(number) for number in HANDLED_SIGNALS}
    yield
    for number, handler in installed.items():
        signal.signal(number, handler)


@pytest.fixture(autouse=True)
def terminal_outcomes(monkeypatch):
    """Collect what the shared boundary reports, without a usage session."""
    outcomes = []
    monkeypatch.setattr(
        _telemetry.usage, "report_exit", lambda outcome, **_kwargs: outcomes.append(outcome)
    )
    monkeypatch.setattr(_telemetry.usage, "apply_usage_session_config", lambda *a, **k: True)
    monkeypatch.setattr(_telemetry.usage, "set_lifecycle_phase", lambda *a, **k: None)
    monkeypatch.setattr(_telemetry.usage, "get_observed_signal", lambda: 0)
    monkeypatch.setattr(_telemetry.usage, "record_observed_signal", lambda *a, **k: None)
    return outcomes


class Resource:
    """Stands in for what a command holds; records how it was taken and released."""

    def __init__(self, held):
        self.held = held
        self.released = False
        self.taken_with = None

    @contextlib.contextmanager
    def holding(self, *args, **kwargs):
        self.taken_with = (args, kwargs)
        try:
            yield self.held
        finally:
            self.released = True


def run(*args: str) -> None:
    group = _telemetry.TelemetryGroup(
        name="trtllm-serve",
        telemetry_usage_context=UsageContext.CLI_SERVE,
        telemetry_component="server",
        commands={
            "mooncake_master_m3": m3_commands.mooncake_master_m3,
            "mooncake_donor": m3_commands.mooncake_donor,
        },
    )
    group.main(args=list(args), prog_name="trtllm-serve", standalone_mode=False)


def signal_on_first_sleep(monkeypatch, number: int, on_idle=None) -> None:
    """Deliver `number` from the commands' first idle wait."""
    state = {"sent": False}

    def sleep(_seconds):
        if not state["sent"]:
            state["sent"] = True
            if on_idle is not None:
                on_idle()
            os.kill(os.getpid(), number)

    monkeypatch.setattr(m3_commands, "time", SimpleNamespace(sleep=sleep, monotonic=time.monotonic))


# ---- mooncake_master_m3 ----


@pytest.fixture
def master(monkeypatch, tmp_path) -> Resource:
    resource = Resource(
        SimpleNamespace(
            address="127.0.0.1:50071",
            log_path=str(tmp_path / "mooncake_master.log"),
            process=SimpleNamespace(poll=lambda: None),
        )
    )
    monkeypatch.setattr(mooncake_store_m3, "running_master", resource.holding)
    return resource


@pytest.mark.parametrize("number", HANDLED_SIGNALS)
def test_master_signal_releases_the_master(master, monkeypatch, tmp_path, number):
    held_on_idle = []
    signal_on_first_sleep(monkeypatch, number, on_idle=lambda: held_on_idle.append(True))

    with pytest.raises(_telemetry.SignalExit) as stopped:
        run("mooncake_master_m3", "--run_dir", str(tmp_path))

    assert stopped.value.signal_number == number
    assert held_on_idle == [True]
    assert master.released


def test_master_options_reach_the_pool_it_runs(master, monkeypatch, tmp_path):
    """The harness's flags: ports, eviction ratio, address file and a wrapper binary."""
    signal_on_first_sleep(monkeypatch, signal.SIGTERM)
    address_file = tmp_path / "master.addr"

    with pytest.raises(_telemetry.SignalExit):
        run(
            "mooncake_master_m3",
            "--rpc_port",
            "50071",
            "--metrics_port",
            "9004",
            "--eviction_ratio",
            "0.03",
            "--address_file",
            str(address_file),
            "--run_dir",
            str(tmp_path),
            "--binary",
            "mooncake_master_shim.sh",
            "--heartbeat_seconds",
            "30",
        )

    (pool, run_dir), kwargs = master.taken_with
    assert pool.launch_master
    assert (pool.master_port, pool.master_metrics_port) == (50071, 9004)
    assert pool.master_eviction_ratio == pytest.approx(0.03)
    assert run_dir == str(tmp_path)
    assert kwargs == {"address_file": str(address_file), "binary": "mooncake_master_shim.sh"}


def test_master_run_dir_defaults_to_the_environment(master, monkeypatch, tmp_path):
    monkeypatch.setenv("TRTLLM_MOONCAKE_RUN_DIR", str(tmp_path))
    signal_on_first_sleep(monkeypatch, signal.SIGTERM)

    with pytest.raises(_telemetry.SignalExit):
        run("mooncake_master_m3")

    (_pool, run_dir), _kwargs = master.taken_with
    assert run_dir == str(tmp_path)


# ---- mooncake_donor ----


@pytest.fixture
def donor(monkeypatch) -> Resource:
    resource = Resource("10.0.0.5")
    monkeypatch.setattr(mooncake_store_m3, "donate_segment", resource.holding)
    # Reaching the master is not what these tests are about.
    monkeypatch.setattr(mooncake_store_m3, "wait_for_master", lambda _address: 0.0)
    return resource


@pytest.mark.parametrize("number", HANDLED_SIGNALS)
def test_donor_signal_withdraws_the_segment(donor, monkeypatch, number):
    signal_on_first_sleep(monkeypatch, number)

    with pytest.raises(_telemetry.SignalExit) as stopped:
        run("mooncake_donor", "--master_server_address", "10.0.0.1:50051")

    assert stopped.value.signal_number == number
    assert donor.released


def test_donor_lends_what_the_harness_asks_for(donor, monkeypatch, tmp_path):
    """The GEN-node donor: a published master address, a node-sized segment, a ready file."""
    address_file = tmp_path / "master.addr"
    address_file.write_text("10.0.0.9:50071\n")
    ready_file = tmp_path / "donor.ready"
    announced = []
    signal_on_first_sleep(
        monkeypatch, signal.SIGTERM, on_idle=lambda: announced.append(ready_file.read_text())
    )

    with pytest.raises(_telemetry.SignalExit):
        run(
            "mooncake_donor",
            "--master_server_address",
            f"file://{address_file}",
            "--segment_size",
            "640GiB",
            "--protocol",
            "rdma",
            "--device_name",
            "mlx5_0,mlx5_1",
            "--metadata_server",
            "P2PHANDSHAKE",
            "--ready_file",
            str(ready_file),
        )

    (master_address, segment_size), kwargs = donor.taken_with
    assert master_address == "10.0.0.9:50071"
    assert segment_size == 640 * GIB
    assert kwargs["protocol"] == "rdma"
    assert kwargs["device_name"] == "mlx5_0,mlx5_1"
    assert kwargs["metadata_server"] == "P2PHANDSHAKE"
    # The segment was mounted before the launcher was told.
    assert announced == [f"10.0.0.5 {640 * GIB}\n"]


def test_donor_reads_what_it_was_not_told_from_the_pool_config(donor, monkeypatch, tmp_path):
    config = tmp_path / "mooncake.json"
    config.write_text(
        json.dumps(
            {
                "master_server_address": "10.0.0.2:50051",
                "protocol": "tcp",
                "device_name": "",
                "metadata_server": "",
            }
        )
    )
    monkeypatch.setenv(CONFIG_PATH_ENV, str(config))
    signal_on_first_sleep(monkeypatch, signal.SIGTERM)

    with pytest.raises(_telemetry.SignalExit):
        run("mooncake_donor", "--segment_size", "1GiB")

    (master_address, segment_size), kwargs = donor.taken_with
    assert master_address == "10.0.0.2:50051"
    assert segment_size == GIB
    assert kwargs["protocol"] == "tcp"
    # An empty connstring is not a form setup accepts; the handshake is the default.
    assert kwargs["metadata_server"] == "P2PHANDSHAKE"


def test_donor_without_a_master_is_a_usage_error(donor):
    import click

    with pytest.raises(click.UsageError, match="No master to join"):
        run("mooncake_donor")
    assert donor.taken_with is None


def test_donor_reports_an_unreachable_master_before_mounting(donor, monkeypatch):
    def refuse(address):
        raise TimeoutError(f"The Mooncake master at {address} did not accept connections")

    monkeypatch.setattr(mooncake_store_m3, "wait_for_master", refuse)

    with pytest.raises(TimeoutError, match="10.0.0.1:50051"):
        run("mooncake_donor", "--master_server_address", "10.0.0.1:50051")
    assert donor.taken_with is None
