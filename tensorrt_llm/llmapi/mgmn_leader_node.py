'''
This script is used to start the MPICommSession in the rank0 and wait for the
MPI Proxy process to connect and get the MPI task to run.
'''
from typing import Literal

import os
import sys
import signal
import faulthandler
import threading
import time

import click
import zmq

from tensorrt_llm._utils import global_mpi_rank, mpi_world_size
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.executor.utils import get_spawn_proxy_process_ipc_addr_env
from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer
from tensorrt_llm.llmapi.utils import print_colored_debug

# -------------------- faulthandler setup --------------------
# Enable faulthandler (helps on fatal errors)
faulthandler.enable(file=sys.stderr, all_threads=True)

# Register SIGUSR1 to dump all thread stacks on demand: `kill -USR1 <pid>`
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception:
    # Some platforms/container policies may forbid signal registration; ignore.
    pass

# Nice banner so you can see which PID/rank is armed
try:
    _rank = global_mpi_rank()
except Exception:
    _rank = 0
print_colored_debug(f"[faulthandler] armed (rank={_rank}, pid={os.getpid()})\n", "yellow")
# -----------------------------------------------------------


def _start_periodic_dumps(every_seconds: int):
    """Periodically dump stacks. Uses faulthandler.dump_traceback_later(repeat=True)."""
    if every_seconds > 0:
        # dump_traceback_later is global; calling once is enough.
        faulthandler.dump_traceback_later(every_seconds, repeat=True)
        print_colored_debug(f"[faulthandler] periodic dump every {every_seconds}s enabled\n", "yellow")


def launch_server_main(sub_comm=None):
    num_ranks = sub_comm.Get_size() if sub_comm is not None else mpi_world_size()
    print_colored_debug(f"Starting MPI Comm Server with {num_ranks} workers\n", "yellow")
    server = RemoteMpiCommSessionServer(
        comm=sub_comm,
        n_workers=num_ranks,
        addr=get_spawn_proxy_process_ipc_addr_env(),
        is_comm=True,
    )
    print_colored_debug(
        f"MPI Comm Server started at {get_spawn_proxy_process_ipc_addr_env()}\n", "green"
    )
    try:
        server.serve()
    finally:
        print_colored_debug("RemoteMpiCommSessionServer stopped\n", "yellow")


def stop_server_main():
    queue = ZeroMqQueue(
        (get_spawn_proxy_process_ipc_addr_env(), None),
        use_hmac_encryption=False,
        is_server=False,
        socket_type=zmq.PAIR,
    )
    try:
        print_colored_debug(
            f"RemoteMpiCommSessionClient [rank{global_mpi_rank()}] send shutdown signal to server\n",
            "green",
        )
        queue.put(None)  # ask RemoteMpiCommSessionServer to shutdown
    except zmq.error.ZMQError as e:
        print_colored_debug(
            f"Error during RemoteMpiCommSessionClient shutdown: {e}\n", "red"
        )


@click.command()
@click.option("--action", type=click.Choice(["start", "stop"]), default="start")
@click.option(
    "--dump-interval",
    type=int,
    default=60,
    show_default=True,
    help="If >0, periodically dump Python stack traces every N seconds.",
)
def main(action: Literal["start", "stop"] = "start", dump_interval: int = 60):
    '''
    Arguments:
        action: The action to perform.
            start: Start the MPI Comm Server.
            stop: Stop the MPI Comm Server.
        dump-interval: If >0, periodically dump stacks every N seconds.
    '''
    _start_periodic_dumps(dump_interval)

    if action == "start":
        launch_server_main()
    elif action == "stop":
        stop_server_main()
    else:
        raise ValueError(f"Invalid action: {action}")


if __name__ == '__main__':
    main()
