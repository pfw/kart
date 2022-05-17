import json
from pathlib import Path

from .socket_utils import logger, send_fds, wait_for_socket
import sys
import os
import socket
import time
import signal


def entrypoint():
    socket_filename = "kart.socket"
    if not os.environ.get("USE_HELPER"):
        from .cli import _entrypoint as cli_entrypoint

        cli_entrypoint()
    else:
        if not os.path.exists(socket_filename):
            # there should be a better check to see if the helper is running, signal to pid maybe

            # start the helper in the background
            cmd = Path.cwd() / "kart"
            cmdline = [Path.cwd() / "kart", *sys.argv[1:]]
            os.spawnvpe(os.P_NOWAIT, cmd, cmdline, os.environ)

        sock = socket.socket(family=socket.AF_UNIX)
        if wait_for_socket(sock, socket_filename):

            # here we send a payload to the helper
            # it should include all the cli's environment and args
            # we also send fds to the helper which should include the current directory, stdout, etc. etc.

            payload = json.dumps(
                {"pid": os.getpid(), "argv": sys.argv, "environ": dict(os.environ)}
            )

            send_fds(
                sock,
                bytes(payload, encoding="ascii"),
                [
                    sys.stdin.fileno(),
                    sys.stdout.fileno(),
                    sys.stderr.fileno(),
                    os.open(os.getcwd(), os.O_RDONLY),
                ],
            )

            # the helper sends a signal to say it is done
            def handler(signum, frame):
                sys.exit()

            signal.signal(signal.SIGALRM, handler)

            # wait for some time to see if a result comes back
            time.sleep(20)
        else:
            print("no socket connection to helper after 10 attempts")


if __name__ == "__main__":
    entrypoint()
