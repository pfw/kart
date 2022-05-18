"""
Fast start up client to 'kart helper' mode.

To run with the helper:

cd build
PATH=./venv/bin:$PATH ./venv/bin/python -S ../kartcli/__main__.py

There is a socket created by default at ~/.kart.socket

The helper will time out and shutdown after 5min by default but killing it directly
is enough to restart everyhing. There is no need to remove the socket if the helper
has died.

Open questions...
- how to organise the caller into 'kart' command
- TODOs as below
- Should there be a log file to trace behaviour of client and helper?
"""
import time

s = time.time()
import marshal
import array
import sys
import os
import socket
import signal

# print(f"imports done [{(time.time() - s):.3f}]")


def send_fds(sock, msg, fds):
    return sock.sendmsg(
        [msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
    )


def wait_for_socket(sock, socket_filename):
    # Simple retry; helper may not have created the socket_filename yet.
    for _ in range(10):
        try:
            sock.connect(socket_filename)
            return True
        except OSError as e:
            time.sleep(0.25)
            pass
    return False


def entrypoint():

    # TODO - resolve where the socket should live and how the processes can find it
    #  for now just create in the home directory with user only permissions

    socket_filename = os.path.join(os.path.expanduser("~"), ".kart.socket")

    try:
        sys.argv.remove("--use-helper")
    except ValueError:
        pass

    sock = socket.socket(family=socket.AF_UNIX)
    try:
        sock.connect(socket_filename)
    except OSError:
        # start the helper in the background
        cmd = "kart"
        cmdline = [cmd, "helper", "--socket", socket_filename]
        os.environ.update(
            {
                "NO_CONFIGURE_PROCESS_CLEANUP": "1",  # TODO - need to figure out exactly process cleanup
            }
        )

        os.spawnvpe(os.P_NOWAIT, cmd, cmdline, os.environ)

        if not wait_for_socket(sock, socket_filename):
            print("no socket connection to helper after 10 attempts")
            sys.exit()

    # here we send a payload to the helper
    # it should include all the cli's environment and args
    # we also send fds to the helper which should include the current directory, stdout, etc. etc.
    payload = marshal.dumps(
        {"pid": os.getpid(), "argv": sys.argv, "environ": dict(os.environ)}
    )

    # TODO - check the payload is smaller than what the helper will call recvmsg with, currently 4000
    # print(f"sendfds [{(time.time() - s):.3f}]")

    send_fds(
        sock,
        payload,
        [
            sys.stdin.fileno(),
            sys.stdout.fileno(),
            sys.stderr.fileno(),
            os.open(os.getcwd(), os.O_RDONLY),
        ],
    )
    # print(f"done sendfds [{(time.time() - s):.3f}]")

    # the helper sends a signal to say it is done
    def handler(signum, frame):
        # print(f"exiting on signal {signum} [{(time.time() - s):.3f}]")
        sys.exit()

    signal.signal(signal.SIGALRM, handler)

    # wait for some time to see if a result comes back
    # TODO - could be configurable, an ext-run script might want to stay open and stream data
    time.sleep(60)
    print("no results back from helper within 60 seconds")


if __name__ == "__main__":
    entrypoint()
