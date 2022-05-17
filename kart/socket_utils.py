import logging

import time
import array
import socket

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)-15s pid=%(process)d %(side)s: %(message)s", level=logging.INFO
)

socket_filename = "frogs.sock"

# Function from https://docs.python.org/3/library/socket.html#socket.socket.recvmsg
def recv_fds(sock, msglen, maxfds):
    fds = array.array("i")  # Array of ints
    msg, ancdata, flags, addr = sock.recvmsg(
        msglen, socket.CMSG_LEN(maxfds * fds.itemsize)
    )
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
    return msg, list(fds)


# Function from https://docs.python.org/3/library/socket.html#socket.socket.sendmsg
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
            time.sleep(0.5)
            pass
    return False
