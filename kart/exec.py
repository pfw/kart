import os
import subprocess
import sys

from . import is_windows
from .cli_util import tool_environment


def _kart_no_exec(cmd, args, env):
    # used in testing. This is pretty hackzy
    p = subprocess.run([cmd] + args[1:], capture_output=True, encoding="utf-8", env=env)
    sys.stdout.write(p.stdout)
    sys.stdout.flush()
    sys.stderr.write(p.stderr)
    sys.stderr.flush()
    sys.exit(p.returncode)


def execvpe(cmd, args, env):
    env = tool_environment(env)

    if "_KART_NO_EXEC" in os.environ:
        _kart_no_exec(cmd, args, env)
    elif is_windows:
        p = subprocess.run([cmd] + args[1:], env=env)
        sys.exit(p.returncode)
    else:  # Posix
        os.execvpe(cmd, args, env)


def execvp(cmd, args):
    return execvpe(cmd, args, os.environ)


def spawn_and_wait(cmd, args):
    """
    spawnvpe a process and wait for it to exit, this is required
    when in helper mode as execvpe overwrites the process so
    the caller can't be notified when the command is complete
    """
    env = tool_environment(os.environ)

    if "_KART_NO_EXEC" in os.environ:
        _kart_no_exec(cmd, args, env)
    elif is_windows:
        raise RuntimeError("Not on windows yet")
    else:  # Posix
        os.spawnvpe(os.P_WAIT, cmd, args, env)
