"""Process setup that has to happen before GStreamer loads."""

from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path


def libpython_dylib() -> Path | None:
    """Return pyenv's ``libpythonX.Y.dylib``, which dyld does not search."""
    matches = sorted((Path(sys.base_prefix) / "lib").glob("libpython*.dylib"))
    return matches[0] if matches else None


def gstreamer_library_dir() -> Path:
    """Directory named by the Reachy camera plugin's rpath."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        Path(sys.prefix)
        / "lib"
        / f"python{version}"
        / "site-packages"
        / "gstreamer_libs"
        / "lib"
    )


def expose_libpython(library_dir: Path | None = None) -> Path | None:
    """Let the camera plugin load ``libpython`` without ``DYLD_LIBRARY_PATH``.

    ``libgstpython`` asks dyld for ``@rpath/libpythonX.Y.dylib``. On a pyenv
    build that library lives in ``<prefix>/lib``, and macOS drops
    ``DYLD_LIBRARY_PATH`` for the GStreamer plugin scanner. The plugin
    already searches ``gstreamer_libs/lib``, so a symlink there is enough.
    """
    source = libpython_dylib()
    if source is None:
        return None
    dest_dir = gstreamer_library_dir() if library_dir is None else library_dir
    if not dest_dir.is_dir():
        return None
    dest = dest_dir / source.name
    if not dest.exists() and not dest.is_symlink():
        dest.symlink_to(source)
    return dest


def install_force_stop() -> None:
    """Make Ctrl+C end the process while the camera plugin is loading.

    GStreamer blocks inside native code and keeps its own threads alive.
    Python only turns SIGINT into ``KeyboardInterrupt`` between bytecode
    instructions, so Ctrl+C can print a traceback and still leave the
    process running. Waiting for SIGINT on a side thread exits immediately.
    """
    if getattr(install_force_stop, "_installed", False):
        return
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})

    def _wait() -> None:
        signal.sigwait({signal.SIGINT})
        os._exit(130)

    threading.Thread(target=_wait, name="reachy-lego-sigint", daemon=True).start()
    install_force_stop._installed = True
