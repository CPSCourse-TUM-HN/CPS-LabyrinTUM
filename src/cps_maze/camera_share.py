"""Shared-memory frame transport so ONE process can hold the camera open and
every script reads frames from it instead of re-opening the device.

Motivation: on Windows the camera must be opened via MSMF to reach 120 fps
(see camera.py), but MSMF can take 10-30 s to open. Paying that per script
across a calibration/lab session is painful. `scripts/camera_server.py` opens
the device once and publishes the latest frame here; other scripts attach in
milliseconds via ``SharedFrameReader``.

Bonus: each frame carries the server's true capture timestamp (monotonic at
grab time), which is process-wide comparable on both Windows (QPC) and Linux
(CLOCK_MONOTONIC). That removes the "burst frame" artifact of reading the
device directly, where two buffered frames get near-identical read-time stamps
and manufacture a huge phantom velocity.

Layout of the single shared block (little-endian):
    offset  0  seq        uint64   seqlock counter (odd = write in progress)
    offset  8  magic      uint32   'CPSC'
    offset 12  width      int32
    offset 16  height     int32
    offset 20  channels   int32
    offset 24  timestamp  float64  monotonic capture time
    offset 64  frame      uint8[height*width*channels]  BGR, row-major
"""
from __future__ import annotations

import struct
from time import monotonic, sleep

import numpy as np

try:
    from multiprocessing import shared_memory
except Exception:  # pragma: no cover - very old Python
    shared_memory = None  # type: ignore

SHM_NAME = "cps_maze_camera"
_MAGIC = 0x43505343  # 'CPSC'
_HEADER_SIZE = 64
_SEQ_OFF = 0
_META_OFF = 8  # magic, width, height, channels
_TS_OFF = 24
_META_FMT = "<Iiii"  # magic, width, height, channels


def _frame_bytes(width: int, height: int, channels: int) -> int:
    return int(width) * int(height) * int(channels)


class SharedFrameWriter:
    """Server side: create the block and publish frames into it."""

    def __init__(self, width: int, height: int, channels: int = 3,
                 name: str = SHM_NAME):
        if shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory unavailable")
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        size = _HEADER_SIZE + _frame_bytes(width, height, channels)
        # Best-effort remove a stale block first: on POSIX unlink() frees it so
        # we recreate cleanly. On Windows unlink() is a no-op and the block
        # lingers until its creating process exits, so a crashed/zombie server
        # can leave one behind - in that case reuse it (below) instead of
        # crashing.
        try:
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name)  # reuse existing
            if self.shm.size < size:
                self.shm.close()
                raise RuntimeError(
                    "existing shared camera block is too small to reuse; "
                    "stop the other camera_server process first")
        self._seq = 0
        struct.pack_into(_META_FMT, self.shm.buf, _META_OFF,
                         _MAGIC, self.width, self.height, self.channels)
        struct.pack_into("<Q", self.shm.buf, _SEQ_OFF, 0)

    def write(self, frame: np.ndarray, timestamp_s: float | None = None) -> None:
        if frame.shape != (self.height, self.width, self.channels):
            raise ValueError(
                f"frame shape {frame.shape} != "
                f"({self.height}, {self.width}, {self.channels})")
        if timestamp_s is None:
            timestamp_s = monotonic()
        data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        # seqlock: odd during write, even when done, so a reader can detect a
        # torn read and retry.
        self._seq += 1
        struct.pack_into("<Q", self.shm.buf, _SEQ_OFF, self._seq)  # odd: writing
        self.shm.buf[_HEADER_SIZE:_HEADER_SIZE + len(data)] = data
        struct.pack_into("<d", self.shm.buf, _TS_OFF, float(timestamp_s))
        self._seq += 1
        struct.pack_into("<Q", self.shm.buf, _SEQ_OFF, self._seq)  # even: done

    def close(self, unlink: bool = True) -> None:
        try:
            self.shm.close()
        finally:
            if unlink:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass


class SharedFrameReader:
    """Client side: attach to the block and copy out the latest frame."""

    def __init__(self, name: str = SHM_NAME) -> None:
        if shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory unavailable")
        self.name = name
        self.shm = shared_memory.SharedMemory(name=name)  # raises if absent
        magic, w, h, c = struct.unpack_from(_META_FMT, self.shm.buf, _META_OFF)
        if magic != _MAGIC:
            self.shm.close()
            raise RuntimeError("shared camera block has wrong magic")
        self.width, self.height, self.channels = int(w), int(h), int(c)
        self._last_seq = -1

    def _read_seq(self) -> int:
        return struct.unpack_from("<Q", self.shm.buf, _SEQ_OFF)[0]

    def latest(self, retries: int = 8) -> tuple[np.ndarray, float, int]:
        """Return (frame_copy, capture_timestamp_s, seq). Retries on torn read."""
        n = _frame_bytes(self.width, self.height, self.channels)
        for _ in range(retries):
            s1 = self._read_seq()
            if s1 & 1:  # writer mid-update
                continue
            ts = struct.unpack_from("<d", self.shm.buf, _TS_OFF)[0]
            frame = np.ndarray(
                (self.height, self.width, self.channels), dtype=np.uint8,
                buffer=self.shm.buf, offset=_HEADER_SIZE,
            ).copy()
            s2 = self._read_seq()
            if s1 == s2:
                self._last_seq = s1
                return frame, float(ts), s1
        # Give up on a clean read; return whatever is there now.
        ts = struct.unpack_from("<d", self.shm.buf, _TS_OFF)[0]
        frame = np.ndarray(
            (self.height, self.width, self.channels), dtype=np.uint8,
            buffer=self.shm.buf, offset=_HEADER_SIZE,
        ).copy()
        return frame, float(ts), self._read_seq()

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass


def server_running(heartbeat_s: float = 0.0, name: str = SHM_NAME) -> bool:
    """True if a shared camera block exists (and, if heartbeat_s>0, is live).

    Attaching fails fast when no server is up, so this is cheap to call from
    every CameraCapture.open(). With heartbeat_s>0 it also confirms the seq is
    advancing, catching a stale block left by a crashed server on Linux (on
    Windows the OS frees the block when the last handle closes, so a stale
    block is unlikely).
    """
    if shared_memory is None:
        return False
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    except Exception:
        return False
    try:
        magic = struct.unpack_from("<I", shm.buf, _META_OFF)[0]
        if magic != _MAGIC:
            return False
        if heartbeat_s > 0.0:
            s1 = struct.unpack_from("<Q", shm.buf, _SEQ_OFF)[0]
            sleep(heartbeat_s)
            s2 = struct.unpack_from("<Q", shm.buf, _SEQ_OFF)[0]
            return s2 != s1
        return True
    finally:
        shm.close()
