import os

import numpy as np
import pytest

from cps_maze import camera_share


@pytest.fixture()
def shm_name():
    # Use a unique block name per test so the suite never collides with the
    # production block, a running camera_server, or a leftover from an
    # interrupted run (on Windows a block lingers until its creator exits).
    name = f"cps_test_cam_{os.getpid()}"
    _drop(name)
    yield name
    _drop(name)


def _drop(name: str) -> None:
    try:
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        try:
            shm.unlink()
        except Exception:
            pass
    except Exception:
        pass


def test_no_server_reports_not_running(shm_name):
    assert camera_share.server_running(name=shm_name) is False
    with pytest.raises(Exception):
        camera_share.SharedFrameReader(name=shm_name)


def test_write_then_read_roundtrips_frame_and_timestamp(shm_name):
    writer = camera_share.SharedFrameWriter(width=8, height=4, channels=3, name=shm_name)
    try:
        assert camera_share.server_running(name=shm_name) is True
        frame = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
        writer.write(frame, timestamp_s=123.5)

        reader = camera_share.SharedFrameReader(name=shm_name)
        got, ts, seq = reader.latest()
        assert got.shape == (4, 8, 3)
        assert np.array_equal(got, frame)
        assert ts == 123.5
        assert seq % 2 == 0  # even seq = a completed (non-torn) write
        reader.close()
    finally:
        writer.close(unlink=True)


def test_reader_sees_latest_after_multiple_writes(shm_name):
    writer = camera_share.SharedFrameWriter(width=4, height=4, channels=3, name=shm_name)
    reader = camera_share.SharedFrameReader(name=shm_name)
    try:
        for i in range(5):
            writer.write(np.full((4, 4, 3), i, dtype=np.uint8), timestamp_s=float(i))
        got, ts, _ = reader.latest()
        assert np.all(got == 4)
        assert ts == 4.0
    finally:
        reader.close()
        writer.close(unlink=True)
