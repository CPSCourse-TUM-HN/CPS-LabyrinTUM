import numpy as np
import pytest

from cps_maze import camera_share


@pytest.fixture()
def cleanup_shm():
    yield
    # ensure no block leaks between tests
    try:
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=camera_share.SHM_NAME)
        shm.close()
        shm.unlink()
    except Exception:
        pass


def test_no_server_reports_not_running(cleanup_shm):
    # A reader cannot attach and server_running is False when nothing published.
    assert camera_share.server_running() is False
    with pytest.raises(Exception):
        camera_share.SharedFrameReader()


def test_write_then_read_roundtrips_frame_and_timestamp(cleanup_shm):
    writer = camera_share.SharedFrameWriter(width=8, height=4, channels=3)
    try:
        assert camera_share.server_running() is True
        frame = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
        writer.write(frame, timestamp_s=123.5)

        reader = camera_share.SharedFrameReader()
        got, ts, seq = reader.latest()
        assert got.shape == (4, 8, 3)
        assert np.array_equal(got, frame)
        assert ts == 123.5
        assert seq % 2 == 0  # even seq = a completed (non-torn) write
        reader.close()
    finally:
        writer.close(unlink=True)


def test_reader_sees_latest_after_multiple_writes(cleanup_shm):
    writer = camera_share.SharedFrameWriter(width=4, height=4, channels=3)
    reader = camera_share.SharedFrameReader()
    try:
        for i in range(5):
            writer.write(np.full((4, 4, 3), i, dtype=np.uint8), timestamp_s=float(i))
        got, ts, _ = reader.latest()
        assert np.all(got == 4)
        assert ts == 4.0
    finally:
        reader.close()
        writer.close(unlink=True)
