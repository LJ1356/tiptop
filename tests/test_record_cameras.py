"""record_cameras' failure handling: who finalizes an SVO, and what a bad one costs.

Both behaviours here were regressions found from a real eval run whose wrist SVO came back
"Corruption detected ... INVALID SVO FILE": the conversion error propagated out of the context
manager and killed a rollout the robot had already executed, and the stop path could finalize an
SVO from the main thread while the grab thread was still writing to it.

No hardware: ``tiptop.recording.ZedCamera`` is swapped for a fake, which is what the module's
``isinstance`` checks look at.
"""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

import tiptop.recording as recording
from tiptop.perception.cameras.zed_camera import CorruptSVOError


class FakeZed:
    """A ZED that records by touching the .svo2 file the SDK would write."""

    def __init__(self, serial: str, grab_delay: float = 0.0):
        self.serial = serial
        self.grab_delay = grab_delay
        self.grabs = 0
        self.stop_threads: list[int] = []
        self._recording_path: Path | None = None

    def read_camera(self):
        if self.grab_delay:
            time.sleep(self.grab_delay)
        self.grabs += 1
        return type("F", (), {"rgb": np.zeros((4, 4, 3), np.uint8)})()

    def start_recording(self, filename: str):
        # The SDK appends the real extension, which is what recording.py's .svo -> .svo2 fallback
        # exists to find; reproduce that so the test covers it.
        self._recording_path = Path(filename).with_suffix(".svo2")
        self._recording_path.parent.mkdir(parents=True, exist_ok=True)
        self._recording_path.write_bytes(b"svo")

    def stop_recording(self):
        self.stop_threads.append(threading.get_ident())


@pytest.fixture
def fake_zed(monkeypatch):
    monkeypatch.setattr(recording, "ZedCamera", FakeZed)
    return FakeZed


def test_zed_svo_is_finalized_on_its_own_grab_thread(fake_zed, tmp_path, monkeypatch):
    """stop_recording must never run on the main thread: one sl.Camera cannot finalize an SVO and
    grab into it at the same time, and doing both truncates the file."""
    monkeypatch.setattr(recording, "convert_svo_to_mp4", lambda src, dst: dst.write_bytes(b"mp4"))
    cam = FakeZed("111")

    with recording.record_cameras([(cam, tmp_path / "hand_cam.svo", tmp_path / "hand_cam.mp4")]):
        while cam.grabs < 2:
            time.sleep(0.01)

    assert cam.stop_threads, "the recording was never finalized"
    assert threading.get_ident() not in cam.stop_threads


def test_a_slow_grab_is_waited_out_rather_than_raced(fake_zed, tmp_path, monkeypatch):
    """A grab that outlives the old 5 s join used to get its recording closed from under it. The
    join now has to actually finish, so a slow camera still finalizes on its own thread."""
    monkeypatch.setattr(recording, "convert_svo_to_mp4", lambda src, dst: dst.write_bytes(b"mp4"))
    monkeypatch.setattr(recording, "_STOP_JOIN_TIMEOUT", 10.0)
    cam = FakeZed("111", grab_delay=0.4)

    with recording.record_cameras([(cam, tmp_path / "hand_cam.svo", tmp_path / "hand_cam.mp4")]):
        while cam.grabs < 1:
            time.sleep(0.01)

    assert len(cam.stop_threads) == 1
    assert threading.get_ident() not in cam.stop_threads


def test_a_corrupt_svo_does_not_abort_the_episode(fake_zed, tmp_path, caplog):
    """The caller dumps the episode *after* this context manager exits, so a camera whose SVO will
    not open must not raise -- the proprioception and the other cameras are still good."""
    external, hand = FakeZed("ext"), FakeZed("hand")

    def convert(src: Path, dst: Path):
        if "hand" in src.name:
            raise CorruptSVOError(f"Cannot open SVO file (corrupted or truncated): {src}")
        dst.write_bytes(b"mp4")

    caplog.set_level("ERROR")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(recording, "convert_svo_to_mp4", convert)
        with recording.record_cameras([
            (external, tmp_path / "external_cam.svo", tmp_path / "external_cam.mp4"),
            (hand, tmp_path / "hand_cam.svo", tmp_path / "hand_cam.mp4"),
        ]) as window:
            while external.grabs < 1 or hand.grabs < 1:
                time.sleep(0.01)

    # The window is still stamped, so the export can align what did record.
    assert window["t_stop"] >= window["t_start"]
    assert (tmp_path / "external_cam.mp4").exists()
    assert not (tmp_path / "hand_cam.mp4").exists()
    # The bad SVO is kept for inspection rather than cleaned up.
    assert (tmp_path / "hand_cam.svo2").exists()
    assert "corrupted or truncated" in caplog.text


def test_a_missing_svo_does_not_abort_the_episode(fake_zed, tmp_path, caplog, monkeypatch):
    """Same reasoning for a recording that produced no file at all."""
    monkeypatch.setattr(recording, "convert_svo_to_mp4", lambda src, dst: dst.write_bytes(b"mp4"))
    cam = FakeZed("111")
    monkeypatch.setattr(cam, "start_recording", lambda filename: None)  # writes nothing

    caplog.set_level("ERROR")
    with recording.record_cameras([(cam, tmp_path / "hand_cam.svo", tmp_path / "hand_cam.mp4")]):
        while cam.grabs < 1:
            time.sleep(0.01)

    assert not (tmp_path / "hand_cam.mp4").exists()
    assert "SVO file not found" in caplog.text
