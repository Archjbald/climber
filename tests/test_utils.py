"""Tests for src.utils.check_vid."""

from src.utils import check_vid

# --- check_vid -----------------------------------------------------------------


def test_check_vid_valid(sample_video_path):
    assert check_vid(sample_video_path)


def test_check_vid_fake(sample_fake_path):
    assert not check_vid(sample_fake_path)
