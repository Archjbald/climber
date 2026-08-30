"""Tests for src.utils.check_vid."""

from pathlib import Path
from  tests.helpers import sample_video_path, sample_fake_path
from src.utils import check_vid

# ASSETS = Path(__file__).parent / "assets"

"""
Test check vid
"""

# Test valid video
def test_check_vid_valid(sample_video_path):
    assert check_vid(sample_video_path)

# Test fake video
def test_check_vid_fake(sample_fake_path):
    assert not check_vid(sample_fake_path)