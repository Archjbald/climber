from pathlib import Path

from src.utils import check_vid

ASSETS = Path(__file__).parent / "assets"

"""
Test check vid
"""

# Test valid video
def test_check_vid_valid():
    assert check_vid(ASSETS / "valid.mp4")

# Test fake video
def test_check_vid_fake():
    assert not check_vid(ASSETS / "fake.mp4")