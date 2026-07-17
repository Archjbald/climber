from unittest.mock import patch
from tests.helpers import make_pose, make_motions
import numpy as np
from src.analyse import analyse_climb
from pytest import approx

@patch("src.analyse.count_moves")
def test_analyse_climb_return(mock_count_moves):
    pose, _ = make_pose()

    fps = 30

    motions = make_motions(len(pose), 1)
    expected = {
        "move_count": 42,
        "static_time":approx(np.sum(motions) / fps)
    }

    mock_count_moves.return_value = np.array(expected["move_count"]), motions

    analysis = analyse_climb(pose, fps=fps)

    assert analysis == expected
    mock_count_moves.assert_called_once_with(
        pose, fps=fps
    )