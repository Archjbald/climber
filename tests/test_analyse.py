"""Tests for src.analyse."""

from unittest.mock import patch

import numpy as np
import pytest
from pytest import approx

from src.analyse import (
    analyse_climb,
    compute_joint_speed,
    compute_kp_state,
    count_moves,
    moving_average,
    threshold_window,
)
from tests.helpers import make_climber_track, make_motions, make_pose


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


# --- compute_joint_speed ---------------------------------------------------


def test_compute_joint_speed_matches_constant_velocity():
    pose = np.zeros((5, 17, 2))
    pose[:, 9, 0] = np.arange(5) * 3.0  # left wrist drifts 3 px/frame

    speed = compute_joint_speed(pose)

    assert speed.shape == (5, 17)
    assert speed[1:, 9] == approx(3.0)
    assert speed[:, 10] == approx(0.0)


# --- moving_average ------------------------------------------------------------


def test_moving_average_window_leq_one_returns_signal_unchanged():
    x = np.array([1.0, 4.0, 2.0, 9.0])
    assert np.array_equal(moving_average(x, 1), x)


def test_moving_average_spreads_an_interior_spike():
    x = np.zeros(15)
    x[7] = 9.0

    out = moving_average(x, 3)

    assert out.max() < 9.0
    assert np.sum(out > 0.0) >= 3


# --- compute_kp_state --------------------------------------------------------


def test_compute_kp_state_stays_static_below_threshold():
    speed = np.full(40, 0.05)
    assert np.all(compute_kp_state(speed, threshold=0.2, static_frames=5) == 0)


def test_compute_kp_state_enters_then_exits_moving():
    speed = np.zeros(60)
    speed[10:25] = 1.0

    states = compute_kp_state(speed, threshold=0.2, static_frames=8)

    assert states[5] == 0
    assert states[15] == 1
    assert states[-1] == 0
    assert np.sum(np.diff(states.astype(int)) < 0) == 1


def test_compute_kp_state_hysteresis_ignores_mild_speed():
    # speed sits below the enter threshold (1.2 * threshold) the whole time
    speed = np.full(30, 0.2)
    assert np.all(compute_kp_state(speed, threshold=0.2, static_frames=5) == 0)


# --- count_moves -----------------------------------------------------------


def test_count_moves_zero_for_a_motionless_track():
    moves, states = count_moves(make_climber_track(90), fps=30)

    assert moves == 0
    assert np.all(states == 0)


def test_count_moves_one_per_wrist_reach():
    track = make_climber_track(150, reaches=((40, 55), (95, 110)))
    moves, _ = count_moves(track, fps=30)
    assert moves == 2


def test_count_moves_watches_ankles_too():
    track = make_climber_track(95, reaches=((40, 55),), keypoint=16)
    moves, _ = count_moves(track, fps=30)
    assert moves == 1


# --- threshold_window ------------------------------------------------------


def test_threshold_window_sup_and_inf_split_a_step_signal():
    vals = np.concatenate([np.zeros(10), np.full(10, 5.0)])

    sup = threshold_window(vals, thresh=2.0, wind=4, keep="sup")
    inf = threshold_window(vals, thresh=2.0, wind=4, keep="inf")

    assert not sup[:10].any() and sup[10:].all()
    assert inf[:10].all() and not inf[10:].any()


def test_threshold_window_rejects_unknown_keep():
    with pytest.raises(AssertionError):
        threshold_window(np.zeros(5), thresh=1.0, wind=3, keep="middle")
