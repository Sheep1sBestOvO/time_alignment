# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd

from generic_neuromotor_interface.vision_gesture_qc import (
    INDEX_TIP,
    MIDDLE_TIP,
    NUM_LANDMARKS,
    THUMB_TIP,
    WRIST,
    VisionQCConfig,
    VisionQCSequence,
    compute_vision_features,
    generate_qc_proposals_from_sequence,
    save_landmarks_csv,
)


def _base_frames(timestamps):
    frames = np.zeros((len(timestamps), NUM_LANDMARKS, 3), dtype=float)
    frames[:, WRIST] = [0.0, 0.0, 0.0]
    frames[:, 5] = [0.22, 0.6, 0.0]  # index MCP
    frames[:, 9] = [0.0, 0.7, 0.0]  # middle MCP
    frames[:, 17] = [-0.22, 0.6, 0.0]  # pinky MCP
    frames[:, INDEX_TIP] = [0.25, 1.0, 0.0]
    frames[:, MIDDLE_TIP] = [0.0, 1.05, 0.0]
    frames[:, THUMB_TIP] = [0.55, 0.55, 0.0]
    return frames


def _sequence(frames, timestamps):
    return VisionQCSequence(
        timestamps=np.asarray(timestamps, dtype=float),
        landmarks=np.asarray(frames, dtype=float),
        detected=np.ones(len(timestamps), dtype=bool),
        handedness="Right",
        fps=60.0,
    )


def test_save_landmarks_csv_writes_one_row_per_frame(tmp_path):
    timestamps = np.linspace(0, 0.2, 3)
    frames = _base_frames(timestamps)
    sequence = _sequence(frames, timestamps)

    out = tmp_path / "landmarks.csv"
    save_landmarks_csv(sequence, out)

    saved = pd.read_csv(out)
    assert list(saved["timestamp"]) == list(timestamps)
    assert saved["detected"].tolist() == [True, True, True]
    assert "landmark_4_x" in saved.columns
    assert "landmark_20_z" in saved.columns


def test_compute_vision_features_normalizes_contact_distance_and_palm_frame():
    timestamps = np.linspace(0, 1, 20)
    frames = _base_frames(timestamps)
    frames[:, THUMB_TIP, :2] = frames[:, INDEX_TIP, :2]
    sequence = _sequence(frames, timestamps)

    features = compute_vision_features(sequence, VisionQCConfig(contact_threshold=0.1))

    assert np.nanmax(features.d_index) < 0.1
    assert features.contact_index.all()
    assert features.thumb_position_palm.shape == (len(timestamps), 2)


def test_index_press_valid_when_thumb_enters_index_contact():
    timestamps = np.linspace(0, 2, 121)
    frames = _base_frames(timestamps)
    close = np.clip((timestamps - 0.5) / 0.35, 0, 1)
    frames[:, THUMB_TIP, :2] = (
        (1 - close)[:, None] * np.array([0.55, 0.55])
        + close[:, None] * frames[:, INDEX_TIP, :2]
    )
    prompts = pd.DataFrame(
        {"trial_id": [7], "prompt_gesture": ["index_press"], "t_prompt": [0.3]}
    )

    out = generate_qc_proposals_from_sequence(
        _sequence(frames, timestamps),
        prompts,
        VisionQCConfig(confidence_threshold=0.45, contact_threshold=0.12),
    )

    row = out.iloc[0]
    assert row["trial_id"] == 7
    assert row["predicted_gesture"] == "index_press"
    assert bool(row["valid_flag"])
    assert 0.45 <= row["t_cam_onset"] <= row["t_cam_peak"] <= 0.95


def test_wrong_prompt_is_invalid_even_when_event_is_confident():
    timestamps = np.linspace(0, 2, 121)
    frames = _base_frames(timestamps)
    close = np.clip((timestamps - 0.5) / 0.35, 0, 1)
    frames[:, THUMB_TIP, :2] = (
        (1 - close)[:, None] * np.array([0.55, 0.55])
        + close[:, None] * frames[:, INDEX_TIP, :2]
    )
    prompts = pd.DataFrame(
        {"trial_id": [8], "prompt_gesture": ["middle_press"], "t_prompt": [0.3]}
    )

    out = generate_qc_proposals_from_sequence(
        _sequence(frames, timestamps),
        prompts,
        VisionQCConfig(confidence_threshold=0.45, contact_threshold=0.12),
    )

    row = out.iloc[0]
    assert row["predicted_gesture"] == "index_press"
    assert not bool(row["valid_flag"])
    assert row["reason"] == "gesture_mismatch"


def test_thumb_out_swipe_uses_palm_frame_direction():
    timestamps = np.linspace(0, 2, 121)
    frames = _base_frames(timestamps)
    move = np.clip((timestamps - 0.55) / 0.35, 0, 1)
    frames[:, THUMB_TIP, 0] = 0.15 + 0.55 * move
    frames[:, THUMB_TIP, 1] = 0.55
    prompts = pd.DataFrame(
        {"trial_id": [9], "prompt_gesture": ["thumb_out"], "t_prompt": [0.4]}
    )

    out = generate_qc_proposals_from_sequence(
        _sequence(frames, timestamps),
        prompts,
        VisionQCConfig(confidence_threshold=0.4, min_swipe_displacement=0.2),
    )

    row = out.iloc[0]
    assert row["predicted_gesture"] == "thumb_out"
    assert bool(row["valid_flag"])
    assert 0.5 <= row["t_cam_peak"] <= 1.1
