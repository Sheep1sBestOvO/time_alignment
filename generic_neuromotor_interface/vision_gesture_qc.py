# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vision-based gesture QC and coarse event-time proposals.

This module is intentionally conservative. It emits high-confidence auxiliary
labels for EMG data QC and EMG time-alignment initialization, not final labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
PINKY_MCP = 17
NUM_LANDMARKS = 21

GESTURES = (
    "index_press",
    "index_release",
    "middle_press",
    "middle_release",
    "thumb_click",
    "thumb_down",
    "thumb_in",
    "thumb_out",
    "thumb_up",
)


@dataclass(frozen=True)
class VisionQCConfig:
    """Thresholds and timing settings for vision QC."""

    window_start_offset: float = 0.1
    window_end_offset: float = 1.5
    contact_threshold: float = 0.35
    confidence_threshold: float = 0.65
    min_tracking_coverage: float = 0.9
    max_missing_frames: int = 0
    min_contact_delta: float = 0.12
    min_release_delta: float = 0.12
    min_swipe_displacement: float = 0.25
    min_click_displacement: float = 0.16
    max_click_duration: float = 0.45
    min_class_margin: float = 0.08
    smoothing_window: int = 5
    max_interp_gap: int = 3
    preferred_handedness: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "VisionQCConfig":
        if path is None:
            return cls()
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - pyyaml is in env.yml
            raise ImportError("Loading config.yaml requires pyyaml.") from exc

        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown config key(s): {unknown}")
        return cls(**data)


@dataclass(frozen=True)
class VisionQCSequence:
    """Per-frame hand landmarks extracted from a video."""

    timestamps: np.ndarray
    landmarks: np.ndarray
    detected: np.ndarray
    fps: float
    handedness: str | None = None


@dataclass(frozen=True)
class VisionFeatures:
    """Signals used by the conservative gesture detectors."""

    timestamps: np.ndarray
    d_index: np.ndarray
    d_middle: np.ndarray
    contact_index: np.ndarray
    contact_middle: np.ndarray
    thumb_position_palm: np.ndarray
    thumb_velocity_palm: np.ndarray
    thumb_speed: np.ndarray
    detected: np.ndarray
    finite_landmarks: np.ndarray


@dataclass(frozen=True)
class Candidate:
    gesture: str
    onset: float
    peak: float
    confidence: float
    strength: float
    reason: str = ""


def extract_hand_landmarks(
    video_path: str | Path,
    *,
    preferred_handedness: str | None = None,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> VisionQCSequence:
    """Run MediaPipe Hands on a video and return a timestamped sequence."""
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional runtime deps
        raise ImportError(
            "Video extraction requires `mediapipe` and `opencv-python`."
        ) from exc

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
    timestamps: list[float] = []
    landmarks: list[np.ndarray] = []
    detected: list[bool] = []
    handedness_votes: list[str] = []

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    try:
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pos_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            selected, handedness = _select_hand(result, preferred_handedness)
            timestamps.append(float(timestamp))
            if selected is None:
                landmarks.append(np.full((NUM_LANDMARKS, 3), np.nan))
                detected.append(False)
            else:
                landmarks.append(selected)
                detected.append(True)
                if handedness is not None:
                    handedness_votes.append(handedness)
            frame_idx += 1
    finally:
        hands.close()
        capture.release()

    handedness = preferred_handedness
    if handedness is None and handedness_votes:
        handedness = max(set(handedness_votes), key=handedness_votes.count)

    return VisionQCSequence(
        timestamps=np.asarray(timestamps, dtype=float),
        landmarks=np.asarray(landmarks, dtype=float),
        detected=np.asarray(detected, dtype=bool),
        fps=fps,
        handedness=handedness,
    )


def _select_hand(
    result: Any,
    preferred_handedness: str | None,
) -> tuple[np.ndarray | None, str | None]:
    hands = getattr(result, "multi_hand_landmarks", None)
    if not hands:
        return None, None

    handedness = getattr(result, "multi_handedness", None)
    chosen = 0
    chosen_label = None
    if handedness:
        labels = [hd.classification[0].label for hd in handedness]
        scores = [hd.classification[0].score for hd in handedness]
        if preferred_handedness in labels:
            chosen = labels.index(preferred_handedness)
        else:
            chosen = int(np.argmax(scores))
        chosen_label = labels[chosen]

    lm = hands[chosen].landmark
    return np.asarray([[p.x, p.y, p.z] for p in lm], dtype=float), chosen_label


def save_landmarks_csv(sequence: VisionQCSequence, output_path: str | Path) -> None:
    """Save raw landmark trajectories to CSV."""
    rows: dict[str, Any] = {
        "timestamp": sequence.timestamps,
        "detected": sequence.detected,
    }
    for idx in range(NUM_LANDMARKS):
        rows[f"landmark_{idx}_x"] = sequence.landmarks[:, idx, 0]
        rows[f"landmark_{idx}_y"] = sequence.landmarks[:, idx, 1]
        rows[f"landmark_{idx}_z"] = sequence.landmarks[:, idx, 2]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _interpolate_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    values = np.array(values, dtype=float, copy=True)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False

    idx = np.arange(values.shape[0])
    for col in range(values.shape[1]):
        series = values[:, col]
        valid = np.isfinite(series)
        if valid.sum() < 2:
            continue
        filled = np.interp(idx, idx[valid], series[valid])
        start = None
        for i, ok in enumerate(valid):
            if not ok and start is None:
                start = i
            elif ok and start is not None:
                if i - start > max_gap:
                    filled[start:i] = np.nan
                start = None
        if start is not None and len(valid) - start > max_gap:
            filled[start:] = np.nan
        values[:, col] = filled

    return values[:, 0] if squeeze else values


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < window:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    out = np.array(values, copy=True)
    for col in range(values.shape[1]):
        series = values[:, col]
        nan = ~np.isfinite(series)
        filled = _interpolate_gaps(series, max_gap=len(series))
        out[:, col] = np.convolve(filled, kernel, mode="same")
        out[nan, col] = np.nan
    return out


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 1e-9)


def compute_vision_features(
    sequence: VisionQCSequence,
    config: VisionQCConfig | None = None,
) -> VisionFeatures:
    """Normalize landmarks and compute contact/thumb-motion features."""
    config = config or VisionQCConfig()
    timestamps = np.asarray(sequence.timestamps, dtype=float)
    landmarks = np.asarray(sequence.landmarks, dtype=float)
    finite_landmarks = np.isfinite(landmarks).all(axis=(1, 2))

    flat = landmarks.reshape(landmarks.shape[0], -1)
    flat = _interpolate_gaps(flat, max_gap=config.max_interp_gap)
    flat = _smooth(flat, config.smoothing_window)
    coords = flat.reshape(-1, NUM_LANDMARKS, 3)[:, :, :2]

    wrist = coords[:, WRIST]
    middle_mcp = coords[:, MIDDLE_MCP]
    index_mcp = coords[:, INDEX_MCP]
    pinky_mcp = coords[:, PINKY_MCP]
    thumb = coords[:, THUMB_TIP]

    palm_scale = np.linalg.norm(middle_mcp - wrist, axis=1)
    fallback_scale = np.nanmedian(palm_scale[np.isfinite(palm_scale)])
    if not np.isfinite(fallback_scale) or fallback_scale <= 1e-9:
        fallback_scale = 1.0
    palm_scale = np.where(palm_scale > 1e-9, palm_scale, fallback_scale)

    y_axis = _unit(middle_mcp - wrist)
    x_axis = _unit(index_mcp - pinky_mcp)
    handedness = (sequence.handedness or config.preferred_handedness or "").lower()
    if handedness == "left":
        x_axis = -x_axis

    rel_thumb = thumb - wrist
    thumb_palm = np.stack(
        [
            np.sum(rel_thumb * x_axis, axis=1) / palm_scale,
            np.sum(rel_thumb * y_axis, axis=1) / palm_scale,
        ],
        axis=1,
    )

    if len(timestamps) > 1:
        thumb_velocity = np.gradient(thumb_palm, timestamps, axis=0)
    else:
        thumb_velocity = np.zeros_like(thumb_palm)
    thumb_speed = np.linalg.norm(thumb_velocity, axis=1)

    d_index = np.linalg.norm(thumb - coords[:, INDEX_TIP], axis=1) / palm_scale
    d_middle = np.linalg.norm(thumb - coords[:, MIDDLE_TIP], axis=1) / palm_scale

    return VisionFeatures(
        timestamps=timestamps,
        d_index=d_index,
        d_middle=d_middle,
        contact_index=d_index < config.contact_threshold,
        contact_middle=d_middle < config.contact_threshold,
        thumb_position_palm=thumb_palm,
        thumb_velocity_palm=thumb_velocity,
        thumb_speed=thumb_speed,
        detected=np.asarray(sequence.detected, dtype=bool),
        finite_landmarks=finite_landmarks,
    )


def _window_indices(
    timestamps: np.ndarray,
    start: float,
    end: float,
) -> tuple[int, int]:
    i0 = int(np.searchsorted(timestamps, start, side="left"))
    i1 = int(np.searchsorted(timestamps, end, side="right"))
    return i0, min(i1, len(timestamps))


def _onset_from_activity(activity: np.ndarray, peak_idx: int, i0: int) -> int:
    segment = activity[i0 : peak_idx + 1]
    finite = segment[np.isfinite(segment)]
    if finite.size == 0:
        return peak_idx
    baseline = float(np.nanpercentile(finite, 20))
    peak = float(activity[peak_idx])
    threshold = baseline + 0.25 * max(peak - baseline, 0.0)
    onset = peak_idx
    for i in range(peak_idx, i0 - 1, -1):
        if not np.isfinite(activity[i]) or activity[i] <= threshold:
            onset = min(i + 1, peak_idx)
            break
        onset = i
    return onset


def _conf(value: float, scale: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return float(value / (value + scale))


def _detect_press(
    features: VisionFeatures,
    i0: int,
    i1: int,
    *,
    gesture: str,
    distance: np.ndarray,
    contact: np.ndarray,
    config: VisionQCConfig,
) -> Candidate | None:
    seg_contact = contact[i0:i1]
    entered = np.flatnonzero(seg_contact & ~np.r_[False, seg_contact[:-1]])
    if entered.size == 0:
        return None
    peak_idx = i0 + int(entered[0])
    local = distance[i0 : peak_idx + 1]
    strength = float(np.nanmax(local) - np.nanmin(distance[i0:i1]))
    if strength < config.min_contact_delta:
        return None
    closing = np.clip(-np.gradient(distance, features.timestamps), 0, None)
    onset_idx = _onset_from_activity(closing, peak_idx, i0)
    return Candidate(
        gesture=gesture,
        onset=float(features.timestamps[onset_idx]),
        peak=float(features.timestamps[peak_idx]),
        confidence=_conf(strength, config.min_contact_delta),
        strength=strength,
    )


def _detect_release(
    features: VisionFeatures,
    i0: int,
    i1: int,
    *,
    gesture: str,
    distance: np.ndarray,
    contact: np.ndarray,
    config: VisionQCConfig,
) -> Candidate | None:
    seg_contact = contact[i0:i1]
    exited = np.flatnonzero(~seg_contact & np.r_[False, seg_contact[:-1]])
    if exited.size == 0:
        return None
    crossing_idx = i0 + int(exited[0])
    opening = np.clip(np.gradient(distance, features.timestamps), 0, None)
    search = opening[max(i0, crossing_idx - 3) : i1]
    if search.size == 0 or not np.isfinite(search).any():
        return None
    peak_idx = max(i0, crossing_idx - 3) + int(np.nanargmax(search))
    strength = float(
        np.nanmax(distance[crossing_idx:i1])
        - np.nanmin(distance[i0 : crossing_idx + 1])
    )
    if strength < config.min_release_delta:
        return None
    onset_idx = _onset_from_activity(opening, peak_idx, i0)
    return Candidate(
        gesture=gesture,
        onset=float(features.timestamps[onset_idx]),
        peak=float(features.timestamps[peak_idx]),
        confidence=_conf(strength, config.min_release_delta),
        strength=strength,
    )


def _detect_swipes(
    features: VisionFeatures,
    i0: int,
    i1: int,
    config: VisionQCConfig,
) -> list[Candidate]:
    pos = features.thumb_position_palm[i0:i1]
    speed = features.thumb_speed[i0:i1]
    valid = np.isfinite(pos).all(axis=1) & np.isfinite(speed)
    if valid.sum() < 2:
        return []
    first = pos[valid][0]
    last = pos[valid][-1]
    displacement = last - first
    axes = {
        "thumb_out": np.array([1.0, 0.0]),
        "thumb_in": np.array([-1.0, 0.0]),
        "thumb_up": np.array([0.0, 1.0]),
        "thumb_down": np.array([0.0, -1.0]),
    }
    candidates = []
    peak_idx = i0 + int(np.nanargmax(speed))
    onset_idx = _onset_from_activity(features.thumb_speed, peak_idx, i0)
    for gesture, axis in axes.items():
        signed = float(np.dot(displacement, axis))
        orth = float(abs(np.dot(displacement, axis[::-1] * np.array([1, -1]))))
        if signed < config.min_swipe_displacement:
            continue
        margin = max(signed - orth, 0.0)
        confidence = _conf(signed, config.min_swipe_displacement) * _conf(
            margin, config.min_class_margin
        )
        candidates.append(
            Candidate(
                gesture=gesture,
                onset=float(features.timestamps[onset_idx]),
                peak=float(features.timestamps[peak_idx]),
                confidence=confidence,
                strength=signed,
            )
        )
    return candidates


def _detect_click(
    features: VisionFeatures,
    i0: int,
    i1: int,
    config: VisionQCConfig,
) -> Candidate | None:
    speed = features.thumb_speed[i0:i1]
    pos = features.thumb_position_palm[i0:i1]
    valid = np.isfinite(speed) & np.isfinite(pos).all(axis=1)
    if valid.sum() < 3:
        return None
    peak_idx = i0 + int(np.nanargmax(speed))
    onset_idx = _onset_from_activity(features.thumb_speed, peak_idx, i0)
    duration = float(features.timestamps[peak_idx] - features.timestamps[onset_idx])
    if duration > config.max_click_duration:
        return None
    excursion = float(np.nanmax(np.linalg.norm(pos - pos[0], axis=1)))
    return_dist = float(np.linalg.norm(pos[-1] - pos[0]))
    if excursion < config.min_click_displacement or return_dist > excursion * 0.55:
        return None
    confidence = _conf(excursion, config.min_click_displacement) * (
        1 - return_dist / excursion
    )
    return Candidate(
        gesture="thumb_click",
        onset=float(features.timestamps[onset_idx]),
        peak=float(features.timestamps[peak_idx]),
        confidence=float(max(0.0, confidence)),
        strength=excursion,
    )


def detect_candidates(
    features: VisionFeatures,
    t_prompt: float,
    config: VisionQCConfig,
) -> list[Candidate]:
    """Return conservative candidates inside one prompt reaction window."""
    i0, i1 = _window_indices(
        features.timestamps,
        t_prompt + config.window_start_offset,
        t_prompt + config.window_end_offset,
    )
    if i1 - i0 < 3:
        return []
    candidates: list[Candidate] = []
    for gesture, distance, contact in (
        ("index_press", features.d_index, features.contact_index),
        ("middle_press", features.d_middle, features.contact_middle),
    ):
        candidate = _detect_press(
            features,
            i0,
            i1,
            gesture=gesture,
            distance=distance,
            contact=contact,
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)
    for gesture, distance, contact in (
        ("index_release", features.d_index, features.contact_index),
        ("middle_release", features.d_middle, features.contact_middle),
    ):
        candidate = _detect_release(
            features,
            i0,
            i1,
            gesture=gesture,
            distance=distance,
            contact=contact,
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(_detect_swipes(features, i0, i1, config))
    click = _detect_click(features, i0, i1, config)
    if click is not None:
        candidates.append(click)
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)


def _tracking_status(
    features: VisionFeatures,
    t_prompt: float,
    config: VisionQCConfig,
) -> tuple[bool, str, float, int]:
    i0, i1 = _window_indices(
        features.timestamps,
        t_prompt + config.window_start_offset,
        t_prompt + config.window_end_offset,
    )
    if i1 <= i0:
        return False, "empty_window", 0.0, 0
    good = features.detected[i0:i1] & features.finite_landmarks[i0:i1]
    coverage = float(good.mean()) if good.size else 0.0
    max_run = 0
    run = 0
    for ok in good:
        if ok:
            run = 0
        else:
            run += 1
            max_run = max(max_run, run)
    if coverage < config.min_tracking_coverage:
        return False, "unstable_tracking", coverage, max_run
    if max_run > config.max_missing_frames:
        return False, "missing_landmarks", coverage, max_run
    return True, "", coverage, max_run


def _proposal_row(
    trial_id: Any,
    prompt_gesture: str,
    t_prompt: float,
    candidates: list[Candidate],
    tracking_ok: bool,
    tracking_reason: str,
    tracking_coverage: float,
    missing_run: int,
    config: VisionQCConfig,
) -> dict[str, Any]:
    best = candidates[0] if candidates else None
    predicted = best.gesture if best is not None else ""
    confidence = best.confidence if best is not None else 0.0
    onset = best.onset if best is not None else np.nan
    peak = best.peak if best is not None else np.nan

    if best is None:
        valid = False
        reason = tracking_reason or "no_event"
    elif not tracking_ok:
        valid = False
        reason = tracking_reason
    elif confidence < config.confidence_threshold:
        valid = False
        reason = "low_confidence"
    elif predicted != prompt_gesture:
        valid = False
        reason = "gesture_mismatch"
    else:
        valid = True
        reason = "valid"

    return {
        "trial_id": trial_id,
        "prompt_gesture": prompt_gesture,
        "predicted_gesture": predicted,
        "t_prompt": t_prompt,
        "t_cam_onset": onset,
        "t_cam_peak": peak,
        "confidence": confidence,
        "valid_flag": bool(valid),
        "reason": reason,
        "tracking_coverage": tracking_coverage,
        "max_missing_frames": missing_run,
    }


def generate_qc_proposals_from_sequence(
    sequence: VisionQCSequence,
    prompts: pd.DataFrame,
    config: VisionQCConfig | None = None,
) -> pd.DataFrame:
    """Run QC proposal generation from an already-extracted landmark sequence."""
    config = config or VisionQCConfig()
    required = {"trial_id", "prompt_gesture", "t_prompt"}
    missing = sorted(required - set(prompts.columns))
    if missing:
        raise ValueError(f"prompts.csv missing required column(s): {missing}")

    features = compute_vision_features(sequence, config)
    rows = []
    for row in prompts.itertuples(index=False):
        trial_id = getattr(row, "trial_id")
        prompt_gesture = str(getattr(row, "prompt_gesture"))
        t_prompt = float(getattr(row, "t_prompt"))
        if prompt_gesture not in GESTURES:
            raise ValueError(f"Unsupported prompt_gesture: {prompt_gesture}")
        tracking_ok, tracking_reason, coverage, missing_run = _tracking_status(
            features, t_prompt, config
        )
        candidates = detect_candidates(features, t_prompt, config)
        rows.append(
            _proposal_row(
                trial_id,
                prompt_gesture,
                t_prompt,
                candidates,
                tracking_ok,
                tracking_reason,
                coverage,
                missing_run,
                config,
            )
        )
    return pd.DataFrame(rows)


def generate_qc_proposals(
    video_path: str | Path,
    prompts_path: str | Path,
    out_path: str | Path,
    landmarks_path: str | Path,
    config: VisionQCConfig | None = None,
) -> pd.DataFrame:
    """End-to-end video -> landmarks CSV + QC proposals CSV."""
    config = config or VisionQCConfig()
    sequence = extract_hand_landmarks(
        video_path, preferred_handedness=config.preferred_handedness
    )
    save_landmarks_csv(sequence, landmarks_path)
    prompts = pd.read_csv(prompts_path)
    proposals = generate_qc_proposals_from_sequence(sequence, prompts, config)
    proposals.to_csv(out_path, index=False)
    return proposals


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Vision-based discrete gesture QC and coarse event-time proposals."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--prompts", required=True, help="prompts.csv path")
    parser.add_argument("--out", required=True, help="Output proposals.csv path")
    parser.add_argument("--landmarks", required=True, help="Output landmarks.csv path")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = VisionQCConfig.from_yaml(args.config)
    generate_qc_proposals(args.video, args.prompts, args.out, args.landmarks, config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
