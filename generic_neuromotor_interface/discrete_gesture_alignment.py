# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for discrete-gesture prompt pattern analysis and time alignment.

The alignment implementation follows the forced-alignment idea described in the
paper: initialize event times from prompt times, estimate gesture-specific
templates, then refine event times by finding the sequence of template placements
that best explains the observed continuous signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_MPF_FREQUENCY_BINS: tuple[tuple[float, float], ...] = (
    (0, 50),
    (30, 100),
    (100, 225),
    (225, 375),
    (375, 700),
    (700, 1000),
)


@dataclass(frozen=True)
class TemplateBank:
    """Gesture templates with a shared time support around event time zero."""

    templates: dict[str, np.ndarray]
    pre_samples: int
    post_samples: int
    sample_rate: float

    @property
    def length(self) -> int:
        return self.pre_samples + self.post_samples


@dataclass(frozen=True)
class SequenceStats:
    """Summary of prompt sequences whose uncertainty intervals overlap."""

    num_prompts: int
    num_sequences: int
    sequence_lengths: list[int]
    prompt_interval_seconds: dict[str, float]


def load_discrete_gesture_hdf5(
    hdf5_path: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load timeseries and prompts from a discrete gesture HDF5 file."""

    import h5py

    hdf5_path = Path(hdf5_path).expanduser()
    with h5py.File(hdf5_path, "r") as file:
        timeseries = file["data"][:]
        task = file["data"].attrs["task"]
        if task != "discrete_gestures":
            raise ValueError(f"Expected discrete_gestures task, got {task!r}")

    prompts = pd.read_hdf(hdf5_path, "prompts")
    return timeseries, prompts


def infer_sample_rate(times: np.ndarray) -> float:
    """Infer sample rate from monotonic timestamps."""

    if len(times) < 2:
        raise ValueError("Need at least two timestamps to infer sample rate")
    dt = np.diff(times)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Timestamps must contain positive increments")
    return float(1.0 / np.median(dt))


def emg_envelope_features(
    emg: np.ndarray,
    sample_rate: float,
    smoothing_ms: float = 50.0,
    zscore: bool = True,
) -> np.ndarray:
    """Create simple continuous features suitable for alignment.

    The paper used MPF features. This helper is intentionally modest: rectify
    EMG, smooth it with a moving average, and optionally z-score each channel.
    It gives the alignment code a usable default while keeping the feature
    extraction swappable.
    """

    features = np.abs(np.asarray(emg, dtype=np.float64))
    window = max(1, int(round(sample_rate * smoothing_ms / 1000.0)))
    if window > 1:
        kernel = np.ones(window, dtype=np.float64) / window
        features = np.vstack(
            [
                np.convolve(features[:, channel], kernel, mode="same")
                for channel in range(features.shape[1])
            ]
        ).T

    if zscore:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        features = (features - mean) / np.maximum(std, 1e-8)

    return features


def multivariate_power_frequency_features(
    emg: np.ndarray,
    times: np.ndarray,
    window_length: int = 200,
    stride: int = 40,
    n_fft: int = 64,
    fft_stride: int = 10,
    fs: float | None = None,
    frequency_bins: Sequence[tuple[float, float]] | None = DEFAULT_MPF_FREQUENCY_BINS,
    chunk_output_frames: int = 4096,
    zscore: bool = True,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MPF features from continuous EMG for alignment.

    This is a NumPy-friendly wrapper around the repository's
    ``MultivariatePowerFrequencyFeatures`` module. It converts raw EMG of shape
    ``(time, channels)`` into flattened MPF frames of shape
    ``(mpf_time, frequency * channels * channels)`` and returns the corresponding
    downsampled timestamps.

    The default parameters match the MPF featurizer used by ``WristArchitecture``
    in this repository: ``window_length=200``, ``stride=40``, ``n_fft=64``, and
    ``fft_stride=10`` with six broad frequency bands.
    """

    import torch

    emg = np.asarray(emg, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    if emg.ndim != 2:
        raise ValueError("emg must have shape (time, channels)")
    if len(emg) != len(times):
        raise ValueError("emg and times must have the same time length")
    if fs is None:
        fs = infer_sample_rate(times)

    if window_length < n_fft:
        raise ValueError("window_length must be greater than n_fft")
    if fft_stride > n_fft:
        raise ValueError("fft_stride must be lower than n_fft")
    if fft_stride > stride:
        raise ValueError("stride must be greater than fft_stride")
    if stride % fft_stride != 0:
        raise ValueError("stride must be a multiple of fft_stride")

    left_context = window_length - fft_stride + n_fft - 1
    output_frame_indices = np.arange(left_context, len(emg), stride)
    if len(output_frame_indices) == 0:
        raise ValueError(
            "EMG recording is too short for the requested MPF window parameters"
        )

    window = torch.hann_window(n_fft, periodic=False)
    window_normalization_factor = torch.linalg.vector_norm(window)
    freq_masks = (
        _build_mpf_frequency_masks(n_fft, float(fs), frequency_bins)
        if frequency_bins is not None
        else None
    )

    feature_chunks = []
    num_chunks = int(np.ceil(len(output_frame_indices) / chunk_output_frames))
    with torch.no_grad():
        for chunk_number, chunk_start in enumerate(
            range(0, len(output_frame_indices), chunk_output_frames), start=1
        ):
            if progress:
                print(
                    "Computing MPF chunk "
                    f"{chunk_number}/{num_chunks} "
                    f"({chunk_start}/{len(output_frame_indices)} frames)",
                    flush=True,
                )
            chunk_indices = output_frame_indices[
                chunk_start : chunk_start + chunk_output_frames
            ]
            raw_start = int(chunk_indices[0] - left_context)
            raw_stop = int(chunk_indices[-1] + 1)
            raw_chunk = torch.from_numpy(emg[raw_start:raw_stop].T).unsqueeze(0)
            mpf_chunk = _compute_mpf_torch(
                raw_chunk,
                window_length=window_length,
                stride=stride,
                n_fft=n_fft,
                fft_stride=fft_stride,
                window=window,
                window_normalization_factor=window_normalization_factor,
                freq_masks=freq_masks,
            ).squeeze(0)
            mpf_chunk = mpf_chunk.permute(3, 0, 1, 2).reshape(
                mpf_chunk.shape[-1], -1
            )
            feature_chunks.append(mpf_chunk.cpu().numpy())

    features = np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)
    if zscore:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        features = (features - mean) / np.maximum(std, 1e-6)

    return features, times[output_frame_indices]


def _compute_mpf_torch(
    inputs: "torch.Tensor",
    window_length: int,
    stride: int,
    n_fft: int,
    fft_stride: int,
    window: "torch.Tensor",
    window_normalization_factor: "torch.Tensor",
    freq_masks: "torch.Tensor | None",
) -> "torch.Tensor":
    """Torch MPF implementation matching ``MultivariatePowerFrequencyFeatures``."""

    import torch

    batch_size, num_channels, _ = inputs.shape
    num_freqs = n_fft // 2 + 1

    spectrogram = (
        torch.stft(
            inputs.reshape(batch_size * num_channels, -1),
            n_fft=n_fft,
            hop_length=fft_stride,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        / window_normalization_factor
    )

    spectrogram = spectrogram.unfold(
        dimension=-1,
        size=window_length // fft_stride,
        step=stride // fft_stride,
    )

    _, _, num_windows, window_size = spectrogram.shape
    spectrogram = spectrogram.reshape(
        batch_size,
        num_channels,
        num_freqs,
        num_windows,
        window_size,
    )
    spectrogram = spectrogram.transpose(1, 3)

    outputs = _compute_strided_cross_spectral_density(spectrogram)

    if freq_masks is not None:
        outputs = torch.stack(
            [
                (outputs * freq_mask).sum(2) / freq_mask.sum(2)
                for freq_mask in freq_masks.unbind(2)
            ],
            dim=2,
        )

    eigvals, eigvecs = torch.linalg.eigh(outputs)
    eigvals = eigvals.log().nan_to_num(nan=0.0, neginf=0.0)
    outputs = (eigvecs * eigvals.unsqueeze(dim=-2)) @ eigvecs.transpose(-1, -2)

    return outputs.permute(0, 2, 3, 4, 1)


def _build_mpf_frequency_masks(
    n_fft: int,
    fs: float,
    frequency_bins: Sequence[tuple[float, float]],
) -> "torch.Tensor":
    import torch

    freqs_hz = torch.fft.fftfreq(n_fft, d=1.0 / fs)[: (n_fft // 2 + 1)].abs()
    freq_masks = torch.stack(
        [
            torch.logical_and(freqs_hz > start_freq, freqs_hz <= end_freq)
            for start_freq, end_freq in frequency_bins
        ]
    ).to(dtype=torch.uint8)
    return freq_masks.reshape(1, 1, len(frequency_bins), len(freqs_hz), 1, 1)


def _compute_strided_cross_spectral_density(inputs: "torch.Tensor") -> "torch.Tensor":
    input_dims = inputs.shape
    num_channels, window_size = input_dims[-2:]
    outputs = inputs.reshape(-1, num_channels, window_size)
    outputs = (outputs @ outputs.transpose(-2, -1).conj()) / window_size
    outputs = outputs.abs().pow(2)
    return outputs.reshape(*input_dims[:-2], num_channels, num_channels)


def summarize_prompt_pattern(
    prompts: pd.DataFrame,
    uncertainty_window: tuple[float, float] | None = None,
) -> SequenceStats:
    """Summarize prompt intervals and optional overlapping uncertainty sequences."""

    prompts = _valid_sorted_prompts(prompts)
    times = prompts["time"].to_numpy(dtype=float)
    intervals = np.diff(times)
    interval_summary = {
        "count": float(len(intervals)),
        "mean": float(np.mean(intervals)) if len(intervals) else np.nan,
        "median": float(np.median(intervals)) if len(intervals) else np.nan,
        "min": float(np.min(intervals)) if len(intervals) else np.nan,
        "max": float(np.max(intervals)) if len(intervals) else np.nan,
    }

    if uncertainty_window is None:
        sequence_lengths = [1] * len(prompts)
    else:
        sequence_lengths = [
            len(sequence)
            for sequence in group_overlapping_sequences(prompts, uncertainty_window)
        ]

    return SequenceStats(
        num_prompts=len(prompts),
        num_sequences=len(sequence_lengths),
        sequence_lengths=sequence_lengths,
        prompt_interval_seconds=interval_summary,
    )


def group_overlapping_sequences(
    prompts: pd.DataFrame,
    uncertainty_window: tuple[float, float],
) -> list[pd.DataFrame]:
    """Group prompts whose timing uncertainty intervals overlap.

    ``uncertainty_window`` is relative to each prompt time, e.g. ``(-0.3, 0.8)``.
    """

    prompts = _valid_sorted_prompts(prompts)
    if len(prompts) == 0:
        return []

    lower, upper = uncertainty_window
    if lower > upper:
        raise ValueError("uncertainty_window must be ordered as (lower, upper)")

    sequences: list[pd.DataFrame] = []
    current_indices = [prompts.index[0]]
    current_end = float(prompts.iloc[0]["time"]) + upper

    for idx, row in prompts.iloc[1:].iterrows():
        start = float(row["time"]) + lower
        end = float(row["time"]) + upper
        if start <= current_end:
            current_indices.append(idx)
            current_end = max(current_end, end)
        else:
            sequences.append(prompts.loc[current_indices].copy())
            current_indices = [idx]
            current_end = end

    sequences.append(prompts.loc[current_indices].copy())
    return sequences


def estimate_templates(
    features: np.ndarray,
    times: np.ndarray,
    prompts: pd.DataFrame,
    pre_s: float,
    post_s: float,
    aligned_time_col: str = "aligned_time",
    method: str = "average",
    ridge: float = 1e-3,
    fit_intercept: bool = True,
) -> TemplateBank:
    """Estimate gesture templates around event times.

    Parameters
    ----------
    method:
        ``"average"`` estimates each template by averaging event-centered
        windows. ``"rerp"`` estimates all templates jointly with a
        regression-based event-related-potential estimator, which can separate
        overlapping gesture contributions better than simple averaging.
    ridge:
        L2 regularization used by the rERP estimator.
    fit_intercept:
        If True, include a constant nuisance regressor in the rERP design matrix.
    """

    prompts = _valid_sorted_prompts(prompts)
    sample_rate = infer_sample_rate(times)
    pre_samples = int(round(pre_s * sample_rate))
    post_samples = int(round(post_s * sample_rate))
    if pre_samples < 0 or post_samples <= 0:
        raise ValueError("pre_s must be non-negative and post_s must be positive")

    if method == "average":
        return estimate_templates_average(
            features,
            times,
            prompts,
            pre_samples,
            post_samples,
            sample_rate,
            aligned_time_col=aligned_time_col,
        )
    if method == "rerp":
        return estimate_templates_rerp(
            features,
            times,
            prompts,
            pre_samples,
            post_samples,
            sample_rate,
            aligned_time_col=aligned_time_col,
            ridge=ridge,
            fit_intercept=fit_intercept,
        )
    raise ValueError("method must be either 'average' or 'rerp'")


def estimate_templates_average(
    features: np.ndarray,
    times: np.ndarray,
    prompts: pd.DataFrame,
    pre_samples: int,
    post_samples: int,
    sample_rate: float,
    aligned_time_col: str = "aligned_time",
) -> TemplateBank:
    """Estimate templates by averaging windows around event times."""

    time_col = aligned_time_col if aligned_time_col in prompts.columns else "time"
    templates: dict[str, np.ndarray] = {}
    for name, group in prompts.groupby("name", sort=False):
        windows = []
        for event_time in group[time_col].to_numpy(dtype=float):
            center = int(np.searchsorted(times, event_time))
            start = center - pre_samples
            stop = center + post_samples
            if start < 0 or stop > len(features):
                continue
            windows.append(features[start:stop])
        if windows:
            templates[str(name)] = np.mean(np.stack(windows, axis=0), axis=0)

    if not templates:
        raise ValueError("No templates could be estimated from the provided prompts")

    return TemplateBank(
        templates=templates,
        pre_samples=pre_samples,
        post_samples=post_samples,
        sample_rate=sample_rate,
    )


def estimate_templates_rerp(
    features: np.ndarray,
    times: np.ndarray,
    prompts: pd.DataFrame,
    pre_samples: int,
    post_samples: int,
    sample_rate: float,
    aligned_time_col: str = "aligned_time",
    ridge: float = 1e-3,
    fit_intercept: bool = True,
) -> TemplateBank:
    """Estimate gesture templates with a regression-based ERP analogue.

    This solves a linear model over the full continuous feature sequence:

    ``features ~= design_matrix @ template_coefficients + intercept``.

    Each gesture class gets ``pre_samples + post_samples`` lag regressors. For
    every event of that class, those lag regressors are placed at the samples
    covered by the event-centered template window. Least-squares coefficients are
    then reshaped into one template per gesture class.
    """

    from scipy import sparse

    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must have shape (time, feature_dim)")

    time_col = aligned_time_col if aligned_time_col in prompts.columns else "time"
    gesture_names = [str(name) for name in prompts["name"].drop_duplicates()]
    template_len = pre_samples + post_samples
    if template_len <= 0:
        raise ValueError("template length must be positive")

    gesture_to_offset = {
        name: gesture_idx * template_len
        for gesture_idx, name in enumerate(gesture_names)
    }
    num_template_params = len(gesture_names) * template_len
    if num_template_params > 10_000:
        raise ValueError(
            "rERP template estimation would create too many template parameters "
            f"({num_template_params}). Use MPF/downsampled features, reduce the "
            "template window, or choose method='average'."
        )

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for row in prompts.itertuples():
        name = str(row.name)
        event_time = float(getattr(row, time_col))
        center = int(np.searchsorted(times, event_time))
        start = center - pre_samples
        stop = center + post_samples
        if start < 0 or stop > len(features):
            continue
        rows.append(np.arange(start, stop, dtype=np.int64))
        cols.append(
            gesture_to_offset[name] + np.arange(template_len, dtype=np.int64)
        )

    if not rows:
        raise ValueError("No valid events could be used for rERP template estimation")

    design_rows = np.concatenate(rows)
    design_cols = np.concatenate(cols)
    design_data = np.ones(len(design_rows), dtype=np.float64)

    num_columns = num_template_params + int(fit_intercept)
    if fit_intercept:
        intercept_col = np.full(len(features), num_template_params, dtype=np.int64)
        design_rows = np.concatenate(
            [design_rows, np.arange(len(features), dtype=np.int64)]
        )
        design_cols = np.concatenate([design_cols, intercept_col])
        design_data = np.concatenate(
            [design_data, np.ones(len(features), dtype=np.float64)]
        )

    design = sparse.coo_matrix(
        (design_data, (design_rows, design_cols)),
        shape=(len(features), num_columns),
    ).tocsr()

    xtx = (design.T @ design).toarray()
    if ridge > 0:
        regularizer = np.full(num_columns, ridge, dtype=np.float64)
        if fit_intercept:
            regularizer[-1] = 0.0
        xtx.flat[:: num_columns + 1] += regularizer
    xty = design.T @ features
    coefficients = np.linalg.solve(xtx, xty)
    coefficients = coefficients[:num_template_params]

    templates = {}
    for name, offset in gesture_to_offset.items():
        templates[name] = coefficients[offset : offset + template_len]

    return TemplateBank(
        templates=templates,
        pre_samples=pre_samples,
        post_samples=post_samples,
        sample_rate=sample_rate,
    )


def align_prompt_times(
    features: np.ndarray,
    times: np.ndarray,
    prompts: pd.DataFrame,
    pre_s: float,
    post_s: float,
    uncertainty_window: tuple[float, float],
    max_iterations: int = 5,
    beam_width: int = 30,
    candidate_step_s: float = 0.02,
    tolerance_s: float = 0.005,
    recenter_templates: bool = False,
    template_estimator: str = "rerp",
    template_ridge: float = 1e-3,
    enforce_monotonic: bool = True,
    min_event_separation_s: float = 0.0,
    prompt_prior_weight: float = 0.0,
    progress: bool = False,
    init_templates: "TemplateBank | None" = None,
    uncertainty_schedule: "Sequence[tuple[float, float]] | None" = None,
) -> tuple[pd.DataFrame, TemplateBank]:
    """Iteratively align discrete gesture prompts to continuous features.

    ``uncertainty_schedule`` optionally runs coarse-to-fine annealing: a list of
    search windows (narrow -> wide). Each stage runs the EM loop to convergence
    (or ``max_iterations``) with that window, carrying the alignment forward.
    Starting narrow keeps the cold-start template from scattering events; widening
    later lets the (now sharper) template localize events that are further off.
    If omitted, a single stage with ``uncertainty_window`` is used.

    ``init_templates`` optionally seeds the FIRST iteration with a given template
    bank (e.g. an oracle bank estimated from ground-truth times) instead of
    estimating templates from the current (perturbed) positions. Later iterations
    re-estimate normally. This tests whether a good initialization lets EM
    sustain a correct solution or whether it degrades back to the perturbed fixed
    point.

    This implements the paper's EM-style generative inference: estimate
    gesture-specific templates from current event times (rERP), then re-infer
    event times via beam search over the generative residual, repeating until the
    timestamp updates fall below ``tolerance_s``.

    ``recenter_templates`` defaults to ``False`` to match the paper, which keeps
    the within-gesture timing offset indeterminate during EM and resolves it once
    at the end via a global recentering against the grand-average template (see
    ``global_recenter_aligned_prompts``). Setting it to ``True`` applies a
    per-iteration energy-peak recentering, which is only a rough single-session
    fallback when no cross-participant reference is available.
    """

    aligned = _valid_sorted_prompts(prompts).copy()
    aligned["prompt_time"] = aligned["time"].to_numpy(dtype=float)
    aligned["aligned_time"] = aligned["prompt_time"]

    schedule = list(uncertainty_schedule) if uncertainty_schedule else [uncertainty_window]

    templates: TemplateBank | None = None
    first_iteration_overall = True
    for stage_idx, stage_window in enumerate(schedule):
        if progress and len(schedule) > 1:
            print(
                f"[align] === stage {stage_idx + 1}/{len(schedule)}: "
                f"uncertainty window {stage_window} ===",
                flush=True,
            )
        for iteration in range(max_iterations):
            if first_iteration_overall and init_templates is not None:
                if progress:
                    print(
                        "[align] iteration 1: using provided init_templates "
                        "(seeding first iteration)",
                        flush=True,
                    )
                templates = init_templates
            else:
                if progress:
                    print(
                        f"[align] stage {stage_idx + 1} iter {iteration + 1}/"
                        f"{max_iterations}: estimating templates ({template_estimator})",
                        flush=True,
                    )
                templates = estimate_templates(
                    features,
                    times,
                    aligned,
                    pre_s,
                    post_s,
                    aligned_time_col="aligned_time",
                    method=template_estimator,
                    ridge=template_ridge,
                )
            first_iteration_overall = False
            if recenter_templates:
                templates = recenter_template_bank(templates)
            previous = aligned["aligned_time"].to_numpy(dtype=float).copy()
            sequences = group_overlapping_sequences(aligned, stage_window)
            if progress:
                print(
                    f"[align] stage {stage_idx + 1} iter {iteration + 1}/"
                    f"{max_iterations}: aligning {len(sequences)} sequences",
                    flush=True,
                )
            pieces = []
            for seq_idx, sequence in enumerate(sequences, start=1):
                pieces.append(
                    _align_sequence(
                        features=features,
                        times=times,
                        sequence=sequence,
                        templates=templates,
                        uncertainty_window=stage_window,
                        beam_width=beam_width,
                        candidate_step_s=candidate_step_s,
                        enforce_monotonic=enforce_monotonic,
                        min_event_separation_s=min_event_separation_s,
                        prompt_prior_weight=prompt_prior_weight,
                    )
                )
                if progress and (seq_idx % 200 == 0 or seq_idx == len(sequences)):
                    print(
                        f"[align]   stage {stage_idx + 1} iter {iteration + 1}: "
                        f"{seq_idx}/{len(sequences)} sequences aligned",
                        flush=True,
                    )
            aligned = pd.concat(pieces, axis=0).sort_index()
            update = np.max(
                np.abs(aligned["aligned_time"].to_numpy(dtype=float) - previous)
            )
            aligned["alignment_iteration"] = iteration + 1
            if progress:
                print(
                    f"[align] stage {stage_idx + 1} iter {iteration + 1}/"
                    f"{max_iterations}: max update = {update:.4f}s "
                    f"(tolerance {tolerance_s:.4f}s)",
                    flush=True,
                )
            if update < tolerance_s:
                if progress:
                    print(
                        f"[align] stage {stage_idx + 1} converged after "
                        f"{iteration + 1} iterations",
                        flush=True,
                    )
                break

    if templates is None:
        raise ValueError("max_iterations must be at least 1")

    aligned["alignment_offset"] = aligned["aligned_time"] - aligned["prompt_time"]
    return aligned, templates


@dataclass(frozen=True)
class AdaptiveTemplateBank:
    """Templates with a PER-GESTURE time window (pre/post samples)."""

    templates: dict[str, np.ndarray]
    window_samples: dict[str, tuple[int, int]]  # name -> (pre_n, post_n)
    sample_rate: float

    def pre(self, name: str) -> int:
        return self.window_samples[name][0]

    def length(self, name: str) -> int:
        pre_n, post_n = self.window_samples[name]
        return pre_n + post_n


def _estimate_adaptive_templates(
    features: np.ndarray,
    times: np.ndarray,
    prompts: pd.DataFrame,
    window_samples: dict[str, tuple[int, int]],
    sample_rate: float,
    aligned_time_col: str,
    method: str,
    ridge: float,
) -> AdaptiveTemplateBank:
    features = np.asarray(features, dtype=np.float64)
    time_col = aligned_time_col if aligned_time_col in prompts.columns else "time"
    names = [str(n) for n in prompts["name"].drop_duplicates() if str(n) in window_samples]

    if method == "average":
        templates: dict[str, np.ndarray] = {}
        for name, group in prompts.groupby("name", sort=False):
            name = str(name)
            if name not in window_samples:
                continue
            pre_n, post_n = window_samples[name]
            windows = []
            for et in group[time_col].to_numpy(dtype=float):
                c = int(np.searchsorted(times, et))
                s, e = c - pre_n, c + post_n
                if s < 0 or e > len(features):
                    continue
                windows.append(features[s:e])
            if windows:
                templates[name] = np.mean(np.stack(windows, axis=0), axis=0)
        if not templates:
            raise ValueError("No adaptive templates could be estimated")
        return AdaptiveTemplateBank(templates, window_samples, sample_rate)

    # rERP with per-gesture (variable-length) lag regressors
    from scipy import sparse

    lengths = {n: window_samples[n][0] + window_samples[n][1] for n in names}
    offsets: dict[str, int] = {}
    off = 0
    for n in names:
        offsets[n] = off
        off += lengths[n]
    total = off
    if total > 20000:
        raise ValueError(f"adaptive rERP would create {total} params; reduce windows")
    rows, cols = [], []
    for row in prompts.itertuples():
        name = str(row.name)
        if name not in window_samples:
            continue
        c = int(np.searchsorted(times, float(getattr(row, time_col))))
        pre_n, post_n = window_samples[name]
        s, e = c - pre_n, c + post_n
        if s < 0 or e > len(features):
            continue
        rows.append(np.arange(s, e, dtype=np.int64))
        cols.append(offsets[name] + np.arange(lengths[name], dtype=np.int64))
    if not rows:
        raise ValueError("No valid events for adaptive rERP")
    drows = np.concatenate(rows)
    dcols = np.concatenate(cols)
    ddata = np.ones(len(drows), dtype=np.float64)
    ncol = total + 1  # intercept
    drows = np.concatenate([drows, np.arange(len(features), dtype=np.int64)])
    dcols = np.concatenate([dcols, np.full(len(features), total, dtype=np.int64)])
    ddata = np.concatenate([ddata, np.ones(len(features), dtype=np.float64)])
    design = sparse.coo_matrix((ddata, (drows, dcols)), shape=(len(features), ncol)).tocsr()
    xtx = (design.T @ design).toarray()
    if ridge > 0:
        reg = np.full(ncol, ridge, dtype=np.float64)
        reg[-1] = 0.0
        xtx.flat[:: ncol + 1] += reg
    xty = design.T @ features
    coef = np.linalg.solve(xtx, xty)[:total]
    templates = {n: coef[offsets[n]: offsets[n] + lengths[n]] for n in names}
    return AdaptiveTemplateBank(templates, window_samples, sample_rate)


def auto_window_samples(
    features: np.ndarray,
    times: np.ndarray,
    prompts: pd.DataFrame,
    sample_rate: float,
    pre_max_s: float = 0.5,
    post_max_s: float = 1.0,
    energy_threshold: float = 0.15,
    time_col: str = "time",
    min_pre_s: float = 0.05,
    min_post_s: float = 0.10,
) -> dict[str, tuple[int, int]]:
    """Derive a per-gesture (pre, post) window from each gesture's template energy
    support — no ground truth, no sweep. Estimates a long-window average template
    per gesture (at the given event times), finds the contiguous region around the
    energy peak that exceeds ``energy_threshold`` x peak, and crops the window to it.
    """

    pre_max = int(round(pre_max_s * sample_rate))
    post_max = int(round(post_max_s * sample_rate))
    bank = estimate_templates_average(
        features, times, _valid_sorted_prompts(prompts), pre_max, post_max,
        sample_rate, aligned_time_col=time_col,
    )
    min_pre = max(1, int(round(min_pre_s * sample_rate)))
    min_post = max(1, int(round(min_post_s * sample_rate)))
    out: dict[str, tuple[int, int]] = {}
    for name, tmpl in bank.templates.items():
        energy = np.sum(np.asarray(tmpl, dtype=np.float64) ** 2, axis=1)
        peak = int(np.argmax(energy))
        thr = energy[peak] * energy_threshold
        lo = peak
        while lo > 0 and energy[lo - 1] >= thr:
            lo -= 1
        hi = peak
        while hi < len(energy) - 1 and energy[hi + 1] >= thr:
            hi += 1
        # event reference sits at index pre_max within the template
        pre_n = max(min_pre, pre_max - lo)
        post_n = max(min_post, hi - pre_max + 1)
        out[str(name)] = (int(pre_n), int(post_n))
    return out


def _subtract_template_adaptive(residual, residual_start_idx, center_idx, template, pre_n):
    start = center_idx - pre_n
    stop = start + len(template)
    a = max(start, residual_start_idx)
    b = min(stop, residual_start_idx + len(residual))
    if a >= b:
        return
    ts = a - start
    rs = a - residual_start_idx
    residual[rs: rs + (b - a)] -= template[ts: ts + (b - a)]


def _align_single_event_adaptive(features, times, sequence, bank, uncertainty_window,
                                 candidate_step_s, prompt_prior_weight=0.0):
    row = sequence.iloc[0]
    name = str(row["name"])
    template = bank.templates[name]
    pre_n = bank.pre(name)
    L = bank.length(name)
    prompt_t = float(row["prompt_time"])
    candidates = _candidate_indices(
        times, prompt_t, uncertainty_window[0], uncertainty_window[1],
        candidate_step_s, bank.sample_rate,
    )
    halfwidth = max(abs(uncertainty_window[0]), abs(uncertainty_window[1]), 1e-9)
    cand, ssr, off = [], [], []
    for c in candidates:
        c = int(c); s = c - pre_n; e = s + L
        if s < 0 or e > len(features):
            continue
        cand.append(c)
        ssr.append(float(np.sum((features[s:e] - template) ** 2)))
        off.append((float(times[c]) - prompt_t) / halfwidth)
    if not cand:
        aligned = sequence.copy()
        aligned["aligned_time"] = float(times[int(np.searchsorted(times, prompt_t))])
        return aligned
    ssr = np.asarray(ssr)
    off = np.asarray(off)
    rng = float(ssr.max() - ssr.min())
    # normalize match cost to [0,1] so prompt_prior_weight is a meaningful mix weight
    ssr_norm = (ssr - ssr.min()) / rng if rng > 1e-12 else np.zeros_like(ssr)
    total = ssr_norm + prompt_prior_weight * off**2
    best = int(np.argmin(total))
    aligned = sequence.copy()
    aligned["aligned_time"] = float(times[cand[best]])
    return aligned


def _align_sequence_adaptive(features, times, sequence, bank, uncertainty_window,
                             beam_width, candidate_step_s, enforce_monotonic,
                             min_event_separation_s, prompt_prior_weight):
    if len(sequence) == 1:
        return _align_single_event_adaptive(
            features, times, sequence, bank, uncertainty_window, candidate_step_s,
            prompt_prior_weight)
    lower, upper = uncertainty_window
    halfwidth = max(abs(lower), abs(upper), 1e-9)
    cand_list = [
        _candidate_indices(times, float(r.prompt_time), lower, upper, candidate_step_s, bank.sample_rate)
        for r in sequence.itertuples()
    ]
    start_t = min(float(t) + lower for t in sequence["prompt_time"])
    end_t = max(float(t) + upper for t in sequence["prompt_time"])
    max_pre = max(bank.pre(str(n)) for n in sequence["name"])
    max_len = max(bank.length(str(n)) for n in sequence["name"])
    start_idx = max(0, int(np.searchsorted(times, start_t)) - max_pre)
    stop_idx = min(len(features), int(np.searchsorted(times, end_t)) + max_len)
    observed = features[start_idx:stop_idx]
    observed_energy = max(float(np.sum(observed**2)), 1e-12)  # normalize residual to ~[0,1]
    sep = int(round(min_event_separation_s * bank.sample_rate))
    prompt_times = sequence["prompt_time"].to_numpy(dtype=float)
    names = [str(n) for n in sequence["name"]]
    beams = [(0.0, observed.copy(), [], 0.0)]
    for pos, cands in enumerate(cand_list):
        nxt = []
        name = names[pos]
        tmpl = bank.templates[name]
        pre_n = bank.pre(name)
        for _, residual, chosen, prior in beams:
            for c in cands:
                c = int(c)
                if enforce_monotonic and chosen and c < chosen[-1] + sep:
                    continue
                nr = residual.copy()
                _subtract_template_adaptive(nr, start_idx, c, tmpl, pre_n)
                offset = float(times[c] - prompt_times[pos]) / halfwidth
                np_ = prior + prompt_prior_weight * offset**2
                resid_term = float(np.sum(nr**2)) / observed_energy
                nxt.append((resid_term + np_, nr, chosen + [c], np_))
        if not nxt:
            raise ValueError("adaptive beam search found no valid candidates")
        nxt.sort(key=lambda x: x[0])
        beams = nxt[:beam_width]
    best = beams[0][2]
    aligned = sequence.copy()
    aligned["aligned_time"] = [float(times[i]) for i in best]
    return aligned


def align_prompt_times_adaptive(
    features, times, prompts, window_samples, uncertainty_window,
    max_iterations=30, beam_width=30, candidate_step_s=0.02, tolerance_s=0.005,
    template_estimator="rerp", template_ridge=1e-3, enforce_monotonic=True,
    min_event_separation_s=0.0, prompt_prior_weight=0.0, progress=False,
):
    """EM alignment with a PER-GESTURE template window (window_samples: name->(pre_n,post_n))."""
    aligned = _valid_sorted_prompts(prompts).copy()
    aligned["prompt_time"] = aligned["time"].to_numpy(dtype=float)
    aligned["aligned_time"] = aligned["prompt_time"]
    sample_rate = infer_sample_rate(times)
    bank = None
    for it in range(max_iterations):
        bank = _estimate_adaptive_templates(
            features, times, aligned, window_samples, sample_rate,
            "aligned_time", template_estimator, template_ridge)
        prev = aligned["aligned_time"].to_numpy(dtype=float).copy()
        pieces = []
        for seq in group_overlapping_sequences(aligned, uncertainty_window):
            pieces.append(_align_sequence_adaptive(
                features, times, seq, bank, uncertainty_window, beam_width,
                candidate_step_s, enforce_monotonic, min_event_separation_s, prompt_prior_weight))
        aligned = pd.concat(pieces, axis=0).sort_index()
        update = float(np.max(np.abs(aligned["aligned_time"].to_numpy(dtype=float) - prev)))
        if progress:
            print(f"[adaptive] iter {it+1}/{max_iterations}: max update = {update:.4f}s", flush=True)
        if update < tolerance_s:
            if progress:
                print(f"[adaptive] converged after {it+1} iterations", flush=True)
            break
    aligned["alignment_offset"] = aligned["aligned_time"] - aligned["prompt_time"]
    return aligned, bank


def recenter_template_bank(templates: TemplateBank) -> TemplateBank:
    """Shift each template so its energy peak is centered on event time zero.

    Prompt-initialized templates can absorb a nearly constant reaction delay into
    the waveform itself. Re-centering makes the optimized timestamp correspond
    more closely to the gesture's dominant EMG event rather than to the original
    prompt-time reference. This is a lightweight analogue of the global
    recentering step described in the paper.
    """

    recentered = {}
    for name, template in templates.templates.items():
        energy = np.sum(template**2, axis=1)
        peak_idx = int(np.argmax(energy))
        shift = peak_idx - templates.pre_samples
        recentered[name] = _shift_with_zeros(template, -shift)
    return TemplateBank(
        templates=recentered,
        pre_samples=templates.pre_samples,
        post_samples=templates.post_samples,
        sample_rate=templates.sample_rate,
    )


def build_global_template_bank(template_banks: Sequence[TemplateBank]) -> TemplateBank:
    """Average session template banks into one global reference bank.

    The paper's final recentering step compares a participant/session template
    against a global template, defined as the grand average across participants.
    This helper builds that reference once several session-level template banks
    have already been estimated.
    """

    if not template_banks:
        raise ValueError("Need at least one template bank")

    reference = template_banks[0]
    for bank in template_banks[1:]:
        _validate_compatible_template_banks(reference, bank)

    templates: dict[str, np.ndarray] = {}
    names = sorted(
        set().union(*(set(bank.templates.keys()) for bank in template_banks))
    )
    for name in names:
        available = [bank.templates[name] for bank in template_banks if name in bank.templates]
        if available:
            templates[name] = np.mean(np.stack(available, axis=0), axis=0)

    return TemplateBank(
        templates=templates,
        pre_samples=reference.pre_samples,
        post_samples=reference.post_samples,
        sample_rate=reference.sample_rate,
    )


def estimate_template_recenter_shifts(
    session_templates: TemplateBank,
    reference_templates: TemplateBank,
    max_shift_s: float | None = None,
) -> dict[str, int]:
    """Estimate per-gesture shifts from session templates to a reference bank.

    The returned value is the integer sample shift that should be applied to the
    session template to maximize normalized correlation with the reference
    template of the same gesture. Positive shifts move the template later in
    template coordinates.
    """

    _validate_compatible_template_banks(session_templates, reference_templates)
    if max_shift_s is None:
        max_shift_samples = session_templates.length - 1
    else:
        if max_shift_s < 0:
            raise ValueError("max_shift_s must be non-negative")
        max_shift_samples = int(round(max_shift_s * session_templates.sample_rate))

    shifts: dict[str, int] = {}
    for name, template in session_templates.templates.items():
        if name not in reference_templates.templates:
            continue
        shifts[name] = _best_correlation_shift(
            template,
            reference_templates.templates[name],
            max_shift_samples=max_shift_samples,
        )
    return shifts


def recenter_template_bank_to_reference(
    session_templates: TemplateBank,
    reference_templates: TemplateBank,
    max_shift_s: float | None = None,
) -> tuple[TemplateBank, dict[str, int]]:
    """Shift session templates to best match a global/reference template bank."""

    shifts = estimate_template_recenter_shifts(
        session_templates,
        reference_templates,
        max_shift_s=max_shift_s,
    )
    recentered = {}
    for name, template in session_templates.templates.items():
        recentered[name] = _shift_with_zeros(template, shifts.get(name, 0))
    return (
        TemplateBank(
            templates=recentered,
            pre_samples=session_templates.pre_samples,
            post_samples=session_templates.post_samples,
            sample_rate=session_templates.sample_rate,
        ),
        shifts,
    )


def apply_recenter_shifts_to_aligned_prompts(
    aligned_prompts: pd.DataFrame,
    template_shifts: Mapping[str, int],
    sample_rate: float,
    time_col: str = "aligned_time",
) -> pd.DataFrame:
    """Apply template recentering shifts to aligned event timestamps.

    If a session template must be shifted by ``s`` samples to match the global
    reference, timestamps are shifted by ``-s / sample_rate`` seconds. This keeps
    the reconstructed signal on the same absolute time axis while changing the
    convention for what point inside the gesture waveform is called event time.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if time_col not in aligned_prompts.columns:
        raise ValueError(f"{time_col!r} is not present in aligned_prompts")

    adjusted = aligned_prompts.copy()
    shifts = adjusted["name"].astype(str).map(template_shifts).fillna(0).to_numpy(
        dtype=float
    )
    adjusted["recenter_shift_samples"] = shifts.astype(int)
    adjusted["recenter_shift_seconds"] = shifts / sample_rate
    adjusted[time_col] = adjusted[time_col].to_numpy(dtype=float) - shifts / sample_rate
    if "prompt_time" in adjusted.columns and time_col == "aligned_time":
        adjusted["alignment_offset"] = adjusted["aligned_time"] - adjusted["prompt_time"]
    return adjusted


def global_recenter_aligned_prompts(
    aligned_prompt_tables: Mapping[str, pd.DataFrame],
    session_template_banks: Mapping[str, TemplateBank],
    max_shift_s: float | None = None,
) -> tuple[TemplateBank, dict[str, pd.DataFrame], pd.DataFrame]:
    """Recenter multiple sessions against a global template reference.

    Parameters
    ----------
    aligned_prompt_tables:
        Mapping from session id to aligned prompts. Each table must include
        ``name`` and ``aligned_time`` columns. ``prompt_time`` is optional but
        allows ``alignment_offset`` to be updated after recentering.
    session_template_banks:
        Mapping from the same session ids to template banks estimated from those
        aligned prompts.
    max_shift_s:
        Optional bound on how far a session template can be shifted when matched
        against the global reference.

    Returns
    -------
    global_templates:
        Grand-average template bank across sessions.
    recentered_tables:
        Aligned prompt tables with recentered ``aligned_time`` values.
    shift_summary:
        Long-form table with one row per session and gesture shift.
    """

    if not aligned_prompt_tables:
        raise ValueError("Need at least one aligned prompt table")
    if set(aligned_prompt_tables) != set(session_template_banks):
        raise ValueError(
            "aligned_prompt_tables and session_template_banks must have identical keys"
        )

    global_templates = build_global_template_bank(list(session_template_banks.values()))
    recentered_tables: dict[str, pd.DataFrame] = {}
    shift_rows = []
    for session_id, template_bank in session_template_banks.items():
        _, shifts = recenter_template_bank_to_reference(
            template_bank,
            global_templates,
            max_shift_s=max_shift_s,
        )
        recentered_tables[session_id] = apply_recenter_shifts_to_aligned_prompts(
            aligned_prompt_tables[session_id],
            shifts,
            sample_rate=template_bank.sample_rate,
        )
        for name, shift_samples in sorted(shifts.items()):
            shift_rows.append(
                {
                    "session_id": session_id,
                    "name": name,
                    "shift_samples": int(shift_samples),
                    "shift_seconds": float(shift_samples / template_bank.sample_rate),
                }
            )

    return global_templates, recentered_tables, pd.DataFrame(shift_rows)


def save_aligned_prompts(
    aligned_prompts: pd.DataFrame,
    output_csv: str | Path,
) -> None:
    """Save aligned prompts to a CSV file."""

    output_csv = Path(output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned_prompts.to_csv(output_csv, index=False)


def _align_sequence(
    features: np.ndarray,
    times: np.ndarray,
    sequence: pd.DataFrame,
    templates: TemplateBank,
    uncertainty_window: tuple[float, float],
    beam_width: int,
    candidate_step_s: float,
    enforce_monotonic: bool = True,
    min_event_separation_s: float = 0.0,
    prompt_prior_weight: float = 0.0,
) -> pd.DataFrame:
    if len(sequence) == 1:
        return _align_single_event(
            features, times, sequence, templates, uncertainty_window, candidate_step_s
        )
    if min_event_separation_s < 0:
        raise ValueError("min_event_separation_s must be non-negative")
    if prompt_prior_weight < 0:
        raise ValueError("prompt_prior_weight must be non-negative")

    lower, upper = uncertainty_window
    candidate_indices = []
    for row in sequence.itertuples():
        candidates = _candidate_indices(
            times,
            float(row.prompt_time),
            lower,
            upper,
            candidate_step_s,
            templates.sample_rate,
        )
        candidate_indices.append(candidates)

    start_t = min(float(t) + lower for t in sequence["prompt_time"])
    end_t = max(float(t) + upper for t in sequence["prompt_time"])
    start_idx = max(0, int(np.searchsorted(times, start_t)) - templates.pre_samples)
    stop_idx = min(
        len(features), int(np.searchsorted(times, end_t)) + templates.post_samples
    )
    observed = features[start_idx:stop_idx]

    min_event_separation_samples = int(
        round(min_event_separation_s * templates.sample_rate)
    )
    prompt_times = sequence["prompt_time"].to_numpy(dtype=float)
    beams: list[tuple[float, np.ndarray, list[int], float]] = [
        (float(np.sum(observed**2)), observed.copy(), [], 0.0)
    ]
    for event_pos, candidates in enumerate(candidate_indices):
        name = str(sequence.iloc[event_pos]["name"])

        def expand(apply_monotonic: bool):
            produced = []
            for _, residual, chosen, prior_cost in beams:
                for candidate in candidates:
                    candidate = int(candidate)
                    if apply_monotonic and chosen:
                        min_allowed = chosen[-1] + min_event_separation_samples
                        if candidate < min_allowed:
                            continue
                    candidate_times = chosen + [int(candidate)]
                    next_residual = residual.copy()
                    _subtract_template(
                        next_residual,
                        start_idx,
                        candidate,
                        templates.templates[name],
                        templates,
                    )
                    offset = float(times[candidate] - prompt_times[event_pos])
                    next_prior_cost = prior_cost + prompt_prior_weight * offset**2
                    cost = float(np.sum(next_residual**2)) + next_prior_cost
                    produced.append(
                        (cost, next_residual, candidate_times, next_prior_cost)
                    )
            return produced

        next_beams = expand(enforce_monotonic)
        if not next_beams and enforce_monotonic:
            # No ordering-consistent placement exists for this event (can happen
            # with wide windows / coarse steps). Degrade gracefully for this
            # event rather than aborting the whole sequence.
            next_beams = expand(False)
        if not next_beams:
            raise ValueError(
                "Beam search produced no candidates. Try increasing the "
                "uncertainty window or candidate density."
            )
        next_beams.sort(key=lambda item: item[0])
        beams = next_beams[:beam_width]

    best_indices = beams[0][2]
    aligned = sequence.copy()
    aligned["aligned_time"] = [float(times[idx]) for idx in best_indices]
    return aligned


def _align_single_event(
    features: np.ndarray,
    times: np.ndarray,
    sequence: pd.DataFrame,
    templates: TemplateBank,
    uncertainty_window: tuple[float, float],
    candidate_step_s: float,
) -> pd.DataFrame:
    row = sequence.iloc[0]
    template = templates.templates[str(row["name"])]
    candidates = _candidate_indices(
        times,
        float(row["prompt_time"]),
        uncertainty_window[0],
        uncertainty_window[1],
        candidate_step_s,
        templates.sample_rate,
    )
    best_cost = np.inf
    best_idx = int(np.searchsorted(times, float(row["prompt_time"])))
    for candidate in candidates:
        start = int(candidate) - templates.pre_samples
        stop = start + templates.length
        if start < 0 or stop > len(features):
            continue
        cost = float(np.sum((features[start:stop] - template) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_idx = int(candidate)

    aligned = sequence.copy()
    aligned["aligned_time"] = float(times[best_idx])
    return aligned


def _candidate_indices(
    times: np.ndarray,
    prompt_time: float,
    lower_s: float,
    upper_s: float,
    candidate_step_s: float,
    sample_rate: float,
) -> np.ndarray:
    if candidate_step_s <= 0:
        raise ValueError("candidate_step_s must be positive")
    step = max(1, int(round(candidate_step_s * sample_rate)))
    lo = int(np.searchsorted(times, prompt_time + lower_s))
    hi = int(np.searchsorted(times, prompt_time + upper_s))
    lo = max(0, min(lo, len(times) - 1))
    hi = max(lo + 1, min(hi, len(times)))
    return np.arange(lo, hi, step, dtype=int)


def _subtract_template(
    residual: np.ndarray,
    residual_start_idx: int,
    center_idx: int,
    template: np.ndarray,
    templates: TemplateBank,
) -> None:
    start = center_idx - templates.pre_samples
    stop = start + templates.length
    overlap_start = max(start, residual_start_idx)
    overlap_stop = min(stop, residual_start_idx + len(residual))
    if overlap_start >= overlap_stop:
        return

    template_start = overlap_start - start
    template_stop = template_start + (overlap_stop - overlap_start)
    residual_start = overlap_start - residual_start_idx
    residual_stop = residual_start + (overlap_stop - overlap_start)
    residual[residual_start:residual_stop] -= template[template_start:template_stop]


def _shift_with_zeros(array: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros_like(array)
    if shift == 0:
        return array.copy()
    if abs(shift) >= len(array):
        return shifted
    if shift > 0:
        shifted[shift:] = array[:-shift]
    else:
        shifted[:shift] = array[-shift:]
    return shifted


def _best_correlation_shift(
    template: np.ndarray,
    reference: np.ndarray,
    max_shift_samples: int,
) -> int:
    if template.shape != reference.shape:
        raise ValueError("template and reference must have the same shape")
    if max_shift_samples < 0:
        raise ValueError("max_shift_samples must be non-negative")

    max_shift_samples = min(max_shift_samples, len(template) - 1)
    best_shift = 0
    best_score = -np.inf
    for shift in range(-max_shift_samples, max_shift_samples + 1):
        shifted = _shift_with_zeros(template, shift)
        score = _normalized_template_correlation(shifted, reference)
        if score > best_score:
            best_score = score
            best_shift = shift
    return best_shift


def _normalized_template_correlation(template: np.ndarray, reference: np.ndarray) -> float:
    template = np.asarray(template, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    template = template - template.mean(axis=0, keepdims=True)
    reference = reference - reference.mean(axis=0, keepdims=True)
    numerator = float(np.sum(template * reference))
    denominator = float(np.linalg.norm(template) * np.linalg.norm(reference))
    if denominator <= 1e-12:
        return -np.inf
    return numerator / denominator


def _validate_compatible_template_banks(
    first: TemplateBank,
    second: TemplateBank,
) -> None:
    if first.pre_samples != second.pre_samples:
        raise ValueError("Template banks must have the same pre_samples")
    if first.post_samples != second.post_samples:
        raise ValueError("Template banks must have the same post_samples")
    # Sample rates inferred from per-session feature timestamps can differ by
    # tiny floating-point amounts (e.g. 50.0025 vs 50.0031 Hz). Only flag banks
    # whose rates differ enough to matter for shift<->seconds conversion.
    if not np.isclose(first.sample_rate, second.sample_rate, rtol=1e-2, atol=1e-2):
        raise ValueError(
            f"Template banks must have the same sample_rate "
            f"({first.sample_rate} vs {second.sample_rate})"
        )
    shared = set(first.templates).intersection(second.templates)
    for name in shared:
        if first.templates[name].shape != second.templates[name].shape:
            raise ValueError(f"Template {name!r} has incompatible shape")


def _valid_sorted_prompts(prompts: pd.DataFrame) -> pd.DataFrame:
    required = {"name", "time"}
    missing = required.difference(prompts.columns)
    if missing:
        raise ValueError(f"prompts is missing required columns: {sorted(missing)}")
    prompts = prompts.dropna(subset=["name", "time"]).copy()
    return prompts.sort_values("time")


def format_sequence_stats(stats: SequenceStats) -> str:
    """Format prompt pattern statistics for CLI output."""

    lengths = np.asarray(stats.sequence_lengths, dtype=int)
    if len(lengths):
        seq_summary = (
            f"count={stats.num_sequences}, mean_len={lengths.mean():.2f}, "
            f"max_len={lengths.max()}, multi_prompt={(lengths > 1).sum()}"
        )
    else:
        seq_summary = "count=0"

    intervals = ", ".join(
        f"{key}={value:.4g}"
        for key, value in stats.prompt_interval_seconds.items()
    )
    return (
        f"prompts={stats.num_prompts}\n"
        f"prompt_intervals_seconds: {intervals}\n"
        f"uncertainty_sequences: {seq_summary}"
    )


def gesture_order_counts(prompts: pd.DataFrame, n: int = 20) -> pd.Series:
    """Return the most common adjacent gesture transitions."""

    prompts = _valid_sorted_prompts(prompts)
    names = prompts["name"].astype(str).to_numpy()
    transitions = [f"{a}->{b}" for a, b in zip(names[:-1], names[1:])]
    return pd.Series(transitions).value_counts().head(n)


def iter_hdf5_files(root: str | Path) -> Iterable[Path]:
    """Yield HDF5 files under a root path."""

    root = Path(root).expanduser()
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*.hdf5"))
    yield from sorted(root.rglob("*.h5"))
