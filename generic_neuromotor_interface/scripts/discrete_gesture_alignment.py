# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Inspect and align discrete gesture prompt timing."""

from __future__ import annotations

from pathlib import Path
import html

import click
import numpy as np
import pandas as pd

from generic_neuromotor_interface.discrete_gesture_alignment import (
    _candidate_indices,
    align_prompt_times,
    emg_envelope_features,
    estimate_templates,
    format_sequence_stats,
    gesture_order_counts,
    global_recenter_aligned_prompts,
    infer_sample_rate,
    iter_hdf5_files,
    load_discrete_gesture_hdf5,
    multivariate_power_frequency_features,
    save_aligned_prompts,
    summarize_prompt_pattern,
)


@click.group()
def main() -> None:
    """Discrete gesture prompt-pattern and alignment tools."""


@main.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--uncertainty",
    nargs=2,
    type=float,
    default=(-0.3, 0.8),
    show_default=True,
    help="Timing uncertainty window relative to prompt time, in seconds.",
)
@click.option(
    "--transitions",
    type=int,
    default=12,
    show_default=True,
    help="Number of adjacent gesture transitions to print.",
)
def inspect(data_path: Path, uncertainty: tuple[float, float], transitions: int) -> None:
    """Print prompt timing pattern statistics for one file or a directory."""

    for hdf5_path in iter_hdf5_files(data_path):
        _, prompts = load_discrete_gesture_hdf5(hdf5_path)
        stats = summarize_prompt_pattern(prompts, uncertainty)
        click.echo(f"\n{hdf5_path}")
        click.echo(format_sequence_stats(stats))
        counts = gesture_order_counts(prompts, transitions)
        if len(counts):
            click.echo("top_transitions:")
            click.echo(counts.to_string())


@main.command()
@click.argument("hdf5_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_svg", type=click.Path(path_type=Path))
@click.option(
    "--aligned-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="CSV produced by the align command. If omitted, only prompts are plotted.",
)
@click.option("--prompt-start", type=int, default=0, show_default=True)
@click.option("--num-prompts", type=int, default=30, show_default=True)
@click.option("--padding", type=float, default=1.0, show_default=True)
@click.option(
    "--aligned-offset",
    type=float,
    default=0.0,
    show_default=True,
    help="Extra offset in seconds added to aligned_time before plotting.",
)
@click.option(
    "--run-local-alignment",
    is_flag=True,
    default=False,
    help="Run a quick local envelope/average alignment for the selected prompts.",
)
@click.option("--pre", type=float, default=0.3, show_default=True)
@click.option("--post", type=float, default=0.9, show_default=True)
@click.option(
    "--uncertainty",
    nargs=2,
    type=float,
    default=(-0.3, 0.8),
    show_default=True,
    help="Timing uncertainty window used if --run-local-alignment is set.",
)
@click.option("--candidate-step", type=float, default=0.02, show_default=True)
@click.option("--beam-width", type=int, default=10, show_default=True)
@click.option("--smoothing-ms", type=float, default=50.0, show_default=True)
def plot(
    hdf5_path: Path,
    output_svg: Path,
    aligned_csv: Path | None,
    prompt_start: int,
    num_prompts: int,
    padding: float,
    aligned_offset: float,
    run_local_alignment: bool,
    pre: float,
    post: float,
    uncertainty: tuple[float, float],
    candidate_step: float,
    beam_width: int,
    smoothing_ms: float,
) -> None:
    """Plot raw sEMG envelope with prompt and aligned timestamps."""

    timeseries, prompts = load_discrete_gesture_hdf5(hdf5_path)
    prompts = prompts.sort_values("time").reset_index(drop=True)
    if prompt_start < 0 or num_prompts <= 0:
        raise click.BadParameter("prompt-start must be >= 0 and num-prompts > 0")
    selected = prompts.iloc[prompt_start : prompt_start + num_prompts].copy()
    if len(selected) == 0:
        raise click.BadParameter("No prompts selected by prompt-start/num-prompts")

    aligned = None
    if aligned_csv is not None:
        aligned = pd.read_csv(aligned_csv).sort_values("prompt_time")
    elif run_local_alignment:
        sample_rate = infer_sample_rate(timeseries["time"])
        features = emg_envelope_features(
            timeseries["emg"], sample_rate=sample_rate, smoothing_ms=smoothing_ms
        )
        aligned, _ = align_prompt_times(
            features=features,
            times=timeseries["time"],
            prompts=selected,
            pre_s=pre,
            post_s=post,
            uncertainty_window=uncertainty,
            max_iterations=3,
            beam_width=beam_width,
            candidate_step_s=candidate_step,
            recenter_templates=True,
            template_estimator="average",
            enforce_monotonic=True,
        )
        generated_csv = output_svg.with_suffix(".aligned_prompts.csv")
        aligned.to_csv(generated_csv, index=False)
        click.echo(f"Saved local aligned prompts to {generated_csv}")

    if aligned is not None:
        if "prompt_time" not in aligned.columns or "aligned_time" not in aligned.columns:
            raise click.BadParameter(
                "aligned CSV must contain prompt_time and aligned_time columns"
            )
        aligned = aligned[
            aligned["prompt_time"].between(
                float(selected["time"].min()) - padding,
                float(selected["time"].max()) + padding,
            )
        ].copy()
        aligned["plot_aligned_time"] = aligned["aligned_time"] + aligned_offset

    event_times = selected["time"].to_numpy(dtype=float)
    if aligned is not None and len(aligned):
        event_times = np.concatenate(
            [event_times, aligned["plot_aligned_time"].to_numpy(dtype=float)]
        )
    window_start = float(event_times.min()) - padding
    window_end = float(event_times.max()) + padding

    raw_times = timeseries["time"]
    start_idx, stop_idx = raw_times.searchsorted([window_start, window_end])
    window = timeseries[start_idx:stop_idx]
    if len(window) == 0:
        raise click.BadParameter("Selected time window has no EMG samples")

    sample_rate = infer_sample_rate(window["time"])
    envelope = _plot_envelope(window["emg"], sample_rate, smoothing_ms)
    _write_alignment_svg(
        output_svg,
        window["time"],
        envelope,
        selected,
        aligned,
        title=f"{hdf5_path.name}: prompts {prompt_start}-{prompt_start + len(selected) - 1}",
    )
    click.echo(f"Saved alignment plot to {output_svg}")


@main.command()
@click.argument("hdf5_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_csv", type=click.Path(path_type=Path))
@click.option("--pre", type=float, default=0.3, show_default=True)
@click.option("--post", type=float, default=0.9, show_default=True)
@click.option(
    "--uncertainty",
    nargs=2,
    type=float,
    default=(-0.3, 0.8),
    show_default=True,
    help="Timing uncertainty window relative to prompt time, in seconds.",
)
@click.option("--iterations", type=int, default=5, show_default=True)
@click.option("--beam-width", type=int, default=30, show_default=True)
@click.option("--candidate-step", type=float, default=0.02, show_default=True)
@click.option(
    "--monotonic/--no-monotonic",
    default=True,
    show_default=True,
    help="Constrain aligned event times to follow prompt order within a sequence.",
)
@click.option(
    "--min-event-separation",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum allowed separation between aligned events in a sequence, seconds.",
)
@click.option(
    "--prompt-prior-weight",
    type=float,
    default=0.0,
    show_default=True,
    help="Quadratic penalty weight for moving aligned times away from prompt times.",
)
@click.option("--smoothing-ms", type=float, default=50.0, show_default=True)
@click.option(
    "--template-estimator",
    type=click.Choice(["rerp", "average"]),
    default="rerp",
    show_default=True,
    help="Template estimation method used inside iterative alignment.",
)
@click.option(
    "--template-ridge",
    type=float,
    default=1e-3,
    show_default=True,
    help="Ridge regularization for rERP template estimation.",
)
@click.option(
    "--feature",
    type=click.Choice(["mpf", "envelope"]),
    default="mpf",
    show_default=True,
    help="Feature representation used for alignment.",
)
@click.option("--mpf-window-length", type=int, default=200, show_default=True)
@click.option("--mpf-stride", type=int, default=40, show_default=True)
@click.option("--mpf-n-fft", type=int, default=64, show_default=True)
@click.option("--mpf-fft-stride", type=int, default=10, show_default=True)
@click.option(
    "--mpf-chunk-output-frames",
    type=int,
    default=4096,
    show_default=True,
    help="Number of downsampled MPF frames to compute per chunk.",
)
@click.option(
    "--recenter/--no-recenter",
    default=True,
    show_default=True,
    help="Recenter templates to their energy peak before time search.",
)
def align(
    hdf5_path: Path,
    output_csv: Path,
    pre: float,
    post: float,
    uncertainty: tuple[float, float],
    iterations: int,
    beam_width: int,
    candidate_step: float,
    monotonic: bool,
    min_event_separation: float,
    prompt_prior_weight: float,
    smoothing_ms: float,
    template_estimator: str,
    template_ridge: float,
    feature: str,
    mpf_window_length: int,
    mpf_stride: int,
    mpf_n_fft: int,
    mpf_fft_stride: int,
    mpf_chunk_output_frames: int,
    recenter: bool,
) -> None:
    """Run iterative forced alignment and save aligned prompts as CSV."""

    timeseries, prompts = load_discrete_gesture_hdf5(hdf5_path)
    times = timeseries["time"]
    sample_rate = infer_sample_rate(times)
    if feature == "mpf":
        features, feature_times = multivariate_power_frequency_features(
            timeseries["emg"],
            times,
            window_length=mpf_window_length,
            stride=mpf_stride,
            n_fft=mpf_n_fft,
            fft_stride=mpf_fft_stride,
            fs=sample_rate,
            chunk_output_frames=mpf_chunk_output_frames,
        )
    else:
        features = emg_envelope_features(
            timeseries["emg"], sample_rate=sample_rate, smoothing_ms=smoothing_ms
        )
        feature_times = times
    aligned, _ = align_prompt_times(
        features=features,
        times=feature_times,
        prompts=prompts,
        pre_s=pre,
        post_s=post,
        uncertainty_window=uncertainty,
        max_iterations=iterations,
        beam_width=beam_width,
        candidate_step_s=candidate_step,
        recenter_templates=recenter,
        template_estimator=template_estimator,
        template_ridge=template_ridge,
        enforce_monotonic=monotonic,
        min_event_separation_s=min_event_separation,
        prompt_prior_weight=prompt_prior_weight,
    )
    save_aligned_prompts(aligned, output_csv)
    click.echo(f"Saved {len(aligned)} aligned prompts to {output_csv}")
    click.echo(
        aligned["alignment_offset"]
        .describe(percentiles=[0.05, 0.5, 0.95])
        .to_string()
    )


@main.command("global-recenter")
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.argument("aligned_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--aligned-suffix",
    type=str,
    default="_aligned_prompts.csv",
    show_default=True,
    help="Aligned CSV filename suffix appended to each HDF5 stem.",
)
@click.option("--pre", type=float, default=0.3, show_default=True)
@click.option("--post", type=float, default=0.9, show_default=True)
@click.option("--max-shift", type=float, default=0.25, show_default=True)
@click.option("--smoothing-ms", type=float, default=50.0, show_default=True)
@click.option(
    "--template-estimator",
    type=click.Choice(["rerp", "average"]),
    default="rerp",
    show_default=True,
    help="Template estimation method used to rebuild session templates.",
)
@click.option(
    "--template-ridge",
    type=float,
    default=1e-3,
    show_default=True,
    help="Ridge regularization for rERP template estimation.",
)
@click.option(
    "--feature",
    type=click.Choice(["mpf", "envelope"]),
    default="mpf",
    show_default=True,
    help="Feature representation used to rebuild templates.",
)
@click.option("--mpf-window-length", type=int, default=200, show_default=True)
@click.option("--mpf-stride", type=int, default=40, show_default=True)
@click.option("--mpf-n-fft", type=int, default=64, show_default=True)
@click.option("--mpf-fft-stride", type=int, default=10, show_default=True)
@click.option(
    "--mpf-chunk-output-frames",
    type=int,
    default=4096,
    show_default=True,
    help="Number of downsampled MPF frames to compute per chunk.",
)
def global_recenter(
    data_path: Path,
    aligned_dir: Path,
    output_dir: Path,
    aligned_suffix: str,
    pre: float,
    post: float,
    max_shift: float,
    smoothing_ms: float,
    template_estimator: str,
    template_ridge: float,
    feature: str,
    mpf_window_length: int,
    mpf_stride: int,
    mpf_n_fft: int,
    mpf_fft_stride: int,
    mpf_chunk_output_frames: int,
) -> None:
    """Apply global template recentering across multiple aligned sessions.

    The command expects one aligned CSV per HDF5 file, named
    ``{hdf5_stem}{aligned_suffix}`` inside ``aligned_dir``. It rebuilds each
    session template from the aligned timestamps, builds a grand-average global
    template bank, finds per-gesture correlation shifts for every session, and
    writes recentered aligned prompt CSVs to ``output_dir``.
    """

    hdf5_paths = list(iter_hdf5_files(data_path))
    if not hdf5_paths:
        raise click.BadParameter("No HDF5 files found in data_path")

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_prompt_tables = {}
    session_template_banks = {}

    for hdf5_path in hdf5_paths:
        session_id = hdf5_path.stem
        aligned_csv = aligned_dir / f"{session_id}{aligned_suffix}"
        if not aligned_csv.exists():
            click.echo(f"Skipping {session_id}: missing {aligned_csv}", err=True)
            continue

        timeseries, _ = load_discrete_gesture_hdf5(hdf5_path)
        aligned = pd.read_csv(aligned_csv)
        if "aligned_time" not in aligned.columns:
            raise click.BadParameter(f"{aligned_csv} must contain aligned_time")
        if "time" not in aligned.columns:
            aligned = aligned.copy()
            aligned["time"] = aligned["aligned_time"]

        times = timeseries["time"]
        sample_rate = infer_sample_rate(times)
        if feature == "mpf":
            features, feature_times = multivariate_power_frequency_features(
                timeseries["emg"],
                times,
                window_length=mpf_window_length,
                stride=mpf_stride,
                n_fft=mpf_n_fft,
                fft_stride=mpf_fft_stride,
                fs=sample_rate,
                chunk_output_frames=mpf_chunk_output_frames,
            )
        else:
            features = emg_envelope_features(
                timeseries["emg"],
                sample_rate=sample_rate,
                smoothing_ms=smoothing_ms,
            )
            feature_times = times

        session_template_banks[session_id] = estimate_templates(
            features=features,
            times=feature_times,
            prompts=aligned,
            pre_s=pre,
            post_s=post,
            aligned_time_col="aligned_time",
            method=template_estimator,
            ridge=template_ridge,
        )
        aligned_prompt_tables[session_id] = aligned
        click.echo(f"Loaded session templates for {session_id}")

    if not session_template_banks:
        raise click.BadParameter("No aligned sessions were found")

    _, recentered_tables, shift_summary = global_recenter_aligned_prompts(
        aligned_prompt_tables,
        session_template_banks,
        max_shift_s=max_shift,
    )

    for session_id, recentered in recentered_tables.items():
        output_csv = output_dir / f"{session_id}_global_recentered_prompts.csv"
        save_aligned_prompts(recentered, output_csv)
        click.echo(f"Saved recentered prompts to {output_csv}")

    summary_csv = output_dir / "global_recenter_shifts.csv"
    shift_summary.to_csv(summary_csv, index=False)
    click.echo(f"Saved recenter shifts to {summary_csv}")
    if len(shift_summary):
        click.echo(
            shift_summary["shift_seconds"]
            .describe(percentiles=[0.05, 0.5, 0.95])
            .to_string()
        )


@main.command("simulate-shift-eval")
@click.argument("hdf5_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--prompt-start", type=int, default=0, show_default=True)
@click.option(
    "--num-prompts",
    type=int,
    default=200,
    show_default=True,
    help="Number of consecutive prompts to perturb and align.",
)
@click.option(
    "--shift-range",
    nargs=2,
    type=float,
    default=(-0.25, 0.25),
    show_default=True,
    help="Uniform random label shift range in seconds, used by --shift-model uniform.",
)
@click.option(
    "--shift-model",
    type=click.Choice(["prompt-delay", "uniform"]),
    default="prompt-delay",
    show_default=True,
    help=(
        "How to synthesize prompt times. prompt-delay makes prompt_time earlier "
        "than ground-truth event_time by a positive reaction delay."
    ),
)
@click.option(
    "--delay-mean",
    type=float,
    default=0.25,
    show_default=True,
    help="Mean reaction delay in seconds for --shift-model prompt-delay.",
)
@click.option(
    "--delay-std",
    type=float,
    default=0.07,
    show_default=True,
    help="Reaction-delay standard deviation for --shift-model prompt-delay.",
)
@click.option(
    "--delay-range",
    nargs=2,
    type=float,
    default=(0.05, 0.55),
    show_default=True,
    help="Min/max positive reaction delay for --shift-model prompt-delay.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--pre", type=float, default=0.3, show_default=True)
@click.option("--post", type=float, default=0.9, show_default=True)
@click.option(
    "--uncertainty",
    nargs=2,
    type=float,
    default=(-0.35, 0.35),
    show_default=True,
    help="Search window around shifted label time, in seconds.",
)
@click.option("--iterations", type=int, default=5, show_default=True)
@click.option("--beam-width", type=int, default=20, show_default=True)
@click.option("--candidate-step", type=float, default=0.02, show_default=True)
@click.option("--min-event-separation", type=float, default=0.0, show_default=True)
@click.option("--prompt-prior-weight", type=float, default=0.0, show_default=True)
@click.option("--smoothing-ms", type=float, default=50.0, show_default=True)
@click.option(
    "--template-estimator",
    type=click.Choice(["rerp", "average"]),
    default="rerp",
    show_default=True,
)
@click.option("--template-ridge", type=float, default=1e-3, show_default=True)
@click.option(
    "--feature",
    type=click.Choice(["mpf", "envelope"]),
    default="mpf",
    show_default=True,
    help="Use envelope for quick checks or MPF for heavier paper-style runs.",
)
@click.option("--mpf-window-length", type=int, default=200, show_default=True)
@click.option("--mpf-stride", type=int, default=40, show_default=True)
@click.option("--mpf-n-fft", type=int, default=64, show_default=True)
@click.option("--mpf-fft-stride", type=int, default=10, show_default=True)
@click.option("--mpf-chunk-output-frames", type=int, default=4096, show_default=True)
@click.option(
    "--plot-prompts",
    type=int,
    default=8,
    show_default=True,
    help="Number of prompts to show in the multichannel SVG.",
)
@click.option("--plot-left", type=float, default=0.45, show_default=True)
@click.option("--plot-right", type=float, default=0.55, show_default=True)
@click.option(
    "--recenter/--no-recenter",
    default=False,
    show_default=True,
    help="Optional per-iteration energy-peak recentering. Keep off for paper-style EM.",
)
@click.option(
    "--with-oracle/--no-oracle",
    default=False,
    show_default=True,
    help="Also compute an oracle upper bound (templates built from ground-truth "
    "times) to separate the search core from template-estimation quality.",
)
@click.option(
    "--oracle-init/--no-oracle-init",
    default=False,
    show_default=True,
    help="Seed EM's FIRST iteration with oracle templates (built from ground "
    "truth), then re-estimate normally. Tests whether a good init lets EM "
    "sustain a correct solution or degrade back to the perturbed fixed point.",
)
@click.option(
    "--coarse-to-fine",
    type=int,
    default=0,
    show_default=True,
    help="If >0, anneal the search window over N stages from narrow to the full "
    "--uncertainty target, to stop the cold-start template from scattering "
    "events. 0 disables (single-stage).",
)
def simulate_shift_eval(
    hdf5_path: Path,
    output_dir: Path,
    prompt_start: int,
    num_prompts: int,
    shift_range: tuple[float, float],
    shift_model: str,
    delay_mean: float,
    delay_std: float,
    delay_range: tuple[float, float],
    seed: int,
    pre: float,
    post: float,
    uncertainty: tuple[float, float],
    iterations: int,
    beam_width: int,
    candidate_step: float,
    min_event_separation: float,
    prompt_prior_weight: float,
    smoothing_ms: float,
    template_estimator: str,
    template_ridge: float,
    feature: str,
    mpf_window_length: int,
    mpf_stride: int,
    mpf_n_fft: int,
    mpf_fft_stride: int,
    mpf_chunk_output_frames: int,
    plot_prompts: int,
    plot_left: float,
    plot_right: float,
    recenter: bool,
    with_oracle: bool,
    oracle_init: bool,
    coarse_to_fine: int,
) -> None:
    """Randomly perturb event labels, align them, and evaluate against original labels.

    Public ``prompts.time`` is treated as pseudo ground truth. This command
    creates a shifted-label proxy for raw prompt times, runs time alignment, and
    writes CSV metrics plus a separated multi-channel visualization.
    """

    if num_prompts <= 0:
        raise click.BadParameter("num-prompts must be positive")
    if shift_range[0] > shift_range[1]:
        raise click.BadParameter("shift-range must be ordered as low high")
    if delay_range[0] < 0 or delay_range[0] > delay_range[1]:
        raise click.BadParameter("delay-range must be ordered positive min max")
    if delay_std < 0:
        raise click.BadParameter("delay-std must be non-negative")
    if shift_model == "prompt-delay":
        effective_shift_range = (-delay_range[1], -delay_range[0])
    else:
        effective_shift_range = shift_range
    if uncertainty[0] > min(effective_shift_range[0], 0.0) or uncertainty[1] < max(
        effective_shift_range[1], 0.0
    ):
        click.echo(
            "Warning: uncertainty window does not fully cover synthetic shift range "
            "and zero; "
            "some ground-truth times may be unreachable.",
            err=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Loading HDF5: {hdf5_path}", err=True)
    timeseries, prompts = load_discrete_gesture_hdf5(hdf5_path)
    prompts = prompts.sort_values("time").reset_index(drop=True)
    selected = prompts.iloc[prompt_start : prompt_start + num_prompts].copy()
    if len(selected) == 0:
        raise click.BadParameter("No prompts selected")
    click.echo(
        f"Selected {len(selected)} prompts "
        f"from index {prompt_start} to {prompt_start + len(selected) - 1}",
        err=True,
    )

    rng = np.random.default_rng(seed)
    if shift_model == "prompt-delay":
        if delay_std == 0:
            delays = np.full(len(selected), delay_mean, dtype=float)
        else:
            delays = rng.normal(delay_mean, delay_std, size=len(selected))
        delays = np.clip(delays, delay_range[0], delay_range[1])
        shifts = -delays
    else:
        delays = np.full(len(selected), np.nan, dtype=float)
        shifts = rng.uniform(shift_range[0], shift_range[1], size=len(selected))
    shifted_prompts = selected.copy()
    shifted_prompts["ground_truth_time"] = selected["time"].to_numpy(dtype=float)
    shifted_prompts["random_shift_seconds"] = shifts
    shifted_prompts["simulated_reaction_delay_seconds"] = delays
    shifted_prompts["shift_model"] = shift_model
    shifted_prompts["time"] = shifted_prompts["ground_truth_time"] + shifts
    shifted_prompts["original_prompt_index"] = selected.index.to_numpy(dtype=int)
    if shift_model == "prompt-delay":
        click.echo(
            "Created simulated prompt-time labels: "
            f"delay range=({delays.min():.3f}, {delays.max():.3f})s, "
            f"delay mean={delays.mean():.3f}s; "
            "synthetic prompt time is earlier than ground truth.",
            err=True,
        )
    else:
        click.echo(
            "Created simulated shifted labels: "
            f"range=({shifts.min():+.3f}, {shifts.max():+.3f})s, "
            f"mean={shifts.mean():+.3f}s",
            err=True,
        )

    times = timeseries["time"]
    sample_rate = infer_sample_rate(times)
    click.echo(
        f"Preparing {feature} features from {len(timeseries):,} EMG samples "
        f"at {sample_rate:.1f} Hz",
        err=True,
    )
    if feature == "mpf":
        features, feature_times = multivariate_power_frequency_features(
            timeseries["emg"],
            times,
            window_length=mpf_window_length,
            stride=mpf_stride,
            n_fft=mpf_n_fft,
            fft_stride=mpf_fft_stride,
            fs=sample_rate,
            chunk_output_frames=mpf_chunk_output_frames,
            progress=True,
        )
    else:
        features = emg_envelope_features(
            timeseries["emg"], sample_rate=sample_rate, smoothing_ms=smoothing_ms
        )
        feature_times = times
    click.echo(
        f"Feature matrix ready: shape={features.shape}, "
        f"time_points={len(feature_times):,}",
        err=True,
    )

    click.echo(
        "Running time alignment: "
        f"template_estimator={template_estimator}, iterations={iterations}, "
        f"beam_width={beam_width}, candidate_step={candidate_step}s",
        err=True,
    )
    uncertainty_schedule = None
    if coarse_to_fine > 0:
        uncertainty_schedule = [
            (uncertainty[0] * k / coarse_to_fine, uncertainty[1] * k / coarse_to_fine)
            for k in range(1, coarse_to_fine + 1)
        ]
        click.echo(
            f"Coarse-to-fine annealing over {coarse_to_fine} stages: "
            f"{[(round(a,3), round(b,3)) for a, b in uncertainty_schedule]}",
            err=True,
        )

    init_bank = None
    if oracle_init:
        click.echo(
            "Seeding EM iteration 1 with oracle templates (from ground truth).",
            err=True,
        )
        truth = shifted_prompts.copy()
        truth["time"] = truth["ground_truth_time"].to_numpy(dtype=float)
        init_bank = estimate_templates(
            features, feature_times, truth, pre, post,
            aligned_time_col="time", method=template_estimator, ridge=template_ridge,
        )

    aligned, _ = align_prompt_times(
        features=features,
        times=feature_times,
        prompts=shifted_prompts[["name", "time", "original_prompt_index"]],
        pre_s=pre,
        post_s=post,
        uncertainty_window=uncertainty,
        max_iterations=iterations,
        beam_width=beam_width,
        candidate_step_s=candidate_step,
        recenter_templates=recenter,
        template_estimator=template_estimator,
        template_ridge=template_ridge,
        enforce_monotonic=True,
        min_event_separation_s=min_event_separation,
        prompt_prior_weight=prompt_prior_weight,
        progress=True,
        init_templates=init_bank,
        uncertainty_schedule=uncertainty_schedule,
    )
    click.echo("Alignment finished. Computing errors against ground truth.", err=True)

    shifted_by_index = shifted_prompts.sort_index()
    aligned = aligned.sort_index().copy()
    aligned["ground_truth_time"] = shifted_by_index["ground_truth_time"].to_numpy(
        dtype=float
    )
    aligned["random_shift_seconds"] = shifted_by_index[
        "random_shift_seconds"
    ].to_numpy(dtype=float)
    aligned["simulated_reaction_delay_seconds"] = shifted_by_index[
        "simulated_reaction_delay_seconds"
    ].to_numpy(dtype=float)
    aligned["shift_model"] = shifted_by_index["shift_model"].to_numpy()
    aligned["original_prompt_index"] = shifted_by_index[
        "original_prompt_index"
    ].to_numpy(dtype=int)
    aligned["error_before"] = aligned["prompt_time"] - aligned["ground_truth_time"]
    aligned["error_after"] = aligned["aligned_time"] - aligned["ground_truth_time"]
    aligned["abs_error_before"] = np.abs(aligned["error_before"])
    aligned["abs_error_after"] = np.abs(aligned["error_after"])
    aligned["error_improvement"] = (
        aligned["abs_error_before"] - aligned["abs_error_after"]
    )

    if with_oracle:
        click.echo(
            "Computing oracle upper bound (templates from ground-truth times).",
            err=True,
        )
        oracle_times = _oracle_align_times(
            features, feature_times, aligned, pre, post, uncertainty, candidate_step
        )
        aligned["oracle_time"] = oracle_times
        aligned["error_oracle"] = aligned["oracle_time"] - aligned["ground_truth_time"]
        aligned["abs_error_oracle"] = np.abs(aligned["error_oracle"])

    stem = hdf5_path.stem
    shifted_csv = output_dir / f"{stem}_simulated_shifted_prompts.csv"
    aligned_csv = output_dir / f"{stem}_simulated_aligned_prompts.csv"
    metrics_csv = output_dir / f"{stem}_simulated_alignment_metrics.csv"
    plot_svg = output_dir / f"{stem}_simulated_multichannel_alignment.svg"
    shifted_prompts.to_csv(shifted_csv, index=False)
    aligned.to_csv(aligned_csv, index=False)
    _write_simulation_metrics(aligned, metrics_csv)
    click.echo("Writing multi-channel SVG visualization.", err=True)
    _write_multichannel_simulation_svg(
        plot_svg,
        timeseries,
        aligned,
        max_prompts=plot_prompts,
        left_s=plot_left,
        right_s=plot_right,
        title=f"{stem}: simulated shifted labels vs recovered alignment",
    )

    click.echo(f"Saved shifted prompts to {shifted_csv}")
    click.echo(f"Saved aligned prompts to {aligned_csv}")
    click.echo(f"Saved metrics to {metrics_csv}")
    click.echo(f"Saved multichannel plot to {plot_svg}")
    summary_cols = ["abs_error_before", "abs_error_after", "error_improvement"]
    if with_oracle:
        summary_cols.insert(2, "abs_error_oracle")
    click.echo(
        aligned[summary_cols].describe(percentiles=[0.05, 0.5, 0.95]).to_string()
    )
    if with_oracle:
        click.echo(
            "\nOracle = upper bound with ground-truth templates. If abs_error_after "
            "is far above abs_error_oracle, the gap is template-estimation/EM "
            "convergence, not the search core or task difficulty."
        )


def _oracle_align_times(
    features: np.ndarray,
    feature_times: np.ndarray,
    aligned: pd.DataFrame,
    pre: float,
    post: float,
    uncertainty: tuple[float, float],
    candidate_step: float,
) -> np.ndarray:
    """Upper-bound control: build templates from the TRUE (ground_truth) times,
    then do a single matched-filter search around each shifted prompt time.

    This 'cheats' by using ground truth to estimate sharp, correctly-centered
    templates, isolating the search core from the (failing) self-bootstrapped
    template estimation. Returns the oracle-aligned time for each row of
    ``aligned`` (index-aligned).
    """

    truth = aligned.copy()
    truth["time"] = truth["ground_truth_time"].to_numpy(dtype=float)
    bank = estimate_templates(
        features, feature_times, truth, pre, post,
        aligned_time_col="time", method="average",
    )
    pre_n, post_n, length = bank.pre_samples, bank.post_samples, bank.length

    out = np.full(len(aligned), np.nan)
    prompt_times = aligned["prompt_time"].to_numpy(dtype=float)
    names = aligned["name"].astype(str).to_numpy()
    for i in range(len(aligned)):
        template = bank.templates.get(names[i])
        if template is None:
            out[i] = prompt_times[i]
            continue
        cands = _candidate_indices(
            feature_times, prompt_times[i],
            uncertainty[0], uncertainty[1], candidate_step, bank.sample_rate,
        )
        best_cost, best_idx = np.inf, int(
            np.searchsorted(feature_times, prompt_times[i])
        )
        for c in cands:
            c = int(c)
            start = c - pre_n
            stop = start + length
            if start < 0 or stop > len(features):
                continue
            cost = float(np.sum((features[start:stop] - template) ** 2))
            if cost < best_cost:
                best_cost, best_idx = cost, c
        out[i] = float(feature_times[best_idx])
    return out


def _write_simulation_metrics(aligned: pd.DataFrame, output_csv: Path) -> None:
    rows = [
        {
            "metric": "count",
            "value": float(len(aligned)),
        },
        {
            "metric": "mae_before",
            "value": float(aligned["abs_error_before"].mean()),
        },
        {
            "metric": "mae_after",
            "value": float(aligned["abs_error_after"].mean()),
        },
        {
            "metric": "median_abs_error_before",
            "value": float(aligned["abs_error_before"].median()),
        },
        {
            "metric": "median_abs_error_after",
            "value": float(aligned["abs_error_after"].median()),
        },
        {
            "metric": "mean_improvement",
            "value": float(aligned["error_improvement"].mean()),
        },
        {
            "metric": "fraction_improved",
            "value": float((aligned["error_improvement"] > 0).mean()),
        },
    ]
    if "abs_error_oracle" in aligned.columns:
        rows.extend(
            [
                {
                    "metric": "mae_oracle",
                    "value": float(aligned["abs_error_oracle"].mean()),
                },
                {
                    "metric": "median_abs_error_oracle",
                    "value": float(aligned["abs_error_oracle"].median()),
                },
            ]
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def _write_multichannel_simulation_svg(
    output_svg: Path,
    timeseries: np.ndarray,
    aligned: pd.DataFrame,
    max_prompts: int,
    left_s: float,
    right_s: float,
    title: str,
) -> None:
    if max_prompts <= 0 or len(aligned) == 0:
        return
    if left_s <= 0 or right_s <= 0:
        raise click.BadParameter("plot-left and plot-right must be positive")

    examples = aligned.sort_values("abs_error_before", ascending=False).head(
        max_prompts
    )
    raw_times = timeseries["time"]
    raw_emg = timeseries["emg"]

    width = 1600
    left_margin = 95
    right_margin = 45
    top = 90
    panel_h = 360
    gap = 54
    panel_w = width - left_margin - right_margin
    height = top + len(examples) * panel_h + (len(examples) - 1) * gap + 105
    plot_pad_l = 64
    plot_pad_r = 20
    plot_pad_t = 45
    plot_pad_b = 38
    inner_w = panel_w - plot_pad_l - plot_pad_r
    inner_h = panel_h - plot_pad_t - plot_pad_b

    colors = {
        "ground_truth": "#16a34a",
        "shifted": "#2563eb",
        "aligned": "#dc2626",
        "signal": "#111827",
        "grid": "#e5e7eb",
        "muted": "#4b5563",
        "text": "#111827",
    }
    channel_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#60a5fa",
        "#fb923c",
        "#4ade80",
        "#f87171",
        "#c084fc",
        "#94a3b8",
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left_margin}" y="35" font-family="Arial" font-size="25" '
        f'font-weight="700" fill="{colors["text"]}">{html.escape(title)}</text>',
        f'<text x="{left_margin}" y="62" font-family="Arial" font-size="14" '
        f'fill="{colors["muted"]}">Raw sEMG is unchanged. Only event labels are '
        "randomly shifted and then recovered by alignment. "
        "Bold lines = focus event; faded thin lines = neighbouring gestures in window.</text>",
    ]

    for panel_idx, (_, row) in enumerate(examples.iterrows()):
        center = float(row["ground_truth_time"])
        window_start = center - left_s
        window_end = center + right_s
        start_idx, stop_idx = raw_times.searchsorted([window_start, window_end])
        window_emg = raw_emg[start_idx:stop_idx].astype(float)
        window_times = raw_times[start_idx:stop_idx]
        if len(window_emg) == 0:
            continue

        display = _robust_multichannel_display(window_emg)
        step = max(1, len(display) // 1200)
        display = display[::step]
        window_times = window_times[::step]

        panel_x = left_margin
        panel_y = top + panel_idx * (panel_h + gap)
        plot_x = panel_x + plot_pad_l
        plot_y = panel_y + plot_pad_t
        n_channels = display.shape[1]
        lane_h = inner_h / n_channels

        def sx(value: float) -> float:
            return plot_x + (value - window_start) / (window_end - window_start) * inner_w

        parts.extend(
            [
                f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" '
                f'height="{panel_h}" fill="#ffffff" stroke="#d1d5db" rx="4"/>',
                f'<text x="{panel_x + 14}" y="{panel_y + 25}" font-family="Arial" '
                f'font-size="16" font-weight="700" fill="{colors["text"]}">'
                f'#{int(row["original_prompt_index"])} {html.escape(str(row["name"]))}'
                "</text>",
                f'<text x="{panel_x + panel_w - 14}" y="{panel_y + 25}" '
                f'font-family="Arial" font-size="12" text-anchor="end" '
                f'fill="{colors["muted"]}">before {row["error_before"]:+.3f}s; '
                f'after {row["error_after"]:+.3f}s</text>',
                f'<rect x="{plot_x}" y="{plot_y}" width="{inner_w}" '
                f'height="{inner_h}" fill="#f9fafb" stroke="#e5e7eb"/>',
            ]
        )

        tick_start = np.ceil(-left_s / 0.1) * 0.1
        tick_stop = np.floor(right_s / 0.1) * 0.1
        for rel in np.arange(tick_start, tick_stop + 1e-9, 0.1):
            x = sx(center + float(rel))
            stroke = "#d1d5db" if abs(rel) < 1e-9 else colors["grid"]
            parts.append(
                f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{plot_y}" '
                f'y2="{plot_y + inner_h}" stroke="{stroke}" stroke-width="0.8"/>'
            )
            parts.append(
                f'<text x="{x:.2f}" y="{plot_y + inner_h + 17}" '
                f'font-family="Arial" font-size="10" text-anchor="middle" '
                f'fill="{colors["muted"]}">{rel:+.1f}</text>'
            )

        # Faded thin lines for ALL OTHER events falling in this panel window, so
        # consecutive/overlapping gestures are visible (the focus event stays bold).
        main_idx = int(row["original_prompt_index"])
        neighbours = aligned[
            (aligned["ground_truth_time"] >= window_start)
            & (aligned["ground_truth_time"] <= window_end)
            & (aligned["original_prompt_index"] != main_idx)
        ]
        for _, nrow in neighbours.iterrows():
            for nkey, ncolor in [
                ("ground_truth_time", "ground_truth"),
                ("prompt_time", "shifted"),
                ("aligned_time", "aligned"),
            ]:
                nvalue = float(nrow[nkey])
                if window_start <= nvalue <= window_end:
                    nx = sx(nvalue)
                    ndash = ' stroke-dasharray="4 4"' if ncolor == "shifted" else ""
                    parts.append(
                        f'<line x1="{nx:.2f}" x2="{nx:.2f}" y1="{plot_y}" '
                        f'y2="{plot_y + inner_h}" stroke="{colors[ncolor]}" '
                        f'stroke-width="1.0" opacity="0.30"{ndash}/>'
                    )
            ngx = sx(float(nrow["ground_truth_time"]))
            parts.append(
                f'<text x="{ngx + 2:.2f}" y="{plot_y + inner_h - 4:.2f}" '
                f'font-family="Arial" font-size="8" fill="{colors["muted"]}" '
                f'opacity="0.75">#{int(nrow["original_prompt_index"])} '
                f'{html.escape(str(nrow["name"]))}</text>'
            )

        for key, color_key, label, dash in [
            ("ground_truth_time", "ground_truth", "ground truth", ""),
            ("prompt_time", "shifted", "shifted input", ' stroke-dasharray="6 5"'),
            ("aligned_time", "aligned", "recovered", ""),
        ]:
            value = float(row[key])
            if window_start <= value <= window_end:
                x = sx(value)
                parts.append(
                    f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{plot_y}" '
                    f'y2="{plot_y + inner_h}" stroke="{colors[color_key]}" '
                    f'stroke-width="2.2"{dash}/>'
                )
                parts.append(
                    f'<text x="{x + 4:.2f}" y="{plot_y + 14}" font-family="Arial" '
                    f'font-size="10" fill="{colors[color_key]}" '
                    f'transform="rotate(-25 {x + 4:.2f},{plot_y + 14})">'
                    f"{label}</text>"
                )

        for channel in range(n_channels):
            base = plot_y + (channel + 0.5) * lane_h
            amp = lane_h * 0.5
            y_values = np.clip(display[:, channel], -1.5, 1.5) / 1.5
            points = " ".join(
                f"{sx(float(t)):.2f},{base - amp * float(y):.2f}"
                for t, y in zip(window_times, y_values)
            )
            parts.append(
                f'<line x1="{plot_x}" x2="{plot_x + inner_w}" y1="{base:.2f}" '
                f'y2="{base:.2f}" stroke="#edf2f7" stroke-width="0.7"/>'
            )
            parts.append(
                f'<polyline points="{points}" fill="none" '
                f'stroke="{channel_colors[channel % len(channel_colors)]}" '
                f'stroke-width="0.72" opacity="0.95"/>'
            )
            parts.append(
                f'<text x="{plot_x - 10}" y="{base + 3:.2f}" font-family="Arial" '
                f'font-size="9" text-anchor="end" fill="{colors["muted"]}">'
                f"ch{channel:02d}</text>"
            )

    legend_y = height - 48
    legend_x = left_margin + 300
    for color_key, label, dash in [
        ("ground_truth", "Ground truth event time", ""),
        ("shifted", "Randomly shifted input label", ' stroke-dasharray="6 5"'),
        ("aligned", "Recovered aligned time", ""),
    ]:
        parts.append(
            f'<line x1="{legend_x}" x2="{legend_x + 60}" y1="{legend_y}" '
            f'y2="{legend_y}" stroke="{colors[color_key]}" stroke-width="3"{dash}/>'
            f'<text x="{legend_x + 70}" y="{legend_y + 5}" font-family="Arial" '
            f'font-size="13" fill="{colors["text"]}">{label}</text>'
        )
        legend_x += 365

    parts.append("</svg>")
    output_svg.write_text("\n".join(parts), encoding="utf-8")


def _robust_multichannel_display(emg: np.ndarray) -> np.ndarray:
    centered = emg - np.median(emg, axis=0, keepdims=True)
    scale = np.percentile(np.abs(centered), 98, axis=0, keepdims=True)
    return centered / np.maximum(scale, 1e-8)


def _plot_envelope(emg: np.ndarray, sample_rate: float, smoothing_ms: float) -> np.ndarray:
    envelope = np.sqrt(np.mean(np.asarray(emg, dtype=np.float64) ** 2, axis=1))
    window = max(1, int(round(sample_rate * smoothing_ms / 1000.0)))
    if window > 1:
        kernel = np.ones(window) / window
        envelope = np.convolve(envelope, kernel, mode="same")
    low, high = np.percentile(envelope, [1, 99])
    envelope = np.clip(envelope, low, high)
    return (envelope - envelope.min()) / max(np.ptp(envelope), 1e-12)


def _write_alignment_svg(
    output_svg: Path,
    times: np.ndarray,
    envelope: np.ndarray,
    prompts: pd.DataFrame,
    aligned: pd.DataFrame | None,
    title: str,
) -> None:
    import html

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 720
    left, right, top, bottom = 90, 30, 80, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    t0, t1 = float(times[0]), float(times[-1])

    def x_pos(t: float) -> float:
        return left + (float(t) - t0) / max(t1 - t0, 1e-12) * plot_w

    def y_pos(v: float) -> float:
        return top + (1.0 - float(v)) * plot_h

    max_points = 2500
    stride = max(1, int(np.ceil(len(times) / max_points)))
    xs = [x_pos(t) for t in times[::stride]]
    ys = [y_pos(v) for v in envelope[::stride]]
    path = " ".join(
        f"{'M' if i == 0 else 'L'} {x:.2f} {y:.2f}"
        for i, (x, y) in enumerate(zip(xs, ys))
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', Arial, sans-serif; fill: #111827; }",
        ".small { font-size: 12px; } .label { font-size: 10px; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="36" font-size="22" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{left}" y="58" class="small">Raw sEMG RMS envelope. Blue dashed = prompt time, red solid = aligned time.</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>',
        f'<path d="{path}" fill="none" stroke="#111827" stroke-width="1.4"/>',
    ]

    for tick in np.linspace(t0, t1, 7):
        x = x_pos(tick)
        rel = tick - t0
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#6b7280"/>')
        lines.append(f'<text x="{x - 18:.2f}" y="{top + plot_h + 24}" class="small">{rel:.1f}s</text>')

    lines.append(f'<text x="{left}" y="{height - 18}" class="small">Window start absolute time: {t0:.6f}</text>')
    lines.append(f'<line x1="{left + 470}" y1="{height - 22}" x2="{left + 520}" y2="{height - 22}" stroke="#2563eb" stroke-width="2" stroke-dasharray="5 5"/>')
    lines.append(f'<text x="{left + 530}" y="{height - 18}" class="small">Prompt</text>')
    lines.append(f'<line x1="{left + 610}" y1="{height - 22}" x2="{left + 660}" y2="{height - 22}" stroke="#dc2626" stroke-width="2"/>')
    lines.append(f'<text x="{left + 670}" y="{height - 18}" class="small">Aligned</text>')

    for i, row in enumerate(prompts.itertuples()):
        x = x_pos(float(row.time))
        label_y = top + 16 + (i % 4) * 15
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#2563eb" stroke-width="1.1" stroke-dasharray="5 5" opacity="0.75"/>')
        lines.append(f'<text x="{x + 3:.2f}" y="{label_y}" class="label" fill="#1d4ed8" transform="rotate(-25 {x + 3:.2f} {label_y})">{html.escape(str(row.name))}</text>')

    if aligned is not None:
        for i, row in enumerate(aligned.itertuples()):
            x = x_pos(float(row.plot_aligned_time))
            label_y = top + plot_h - 12 - (i % 4) * 15
            name = getattr(row, "name", "")
            offset = float(row.plot_aligned_time - row.prompt_time)
            lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#dc2626" stroke-width="1.4" opacity="0.85"/>')
            lines.append(f'<text x="{x + 3:.2f}" y="{label_y}" class="label" fill="#b91c1c" transform="rotate(-25 {x + 3:.2f} {label_y})">{html.escape(str(name))} {offset:+.2f}s</text>')

    lines.append("</svg>")
    output_svg.write_text("\n".join(lines), encoding="utf-8")


def _compute_features_for_session(
    timeseries,
    feature: str,
    smoothing_ms: float,
    mpf_window_length: int,
    mpf_stride: int,
    mpf_n_fft: int,
    mpf_fft_stride: int,
    mpf_chunk_output_frames: int,
):
    """Compute alignment features (mpf or envelope) and their timestamps once."""

    times = timeseries["time"]
    sample_rate = infer_sample_rate(times)
    if feature == "mpf":
        features, feature_times = multivariate_power_frequency_features(
            timeseries["emg"],
            times,
            window_length=mpf_window_length,
            stride=mpf_stride,
            n_fft=mpf_n_fft,
            fft_stride=mpf_fft_stride,
            fs=sample_rate,
            chunk_output_frames=mpf_chunk_output_frames,
            progress=True,
        )
    else:
        features = emg_envelope_features(
            timeseries["emg"], sample_rate=sample_rate, smoothing_ms=smoothing_ms
        )
        feature_times = times
    return features, feature_times


@main.command("multi-session-recenter-eval")
@click.argument("hdf5_paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--offsets",
    type=str,
    default=None,
    help=(
        "Comma-separated per-session systematic offsets in seconds, one per HDF5 "
        "file (e.g. '-0.25,-0.10,-0.40'). If omitted, offsets are spread evenly "
        "across [-0.40, -0.10]."
    ),
)
@click.option(
    "--jitter",
    type=float,
    default=0.05,
    show_default=True,
    help="Uniform +/- per-event jitter added on top of each session offset.",
)
@click.option("--num-prompts", type=int, default=100000, show_default=True)
@click.option("--pre", type=float, default=0.3, show_default=True)
@click.option("--post", type=float, default=0.9, show_default=True)
@click.option(
    "--uncertainty",
    nargs=2,
    type=float,
    default=(-0.15, 0.55),
    show_default=True,
    help="Search window; must cover the full injected shift range and zero.",
)
@click.option("--iterations", type=int, default=5, show_default=True)
@click.option("--beam-width", type=int, default=30, show_default=True)
@click.option("--candidate-step", type=float, default=0.02, show_default=True)
@click.option("--min-event-separation", type=float, default=0.02, show_default=True)
@click.option(
    "--template-estimator",
    type=click.Choice(["rerp", "average"]),
    default="rerp",
    show_default=True,
)
@click.option("--template-ridge", type=float, default=1e-3, show_default=True)
@click.option(
    "--max-shift",
    type=float,
    default=0.5,
    show_default=True,
    help="Max template shift allowed when matching to the global reference.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--smoothing-ms", type=float, default=50.0, show_default=True)
@click.option(
    "--feature",
    type=click.Choice(["mpf", "envelope"]),
    default="mpf",
    show_default=True,
)
@click.option("--mpf-window-length", type=int, default=200, show_default=True)
@click.option("--mpf-stride", type=int, default=40, show_default=True)
@click.option("--mpf-n-fft", type=int, default=64, show_default=True)
@click.option("--mpf-fft-stride", type=int, default=10, show_default=True)
@click.option("--mpf-chunk-output-frames", type=int, default=4096, show_default=True)
@click.option(
    "--oracle-init/--no-oracle-init",
    default=False,
    show_default=True,
    help="Seed each session's EM first iteration with oracle templates (from "
    "ground truth). Isolates the global-recenter step from cold-start EM failure.",
)
def multi_session_recenter_eval(
    hdf5_paths: tuple[Path, ...],
    output_dir: Path,
    offsets: str | None,
    jitter: float,
    num_prompts: int,
    pre: float,
    post: float,
    uncertainty: tuple[float, float],
    iterations: int,
    beam_width: int,
    candidate_step: float,
    min_event_separation: float,
    template_estimator: str,
    template_ridge: float,
    max_shift: float,
    seed: int,
    smoothing_ms: float,
    feature: str,
    mpf_window_length: int,
    mpf_stride: int,
    mpf_n_fft: int,
    mpf_fft_stride: int,
    mpf_chunk_output_frames: int,
    oracle_init: bool,
) -> None:
    """Validate the full pipeline: inject DIFFERENT systematic offsets per session,
    align each with EM, then run global recentering and check whether the
    between-session offset differences collapse.

    Single-session EM cannot recover a systematic (constant) offset, so each
    session's EM error stays near its injected offset. Global recentering anchors
    all sessions to a shared grand-average template, which should remove the
    BETWEEN-session offset differences (the absolute common offset stays
    unidentifiable, by design).
    """

    if len(hdf5_paths) < 2:
        raise click.BadParameter("Provide at least two HDF5 files for a multi-session test")

    if offsets is None:
        n = len(hdf5_paths)
        offset_values = list(np.linspace(-0.40, -0.10, n))
    else:
        offset_values = [float(x) for x in offsets.split(",")]
        if len(offset_values) != len(hdf5_paths):
            raise click.BadParameter(
                f"--offsets has {len(offset_values)} values but {len(hdf5_paths)} files given"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    aligned_tables: dict[str, pd.DataFrame] = {}
    template_banks = {}
    per_event_rows = []

    for path, offset in zip(hdf5_paths, offset_values):
        session_id = path.stem
        click.echo(f"\n=== Session {session_id}: injected offset {offset:+.3f}s ===", err=True)
        timeseries, prompts = load_discrete_gesture_hdf5(path)
        prompts = prompts.sort_values("time").reset_index(drop=True)
        selected = prompts.iloc[:num_prompts].copy()

        gt = selected["time"].to_numpy(dtype=float)
        shift = offset + rng.uniform(-jitter, jitter, size=len(selected))
        shifted = selected.copy()
        shifted["ground_truth_time"] = gt
        shifted["time"] = gt + shift
        shifted["original_prompt_index"] = selected.index.to_numpy(dtype=int)

        features, feature_times = _compute_features_for_session(
            timeseries, feature, smoothing_ms,
            mpf_window_length, mpf_stride, mpf_n_fft, mpf_fft_stride,
            mpf_chunk_output_frames,
        )

        init_bank = None
        if oracle_init:
            click.echo("  seeding EM with oracle templates (from ground truth)", err=True)
            truth = shifted.copy()
            truth["time"] = truth["ground_truth_time"].to_numpy(dtype=float)
            init_bank = estimate_templates(
                features, feature_times, truth, pre, post,
                aligned_time_col="time", method=template_estimator, ridge=template_ridge,
            )

        aligned, templates = align_prompt_times(
            features=features,
            times=feature_times,
            prompts=shifted[["name", "time", "ground_truth_time", "original_prompt_index"]],
            pre_s=pre,
            post_s=post,
            uncertainty_window=uncertainty,
            max_iterations=iterations,
            beam_width=beam_width,
            candidate_step_s=candidate_step,
            recenter_templates=False,
            template_estimator=template_estimator,
            template_ridge=template_ridge,
            enforce_monotonic=True,
            min_event_separation_s=min_event_separation,
            progress=True,
            init_templates=init_bank,
        )
        aligned_tables[session_id] = aligned
        template_banks[session_id] = templates

    click.echo("\n=== Running global recentering across sessions ===", err=True)
    _, recentered_tables, shift_summary = global_recenter_aligned_prompts(
        aligned_tables, template_banks, max_shift_s=max_shift,
    )

    # Evaluate each stage against ground truth.
    summary_rows = []
    for session_id, offset in zip([p.stem for p in hdf5_paths], offset_values):
        em = aligned_tables[session_id].sort_index()
        rc = recentered_tables[session_id].sort_index()
        gt = em["ground_truth_time"].to_numpy(dtype=float)
        err_raw = em["prompt_time"].to_numpy(dtype=float) - gt
        err_em = em["aligned_time"].to_numpy(dtype=float) - gt
        err_rc = rc["aligned_time"].to_numpy(dtype=float) - gt

        def demean_std(e):
            return float(np.std(e - np.median(e)))

        summary_rows.append(
            {
                "session": session_id,
                "injected_offset": offset,
                "n": len(gt),
                "mean_err_raw": float(np.mean(err_raw)),
                "mean_err_em": float(np.mean(err_em)),
                "mean_err_recenter": float(np.mean(err_rc)),
                "demeaned_std_raw": demean_std(err_raw),
                "demeaned_std_em": demean_std(err_em),
                "demeaned_std_recenter": demean_std(err_rc),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "multi_session_recenter_summary.csv"
    summary.to_csv(summary_csv, index=False)
    shift_summary.to_csv(output_dir / "multi_session_recenter_shifts.csv", index=False)
    for session_id, rc in recentered_tables.items():
        rc.to_csv(output_dir / f"{session_id}_recentered_eval.csv", index=False)

    between_em = float(np.std(summary["mean_err_em"]))
    between_rc = float(np.std(summary["mean_err_recenter"]))

    click.echo("\n================ RESULT ================")
    click.echo(summary.to_string(index=False))
    click.echo(
        "\nBetween-session spread of mean error (the thing global recenter should fix):"
    )
    click.echo(f"  after EM only       : std = {between_em:.4f}s")
    click.echo(f"  after global recenter: std = {between_rc:.4f}s")
    if between_rc < between_em:
        click.echo(
            f"  -> recenter REDUCED between-session spread "
            f"({between_em:.4f} -> {between_rc:.4f}). Global recenter is working."
        )
    else:
        click.echo(
            "  -> recenter did NOT reduce between-session spread; "
            "check templates/feature/num-prompts."
        )
    click.echo(
        "\nNote: the absolute common offset is unidentifiable by design; judge "
        "EM by demeaned_std (relative timing) and recenter by between-session spread."
    )
    click.echo(f"\nSaved summary to {summary_csv}")


if __name__ == "__main__":
    main()
