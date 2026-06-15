import numpy as np
import pandas as pd

from generic_neuromotor_interface.discrete_gesture_alignment import (
    TemplateBank,
    _align_sequence,
    apply_recenter_shifts_to_aligned_prompts,
    align_prompt_times,
    build_global_template_bank,
    estimate_templates,
    estimate_template_recenter_shifts,
    estimate_templates_rerp,
    global_recenter_aligned_prompts,
    group_overlapping_sequences,
    multivariate_power_frequency_features,
    recenter_template_bank_to_reference,
    summarize_prompt_pattern,
)


def test_group_overlapping_sequences():
    prompts = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "time": [0.0, 0.5, 2.0, 2.4],
        }
    )

    sequences = group_overlapping_sequences(prompts, (-0.2, 0.4))

    assert [len(sequence) for sequence in sequences] == [2, 2]


def test_summarize_prompt_pattern_with_uncertainty():
    prompts = pd.DataFrame(
        {
            "name": ["a", "b", "c"],
            "time": [1.0, 1.5, 3.0],
        }
    )

    stats = summarize_prompt_pattern(prompts, (-0.4, 0.4))

    assert stats.num_prompts == 3
    assert stats.num_sequences == 2
    assert stats.sequence_lengths == [2, 1]
    assert stats.prompt_interval_seconds["min"] == 0.5


def test_estimate_templates_uses_prompt_times_initially():
    sample_rate = 100
    times = np.arange(0, 4, 1 / sample_rate)
    features = np.zeros((len(times), 1))
    prompts = pd.DataFrame({"name": ["tap", "tap"], "time": [1.0, 3.0]})

    for event_time in prompts["time"]:
        center = np.searchsorted(times, event_time)
        features[center - 2 : center + 3, 0] = [0, 1, 2, 1, 0]

    bank = estimate_templates(features, times, prompts, pre_s=0.02, post_s=0.03)

    assert "tap" in bank.templates
    np.testing.assert_allclose(bank.templates["tap"][:, 0], [0, 1, 2, 1, 0])


def test_align_prompt_times_recovers_shifted_event():
    sample_rate = 100
    times = np.arange(0, 4, 1 / sample_rate)
    features = np.zeros((len(times), 1))
    prompts = pd.DataFrame(
        {
            "name": ["tap", "tap", "tap"],
            "time": [0.9, 1.9, 2.9],
        }
    )
    true_times = np.array([1.0, 2.0, 3.0])
    waveform = np.array([0, 1, 3, 1, 0], dtype=float)
    for event_time in true_times:
        center = np.searchsorted(times, event_time)
        features[center - 2 : center + 3, 0] += waveform

    aligned, _ = align_prompt_times(
        features=features,
        times=times,
        prompts=prompts,
        pre_s=0.02,
        post_s=0.15,
        uncertainty_window=(-0.05, 0.2),
        max_iterations=3,
        beam_width=5,
        candidate_step_s=0.01,
        tolerance_s=0.001,
        recenter_templates=True,
        template_estimator="average",
    )

    np.testing.assert_allclose(aligned["aligned_time"], true_times, atol=0.011)


def test_rerp_template_estimation_separates_overlapping_events():
    sample_rate = 100
    times = np.arange(0, 6, 1 / sample_rate)
    pre_samples = 2
    post_samples = 3
    template_a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])[:, None]
    template_b = np.array([0.0, -0.5, -1.0, -0.5, 0.0])[:, None]
    true_templates = {"a": template_a, "b": template_b}
    prompts = pd.DataFrame(
        {
            "name": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "time": [1.00, 1.02, 2.00, 2.03, 3.00, 3.02, 4.00, 4.03],
        }
    )
    features = np.zeros((len(times), 1))
    for row in prompts.itertuples():
        center = np.searchsorted(times, row.time)
        start = center - pre_samples
        stop = center + post_samples
        features[start:stop] += true_templates[row.name]

    bank = estimate_templates_rerp(
        features,
        times,
        prompts,
        pre_samples=pre_samples,
        post_samples=post_samples,
        sample_rate=sample_rate,
        aligned_time_col="time",
        ridge=1e-8,
        fit_intercept=False,
    )

    np.testing.assert_allclose(bank.templates["a"], template_a, atol=1e-5)
    np.testing.assert_allclose(bank.templates["b"], template_b, atol=1e-5)


def test_beam_search_enforces_monotonic_event_order():
    sample_rate = 100
    times = np.arange(0, 3, 1 / sample_rate)
    features = np.zeros((len(times), 1))
    waveform = np.array([0, 1, 3, 1, 0], dtype=float)[:, None]
    for event_time in [1.0, 1.2]:
        center = np.searchsorted(times, event_time)
        features[center - 2 : center + 3] += waveform

    sequence = pd.DataFrame(
        {
            "name": ["tap", "tap"],
            "time": [1.15, 0.95],
            "prompt_time": [1.15, 0.95],
        }
    )
    templates = TemplateBank(
        templates={"tap": waveform},
        pre_samples=2,
        post_samples=3,
        sample_rate=sample_rate,
    )

    aligned = _align_sequence(
        features=features,
        times=times,
        sequence=sequence,
        templates=templates,
        uncertainty_window=(-0.3, 0.3),
        beam_width=5,
        candidate_step_s=0.01,
        enforce_monotonic=True,
    )

    assert np.all(np.diff(aligned["aligned_time"]) >= 0)


def test_multivariate_power_frequency_features_shape_and_times():
    sample_rate = 2000
    times = np.arange(1000) / sample_rate
    emg = np.random.default_rng(0).normal(size=(len(times), 4)).astype(np.float32)

    features, feature_times = multivariate_power_frequency_features(
        emg,
        times,
        window_length=200,
        stride=40,
        n_fft=64,
        fft_stride=10,
        frequency_bins=((0, 50), (50, 150)),
        chunk_output_frames=7,
    )

    assert features.shape[0] == len(feature_times)
    assert features.shape[1] == 2 * 4 * 4
    assert np.all(np.diff(feature_times) > 0)


def test_global_template_bank_averages_matching_gestures():
    first = TemplateBank(
        templates={"tap": np.array([[0.0], [1.0], [0.0]])},
        pre_samples=1,
        post_samples=2,
        sample_rate=100,
    )
    second = TemplateBank(
        templates={"tap": np.array([[0.0], [3.0], [0.0]])},
        pre_samples=1,
        post_samples=2,
        sample_rate=100,
    )

    global_bank = build_global_template_bank([first, second])

    np.testing.assert_allclose(global_bank.templates["tap"], [[0.0], [2.0], [0.0]])


def test_reference_recenter_finds_template_shift():
    reference_waveform = np.array([0, 0, 0, 1, 4, 1, 0, 0, 0], dtype=float)[:, None]
    session_waveform = np.array([0, 1, 4, 1, 0, 0, 0, 0, 0], dtype=float)[:, None]
    session = TemplateBank(
        templates={"tap": session_waveform},
        pre_samples=4,
        post_samples=5,
        sample_rate=100,
    )
    reference = TemplateBank(
        templates={"tap": reference_waveform},
        pre_samples=4,
        post_samples=5,
        sample_rate=100,
    )

    recentered, shifts = recenter_template_bank_to_reference(session, reference)

    assert shifts["tap"] == 2
    np.testing.assert_allclose(recentered.templates["tap"], reference_waveform)


def test_recenter_shift_updates_aligned_times_with_opposite_sign():
    aligned = pd.DataFrame(
        {
            "name": ["tap", "tap"],
            "prompt_time": [1.0, 2.0],
            "aligned_time": [1.2, 2.2],
        }
    )

    adjusted = apply_recenter_shifts_to_aligned_prompts(
        aligned,
        {"tap": 20},
        sample_rate=100,
    )

    np.testing.assert_allclose(adjusted["aligned_time"], [1.0, 2.0])
    np.testing.assert_allclose(adjusted["alignment_offset"], [0.0, 0.0])


def test_global_recenter_aligned_prompts_returns_tables_and_shift_summary():
    reference = np.array([0, 0, 0, 1, 4, 1, 0, 0, 0], dtype=float)[:, None]
    early = np.array([0, 1, 4, 1, 0, 0, 0, 0, 0], dtype=float)[:, None]
    session_templates = {
        "s0": TemplateBank(
            templates={"tap": early},
            pre_samples=4,
            post_samples=5,
            sample_rate=100,
        ),
        "s1": TemplateBank(
            templates={"tap": reference},
            pre_samples=4,
            post_samples=5,
            sample_rate=100,
        ),
    }
    aligned_tables = {
        "s0": pd.DataFrame(
            {"name": ["tap"], "prompt_time": [1.0], "aligned_time": [1.2]}
        ),
        "s1": pd.DataFrame(
            {"name": ["tap"], "prompt_time": [2.0], "aligned_time": [2.0]}
        ),
    }

    _, recentered, summary = global_recenter_aligned_prompts(
        aligned_tables,
        session_templates,
        max_shift_s=0.05,
    )

    assert set(recentered) == {"s0", "s1"}
    assert set(summary["session_id"]) == {"s0", "s1"}
    assert "shift_seconds" in summary.columns
    assert recentered["s0"]["aligned_time"].iloc[0] < 1.2
