# Simulated Shift Time Alignment Evaluation

这个实验用于测试 time alignment 代码是否能从人为偏移的 gesture labels 中恢复出接近原始公开 event time 的 aligned time。

## 实验逻辑

公开数据中的 `prompts.time` 被当作 pseudo ground truth：

```text
ground_truth_time = public prompts.time
```

然后随机偏移这些 label：

```text
shifted_time = ground_truth_time + random_shift
```

再将 `shifted_time` 输入 time alignment：

```text
shifted_time -> alignment -> recovered aligned_time
```

最后比较：

```text
error_before = shifted_time - ground_truth_time
error_after  = aligned_time - ground_truth_time
```

注意：这个实验不改变 raw sEMG。被偏移的是 label time，不是 EMG signal。

## 快速本地测试

这个版本使用 envelope + average template，速度快，适合检查代码是否能跑通：

```bash
cd /Users/yoooung/Desktop/generic-neuromotor-interface
source .venv/bin/activate

python -m generic_neuromotor_interface.scripts.discrete_gesture_alignment simulate-shift-eval \
  data/discrete_gestures_user_000_dataset_000.hdf5 \
  generic_neuromotor_interface/reports/time_alignment_reproduction/sim_shift_smoke \
  --num-prompts 24 \
  --shift-range -0.25 0.25 \
  --uncertainty -0.35 0.35 \
  --feature envelope \
  --template-estimator average \
  --iterations 3 \
  --beam-width 8 \
  --candidate-step 0.04 \
  --plot-prompts 4 \
  --seed 7 \
  --recenter
```

## HPC / paper-style run

这个版本使用 MPF + rERP，更接近论文方法，但计算更重：

```bash
python -m generic_neuromotor_interface.scripts.discrete_gesture_alignment simulate-shift-eval \
  data/discrete_gestures_user_000_dataset_000.hdf5 \
  outputs/sim_shift_user_000_mpf_rerp \
  --num-prompts 500 \
  --shift-range -0.25 0.25 \
  --uncertainty -0.35 0.35 \
  --feature mpf \
  --template-estimator rerp \
  --template-ridge 0.001 \
  --iterations 5 \
  --beam-width 30 \
  --candidate-step 0.02 \
  --min-event-separation 0.02 \
  --prompt-prior-weight 0.05 \
  --plot-prompts 8 \
  --seed 0
```

如果运行太慢，可以先降低：

```text
--num-prompts
--beam-width
```

或者增大：

```text
--candidate-step
```

## 输出文件

命令会输出：

```text
*_simulated_shifted_prompts.csv
  人为偏移后的 label time。

*_simulated_aligned_prompts.csv
  alignment 后的 recovered aligned_time，并包含 before/after error。

*_simulated_alignment_metrics.csv
  MAE before、MAE after、improvement 等指标。

*_simulated_multichannel_alignment.svg
  多通道分离可视化图。
```

SVG 图中：

```text
绿色 = ground truth event time，也就是公开 prompts.time
蓝色虚线 = 随机偏移后的 input label time
红色 = alignment 恢复出的 aligned_time
彩色曲线 = 16 个 raw sEMG channel 分开堆叠显示
```

