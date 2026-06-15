# Discrete Gesture Time Alignment — 复现流程与参数说明

本报告说明本仓库 `discrete_gesture_alignment` 这套复现的**完整运行流程**，以及**需要根据数据/实验目标调的参数**及其作用。代码位置：
- 算法库：`generic_neuromotor_interface/discrete_gesture_alignment.py`
- 命令行：`generic_neuromotor_interface/scripts/discrete_gesture_alignment.py`

---

## 一、背景与思路

论文只公开了**对齐后**的 gesture prompt time（即每个手势事件的"真值"时间），没有公开对齐**之前**的 raw prompt time，也没公开对齐代码。

本复现做两件事：
1. **复现对齐算法**：给定连续 sEMG 特征和一组（带时间误差的）prompt time，用 EM 式的"生成-推断"把每个 prompt 的时间挪到最能解释 EMG 的位置（forced alignment）。
2. **构造测试**：把公开的 aligned time 当作**伪真值**，人为加偏移造出"raw prompt time"，跑对齐看能否把它恢复回真值，以此验证算法是否有效。

核心生成模型假设：连续特征 ≈ 把每个手势的**模板波形 (template)** 放到各自事件时间上叠加而成。对齐 = 找一组事件时间，使"模板叠加"与真实特征的残差最小。

---

## 二、完整运行流程

以 `simulate-shift-eval`（构造偏移 + 对齐 + 评估）为主线，逐步说明：

### 步骤 0：加载数据
`load_discrete_gesture_hdf5` 读出：
- `timeseries`：连续 sEMG（含 `emg` 多通道 + `time` 时间戳，2 kHz）
- `prompts`：每个手势的 `name`（类别）和 `time`（公开的 aligned 时间）

### 步骤 1：构造 raw prompt time（仅评估时）
取前 `--num-prompts` 个事件，把它们的 aligned time 当 `ground_truth_time`，加一个随机偏移得到 `prompt_time`（= 模拟的 raw prompt time），输入给对齐。偏移由 `--shift-range` 控制。

### 步骤 2：特征提取
把原始 sEMG 变成对齐用的连续特征：
- **MPF**（`--feature mpf`，论文用法）：Multivariate Power-Frequency。STFT → 跨通道交叉谱密度 → 按频段聚合 → 特征维 = 频段数 × 通道 × 通道（默认 6×16×16 = 1536 维）。时间分辨率 = `stride / fs`，默认 40/2000 = **20 ms/帧**。
- **envelope**（`--feature envelope`）：整流 + 滑动平均 + z-score。轻量、快，但判别力弱，**只适合 smoke test**。

### 步骤 3：EM 迭代对齐（核心，`align_prompt_times`）
重复最多 `--iterations` 轮：

1. **估模板（E-ish step）** `estimate_templates`
   - `average`：把每个事件周围 `[-pre, +post]` 的特征窗口对齐叠平均，得到每类手势一个模板。
   - `rerp`（论文式，默认）：用稀疏线性回归**联合**估所有类别模板，能把时间上重叠的相邻手势贡献分离开（比单纯平均更干净）。`--template-ridge` 是 L2 正则。

2. **分组重叠序列** `group_overlapping_sequences`
   - 把时间不确定窗口（`uncertainty`）会互相重叠的相邻事件归为一个 sequence，在序列内联合搜索（避免两个相邻事件抢同一个 EMG burst）。

3. **beam search 定位（M step）** `_align_sequence`
   - 对每个事件，在 `[prompt_time + uncertainty_low, prompt_time + uncertainty_high]` 范围内、以 `--candidate-step` 为步长枚举候选时间；
   - 每放一个候选，就从残差里减掉该类模板，选使总残差最小的组合；
   - `beam-width` 控制保留多少条候选路径；
   - `enforce_monotonic` + `min-event-separation` 约束事件不能乱序/挤在一起；
   - `prompt-prior-weight` 给"偏离原 prompt 太远"加二次惩罚。

4. **收敛判断**：本轮所有事件时间的最大移动量 < `--tolerance`（默认 0.005s）就提前停；否则跑满 `--iterations`。

### 步骤 4（可选）：跨 session 全局 recenter
`global_recenter_aligned_prompts`：EM 内部的"手势零点"是不定的（模板可能整体吸收了一个固定反应延迟）。论文最后用**跨参与者的 grand-average 模板**做一次全局对齐，把每个 session 的模板/时间统一到共同零点。单 session 测试用不到，多 session 才需要。

### 步骤 5：评估 + 可视化
- 计算 `error_before`（raw−真值）和 `error_after`（对齐后−真值），写 `*_alignment_metrics.csv`。
- 画 `*_multichannel_alignment.svg`：每个事件叠在多通道 EMG 上，🟢真值 / 🔵你造的 raw prompt / 🔴恢复结果。

---

## 三、需要 figure out 的参数（按重要性排序）

### ⭐ 1. `--uncertainty LOW HIGH`（时间不确定/搜索窗）
**作用**：每个事件允许搜索的时间范围，相对 prompt_time。这是**最关键**的参数。
**规则**：必须**覆盖你注入偏移的全范围，并且包含 0**，否则真值落在窗外，再准也够不到。
- 例：raw prompt 比真手势早 0.1–0.4s（`--shift-range -0.4 -0.1`），真值在 prompt 之后 +0.1~+0.4s，所以上界要 ≥ 0.4（留余量 → 0.5），下界给点 slack（−0.1）：`--uncertainty -0.1 0.5`。
**权衡**：太窄 → 够不到真值；太宽 → 搜索慢、且容易被邻近别的手势的 EMG 误导（误对齐）。脚本会在窗没覆盖 shift 时打 warning。

### ⭐ 2. `--pre` / `--post`（模板时间窗，秒）
**作用**：模板从事件时间往前 `pre`、往后 `post` 截取的长度，应**覆盖一次手势的 EMG 活动**。默认 0.3 / 0.9。
**权衡**：太短 → 模板抓不全手势波形，判别力差；太长 → 相邻手势互相污染、rERP 参数爆炸（脚本对 rERP 总参数有 1 万上限保护）。

### ⭐ 3. `--feature` + MPF 参数
- `mpf` 用于正式实验，`envelope` 只做 smoke test。
- `--mpf-stride`（默认 40）决定**时间分辨率**：40/2000 = 20 ms。这是对齐精度的**物理下限**——`candidate-step` 设得比它小也没意义。
- `--mpf-window-length / --mpf-n-fft / --mpf-fft-stride`：谱估计的窗长/FFT 点数，一般保持默认（与仓库 `WristArchitecture` 的 MPF 一致）。

### 4. `--template-estimator` + `--template-ridge`
- `rerp`（默认，推荐）：联合反卷积，分离重叠手势。`--template-ridge`（默认 1e-3）防止病态/过拟合；事件少时调大更稳。
- `average`：快但对重叠手势会糊。

### 5. `--candidate-step`（搜索步长，秒）
**作用**：beam search 枚举候选时间的粒度。默认 0.02s（= 1 个 MPF 帧）。
**权衡**：更小 → 更精细但更慢，且小于 MPF 帧分辨率收益递减；更大 → 快但精度粗。

### 6. `--beam-width`
**作用**：序列内保留的候选路径数。默认 30。
**权衡**：大 → 更不容易陷入局部最优，但越大越慢。短序列影响不大；长重叠序列建议 ≥ 20。

### 7. `--iterations` + `--tolerance`
EM 最多轮数 / 收敛阈值。观察每轮 `max timestamp update`：逐轮变小是正常的；若早早 < tolerance 会提前停。一般 5 轮足够。

### 8. `--min-event-separation`（秒）
**作用**：序列内相邻对齐事件的最小间隔，防止多个事件塌缩到同一处。按手势最快连发的真实节奏设（如 0.02–0.05s）。

### 9. `--prompt-prior-weight`
**作用**：对"对齐时间偏离原 prompt 太远"的二次惩罚权重。
**何时用**：当你**相信原 prompt 已经比较准**、只想做小幅修正时调大；当 raw prompt 偏差大、要大幅搬动时设小或 0。

### 10. `--recenter / --no-recenter`
每轮把模板按能量峰重新居中。论文式 EM 用 `--no-recenter`（最后靠全局 recenter 解决零点）；单 session 无全局参考时，`--recenter` 是个粗略替代。

### 11.（评估用）`--shift-range LOW HIGH`、`--seed`、`--num-prompts`
- `--shift-range`：注入偏移的分布（均匀）。负值 = raw prompt 比手势早（更接近真实反应时）。
- `--seed`：随机种子，多 seed 跑出误差棒。
- `--num-prompts`：参与的事件数。**越多模板越干净、EM 越容易收敛**——这是之前 100 个事件失败、全量成功的关键原因。设很大（如 100000）= 用全部事件。

---

## 四、判读结果

看 `*_alignment_metrics.csv`：
- **有效**：`mae_after` 明显 < `mae_before`，`median_abs_error_after` ≲ 0.025s（1–2 个 MPF 帧），`fraction_improved` > 0.8。
- **没收敛**：`after ≈ before`，`fraction_improved ≈ 0.4–0.5`。

**建议加 oracle 对照**（用未偏移真值估模板再单次匹配）作为能力上界——本地测得 median ≈ 12 ms。有了上界，才能区分"方法没生效"还是"任务在 MPF 20ms 分辨率下本就这么难"。

---

## 五、典型命令（往前偏移、全量、MPF+rERP）

```bash
python -u -m generic_neuromotor_interface.scripts.discrete_gesture_alignment simulate-shift-eval \
  $DATA /scratch/.../out/mpf_rerp_u000_seed0 \
  --num-prompts 100000 \
  --shift-range -0.40 -0.10 \
  --uncertainty -0.10 0.50 \
  --feature mpf --template-estimator rerp --template-ridge 0.001 \
  --iterations 5 --beam-width 30 --candidate-step 0.02 \
  --min-event-separation 0.02 --prompt-prior-weight 0.05 \
  --plot-prompts 12 --seed 0
```
