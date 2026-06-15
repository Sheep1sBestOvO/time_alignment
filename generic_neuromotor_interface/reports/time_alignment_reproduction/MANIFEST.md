# 报告包说明

这个目录包含当前 discrete gesture time alignment 复现工作的报告材料。

## 文件

- `README.md`
  - 主报告。
  - 内容包括：方法理解、open data 观察、当前实现状态、和原论文的差距、关键参数讨论、自采数据建议、下一步计划。

## 相关源码

报告中提到的实现代码位于：

- `generic_neuromotor_interface/discrete_gesture_alignment.py`
  - prompt pattern 分析和 time alignment 的核心实现。

- `generic_neuromotor_interface/scripts/discrete_gesture_alignment.py`
  - 命令行工具，用于 inspect prompt pattern 和运行 alignment。

- `generic_neuromotor_interface/tests/test_discrete_gesture_alignment.py`
  - 合成数据和基础逻辑测试，包括 sequence grouping、template estimation 和简单 alignment recovery。

## 常用命令

查看 prompt timing pattern：

```bash
python3 -m generic_neuromotor_interface.scripts.discrete_gesture_alignment inspect \
  /Users/yoooung/Desktop/generic-neuromotor-interface/data \
  --uncertainty -0.3 0.8
```

对一个 HDF5 文件运行 alignment：

```bash
python3 -m generic_neuromotor_interface.scripts.discrete_gesture_alignment align \
  /Users/yoooung/Desktop/generic-neuromotor-interface/data/discrete_gestures_user_000_dataset_000.hdf5 \
  /Users/yoooung/Desktop/generic-neuromotor-interface/data/user_000_aligned_prompts.csv \
  --pre 0.3 \
  --post 0.9 \
  --uncertainty -0.3 0.8 \
  --candidate-step 0.02 \
  --beam-width 30
```
