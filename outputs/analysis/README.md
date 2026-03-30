# Analysis Outputs Index

结果目录已按用途收口到：

- `error/`
- `motif/`
- `diagnostics/`
- `eval/`

顶层旧文件名目前保留为软链接，便于兼容旧路径和已打开的文件标签页。
后续新结果建议直接写入对应子目录。

这份目录索引只保留当前还值得看的分析结果，并标清每个文件的用途。

## 结果分组

- `error/`
  - bind / together 错误分析
- `motif/`
  - motif 与 pairwise motif 结果
- `diagnostics/`
  - 诊断矩阵、原始数组、summary
- `eval/`
  - 独立评估结果

## 当前最值得看的文件

### 1. 单任务 bind 当前主结果（weight 版）
- [etd_bind_diagnostics_weight_last.md](/media/scw-workspace/m6a_dataset/outputs/analysis/diagnostics/etd_bind_diagnostics_weight_last.md)
  - 当前最推荐看的 bind 诊断摘要。
  - 对应 checkpoint：`outputs/etd_only/etd_bind_m6a_reader_neg_weight_0320/checkpoints/last.pt`
  - 内容：共现矩阵、分数分布、最优阈值、混淆矩阵、top-k、全局多标签指标。
  - 结论：weight 版优于无 weight 版，改善了 `EIF3 / YTHDC / FMR1 / YTHDF`，但 `ELAVL1 / RBMX` 仍然没学出来。

- [etd_bind_error_analysis_weight_last.md](/media/scw-workspace/m6a_dataset/outputs/analysis/error/etd_bind_error_analysis_weight_last.md)
  - 当前最推荐看的 bind 错误分析。
  - 内容：per-family TP/FP/FN/TN、多阈值 summary、hardest FP/FN。
  - 用途：看具体错成谁、哪些 family 是阈值问题、哪些 family 根本没学出来。

### 2. 单任务 bind 对照结果（无 weight 基线）
- [etd_bind_error_analysis_wo_weight.md](/media/scw-workspace/m6a_dataset/outputs/analysis/error/etd_bind_error_analysis_wo_weight.md)
  - 无 weight 版本错误分析。
  - 用途：与 `weight_last` 对照，看 weighting 到底改善了哪些 family。

- [etd_bind_diagnostics_matrices.md](/media/scw-workspace/m6a_dataset/outputs/analysis/diagnostics/etd_bind_diagnostics_matrices.md)
  - 较早的无 weight 诊断矩阵整理版。
  - 用途：看无 weight 版的整体行为模式。

### 3. mod / DRACH 分析
- [mod_drach_negative_eval.md](/media/scw-workspace/m6a_dataset/outputs/analysis/eval/mod_drach_negative_eval.md)
  - 多任务 mod 头在 DRACH 定义下的 4 套对照评估。
  - 还包含 `DRH` 18 种组合占比。
  - 结论：mod 头强依赖 DRACH motif，但在 DRACH 内部仍有一定区分真 m6A / 普通 A 的能力。

- [drach_motif_stats_readable.md](/media/scw-workspace/m6a_dataset/outputs/analysis/motif/drach_motif_stats_readable.md)
  - 数据统计：所有 A 里 DRACH 占比、m6A 里 DRACH 占比、train/val/test 分布。

## 仍可保留但优先级较低的文件

### bind 诊断原始文本 / 数组
- [etd_bind_diagnostics.txt](/media/scw-workspace/m6a_dataset/outputs/analysis/diagnostics/etd_bind_diagnostics.txt)
- [etd_bind_diagnostics_arrays.npz](/media/scw-workspace/m6a_dataset/outputs/analysis/diagnostics/etd_bind_diagnostics_arrays.npz)
- [etd_bind_diagnostics_weight_last.txt](/media/scw-workspace/m6a_dataset/outputs/analysis/diagnostics/etd_bind_diagnostics_weight_last.txt)
- [etd_bind_diagnostics_weight_last.npz](/media/scw-workspace/m6a_dataset/outputs/analysis/diagnostics/etd_bind_diagnostics_weight_last.npz)

用途：原始输出或复查用，不建议作为第一入口。

### bind 错误分析原始 json
- [etd_bind_error_analysis.json](/media/scw-workspace/m6a_dataset/outputs/analysis/error/etd_bind_error_analysis.json)
- [etd_bind_error_analysis_weight.json](/media/scw-workspace/m6a_dataset/outputs/analysis/error/etd_bind_error_analysis_weight.json)
- [etd_bind_error_analysis_weight_last.json](/media/scw-workspace/m6a_dataset/outputs/analysis/error/etd_bind_error_analysis_weight_last.json)

用途：脚本后处理或程序读取。

### bind 中间版本 md
- [etd_bind_error_analysis_weight.md](/media/scw-workspace/m6a_dataset/outputs/analysis/error/etd_bind_error_analysis_weight.md)
  - 注意：这是 `weight` 版 `best.pt` 的结果，而当前 `best.pt` 是按 `val_bind_loss` 保存的 epoch 1，不代表最终较优 checkpoint。

- [etd_bind_diagnostics_summary.md](/media/scw-workspace/m6a_dataset/outputs/analysis/diagnostics/etd_bind_diagnostics_summary.md)
  - 早期 summary，信息已基本被 `etd_bind_diagnostics_matrices.md` 和 `etd_bind_diagnostics_weight_last.md` 覆盖。

## 推荐阅读顺序

如果你现在要快速看当前结论，按下面顺序：

1. [etd_bind_diagnostics_weight_last.md](/media/scw-workspace/m6a_dataset/outputs/analysis/diagnostics/etd_bind_diagnostics_weight_last.md)
2. [etd_bind_error_analysis_weight_last.md](/media/scw-workspace/m6a_dataset/outputs/analysis/error/etd_bind_error_analysis_weight_last.md)
3. [etd_bind_error_analysis_wo_weight.md](/media/scw-workspace/m6a_dataset/outputs/analysis/error/etd_bind_error_analysis_wo_weight.md)
4. [mod_drach_negative_eval.md](/media/scw-workspace/m6a_dataset/outputs/analysis/eval/mod_drach_negative_eval.md)
5. [drach_motif_stats_readable.md](/media/scw-workspace/m6a_dataset/outputs/analysis/motif/drach_motif_stats_readable.md)

## 当前一句话结论

- 单任务 bind 当前最佳版本是 `etd_bind_m6a_reader_neg_weight_0320` 的 `last.pt`。
- weighting 带来了真实但有限的提升，主要改善 `EIF3 / YTHDC / FMR1 / YTHDF`。
- `ELAVL1 / RBMX` 仍然没有学出来。
- mod 头对 DRACH motif 依赖很强，但不是纯 motif 检测器。
