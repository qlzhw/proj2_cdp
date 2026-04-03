# Pipeline Summary

**Problem**: CDP固定窗口(32步历史动作+32帧观测)丢失长程历史，扩散策略不利用历史
**Final Method Thesis**: CDP local buffer负责短期因果连续性，HistoryMamba负责extra-window memory，QC-Remote-PTP逼模型真正使用这段memory
**Final Verdict**: REVISE* (8.5/10, 方法闭环可实现，新颖性取决于实验结果)
**Date**: 2026-04-03

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- **Dominant contribution**: Extra-window Mamba dual-history memory (obs+action, split-pool chunk tokens) + QC-Remote-PTP (query-conditioned, strictly extra-window utilization supervision) for CDP
- **Optional supporting contribution**: Two-stage training with cached obs embeddings (LDP-validated recipe)
- **Explicitly rejected complexity**: Self-verification, multi-scale compression, memory bank/retrieval, separate obs/action encoders, Transformer for history

## Must-Prove Claims
- C1: Extra-window memory improves long-horizon success rate (≥5% vs CDP)
- C2: QC-Remote-PTP is essential for utilization (w/o-PTP degrades)
- C3: Model truly uses history semantics (corruption → ≥10% drop)
- C4: Inference cost acceptable (<30% latency increase)

## First Runs to Launch
1. **M0 Sanity**: ADP-full overfit on adroit_hammer (验证pipeline)
2. **M1 Gating**: CDP-L20/L50/L100 × 3 tasks (验证fixed window是瓶颈)
3. **M2 Cache**: 缓存所有task的DP3Encoder obs embeddings

## Main Risks
- **[HIGH] Gating diagnostic negative**: CDP-L100≈L20 → 固定窗口可能不是bottleneck → 调整paper story为"memory+PTP makes longer context actually useful"
- **[MEDIUM] ADP-full ≤ CDP**: Extra-window memory在当前benchmark不有效 → 寻找更长horizon任务
- **[LOW] Mamba step()推理不稳定**: 降级为GRU backup

## Next Action
- `/run-experiment` to deploy M0 sanity + M1 gating diagnostic
