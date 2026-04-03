# Review Summary

**Problem**: CDP固定窗口历史 + 扩散策略不利用历史 → 如何让CDP用到窗外历史
**Initial Approach**: GRU压缩全历史动作 + PTP辅助loss
**Date**: 2026-04-03
**Rounds**: 5 / 5
**Final Score**: 8.5 / 10
**Final Verdict**: REVISE* (方法已闭环，可实现；未达9是因新颖性上限取决于实验结果)

## Problem Anchor
CDP的CausalTransformer在固定horizon=40的动作序列窗口内运行（前32步历史动作buffer + 后8步待生成动作），窗口外所有历史被丢弃。Paper identity: CDP扩展论文。

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Fixed | Solved? | Remaining |
|-------|-------------------------|----------------------|---------|-----------|
| 1 | HistoryGRU接口错误; PTP不约束窗外; PTP应为核心不是optional; 实验太散 | — (baseline review) | — | All |
| 2 | PTP→Remote-PTP(窗外); PTP升核心; Self-verification删除; GRU→Mamba; 加obs历史; 实验压缩 | ✓ PTP机制, ✓ Self-verif删除, ✓ 双历史 | Partial | Drift detected; Remote-PTP不严格; 序列化不精确 |
| 3 | Drift修复→锁回CDP; Remote-PTP→QC(query-conditioned); 序列化契约精确化; Obs claim收缩 | ✓ Drift修复, ✓ QC-Remote-PTP, ✓ 序列化 | Partial | 时间索引不统一; QC queries可能落在local buffer内; Train/infer不一致 |
| 4 | τ-based统一索引; HistoryMamba→only extra-window; QC queries严格extra-window; Chunk-level同构 | ✓ τ索引, ✓ Extra-window only, ✓ 同构 | Partial | Chunk mean-pool vs 4-step raw target粒度失配 |
| 5 | Split-pool替代mean-pool; QC-Remote-PTP target对齐为chunk summary | ✓ 粒度对齐 | ✓ All | None (方法层面) |

## Overall Evolution
- **Round 1→2**: 从action-only GRU扩展到obs+action Mamba双历史（用户纠正+文献驱动）
- **Round 2→3**: 修复Problem Anchor drift，锁回CDP扩展定位
- **Round 3→4**: 精确化时间索引(τ-based)，HistoryMamba收缩为extra-window only
- **Round 4→5**: 最后一个粒度对齐修复(split-pool)
- **每轮都在简化而非添加**: self-verification删除, adp2降为后续, 实验压缩, paper identity收窄

## Final Status
- Anchor: **preserved** (confirmed Round 3-5)
- Focus: **tight** — "extra-window memory + utilization supervision for CDP"
- Modernity: **appropriate** — Mamba + AdaLN-Zero + cached embeddings + PTP-style supervision
- Strongest: 清晰的三区划分(extra-window / local buffer / generation), QC-Remote-PTP机制严格性
- Remaining: 新颖性上限取决于实验结果能否证明"必要且有效的机制改进"而非"干净工程整合"
