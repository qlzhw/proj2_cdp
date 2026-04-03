# Refinement Report

**Problem**: CDP固定窗口历史限制long-horizon任务成功率
**Initial Approach**: GRU压缩全历史动作 + PTP
**Date**: 2026-04-03
**Rounds**: 5 / 5
**Final Score**: 8.5 / 10
**Final Verdict**: REVISE* (方法闭环，可实现)

## Problem Anchor
CDP的CausalTransformer在固定horizon=40的动作序列窗口内运行，窗口外所有历史被丢弃。Paper identity: CDP扩展。

## Output Files
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 8                | 5                  | 6                    | 7                 | 7           | 5                | 6               | 6.3     | REVISE  |
| 2     | 6                | 6                  | 7                    | 8                 | 8           | 8                | 7               | 6.9     | REVISE  |
| 3     | 8                | 6                  | 8                    | 8                 | 8           | 8                | 7               | 7.5     | REVISE  |
| 4     | 8                | 7                  | 8                    | 8                 | 8           | 8                | 8               | 7.8     | REVISE  |
| 5     | 9                | 9                  | 8                    | 8                 | 9           | 8                | 8               | 8.5     | REVISE* |

## Round-by-Round Review Record

| Round | Main Concerns | What Changed | Result |
|-------|---------------|--------------|--------|
| 1 | GRU接口错误; PTP不约束窗外; 实验太散 | baseline评审 | 6.3 |
| 2 | 加obs历史+Mamba; Remote-PTP; 删self-verif → **drift detected** | 重大修订 | 6.9 |
| 3 | drift修复; QC-Remote-PTP; 序列化契约 | 锁回CDP扩展 | 7.5 |
| 4 | τ-based索引; extra-window only; 同构化 | 精确化 | 7.8 |
| 5 | split-pool粒度对齐 | 最终修复 | 8.5 |

## Final Proposal Snapshot
- CDP local buffer(32步)负责短期因果连续性
- HistoryMamba(2层Mamba-2, split-pool chunk tokens)编码extra-window obs+action历史
- AdaLN-Zero将z_hist注入CausalTransformer各层
- QC-Remote-PTP通过query-conditioned监督强制利用extra-window memory
- Two-stage training(LDP-validated): 冻结DP3Encoder, 缓存embedding

## Method Evolution Highlights
1. **最重要的简化**: 删除self-verification, 只做adp3, 收窄paper identity
2. **最重要的机制升级**: PTP→QC-Remote-PTP(query-conditioned, strictly extra-window, granularity-aligned)
3. **最重要的架构改变**: GRU→Mamba, action-only→obs+action双历史(用户主导+文献驱动)

## Pushback / Drift Log
| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | PTP是optional | 升为核心(GPT-5.4正确) | accepted |
| 1 | 加memory tokens | 拒绝(Mamba state够用) | rejected |
| 2 | Problem Anchor drifted | 锁回CDP扩展(GPT-5.4正确) | accepted |
| 3 | HistoryMamba应只编码extra-window | 接受(更清晰) | accepted |
| 4 | Mean-pool与target粒度失配 | 改为split-pool(GPT-5.4正确) | accepted |

## Remaining Weaknesses
- 贡献本质是"已知现代组件在CDP上的干净整合"，新颖性上限取决于实验结果
- QC-Remote-PTP可能受数据分布中phase prior影响(需history intervention排除)
- Mamba-2的step()推理接口稳定性需要验证

## Next Steps
- Proceed to `/experiment-plan` for detailed experiment roadmap
- Then `/run-experiment` to deploy
