# Experiment Plan

**Problem**: CDP固定窗口丢失长程历史 + 扩散策略不利用历史
**Method Thesis**: Extra-window Mamba memory + QC-Remote-PTP for CDP
**Date**: 2026-04-03

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: Extra-window memory improves long-horizon success | 主论文核心 | ADP-full > CDP ≥5% on ≥3 tasks | B1, B2 |
| C2: QC-Remote-PTP essential for utilization | 证明不只是加参数 | ADP-w/o-PTP < ADP-full | B3 |
| C3: Model truly uses history semantics | 排除phase clock | History corruption → ≥10% drop | B4 |
| C4: Inference cost acceptable | 实际可用性 | Latency <30% increase | B5 |

## Paper Storyline
- Main paper must prove: C1 (main table) + C2 (PTP necessity) + C3 (history intervention) + C4 (latency)
- Appendix can support: obs-only vs act-only vs full分离; Mamba vs GRU; AdaLN vs CrossAttn
- Experiments intentionally cut: adp2; test-time self-verification; multi-scale compression

## Experiment Blocks

### Block 0: Gating Diagnostic (MUST-RUN, Pre-ADP)
- **Claim tested**: CDP的固定窗口确实是瓶颈
- **Why**: 如果CDP窗口从20扩到100不变化，整个方向失去立项基础
- **Dataset/task**: Adroit hammer, DexArt laptop, MetaWorld box-close
- **Compared systems**: CDP-L20(default), CDP-L50, CDP-L100 (truncated attention, no PTP)
- **Metrics**: Success rate (1 seed足够做go/no-go决策)
- **Setup**: 修改cdp3的n_obs_steps/horizon参数, 其他不变
- **Success criterion**: (a) L100 > L20有提升 → capacity有价值, proceed; (b) L100 ≈ L20 → 需要PTP, still proceed but adjust story; (c) L100 < L20 → 方向有问题, pause
- **Failure interpretation**: 如果L100崩溃, 印证LDP的"不加PTP则利用率极低"发现
- **Table/figure target**: 论文Table 1的motivation experiment
- **Priority**: MUST-RUN (最先做)

### Block 1: Main Comparison Table (MUST-RUN)
- **Claim tested**: C1 — Extra-window memory improves long-horizon
- **Why**: 论文核心结果
- **Dataset/task**: DexArt laptop, Adroit hammer, Adroit pen (long-horizon) + MetaWorld reach (short, negative control)
- **Compared systems**: (1) CDP, (2) ADP-obs-only, (3) ADP-full, (4) CDP-L100
- **Metrics**: Success rate ± std (3 seeds)
- **Setup**: Stage 1用现有CDP checkpoint; Stage 2训练2000 epochs; 相同seed/eval protocol
- **Success criterion**: ADP-full > CDP ≥5% on all 3 long-horizon tasks; ADP-full ≈ CDP on short-horizon
- **Failure interpretation**: 如果ADP-full ≤ CDP → extra-window memory对这些任务不必要, 可能需要更长horizon任务
- **Table/figure target**: Table 2 (Main Results)
- **Priority**: MUST-RUN

### Block 2: Obs vs Action History Decomposition (MUST-RUN, part of main table)
- **Claim tested**: C1 supplementary — obs和action历史的各自贡献
- **Why**: 验证双历史设计的必要性
- **Dataset/task**: Same as Block 1
- **Compared systems**: ADP-full vs ADP-obs-only (已在Block 1) + ADP-act-only (appendix)
- **Metrics**: Same
- **Success criterion**: ADP-full > ADP-obs-only ≥ ADP-act-only
- **Failure interpretation**: 如果obs-only≈full → action history贡献小, 可收缩claim
- **Table/figure target**: Table 2 (same table, ADP-obs-only column)
- **Priority**: MUST-RUN (obs-only in main, act-only in appendix)

### Block 3: PTP Necessity Ablation (MUST-RUN)
- **Claim tested**: C2 — QC-Remote-PTP is essential
- **Why**: 区分"加了记忆"和"加了记忆+强制利用"的差异; 复现LDP核心发现
- **Dataset/task**: Same 4 tasks as Block 1
- **Compared systems**: ADP-full vs ADP-full-w/o-PTP (identical architecture, ptp_weight=0)
- **Metrics**: Success rate ± std (3 seeds)
- **Success criterion**: ADP-w/o-PTP < ADP-full, gap ≥ 3%
- **Failure interpretation**: 如果w/o-PTP≈full → Mamba memory自身已足够利用历史, QC-Remote-PTP不关键, 需调整paper story
- **Table/figure target**: Table 3 (Ablation)
- **Priority**: MUST-RUN

### Block 4: History Intervention Test (MUST-RUN)
- **Claim tested**: C3 — Model truly uses history semantics, not phase clock
- **Why**: 排除模型只用步数/阶段信息而非历史语义的可能
- **Dataset/task**: DexArt laptop, Adroit hammer (2 tasks够)
- **Compared systems**: ADP-full with:
  - Normal history (baseline)
  - Truncated: 截断extra-window到最近K/2 chunks
  - Shuffled: 打乱extra-window chunks顺序
  - Cross-episode replaced: 用其他episode的extra-window替换
- **Metrics**: Success rate (1 seed, inference only — 不需要重新训练)
- **Success criterion**: All corruptions → ≥10% drop vs normal
- **Failure interpretation**: 如果shuffle不掉 → 模型只用chunk statistics, 不用temporal order; 如果replace不掉 → 模型不依赖specific episode history
- **Table/figure target**: Table 4 (History Intervention)
- **Priority**: MUST-RUN

### Block 5: Inference Cost (MUST-RUN)
- **Claim tested**: C4 — Acceptable latency and memory
- **Why**: 实际可用性
- **Dataset/task**: Any single task, 100 episodes
- **Compared systems**: CDP vs ADP-full
- **Metrics**: Wall-clock ms/step (mean±std), peak GPU memory (MB)
- **Setup**: 同一GPU上profile, warmup 10 episodes then measure
- **Success criterion**: ADP latency ≤ 1.3× CDP latency
- **Failure interpretation**: 如果>1.3× → 需要优化Mamba step或减少Mamba layers
- **Table/figure target**: Table 5 (Efficiency)
- **Priority**: MUST-RUN

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0: Sanity | Data pipeline + metric正确性 | 1 overfit run on smallest task | Loss收敛, eval runs | 4h | Low |
| M1: Gating Diagnostic | 验证fixed window是瓶颈 | CDP-L20/L50/L100 × 3 tasks × 1 seed = 9 runs | L100≠L20 | 27h (3.4h each) | **HIGH** — 如果不变则方向需调整 |
| M2: Stage 1 + Cache | CDP checkpoint + cache obs embeddings | 1 Stage 1 per task (or reuse) + caching | Embeddings cached | 2h caching | Low |
| M3: Main Method | ADP-full + ADP-obs-only training | 2 systems × 4 tasks × 3 seeds = 24 runs | ADP-full > CDP | 72h (3h each, 8 GPU parallel) | Medium |
| M4: Decision Ablation | PTP necessity + history intervention | 12 runs (training) + 8 inference runs | PTP matters, corruption drops | 36h + 2h | Medium |
| M5: Polish | Latency profiling + appendix experiments | 2 profiling + ~20 appendix runs | Latency acceptable | 60h | Low |

## Compute and Data Budget
- Total estimated GPU-hours: ~560 (must-run ~300, appendix ~260)
- Data preparation: Zero (existing zarr datasets)
- Human evaluation: None (automated success rate)
- Biggest bottleneck: M1 gating diagnostic (如果结果不支持, 需要调整方向)
- Hardware: 8× GPU 48GB, 并行capacity=8 runs

## Risks and Mitigations
- **[HIGH] Gating diagnostic negative**: CDP-L100≈CDP-L20 → 固定窗口不是瓶颈 → Mitigation: 仍然proceed, 但paper story改为"even if longer window doesn't help naively, memory+PTP does"
- **[MEDIUM] ADP-full ≤ CDP**: Extra-window memory在当前benchmark不有效 → Mitigation: 寻找更长horizon的任务; 检查GRU vs Mamba差异
- **[MEDIUM] QC-Remote-PTP无效**: w/o-PTP≈full → Mitigation: 调整ptp_weight; 检查Mamba是否已自主利用历史; 调整paper story
- **[LOW] Mamba step()推理不稳定**: 与training不一致 → Mitigation: 用full forward验证; 降级为GRU
- **[LOW] Stage transition gap**: Stage 2起点差 → Mitigation: 验证缓存; 降低new module lr

## Final Checklist
- [x] Main paper tables are covered (Table 1-5)
- [x] Novelty is isolated (Block 3: PTP ablation)
- [x] Simplicity is defended (not comparing overcomplicated variants in main paper)
- [x] Frontier contribution is justified (Mamba + PTP = modern but natural tools)
- [x] Nice-to-have runs are separated from must-run runs
