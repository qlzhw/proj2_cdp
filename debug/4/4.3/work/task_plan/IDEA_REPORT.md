# Robotics Idea Discovery Report: CDP → ADP

**Direction**: 将CDP (Causal Diffusion Policy) 的固定窗口历史动作条件扩展为全历史/长历史动作条件 (ADP)，提升任务成功率  
**Date**: 2026-04-03  
**Pipeline**: research-lit → idea-creator (robotics framing) → novelty-check → research-review

---

## 1. Robotics Problem Frame

| 维度 | 值 |
|------|-----|
| **Embodiment** | 单臂机械手 (Adroit/MetaWorld/DexArt/RoboFactory) |
| **Task family** | 灵巧操作: hammer, pen, laptop, toilet, bin-picking, box-close, disassemble, reach, lift-barrier, place-food |
| **Observation** | 点云 (1024 pts, DP3Encoder) 或 多视角RGB (ResNet18 + spatial softmax) |
| **Action interface** | 关节角度/末端执行器增量, chunk size = 8, horizon = 40 |
| **Learning regime** | BC + DDPM/TEDi Causal Diffusion |
| **Available assets** | 4仿真环境, CDP代码库, zarr数据集 |
| **Compute budget** | 单卡 GPU |
| **Safety constraints** | 纯仿真, 无真实机器人 |
| **Desired contribution** | 方法改进: CDP → ADP, 引入全历史/长历史动作条件 |

---

## 2. 文献全景 (Landscape Matrix)

### 2.1 指定论文分析

| 论文 | 会议 | 核心机制 | 历史处理方式 | 对ADP的价值 |
|------|------|---------|------------|------------|
| **Hiveformer** (2209.04899) | CoRL 2022 | Full history token拼接 + full attention | 全历史观测+动作, keypose级(~10步) | 证明全历史有效, 但O(T²)不可扩展 |
| **LDP/PTP** (2505.09561) | CoRL 2025 Best Paper | Past-Token Prediction + 多阶段训练 + self-verification | 长观测上下文, 同时预测过去+未来动作 | **最核心参考**: PTP辅助loss + embedding缓存 + test-time候选选择 |
| **CDP** (2506.14769) | CoRL 2025 | TEDi式因果扩散 + CausalTransformer + KV cache | 固定窗口action buffer (L=20步) | ADP的直接基线 |
| **R3DP** (2603.14498) | Preprint 2026 | AFSC快慢系统 + VGGT蒸馏 + PRoPE | 感知端时序建模(TFPNet单步Markov) | 与CDP正交互补, 代码未发布 |

### 2.2 扩展文献关键发现

| 方向 | 代表论文 | 核心洞察 | 对ADP的启示 |
|------|---------|---------|------------|
| **SSM处理长序列** | MTIL (RA-L 2025), RoboMamba (NeurIPS 2024), MaIL (CoRL 2024) | Mamba O(n)编码full history, 已在IL中验证 | SSM是处理全历史的可行backbone, 但在低维action空间优势存疑 |
| **可变噪声扩散** | Diffusion Forcing (NeurIPS 2024) | 每个token独立噪声级别=时间距离编码 | 近历史低噪声/高精度, 远历史高噪声/可压缩 |
| **记忆增强策略** | MemoryVLA (2025), MemER (2025), HAMLET (2025) | Memory bank + cross-attention检索 | Retrieval方案已较拥挤, 不建议走此路线 |
| **多尺度动作生成** | CARP (ICCV 2025) | Multi-scale tokenization + coarse-to-fine AR | 生成端多尺度已有, 条件端多尺度是gap |
| **DiT扩展** | RDT-1B (ICLR 2025), ScaleDP (ICRA 2025), Dita (ICCV 2025) | 大规模DiT训练稳定性, ACI条件注入 | 深层Transformer需要ScaleDP的稳定化技术 |
| **高效推理** | LightDP (ICCV 2025), RDP (RSS 2025 Best Student Paper) | 剪枝+蒸馏压缩去噪步, Slow-fast双层架构 | 推理加速技术可后续集成 |
| **Latent空间长时域** | LoLA (2025) | Latent action space处理长时域比原始空间更有效 | 历史动作编码到latent space再做全历史注意力 |

### 2.3 核心瓶颈与Gap

| Gap | 论文证据 | 严重程度 |
|-----|---------|---------|
| 扩散策略严重低利用长历史 ("反向copycat问题") | LDP: Diffusion no-PTP在长上下文下崩溃至0%成功率 | **极高** |
| CDP固定窗口不支持超长horizon | CDP论文Limitation明确承认 | **高** |
| O(n²) attention不可扩展到全历史 | Hiveformer承认, 300步任务=上千token | **高** |
| 远期历史噪声干扰 vs 信息价值的trade-off | 无直接论文量化, 但物理直觉合理 | **中-高** |
| 条件端(conditioning-side)多尺度历史表示 | CARP只做生成端(generation-side)多尺度, 条件端空白 | **中** |

---

## 3. Ranked Ideas

### Idea 1: Dual-Stream ADP (Mamba History Encoder + PTP) — 可行但需迭代

**一句话**: 用Mamba编码全部历史动作为压缩状态, CausalTransformer处理本地窗口, PTP防止历史低利用。

| 维度 | 值 |
|------|-----|
| Embodiment | 单臂 (同CDP) |
| Benchmark | Adroit/DexArt/MetaWorld/RoboFactory |
| 瓶颈 | 固定窗口丢失长程历史, O(n²)不可扩展 |
| Pilot type | sim |
| 正向信号 | Long-horizon task (DexArt laptop/toilet) 成功率 > CDP+5% |
| Novelty | **MEDIUM** — MTIL (RA-L 2025)已用Mamba编码full history但无扩散过程 |
| 审稿分 | 5.5/10 |
| 硬件风险 | 无 |
| 核心风险 | Cross-attention注入设计薄弱; Mamba在低维action序列上优势未验证; 需证明vs MTIL+diffusion简单组合的增量性 |

**最强竞争者**: MTIL (RA-L 2025) — 已用Mamba-2编码完整历史轨迹, 虽无diffusion但思路重叠。

---

### Idea 2: 层次化历史压缩 + 多尺度因果扩散 — 新颖但过复杂

**一句话**: 三级时间层次(近期raw + 中期pooled tokens + 远期GRU/Mamba状态), 配合与时间距离绑定的可变噪声注入。

| 维度 | 值 |
|------|-----|
| Embodiment | 单臂 (同CDP) |
| Benchmark | Adroit/DexArt/MetaWorld/RoboFactory |
| 瓶颈 | 远期历史噪声干扰, 缺乏条件端多分辨率 |
| Pilot type | sim |
| 正向信号 | Variable noise vs fixed noise在long-horizon task上有显著差异 |
| Novelty | **MEDIUM-HIGH** — 条件端多尺度历史表示无直接竞品 |
| 审稿分 | 5/10 |
| 硬件风险 | 无 |
| 核心风险 | 超参数爆炸(8-10个新超参); multi-scale PTP梯度动态存疑; 消融组合爆炸 |

**最强竞争者**: CARP (ICCV 2025, 生成端多尺度) + Diffusion Forcing (NeurIPS 2024, 可变噪声)。

---

### Idea 3: 自适应历史选择 (Retrospective Attention) — 建议放弃

**一句话**: 维护历史action chunk bank, 注意力检索top-K相关片段, PTP self-verification选择候选。

| 维度 | 值 |
|------|-----|
| Novelty | **LOW-MEDIUM** |
| 审稿分 | 4/10 |
| 核心问题 | MemoryVLA (2025)已做memory bank + retrieval + diffusion, 差异不足; retrieval mechanism未定义; test-time overhead |

---

## 4. Eliminated Ideas

- **Idea 3 (Adaptive Retrieval)** — killed because: MemoryVLA/MemER/STRAP/EchoVLA已使"retrieval-augmented robot policy"极度拥挤, action-chunk memory vs observation memory的差异不足以支撑独立工作, retrieval mechanism是一个完整的研究问题而非可直接使用的module。

---

## 5. 外部评审关键建议与综合最优方案

### 5.1 审稿人的核心质疑 (必须回答)

> **"你还没有empirical evidence证明CDP的fixed-window确实是performance bottleneck。"**

如果现有benchmark上CDP的20步窗口已足够, 那无论如何扩展history, 性能都不会显著提升。**第一步必须做diagnostic experiment:**

```
实验0 (诊断性实验, 在ADP开发前必做):
- 将CDP的window从20缩小到5, 10, 看performance掉多少
- 将CDP的window暴力扩大到50, 100 (truncated attention), 看performance涨多少
- 如果20→100无显著提升, 则需要:
  (a) 找到真正需要long history的task/benchmark
  (b) 或重新审视research direction
```

### 5.2 综合最优方案 — RECOMMENDED

**从三个idea中提取精华, 大幅简化后的统一方案:**

| 组件 | 选择 | 理由 |
|------|------|------|
| **History Encoder** | **GRU** (不是Mamba) | Action序列是低维(7-20D), 长度~300步, 这个regime下GRU的simplicity是优势; Mamba优势(高维长序列)不一定体现; ablation更clean |
| **历史层级** | **2-level** (不是3-level) | Level 1: 最近20步raw actions (现有CDP); Level 2: GRU hidden state (全历史压缩) |
| **条件注入方式** | **Adaptive Layer Norm (AdaLN)** | GRU state通过AdaLN注入CausalTransformer, 比cross-attention更轻量更好训练, 在DiT中已广泛验证 |
| **训练辅助loss** | **PTP** (单尺度, 来自LDP) | 只在local window上做past-token prediction, 不搞multi-scale PTP |
| **噪声策略** | **简化variable noise** | GRU conditioning加一个learnable noise scale, 不搞per-token variable noise |
| **推理** | **GRU增量更新 + KV cache** | GRU state O(1)更新/步, KV cache保留给local window |
| **算法命名** | **adp2, adp3** | 维持现有dp2/cdp2/dp3/cdp3的命名约定 |

**Paper Narrative:**
1. **Problem**: CDP固定窗口限制long-horizon鲁棒性, LDP证明扩散策略低利用历史
2. **Solution**: 轻量GRU压缩全历史为persistent state, AdaLN条件注入, PTP正则化
3. **Core Contribution**: (a) 首个全历史因果扩散策略扩展; (b) 证明简单压缩+PTP即已足够(Mamba/层次化/检索都不必要)
4. **Story核心**: **simplicity works** — 用最简单的方法解决问题, ablation证明复杂方案不值得

**预估审稿分**: 6-6.5/10, ablation扎实+long-horizon有convincing improvement可到7。

### 5.3 LDP Past-Token Prediction 与 CDP/ADP 的结合分析

用户特别关注此问题, 详细分析如下:

**LDP的PTP机制回顾:**
- 训练时: 模型预测horizon内所有位置(包括过去动作位置), loss覆盖past+future
- 推理时: 采样N个候选, 比较每个候选的predicted past actions vs actual past actions的MSE, 选MSE最小的

**与CDP结合的具体方案:**

```python
# CDP的compute_loss当前只在action steps (To:)上计算loss
# PTP修改: 扩展loss到observation steps (0:To), 即也预测历史动作

# 训练时:
pred = self.model(noisy_trajectory, diff_steps, cond=cond)
# 原始: loss = MSE(pred[:, To:], target[:, To:])  # 只预测未来
# PTP:  loss = MSE(pred, target) * loss_mask        # 预测过去+未来

# 推理时 (test-time self-verification):
N_candidates = 5
candidates = [self.conditional_sample(cond) for _ in range(N_candidates)]
past_actions = self.action_buffer[:, :To]  # 实际历史动作
mse_scores = [MSE(c[:, :To], past_actions) for c in candidates]
best = candidates[argmin(mse_scores)]
```

**可行性**: 高。CDP的CausalTransformer已经在horizon的所有位置有输出, 只需扩展loss mask。
**预期收益**: LDP在Diffusion Policy上用PTP将长上下文性能提升3倍。CDP本身已有历史动作条件化, PTP应能进一步强化其利用。
**风险**: 如果CDP的causal conditioning已经足够强制利用历史, PTP的边际收益可能较小。需要实验验证。

### 5.4 用户六点关切的逐一回应

| 用户关切 | 方案回应 |
|---------|---------|
| **①远期动作噪声干扰** | GRU的hidden state天然实现指数衰减加权, 近期信息占主导, 远期信息自然衰减, 无需手动设计权重; R3DP中TFPNet的类似做法可参考 |
| **②历史动作训练trick** | CDP的noise injection保留(causal_condition_noise_weight); 新增PTP辅助loss; 可选: 对GRU输入做小概率mask(Hiveformer启发的dropout) |
| **③计算复杂度/推理延迟** | GRU更新O(1)/步, CausalTransformer local window不变, KV cache保留 → 推理延迟几乎不增加; 后续可集成Flash Attention |
| **④LLM百万上下文类比** | 技术可行但非必要: 300步action序列远小于LLM上下文; GRU压缩比全注意力更高效; 全注意力方案作为ablation保留 |
| **⑤历史动作 vs 历史观测 vs 两者** | 推荐先只做历史动作(ADP核心); 观测端可后续集成R3DP的AFSC; LDP的启示是两者各有价值但机制不同 |
| **⑥LDP能否与CDP/全历史结合** | **能。** 具体方案见5.3节。PTP+GRU全历史压缩是最优组合。Test-time self-verification可直接复用CDP的action buffer。 |

---

## 6. Evidence Package for the Recommended Approach

### 6.1 必做实验

| 实验 | 目的 | 优先级 |
|------|------|--------|
| **实验0: 诊断性窗口扫描** | 验证固定窗口确实是瓶颈 (window=5/10/20/50/100) | **P0 (最先做)** |
| **实验1: ADP(GRU+AdaLN+PTP) vs CDP** | 主实验, 7-8个task, 3 seeds | P1 |
| **实验2: 消融 — ADP w/o PTP** | 隔离PTP的贡献 | P1 |
| **实验3: 消融 — ADP w/o GRU (只用PTP)** | 隔离GRU的贡献 | P1 |
| **实验4: GRU vs Mamba vs LSTM** | 证明GRU足够或说明何时需要更强模型 | P2 |
| **实验5: AdaLN vs Cross-Attention vs FiLM** | 条件注入方式对比 | P2 |
| **实验6: History length scaling** | 成功率 vs 有效历史长度曲线 (20/50/100/200/full) | P1 |
| **实验7: Test-time self-verification** | N=1/3/5/10候选的成功率对比 | P2 |
| **实验8: 推理延迟对比** | Wall-clock time per step, 与CDP和暴力扩窗对比 | P1 |

### 6.2 必须对比的Baseline

- CDP (固定窗口20步) — 直接前身
- DP3 (无历史动作) — lower bound
- CDP + PTP (无GRU, 只加PTP loss) — 隔离PTP
- CDP (窗口扩大到100, truncated attention) — naive scaling
- LDP/PTP原始方法 (如果能在相同benchmark复现)

### 6.3 必须报告的Metrics

- 成功率 (success rate) ± std, 3 seeds
- 推理延迟 (ms/step)
- GPU显存占用 (MB)
- 训练时间 (hours)
- 失败案例分析 (至少3个task的failure mode对比)

### 6.4 是否需要真实机器人验证

**不需要** (当前阶段)。仿真实验覆盖4个环境10个task已足够。真实机器人验证可作为future work。

---

## 7. 实现路线图

### 7.1 代码架构变更

```
diffusion_policy/
├── policy/
│   ├── adp2.py          # NEW: ADP with multi-image obs
│   ├── adp3.py          # NEW: ADP with point cloud obs
│   ├── cdp2.py          # UNCHANGED
│   └── cdp3.py          # UNCHANGED
├── model/
│   └── diffusion/
│       ├── causal_transformer.py  # MODIFIED: 加入AdaLN conditioning from GRU state
│       └── history_encoder.py     # NEW: GRU history encoder
config/
├── adp2.yaml            # NEW
├── adp3.yaml            # NEW
train.py                  # MODIFIED: _POLICY_MAP加入adp2, adp3
scripts/train_policy.sh   # MODIFIED: 支持adp2, adp3
```

### 7.2 开发顺序

```
Phase 1 (诊断, ~1-2天):
  - 实验0: CDP窗口扫描 (验证方向有效性)

Phase 2 (核心开发, ~1周):
  - 实现history_encoder.py (GRU)
  - 实现adp3.py (基于cdp3.py改)
  - 实现AdaLN条件注入
  - 实现PTP训练loss
  - 配置adp3.yaml

Phase 3 (验证, ~1周):
  - 实验1-3, 6, 8 (主实验+关键消融)
  - 分析结果, 调整方案

Phase 4 (完善, ~3-5天):
  - 实验4-5, 7 (次要消融)
  - adp2实现和验证
  - 失败案例分析
```

---

## 8. Next Steps

- [ ] **最优先**: 跑实验0 — CDP窗口扫描诊断 (window=5/10/20/50/100), 验证方向有效性
- [ ] 确认实验0结果后, 决定是否推进ADP开发
- [ ] 实现GRU history encoder + AdaLN + PTP
- [ ] 在Adroit/DexArt的long-horizon task上优先验证
- [ ] 只在确认仿真结果后考虑真实机器人

---

## 9. 附录: 关键文献引用

### 指定论文
1. Hiveformer: Instruction-driven history-aware policies for robotic manipulations (CoRL 2022) [arXiv:2209.04899]
2. LDP: Learning Long-Context Diffusion Policies via Past-Token Prediction (CoRL 2025, Best Paper) [arXiv:2505.09561]
3. CDP: Towards Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion (CoRL 2025) [arXiv:2506.14769]
4. R3DP: Real-Time 3D-Aware Policy for Embodied Manipulation (2026 preprint) [arXiv:2603.14498]

### 核心扩展文献
5. TEDi: Temporally-Entangled Diffusion for Long-Term Motion Synthesis (SIGGRAPH 2024) [arXiv:2307.15042]
6. TEDi Policy: Temporally Entangled Diffusion for Robotic Control (2024) [arXiv:2406.04806]
7. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion (NeurIPS 2024)
8. MTIL: Encoding Full History with Mamba for Temporal Imitation Learning (RA-L 2025) [arXiv:2505.12410]
9. RoboMamba: Multimodal State Space Model for Efficient Robot Reasoning (NeurIPS 2024) [arXiv:2406.04339]
10. MaIL: Improving Imitation Learning with Mamba (CoRL 2024) [arXiv:2406.08234]
11. CARP: Visuomotor Policy Learning via Coarse-to-Fine Autoregressive Prediction (ICCV 2025) [arXiv:2412.06782]
12. MemoryVLA: Perceptual-Cognitive Memory in VLA Models (2025) [arXiv:2508.19236]
13. RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation (ICLR 2025) [arXiv:2410.07864]
14. Dita: Scaling Diffusion Transformer for Generalist VLA Policy (ICCV 2025) [arXiv:2503.19757]
15. ScaleDP: Scaling Diffusion Policy in Transformer (ICRA 2025)
16. LightDP: On-Device Diffusion Transformer Policy (ICCV 2025)
17. RDP: Reactive Diffusion Policy (RSS 2025 Best Student Paper Finalist) [arXiv:2503.02881]
18. HAMLET: Switch your VLA Model into a History-Aware Policy (2025) [arXiv:2510.00695]
19. LoLA: Long Horizon Latent Action Learning (2025) [arXiv:2512.20166]
20. TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness (ICLR 2025) [arXiv:2412.10345]
21. ACT: Action Chunking with Transformers (2023) [arXiv:2304.13705]
22. HULC/HULC++: Hierarchical Universal Language Conditioned Policies (RA-L 2022 / ICRA 2023)
23. MemER: Scaling Up Memory for Robot Control via Experience Retrieval (2025) [arXiv:2510.20328]
24. STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning (ICLR 2025) [arXiv:2412.15182]
25. RoboMME: Benchmarking Memory for Robotic Generalist Policies (2026) [arXiv:2603.04639]

---

## 10. GPT-5.4 独立外部评审 (via Codex MCP)

> 以下为 GPT-5.4 作为独立 CoRL/RSS/ICRA 审稿人的评审意见, 与上述 Claude 评审形成交叉验证。

### 10.1 评分对比

| 方案 | Claude 评审 | GPT-5.4 评审 | 建议 |
|------|-----------|-------------|------|
| **推荐组合** (GRU+AdaLN+PTP) | 6-6.5/10 | **6/10** | pursue |
| Idea 1 (Dual-Stream Mamba) | 5.5/10 | **5/10** | iterate |
| Idea 2 (层次化压缩) | 5/10 | **4/10** | **abandon** |
| Idea 3 (自适应检索) | 4/10 | **5/10** | iterate |

两位审稿人一致认为: **推荐组合方案是最佳起点, Idea 2 风险最高。**

### 10.2 GPT-5.4 提出的关键新洞察 (Claude 未覆盖)

**① 核心判断: "问题到底是不是容量(capacity)"**

> 300步根本不够长到justify三层层级压缩。真正高概率瓶颈是**历史利用不足(history under-utilization)**，不是历史装不下。如果你的表里没有朴素 CDP-L300、CDP-L300+PTP 这些基线，reviewer非常倾向拒稿。

**② 动作历史可能只是 phase clock**

> 模型可能只是靠"已经执行了多少步"推断阶段，而不是在用有语义的历史信息。必须通过 history intervention（截断、打乱、跨episode替换）来验证模型是否真正利用了历史语义。

**③ Action-only 假设太强**

> 真正有效的长期信息，往往来自观测-动作联合轨迹(obs+action joint history)，不是动作单独一条线。必须加 `action-only` vs `obs+action` 消融实验。

**④ Exposure bias 随历史长度恶化**

> 历史越长，条件里越多自生成动作（而非专家动作）；这可能让 long-history 方案在闭环 rollout 中更脆弱。CDP现有的历史动作噪声注入未必足够应对300-step自条件漂移。

**⑤ Self-verification 容易偷算力**

> 如果多采样N个候选再选最优，而baseline没有同等预算，reviewer会认为这是unfair compute advantage。必须对齐计算预算。

**⑥ 命名过度承诺**

> GRU hidden state不是"all-history access"，只是"full-history compression"。叫"ADP (All-history)"容易被抓字眼。建议改为 long-history 或 history-utilized diffusion policy。

### 10.3 GPT-5.4 建议的最强 paper 叙事

> 你真正该讲的故事不是"All-history Diffusion Policy"，而是:
> **"Diffusion policy 不会用历史；给它全历史访问并不够，还要显式逼它利用历史"**
> 这个叙事比ADP更硬，也更容易自圆其说。

### 10.4 GPT-5.4 加强版诊断实验 (实验0)

```
必须在commit任何方案前完成:

① 窗口扫描: CDP-L20 / CDP-L100 / CDP-L300
② PTP独立贡献: CDP-L20+PTP / CDP-L300+PTP
③ 测试时历史干预 (history intervention):
   - truncate: 截断20步之前的历史
   - shuffle: 打乱历史动作顺序
   - cross-episode replace: 用其他episode的动作替换远期历史
④ 以上全部在正常观测 + 观测退化两种条件下评估

判断标准:
- L300 ≈ L20 → 方向错了，别做all-history
- PTP提升大但L300本身不提升 → 问题是utilization，不是capacity
- L300+PTP显著提升 + 对old-action corruption敏感 → 值得继续投memory architecture
```

### 10.5 GPT-5.4 指出的盲点

- 300步不算真正长序列, 复杂记忆结构缺乏necessity argument
- 动作历史可能只编码了"phase"而非语义信息
- 全历史动作条件会放大exposure bias (自条件漂移)
- Self-verification的计算预算必须与baseline对齐
- MetaWorld某些任务过短, RoboFactory不够主流, benchmark bias风险
- 如果gain只在观测退化下出现, claim应收缩为robustness而非general improvement
