# 对话总结：Causal-Diffusion-Policy 与 3D-Diffusion-Policy 排查记录

## 1. 仓库关系与总体判断

- `Causal-Diffusion-Policy` 明显基于 `3D-Diffusion-Policy` 演化而来。
- 但差异**不只在算法代码**，非算法层也有实质变化，主要包括：
  - 训练入口与日志逻辑
  - 数据集字段与观测定义
  - 任务配置
  - 评测入口
  - runner 输入字段

结论：

- 两者底座和大量工程代码同源。
- 但如果说“除了算法其余都一样”，这个判断不成立。

## 2. 数据生成、训练、评测的比较结论

### 数据生成

- 两个仓库都沿用同一批第三方专家演示生成流程：
  - Adroit 走 `VRL3`
  - DexArt 走 `dexart-release`
  - Metaworld 走 `third_party/Metaworld`
- 因此数据生成流程**大体一致**。
- 但脚本细节并非完全相同，例如输出目录、默认 GPU 设置等有差异。

### 训练

- `3D-Diffusion-Policy` 的训练入口相对简单，主要围绕 `DP3`。
- `Causal-Diffusion-Policy` 的训练入口支持 `dp2/cdp2/dp3/cdp3` 多种策略映射。
- `Causal-Diffusion-Policy` 还改了日志行为、validation 流程以及采样误差统计逻辑。

### 评测

- `3D-Diffusion-Policy` 有独立 `eval.py`。
- `Causal-Diffusion-Policy` 没有独立同级 `eval.py`，评测逻辑主要在 `train.py` 内部保留 `eval()` 方法。
- 两者 runner 也不完全相同，特别是 `AdroitRunner` 的输入字段与统计方式有差异。

## 3. 关于 horizon、n_obs_steps、n_action_steps 的澄清

这三个量分别控制不同的语义：

- `horizon`：模型一次采样生成的整条未来动作序列长度。
- `n_obs_steps`：模型输入的历史观测长度。
- `n_action_steps`：从整条预测动作序列中切出来、真正执行给环境的动作长度。

### 默认配置下的含义

#### 3D-Diffusion-Policy

- `horizon = 16`
- `n_obs_steps = 2`
- `n_action_steps = 8`

含义：

- 预测 16 步动作
- 看 2 步历史观测
- 实际执行 8 步动作

#### Causal-Diffusion-Policy（CDP）

- `horizon = 40`
- `n_obs_steps = 32`
- `n_action_steps = 8`

含义：

- 预测 40 步动作
- 看 32 步历史观测
- 实际执行 8 步动作

## 4. 关于“训练误差/采样误差”的关键结论

这里区分两种东西：

- 主训练损失：真正参与反向传播更新参数的 loss
- 采样误差日志：训练中额外打印出来监控的指标

### CDP 中的真实训练损失

- CDP 的主训练损失在 policy 内部，是 masked MSE。
- 这部分不是只比较固定窗口，也不是只比较 `16:20`。

也就是说：

- **论文训练目标不是只对 `16:20` 做 loss**
- `16:20` 只是训练脚本里一个额外日志指标的切片方式

### CDP 中原先的问题

在 `Causal-Diffusion-Policy/train.py` 中，原先日志误差写死为：

- `pred_action[:, 16:20, :]`
- `gt_action[:, 16:20, :]`

这带来两个问题：

- 它只是一个固定窗口，不是动态和配置对齐的窗口
- 它和当前 CDP 默认真实执行窗口并不一致

CDP 默认实际执行窗口对应：

- `start = n_obs_steps = 32`
- `end = start + n_action_steps = 40`

所以原来的 `16:20`：

- 不影响主训练损失
- 不影响正常 rollout 评测
- 但会误导训练日志解读
- 在改配置时还可能导致指标失真甚至失效

## 5. 是否在论文中找到 16:20 的解释

检索结论：

- 没有在 CDP 论文、项目页、README 中找到对 `16:20` 的专门解释。
- 论文解释了：
  - 引入历史动作的原因
  - error accumulation 的问题
  - causal diffusion 的设计动机
- 但**没有解释为什么日志误差要看 `16:20`**。

因此更稳妥的判断是：

- `16:20` 不是论文明确规定的方法设计
- 更像是代码里的硬编码遗留监控窗口

## 6. 3D-Diffusion-Policy 里有没有同样的问题

结论：

- `3D-Diffusion-Policy` **没有** `16:20` 这个硬编码问题。

它的日志误差写法是：

- 直接对整段 `pred_action` 和 `gt_action` 做 MSE

因此：

- 没有固定神秘窗口
- 不存在和当前配置明显错位的 `16:20` 问题

但仍要注意：

- 它记录的是**整段预测误差**
- 不是“实际执行 chunk 的误差”

所以 3D 没有 CDP 那种硬编码脏点，但日志口径也不等于执行窗口口径。

## 7. 已对 CDP 做的代码修改

按照讨论结果，已经对 `Causal-Diffusion-Policy` 做了修正。

### 修改目标

- 去掉 CDP 训练脚本里固定写死的 `16:20`
- 改为基于配置动态计算误差窗口

### 修改原则

- 不改主训练损失
- 不改 rollout 评测逻辑
- 只修正日志误差窗口
- 同时保持 DP 与 CDP 各自原本的动作起点语义

### 修改后的逻辑

在 `train.py` 中新增：

- `sample_mse_start = cfg.n_obs_steps`
- 如果是 `train_dp2` 或 `train_dp3`，则 `sample_mse_start -= 1`
- `sample_mse_end = sample_mse_start + cfg.n_action_steps`

因此：

- 对 `cdp2/cdp3`，日志误差窗口变为：
  - `start = n_obs_steps`
  - `end = n_obs_steps + n_action_steps`
- 对 `dp2/dp3`，日志误差窗口保持和原执行窗口对齐：
  - `start = n_obs_steps - 1`
  - `end = start + n_action_steps`

### 修改效果

对默认 CDP 配置：

- 原先误差窗口：`16:20`
- 修改后误差窗口：`32:40`

这与 CDP 实际执行动作段一致，因此比原逻辑更合理。

## 8. 修改后的验证结果

已做的验证：

- 编辑器诊断无报错
- `py_compile` 语法编译通过

结论：

- 此次修改语法正确
- 逻辑上不会影响主训练过程
- 能修正 CDP 日志误差窗口与执行窗口不一致的问题

## 9. 最终结论

### 对仓库比较的最终结论

- `Causal-Diffusion-Policy` 与 `3D-Diffusion-Policy` 同源
- 但差异不止算法，还包括训练、配置、数据字段和评测逻辑

### 对误差问题的最终结论

- CDP 原先存在一个固定 `16:20` 的日志误差窗口问题
- 该问题不影响主训练损失和正常 rollout
- 但会误导日志解读，并在改配置时可能带来隐患

### 当前状态

- 该问题已经在 `Causal-Diffusion-Policy/train.py` 中修正为动态窗口逻辑
- 现在日志误差与实际执行 chunk 对齐，更适合继续训练与分析
