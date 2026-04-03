# Adroit Pen: `cdp3` / `dp3` / `3D-DP3` 成功率、训练流程与训练中评测对比

## 最核心更新

- **[共同底层信号]** 官方 `Adroit` 和 `DP3 / CDP3` 都依赖同一个 step-level `goal_achieved`，来源是 `mj_envs` 的 Adroit 环境。
- **[官方指标]** 官方 `Adroit` 用整条 episode 的 `goal_achieved` 做聚合。对 `pen` 任务，官方 episode success 是 `sum_t goal_achieved_t > 20`，然后对多条 episode (`paths`) 计算 `success_percentage`。
- **[DP3 / Causal 当前日志指标]** 训练时 `runner` 记录的 `test_mean_score` 不是这个官方 episode-level success，而是 `MultiStepWrapper` 返回的**末尾窗口均值**。
- **[Causal vs 3D]** `Causal` 的 `cdp3 / dp3` 看末尾 `32` 个 primitive steps，`3D-Diffusion-Policy` 的 `dp3` 看末尾 `2` 个 primitive steps。
- **[真正差异]** 差异不止是窗口长度，还包括：
  - **[时间范围]** 整局 episode vs episode 末尾尾窗
  - **[步数数量]** 全部有效步 vs `32 / 2`
  - **[聚合方式]** episode-level `0/1 success` vs tail-window mean
- **[结论]** `Causal` 内部 `cdp3` 和 `dp3` 的 `test_mean_score` 可比；`Causal` 与 `3D-Diffusion-Policy` 的 `test_mean_score` **不能当同一个成功率直接横比**。

## 对比对象

- **[Causal / cdp3]** `conda run --no-capture-output -n cdp bash scripts/train_policy.sh cdp3 adroit_pen 0403 0 4`
- **[Causal / dp3]** `conda run --no-capture-output -n cdp bash scripts/train_policy.sh dp3 adroit_pen 0403 0 1`
- **[3D-Diffusion-Policy / dp3]** `bash scripts/train_policy.sh dp3 adroit_pen 0112 0 0`

## 1. 官方 Adroit 指标 vs DP3 当前指标

### 1.1 官方 Adroit

- **[step-level]** `mj_envs/.../pen_v0.py` 当前代码把 `goal_achieved` 定义为 `orien_similarity > 0.95`。
- **[episode-level]** `evaluate_success(paths)` 对整条轨迹做判定：`pen` 任务满足 `sum(path['env_infos']['goal_achieved']) > 20` 即 success。
- **[批量评测]** `paths` 不是一个 episode 内部的“很多 path”，而是“很多条 episode / trajectory 的列表”；因此 `success_percentage = num_success / num_paths * 100` 是跨 episode 的成功率。
- **[参考实现]** `third_party/VRL3/src/train_adroit.py` 使用同样的阈值：`pen=20`，其他 Adroit 任务 `25`。

### 1.2 DP3 / CDP3 训练时日志

- **[数据流]** Adroit env -> `rrl_multicam.py` -> `MujocoPointcloudWrapperAdroit` -> `MultiStepWrapper` -> `AdroitRunner`
- **[关键行为]** `MultiStepWrapper.step()` 返回 `dict_take_last_n(self.info, self.n_obs_steps)`，因此 `info['goal_achieved']` 只是一段最近窗口。
- **[Causal / cdp3, dp3]** `n_obs_steps=32`，`test_mean_score` 近似 `mean_e mean(last_32_goal_flags)`
- **[3D / dp3]** `n_obs_steps=2`，`test_mean_score` 近似 `mean_e mean(last_2_goal_flags)`
- **[含义]** 它不是官方 episode success，而是“episode 末尾窗口分数”。

## 2. 三个实验的 success 语义对比

| 指标 | 官方 Adroit (`pen`) | Causal / cdp3 | Causal / dp3 | 3D / dp3 |
|---|---|---|---|---|
| 单步信号 | `goal_achieved` | 同左 | 同左 | 同左 |
| 时间范围 | 整条 episode | 末尾 32 步 | 末尾 32 步 | 末尾 2 步 |
| 单 episode 聚合 | `1[sum > 20]` | `mean(last_32)` | `mean(last_32)` | `mean(last_2)` |
| 多 episode 聚合 | 成功 episode 比例 | `mean` over episodes | `mean` over episodes | `mean` over episodes |
| 能否与官方 success 直接等同 | 否 | 否 | 否 | 否 |
| 与谁可直接比较 | 官方内部 | Causal / dp3 | Causal / cdp3 | 仅同配置 3D dp3 |

- **[硬结论]** `Causal / cdp3` 与 `Causal / dp3` 的 `test_mean_score` 可直接比；`Causal` 与 `3D` 的 `test_mean_score` 不可直接横比。
- **[补充]** `mean_n_goal_achieved` 也不干净，因为 `MultiStepWrapper` 是滑动窗口，相邻 policy step 会重复覆盖同一批 primitive steps。

## 3. 训练与评测框架差异

| 项目 | Causal / cdp3 | Causal / dp3 | 3D / dp3 |
|---|---|---|---|
| 主干模型 | `CausalTransformer` | `ConditionalUnet1D` | `ConditionalUnet1D` |
| `horizon` | 40 | 40 | 16 |
| `n_obs_steps` | 32 | 32 | 2 |
| `n_action_steps` | 8 | 8 | 8 |
| `rollout_every` | 50 | 50 | 200 |
| `validation` | 开启 | 开启 | 关闭 |
| `checkpoint.topk.k` | 5 | 5 | 1 |
| `resume` 语义 | 恢复后只跑剩余 epoch | 恢复后只跑剩余 epoch | 恢复后仍可能再跑完整 `num_epochs` |
| GPU 语义 | 直接 `training.device=cuda:id` | 同左 | `CUDA_VISIBLE_DEVICES + cuda:0` |

- **[Causal 内部差异]** 主要是模型、scheduler、优化器、batch size、推理步数不同；训练循环和评测框架相同。
- **[Causal vs 3D 差异]** 除 `n_obs_steps / horizon` 外，还包括 validation 开关、rollout 频率、wandb 记录粒度、checkpoint 策略、resume 语义、GPU 可见性处理。
- **[数据管线]** `Causal` 的 `adroit_pen` 额外包含 `image`，但 `dp3 / cdp3` 的核心输入仍是 `point_cloud + agent_pos`，这不是成功率定义差异的主因。

## 4. 如果要做公平对比

- **[`n_obs_steps`]** 必须对齐，否则 success 指标语义先天不同。
- **[`horizon`]** 最好对齐，否则 rollout 方式不同。
- **[`rollout_every`]** 最好对齐，否则训练过程中的“最佳点”搜索机会不同。
- **[`resume`]** 必须对齐，否则训练总时长不一致。
- **[GPU 语义]** 最好统一为 `CUDA_VISIBLE_DEVICES + training.device=cuda:0`，避免裸 `.cuda()` 造成设备漂移。
- **[最靠谱做法]** 重写 `AdroitRunner`，显式累计整条 episode 的 `goal_achieved`，再按 `mj_envs.evaluate_success` 的官方阈值计算 episode success。
- **[别自欺欺人]** 如果继续沿用现在的 `test_mean_score`，那它只能叫“tail-window success score”，别叫官方 Adroit success rate。

## 5. 关键依据文件

- **[官方 Adroit 指标]** `3D-Diffusion-Policy/third_party/rrl-dependencies/mj_envs/mj_envs/hand_manipulation_suite/pen_v0.py`
- **[官方阈值参考实现]** `3D-Diffusion-Policy/third_party/VRL3/src/train_adroit.py`
- **[Causal train]** `Causal-Diffusion-Policy/Causal-Diffusion-Policy/train.py`
- **[3D train]** `3D-Diffusion-Policy/3D-Diffusion-Policy/train.py`
- **[Causal runner]** `Causal-Diffusion-Policy/Causal-Diffusion-Policy/diffusion_policy/env_runner/adroit_runner.py`
- **[3D runner]** `3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/adroit_runner.py`
- **[Causal MultiStepWrapper]** `Causal-Diffusion-Policy/Causal-Diffusion-Policy/diffusion_policy/gym_util/multistep_wrapper.py`
- **[3D MultiStepWrapper]** `3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/gym_util/multistep_wrapper.py`
- **[Causal task config]** `Causal-Diffusion-Policy/Causal-Diffusion-Policy/diffusion_policy/config/task/adroit_pen.yaml`
- **[3D task config]** `3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task/adroit_pen.yaml`
