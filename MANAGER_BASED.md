# Qmini Manager-Based 任务

项目保留原有 Direct 任务，并新增一套独立的 Isaac Lab `ManagerBasedRLEnv` 实现。两套任务共享 Qmini URDF、执行器参数和 RSL-RL 网络结构，策略接口保持一致：

- 动作：12 维（2 个步频动作 + 10 个关节增量动作）；
- 单帧观察：43 维；
- 历史：3 帧，最终策略输入 129 维；
- 物理步长：0.001 s；
- 控制步长：0.015 s（decimation 15）。

## 任务 ID

训练任务：

- `Qmini-Velocity-Flat-Manager-v0`
- `Qmini-Velocity-Rough-Manager-v0`

可视化/测试任务：

- `Qmini-Velocity-Flat-Manager-Play-v0`
- `Qmini-Velocity-Rough-Manager-Play-v0`

## Ubuntu 更新与检查

复制新增文件后，在 Ubuntu 执行：

```bash
conda activate env_isaaclab
cd ~/WS/qmini_ws
python -m pip install -e source/qmini_lab
python scripts/list_envs.py
```

首先进行单环境零动作测试：

```bash
python scripts/zero_agent.py \
  --task Qmini-Velocity-Flat-Manager-Play-v0 \
  --num_envs 1 --visualizer kit
```

然后进行随机动作测试：

```bash
python scripts/random_agent.py \
  --task Qmini-Velocity-Flat-Manager-Play-v0 \
  --num_envs 4 --visualizer kit
```

启动时应出现类似输出：

```text
[INFO]: Enabled Qmini contact reporting on 11 source bodies before Manager-Based scene cloning.
```

如果不是 11 个刚体，请先停止训练并检查 URDF 是否完整。

## 训练

先进行 20 次迭代的短测试：

```bash
python scripts/rsl_rl/train.py \
  --task Qmini-Velocity-Flat-Manager-v0 \
  --num_envs 512 \
  --max_iterations 20
```

确认 episode length、奖励和终止统计正常后进行正式训练：

```bash
python scripts/rsl_rl/train.py \
  --task Qmini-Velocity-Flat-Manager-v0 \
  --num_envs 4096
```

粗糙地形：

```bash
python scripts/rsl_rl/train.py \
  --task Qmini-Velocity-Rough-Manager-v0 \
  --num_envs 4096
```

Manager-Based 日志分别写入：

```text
logs/rsl_rl/qmini_velocity_flat_manager/
logs/rsl_rl/qmini_velocity_rough_manager/
```

## 代码结构

```text
tasks/manager_based/qmini_velocity/
├── __init__.py                    # Gym 任务注册
├── qmini_velocity_env_cfg.py      # Scene 和所有 Manager 配置
├── agents/
│   └── rsl_rl_ppo_cfg.py          # Manager 任务的 PPO 配置
└── mdp/
    ├── actions.py                 # 12 维动作和关节目标积分
    ├── commands.py                # 速度指令、站立比例和 deadband
    ├── observations.py            # 43×3 策略观察
    ├── rewards.py                 # 奖励项
    └── terminations.py            # 高度和倾角终止项
```

Qmini 的 URDF 刚体层级比较特殊，通用接触传感器初始化只会识别 `base_link`。Manager 版本在 articulation 的 `_post_spawn()` 阶段、场景克隆之前给全部 11 个刚体添加接触报告 schema，使克隆后的所有环境都能识别脚部、髋部和膝部接触。

