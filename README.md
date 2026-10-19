# Qmini Isaac Lab 工作区

本项目把 Qmini、RoboTamer4Qmini 和 RoboTamerSdk4Qmini 中与训练及部署有关的内容，整理为可用于 **Isaac Sim 6.0.1 + Isaac Lab 3.0** 的外部 Isaac Lab 项目。项目先在 Windows 上生成，但运行目标是 Ubuntu 24.04；复制整个 `qmini_ws` 文件夹即可。

> 当前状态：项目结构、机器人资源、任务、RSL-RL 配置、训练/回放脚本及 ONNX 部署工具已完成，并通过 Python 编译和静态一致性测试。由于当前 Windows 主机没有可运行的 Isaac Sim 6.0.1 + Isaac Lab 3.0 GPU 环境，首次仿真和训练仍需在 Ubuntu 上完成。

项目同时提供 Direct 和 Manager-Based 两种任务实现。Manager-Based 的任务 ID、代码结构、Ubuntu 更新及训练方法见 [MANAGER_BASED.md](MANAGER_BASED.md)。

## 为什么没有执行 `./isaaclab.sh --new`

Isaac Lab 的 `--new` 是交互式项目生成器，适合在已经安装好的 Ubuntu Isaac Lab 源码目录中创建模板。本工作区需要先在 Windows 生成、再整体迁移到 Ubuntu，因此这里按官方 external project 模板直接生成了等价的独立项目：

- 不需要把 `qmini_ws` 放进 IsaacLab 仓库；
- IsaacLab 可以升级或重新克隆而不影响本项目；
- 在 Ubuntu 中使用 editable install 注册任务；
- 脚本通过 `ISAACLAB_ROOT` 调用版本匹配的官方训练/回放入口。

如果以后希望用生成器创建另一个空模板，可以在 Ubuntu 的 IsaacLab 目录执行 `./isaaclab.sh --new`，但不应覆盖本项目。

## 已转换的内容

| 原项目内容 | 本项目实现 |
|---|---|
| Isaac Gym `LeggedRobot` 任务 | Isaac Lab `DirectRLEnv` 与 `ManagerBasedRLEnv` |
| Qmini URDF/STL | `UrdfFileCfg` 首次运行时导入 USD |
| 原 PPO 配置 | RSL-RL PPO 配置 |
| 10 个关节的位置增量控制 | 12 维动作：2 个步频 + 10 个关节增量 |
| BIRL 43 维单帧观测 | 保留 43 维契约并堆叠 3 帧，共 129 维 |
| 摩擦、质量、增益、推力随机化 | Isaac Lab event/domain randomization |
| SDK ONNX 策略 | 导出、动作限幅、检查和 SDK 安装脚本 |

保留的控制周期为 `0.015 s`（物理步长 `0.001 s`、decimation 15）。动作和观测契约见 [policy_contract.json](scripts/deploy/policy_contract.json)。

## 项目结构

```text
qmini_ws/
├── source/qmini_lab/              # 可安装的 Isaac Lab 扩展
│   └── qmini_lab/
│       ├── assets/                # Qmini URDF、STL 和机器人配置
│       └── tasks/
│           ├── direct/qmini_velocity/
│           └── manager_based/qmini_velocity/
├── scripts/
│   ├── rsl_rl/                    # 训练和回放入口
│   ├── deploy/                    # ONNX/SDK 工具
│   ├── check_install.py
│   └── zero_agent.py / random_agent.py
├── tests/                         # 无需 Isaac Sim 的静态测试
├── requirements.txt               # 项目额外依赖
├── requestment.txt                # 兼容用户要求的文件名
└── environment.yml                # 可选 env_qmini 环境
```

## 1. Ubuntu 24.04 环境准备

目标版本：

- Ubuntu 24.04；
- Python 3.12；
- Isaac Sim `6.0.1.0`（pip）；
- Isaac Lab `release/3.0.0-beta2`；
- NVIDIA GPU、驱动及显存应满足 Isaac Sim 6.0.1 要求。

### 复用现有 `env_isaaclab`

先检查已有环境，不要无意义地创建第二个环境：

```bash
conda activate env_isaaclab
python --version
python -m pip show isaacsim isaaclab rsl-rl-lib
```

只有当 Python 是 3.12，且 Isaac Sim/Isaac Lab 版本与上面目标一致时才继续复用。若已有环境是旧版 Isaac Sim 或 Isaac Lab，不建议在原环境内直接覆盖升级。

### 创建隔离的 `env_qmini`（推荐用于版本不匹配时）

把本文件夹复制到 Ubuntu，例如 `~/qmini_ws`，然后执行：

```bash
cd ~/qmini_ws
conda env create -f environment.yml
conda activate env_qmini
python -m pip install --upgrade pip
```

安装 Isaac Sim 6.0.1：

```bash
python -m pip install --pre "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com
```

安装官方为 x86_64 平台指定的 CUDA PyTorch：

```bash
python -m pip install -U torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

接受 NVIDIA EULA 后验证：

```bash
python -c "import isaacsim; print('Isaac Sim import OK')"
```

安装官方版本匹配的 Isaac Lab：

```bash
sudo apt update
sudo apt install -y cmake build-essential
cd ~
git clone --branch release/3.0.0-beta2 \
  https://github.com/isaac-sim/IsaacLab.git
cd ~/IsaacLab
./isaaclab.sh --install rsl_rl
```

CUDA PyTorch 已在前一步按官方版本显式安装；`isaaclab.sh --install rsl_rl` 会安装 Isaac Lab 子模块和 RSL-RL。不要把旧 RoboTamer 仓库中的 Python 3.8、旧 PyTorch 或 Isaac Gym 依赖装进该环境。

## 2. 安装本项目

```bash
conda activate env_qmini       # 或 env_isaaclab
export ISAACLAB_ROOT="$HOME/IsaacLab"
cd ~/qmini_ws
python -m pip install -r requirements.txt
python -m pip install -e source/qmini_lab
python scripts/check_install.py
python scripts/list_envs.py
```

`check_install.py` 应显示 Python 3.12、Isaac Sim、Isaac Lab、RSL-RL、qmini-lab，并列出 8 个任务：

- `Qmini-Velocity-Flat-Direct-v0`
- `Qmini-Velocity-Flat-Direct-Play-v0`
- `Qmini-Velocity-Rough-Direct-v0`
- `Qmini-Velocity-Rough-Direct-Play-v0`
- `Qmini-Velocity-Flat-Manager-v0`
- `Qmini-Velocity-Flat-Manager-Play-v0`
- `Qmini-Velocity-Rough-Manager-v0`
- `Qmini-Velocity-Rough-Manager-Play-v0`

每次新开终端都要激活 Conda 环境并设置 `ISAACLAB_ROOT`。也可以把 export 写进该 Conda 环境的 activate 脚本。

## 3. 首次仿真检查

先用单环境零动作检查 URDF、关节方向和接触体：

```bash
cd ~/qmini_ws
python scripts/zero_agent.py \
  --task Qmini-Velocity-Flat-Direct-Play-v0 \
  --num_envs 1 --visualizer kit
```

然后检查随机动作：

```bash
python scripts/random_agent.py \
  --task Qmini-Velocity-Flat-Direct-Play-v0 \
  --num_envs 4 --visualizer kit
```

首次启动会把 URDF/STL 导入 USD，时间会比之后长。确认机器人朝向、左右腿、膝盖弯曲方向和足部接触正常后再开始大规模训练。若当前 Isaac Lab 小版本的命令行参数有变化，请运行相应脚本的 `--help`；Isaac Lab 3.0 使用 `--visualizer kit` 显示窗口，默认运行可不加该参数。

## 4. 训练

先用较少并行环境验证训练能稳定推进：

```bash
python scripts/rsl_rl/train.py \
  --task Qmini-Velocity-Flat-Direct-v0 \
  --num_envs 512
```

验证通过并确认显存足够后，可提高到配置默认的 4096 个环境：

```bash
python scripts/rsl_rl/train.py \
  --task Qmini-Velocity-Flat-Direct-v0 \
  --num_envs 4096
```

粗糙地形训练：

```bash
python scripts/rsl_rl/train.py \
  --task Qmini-Velocity-Rough-Direct-v0 \
  --num_envs 4096
```

常用选项可通过下面命令查看：

```bash
python scripts/rsl_rl/train.py --help
```

日志和 checkpoint 使用官方 RSL-RL 脚本的默认位置，通常位于 `logs/rsl_rl/<experiment_name>/...`。需要恢复训练时，使用 `--resume` 以及 `--load_run`/`--checkpoint`（以 `--help` 中该版本参数为准）。

## 5. 回放及导出 ONNX

```bash
python scripts/rsl_rl/play.py \
  --task Qmini-Velocity-Flat-Direct-Play-v0 \
  --checkpoint /absolute/path/to/model_XXXX.pt \
  --num_envs 1 --visualizer kit
```

官方 RSL-RL play 脚本会在 checkpoint 对应日志目录的 `exported/` 下导出 JIT 和 ONNX 策略。请先在仿真中确认导出策略正常。

新的 RSL-RL actor 均值输出不保证带最终 `tanh`，仿真包装器会将动作限制到 `[-1, 1]`，而原 Qmini SDK 直接使用 ONNX 输出。因此，部署前必须生成显式带 `Clip` 的模型：

```bash
python scripts/deploy/clip_onnx_actions.py \
  /path/to/exported/policy.onnx \
  /path/to/exported/qmini_policy_bounded.onnx

python scripts/deploy/validate_onnx.py \
  /path/to/exported/qmini_policy_bounded.onnx
```

检查器会验证输入 129 维、输出 12 维、有限数值以及动作范围。

## 6. 安装策略到 RoboTamerSdk4Qmini

在 Ubuntu 中先按原 SDK README 编译 SDK，再执行：

```bash
python scripts/deploy/install_policy_to_sdk.py \
  /path/to/exported/policy.onnx \
  /path/to/RoboTamerSdk4Qmini
```

脚本会：

1. 在 SDK 的 `bin/policy.onnx` 写入带 `[-1, 1]` 限幅的模型；
2. 验证 ONNX 输入/输出契约；
3. 把匹配本训练任务的 `config.yaml` 写入 SDK `bin/`。

此操作会覆盖 SDK `bin/policy.onnx` 和 `bin/config.yaml`，需要保留旧文件时请先备份。

> 真机安全：第一次测试应架空机器人或使用保护架，准备急停，限制速度，并逐项核对电机编号、零位、关节方向、限位和增益。仿真策略不能在未经这些检查时直接上机。

## 策略接口

关节顺序固定为：

```text
hip_yaw_l, hip_roll_l, hip_pitch_l, knee_pitch_l, ankle_pitch_l,
hip_yaw_r, hip_roll_r, hip_pitch_r, knee_pitch_r, ankle_pitch_r
```

动作：

- `action[0:2]`：左右腿步频，归一化 `[-1, 1]` 映射到 `[0.5, 3.5] Hz`；
- `action[2:12]`：10 个关节的目标位置增量速率，范围 `[-15, 15] rad/s`，每 0.015 s 积分一次并裁剪到 URDF 限位。

每帧 43 维观测依次为：速度命令/偏航命令、roll/pitch、基座角速度、关节位置差、关节速度、当前关节目标、左右相位 sin/cos、左右步频；按时间堆叠 3 帧后输入 ONNX，共 129 维。精确定义以 [policy_contract.json](scripts/deploy/policy_contract.json) 和环境源码为准。

## 主要可调文件

- 机器人与执行器参数：[qmini.py](source/qmini_lab/qmini_lab/assets/qmini.py)
- 环境、随机化、奖励与命令范围：[qmini_velocity_env_cfg.py](source/qmini_lab/qmini_lab/tasks/direct/qmini_velocity/qmini_velocity_env_cfg.py)
- 观测、动作、奖励和 reset 实现：[qmini_velocity_env.py](source/qmini_lab/qmini_lab/tasks/direct/qmini_velocity/qmini_velocity_env.py)
- PPO 参数：[rsl_rl_ppo_cfg.py](source/qmini_lab/qmini_lab/tasks/direct/qmini_velocity/agents/rsl_rl_ppo_cfg.py)
- SDK 部署配置：[qmini_sdk_config.yaml](scripts/deploy/qmini_sdk_config.yaml)

修改观测或动作定义时，必须同步修改环境、`policy_contract.json`、ONNX 验证器和 SDK C++ 端；否则维度相同也可能产生危险的语义错位。

## 静态测试

不启动 Isaac Sim 时可运行：

```bash
python -m unittest discover -s tests -v
python -m compileall -q source scripts tests
```

静态测试会检查策略维度、URDF 关节顺序及限位、所有 STL 文件和基座碰撞体。它不能替代 Isaac Sim 内的物理、渲染和训练测试。

## 常见问题

- 找不到 Isaac Lab 脚本：确认 `export ISAACLAB_ROOT="$HOME/IsaacLab"` 指向源码 checkout，而不是 Python site-packages。
- 找不到 Qmini task：重新执行 `python -m pip install -e source/qmini_lab`，并确认当前终端使用正确的 Conda Python。
- CUDA/torch 依赖冲突：不要安装旧仓库的 requirements；新环境中让 `isaaclab.sh --install rsl_rl` 管理核心版本。
- 显存不足：先把 `--num_envs` 降到 512、128 或更低；粗糙地形比平地占用更多显存。
- Ubuntu 路径报错：Linux 区分文件名大小写，请原样复制整个 `qmini_ws`，不要单独重命名 mesh。
- URDF 首次导入慢：属于正常现象；后续会使用生成的缓存。

## 许可证和来源

Python 项目代码使用 MIT License。Qmini URDF/STL 资源来自 Unitree Qmini 仓库，遵循 **CC BY-NC-SA 4.0**，包含非商业使用限制；RoboTamer 两个仓库的相关代码分别保留其 MIT 许可。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 `third_party_licenses/`。

参考：

- [unitreerobotics/Qmini](https://github.com/unitreerobotics/Qmini)
- [vsislab/RoboTamer4Qmini](https://github.com/vsislab/RoboTamer4Qmini)
- [vsislab/RoboTamerSdk4Qmini](https://github.com/vsislab/RoboTamerSdk4Qmini)
- [Isaac Lab 3.0 安装文档](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/index.html)
- [Isaac Sim pip 安装文档](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html)
# qmini_ws
# qmini_ws
# qmini_ws
