# Movens-sim2real 开发进度

更新：2026-11-18 02:14（本节时间为服务器本地 UTC+08:00）。本轮仿真、回训和离线验证结束；真机未启动。

| 阶段 | 状态 | 验证层级与证据 |
|---|---|---|
| M0 本机接口 | PASS | `PROJECT_PLAN.md` §7：报告指纹，真实数据、原入口、编译后 43 关节/actuator、位置伺服、80 个物理步/命令；源码及上游现有未提交内容保留 |
| M1 数据/打标 | PASS | 357 Apple 原始回放、12 个定向真积分补录、1,000 SONIC；`runs/m1/verification.json` 通过时间/命令配对、同源、因果、reset 与组合检查 |
| M2 双模型预训练 | PASS | `runs/m2_formal`，正式 440 次，两模型权重、优化器、统计和恢复经过验证，完整留出评估；动力学劣于部分简单基线 |
| M3 仿真执行 | PASS | 两条完整零残差评估、ep1 全程残差执行和物理断点恢复、同渲染设置下 D0 预测开关逐位相同 |
| M4 S0 闭环学习 | PASS | 两组各采集 1,721 步、更新 244 次；固定留出原 Movens 1/2、随机残差 1/2、预训练残差 0/2，未见收益；两臂退出 0，恢复与 D0 旁路检查通过 |
| M5 S1 部署适配 | PASS | 3,428 周期、480 更新，训练 1/4；四条件 S0→S1：干净 0→1、物理 2→2、噪声 0→1、部署 1→1（每组两条）；90 步部署恢复逐位通过 |
| M6 导出与回训 | PASS | 首轮两条、累计轮三条连续训练成功，原接口、回训、固定留出闭环通过；独立/加同一 S0 残差：原权重 1/2、0/2，首轮 2/2、1/2，累计轮 1/2、2/2 |
| M7 Agent/真机软件 | PASS（离线） | S1 actor/双 Q/D0 与原真机归档、厂商 SDK 报文、USDA 任务缓存和注册深度适配均已验证；真机 RGB-D、SAM 3 权重、现场标定和执行链仍待现场条件 |
| M8 R0 真机校准 | BLOCKED | 缺明确当次真机执行条件、设备与可信实测日志 |
| M9 R1 真机适应 | BLOCKED | 依赖 M8 |
| X1 S2 模型辅助 | NOT_STARTED | 可选实验；D0 不接入动作选择 |

M0 现场：共享盘 1017 GiB 可用，2× RTX 5090 无现有 CUDA 任务，screen 可用、tmux 未装。旧 80k 评估队列和相邻续训记录存在，未操作；不以它们的历史 PID 判定当前状态。原 Apple sparse 数据是实测原视频时钟，10 Hz 精确交集；4000 Hz 仅有苹果及接触日志，没有 50 Hz 机器人状态。原控制目标是 50 Hz 区间右端编号。SONIC 原档无 actuator/速度/物体真值，不能作为同控制链的动作转移。补录的左手组接触按该次 solver 关节和几何索引读取。

当前小样本命令（cwd 本项目）：
```bash
export PYTHONPATH=src:../Movens-Interact OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=../Movens-Interact/.local/train-env/bin/python
"$PY" -B -m movens_sim2real.data apple --out runs/m1_smoke --episodes 0 1 7 12 44 --workers 5
"$PY" -B -m movens_sim2real.data replay --out runs/m1_smoke/dense --episode 1 --seconds .2 --checkpoint-every 5
```

短试跑：原生 ep1 编译约 22.2 s，原链 800 步/0.2 s 实际积分约 7.76 s。首个打标读取发现早期逐步接触行只有 geom ID，无 body 名；改用实际 XML 编译映射并以历史 left-finger 摘要独立核对。原场景物体和盘面 collider 在多级子体，已按拓扑映射。两项修复重试中。暂未启动长训练；阶段结果以实际产物为准。

M1 实际试跑通过：ep1/7 识别完整抓放成功，ep0/12/44 未达到 v3 稳定抓握；五条指组日志与原逐物理步 left-finger 摘要一致。短补录 ep1 积分 10 步决策、80 个物理步/决策；3 个精确重合状态的关节最大差 0，苹果位置最大差 27.1 µm，补录为同源独立执行。全量历史打标/定向补录的最终结果另记。

M1 已启动有上限批处理（2026-11-17 23:00:55 服务器本地时间）：
- `981766.movens-v3-m1-apple`，timeout PID 981773，12 CPU workers，`runs/m1/apple.log`，357 完整历史运行和 45 排除，逐 episode 原子保存 `runs/m1/sparse/`。
- `981770.movens-v3-m1-sonic`，timeout PID 981777，CPU；已退出 0，1,000 条、0 排除，`runs/m1/sonic_summary.json`，每条按 XML 名称映射，并仅取真实时间刻。
- `981775.movens-v3-m1-dense`，timeout PID 981785，最多 8 CPU workers；12 个确定的源 ID（train 1/2/4/8/12/33，val 19/20/44，test 7/16/23），日志 `runs/m1/dense.log`，断点 `runs/m1/dense/apple_XXXXXX/resume.npz` 每 100 控制步。训练样本不跨 split。每项 55 分钟上限，无 GPU。

实际启动/恢复：先查 `screen -ls` 和对应 worker/exit；已退出且需要续跑时，从本项目执行：
```bash
mode=dense  # apple / sonic / dense
screen -dmS "movens-v3-m1-$mode" bash -c 'timeout -k 30s 55m scripts/m1.sh "$1" >>"runs/m1/$1.log" 2>&1; rc=$?; printf "%s\n" "$rc" >"runs/m1/$1.exit"; exit "$rc"' _ "$mode"
```

23:20 恢复核验：Apple、SONIC 批处理均退出 0。`runs/m1/apple_summary.json`：402 条清单、357 条标注（288/37/32），45 条排除，134 条成功（106/14/14），216 条未稳定抓住、4 条未到可抓距离、3 条滑脱或盘外释放；历史数据仅 62 段偶然精确对齐的 20 ms 转移，其余保留实际命令序列。`runs/m1/sonic_summary.json` 有 1,000 条状态辅助记录，无排除。`runs/m1/dense.log` 仍推进；尚未启动 M2 正式训练。两个 RTX 5090 分别由本项目 PID 11623/11625 使用约 13 GiB，盘余 1017 GiB，无需改动相邻项目。

已运行的后续小样本及恢复入口（先核对 `screen -ls`、PID、`exit` 和产物；不要重复启动同目录）：
- M2 `runs/m2_smoke/{latest.pt,metrics.jsonl,evaluation.json,config.json}`，种子 42，16 次主监督、4 次 SONIC 辅助，batch 16。无动力学留出评估，正式任务另用 `runs/m1` 全量索引与固定补录。
- M3 `11613.movens-v3-m3-zero`，timeout 11619、Python 11623，32 分钟，测试 ep7/16，日志 `runs/m3_zero/run.log`，恢复用同一参数 `-m movens_sim2real.rollout --out runs/m3_zero --episode 7 16 --zero --side`；GPU/EGL 0。原 21 步截断视频在 `runs/m3_probe/apple_000007/`。
- M4 `11616.movens-v3-m4-probe`，timeout 11624、Python 11625，9 分钟，日志 `runs/m4_probe/run.log`，恢复在原参数后加 `--resume`：`-m movens_sim2real.learn --pretrained runs/m2_smoke/latest.pt --out runs/m4_probe --train 1 --eval --steps 40 --max-updates 12 --warmup 10 --batch 16 --save-every 4`；GPU/EGL 1，种子 42。检查更新、转移、版本固定与截断；这个结果不当作 S0 效果。

M1 结束：`dense.exit=0`，按 train/val/test 真正重新积分 4,956 / 2,152 / 1,990 条；历史偶然对齐另有 54 / 4 / 4 条。所有帧逐条核验与源实际命令对应，每 20 ms 为 80 个原生物理步。用 `PYTHONPATH=src:../Movens-Interact "$PY" -B scripts/check_data.py runs/m1` 复验；覆盖抓放成功、未抓住、滑脱或盘外释放、未达抓握位置，前缀不变、reset 拒绝和合成误差至多 2.22e-16 rad。密集补录是新物理执行，原运行仅用于配对校验，非插值。

M4 链路修复：原 probe 采集成功 40 步但训练退出 1；嵌套当前/下一观测不能作为对象数组传入 Torch。修复后使用原日志完成真实 SAC+D0 更新；另在 `runs/m4_resume_check` 重新采集 40 步，screen 14166 / timeout 14172 / Python 14173 已在第 4 次更新退出 0，待 `--resume` 到 12 次。恢复使用上述小样本参数，将 `--out` 改为 `runs/m4_resume_check`；种子 42、GPU/EGL 1、8 分钟；原错误产物保留。部分验证尾段被截断时，未完整的最后策略回报不进入 SAC，该段已执行转移仍进入 D0。

M2 正式阶段开始：选定种子 42、batch 64、40 次 SONIC 历史辅助启动 + 400 次联合批更新（两模型不共享参数；训练仍混合 SONIC 辅助，动力学只用实测动作转移），保存每 50 次，两种名义及全零对照均训练；按固定留出全量评估。小样本主更新每次约 0.24–0.40 s，正式任务保留 35 分钟预算含样本构建、三成员多步评估与一次恢复验证；不扩大搜索。首次第 100 次保存退出，随后继续到 440。

正式 M2 已完成，screen 15845（恢复段 timeout 16242 / Python 16243）、退出码 0。命令：`CUDA_VISIBLE_DEVICES=1 "$PY" -B -m movens_sim2real.pretrain --data runs/m1 --out runs/m2_formal --steps 400 --aux-steps 40 --seed 42 --batch 64 --ablations --save-every 50`，首次带 `--stop-after 100`，第二次 `--resume`。权重/恢复 `runs/m2_formal/latest.pt`，证据 `config.json`、`metrics.jsonl`、`evaluation.json`。train/val/test 真实动力学转移 5010/2156/1994；test 残差归一化 MAE 0.1840 vs 仅名义 0.2237、零修正 0.3838，正确名义仍有非零扰动。动力学 test 20 ms 位置误差 p95 2.96 mm、关节 0.0136 rad，对照保持 3.13 mm、0.0030 rad；不能认为整体优于保持/匀速。按 v3 保持 D0。

M4 试跑恢复成功：screen 15848 已退出 0，从第 4 次更新用同一真实采集恢复至 12 次（40 个实际控制步，未重新采集），产物 `runs/m4_resume_check/{latest.pt,final.pt,evaluation.json,metrics.jsonl}`；没有完整抓放效果结论。

M3 冻结模型 ep7 完整成功，854 步，最后稳定 7.10 s，证据 `runs/m3_zero/apple_000007/{ego.mp4,side.mp4,summary.json,trajectory.npz}`，ep16 仍在运行。用 `-m movens_sim2real.export --audit-only <summary.json>` 核对实际命令重建、时间、分块和物理成功。ep7 的 804 个动作、161 个原生 RGB 观测通过原训练器 `window_arrays` 编码与解码：Body29 最大误差 2.38e-7 rad，Hand14 最大误差 7.45e-9 rad，`codec_review.json`；该轨迹属于 test，没有导出到训练。

M3 残差运行预算：预训练正式 checkpoint、固定训练源初态 ep1（不计入留出结果）、种子 42、完整 940 步、双视角、D0 单独预测。使用原零残差实际计时约 0.7–0.9 s/控制步，18 分钟含两次加载；首次约 115 秒受控中断并保存，再从同一固定初态的整合状态恢复。另用相同控制配置的 21 步无预测运行检查 D0 等价。

M3 验收：零残差 screen 11613 退出 0，ep7 成功，ep16 稳定 0.92 s 按固定 1.0 s 标准判失败；合计 1/2 成功，未调整成功条件。M3 残差 screen `17450.movens-v3-m3-residual`，首次 timeout 17453，受控退出 124 在 tick 69 保存了积分状态、历史、动作块、阶段与 RNG。恢复段 timeout 18025 / Python 18026，`runs/m3_residual/{run.log,pause.exit,exit}`，断点 `apple_000001/resume.pt`；恢复执行 `-m movens_sim2real.rollout --out runs/m3_residual --episode 1 --small runs/m2_formal/latest.pt --side --seed 42 --checkpoint-every 30`，GPU/EGL 1。匹配侧视角开关的无预测对照 screen 19820 退出 0：前 21 个实际命令、状态和 RGB 逐位相等（跨一次动作块刷新），`runs/m3_residual/d0_equivalence.json`。第一次对照漏了 `--side`，导致 ego RGB 在渲染后续帧时出现差异，保留在 `runs/m3_no_d0`，不用于结论。正式各组评估统一双视角渲染。

M4 正式预算：seed 42；每臂同一固定训练源 ep1/8，1,721 原生 20 ms 控制步（包含验证尾段），全程不重置拼接；最多 256 次更新，暖启动 500 个实际决策、采样比 .15、batch 64。两个学习臂除残差初始化外一致，动力学均从正式 M2 开始且旁路。后续固定 ep7/16 评估，统一双视角、seed 42，与 M3 冻结基准比较；每臂约 50 分钟，硬上限 55 分钟。报告 200k 控制步每种子在本机原生物理耗时约 40–50 小时；此次为一个预先限定种子的研究轮次，独立留出不更换，不自动扩成调参或宣称统计显著。

M4 启动 23:44:24：预训练臂 `20745.movens-v3-m4-pretrained` / timeout 20755 / Python 20757（GPU/EGL 0）；随机初始化臂 `20747.movens-v3-m4-cold` / timeout 20753 / Python 20756（GPU/EGL 1）。各 `runs/m4_s0_{pretrained,cold}/{run.log,latest.pt,exit}`，完成训练保存 `final.pt`，固定评估随后写 `evaluation.json`；采集与评估有独立 `resume.pt`。实际命令 `timeout -k 90s 55m bash scripts/s0.sh pretrained 0`、`timeout -k 90s 55m bash scripts/s0.sh cold 1`；确认原进程退出后给同一脚本追加 `--resume`，不改模型、数据、渲染开关或预算。

M5 独立短联调 23:45:10：`21058.movens-v3-m5-probe` / timeout 21064 / Python 21065，GPU/EGL 0；命令 `"$PY" -B scripts/check_deployment.py --small runs/m2_formal/latest.pt --out runs/m5_probe --episode 1 --seconds 1.8`，90 个实际控制步、8 分钟上限。验收有可见/缺失窗口、部分点云、延迟、实际预测和 actor/critic 更新；旧 M2 权重仅用于这项集成检查，S1 正式训练仍等待准确输入 S0。

M5 短联调退出 0，`runs/m5_probe/verification.json`：90 个实积分控制步，观测边界 72 可见 / 19 缺失，可见点云 256 个可见表面采样点，缓存最大年龄约 80 ms，估计位置与真值最大差约 11.0 mm。未知物理字段掩码生效；SAC actor、双 Q、D0 成员均利用同一份实际记录完成有限梯度更新。此 seed 的显式机器人状态延迟是 0 tick，正式 S1 预选 ep2/4 的部署扰动各为 1 tick。

M3 补充残差完整运行退出 0，但 ep1 940 步未稳定抓住（不是工程失败），双视角和整集记录都在 `runs/m3_residual/apple_000001`。为避免 M6 等同于等待测试组成功，增加一次有限采集：已验真的训练来源 ep2（正式预训练残差+D0）、ep4（冻结基础模型零修正），各只执行一次、seed 42、保留完整起点/尾段且不用于学习臂效果比较。只有满足 v3 原成功门限的训练源可进入待回训清单，测试集不变。各 18 分钟上限，无自动重试或调参。

M7 已运行：`"$PY" -B -m movens_sim2real.real_replay --episode 1 --frames 90 --out runs/m7_apple_raw`。原始 30 Hz 真机 RGB 与 43 编码器位置最大时钟差 0.89 µs；二维颜色跟踪 90/90 有候选（无标注，不能当准确率），陈旧编码器和未来包拒绝；3D、力、速度和成功均显式无效。`observations.npz`、`summary.json`、`priors.json`、`tracked_rgb.png`。Unitree Body/Dex3 厂商实现存在于相邻 SONIC 上游，当前环境无 DDS Python 运行包、设备映射与时钟校准，无同步 RGB-D。FDA/USDA 本机网络代理连接失败；现用带来源的离线 USDA 参考份量卡，质量区间不是实测，没几何时不填惯性。已实现命名的 SDK 双向关节映射、SI 值/有效位/置信度、RGB-D 部分观测和离线工具；SAM 3 权重、真实 RGB-D 追踪、设备接口仍待对应资产条件。

23:56 接续核验：项目只有 `AGENTS.md`；v3 DOCX 指纹与计划一致。两块 RTX 5090 各约 26.5 GiB 已用（均为本项目），共享盘约 1017 GiB 可用，暂不再并发启动 GPU 作业。S0 两臂分别已过 630/720 控制步，均在写物理断点，尚未形成首条完整训练 episode。

补充采集启动于 23:49:45：`22492.movens-v3-capture-2` / timeout 22497 / Python 22502，GPU/EGL 0；`22495.movens-v3-capture-4` / timeout 22503 / Python 22504，GPU/EGL 1。退出码为 `runs/m4_capture_{2,4}/exit`，日志为同目录 `run.log`，物理断点为 `apple_000002/resume.pt` / `apple_000004/resume.pt`。两条均已过 360 个实际控制周期。若原任务中断且预算允许，从本项目沿用原命令与 `--out`（rollout 自动恢复已存在的物理断点）：`timeout -k 60s 18m env CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_EGL_DEVICE_ID=<gpu> MUJOCO_GL=egl PYTHONPATH=src:../Movens-Interact OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" -u -B -m movens_sim2real.rollout --out runs/m4_capture_<ep> --episode <ep> --side --seed 42 <mode>`，ep2 为 `--small runs/m2_formal/latest.pt`，ep4 为 `--zero`。保留已有日志，不并跑同一目录。

M7 几何补充联调：`MUJOCO_GL=egl "$PY" -B scripts/check_sensor_geometry.py runs/m3_no_d0_side/apple_000001 --out runs/m7_apple_raw/geometry_interface_check.json`。实际 MuJoCo 光学深度及分割 ID 产生 256 个部分点，初始中心误差 6.31 mm；单帧姿态与速度无效，惯性仅由质量先验和度量尺寸计算对角项。该项验证光学坐标与几何接口，真机证据仍是上面的单目日志。

00:05：S0 两臂均已完成首条 940 个真实周期并更新 134 次，首条都未稳定抓握，正在执行 ep8；GPU 资源与上游进程均未改动。补充采集 ep2/ep4 退出 0，分别完成 845/862 步，均未稳定抓握。M6 不能使用这些轨迹或已成功的 test ep7。仅再进行一批两条训练源的固定采集：从相邻旧 DWA 同一 checkpoint/同任务文本的 `results.csv` 中，取历史报告抓放成功且本项目 M1 原链也验收成功的 train 源，按 ID 升序只取前两条 ep15/32。旧评的这两条 `physics_valid=False`，旧记录不直接导出。两条均在本项目原链从规定初态全程重跑，冻结基础模型、零残差、D0、seed 42、双视角；每条硬上限 18 分钟。仅用于取得 M6 数据，和 S0 的学习效果比较分开；成功条件和固定留出 ep7/16 保持一致。该批与 S0 的训练源日志完成后统一筛选，无自动扩批。

M7 厂商 SDK 离线验证通过：仅在 `runs/m7_sdk/env` 使用 Python 3.12 和本机已有 cyclonedds 0.10.2 wheel；通过 `PYTHONPATH` 引用 SONIC 中 `external_dependencies/unitree_sdk2_python`，通过 `LD_LIBRARY_PATH` 引用其 `gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64`，不修改相邻环境。`runs/m7_sdk/env/bin/python -B scripts/check_unitree_packets.py runs/m7_apple_raw/observations.npz --out runs/m7_sdk/verification.json` 用真实归档位置和测试标定做三个反馈及三个命令 SDK 包 roundtrip，位置最大差 0 rad，CRC、过期和坏包拒绝通过，未打开 DDS 或发布。实际设备标定、确认执行与 RGB-D 同步仍未验证。

M6 定向采集启动于 00:05:34：`26998.movens-v3-m6-capture-15` / timeout 27005 / Python 27009（GPU/EGL 0）；`27002.movens-v3-m6-capture-32` / timeout 27010 / Python 27011（GPU/EGL 1）。日志 `runs/m6_capture_{15,32}/run.log`，退出 `exit`，各自 `apple_0000{15,32}/resume.pt` 每 90 步更新；完整预算 714/704 步。恢复用上面的 18 分钟同源采集命令，`--out runs/m6_capture_<ep> --episode <ep> --side --seed 42 --zero`，确认对应任务结束且剩余预算允许后再用。

00:17：ep15/32 均退出 0，714/704 个连续原生周期，独立成功门限不变；ep15 最终稳定 6.48 s。筛选本轮完整采集与 S0 两臂训练轨迹，CPU 导出，先做原 Qwen 缓存单 chunk 试跑再续完整缓存，沿用旧归一化与现有 Body/Hand flow、FK、物理损失。真机单目离线策略接线使用另一张 GPU 的剩余显存，25 个原源观测、最多 6 分钟；固定准确输入 S0 预训练臂 `final.pt` 仅用于验证 actor、双 Q、D0 的输入输出，不当作正式 S1 或真机控制。

M6 导出与原缓存完成：ep15/32 稳定 6.48/2.98 s；8 条真实候选只收 2 条，另 6 条均未稳定抓握。`runs/m6_export/{selection,index,verification}.json`：1,318 动作、264 RGB、133 个原训练窗口；Body29/Hand14 解码最大差 2.38e-7/7.45e-9 rad。Qwen 单 chunk 试跑 `30841.movens-v3-m6-cache-probe`，timeout 30847，32 帧用约 1.30 s、CUDA 峰值 8.94 GiB；`bash scripts/m6_cache.sh 0 --max-chunks 1`，8 分钟上限。续跑 `31203.movens-v3-m6-cache`，`bash scripts/m6_cache.sh 0 --resume --verify-cache-hashes`，6 分钟上限，`runs/m6_context/{run.log,exit,manifest.json}`，退出 0、完整 264 帧、约 120 MB。两次均复用原冻结缓存入口。

M6 训练短试跑开始：GPU 0 当前另有 S0 约 13.4 GiB，试跑限制 CUDA 分配比例 0.5；种子 42，3 次更新，batch 8（4 原模拟窗口 + 4 新成功窗口，microbatch 2），4 个固定旧验证窗口，10 分钟上限。只学习两套原专家模块；正式步数与固定闭环轮次根据这次实际耗时和资源另定。

M7 策略回放退出 0：`-m movens_sim2real.real_replay --episode 1 --frames 25 --small runs/m4_s0_pretrained/final.pt --out runs/m7_policy_offline`（GPU 1，6 分钟上限）。25 个原始单目/编码器观测跑过 actor、两个 Q 和 D0，3 个冻结名义块，prefix 为 0/10/10；首次推理 0.615 s，后两次约 0.240 s。该时延是离线共享 GPU 测量，真实部署需先热身并验证 0.4 s 换块和控制时钟。记录没有实际下发命令；R0 数据读取器已验证会拒绝该归档。SAM 3 已接本地权重加载及因果实例关联，依赖外部兼容环境/权重，未跑模型。

M6 回训短试跑完成：`31843.movens-v3-m6-refit-probe` / timeout 31849 / Python 31850，退出 0，`runs/m6_refit_probe/{metrics.jsonl,evaluation.json,checkpoint_last.pt}`。三次都读取 4 原 + 4 新窗口，原 Body、Hand 损失均产生有限且非零梯度；首步 0.96 s，随后约 0.38 s/步。正式轮固定种子 42、64 次更新、8 窗口（4 原 + 4 新）、microbatch 2，Body/Hand LR 1e-5/4e-5，旧 val 均匀检查 32 窗口，原冻结基准不变。不以 2 条新轨迹估计总体收益。第一次第 8 步保存、同预算续跑到 64（两个运行段各最多 8 分钟），之后固定 ep7/16 分别评估新基础权重独立及加 S0 残差，与已有旧基础闭环对照。

M6 正式回训结束：`32548.movens-v3-m6-refit` / timeout 32554 / Python 32555，8 步保存，`pause.exit=0`；`33095.movens-v3-m6-refit-resume` / timeout 33101 / Python 33102，同配置加 `--resume`，`exit=0`。命令 `-m movens_sim2real.refit --new-index runs/m6_export/index.json --new-context runs/m6_context --out runs/m6_refit --steps 64 --batch 8 --microbatch 2 --new-per-batch 4 --eval-batches 16 --save-every 16 --seed 42 --gpu-memory-fraction .5`，首段加 `--stop-after 8`。`runs/m6_refit/checkpoint_last.pt` step 80064；`weight_check.json` 证明 64 步不重复，Body 89/92、Hand 77/78 个张量更新。32 个留出旧窗口 Body 关节 RMSE 0.0281 rad、左手 B_hat MAE 0.0257 rad，仅为开环诊断。接下来两臂固定同套 ep7/16、同 seed 42、双视角，每臂 1,353 个真实控制步、35 分钟上限，复用 S0 预训练臂残差作新分布实测。

M4 有限轮验收结束：两臂 `exit=0`，各完整训练 1,721 步（34.42 s 仿真）、244 次更新，成功示范产率各 0/2。固定 ep7/16 各臂 1,353 次交互；基准仅 ep7 成功（7.10 s 稳定），随机残差仅 ep16 成功（2.64 s），预训练残差均失败。`runs/m4_s0_comparison.json`、各 `evaluation.json` 保存逐轨迹和 D0 预测误差；没有预训练残差收益证据。`runs/m4_s0_contact_diagnostic.json`：ep7 都接触并抬起，但最长稳定抓持证据基准 0.28 s，随机/预训练仅 0.12/0.10 s，低于 0.20 s。保持原成功规则和固定测试；以原预训练臂 `final.pt` 为本轮 S1 的准确输入基准（按预定计划，不按评估结果挑模型）。旧 S0 核心源码与配置记录指纹一致，保留在 `runs/m4_source_snapshot/src/` 便于复现。

M5 v3 观测适配：仿真物理变化与观测误差分别记录。部署模式走渲染度量深度和可见表面拟合，保留原始估计与捕获时间；旋转、角速度和成段缺失的编码器速度有独立掩码，质量/惯性/摩擦未知均隐藏。接触代理只驱动观测阶段，实际成功、奖励及 D0 监督仍用物理事实。先固定 train ep2、seed 42、90 步真实联调验证年龄、掩码及双 Q/D0 梯度；之后按试跑估计时间启动 3,428 原生步、最多 480 次更新、clean/changed 各半的正式 S1，并在固定 ep7/16 做四种配对条件。

M7 固定任务工具验证：`-m movens_sim2real.object_info --request runs/m7_task_setup/request.json --out runs/m7_task_setup/task.json`，离线回放带 `--task-info` 读取同一结构化工具配置与来源卡；12/12 原始帧检测到候选、无三维真值，改变稳定门限的请求被拒绝（同目录 `summary.json`、`negative_check.json`）。R0/R1 数据输入/更新入口已实现；无可信真机转移，未运行训练或给出真机结论。

00:32 启动固定闭环验证：`34737.movens-v3-m6-eval-solo` / timeout 34741 / Python 34746（GPU/EGL 1）；`34742.movens-v3-m6-eval-residual` / timeout 34750 / Python 34751（GPU/EGL 0）。命令 `"$PY" -u -B scripts/evaluate_policy.py --out runs/m6_eval_<arm> --episode 7 16 --base-adapter runs/m6_refit/checkpoint_last.pt --seed 42`，残差臂加 `--small runs/m4_s0_pretrained/final.pt`。`run.log`、`exit`、每条 `clean/apple_*/resume.pt` 同目录；35 分钟上限，受控中断后同命令/原目录自动使用物理断点。两臂均已加载 step 80064 的 92 个 Body / 78 个 Hand 张量并完成精确核验，闭环尚未结束。

M5 v3 联调 `34747.movens-v3-m5-probe-v3` / timeout 34755 / Python 34756（GPU/EGL 0，8 分钟上限）：`"$PY" -u -B scripts/check_deployment.py --small runs/m4_s0_pretrained/final.pt --episode 2 --seconds 1.8 --out runs/m5_probe_v3`，`runs/m5_probe_v3/{run.log,exit,verification.json}`。此固定短试种子在 trial 0 的编码器延迟为 0 tick；正式训练按 episode/trial 独立预定配置检查 1 tick 档位。这是接口及真实转移联调，尚无 S1 正式结果。

00:37 接续：M5 首次 v3 联调 `exit=1`，渲染模型中苹果碰撞件可位于未命名子体，跨编译模型按名称搬运碰撞子体 ID 时失败。已有修复改为在渲染模型内遍历 `apple_root` 子树取可见几何 ID。旧失败目录保留；在新目录 `runs/m5_probe_v3_fixed` 用同一准确 S0 权重和正式采集的 ep2/trial 2（seed 42、编码器延迟 1 tick）重跑 90 个实际周期，硬上限 8 分钟。M6 两臂均已推进过 ep7 的 180 步；资源复查 GPU 各约 13 GiB、均为当前 M6 进程，共享盘约 1008 GiB 可用。暂不启动未通过短联调的 S1。

00:40 M5 v3 联调通过：`36424.movens-v3-m5-probe-fixed` / timeout 36427 / Python 36428 退出 0。`"$PY" -u -B scripts/check_deployment.py --small runs/m4_s0_pretrained/final.pt --episode 2 --trial 2 --seconds 1.8 --out runs/m5_probe_v3_fixed`；`verification.json` 有 90 个真实周期、81/10 个有/无物体观测的边界、24 个速度缺失边界、0 个旋转有效边界；可见时 256 点，最大观测年龄 80 ms、编码器 20 ms、物体中心诊断误差 11.0 mm。actor、双 Q、独立 D0 三成员以真实样本各更新一次。90 步是预定截断，不能作为抓放评估。S1 恢复身份固定适配器/场景源码、S0 权重和原 replay 指纹，禁止把已扰动轨迹混进干净样本。

M5 正式预算：由短试跑稳态 0.90 s/原生周期定界。一次种子 42，从准确 S0 预训练臂分支；固定 train ep1/8/2/4 分别执行物理变化、观测误差、部署输入、部署输入，3,428 个真实 20 ms 周期、最多 480 次更新、batch 64（32 干净 S0 + 32 新采集）、75 分钟硬上限。固定 test ep7/16、双视角、四种条件各 1,353 周期。已有准确输入 S0 干净结果保留；同一 S0 权重在其他三种条件补齐配对基准（4,059 周期、85 分钟上限）。S1 权重完成后各条件通过独立评估入口执行，单条件 35 分钟上限。扰动范围仍为 v3 候选，未依据缺失的真机传感统计定标。启动前复查两块 RTX 5090 各仅本项目 M6 占约 13 GiB，共享盘约 1008 GiB 可用；各 GPU 只追加一项约 13 GiB 的前台计算。

00:40:47 启动正式 S1：`37344.movens-v3-m5-s1` / timeout 37349 / Python 37351（GPU/EGL 0），`timeout -k 90s 75m bash scripts/s1.sh 0`，训练恢复入口同命令追加 `--resume`（先确认该 screen/进程已退出，且剩余预算可用）。`runs/m5_s1/{run.log,exit,config.json,latest.pt}`，每轨物理断点 `train/*/resume.pt`，结束保存 `final.pt`，训练汇总 `evaluation.json`。命令将评估交给固定独立入口以并行执行。

00:40:47 启动 S0 受扰配对：`37347.movens-v3-m5-eval-s0` / timeout 37352 / Python 37354（GPU/EGL 1），`"$PY" -u -B scripts/evaluate_policy.py --out runs/m5_eval_s0 --small runs/m4_s0_pretrained/final.pt --episode 7 16 --modes physics noisy deployment --seed 42`；`runs/m5_eval_s0/{run.log,exit,evaluation.json}`，每条 `<mode>/apple_*/resume.pt`；中断后在同参数和目录自动续跑，已完成轨迹可复用。配对工具现新增积分初态、实际动作与阶段复核；在已有 M4 三臂六条真实轨迹通过（`runs/m4_pair_audit.json`）。M6 两臂 ep7 各完结 854 步：独立成功、加残差未稳定抓握，固定 ep16 尚未完成。

00:54 M6 验收：固定 ep7/16，各臂 1,353 个原生步、64 次名义计划调用，`m6_eval_{solo,residual}/exit` 均 0。`runs/m6_comparison.json` 有四臂的逐轨迹初态、动作、阶段与成功复核。原 Movens 独立 1/2（ep16 稳定 0.92 s 未达标），原 Movens+预训练 S0 残差 0/2，新权重独立 2/2（稳定 6.80/2.14 s），新权重+同一残差 1/2（ep7 未稳定抓握；ep16 稳定 2.02 s）。本轮两条 train 成功回训没有用 test 源，原训练接口读取、缓存、编码、步数和恢复均已证实。独立模式在小留出集改善，旧残差未保持新基础模型的表现，不能外推总体收益。

M7 控制边界集成：准备离线按 50 Hz 逻辑时钟读取原始 30 Hz 编码器及 RGB，检查整块换块、关节映射、残差只改左手、SDK 命令限幅限速、D0 输入区分有无实际施加。短试用 33 个 native 归档帧、准确 S0 checkpoint、固定任务、6 分钟上限。测试用的轴顺序/方向/零点及速度包络明确无真机标定意义，范围来自原模型；在 `runs/m7_control_s0/test_calibration.json`。GPU/EGL 1 仅保留 S0 受扰评估约 13 GiB，盘余约 1007 GiB。真实同步深度、现场时钟、控制器 applied 回执和执行授权仍缺。

M7 控制联调 `41368.movens-v3-m7-control-s0` 退出 0：`"$PY" -u -B scripts/check_robot_control.py --small runs/m4_s0_pretrained/final.pt --task-info runs/m7_task_setup/task.json --calibration runs/m7_control_s0/test_calibration.json --out runs/m7_control_s0 --episode 1 --frames 33`。32 个原生采样供 54 个逻辑决策、3 个计划块；编码器最大年龄 33.3 ms，缺块/陈旧拒绝通过，限速/限位及身体和右手无残差差异通过。`verification.json`、`proposals.npz`、`plans.npz`。再用上述隔离 Unitree 环境执行 `scripts/check_unitree_packets.py runs/m7_control_s0/proposals.npz --calibration runs/m7_control_s0/test_calibration.json --out runs/m7_control_s0/packets.json`；策略实际提案的报文往返最大差 5.38e-8 rad，未发布。与 S0 权重的接口联调通过，仍会拿 S1 权重走同一离线路径；未验证实时 50 Hz 时限。

M9 数据回流软件前置：`-m movens_sim2real.export --hardware --out <run> <trial.json...>` 只收独立确认的连续 train 成功，必须有已施加区间、校准根、因果 RGB、原动作块和组划分；来源明确保留为硬件，训练仍复用原 window/损失。用已验收的 ep15 原生 MuJoCo 日志验证时间索引适配，输出 664 个原目标、133 个 RGB，原 codec 数组逐位一致；未来 RGB 拒绝（`runs/m7_real_export_checks/paired.json`）。真实单目档案因无施加回执被拒绝，实际硬件回训尚无有效输入。

M5 新增部署观测断点验收：相同 90 步、同 ep2/trial 2/seed 42/同 GPU0，在 tick 37 保存训练外的物理积分、感知缓存、丢帧时钟、历史和 RNG，再从同目录恢复，与已通过的未中断试跑对照。12 分钟硬上限。启动前 GPU 0 仅 S1 约 13.4 GiB，盘余 1007 GiB。

续跑 `42293.movens-v3-m5-deploy-resume` / timeout 42295 / 首段 Python 42297，执行原 probe 命令，首段加 `--pause-after 37`，同目录去掉该参数恢复；两个 90 步对照运行都退出 0，但严格等价失败（`runs/m5_resume_probe/equivalence.json`）。实际关节和目标逐位一致；恢复后第一周期起苹果软触碰求解差异积累到约 24 µm，末帧接触数差 1。用同一真实命令做 CPU 分支证明：复制或序列化 `MjData`（约 176 MB）仍失配，`mj_forward` 先构造求解器工作区、再覆盖原积分状态（含 warmstart）后，后续 5 个完整物理周期逐位相同（`native_snapshot.json`）。用独立源码 `runs/m5_restore_candidate` 做完整感知/控制恢复验证，随后再把补丁并回，防止正在运行的 S1/评估记录错误的代码指纹；原运行源码的指纹归档在 `runs/m5_source_snapshot`。首个 S1 段已完成 940 步、134 次更新，正在 ep8；S0 物理变化对照已完成 2/2，其余条件运行中。

修复候选验收预算：同 GPU0、ep2/trial 2、seed 42、90 步，tick 37 暂停后严格比较完整状态、实际目标、感知掩码/历史、D0 输出和 19 个 RGB；12 分钟硬上限，不改原作业或未中断参考。启动前 GPU0 仅 PID 37351 约 13.4 GiB，盘余 1007 GiB。

01:10 恢复修复候选通过：`44881.movens-v3-m5-resume-exact` / timeout 44883 / 恢复 Python 44884，退出 0。在本项目以 `PYTHONPATH=runs/m5_restore_candidate/src:../Movens-Interact` 执行相同 `scripts/check_deployment.py` 参数，输出 `runs/m5_resume_exact`，首段 `--pause-after 37`、续段 `--reference runs/m5_probe_v3_fixed`。`equivalence.json` 显示 109 个数据字段与 19 帧 RGB 全部逐位一致。候选只改变恢复分支，不改变原始初态、正常推进或成功定义；现有长任务未中断，待其读完旧源码后并回。S1 头两条累计 1,721 步/244 次更新，正在 ep2 部署输入；固定评估来自测试 ep7/16，按 seed 42 分别是 0/1 tick 的编码器延迟。

M7 边界复查：当前训练环境 transformers 4.57.6 无 `transformers.models.sam3`，本环境无本地相机/串口设备节点；离线接口验证仍有效，实际 SAM 3 与设备时钟/施加链需要独立依赖、权重和现场条件。现有 30 Hz 归档支持单目和编码器验证，不据此宣称现场延迟分布。

01:18 核验：S1 模型断点保存 1,721 个完整采集周期及 244 次更新，包含优化器、随机数、两条干净回放与代码指纹；第三条实际部署输入已过 tick 720。记录 R0 必须匹配该批冻结 checkpoint 的指纹，事件标签必须有独立实际证据源；真实单目归档拒绝的返回码及原因在 `runs/m7_real_export_checks/negative_archive_check.json`。M6 原 64 步训练代码按运行 `config.json` 指纹复原并保存在 `runs/m6_source_snapshot`，旧缓存路径在本次接口改动后仍读取到 133 个真实窗口与 264 个上下文帧。

01:22：S1 完成三条、2,566 个真实周期及 363 次更新，最后一条 ep4 在执行。S0 `physics` 2/2、`noisy` 0/2 均完成，无工程失败；`runs/m5_eval_s0/noisy_isolation.json` 核验 `clean/noisy` 的两对实际初态和真实参数逐位一致，ep7/16 观测位置误差 p95 8.08/9.15 mm，最大年龄 0/20 ms；S0 部署输入条件正在跑。

M6 累计数据轮前置确认：S1 的 train ep8 在 `noisy` 条件通过原验收（`runs/m5_s1/train/001_apple_000008/summary.json`，781 原生步，含 4.80 s 最终稳定；731 个实际任务动作）。等待 ep4 完成后将全部 S1 合格 train 轨迹与已验收 ep15/32 一次合并，另写 `runs/m6_export_s1`。不覆盖第一次 M6 的结果，不从留出轨迹收样。由已验收的 64 步短耗时确定追加唯一固定同配方：同 80k 父权重、seed 42、64 次更新、batch 8（4 原 + 4 合格示范）、microbatch 2、LR 1e-5/4e-5、旧 val 32 窗口、训练 8 分钟上限，冻结 Qwen 缓存 8 分钟上限；各新闭环臂固定 ep7/16、双视角、seed 42、1,353 周期、35 分钟上限。残差臂保持第一次 M6 的 S0 残差，仅改变监督示范集合；全部结果保留，不因更好的测试结果回选配方或数据。

01:32 接续：v3 DOCX SHA-256 与计划一致，仍只有现行 `AGENTS.md`；独立 Git 尚无提交。S1 退出 0，完成四条、3,428 周期（68.56 s 仿真）、480 次更新；1/4 通过、3 条未稳定抓握、无工程失败。`runs/m5_s1/{evaluation.json,final.pt}`；S0 权重指纹未改。S0 受扰对照已完结五条，部署 ep16 在约 tick 90。两项旧运行的 `config/run_config` 源码指纹均与 `runs/m5_source_snapshot/src` 匹配；S0 最后一条在加载旧源码后已启动。将已验收候选的 `scene.py`/`rollout.py` 原样并回供后续运行；不改正在执行的进程或旧结果。

01:32:28 固定 S1 评估启动：`51914.movens-v3-m5-eval-s1-clean` / timeout 51919 / Python 51921（GPU/EGL 0），`51922.movens-v3-m5-eval-s1-physics` / timeout 51926 / Python 51927（GPU/EGL 1）。`timeout -k 90s 35m bash scripts/s1_eval.sh <gpu> <mode>`；每条件同 ep7/16、seed 42、1,353 步，`runs/m5_eval_s1/<mode>/{run.log,exit,evaluation.json}` 和各 `apple_*/resume.pt`。若中断，先确认对应进程/锁已释放，沿用同源码、设备、参数和目录，评估入口自动续跑物理断点。GPU 0 启动前空闲，GPU 1 仅旧 S0 评估约 13 GiB；盘余约 1007 GiB。待缓存/回训及 S0 作业释放槽位后启动其余两项条件。M6 累计导出已从两个定向训练源和四个 S1 train receipt 启动（CPU，PID 51900），命令 `-m movens_sim2real.export --out runs/m6_export_s1 runs/m6_capture_{15,32}/apple_*/summary.json runs/m5_s1/train/*/summary.json`。

01:37 累计导出和回训结束：入选 train ep8/15/32（3/6），2,049 个实际动作、411 个原生 RGB、207 个训练窗口；Body/Hand 重编码最大差 2.38e-7/8.48e-8 rad（`runs/m6_export_s1/{selection,index,verification}.json`）。`52991.movens-v3-m6-cumulative` 依次在 GPU 0 运行 `MOVENS_REFIT_INDEX="$PWD/runs/m6_export_s1/index.json" MOVENS_REFIT_CONTEXT="$PWD/runs/m6_context_s1" bash scripts/m6_cache.sh 0` 和 `-m movens_sim2real.refit --new-index runs/m6_export_s1/index.json --new-context runs/m6_context_s1 --out runs/m6_refit_s1 --steps 64 --batch 8 --microbatch 2 --new-per-batch 4 --eval-batches 16 --save-every 16 --seed 42 --gpu-memory-fraction .5`（timeout 53222、Python 53223）。各 8 分钟上限，均退出 0；缓存实耗 22 s，训练实际梯度段约 39 s。恢复分别在原命令追加 `--resume --verify-cache-hashes` / `--resume`，在对应目录已有进程退出后使用。产物 `runs/m6_{context,refit}_s1/{run.log,exit}`、`runs/m6_refit_s1/checkpoint_last.pt`；固定新闭环两臂待 GPU 槽位。

01:37 S0 配对基准结束：`runs/m5_eval_s0/exit=0`，物理变化 2/2、噪声 0/2、部署 1/2；部署 ep7 稳定成功、ep16 释放不稳定或盘外。加原干净臂共四组，待 S1 对照。逐条件每组 1,353 原生步，无工程错误。

01:37:43 已启动剩余 S1 评估：`53850.movens-v3-m5-eval-s1-deployment`（GPU/EGL 1）；GPU/EGL 0 `53853.movens-v3-m7-s1-then-noisy` 先运行 `scripts/check_robot_control.py --small runs/m5_s1/final.pt --task-info runs/m7_task_setup/task_usda.json --calibration runs/m7_control_s0/test_calibration.json --out runs/m7_control_s1 --episode 1 --frames 33`（timeout 53861、Python 53863，6 分钟），再运行 `timeout -k 90s 35m bash scripts/s1_eval.sh 0 noisy`。部署/噪声固定预算同上，恢复直接使用相应 `s1_eval.sh`，不重复离线控制检查。启动前两个 GPU 各约 13 GiB、仅本项目的 clean 或 physics 作业，盘余约 1003 GiB。

M7 外部物体信息验证：`-m movens_sim2real.object_info --request runs/m7_task_setup/request.json --apple-fdc-cache runs/m7_task_setup/usda_171688.json --out runs/m7_task_setup/task_usda.json` 真实访问 USDA SR Legacy 171688；本机旧代理端口不可用，查询进程仅对 `api.nal.usda.gov` 直连。响应记录三个公开部分重量 0.149/0.182/0.223 kg，计算直径和来源置信度，无形状时惯性仍未知。以 `--catalog runs/m7_task_setup/usda_171688.json` 离线重算结构化任务逐位一致；真实质量未测。SAM 3 本地仍无权重，直接访问 Hugging Face 超时，未更改训练依赖。

01:39 M7 S1 控制联调退出 0（`runs/m7_control_s1/verification.json`）：原 30 Hz 归档 32 个实际采样支撑 54 个逻辑提案/3 个计划块，记录的是测试包络和提案，0 个实际施加。首块/后续规划约 0.58/0.24 s；部署仍需实测控制时钟。用 `runs/m7_sdk/env/bin/python -B scripts/check_unitree_packets.py runs/m7_control_s1/proposals.npz --calibration runs/m7_control_s0/test_calibration.json --out runs/m7_control_s1/packets.json` 做真实厂商 IDL 往返，误差 5.38e-8 rad、坏 CRC/陈旧拒绝、未发布。部署评估 timeout 53856/Python 53860，噪声 timeout 54083/Python 54084。

M6 累计完成权重核对：`runs/m6_refit_s1/weight_check.json` 确认原 80000 → 80064，64 个不同更新，共 256+256 个窗口；原 Body 89/92、Hand 77/78 个张量变化，保留原父权重、归一化。`evaluation.json` 旧 val 32 窗口 Body RMSE 0.02777 rad。此开环指标不代替两臂的闭环验收。

M1 补录短恢复核验：`55157.movens-v3-m1-resume-exact`，CPU，`-m movens_sim2real.data replay --out runs/m1_dense_resume_exact --episode 1 --seconds .6 --checkpoint-every 10`，首段在第 10 步落盘后 TERM（143），同参数续跑（timeout 55442 / Python 55443）退出 0。`equivalence.json` 与历史连续补录同前 30 步的关节、物体、接触、真实命令、阶段等 9 个字段逐位一致。不重跑已完成原补录。

M7 注册深度离线通过：`scripts/check_sensor_geometry.py runs/m3_no_d0_side/apple_000001 --recorded-worker --out runs/m7_rgbd_worker/verification.json`（EGL 0，3 分钟上限）。RGB 检测掩码独立于 MuJoCo ID，对同帧光学深度构造 256 个部分点、中心误差 6.30 mm、与仿真真值掩码 IoU 0.984；记录的逻辑到达时序验证未到/陈旧拒绝，无旋转真值或真机误差结论。官方份量与仿真度量尺寸组合的惯性/来源为 `runs/m7_task_setup/geometry_priors_usda.json`。`-m movens_sim2real.real_replay --episode 1 --frames 12 --task-info runs/m7_task_setup/task_usda.json --out runs/m7_task_usda_replay` 原真机单目 12/12 个检测候选、0 个可确认三维帧。本机仍缺实际同步 RGB-D、设备时间/控制标定、独立 applied/接触与成功证据及当次真机授权；M8/M9 未启动。

待现场的 R0/R1 接口核查：应用命令区间连续性统一到 0.2 ms；R1 只从完整 R0 或已完成 R1 批次初始化，首批可用原 S1 或仅更新 D0 后的同 actor 权重。加真实试验的更新前/后配对入口；只评估有设备证据和模型指纹的固定留出组，遇未知结果保留样本但不计算成功差值。使用现有真机单目归档确认 R0 拒绝缺施加日志的输入，以及 R1 配对拒绝 S1 权重（`runs/m7_real_input_checks/{r0,r1}.log`）。RGB-D adapter 拒绝错位帧及把 MuJoCo 域当真机域（`runs/m7_rgbd_worker/rejected.json`）。没有可信实际 applied 数据，以上只验输入保护，不宣称 R0/R1 训练或对照运行。

01:54 S1 的干净/物理变化配对分别退出 0，均完成 1,353 真实周期，成功 1/2、2/2；噪声和部署输入还各有一条在执行。配对检查现核验每一臂冻结模型身份一致；在此前 M6 的四臂共 8 条完整真实轨迹通过（`runs/m6_refit_s1/baseline_pair_check.json`）。

01:54:59 M6 累计回训固定闭环启动：`59016.movens-v3-m6-s1-eval-solo` / timeout 59023 / Python 59025（GPU/EGL 1）；`59020.movens-v3-m6-s1-eval-residual` / timeout 59024 / Python 59026（GPU/EGL 0）。两块卡在启动前各仅对应 M5 余下进程占约 13 GiB，盘余 1002 GiB。命令 `timeout -k 90s 35m "$PY" -u -B scripts/evaluate_policy.py --out runs/m6_eval_s1_<arm> --episode 7 16 --base-adapter runs/m6_refit_s1/checkpoint_last.pt --seed 42`，残差臂附 `--small runs/m4_s0_pretrained/final.pt`；两臂均精确加载 92 个 Body、78 个 Hand 张量。`runs/m6_eval_s1_<arm>/{run.log,exit,evaluation.json}` 与 `clean/apple_*/resume.pt`；确认进程退出后沿用相同目录及原参数续跑。两臂各 1,353 真实周期，无进一步调参或追加数据计划。

02:00 M5 正式验收：各条件均退出 0，`"$PY" -B scripts/check_s1.py --out runs/m5_comparison.json` 通过。固定 ep7/16，S0/S1 四条件各 1,353 原生步，全对照 10,824 步；无工程错误。S0→S1 成功计数干净 0→1、物理变化 2→2、噪声 0→1、部署 1→1，每条件仅 2 条；部署 ep16 仍失败（S0 放置不稳或盘外，S1 未稳定抓持）。核对了逐对积分初态、实际目标及阶段、物理/观测隔离和 D0 误差。S1 部署 1194/1355 边界有可见估计，161 缺失，最大年龄 80 ms、姿态/物性按独立掩码处理；D0 关节一步 p95 由 0.00382 降至 0.00245 rad，物体位置 p95 约 2.87→2.99 mm，不能据此声称动力学整体提高或部署成功率改善。`runs/m5_s1/weight_check.json` 确认 actor、双 Q、三成员实际更新，最新可恢复权重与最终权重逐位相同、原 S0 SHA 保留。旧正式运行均结束；其历史恢复命令对应 `runs/m5_source_snapshot/src` 的旧源码指纹，新物理恢复实现用于新运行。

M5 有限失败诊断：`runs/m5_eval_s1/ep16_physical_diagnostic.json` 用实际物体升高、多指接触、支撑及相对速度复核 ep16。S1 部署实际抬高 52.1 mm，组合稳定抓握证据仅 0.08 s（要求 0.20 s）；相同物理条件的 S1 `physics` 有 0.24 s、成功。S0 部署有 0.32 s 抓握证据但放置失败。只定位这两条失败发生的阶段，不能单独归因于某项输入误差。

02:14 M6 累计轮闭环验收：两个 `runs/m6_eval_s1_{solo,residual}/exit` 均 0；执行 `"$PY" -B scripts/compare_rollouts.py`，以 `frozen`、`frozen_residual`、`refit_solo`、`refit_residual`、`cumulative_solo`、`cumulative_residual` 为六臂，输入前述各 runs 下固定 ep7/16 的 `summary.json`，输出 `runs/m6_comparison_s1.json`。六臂各 1,353 原生步、无工程错误，开始积分状态、实际动作、成功标签和每臂冻结模型身份检查通过。累计轮独立 1/2（ep7 稳定 6.02 s；ep16 0.68 s，未达 1.0 s），累计轮加同一 S0 残差 2/2（稳定 7.66/1.26 s）。首轮独立 2/2、加残差 1/2；原权重独立 1/2、加残差 0/2。新增一条训练源后，独立执行在这两个案例下降，残差组合上升；保留两个权重、所有对照及失败，不回选训练配方或外推总体收益。两套 80064 权重分别记录在 `runs/m6_refit_s1/checkpoint_digests.sha256`。

本轮收尾：M0–M6 的既定有限预算均有实际验证，M7 现有软件与离线验收通过。最后查询无在运行的项目 screen，先前 PID 已退出，两块 GPU 当时均 0 MiB，共享盘约 1002 GiB 可用。若未来接续，保留 `runs/m4_s0_pretrained/final.pt` 为准确输入基准，`runs/m5_s1/final.pt` 为部署输入的 D0 分支；回训结果分别在 `runs/m6_refit/checkpoint_last.pt` 与 `runs/m6_refit_s1/checkpoint_last.pt`。M8/M9 的恢复入口见 `PROJECT_PLAN.md` §6–7：先具备实际设备、当次执行授权、现场时钟/标定和独立 applied 数据，再按 source 分开 train/test，使用 `-m movens_sim2real.calibration --checkpoint runs/m5_s1/final.pt --receipt <已审计真机记录> --out <R0运行目录> --steps <实测后登记预算>`；通过后才做 R1 配对及成功回流。SAM 3 权重/兼容环境和实际 RGB-D 仍待提供。X1/D1 未启动。

GitHub 发布准备（用户已指定公开仓库 `BLACKCXX/Movens-sim2real`）：整理现有代码、配置、v3 报告与项目文档，保留全部研究结果说明；`runs/`、原始数据、权重、视频和环境密钥不进入 Git。45 个候选文件约 2.36 MB，常见凭据模式扫描和 Python/Bash/JSON 语法检查通过。当前服务器尚无 GitHub 登录凭据；先完成本地初始提交，再通过官方 CLI 登录、建立远程并推送，实际发布状态在完成后追加。
