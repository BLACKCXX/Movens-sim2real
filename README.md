# Movens-sim2real

本项目实现 v3 的真实回放标注、两个小模型预训练、冻结 Movens 的 MuJoCo 闭环，以及成功轨迹回训和部署适配。实际完成状态和恢复入口见 `PROGRESS.md`，接口与运行命令见 `PROJECT_PLAN.md`。在服务器本项目根目录工作，复用相邻 Movens-Interact 的 Python 环境和只读资产。原始数据、运行日志与权重不提交到仓库。

| 文件 | 用途 | 后续维护 |
|---|---|---|
| `AGENTS.md` | 稳定约束、阅读顺序、工作方式、简洁实现要求 | 只有稳定规则变化才改 |
| `PROJECT_PLAN.md` | 全流程架构、接口、依赖、阶段验收与运行表 | 接口核验或方案决定变化时改 |
| `PROGRESS.md` | 当前状态、证据、阻塞、长任务和恢复入口 | 阶段结束/中断/阻塞时改 |
| `START_PROMPT.md` | 一次性启动指令及续接短句 | 主要供用户复制 |
| `README.md` | 使用方式与方法来源 | 按需更新 |

`AGENTS.md` 是 Codex 识别的项目指令文件；另外两份计划/进度文件由它明确指向，不需要特殊文件名或额外插件。新建文件后，在项目根目录开始新任务，使 Codex 读取当前约定。若继承的上级指令存在冲突，由本地 Codex 报告具体来源并按实际指令优先级处理。

## 这套安排解决什么

- 完整目标一次交付，实施按真实依赖展开；通过验收自动继续，不设人为逐阶段确认关卡。
- 实验结果、验收证据和待办按实际执行维护。
- 在完整功能与验收要求下精简代码和架构。先用少量样本验证完整路径，再完成约定训练与评估，不能以阶段性演示替代最终交付。
- 复用已有数据和上游代码；省去冗余层与无关框架，不省必做功能、数据、恢复能力或必要验证。
- 动力学 D0、模型辅助 D1、基础 checkpoint 回训及真机条件分清，避免研究对照混杂。
- 长训练有有限预算、进程与恢复记录；换会话可以续接。文件能提供恢复上下文，但不能保证一次会话永久运行或自动克服缺少的硬件和额度。

## 查阅的方法资料

以下是本启动包采用的方法来源，核对日期为 2026-11-17。这里只提取适合当前科研工程的原则，没有照搬来源中的全部流程或审批要求。具体模型、阶段和数据约定来自用户的 v3 方案及后续讨论。

1. [OpenAI：AGENTS.md 项目指令](https://developers.openai.com/codex/guides/agents-md)：将稳定约定放在项目指令文件，通过明确引用连接计划与运行信息。
2. [OpenAI：Using PLANS.md for multi-hour problem solving](https://developers.openai.com/cookbook/articles/codex_exec_plans)：可持续更新的执行计划、明确里程碑、可观察的验收结果和恢复上下文；采用其思想，不另加一份重复的 PLANS.md。
3. [OpenAI：Run long horizon tasks with Codex](https://developers.openai.com/blog/run-long-horizon-tasks-with-codex)：实现、运行、修复、记录的循环，以及长任务中的项目记忆。
4. [OpenAI：Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)：按需读取资料，删除过时或过量规则，避免不必要的测试和人为停止点；这些约定不依赖特定模型才能使用。
5. [Google：What to look for in a code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html)：检查复杂度，避免没有当前需求的泛化，测试自身也应清晰且有价值。
6. [Google：Small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)：围绕一个完整行为组织改动，便于理解和回退；不设置机械的文件数或代码行数限制。
7. [Martin Fowler：Yagni](https://martinfowler.com/bliki/Yagni.html)：推迟尚未需要的能力，并保留让当前代码易于修改的必要设计。
8. [PyTorch：Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)：记录种子、环境和数据处理条件，不把相同 seed 当成跨版本/硬件严格一致的保证。

## 维护规模

常驻规则保持简短；模型参数、运行命令与实际进度各有唯一位置。不要为维护这些 Markdown 文件再开发一个项目管理系统。若未来任务规模确实超出当前结构，再依据已经出现的需求拆分。
