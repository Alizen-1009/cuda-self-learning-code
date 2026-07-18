# AGENTS.md — CUDA / Tensor Core 学习工作区

这是一个用 `/teach` 技能驱动的**长期学习工作区**。每次会话开始，先读下面的指针进入状态。

## 学习者画像
- 强算法背景（ACM 获奖），多段推理框架实习。抽象/工程能力强，**无需从 C++/CUDA 基础教起**。
- 熟练 CUDA core 算子；**tensor core 零基础**；Hopper/Blackwell 特性不熟。
- 硬件：H20(Hopper) · B200 / B300(Blackwell)，全部为数据中心真机。
- 高强度投入 >15h/周。

## 核心目标（详见 [MISSION.md](./MISSION.md)）
不是闭门手写算子，而是**深度到能与 AI Agent 平等协作、看穿黑盒**：读懂 / 质疑 / 指方向 / 纠错。
- 深度终点：**PTX 指令层**（`mma.sync` / `wgmma` / `tcgen05`-TMMA）。
- 主线：**MoE 计算侧** kernel（grouped/量化 GEMM），Hopper + Blackwell 双目标。
- 通信层（NVSHMEM/all2all/DeepEP）排除出主线。

## 工作区导航（详见 [STRUCTURE.md](./STRUCTURE.md)）
- `MISSION.md` 使命 · `NOTES.md` 教学偏好 · `RESOURCES.md` 知识源 · `STRUCTURE.md` 目录说明
- `learning-records/` 学习记录（算 ZPD，决定下一步教什么）
- `reference/` 参考文档（`glossary.html` 术语表——每课遵循）
- `lessons/` 课程（编号递增 HTML）· `assets/` 共享样式/组件
- `repos/` 你 clone 的 8 个参考仓库（含 `modern-gpu-programming-for-mlsys`：MLC/CMU 书，Blackwell 心智模型+图解，TIRx DSL 作第二对照透镜）

## 教学约定
- 每节课只锁定**一个 tangible win**，短、可快速完成，落脚在"你现在能看懂/质疑 Agent 说的 X"。
- 读代码 + 判断权衡 优先于 从空文件手写；深度到 PTX，SASS 仅随 `ncu` 顺带。
- 练习尽量绑定真实项目的 MoE 计算侧 kernel；直接在 H20/B200/B300 上迭代与验证专属特性。
- 每课含即时回忆题（答案等长，不靠格式泄题）、主源推荐、术语表链接、"随时追问"提示。
- 新术语先进 `glossary.html`；有非平凡的掌握/纠误/目标变化就写 `learning-records/`。

## 学习路径（阶段）
0 性能心智模型 → **1 `mma.sync`+ldmatrix**（当前，4060ti）→ 2 GEMM+CuTe → 3 Hopper(wgmma+TMA) → 4 Blackwell(tcgen05+TMEM+FP4) → 5 MoE 毕业作品。**Triton + TIRx 作两条对照透镜穿插**（TIRx 见 MLC 书）。阶段 3~4 的 Blackwell 心智模型与图解主要来自 `repos/modern-gpu-programming-for-mlsys`。

## 进度
- ✅ 初始访谈完成，mission/路径确立（`learning-records/0001`）
- ✅ 第 1 课：从 CUDA Core 到 Tensor Core（`lessons/0001`）
- ✅ 新增知识源：`modern-gpu-programming-for-mlsys`（Blackwell 主脊 + TIRx 透镜，`learning-records/0002`）
- ⏭ 下一课：手写 `mma.sync` + `ldmatrix`（拆开 wmma，图解 fragment 映射）

## 待确认（见 NOTES.md）
真实项目具体框架（vLLM/SGLang/TRT-LLM/自研）· Triton 权重 · 精度顺序(FP16→FP8→FP4)。

---
> 原始需求（存档）：见首次 `/teach` 记录——"从 cuda core 进阶到精通 tensor core / MMA / TMMA / CUTLASS，掌握内存层次与线程编排"。已展开为上面的 MISSION。
