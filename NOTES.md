# Teaching Notes

## 学习者画像
- 强算法背景（ACM 获奖），多段推理框架实习。抽象能力强，学得快，可跳过基础语法。
- 只写过 CUDA core 算子；tensor core 零基础；Hopper/Blackwell 特性不熟。
- 高强度投入 >15h/周。

## 教学偏好 / 原则
- **看穿黑盒导向**：每节课的落脚点是"你现在能读懂/质疑 Agent 说的 X"，而不只是"你能从零写出 X"。读代码 + 判断权衡 优先于 从空文件手写。
- 深度到 **PTX 指令层**；SASS 只在 `ncu` 定位瓶颈时顺带认，不逐条读。
- 用本地 4060ti(Ada) 做快速迭代练习；H20/B200 用于 Hopper/Blackwell 专属特性的真机验证。
- 概念可以讲得快、密度可以高（工作记忆容量比一般学习者大），但每节课仍只锁定"一个 tangible win"。
- 尽量把练习绑定到他真实项目的 MoE 计算侧 kernel。

## 待确认 / 后续可深挖
- 真实项目具体是哪个框架（vLLM / SGLang / TRT-LLM / 自研）——下次可问，用来选更精准的参考 kernel。
- 是否需要把 Triton 作为并列主力（目前定位：作为对照透镜 + 快速 baseline，穿插在各阶段）。
- 精度重点顺序：FP16/BF16 打底 → FP8 → Blackwell FP4/MXFP（MoE 量化会重度用到）。
