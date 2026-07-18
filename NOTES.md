# Teaching Notes

## 学习者画像
- 强算法背景（ACM 获奖），多段推理框架实习。抽象能力强，学得快，可跳过基础语法。
- 熟练 CUDA core 算子；已用 WMMA 在远程 GPU 实现并验证 tensor core 算子；正在进入裸 `mma.sync`/`ldmatrix` 层。Hopper/Blackwell 特性不熟。
- 高强度投入 >15h/周。

## 教学偏好 / 原则
- **看穿黑盒导向**：每节课的落脚点是"你现在能读懂/质疑 Agent 说的 X"，而不只是"你能从零写出 X"。读代码 + 判断权衡 优先于 从空文件手写。
- 深度到 **PTX 指令层**；SASS 只在 `ncu` 定位瓶颈时顺带认，不逐条读。
- 直接在 H20/B200/B300 数据中心真机上做练习与验证：H20 跑 Hopper 专属特性，B200 / B300 跑 Blackwell 专属特性（无本地卡）。
- 概念可以讲得快、密度可以高（工作记忆容量比一般学习者大），但每节课仍只锁定"一个 tangible win"。
- 尽量把练习绑定到他真实项目的 MoE 计算侧 kernel。

## 待确认 / 后续可深挖
- 真实项目具体是哪个框架（vLLM / SGLang / TRT-LLM / 自研）——下次可问，用来选更精准的参考 kernel。
- 两条对照透镜（Triton / TIRx）的相对权重：目前均定位为对照透镜（Triton 偏快速 baseline，TIRx 偏 Blackwell 概念/图解），非主线实操；是否要给某一条更多篇幅，待用户偏好确认。
- 精度重点顺序：FP16/BF16 打底 → FP8 → Blackwell FP4/MXFP（MoE 量化会重度用到）。
