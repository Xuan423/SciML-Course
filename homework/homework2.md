下面这版可以直接当作**面向 Codex 的实现说明**发给它：

---

请用 **PyTorch** 实现一个 **Allen-Cahn 方程的一维 PINN 求解器**，采用连续时间 PINN 形式：用神经网络 (u_\theta(x,t)) 逼近解，并通过自动微分计算 (u_t,u_x,u_{xx})，构造 PDE 残差
[
f(x,t)=u_t-Du_{xx}+5(u^3-u),\quad D=10^{-4},
]
求解区域为 (x\in[-1,1],, t\in[0,1])，初值为
[
u(x,0)=x^2\cos(\pi x),
]
边界条件为
[
u(-1,t)=u(1,t)=-1.
]
PINN 的基本训练目标应包含 PDE 残差项、初值项和边界项，这是经典 PINN 的标准做法；自动求导请基于 PyTorch `autograd` 实现。([ISU Sites][1])

作业需要覆盖以下内容，请全部实现并可切换：

1. **采样策略**：支持 `uniform` 和 `random` 两种内部配点方式。
2. **自适应加点**：先训练若干 epoch，再在候选点上计算残差 (|f(x,t)|)，把残差最大的若干点加入 collocation set，继续训练。
3. **损失权重策略**：支持

   * `equal_weights`：所有权系数均为 1；
   * `adaptive_weights`：根据各项 loss 的相对大小动态更新 (\lambda_f,\lambda_{ic},\lambda_{bc})。
4. **实验对比**：至少输出以下四组结果：

   * uniform + equal_weights
   * random + equal_weights
   * random + adaptive_points
   * random + adaptive_weights
5. **结果展示**：保存

   * 总 loss 曲线及各子损失曲线；
   * (u(x,t)) 的二维热力图；
   * 若干时刻的截面曲线；
   * 自适应加点前后的配点分布图；
   * 各方案的误差指标表（如 PDE residual MSE、IC/BC MSE，若有参考解再给相对 (L_2) 误差）。Allen-Cahn 是 DeepXDE 官方 PINN 示例之一，这样的展示方式与常见 PINN forward-problem 实现一致。([DeepXDE][2])

实现要求尽量简洁，建议文件结构为：

* `model.py`：MLP 网络；
* `sampler.py`：uniform/random/adaptive sampling；
* `losses.py`：PDE、IC、BC loss 与自适应权重；
* `train.py`：训练主流程；
* `plot.py`：画图与结果保存；
* `config.yaml`：超参数配置。

默认网络建议使用 `tanh` 激活的全连接 MLP，例如 4~6 层、每层 64~128 单元；优化器先用 Adam，再可选 L-BFGS 精修。训练脚本应支持随机种子、GPU、结果复现和命令行切换实验配置。([ISU Sites][1])

验收标准：代码可直接运行；四组实验都能自动完成；输出图表和结果表；最终生成一个简短 `report.md`，说明 PDE、损失函数、采样方式、自适应加点方法、自适应加权方法及四组实验现象对比。

---

如果你要，我可以继续把这段直接扩成一版**可复制给 Codex 的英文 prompt**，或者直接替你写成**项目目录 + 核心代码骨架**。

[1]: https://faculty.sites.iastate.edu/hliu/files/inline-files/PINN_RPK_2019_1.pdf?utm_source=chatgpt.com "Physics-informed neural networks"
[2]: https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html?utm_source=chatgpt.com "Allen-Cahn equation - DeepXDE - Read the Docs"
