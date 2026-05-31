按当前仓库结构和作业线索，更合理的第三次作业理解是：

PPT/课程内容强调 **DeepONet** 与 **FNO** 两类 neural operator；仓库中已有两个一维 demo：

- `Codes/Neural_Operator/DeepONet`：TensorFlow v1 风格的 ODE operator demo。
- `Codes/Neural_Operator/FNO_earthquake1D`：PyTorch 1D FNO earthquake demo。

同时，`Data/Cavity` 提供了 cavity flow 数据，但没有现成训练代码。因此主任务应当是：**在同一个 Cavity Flow 数据集上分别实现 DeepONet 与 FNO，并对比两种方法的误差和预测效果**。已有 demo 作为参考或最小复现材料，不应替代 Cavity Flow 上的双模型对比。

---

请基于 `XuhuiM/SciML-Course` 仓库完成**第三次作业**。要求以 **Cavity Flow** 为主实验对象，分别实现 **DeepONet** 和 **FNO** 两种 neural operator，并在同一训练/测试数据、同一指标体系下进行对比。

### 总体目标

完成一个可复现的神经算子小项目，包含三部分：

1. **现有 demo 说明或最小复现**：说明仓库中的 `DeepONet` ODE demo 和 `FNO_earthquake1D` demo 的作用、代码入口、依赖和可运行方式。
2. **Cavity DeepONet 实现**：基于 `Data/Cavity/Cavity_Flow.mat` 和 `Data/Cavity/Cavity_Flow_Test.mat`，实现用于 cavity flow 的 DeepONet。
3. **Cavity FNO 实现**：基于相同数据，实现 2D FNO，并与 DeepONet 进行公平对比。

### 已知仓库信息

- DeepONet demo 路径：`Codes/Neural_Operator/DeepONet`
- FNO 1D demo 路径：`Codes/Neural_Operator/FNO_earthquake1D`
- Cavity 数据路径：`Data/Cavity`
- Cavity 数据实际字段：
  - `Cavity_Flow.mat`
    - `u_bc`: `(100, 65)`，训练输入，上壁边界条件
    - `u_data`: `(100, 65, 65)`，训练输出，水平速度场
    - `v_data`: `(100, 65, 65)`，训练输出，垂直速度场
    - `x_2d`, `y_2d`: `(65, 65)`，空间网格
  - `Cavity_Flow_Test.mat`
    - `u_bc`: `(10, 65)`，测试输入
    - `u_data`: `(10, 65, 65)`，测试水平速度场
    - `v_data`: `(10, 65, 65)`，测试垂直速度场
    - `x_2d`, `y_2d`: `(65, 65)`，空间网格

注意：`.mat` 文件没有显式提供归一化参数。实现时应先保存数据字段、shape、dtype 和数值范围，再决定是否做标准化或归一化。

### 具体实现要求

#### A. Demo 说明或最小复现

1. `reproduce_deeponet.py` 或 `README` 说明
   - 说明原始 DeepONet demo 是 ODE operator，不是 cavity flow。
   - 说明其依赖 TensorFlow v1 风格 API。
   - 如果当前环境缺少 TensorFlow，可基于已有 checkpoint/输出文件说明最小复现方式。

2. `reproduce_fno_earthquake1d.py`
   - 参考 `Codes/Neural_Operator/FNO_earthquake1D/fourier_1d.py`。
   - 输出训练/测试 loss 曲线和至少一个测试样本预测曲线。
   - 原始代码中预测保存部分是注释状态，可封装恢复。

#### B. 主任务：Cavity DeepONet

请实现一个 PyTorch 版 DeepONet，用于学习：

`u_bc(y) -> [u(x,y), v(x,y)]`

推荐设计：

- Branch net 输入：`u_bc`，shape 为 `(65,)`。
- Trunk net 输入：空间坐标点 `(x, y)`。
- 输出：每个空间点上的两个速度分量 `(u, v)`。
- 可以采用两个输出头，或分别训练 `u`、`v` 两个 DeepONet。
- 训练时将每个样本的 `65 x 65` 网格展开为点集，预测所有网格点。
- 支持 batch 训练、CPU/GPU、固定随机种子、模型保存和日志保存。

#### C. 主任务：Cavity FNO

请实现一个 2D FNO，用于同一映射：

`u_bc(y) -> [u(x,y), v(x,y)]`

推荐设计：

- 将 `u_bc` 扩展到二维网格，作为输入通道。
- 拼接坐标通道 `x_2d, y_2d`。
- 输入通道示例：`[u_bc_field, x, y]`。
- 输出通道：`[u, v]`。
- 参考 1D FNO，把 `SpectralConv1d` 扩展为 `SpectralConv2d`。
- FNO 主体建议包含 lifting layer、4 个 Fourier blocks 和 projection head。

#### D. 训练与评估

DeepONet 和 FNO 必须使用相同训练/测试划分，并输出同一套指标：

- overall relative L2 error
- overall MSE
- `u` 分量 relative L2 / MSE
- `v` 分量 relative L2 / MSE
- 每个测试样本的 relative L2 error
- 平均测试误差和标准差

训练要求：

- 支持 CPU/GPU。
- 固定随机种子。
- 保存最优模型。
- 保存训练日志。
- 配置 batch size、epoch、learning rate、网络宽度等超参数。

#### E. 可视化

至少生成以下图表：

- DeepONet 与 FNO 的训练/测试 loss 曲线。
- 同一测试样本上，`u` 的真值、预测、误差热力图。
- 同一测试样本上，`v` 的真值、预测、误差热力图。
- 速度模长 `sqrt(u^2+v^2)` 的真值/预测/误差图。
- DeepONet vs FNO 的误差柱状图或表格图。
- 若容易实现，可补充流线图或矢量图。

#### F. 对比实验

至少完成一组公平对比：

- `Cavity DeepONet`
- `Cavity FNO`

如果时间允许，可补充小规模消融：

- FNO 不同 `modes` / `width`
- DeepONet 不同 latent dimension / width
- 是否加入坐标归一化或输出标准化

### 代码结构建议

建议生成如下目录：

- `third_assignment/`
  - `README.md`
  - `requirements.txt`
  - `reproduce_deeponet.py`
  - `reproduce_fno_earthquake1d.py`
  - `cavity/`
    - `data_loader.py`
    - `deeponet.py`
    - `fno2d.py`
    - `train_deeponet.py`
    - `train_fno.py`
    - `evaluate.py`
    - `visualize.py`
    - `utils.py`
    - `config.yaml`
    - `outputs/`
  - `report_assets/`
  - `report.md`

### 报告必须涵盖的内容

请自动生成一个简短 `report.md`，内容至少包括：

1. **作业目标**：说明第三次作业围绕 neural operator，并在 Cavity Flow 上比较 DeepONet 与 FNO。
2. **方法简介**：
   - DeepONet 如何通过 branch/trunk 网络学习函数到函数映射。
   - FNO 如何通过频域卷积学习算子。
   - 两者在本任务中如何接收 `u_bc` 并输出 `(u,v)`。
3. **数据说明**：
   - 数据文件名。
   - `.mat` 的关键字段、shape、dtype 和数值范围。
   - 训练/测试划分。
   - 是否做了归一化或标准化。
4. **实验设置**：
   - DeepONet 结构。
   - FNO 结构。
   - batch size / epoch / learning rate。
   - 训练设备。
5. **结果展示**：
   - demo 说明或复现结果。
   - Cavity DeepONet 结果。
   - Cavity FNO 结果。
   - 两种方法的误差对比表。
   - 若干可视化图。
6. **结论**：
   - DeepONet 与 FNO 的差异。
   - 哪个方法在 Cavity Flow 上表现更好。
   - 当前结果是否合理，以及可能的改进方向。

### 实现注意事项

- 主实现统一用 **PyTorch**。
- 先检查 Cavity `.mat` 文件内容再写 loader。
- 不要假设 Cavity 已有现成训练脚本；当前仓库只有 demo 和数据。
- DeepONet demo 是 TensorFlow v1 风格；如果当前环境不能运行 TensorFlow，应在 README/report 中说明。
- 所有输出图和表都保存在本地目录，便于直接写作业报告。
- 命令行入口要清晰，代码要能直接运行。

### 验收标准

- 明确说明或最小复现仓库 DeepONet 与 FNO 1D demo。
- Cavity Flow 上有可运行的 PyTorch DeepONet。
- Cavity Flow 上有可运行的 PyTorch 2D FNO。
- 两个模型使用同一数据划分和同一指标。
- 自动生成结果图、误差表和 `report.md`。
- 项目结构清晰，可直接作为课程作业提交。
