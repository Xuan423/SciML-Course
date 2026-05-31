# Homework 3: DeepONet and FNO for Cavity Flow

本目录实现第三次作业的 PyTorch 主任务：在同一 Cavity Flow 数据集上比较 DeepONet 与 2D FNO。

## 目录说明

- `data/`: 从 `Data/Cavity` 复制来的 `Cavity_Flow.mat`, `Cavity_Flow_Test.mat`, `CMAME.pdf`。
- `cavity/`: 数据读取、DeepONet、FNO2d、训练、评估、可视化与 Word 报告生成代码。
- `outputs/`: 模型、预测、日志、指标表。
- `report_assets/`: Word 报告插图。
- `run_all.py`: 一键运行训练、评估、绘图和中文 Word 报告生成。

## 一键运行

```bash
conda activate phmbench
cd Codes/Neural_Operator/third_assignment
python run_all.py
```

运行结束后：

- 数值结果在 `outputs/`
- 图像结果在 `report_assets/`
- Word 报告在 `homework/homework3_report.docx`

## Demo 说明

仓库已有两个参考 demo：

- `Codes/Neural_Operator/DeepONet`: TensorFlow v1 风格 ODE operator demo。当前 `phmbench` 环境未安装 TensorFlow，因此本作业主实现改用 PyTorch DeepONet，并在报告中说明原始 demo 依赖。
- `Codes/Neural_Operator/FNO_earthquake1D`: PyTorch 1D FNO demo。其 `SpectralConv1d` 和 FNO 训练流程被用于参考，主任务中扩展为 `SpectralConv2d`。
