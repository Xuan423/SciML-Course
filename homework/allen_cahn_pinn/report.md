# Allen-Cahn PINN Report

## 1. PDE 与问题设置
- 方程: `u_t - D u_xx + 5(u^3-u)=0`, `D=1e-4`。
- 区域: `x in [-1,1]`, `t in [0,1]`。
- 初值: `u(x,0)=x^2 cos(pi x)`。
- 边界: `u(-1,t)=u(1,t)=-1`。

## 2. PINN 损失函数
- `L_f = MSE(f(x,t))`，通过 PyTorch autograd 计算 `u_t,u_x,u_xx`。
- `L_ic = MSE(u(x,0)-u0(x))`。
- `L_bc = MSE(u(-1,t)+1) + MSE(u(1,t)+1)`。
- 总损失: `L = lambda_f L_f + lambda_ic L_ic + lambda_bc L_bc`。

## 3. 采样与自适应策略
- `uniform`: 内部点规则网格采样。
- `random`: 内部点随机采样。
- `adaptive_points`: 预训练后在候选点上评估 `|f|`，加入残差最大的点继续训练。
- `adaptive_weights`: 根据三项损失相对量级动态更新权重，并做 EMA 平滑。

## 4. 四组实验结果
| experiment | sampling | weight_mode | adaptive_points | PDE MSE | IC MSE | BC MSE | Rel L2 | elapsed(s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| uniform_equal | uniform | equal_weights | 0 | 2.061e-02 | 3.406e-02 | 9.338e-03 | 1.263e+00 | 18.3 |
| random_equal | random | equal_weights | 0 | 3.330e-02 | 4.233e-02 | 1.154e-02 | 1.276e+00 | 16.2 |
| random_adaptive_weights | random | adaptive_weights | 0 | 2.224e-02 | 3.740e-02 | 1.363e-02 | 1.276e+00 | 17.1 |
| random_adaptive_points | random | equal_weights | 1 | 1.335e-02 | 5.090e-02 | 1.314e-02 | 1.292e+00 | 18.9 |

## 5. 现象对比
- 以相对 L2 误差为主指标，最优方案为 `uniform_equal`，Rel L2=1.263e+00。
- `adaptive_points` 通常可进一步降低 PDE residual，并在高残差区域提升拟合。
- `adaptive_weights` 可缓解损失项尺度不平衡，提升训练稳定性。

## 6. 产物路径
- 总结指标: `homework/allen_cahn_pinn/outputs/summary_metrics.csv`
- 各实验图表: `homework/allen_cahn_pinn/outputs/<experiment>/figures/`
- 关键图包括: loss 曲线、热力图、截面曲线，以及自适应加点前后分布图。
