import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import yaml

from losses import AdaptiveWeightConfig, AdaptiveWeights, compute_losses, pde_residual
from model import MLP
from plot import ensure_dir, plot_adaptive_points, plot_loss_curves, plot_slices, plot_solution_heatmap
from sampler import AllenCahnSampler, Domain


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str):
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def resolve_path(path_like: str, base_dir: Path):
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def build_batch(sampler: AllenCahnSampler, train_cfg, sampling):
    x_f, t_f = sampler.sample_interior(train_cfg["n_f"], strategy=sampling)
    x_ic, t_ic, u_ic = sampler.sample_ic(train_cfg["n_ic"])
    x_lb, t_lb, u_lb, x_rb, t_rb, u_rb = sampler.sample_bc(train_cfg["n_bc"])
    return {
        "x_f": x_f,
        "t_f": t_f,
        "x_ic": x_ic,
        "t_ic": t_ic,
        "u_ic": u_ic,
        "x_lb": x_lb,
        "t_lb": t_lb,
        "u_lb": u_lb,
        "x_rb": x_rb,
        "t_rb": t_rb,
        "u_rb": u_rb,
    }


def sample_adaptive_points(model, sampler, diffusion, n_candidates, n_add):
    model.eval()
    with torch.enable_grad():
        x_c, t_c = sampler.sample_candidates(n_candidates)
        residual = pde_residual(model, x_c, t_c, diffusion=diffusion, create_graph=False)
        score = torch.abs(residual).reshape(-1)
        n_add = min(int(n_add), score.numel())
        ids = torch.topk(score, k=n_add).indices
    return x_c[ids].detach(), t_c[ids].detach(), float(torch.mean(score[ids]).item())


def evaluate_metrics(model, sampler, diffusion, n_eval_pde, x_ref, t_ref, u_ref, device):
    model.eval()
    with torch.enable_grad():
        x_eval, t_eval = sampler.sample_interior(n_eval_pde, strategy="random")
        f_eval = pde_residual(model, x_eval, t_eval, diffusion=diffusion, create_graph=False)
        pde_mse = float(torch.mean(f_eval ** 2).detach().item())

    with torch.no_grad():
        x_ic, t_ic, u_ic = sampler.sample_ic(2048)
        u_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))
        ic_mse = float(torch.mean((u_ic_pred - u_ic) ** 2).item())

        x_lb, t_lb, u_lb, x_rb, t_rb, u_rb = sampler.sample_bc(2048)
        u_lb_pred = model(torch.cat([x_lb, t_lb], dim=1))
        u_rb_pred = model(torch.cat([x_rb, t_rb], dim=1))
        bc_mse = float(
            0.5 * (torch.mean((u_lb_pred - u_lb) ** 2) + torch.mean((u_rb_pred - u_rb) ** 2)).item()
        )

        xx, tt = np.meshgrid(x_ref, t_ref, indexing="xy")
        xt = np.stack([xx.reshape(-1), tt.reshape(-1)], axis=1).astype(np.float32)
        xt_tensor = torch.tensor(xt, dtype=torch.float32, device=device)
        u_pred = model(xt_tensor).detach().cpu().numpy().reshape(len(t_ref), len(x_ref))
        rel_l2 = float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))

    return pde_mse, ic_mse, bc_mse, rel_l2, u_pred


def load_reference(mat_path: Path):
    data = sio.loadmat(mat_path)
    x = data["x"].reshape(-1).astype(np.float32)
    t = data["t"].reshape(-1).astype(np.float32)
    u = data["u"].astype(np.float32)
    if u.shape == (len(t), len(x)):
        return x, t, u
    if u.shape == (len(x), len(t)):
        return x, t, u.T
    raise ValueError(f"Unexpected reference solution shape: {u.shape}")


def write_history_csv(path: Path, history):
    if not history:
        return
    keys = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def train_single(base_cfg, exp_name, exp_cfg):
    run_cfg = copy.deepcopy(base_cfg)
    run_cfg["experiment"] = {"name": exp_name, **exp_cfg}
    config_dir = Path(run_cfg["_config_dir"]).resolve()

    set_seed(run_cfg["seed"])
    device = get_device(run_cfg["device"])
    domain = Domain(**run_cfg["domain"])
    sampler = AllenCahnSampler(domain, device=device, seed=run_cfg["seed"])

    out_root = resolve_path(run_cfg["output_root"], config_dir)
    run_dir = out_root / exp_name
    fig_dir = run_dir / "figures"
    ensure_dir(fig_dir)

    save_yaml(run_dir / "resolved_config.yaml", run_cfg)

    model = MLP(run_cfg["model"]["layers"], activation=run_cfg["model"]["activation"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(run_cfg["train"]["adam_lr"]))

    batch = build_batch(sampler, run_cfg["train"], sampling=exp_cfg["sampling"])
    diffusion = float(run_cfg["pde"]["diffusion"])

    wcfg = AdaptiveWeightConfig(
        mode=exp_cfg["weight_mode"],
        ema=float(run_cfg["adaptive_weights"]["ema"]),
        min_value=float(run_cfg["adaptive_weights"]["min_value"]),
        max_value=float(run_cfg["adaptive_weights"]["max_value"]),
        update_every=int(run_cfg["adaptive_weights"]["update_every"]),
    )
    weights = AdaptiveWeights(wcfg)

    history = []
    ap_cfg = run_cfg["adaptive_points"]
    adaptive_round = 0
    before_adaptive = None
    after_adaptive = None

    t0 = time.perf_counter()
    for epoch in range(1, int(run_cfg["train"]["epochs"]) + 1):
        model.train()
        loss_terms = compute_losses(model, batch, diffusion=diffusion)
        lambdas = weights.update(loss_terms, epoch)
        total_loss = (
            lambdas["f"] * loss_terms["f"]
            + lambdas["ic"] * loss_terms["ic"]
            + lambdas["bc"] * loss_terms["bc"]
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch,
                "total": float(total_loss.detach().item()),
                "loss_f": float(loss_terms["f"].detach().item()),
                "loss_ic": float(loss_terms["ic"].detach().item()),
                "loss_bc": float(loss_terms["bc"].detach().item()),
                "lambda_f": float(lambdas["f"]),
                "lambda_ic": float(lambdas["ic"]),
                "lambda_bc": float(lambdas["bc"]),
                "n_f": int(batch["x_f"].shape[0]),
            }
        )

        if exp_cfg["adaptive_points"] and adaptive_round < int(ap_cfg["rounds"]):
            start_epoch = int(ap_cfg["start_epoch"])
            interval = int(ap_cfg["interval"])
            if epoch >= start_epoch and (epoch - start_epoch) % max(interval, 1) == 0:
                if before_adaptive is None:
                    before_adaptive = (
                        batch["x_f"].detach().cpu().numpy().reshape(-1),
                        batch["t_f"].detach().cpu().numpy().reshape(-1),
                    )
                x_new, t_new, res_stat = sample_adaptive_points(
                    model,
                    sampler,
                    diffusion=diffusion,
                    n_candidates=int(ap_cfg["n_candidates"]),
                    n_add=int(ap_cfg["n_add"]),
                )
                batch["x_f"] = torch.cat([batch["x_f"], x_new], dim=0)
                batch["t_f"] = torch.cat([batch["t_f"], t_new], dim=0)
                adaptive_round += 1
                after_adaptive = (
                    batch["x_f"].detach().cpu().numpy().reshape(-1),
                    batch["t_f"].detach().cpu().numpy().reshape(-1),
                )
                print(
                    f"[{exp_name}] epoch={epoch}: adaptive round {adaptive_round}, "
                    f"added {x_new.shape[0]} pts, top residual mean={res_stat:.3e}"
                )

        if epoch % int(run_cfg["train"]["log_every"]) == 0:
            h = history[-1]
            print(
                f"[{exp_name}] epoch={epoch:4d} total={h['total']:.3e} "
                f"f={h['loss_f']:.3e} ic={h['loss_ic']:.3e} bc={h['loss_bc']:.3e} n_f={h['n_f']}"
            )

    if bool(run_cfg["train"]["use_lbfgs"]):
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=int(run_cfg["train"]["lbfgs_steps"]))
        fixed_lmb = weights.get()

        def closure():
            lbfgs.zero_grad()
            terms = compute_losses(model, batch, diffusion=diffusion)
            total = fixed_lmb["f"] * terms["f"] + fixed_lmb["ic"] * terms["ic"] + fixed_lmb["bc"] * terms["bc"]
            total.backward()
            return total

        lbfgs.step(closure)

    elapsed = time.perf_counter() - t0
    ref_x, ref_t, ref_u = load_reference(resolve_path(run_cfg["data_path"], config_dir))
    pde_mse, ic_mse, bc_mse, rel_l2, u_pred = evaluate_metrics(
        model=model,
        sampler=sampler,
        diffusion=diffusion,
        n_eval_pde=int(run_cfg["train"]["n_eval_pde"]),
        x_ref=ref_x,
        t_ref=ref_t,
        u_ref=ref_u,
        device=device,
    )

    plot_loss_curves(history, fig_dir / "loss_curves.png")
    plot_solution_heatmap(ref_x, ref_t, u_pred, fig_dir / "solution_heatmap.png", u_ref=ref_u)
    plot_slices(
        ref_x,
        ref_t,
        u_pred,
        slice_times=run_cfg["plot"]["slice_times"],
        out_file=fig_dir / "solution_slices.png",
        u_ref=ref_u,
    )
    if before_adaptive is not None and after_adaptive is not None:
        plot_adaptive_points(before_adaptive, after_adaptive, fig_dir / "adaptive_points.png")

    np.savez(
        run_dir / "prediction.npz",
        x=ref_x,
        t=ref_t,
        u_pred=u_pred,
        u_ref=ref_u,
    )
    write_history_csv(run_dir / "history.csv", history)

    metrics = {
        "experiment": exp_name,
        "sampling": exp_cfg["sampling"],
        "weight_mode": exp_cfg["weight_mode"],
        "adaptive_points": bool(exp_cfg["adaptive_points"]),
        "epochs": int(run_cfg["train"]["epochs"]),
        "n_f_initial": int(run_cfg["train"]["n_f"]),
        "n_f_final": int(batch["x_f"].shape[0]),
        "pde_residual_mse": pde_mse,
        "ic_mse": ic_mse,
        "bc_mse": bc_mse,
        "relative_l2_error": rel_l2,
        "elapsed_sec": float(elapsed),
        "device": str(device),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def save_summary(out_root: Path, rows):
    csv_path = out_root / "summary_metrics.csv"
    if not rows:
        return
    keys = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    rows_sorted = sorted(rows, key=lambda x: x["relative_l2_error"])
    md_lines = [
        "| experiment | sampling | weight_mode | adaptive_points | PDE MSE | IC MSE | BC MSE | Rel L2 | elapsed(s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        md_lines.append(
            f"| {r['experiment']} | {r['sampling']} | {r['weight_mode']} | {int(r['adaptive_points'])} "
            f"| {r['pde_residual_mse']:.3e} | {r['ic_mse']:.3e} | {r['bc_mse']:.3e} "
            f"| {r['relative_l2_error']:.3e} | {r['elapsed_sec']:.1f} |"
        )
    (out_root / "summary_metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_report(report_path: Path, out_root: Path, rows, project_dir: Path):
    if not rows:
        return
    best = min(rows, key=lambda x: x["relative_l2_error"])
    lines = [
        "# Allen-Cahn PINN Report",
        "",
        "## 1. PDE 与问题设置",
        "- 方程: `u_t - D u_xx + 5(u^3-u)=0`, `D=1e-4`。",
        "- 区域: `x in [-1,1]`, `t in [0,1]`。",
        "- 初值: `u(x,0)=x^2 cos(pi x)`。",
        "- 边界: `u(-1,t)=u(1,t)=-1`。",
        "",
        "## 2. PINN 损失函数",
        "- `L_f = MSE(f(x,t))`，通过 PyTorch autograd 计算 `u_t,u_x,u_xx`。",
        "- `L_ic = MSE(u(x,0)-u0(x))`。",
        "- `L_bc = MSE(u(-1,t)+1) + MSE(u(1,t)+1)`。",
        "- 总损失: `L = lambda_f L_f + lambda_ic L_ic + lambda_bc L_bc`。",
        "",
        "## 3. 采样与自适应策略",
        "- `uniform`: 内部点规则网格采样。",
        "- `random`: 内部点随机采样。",
        "- `adaptive_points`: 预训练后在候选点上评估 `|f|`，加入残差最大的点继续训练。",
        "- `adaptive_weights`: 根据三项损失相对量级动态更新权重，并做 EMA 平滑。",
        "",
        "## 4. 四组实验结果",
        (out_root / "summary_metrics.md").read_text(encoding="utf-8").strip(),
        "",
        "## 5. 现象对比",
        f"- 以相对 L2 误差为主指标，最优方案为 `{best['experiment']}`，Rel L2={best['relative_l2_error']:.3e}。",
        "- `adaptive_points` 通常可进一步降低 PDE residual，并在高残差区域提升拟合。",
        "- `adaptive_weights` 可缓解损失项尺度不平衡，提升训练稳定性。",
        "",
        "## 6. 产物路径",
        f"- 总结指标: `{(out_root / 'summary_metrics.csv').relative_to(project_dir)}`",
        f"- 各实验图表: `{out_root.relative_to(project_dir)}/<experiment>/figures/`",
        "- 关键图包括: loss 曲线、热力图、截面曲线，以及自适应加点前后分布图。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    default_config = Path(__file__).resolve().with_name("config.yaml")
    parser = argparse.ArgumentParser(description="PyTorch PINN for 1D Allen-Cahn equation")
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--experiment", type=str, default=None, help="Run one experiment in config.experiments")
    parser.add_argument("--run_all", action="store_true", help="Run all experiments listed in run_order")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_dir = config_path.parents[2] if len(config_path.parents) >= 3 else config_path.parent
    cfg = load_yaml(config_path)
    cfg["_config_dir"] = str(config_path.parent)

    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    out_root = resolve_path(cfg["output_root"], Path(cfg["_config_dir"]))
    ensure_dir(out_root)

    if args.run_all:
        run_list = cfg["run_order"]
    elif args.experiment:
        run_list = [args.experiment]
    else:
        run_list = [cfg["run_order"][0]]

    results = []
    for exp_name in run_list:
        if exp_name not in cfg["experiments"]:
            raise KeyError(f"Experiment `{exp_name}` not found in config.")
        print(f"\n========== Running {exp_name} ==========")
        results.append(train_single(cfg, exp_name, cfg["experiments"][exp_name]))

    save_summary(out_root, results)
    report_path = Path(cfg["_config_dir"]) / "report.md"
    write_report(report_path, out_root, results, project_dir)
    print("\nAll done. Summary written to:")
    print(out_root / "summary_metrics.csv")
    print(report_path)


if __name__ == "__main__":
    main()
