from pathlib import Path

import torch
import torch.nn.functional as F

from .evaluate import evaluate_predictions, predict_fno
from .fno2d import FNO2d
from .utils import count_parameters, ensure_dir


def train_fno(data, cfg, device, output_dir):
    train_cfg = cfg["training"]
    model_cfg = cfg["fno"]
    output_dir = Path(output_dir)
    ensure_dir(output_dir / "models")

    train_loader = data.fno_loader("train", train_cfg["batch_size"], shuffle=True)
    test_loader = data.fno_loader("test", train_cfg["batch_size"], shuffle=False)
    model = FNO2d(
        in_channels=3,
        out_channels=2,
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        width=model_cfg["width"],
        depth=model_cfg["depth"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.6)

    history = []
    best = float("inf")
    best_path = output_dir / "models" / "fno_best.pt"
    epochs = int(train_cfg["fno_epochs"])
    eval_every = int(train_cfg["eval_every"])

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.shape[0]
            n_seen += x.shape[0]
        scheduler.step()

        row = {"epoch": epoch, "train_mse": total / max(n_seen, 1)}
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            pred = predict_fno(model, data, test_loader, device)
            summary, _, _ = evaluate_predictions("FNO", pred, data.target("test"))
            row["test_relative_l2"] = summary["relative_l2_mean"]
            row["test_mse"] = summary["mse"]
            if summary["relative_l2_mean"] < best:
                best = summary["relative_l2_mean"]
                torch.save(model.state_dict(), best_path)
            print(
                f"[FNO] epoch={epoch:04d} train_mse={row['train_mse']:.3e} "
                f"test_rel_l2={row['test_relative_l2']:.3e}"
            )
        history.append(row)

    model.load_state_dict(torch.load(best_path, map_location=device))
    pred = predict_fno(model, data, test_loader, device)
    summary, per_sample, raw_metrics = evaluate_predictions("FNO", pred, data.target("test"))
    summary["parameters"] = count_parameters(model)
    summary["best_model"] = str(best_path)
    return model, pred, history, summary, per_sample, raw_metrics
