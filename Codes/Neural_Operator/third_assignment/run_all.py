from pathlib import Path

import numpy as np
import torch
import yaml

from cavity.data_loader import CavityDataModule
from cavity.make_report import create_word_report
from cavity.train_deeponet import train_deeponet
from cavity.train_fno import train_fno
from cavity.utils import ensure_dir, get_device, history_to_rows, resolve_path, save_csv, save_json, set_seed
from cavity.visualize import generate_all_plots


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    root = Path(__file__).resolve().parent
    config_path = root / "cavity" / "config.yaml"
    cfg = load_config(config_path)
    base = config_path.parent

    output_dir = resolve_path(cfg["paths"]["output_dir"], base)
    assets_dir = resolve_path(cfg["paths"]["report_assets"], base)
    report_path = resolve_path(cfg["paths"]["word_report"], base)
    ensure_dir(output_dir)
    ensure_dir(assets_dir)

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    cfg["_device"] = str(device)
    print(f"Using device: {device}")

    data = CavityDataModule(
        train_path=resolve_path(cfg["paths"]["train_mat"], base),
        test_path=resolve_path(cfg["paths"]["test_mat"], base),
        output_dir=output_dir,
    )
    data_summary = data.save_data_summary()

    deeponet_model, deeponet_pred, deeponet_history, deeponet_summary, deeponet_sample, _ = train_deeponet(
        data, cfg, device, output_dir
    )
    fno_model, fno_pred, fno_history, fno_summary, fno_sample, _ = train_fno(data, cfg, device, output_dir)

    del deeponet_model, fno_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary_rows = [deeponet_summary, fno_summary]
    per_sample_rows = deeponet_sample + fno_sample
    save_csv(output_dir / "metrics_summary.csv", summary_rows)
    save_csv(output_dir / "per_sample_errors.csv", per_sample_rows)
    save_csv(
        output_dir / "training_history.csv",
        history_to_rows(deeponet_history, "DeepONet") + history_to_rows(fno_history, "FNO"),
    )
    save_json(output_dir / "resolved_config.json", cfg)
    np.savez_compressed(
        output_dir / "predictions.npz",
        deeponet_pred=deeponet_pred,
        fno_pred=fno_pred,
        target=data.target("test"),
        x_2d=data.x_2d,
        y_2d=data.y_2d,
    )

    generate_all_plots(
        data=data,
        deeponet_pred=deeponet_pred,
        fno_pred=fno_pred,
        deeponet_history=deeponet_history,
        fno_history=fno_history,
        summary_rows=summary_rows,
        out_dir=assets_dir,
        sample_index=int(cfg["visualization"]["sample_index"]),
    )
    create_word_report(
        report_path=report_path,
        cfg=cfg,
        data_summary=data_summary,
        summary_rows=summary_rows,
        per_sample_rows=per_sample_rows,
        assets_dir=assets_dir,
        output_dir=output_dir,
    )

    print("Finished homework 3 artifacts.")
    print(f"Results: {output_dir}")
    print(f"Figures: {assets_dir}")
    print(f"Word report: {report_path}")


if __name__ == "__main__":
    main()
