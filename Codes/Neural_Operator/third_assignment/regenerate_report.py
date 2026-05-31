import csv
from pathlib import Path

import numpy as np
import yaml

from cavity.data_loader import CavityDataModule
from cavity.make_report import create_word_report
from cavity.utils import resolve_path
from cavity.visualize import TIMES_FONT, generate_all_plots


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", None):
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def split_history(rows, model):
    result = []
    for row in rows:
        if row.get("model") != model:
            continue
        item = {k: v for k, v in row.items() if k != "model"}
        if "epoch" in item:
            item["epoch"] = int(float(item["epoch"]))
        result.append(item)
    return result


def main():
    root = Path(__file__).resolve().parent
    config_path = root / "cavity" / "config.yaml"
    cfg = load_config(config_path)
    base = config_path.parent
    output_dir = resolve_path(cfg["paths"]["output_dir"], base)
    assets_dir = resolve_path(cfg["paths"]["report_assets"], base)
    report_path = resolve_path(cfg["paths"]["word_report"], base)
    cfg["_device"] = "regenerated from saved outputs"

    data = CavityDataModule(
        train_path=resolve_path(cfg["paths"]["train_mat"], base),
        test_path=resolve_path(cfg["paths"]["test_mat"], base),
        output_dir=output_dir,
    )
    data_summary = read_csv(output_dir / "cavity_mat_summary.csv")
    summary_rows = read_csv(output_dir / "metrics_summary.csv")
    per_sample_rows = read_csv(output_dir / "per_sample_errors.csv")
    history_rows = read_csv(output_dir / "training_history.csv")
    pred = np.load(output_dir / "predictions.npz")

    deeponet_history = split_history(history_rows, "DeepONet")
    fno_history = split_history(history_rows, "FNO")
    generate_all_plots(
        data=data,
        deeponet_pred=pred["deeponet_pred"],
        fno_pred=pred["fno_pred"],
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
    print(f"Regenerated figures with font: {TIMES_FONT}")
    print(f"Updated Word report: {report_path}")


if __name__ == "__main__":
    main()
