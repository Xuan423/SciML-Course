# Repository Guidelines

## Project Structure & Module Organization
`Codes/` contains most runnable SciML examples:
- `forward/`: forward PINN solvers (`PINN-ODE-Forward`, `PINN_RAR`)
- `PINN_Inverse/`: inverse problems (`PINN_ODE`, `PINN_VIV`, `torch_viv`)
- `regression/`: regression baselines in `tf/`, `torch/`, and `RF/`
- `torch_parallel/`: distributed/data-parallel and ensemble PyTorch demos

`Data/` stores shared datasets such as `Data/Burgers/` and `Data/Allen-Cahn/`.  
Most submodules follow a simple layout: `net.py` (model), `dataset.py` (data helper, optional), solver script (`func.py` / `pinn_solver.py` / `torch_pinn.py`), and `Output/` for artifacts.

## Build, Test, and Development Commands
There is no single build system; run scripts directly from repository root.

```bash
python Codes/regression/torch/func.py
python Codes/forward/PINN-ODE-Forward/soft_bc/pinn_solver.py
python Codes/PINN_Inverse/torch_viv/torch_pinn.py
python Codes/torch_parallel/data_parallel/func.py
```

Use these as smoke tests after edits in the corresponding module.  
Dependencies are managed manually in this repo; typical packages include `numpy`, `scipy`, `matplotlib`, `torch`, and `tensorflow` (v1 compatibility APIs are used in several scripts).

## Coding Style & Naming Conventions
- Python with 4-space indentation.
- `snake_case` for functions/variables; `CamelCase` for classes (`FNN`, `DNN`).
- Keep naming consistent with existing files: `net.py`, `dataset.py`, `func.py`, `pinn_*`.
- Preserve reproducibility behavior (`np.random.seed`, `torch.manual_seed`, `tf.set_random_seed`) unless intentionally changing experiment design.

## Testing Guidelines
No formal unit-test suite is currently present. Validate by executing the impacted script end-to-end and checking:
- training loss decreases without runtime errors;
- output artifacts in `Output/` are generated correctly;
- key metrics/logged errors stay reasonable versus prior runs.

For new functionality, include a minimal runnable script in the same folder.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit messages (for example: `update slides`, `Add regression ablation report`). Keep commits focused and scoped by module.

PRs should include:
- summary of what changed and why;
- affected paths (for example `Codes/PINN_Inverse/torch_viv/`);
- exact run commands used for verification;
- key result snapshots (metrics or plots) when behavior changes.
