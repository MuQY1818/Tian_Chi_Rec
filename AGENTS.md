# Repository Guidelines

## Project Structure & Modules
- `src/data`: Exploration and preprocessing (`data_exploration.py`, `simple_preprocessing.py`).
- `src/features`: Feature engineering (`feature_engineering.py`).
- `src/models`: Baselines and ensembling (`simple_baseline_models.py`, others).
- `src/utils`: Helpers and config loaders.
- `config`: Project config (`config.py`) and dependencies (`requirements.txt`).
- `data`: `raw/`, `processed/`, `models/` outputs. Large artifacts are git‑ignored.
- Root helpers: `Makefile`, `main_large_scale.py`, `README.md`.

## Build, Test, Run
- `make install`: Install Python deps from `config/requirements.txt`.
- `make run-all`: Run exploration → preprocessing → features → models.
- Stepwise: `make run-exploration`, `make run-preprocessing`, `make run-feature-engineering`, `make run-models`.
- Large pipeline: `python main_large_scale.py --mode full --model_type ensemble` (see `README.md` for modes).
- Utilities: `make format` (Black + Flake8), `make test` (pytest), `make tree`, `make clean`.

## Coding Style & Naming
- Python, 4‑space indentation, UTF‑8, Unix newlines.
- Files/modules: `snake_case.py`; classes: `CamelCase`; functions/vars: `snake_case`.
- Docstrings: triple quotes with concise summaries; prefer type hints.
- Lint/format: Black + Flake8 via `make format` before pushing.

## Testing Guidelines
- Framework: `pytest` (run with `make test`).
- Location: create `tests/` with files named `test_*.py` mirroring `src/` structure.
- Focus: data transforms, feature outputs, and model utilities. Use small, deterministic fixtures.
- Artifacts: do not read large pickles in unit tests; mock or sample.

## Commit & PR Guidelines
- Style: Conventional Commits (e.g., `feat: ...`, `fix: ...`, `docs: ...`). Example in history: `feat: 初始化天池推荐算法项目`.
- Commits: small, scoped, imperative mood; include affected paths if helpful.
- PRs: clear description, rationale, and scope; link issues; include run command(s), logs/metrics tables (e.g., accuracy/AUC), and any data sampling notes. Screenshots for docs/plots.

## Security & Configuration
- Configure paths and switches in `config/config.py`; do not hardcode secrets or absolute paths.
- Data and large artifacts are ignored by `.gitignore` (e.g., `dataset/`, `*.pkl`, `*.csv`). Do not commit them.
- For deep models, document GPU/CPU assumptions and parameters used.

## Examples
- Run full pipeline: `make run-all`
- Format and lint before PR: `make format`
- Large‑scale run: `python main_large_scale.py --mode preprocess` then `--mode train`/`--mode predict`

