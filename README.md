[![tests](https://github.com/e11bio/volara-ml/actions/workflows/tests.yaml/badge.svg)](https://github.com/e11bio/volara-ml/actions/workflows/tests.yaml)
[![lint](https://github.com/e11bio/volara-ml/actions/workflows/lint.yaml/badge.svg)](https://github.com/e11bio/volara-ml/actions/workflows/lint.yaml)
[![docs](https://github.com/e11bio/volara-ml/actions/workflows/docs.yaml/badge.svg)](https://e11bio.github.io/volara-ml)
[![publish](https://github.com/e11bio/volara-ml/actions/workflows/publish.yaml/badge.svg)](https://pypi.org/project/volara-ml/)

# volara-ml
A plugin for volara that includes ML models and a `Predict` task

**Documentation:** https://e11bio.github.io/volara-ml

## Install

```bash
pip install volara-ml         # base package
pip install volara-ml[torch]  # with PyTorch support
pip install volara-ml[jax]    # with Jax support
```

# Available blockwise operations:
- `Predict`: Model Prediction (requires `torch` or `jax` extra)

## Contributing

The commands below are the ones CI runs, so a green local run means a green PR.

```bash
git clone https://github.com/e11bio/volara-ml
cd volara-ml
uv sync --extra torch --extra jax                   # dev tools + both backends (from uv.lock)
uv pip install -e .                                 # the package itself (no build backend yet)
uv run pytest                                       # tests with coverage
uv run pytest --no-cov tests/<file>.py              # one file, without the coverage floor
uv run --only-group dev ruff check . && uv run --only-group dev ruff format --check .
uv run ty check                                     # types
# docs: see .github/workflows/docs.yaml (notebooks are converted before sphinx-build)
```

`uv.lock` pins the development environment; the `lock-upgrade` workflow refreshes it monthly and opens a pull request.
