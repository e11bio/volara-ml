[![tests](https://github.com/e11bio/volara-ml/actions/workflows/tests.yaml/badge.svg)](https://github.com/e11bio/volara-ml/actions/workflows/tests.yaml)
[![lint](https://github.com/e11bio/volara-ml/actions/workflows/lint.yaml/badge.svg)](https://github.com/e11bio/volara-ml/actions/workflows/lint.yaml)
[![docs](https://github.com/e11bio/volara-ml/actions/workflows/docs.yaml/badge.svg)](https://e11bio.github.io/volara-ml)
[![publish](https://github.com/e11bio/volara-ml/actions/workflows/publish.yaml/badge.svg)](https://pypi.org/project/volara-ml/)

# volara-ml
A plugin for volara that includes ML models and a `Predict` task

**Documentation:** https://e11bio.github.io/volara-ml

## Install

```bash
pip install volara-ml           # base package        (or: uv add volara-ml)
pip install "volara-ml[torch]"  # with PyTorch support (or: uv add "volara-ml[torch]")
pip install "volara-ml[jax]"    # with Jax support     (or: uv add "volara-ml[jax]")
```

# Available blockwise operations:
- `Predict`: Model Prediction (requires `torch` or `jax` extra)

## Contributing

The commands below are the ones CI runs, so a green local run means a green PR.

```bash
git clone https://github.com/e11bio/volara-ml
cd volara-ml
uv sync --extra torch --extra jax                   # the package, dev tools + both backends (from uv.lock)
uv run pytest                                       # tests with coverage
uv run pytest --no-cov tests/<file>.py              # one file, without the coverage floor
uv run --only-group dev ruff check . && uv run --only-group dev ruff format --check .
uv run ty check --error-on-warning                  # types
```

Building the docs needs [pandoc](https://pandoc.org/installing.html) on the `PATH`; the example scripts are converted to notebooks before Sphinx runs:

```bash
uv sync --group docs --extra torch --extra jax
uv run python -m ipykernel install --name volara_env --user
uv run jupytext --to notebook ./examples/cremi/cremi.py
uv run jupytext --to notebook ./examples/jax/identity.py
cp -r examples docs/source
find docs/source/examples -name '*.py' -delete
uv run sphinx-build -W docs/source docs/build/html -b html
```

`uv.lock` pins the development environment; the `lock-upgrade` workflow refreshes it monthly and opens a pull request.
