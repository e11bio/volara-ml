[![tests](https://github.com/e11bio/volara-ml/actions/workflows/tests.yaml/badge.svg)](https://github.com/e11bio/volara-ml/actions/workflows/tests.yaml)
[![ruff](https://github.com/e11bio/volara-ml/actions/workflows/ruff.yaml/badge.svg)](https://github.com/e11bio/volara-ml/actions/workflows/ruff.yaml)
[![ty](https://github.com/e11bio/volara-ml/actions/workflows/ty.yaml/badge.svg)](https://github.com/e11bio/volara-ml/actions/workflows/ty.yaml)
<!-- [![codecov](https://codecov.io/gh/e11bio/volara-ml/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/e11bio/volara-ml) -->

# volara-ml
A plugin for volara that includes ML models and a `Predict` task

## Installation

```bash
pip install volara-ml         # base package
pip install volara-ml[torch]  # with PyTorch support
pip install volara-ml[jax]    # with Jax support
```

# Available blockwise operations:
- `Predict`: Model Prediction (requires `torch` or `jax` extra)
