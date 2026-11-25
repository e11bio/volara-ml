from pathlib import Path
from typing import Any

import pytest
from funlib.geometry import Coordinate

from volara_ml.models import JaxModel, Model, TorchModel

# Every saved model is a 1x1 convolution that multiplies its input by 0.5, so
# the tests state their expected outputs as literals.
CONFIG: dict[str, Any] = {
    "in_channels": 1,
    "out_channels": 1,
    "out_range": (0.0, 1.0),
    "min_input_shape": Coordinate(1, 1),
    "min_output_shape": Coordinate(1, 1),
    "min_step_shape": Coordinate(1, 1),
    "pred_size_growth": Coordinate(7, 7),
}


def save_torch_model(
    directory: Path, save_type: str, checkpoint: str | None = None
) -> TorchModel:
    """With a checkpoint, the saved module multiplies by 0.25 and the
    checkpoint file holds the halving weights."""
    torch = pytest.importorskip("torch")

    def conv(weight: float):
        module = torch.nn.Conv2d(1, 1, 1, bias=False)
        torch.nn.init.constant_(module.weight, weight)
        return module

    module = conv(0.5 if checkpoint is None else 0.25)
    save_path = directory / "model"
    if save_type == "jit":
        torch.jit.save(torch.jit.trace(module, torch.ones(1, 1, 8, 8)), save_path)
    else:
        torch.save(module, save_path)

    checkpoint_file = None
    if checkpoint is not None:
        checkpoint_file = directory / "checkpoint"
        weights = conv(0.5).state_dict()
        torch.save(
            {"model_state_dict": weights} if checkpoint == "checkpoint" else weights,
            checkpoint_file,
        )
    return TorchModel(save_path=save_path, checkpoint_file=checkpoint_file, **CONFIG)


def save_jax_model(directory: Path, save_type: str) -> JaxModel:
    jax = pytest.importorskip("jax")
    nn = pytest.importorskip("flax.linen")
    cloudpickle = pytest.importorskip("cloudpickle")

    class Halve(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = jax.numpy.transpose(x, (0, 2, 3, 1))
            x = nn.Conv(
                features=1,
                kernel_size=(1, 1),
                use_bias=False,
                kernel_init=nn.initializers.constant(0.5),
            )(x)
            return jax.numpy.transpose(x, (0, 3, 1, 2))

    module = Halve()
    params = module.init(jax.random.PRNGKey(0), jax.numpy.ones((1, 1, 8, 8)))
    model_path = directory / "model.pkl"
    model_path.write_bytes(cloudpickle.dumps(module))

    if save_type == "orbax":
        ocp = pytest.importorskip("orbax.checkpoint")
        params_path = directory / "ckpt"
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(str(params_path), params)
        checkpointer.close()
    else:
        from flax.serialization import to_bytes

        params_path = directory / "params.msgpack"
        params_path.write_bytes(to_bytes(params))
    return JaxModel(model_path=model_path, params_path=params_path, **CONFIG)


@pytest.fixture(scope="session")
def saved_model(tmp_path_factory):
    """`saved_model("torch-jit-checkpoint")`: the config of a halving model
    saved in that format, saved once per session."""
    savers = {"torch": save_torch_model, "jax": save_jax_model}
    models: dict[str, Model] = {}

    def get(kind: str) -> Model:
        if kind not in models:
            backend, *options = kind.split("-")
            models[kind] = savers[backend](tmp_path_factory.mktemp(kind), *options)
        return models[kind]

    return get
