"""Builders for the saved models the tests load through volara_ml."""

from pathlib import Path

import pytest
from funlib.geometry import Coordinate

from volara_ml.models import JaxModel, TorchModel

# Every model saved here is a 1x1 convolution that multiplies its input by
# 0.5, so the tests state their expected outputs as literals.


def save_torch_model(
    directory: Path,
    save_type: str = "pickle",
    checkpoint: str | None = None,
    out_range: tuple[float, float] = (0.0, 1.0),
) -> TorchModel:
    """Save a halving model with torch and return the `TorchModel` that loads it.

    With a checkpoint, the saved module multiplies by 0.25 and the checkpoint
    file holds the halving weights.
    """
    torch = pytest.importorskip("torch")

    def conv(weight: float):
        module = torch.nn.Conv2d(1, 1, 1, bias=False)
        torch.nn.init.constant_(module.weight, weight)
        return module

    module = conv(0.5 if checkpoint is None else 0.25)
    if save_type == "jit":
        save_path = directory / "model.jit"
        torch.jit.save(torch.jit.trace(module, torch.ones(1, 1, 8, 8)), save_path)
    else:
        save_path = directory / "model.pth"
        torch.save(module, save_path)

    checkpoint_file = None
    if checkpoint is not None:
        checkpoint_file = directory / "checkpoint.pth"
        weights = conv(0.5).state_dict()
        torch.save(
            {"model_state_dict": weights} if checkpoint == "wrapped" else weights,
            checkpoint_file,
        )

    return TorchModel(
        in_channels=1,
        min_input_shape=Coordinate(1, 1),
        min_output_shape=Coordinate(1, 1),
        min_step_shape=Coordinate(1, 1),
        out_channels=1,
        out_range=out_range,
        save_path=save_path,
        checkpoint_file=checkpoint_file,
        pred_size_growth=Coordinate(7, 7),
    )


def save_jax_model(
    directory: Path,
    save_type: str = "msgpack",
    out_range: tuple[float, float] = (0.0, 1.0),
) -> JaxModel:
    """Save a halving flax module and its parameters; return the `JaxModel`."""
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
    with open(model_path, "wb") as f:
        cloudpickle.dump(module, f)

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

    return JaxModel(
        in_channels=1,
        min_input_shape=Coordinate(1, 1),
        min_output_shape=Coordinate(1, 1),
        min_step_shape=Coordinate(1, 1),
        out_channels=1,
        out_range=out_range,
        model_path=model_path,
        params_path=params_path,
        pred_size_growth=Coordinate(7, 7),
    )
