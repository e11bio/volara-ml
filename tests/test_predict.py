import numpy as np
import pytest
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds
from saved_models import save_jax_model, save_torch_model
from volara.datasets import Raw
from volara.logging import get_log_basedir, set_log_basedir

from volara_ml.blockwise import Predict


@pytest.fixture
def log_dir(tmp_path):
    """Point volara's worker logs and block-done cache at `tmp_path`."""
    previous = get_log_basedir()
    yield set_log_basedir(tmp_path / "volara_logs")
    set_log_basedir(previous)


@pytest.mark.parametrize(
    "save_model",
    [
        pytest.param(save_torch_model, id="torch"),
        pytest.param(save_jax_model, id="jax"),
    ],
)
def test_predict_writes_model_output_as_uint8_over_every_block(
    save_model, tmp_path, log_dir
):
    # Odd values: read as v / 255, halved by the model, and written with
    # out_range (0, 1) as uint8, each voxel lands on v / 2 = k + 0.5 -> k.
    data = np.tile(np.arange(1, 256, 2, dtype=np.uint8), 2).reshape(16, 16)
    raw_array = prepare_ds(
        store=str(tmp_path / "data.zarr/raw"),
        shape=data.shape,
        offset=Coordinate(0, 0),
        voxel_size=Coordinate(1, 1),
        axis_names=["y", "x"],
        units=["nm", "nm"],
        dtype=np.uint8,
        mode="w",
    )
    raw_array[:] = data

    prediction = Raw(store=tmp_path / "data.zarr/prediction")
    task = Predict(
        checkpoint=save_model(tmp_path),
        in_data=Raw(store=tmp_path / "data.zarr/raw", scale_shift=(1 / 255, 0)),
        out_data=[prediction],
    )
    task.run_blockwise(multiprocessing=False)

    written = prediction.array("r")
    assert written.dtype == np.uint8
    np.testing.assert_array_equal(
        written[:], np.tile(np.arange(128, dtype=np.uint8), 2).reshape(1, 16, 16)
    )
