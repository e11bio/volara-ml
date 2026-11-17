import numpy as np
import pytest
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds
from volara.datasets import Raw
from volara.logging import get_log_basedir, set_log_basedir

from volara_ml.blockwise import Predict


@pytest.fixture
def log_dir(tmp_path):
    """Point volara's worker logs and block-done cache at `tmp_path`."""
    previous = get_log_basedir()
    yield set_log_basedir(tmp_path / "volara_logs")
    set_log_basedir(previous)


@pytest.mark.parametrize("kind", ["torch-pickle", "jax-msgpack"])
def test_predict_writes_model_output_as_uint8_over_every_block(
    saved_model, kind, tmp_path, log_dir
):
    # Odd values: read as v / 255, halved by the model, and written with
    # out_range (0, 1) as uint8, each voxel lands on v / 2 = k + 0.5 -> k.
    # The model evaluates 8x8, so 8x16 is two blocks.
    raw_array = prepare_ds(
        store=str(tmp_path / "data.zarr/raw"),
        shape=(8, 16),
        offset=Coordinate(0, 0),
        voxel_size=Coordinate(1, 1),
        axis_names=["y", "x"],
        units=["nm", "nm"],
        dtype=np.uint8,
        mode="w",
    )
    raw_array[:] = np.arange(1, 256, 2, dtype=np.uint8).reshape(8, 16)

    prediction = Raw(store=tmp_path / "data.zarr/prediction")
    task = Predict(
        checkpoint=saved_model(kind),
        in_data=Raw(store=tmp_path / "data.zarr/raw", scale_shift=(1 / 255, 0)),
        out_data=[prediction],
    )
    task.run_blockwise(multiprocessing=False)

    written = prediction.array("r")
    assert written.dtype == np.uint8
    np.testing.assert_array_equal(
        written[:], np.arange(128, dtype=np.uint8).reshape(1, 8, 16)
    )
