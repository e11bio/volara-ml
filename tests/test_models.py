import numpy as np
import pytest
from funlib.geometry import Coordinate
from saved_models import save_jax_model, save_torch_model

backends = pytest.mark.parametrize(
    "save_model",
    [
        pytest.param(save_torch_model, id="torch"),
        pytest.param(save_jax_model, id="jax"),
    ],
)


@backends
def test_eval_shapes_are_min_shapes_plus_pred_size_growth(save_model, tmp_path):
    model = save_model(tmp_path)

    assert model.context == Coordinate(0, 0)
    assert model.eval_input_shape == Coordinate(8, 8)
    assert model.eval_output_shape == Coordinate(8, 8)
    assert model.num_out_channels == [1]


@backends
@pytest.mark.parametrize(
    "out_range, floats",
    [((0.0, 1.0), [0.0, 0.2, 1.0]), ((-1.0, 1.0), [-1.0, -0.6, 1.0])],
)
def test_uint8_maps_linearly_onto_out_range(save_model, out_range, floats, tmp_path):
    model = save_model(tmp_path, out_range=out_range)

    converted = model.from_uint8(np.array([0, 51, 255], dtype=np.uint8))

    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted, floats, rtol=0, atol=1e-6)


@backends
@pytest.mark.parametrize(
    "out_range, floats",
    [
        ((0.0, 1.0), [-0.5, 0.0, 0.5, 1.0, 1.5]),
        ((-1.0, 1.0), [-2.0, -1.0, 0.0, 1.0, 2.0]),
    ],
)
def test_to_out_dtype_maps_out_range_onto_uint8_and_clips(
    save_model, out_range, floats, tmp_path
):
    model = save_model(tmp_path, out_range=out_range)

    converted = model.to_out_dtype(np.array(floats, dtype=np.float32))

    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted, [0, 0, 127, 255, 255])


@pytest.mark.parametrize(
    "save_model, save_options",
    [
        pytest.param(save_torch_model, {"save_type": "jit"}, id="torch-jit"),
        pytest.param(save_torch_model, {"save_type": "pickle"}, id="torch-pickle"),
        pytest.param(
            save_torch_model,
            {"save_type": "jit", "checkpoint": "wrapped"},
            id="torch-jit-checkpoint",
        ),
        pytest.param(
            save_torch_model,
            {"save_type": "pickle", "checkpoint": "wrapped"},
            id="torch-pickle-checkpoint",
        ),
        pytest.param(
            save_torch_model,
            {"save_type": "pickle", "checkpoint": "plain"},
            id="torch-pickle-plain-state-dict",
        ),
        pytest.param(save_jax_model, {"save_type": "orbax"}, id="jax-orbax"),
        pytest.param(save_jax_model, {"save_type": "msgpack"}, id="jax-msgpack"),
    ],
)
def test_predict_applies_the_saved_weights(save_model, save_options, tmp_path):
    model = save_model(tmp_path, **save_options)
    input_data = np.arange(64, dtype=np.float32).reshape(1, 1, 8, 8)

    with model.predict(model.select_device(0)) as predict:
        outputs = predict(input_data)

    assert len(outputs) == 1
    np.testing.assert_array_equal(
        outputs[0], np.arange(0, 32, 0.5, dtype=np.float32).reshape(1, 1, 8, 8)
    )
