import numpy as np
import pytest
from funlib.geometry import Coordinate

backends = pytest.mark.parametrize("kind", ["torch-pickle", "jax-msgpack"])


@backends
def test_eval_shapes_are_min_shapes_plus_pred_size_growth(saved_model, kind):
    model = saved_model(kind)

    assert model.context == Coordinate(0, 0)
    assert model.eval_input_shape == Coordinate(8, 8)
    assert model.eval_output_shape == Coordinate(8, 8)
    assert model.num_out_channels == [1]


@backends
@pytest.mark.parametrize(
    "out_range, floats",
    [((0.0, 1.0), [0.0, 0.2, 1.0]), ((-1.0, 1.0), [-1.0, -0.6, 1.0])],
)
def test_uint8_maps_linearly_onto_out_range(saved_model, kind, out_range, floats):
    model = saved_model(kind).model_copy(update={"out_range": out_range})

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
    saved_model, kind, out_range, floats
):
    model = saved_model(kind).model_copy(update={"out_range": out_range})

    converted = model.to_out_dtype(np.array(floats, dtype=np.float32))

    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted, [0, 0, 127, 255, 255])


@pytest.mark.parametrize(
    "kind",
    [
        "torch-jit",
        "torch-pickle",
        "torch-jit-checkpoint",
        "torch-pickle-checkpoint",
        "torch-pickle-statedict",
        "jax-orbax",
        "jax-msgpack",
    ],
)
def test_predict_applies_the_saved_weights(saved_model, kind):
    model = saved_model(kind)
    input_data = np.arange(64, dtype=np.float32).reshape(1, 1, 8, 8)

    with model.predict(model.select_device(0)) as predict:
        outputs = predict(input_data)

    assert len(outputs) == 1
    np.testing.assert_array_equal(
        outputs[0], np.arange(0, 32, 0.5, dtype=np.float32).reshape(1, 1, 8, 8)
    )
