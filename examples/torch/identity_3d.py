# %% [markdown]
# # Torch 3D identity example
# Reproduces the structure of a 3D prediction pipeline (like the user's
# syn_detection_M3 script) using a dummy 1-channel in / 1-channel out 1x1x1
# 3D conv. Verifies that a 10x10x10 input volume yields a 10x10x10 output
# volume (with a leading channel axis of size 1).

# %%
import shutil
from pathlib import Path

import numpy as np
import torch
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds

from volara.datasets import Raw
from volara_ml.blockwise import Predict
from volara_ml.models import TorchModel


# %% [markdown]
# ## 1. Create a synthetic 10x10x10 uint8 input volume

# %%
work_dir = Path("_torch_identity_3d_workdir")
if work_dir.exists():
    shutil.rmtree(work_dir)
work_dir.mkdir(parents=True)

rng = np.random.default_rng(0)
data = rng.integers(0, 256, size=(10, 10, 10), dtype=np.uint8)

raw_array = prepare_ds(
    store=str(work_dir / "data.zarr/raw"),
    shape=data.shape,
    offset=Coordinate(0, 0, 0),
    voxel_size=Coordinate(1, 1, 1),
    axis_names=["z", "y", "x"],
    units=["nm", "nm", "nm"],
    dtype=np.uint8,
    mode="w",
)
raw_array[:] = data


# %% [markdown]
# ## 2. Define and save a dummy 1x1x1 3D conv model (1 ch in -> 1 ch out)
# A 1x1x1 conv has no context loss, so eval input shape == eval output shape.

# %%
model = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1)

# TorchModel.model() uses torch.load(save_path) and expects a full nn.Module
save_path = work_dir / "model.pt"
torch.save(model, save_path)

# Optional checkpoint with state_dict (matches the pattern in the user's script)
checkpoint_path = work_dir / "model_checkpoint"
torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)


# %% [markdown]
# ## 3. Configure TorchModel + Predict

# %%
torch_model = TorchModel(
    in_channels=1,
    out_channels=1,
    min_input_shape=Coordinate(10, 10, 10),
    min_output_shape=Coordinate(10, 10, 10),
    min_step_shape=Coordinate(1, 1, 1),
    out_range=(-1.0, 1.0),
    save_path=save_path,
    checkpoint_file=checkpoint_path,
    pred_size_growth=Coordinate(0, 0, 0),
)

raw_in = Raw(store=work_dir / "data.zarr/raw", scale_shift=(1 / 255, 0))
raw_out = Raw(store=work_dir / "data.zarr/prediction")

predict_task = Predict(
    in_data=raw_in,
    out_data=[raw_out],
    out_array_dtype=np.dtype(np.float32),
    checkpoint=torch_model,
    num_workers=1,
)


# %% [markdown]
# ## 4. Run blockwise prediction and verify shapes

# %%
predict_task.drop()
predict_task.run_blockwise(multiprocessing=False)

input_data = raw_in.array("r")[:]
output_data = raw_out.array("r")[:]

print(f"input shape:  {input_data.shape}")
print(f"output shape: {output_data.shape}")

assert input_data.shape == (10, 10, 10), (
    f"Expected input shape (10, 10, 10), got {input_data.shape}"
)
assert output_data.shape == (1, 10, 10, 10), (
    f"Expected output shape (1, 10, 10, 10) "
    f"(channel-first 3D), got {output_data.shape}"
)

print("Success! 10x10x10 input -> (10, 10, 10) output as expected.")

# Clean up
shutil.rmtree(work_dir)
shutil.rmtree("volara_logs", ignore_errors=True)
