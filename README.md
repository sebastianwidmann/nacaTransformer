# nacaTransformer

A Vision Transformer, implemented in [JAX](https://github.com/google/jax) /
[Flax](https://github.com/google/flax), for surrogate modeling of steady-state
aerodynamic flow fields (pressure, velocity) around NACA airfoils and bluff
bodies from RANS CFD simulations. The encoder ingests a rasterised
signed-distance/Mach-number field and the decoder predicts the corresponding
flow field over the same grid.

Developed at the Technical University of Munich, Department of Aerospace and
Geodesy.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- An NVIDIA GPU with CUDA 12 is strongly recommended for training and
  preprocessing (CPU works but is very slow)

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
from the repository root:

```bash
uv sync
```

This creates a `.venv` and installs the pinned dependencies from `uv.lock`.
Run any command in that environment with `uv run <command>`, e.g.
`uv run python -m src.main ...`, or activate the environment directly with
`source .venv/bin/activate`.

### GPU support

`uv sync` installs a CPU-only `jaxlib`. For GPU-accelerated training, install
the matching CUDA 12 build afterwards (version must match the `jax` pin in
`pyproject.toml`):

```bash
uv pip install "jax[cuda12]==0.4.26"
```

### Optional: GPU-accelerated bluff-body preprocessing

`src/preprocessing/interpolation.py` uses [faiss](https://github.com/facebookresearch/faiss)
for GPU k-nearest-neighbour search and [cuSpatial](https://github.com/rapidsai/cuspatial)
for point-in-polygon tests when building the bluff-body dataset
(`config.trainer = 'preprocess'`). These are not included in the default
dependency set because they require an NVIDIA GPU, CUDA 12, and RAPIDS
packages hosted on a separate index. They are **not** required for training
or inference. Install them with:

```bash
uv pip install --extra-index-url https://pypi.nvidia.com \
    faiss-gpu-cu12 cudf-cu12 cuspatial-cu12
```

## Usage

All entry points are driven by an [`ml_collections`](https://github.com/google/ml_collections)
config file; see `src/config.py` for every available option and its default.
Any field can be overridden on the command line.

```bash
# Convert raw CFD output (.vtu/.stl) into a TFDS dataset of TFRecords
uv run python -m src.main --config=src/config.py --config.trainer=preprocess

# Train the transformer
uv run python -m src.main --config=src/config.py --config.trainer=train \
    --config.dataset=/path/to/tfds/dataset \
    --config.output_dir=/path/to/outputs

# Fine-tune a trained checkpoint on the bluff-body dataset
uv run python -m src.main --config=src/config.py --config.trainer=train \
    --config.fine_tune.enable=True \
    --config.fine_tune.dataset=/path/to/bluff/tfds/dataset \
    --config.fine_tune.checkpoint_dir=/path/to/checkpoint
```

`config.trainer` selects the mode (`preprocess`, `train`, or `inference` —
the latter is not yet implemented). Preprocessing requires the optional GPU
dependencies described above.

## Project structure

```
src/
├── main.py                  # CLI entry point (absl flags + ml_collections config)
├── config.py                # Default configuration (model, training, preprocessing)
├── train.py                 # Training / evaluation loop, checkpointing
├── transformer/              # Vision Transformer model (encoder, decoder, layers)
├── preprocessing/             # Converts raw CFD output (.vtu/.stl) into TFDS datasets
└── utilities/                 # Dataset filtering, plotting, LR schedulers, misc scripts

Dockerfile-base                # Base image: Python + uv-managed dependencies
Dockerfile-nacavit              # Runtime image built on top of the base image
NACA_transformer_structure.json # Reference parameter tree of a trained checkpoint
```

Some scripts in `src/utilities/` (`airfoilMNIST-*.py`, `postprocessing.py`,
`visualize_normalization_comparison.py`) are standalone tools used for
dataset inspection and thesis figure generation; they are not part of the
`main.py` pipeline and may contain hard-coded paths.

## Docker

```bash
docker build -f Dockerfile-base -t nacavit-base .
docker build -f Dockerfile-nacavit -t nacavit .
docker run --gpus all nacavit
```

## Known limitations

- `config.py` and some `src/utilities/` scripts contain filesystem paths
  from the original development machine; override them via `--config.*`
  flags or edit `src/config.py` for your environment.
- `inference` mode (`config.trainer = 'inference'`) is not implemented yet.
- The pinned `jax`/`jaxlib`/`flax`/`optax`/`orbax-checkpoint` versions are
  held together deliberately (see the comment in `pyproject.toml`) because
  `src/train.py` relies on `jax.sharding.PositionalSharding`, which was
  removed in later JAX releases. Upgrading these packages requires updating
  `src/train.py` accordingly.
