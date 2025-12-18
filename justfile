export PYTHONOPTIMIZE := "1"
export HATCH_BUILD_CLEAN := "1"
export HYDRA_FULL_ERROR := "1"
export TQDM_DISABLE := "1"
export LEROBOT_TEST_DEVICE := "cuda"
export CUDA_VISIBLE_DEVICES := "1"
export HF_LEROBOT_HOME := "/nasa/3rd_party/lerobot"

_default:
    @just --list --unsorted

sync:
    uv sync --extra dev --extra test --extra smolvla --extra export

generate-config:
    ytt --file {{ justfile_directory() }}/rbyte/config/_templates/ \
        --output-files {{ justfile_directory() }}/rbyte/config/ \
        --output yaml \
        --ignore-unknown-comments \
        --strict

install-duckdb-extensions:
    uv run python -c "import duckdb; duckdb.connect().install_extension('spatial')"


train *ARGS: generate-config
    uv run examples/3_train_policy_rbyte.py \
        --config-path {{ justfile_directory() }}/rbyte/config \
        --config-name train.yaml \
        normalization=original \
        model=smolvla_train {{ ARGS }}

predict *ARGS: generate-config
    source .venv/bin/activate & uv run examples/2_evaluate_pretrained_policy_rbyte.py \
        --config-path {{ justfile_directory() }}/rbyte/config \
        --config-name predict.yaml {{ ARGS }}

convert *ARGS: generate-config
    uv run src/lerobot/policies/smolvla/conversion_utils_yaak.py \
        --config-path {{ justfile_directory() }}/rbyte/config \
        --config-name train.yaml {{ ARGS }}

export-onnx *ARGS:
    uv run src/lerobot/scripts/export_onnx.py \
        --config-path {{ justfile_directory() }}/rbyte/config \
        --config-name export/onnx.yaml \
        {{ ARGS }}

export-fast *ARGS: generate-config
    uv run src/lerobot/scripts/export_onnx_fast.py \
        --config-path {{ justfile_directory() }}/rbyte/config \
        --config-name export/onnx_fast.yaml \
        {{ ARGS }}