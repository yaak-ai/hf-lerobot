# Architectural decisions: rbyte dataset integration

## How to Run (from `justfile`)

The `justfile` is the primary entrypoint. All commands are run with `just <command>` from the repository root.

### Commands

Install && generate configs:

```shell
# Install all dependencies (dev, test, smolvla, yaak extras)
just sync                 
# Install DuckDB spatial extension (required once)
just install-duckdb-extensions
# Re-generate Hydra YAML configs from ytt templates 
just generate-config        
```

Dependencies (`pyproject.toml`)

New `yaak` extra:
```toml
yaak = ["rbyte[geo,jpeg,yaak]==0.30.0", "duckdb>=1.2.2", "polars>=1.0.0", "matplotlib>=3.9.0,<4.0.0"]
```

Python version pinned to `>=3.12,<3.13` (up from `>=3.10`).


#### Compute train dataset stats to normalize speed and waypoints

```shell
# Done just once
just rbyte2lerobot-stats
```

Computes per-feature statistics (min, max, mean, std, q01, q99) from the rbyte dataset and writes them to a JSON file. This JSON is required before training (normalization layers read it).

```
Configs (Hydra)
┌─────────────────────────────────────────────────────────────┐
│  rbyte/config/stat.yaml                                     │
│    └── /paths:     yaak/default  ─── lerobot_stats path ───┼──► output path
│    └── /datamodule: yaak/train                              │
│            └── /dataset: yaak/train_clip (ytt-generated)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ instantiate(cfg.datamodule.dataset)
                               ▼
Script
┌──────────────────────────────────────────────────────────────┐
│  conversion_utils_yaak.py :: compute_rbyte2lerobot_stats()   │
│                                                              │
│  action          ← gas / brake / steering columns           │
│    → min, max, mean, std, q01, q99  (shape: 3)              │
│                                                              │
│  observation.state  ← waypoints/xy_normalized (10 × 2)      │
│    → tiled min/max/mean/std  (shape: 20)                     │
│                                                              │
│  observation.state.vehicle  ← VehicleMotion/speed           │
│    → min, max, mean, std, q01, q99  (shape: 1)              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
Output
┌──────────────────────────────────────────────────────────────┐
│  paths.lerobot_stats              │
│  {                                                           │
│    "action":                    { min, max, mean, std, ... } │
│    "observation.state":         { min, max, mean, std, ... } │
│    "observation.state.vehicle": { min, max, mean, std, ... } │
│  }                                                           │
│                         consumed by just train               │
└──────────────────────────────────────────────────────────────┘
```

#### Train the model

```shell
just train
# calls the below under the hood
# uv run examples/training/train_policy_rbyte.py \
#     --config-path <repo>/rbyte/config \
#     --config-name train.yaml \
#     normalization=original \
#     model=smolvla_train [ARGS]
```

Trains SmolVLA from a pretrained VLM checkpoint using the rbyte dataloader. Model config and normalization strategy can be overridden via Hydra CLI.


```shell
Configs (Hydra)
┌──────────────────────────────────────────────────────────────────────────┐
│  rbyte/config/train.yaml                                                 │
│    └── experiment/yaak/finetune.yaml                                     │
│          ├── /datamodule:     yaak/train    ── batch_size, num_workers   │
│          │     └── /dataset:  yaak/train_clip  (ytt-generated)           │
│          ├── /datamodule_val: yaak/val                                   │
│          ├── /paths:          yaak/default  ── lerobot_stats path        |
│          ├── /normalization:  original      ── IDENTITY/MIN_MAX per feat │
│          └── /model:          smolvla_train ── all hyperparameters       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
Inputs
┌─────────────────────────────┐   ┌──────────────────────────────────────┐
│  rbyte.Dataset (train_clip) │   │  dataset_stats.json                  │
│    raw drives on disk       │   │  { action, obs.state,                │
│    DuckDB filter + clip     │   │    obs.state.vehicle }               │
│    → rbyte.Batch per step   │   │  → normalization layer init          │
└──────────────┬──────────────┘   └──────────────────┬───────────────────┘
               │                                      │
               ▼                                      ▼
Script: train_policy_rbyte.py → scripts/train_yaak.py::train()
┌──────────────────────────────────────────────────────────────────────────┐
│  make_policy_yaak()          SmolVLAPolicy (from pretrained VLM)         │
│  make_pre_post_processors()  normalize / rename observations             │
│                                                                          │
│  per step:                                                               │
│    rbyte.Batch                                                           │
│      → __getbatch__()        rbyte format → lerobot dict                 │
│      → preprocessor()        normalize + move to device                  │
│      → policy.forward()      loss + loss_dict (+ tracking_callback)      │
│      → update_policy()       backward + optimizer step                   │
│                                                                          │
│  every eval_freq steps:                                                  │
│    → eval_policy_yaak()      val split forward pass                      │
│    → metric_accum_callback() per-drive GT vs pred plots                  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
Outputs
┌──────────────────────────────────────────────────────────────────────────┐
│  outputs/train/<job_name>/checkpoints/<step>/pretrained_model/           │
│    model.safetensors   policy weights                                    │
│    config.json         SmolVLAConfig + TrainPipelineConfig               │
│    preprocessor/       normalization state                               │
│  W&B run               loss curves, per-action nnz stats, eval images   │
└──────────────────────────────────────────────────────────────────────────┘
```

#### Evaluate the model

```shell
just predict
# Calls the evaluate_pretrained_policy_rbyte under the hood
# uv run examples/training/evaluate_pretrained_policy_rbyte.py \
#     --config-path <repo>/rbyte/config \
#     --config-name predict.yaml [ARGS]
```

Loads a trained policy artifact from W&B, runs inference over a predict split, and writes predictions to a parquet file in the [reye format](#reye-output-format).

```shell
Configs (Hydra)
┌──────────────────────────────────────────────────────────────────────────┐
│  rbyte/config/predict.yaml                                               │
│    └── inference/yaak/default.yaml                                       │
│          ├── /model:      smolvla      ── W&B artifact ref + stats path  │
│          ├── /datamodule: yaak/predict ── predict split dataset          │
│          └── /paths:      yaak/default ── lerobot_stats path             │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
Inputs
┌─────────────────────────────┐   ┌──────────────────────────────────────┐
│  rbyte.Dataset (predict)    │   │  W&B artifact                        │
│    raw drives (holdout)     │   │    model.safetensors  policy weights  │
│    DuckDB filter + clip     │   │    train_config.json  SmolVLAConfig   │
│    → rbyte.Batch per step   │   │    preprocessor/      norm state      │
└──────────────┬──────────────┘   └──────────────────┬───────────────────┘
               │                                      │ load_from_wandb_artifact()
               │                                      │ make_policy_yaak()
               │                                      │ make_pre_post_processors()
               └──────────────────┬───────────────────┘
                                  ▼
Script: evaluate_pretrained_policy_rbyte.py → scripts/eval_yaak.py::predict_main_yaak()
┌──────────────────────────────────────────────────────────────────────────┐
│  per batch:                                                              │
│    rbyte.Batch                                                           │
│      → __getbatch__()          rbyte format → lerobot dict               │
│      → preprocessor()          normalize + move to device                │
│      → policy.forward()        loss + loss_dict                          │
│      → policy.predict_action_chunk()  deterministic action prediction    │
│      → postprocessor()         unnormalize predicted actions             │
│                                                                          │
│  after all batches:                                                      │
│    → metric_accum_callback()   per-drive GT vs pred plots                │
│    → create_reye_df()          assemble output DataFrame                 │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
Outputs
┌──────────────────────────────────────────────────────────────────────────┐
│  /nasa/team-space/artifacts/predictions/lerobot/v044/<artifact>/         │
│    results.parquet   reye format: GT + predictions + L1 scores           │
│                      keyed by timestamp + input_id (drive)               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Component Map

```shell
justfile
├── just train
│   └── examples/training/train_policy_rbyte.py         # entrypoint
│       ├── Hydra: rbyte/config/train.yaml
│       │   └── experiment/yaak/finetune.yaml
│       │       ├── datamodule: yaak/train             → torch.utils.data.DataLoader
│       │       │   └── dataset: yaak/train_clip        → rbyte.Dataset (ytt-generated)
│       │       ├── datamodule_val: yaak/val
│       │       ├── normalization: original
│       │       ├── model: smolvla_train
│       │       └── paths: yaak/default
│       ├── load_dataset_stats()                         # reads paths.lerobot_stats
│       ├── make_cfg_smolvla_clip()                      # builds configs from model/smolvla_train
│       └── src/lerobot/scripts/train_yaak.py::train()  # training loop
│           ├── make_policy_yaak()                       # factory_yaak.py
│           ├── make_pre_post_processors()               # normalize + rename
│           ├── __getbatch__()                           # rbyte→lerobot batch conversion
│           └── eval_policy_yaak()                       # in-loop evaluation
│
├── just predict
│   └── examples/training/evaluate_pretrained_policy_rbyte.py
│       ├── Hydra: rbyte/config/predict.yaml
│       │   └── inference/yaak/default.yaml
│       │       ├── model: smolvla               → load_from_wandb_artifact()
│       │       └── datamodule: yaak/predict
│       └── scripts/eval_yaak.py::predict_main_yaak()
│           ├── eval_policy_yaak_loop()
│           │   └── __getbatch__() + policy.forward() + postprocessor()
│           └── create_reye_df()                        # writes parquet output
│
└── just rbyte2lerobot-stats
    └── src/lerobot/policies/smolvla/conversion_utils_yaak.py
        └── compute_rbyte2lerobot_stats()               # stats → JSON
```

---

## Key Components

### 1. Config System (`rbyte/config/`)

Config composition tree for `train.yaml`:

```
train.yaml
└── experiment/yaak/finetune.yaml
    ├── datamodule/yaak/train.yaml    (→ dataset/yaak/train_clip.yaml)
    ├── datamodule_val/yaak/val.yaml
    ├── paths/yaak/default.yaml
    ├── normalization/original.yaml   (IDENTITY for visual/action, MIN_MAX for state)
    └── model/smolvla_train.yaml      (all SmolVLA hyperparameters)
```

### 2. rbyte Dataset (`rbyte.Dataset`)

The current [`train_clip.yaml`](rbyte/config/_templates/dataset/yaak/train_clip.yaml) reflects several deliberate decisions:

**1. Waypoint frame: translation-only (no heading rotation)**

The `rmind` config applies `ST_Rotate(..., radians("waypoints/heading"))` to rotate waypoints into an ego-heading-aligned frame. The `train_clip` drops the rotation and only translates (`ST_Translate`) to the ego position. The model is expected to learn heading-invariant behavior from the visual context instead.

**2. Waypoint coordinates in raw meters (not scaled)**

The `rmind` config divides normalized waypoint coordinates by 100 (`ST_X(...) / 100`), mapping values roughly to `[-1.5, 1.5]`. `train_clip` outputs raw meter values and delegates scaling to the normalization layer (`MIN_MAX` over the full dataset range), keeping the spatial meaning explicit.

**3. Dense 101-frame buffer with separate context and action windows**

The `rmind` config uses a sparse clip: `period: 60i, gather_every: 10` → 6 frames spanning 50 frame-index steps. `train_clip` buffers 101 consecutive frames (`period: 101i, gather_every: 1`) and then extracts two non-overlapping windows from that buffer in the final DuckDB query:

| Window | Slice | Frames | Purpose |
|---|---|---|---|
| Context (observation) | `list_slice(..., 1, 60, 10)` | 6 frames | Image + speed input to the model |
| Action horizon | `list_slice(..., 52, 101, 1)` | 50 frames | Future actions for chunk prediction |

The context window covers frames 1–60 at stride 10 (roughly the present); the action horizon covers frames 52–101 at stride 1 (the immediate future). This decoupling allows predicting a 50-step action chunk from a sub-sampled visual history.

**4. Lerobot-compatible feature naming**

Earlier configs kept raw rbyte column names (`meta/VehicleMotion/speed`, `waypoints/xy_normalized`, etc.) and left renaming to downstream code. `train_clip` renames columns to lerobot constants directly in the DuckDB query (`observation.state`, `observation.state.vehicle`, `task`), so `__getbatch__` can look up features by their lerobot keys without an extra mapping layer.

### 3. Batch Conversion Layer (`__getbatch__`)

[`conversion_utils_yaak.py:101`](src/lerobot/policies/smolvla/conversion_utils_yaak.py#L101)

Converts a `rbyte.Batch` to the lerobot `dict` format expected by SmolVLA preprocessing:

| rbyte field | lerobot key | Transform |
|---|---|---|
| `meta/VehicleMotion/{gas,brake,steering}_normalized` | `action` | `torch.stack` → shape `(B, T, 3)` |
| `cam_front_left` | `observation.image` | `permute(0,1,4,2,3) / 255` → `(B, T, C, H, W)` |
| `observation.state.vehicle` (speed) | `observation.state.vehicle` | expand dims → `(B, T, 1)` |
| `observation.state` (waypoints) | `observation.state` | expand + repeat → `(B, T, 20)` |

This function is called at the start of every train and eval step, between the DataLoader and the preprocessor.

### 4. Dataset Statistics (`conversion_utils_yaak.py`)

[`conversion_utils_yaak.py:32`](src/lerobot/policies/smolvla/conversion_utils_yaak.py#L32)

`compute_rbyte2lerobot_stats()` iterates the full dataset and computes statistics for:
- `action`: per-dimension (gas, brake, steering) min/max/mean/std/q01/q99
- `observation.state`: waypoints (flattened 10×2 = 20 dims), tile min/max/mean/std
- `observation.state.vehicle`: vehicle speed scalar

Output is a JSON file at `paths.lerobot_stats`. This file is required at train time to initialize the normalization layers.

### 5. Policy Factory (`factory_yaak.py`)

[`factory_yaak.py`](src/lerobot/policies/factory_yaak.py)

A Yaak-specific copy of the upstream `make_policy` factory. The key difference: it accepts `stats: dict` (pre-loaded from the stats JSON) instead of `LeRobotDatasetMetadata`. This is needed because rbyte datasets don't produce `LeRobotDatasetMetadata`.

### 6. SmolVLA Model Changes

Three new configuration flags and supporting architecture changes:

#### `use_context` (temporal multi-frame input)

[`modeling_smolvla.py:434`](src/lerobot/policies/smolvla/modeling_smolvla.py#L434)

When `True`, all `T` frames of a clip are passed as separate image tokens to the VLM (rather than only the last frame). The `prepare_images` method loops over all temporal indices when this flag is set, requiring `ndim == 5` tensors `(B, T, C, H, W)`.

#### `use_separate_intent` (decoupled waypoint + speed embeddings)

[`modeling_smolvla.py:505`](src/lerobot/policies/smolvla/modeling_smolvla.py#L505)

When `True`, waypoints (`observation.state`) and vehicle speed (`observation.state.vehicle`) are projected into separate embedding spaces before being concatenated and fed to the VLM:

```
waypoints → intent_proj (max_intent_dim → hidden)  → intent_emb
speed     → state_proj  (max_state_dim  → hidden)  → vehicle_emb
concat([intent_emb, vehicle_emb], dim=1) → state tokens
```

When `False` (legacy), the two features are concatenated along the feature dim and passed through a single `state_proj`.

`prepare_state_wrapper` is the new dispatch function replacing `prepare_state`, routing to the appropriate behavior based on these flags.

### 7. Normalization Pipeline

The normalization strategy is configured in three layers that compose at runtime:

**Layer 1 — Config** ([`normalization/original.yaml`](rbyte/config/normalization/original.yaml))

Defines per-`FeatureType` normalization modes:

```yaml
visual: IDENTITY   # images already [0,1], no normalization
action:  IDENTITY  # raw CAN signals, no normalization
state:   MIN_MAX   # waypoints and speed both tagged as STATE
```

**Layer 2 — Python** ([`train_policy_rbyte.py`](examples/training/train_policy_rbyte.py#L183))

Hydra instantiates the YAML values into a `normalization_mapping` dict keyed by feature role string, then passed into `SmolVLAConfig`:

```python
normalization_mapping = {
    "VISUAL": NormalizationMode.IDENTITY,   # from normalization.visual
    "STATE":  NormalizationMode.MIN_MAX,    # from normalization.state
    "ACTION": NormalizationMode.IDENTITY,   # from normalization.action
}
```

Both `observation.state` (waypoints) and `observation.state.vehicle` (speed) share the `STATE` `FeatureType`, so they would both receive `MIN_MAX` normalization — but their value ranges differ significantly (waypoints in meters, speed in km/h).

**Layer 3 — Runtime override** ([`normalize_processor.py:37`](src/lerobot/processor/normalize_processor.py#L37))

`patch_norm_mode()` is called inside `_apply_transform` for every key before the normalization math runs:

```python
def patch_norm_mode(norm_mode: NormalizationMode, key: str) -> NormalizationMode:
    return NormalizationMode.ZERO_ONE if key == OBS_STATE_VEHICLE else norm_mode
```

This unconditionally overrides the `STATE` mode to `ZERO_ONE` (maps `[min, max] → [0, 1]`) for `observation.state.vehicle` representing the speed (`[0, 130] -> [0,, 1]`), while leaving waypoints on `MIN_MAX` (maps `[min, max] → [-1, 1]`).

The override exists to support different types of normalizations for the same `FeatureType`.

```
normalization/original.yaml
    state: MIN_MAX
         │
         │  instantiate(hydra_cfg.normalization.state)
         ▼
normalization_mapping["STATE"] = NormalizationMode.MIN_MAX
         │
         │  SmolVLAConfig(..., normalization_mapping=...)
         │  make_pre_post_processors(...)
         ▼
NormalizerProcessorStep._apply_transform(tensor, key, FeatureType.STATE)
    norm_mode = norm_map[STATE]          →  MIN_MAX
    norm_mode = patch_norm_mode(norm_mode, key)
        if key == "observation.state.vehicle"  →  ZERO_ONE   (speed → [0, 1])
        else                                   →  MIN_MAX     (waypoints → [-1, 1])
```

### 8. Evaluation and reye Output

[`eval_yaak.py`](src/lerobot/scripts/eval_yaak.py), [`reye_utils.py`](src/lerobot/utils/reye_utils.py)

The eval loop collects predicted actions and ground truth, then writes a Polars DataFrame in the **reye format** (Yaak's internal prediction evaluation format) to `results.parquet`. The schema includes:

- Input identifiers and timestamps
- Ground truth: `predictions/policy/ground_truth/continuous/{gas,brake,steering}`
- Predictions: `predictions/policy/prediction_value/continuous/{gas,brake,steering}`
- L1 scores: `predictions/policy/score_l1/continuous/{gas,brake,steering}`
- Confidence placeholders (std, probs, logprob)

The `is_without_clip` flag handles the difference between single-frame (non-clip) and multi-frame (clip) timestamp formats.

### 9. W&B Logging Extensions

[`wandb_utils_yaak.py`](src/lerobot/utils/wandb_utils_yaak.py), [`rl/wandb_utils.py`](src/lerobot/rl/wandb_utils.py)

- `tracking_callback`: called inside `SmolVLAPolicy.forward()` after loss computation. Computes per-action loss statistics (std, mean) filtered to non-zero action regions (where gas > 0.04, brake > threshold, |steering| > 0.01). Added to `loss_dict` for W&B logging.
- `metric_accum_callback`: called at the end of each eval epoch. Accumulates per-drive loss arrays, plots GT vs predicted actions with matplotlib, and logs them as W&B `Image` objects.
- `WandBLogger.log_image`: new method added to upstream `WandBLogger` to support logging these images.


