
Setup the environment:

```shell
just sync
just install-duckdb-extensions
```

Evaluate a model artifact:

```shell
just predict model.artifact=yaak/lerobot/policy_smolvla-seed_1000-dataset_yaak-ai_L2D-060000:v19
```

Train a model:

```shell
just train
```
