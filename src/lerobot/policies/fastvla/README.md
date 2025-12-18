

# FastVLM


Wrapper for the FastVLM's vision module.

The goal is to implement image preprocessing to be ina format that FastVLM expects, but also to test export to ONNX.


Running export:

```shell
just export-fast
```

## Note
You will get the `[torch.onnx] Execute the model with ONNX Runtime... ❌` because some constants in normalization get instantiated as float32. Exporting to TensorRT with `--stronglyTyped` will make sure that everything is done in FP16 (next chapter). It is also possible to export with:
```shell
just export-fast dtype=torch.float32
```
And then force FP16 during export with `--precisionConstraints=obey --layerPrecisions=*:fp16  --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16` instead of `--stronglyTyped`. 

## Export to TensorRT
To convert to TensorRT, rsync the ONNX to the delta kit and run the [export_fastvitd.sh](https://github.com/yaak-ai/drahve/blob/vit/whac-a-mole/scripts/export_fastvitd.sh)

```shell
nix develop
bash scripts/export_fastvitd.sh
```

The above script assumes that the ONNX model is saved to `~/fastvithd_16_256_batch_1_nc` directory.