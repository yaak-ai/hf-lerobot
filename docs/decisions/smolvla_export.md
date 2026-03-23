
Real data used for export, not dummy

Model is cast to float16
 - ONNX does not support bfloat16, TRT does


Things outside of TRT:

1. Normalization
- norm. parameters stored to `f: ${hydra:run.dir}/normalization/normalization.pt` (based on @rbyte/config/export/onnx.yaml)

2. Image resizing and normalization to [-1, 1]
- handled in drivr

3. Language tokenization
- computed for a real batch

4. Language embedding
- computed once to reduce inference time
- saved as a buffer to the model:

```python
# Save language embeddings into a buffer
self.policy.register_buffer("lang_emb", lang_emb)
self.policy.register_buffer("lang_masks", lang_masks)
```

SmolVLA exported as 3 TRT models -> 3 PyTorch wrappers

1. Embedding (ViT, nn.Linear for speed, waypoints)
- TRT Embedding of a full episode[6]

Overrides to hf-transformers:
- avoid data-dependent branching
- perfromance: mask creation
ExportSmolVLMVisionTransformer, ExportSmolVLMVisionEmbeddings

Test if transformers have changed & that the overrides produce the same results

Running tests -> prereq for export

SigLip used to be called in the for loop for each image -> creates 6 streams


2. TRT Embedding of the episode[1] of 1 timestamp
Concates the prefix_embs_cache w the latest embeddings

3. Action model (transformer + action expert transformer)
Mask computation moved outside the denoising for loop -> performance