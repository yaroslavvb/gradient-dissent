# Vision model and data implementation notes

The vision helper is [vision.py](vision.py). No paid computation or pretrained weights were used while implementing it. `self_test()` passes on local PyTorch 2.14.0 CPU, including tiny forward/backward tests and exact pruning equivalence checks. Architecture counts were checked with PyTorch meta tensors.

## Shared API

```python
from vision import build_model, describe_model, load_cifar100, prepare_images

model = build_model("vit_cifar100", image_size=32)
# or build_model("convnext_cifar100", image_size=64)
print(describe_model(model))

# Images are already normalized float32 [B,3,H,W].
# `scales` is [B,P], with caller-generated Bernoulli/(1-p) entries.
# `keep` contains integer indices, NOT booleans; None keeps everything.
logits = model(images, scales=scales, keep=None)
full_logits = model(images)
pruned_logits = model(images, keep=model.primary_keep())
```

- `model.prunable_count`: number of shape-preserving residual blocks.
- `model.prunable_ids`: stable names in forward execution order.
- `keep`: tuple/list of zero-based prunable block indices. Reordering this tuple does not reorder the model. Duplicate/out-of-range indices and Boolean masks are rejected. The empty tuple removes every prunable block, retaining mandatory embedding/downsampling/head modules.
- `scales`: tensor of exact shape `[batch, prunable_count]` on the input device. The runner owns schedule construction and independent per-image Bernoulli draws. No stochastic state is hidden in the model.
- `scales=None`: retained branches use their ordinary residual strength (inference scale 1). ConvNeXt's learned LayerScale remains part of each residual branch.
- Passing zero scales computes then zeros the branches; passing `keep` skips their computation. Their outputs agree for finite inputs. Training-time active-work accounting must not be called measured compute savings when using compute-then-mask.

## ViT

Default `vit_cifar100` has **85,219,684 parameters**, 12 blocks, width 768, 12 attention heads, MLP width 3,072, and patch size 4. Native 32×32 inputs produce 64 patch tokens. It uses learned positional embeddings, bidirectional attention, pre-LayerNorm residual updates, final tokenwise LayerNorm, mean pooling, and a 100-class linear head. There is no class token and no attention, embedding, or MLP dropout.

Each block's **same per-example scale multiplies both attention and MLP updates**. The MLP sees the already-scaled attention update, implementing the composition of the paper's shared-mask Eq.6. An attention-only or MLP-only dropout implementation would define a different experiment.

`primary_keep()` retains indices `(0,1,2,3,4,5,6,7)`. Alternative depths and widths are accepted for tests or controls. Increasing `image_size` to 64 while keeping patch size 4 gives 256 tokens and changes attention cost substantially; report this setting explicitly.

The block dimensions follow [the ViT paper](https://arxiv.org/abs/2010.11929) and [the authors' official base-model configuration](https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py). Patch size, mean pooling, CIFAR-resolution training from scratch, and the short experimental schedule differ from the original large-pretraining setting. This is a backbone-family transfer test, not a reproduction of the paper's headline ViT results. The helper's PyTorch attention implementation was written for this experiment; no third-party ViT source or pretrained checkpoint was copied.

## ConvNeXt

Default `convnext_cifar100` has **27,897,028 parameters**. Stage depths are `(3,3,9,3)` and widths `(96,192,384,768)`. Each block contains depthwise 7×7 convolution, channel LayerNorm, a 4× expanded pointwise MLP with GELU, and a learned per-channel LayerScale initialized to 1e−6. All **18** residual blocks are shape-preserving and prunable. The stride-4 stem and three stride-2 stage transitions remain mandatory.

The final representation is spatially averaged, LayerNormed, then classified. There is no BatchNorm and no running-statistics recalibration when pruning. Built-in DropPath is replaced by the supplied ILD scales, so the dense treatment has no hidden stochastic-depth regularization.

`primary_keep()` retains the first `(2,2,6,2)` blocks in the four stages, corresponding to indices `(0,1,3,4,6,7,8,9,10,11,15,16)`. These are 12 of 18 residual blocks. `stagewise_keep(counts)` constructs other stage-preserving prefixes.

The architecture follows [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) and the [official ConvNeXt implementation](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py). The official implementation is MIT-licensed; its copyright and complete permission/disclaimer notice are retained in `vision.py`.

**Resolution caveat:** native CIFAR32 yields spatial stages 8→4→2→1. In the last stage, only the central position of a padded 7×7 depthwise kernel overlaps actual input. Bilinearly upsampling to 64 gives 16→8→4→2; upsampling to 128 gives 32→16→8→4. Upsampling adds no new observed image information. These settings remain different from canonical ImageNet224 spatial geometry.

**LayerScale caveat:** all residual branches start with strength 1e−6. An undertrained model can appear robust to pruning because its branches remain close to zero. The runner should record learned `stages.S.B.gamma` magnitudes per stage, full-model learning curves, and final quality. Apparent pruning robustness without a learned full-model baseline is not useful transfer evidence.

## CIFAR-100 data and augmentation

```python
data = load_cifar100("/data/cifar100", val_per_class=50, split_seed=913071)
train_x, train_y = data["train"]          # uint8 NCHW, int64 fine labels
val_x, val_y = data["validation"]
test_x, test_y = data["test"]

# Use an augmentation generator separate from minibatch and mask generators.
aug = torch.Generator(device="cuda").manual_seed(seed + 200_000)
batch = prepare_images(raw_batch_on_gpu, training=True, generator=aug, image_size=64)
eval_batch = prepare_images(raw_test_batch_on_gpu, image_size=64)
```

The helper downloads the [original CIFAR-100 binary archive](https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz), verifies its published MD5 `03b5dce01913d631647c71ecec9e9cb8`, then parses fixed byte rows from the tar members. It does not use torchvision, Python pickle, or archive extraction. Metadata records archive/member SHA-256 hashes and original train-row indices and hashes.

The original dataset contains 500 training and 100 test images per fine class. A fixed stratified subset of **50 images per class** from official training becomes validation: **45,000 train / 5,000 validation**. All **10,000 official test images** are returned unchanged and in their original order. See the [official dataset description and requested citation to Krizhevsky (2009)](https://www.cs.toronto.edu/~kriz/cifar.html).

`prepare_images` requires uint8 `[B,3,32,32]`. Training uses reflection padding of four pixels, uniformly sampled 32×32 crops, and independently sampled horizontal flips with probability 0.5. Optional bilinear resize (`align_corners=False`) follows cropping/flipping. Evaluation has no crop or flip. Fixed affine normalization `(x/255−0.5)/0.5` maps pixels to [−1,1]; no normalization statistic is fitted on validation or test. The runner must reset the independent augmentation generator to the same seed across paired treatments and pass the chosen `image_size` during both training and evaluation.

Local tests verify exact all-kept/full equivalence; exact explicit skips/zero-scale equivalence; order-preserving keep semantics; deterministic evaluation; finite input and parameter gradients; per-example scales; shared-mask ViT Eq.6 composition; absence of BatchNorm and hidden Dropout; augmentation RNG isolation; deterministic resizing; and disjoint, stratified, reproducible split indices. The actual large archive download is intentionally left to the runner's data preparation rather than exercised during the tiny local model tests.
