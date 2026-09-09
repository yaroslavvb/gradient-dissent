"""Vision backbones and CIFAR-100 helpers for the A100 transfer study.

Common API:
    model = build_model('vit_cifar100' | 'convnext_cifar100', **overrides)
    logits = model(images, scales=None, keep=None)
`scales` is [batch, model.prunable_count], normally Bernoulli/(1-p),
provided by the runner. `keep` is an iterable of zero-based block indices;
None keeps everything. Forward contains no random draws or hidden dropout.
All retained blocks execute in their original order and use the original head.

The ConvNeXt architecture follows the official implementation, with external
per-example residual scales and explicit pruning replacing built-in DropPath.
Upstream attribution and MIT notice (https://github.com/facebookresearch/ConvNeXt):
Copyright (c) Meta Platforms, Inc. and affiliates.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
CIFAR100_MD5 = "03b5dce01913d631647c71ecec9e9cb8"
NORMALIZATION = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def _controls(x: torch.Tensor, count: int, scales, keep):
    """Shape checks avoid data-dependent GPU synchronizations in the hot path."""
    if scales is not None:
        if not isinstance(scales, torch.Tensor) or scales.ndim != 2 or scales.shape != (len(x), count):
            raise ValueError(f"scales must have shape [batch={len(x)}, prunable={count}]")
        if scales.device != x.device:
            raise ValueError("scales and inputs must be on the same device")
    if keep is None:
        return scales, None
    indices = tuple(keep)
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices):
        raise ValueError("keep must contain integer block indices, not a Boolean mask")
    if len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
        raise ValueError("keep contains duplicates or out-of-range block indices")
    return scales, frozenset(int(i) for i in indices)


def _scale_branch(branch, scale):
    if scale is None:
        return branch
    return branch * scale.to(dtype=branch.dtype).reshape(-1, *([1] * (branch.ndim-1)))


class ViTBlock(nn.Module):
    """Bidirectional pre-norm Transformer block; one shared Eq6 layer mask."""
    def __init__(self, width: int, heads: int, mlp_width: int):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads = width, heads
        self.attn_norm = nn.LayerNorm(width, eps=1e-6)
        self.qkv = nn.Linear(width, 3*width)
        self.attn_out = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_width), nn.GELU(), nn.Linear(mlp_width, width))

    def forward(self, x, scale=None):
        b, n, d = x.shape
        qkv = self.qkv(self.attn_norm(x)).reshape(b, n, 3, self.heads, d//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
        attention = attention.transpose(1, 2).reshape(b, n, d)
        # SAME scale is applied at both updates. The second branch sees the
        # scaled first update, matching the shared-mask Eq6 composition.
        h = x + _scale_branch(self.attn_out(attention), scale)
        return h + _scale_branch(self.mlp(self.mlp_norm(h)), scale)


class VisionTransformer(nn.Module):
    def __init__(self, depth=12, width=768, heads=12, mlp_width=3072,
                 patch_size=4, image_size=32, num_classes=100):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image size must be divisible by patch size")
        self.image_size, self.depth = image_size, depth
        self.patch = nn.Conv2d(3, width, patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.empty(1, (image_size//patch_size)**2, width))
        self.blocks = nn.ModuleList([ViTBlock(width, heads, mlp_width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.head = nn.Linear(width, num_classes)
        self.prunable_ids = tuple(f"blocks.{i}" for i in range(depth))
        self.prunable_count = depth
        self.config = {"family": "vit", "depth": depth, "width": width, "heads": heads,
                       "mlp_width": mlp_width, "patch_size": patch_size, "image_size": image_size,
                       "tokens": (image_size//patch_size)**2, "num_classes": num_classes,
                       "pooling": "mean of final-normalized patch tokens", "mask": "shared over attention and MLP residual updates",
                       "initialization": "truncated normal std.02, zero biases, LayerNorm weight1"}
        self.apply(_initialize)
        nn.init.trunc_normal_(self.position, std=.02)

    def forward(self, x, scales=None, keep=None):
        if x.ndim != 4 or x.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"expected [B,3,{self.image_size},{self.image_size}] input")
        scales, selected = _controls(x, self.prunable_count, scales, keep)
        h = self.patch(x).flatten(2).transpose(1, 2) + self.position
        for i, block in enumerate(self.blocks):
            if selected is None or i in selected:
                h = block(h, None if scales is None else scales[:, i])
        return self.head(self.norm(h).mean(dim=1))

    def primary_keep(self):
        return tuple(range(max(1, (2*self.depth)//3)))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each pixel, never across batch or space."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, layer_scale_init=1e-6):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, 4*channels)
        self.project = nn.Linear(4*channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), float(layer_scale_init)))

    def forward(self, x, scale=None):
        branch = self.depthwise(x).permute(0, 2, 3, 1)
        branch = self.project(F.gelu(self.expand(self.norm(branch))))
        branch = (branch * self.gamma).permute(0, 3, 1, 2)
        return x + _scale_branch(branch, scale)


class ConvNeXtTiny(nn.Module):
    def __init__(self, depths=(3, 3, 9, 3), widths=(96, 192, 384, 768),
                 num_classes=100, layer_scale_init=1e-6, image_size=32):
        super().__init__()
        if len(depths) != 4 or len(widths) != 4 or any(d < 1 for d in depths):
            raise ValueError("four nonempty stages are required")
        if image_size < 32 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of32")
        self.image_size, self.depths = image_size, tuple(depths)
        self.downsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, widths[0], 4, stride=4), ChannelLayerNorm(widths[0])),
            *[nn.Sequential(ChannelLayerNorm(widths[i-1]), nn.Conv2d(widths[i-1], widths[i], 2, stride=2))
              for i in range(1, 4)],
        ])
        self.stages = nn.ModuleList([nn.ModuleList([ConvNeXtBlock(width, layer_scale_init)
                                                  for _ in range(n)])
                                     for width, n in zip(widths, depths)])
        self.norm = nn.LayerNorm(widths[-1], eps=1e-6)
        self.head = nn.Linear(widths[-1], num_classes)
        self.prunable_ids = tuple(f"stages.{s}.{b}" for s, n in enumerate(depths) for b in range(n))
        self.prunable_count = sum(depths)
        self.config = {"family": "convnext_tiny", "depths": list(depths), "widths": list(widths),
                       "image_size": image_size, "num_classes": num_classes,
                       "layer_scale_initial_value": layer_scale_init, "depthwise_kernel": 7,
                       "spatial_sizes": [image_size//4, image_size//8, image_size//16, image_size//32],
                       "mandatory": ["stem", "downsample1", "downsample2", "downsample3"],
                       "pooling": "global spatial mean, final LayerNorm, classifier",
                       "initialization": "truncated normal std.02, zero biases, LayerNorm weight1"}
        self.apply(_initialize)

    def forward(self, x, scales=None, keep=None):
        if x.ndim != 4 or x.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"expected [B,3,{self.image_size},{self.image_size}] input")
        scales, selected = _controls(x, self.prunable_count, scales, keep)
        index = 0
        for downsample, stage in zip(self.downsample, self.stages):
            x = downsample(x)  # mandatory even if an entire residual stage is removed
            for block in stage:
                if selected is None or index in selected:
                    x = block(x, None if scales is None else scales[:, index])
                index += 1
        return self.head(self.norm(x.mean(dim=(2, 3))))

    def stagewise_keep(self, counts: Iterable[int]):
        counts = tuple(counts)
        if len(counts) != len(self.depths) or any(k < 0 or k > n for k, n in zip(counts, self.depths)):
            raise ValueError("retained counts must fit each of four stages")
        keep, start = [], 0
        for n, k in zip(self.depths, counts):
            keep.extend(range(start, start+k))
            start += n
        return tuple(keep)

    def primary_keep(self):
        return self.stagewise_keep(tuple((2*n)//3 for n in self.depths))


def _initialize(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_model(task: str, **overrides):
    if task in {"vit", "vit_cifar100", "vit_base"}:
        return VisionTransformer(**overrides)
    if task in {"convnext", "convnext_cifar100", "convnext_tiny"}:
        return ConvNeXtTiny(**overrides)
    raise ValueError(f"Unknown vision task: {task}")


def describe_model(model):
    return {**model.config, "parameter_count": sum(p.numel() for p in model.parameters()),
            "prunable_count": model.prunable_count, "prunable_ids": list(model.prunable_ids),
            "primary_keep": list(model.primary_keep()), "scales_shape": "[batch,prunable_count]",
            "keep_semantics": "zero-based prunable indices; None=all; retained blocks preserve original order"}


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _stratified_split(labels, val_per_class=50, split_seed=913071):
    if not 1 <= val_per_class < 500:
        raise ValueError("val_per_class must be between1 and499")
    rng = np.random.default_rng(split_seed)
    validation = []
    for label in range(100):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) != 500:
            raise ValueError("official training set must contain500 examples of each fine label")
        validation.extend(rng.permutation(candidates)[:val_per_class])
    val_indices = np.sort(np.asarray(validation, dtype=np.int64))
    train_indices = np.setdiff1d(np.arange(len(labels), dtype=np.int64), val_indices)
    assert not np.intersect1d(train_indices, val_indices).size
    assert len(train_indices)+len(val_indices) == 50000
    return train_indices, val_indices


def load_cifar100(root, val_per_class=50, split_seed=913071, download=True):
    """Return {train, validation, test: (uint8[N,3,32,32], int64[N]), metadata}.

    Uses the original binary archive and no torchvision dependency, pickle,
    archive extraction, augmentation, or test-dependent preprocessing. The
    official published archive MD5 is verified before parsing fixed byte rows.
    The original 10,000-image test set is returned whole and in original order.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-100-binary.tar.gz"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(archive)
        temporary = archive.with_name(archive.name+f".part-{uuid.uuid4().hex}")
        try:
            with urllib.request.urlopen(CIFAR100_URL, timeout=90) as response, temporary.open("wb") as target:
                while chunk := response.read(1024*1024):
                    target.write(chunk)
            if hashlib.md5(temporary.read_bytes()).hexdigest() != CIFAR100_MD5:
                raise ValueError("CIFAR100 archive checksum mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    archive_bytes = archive.read_bytes()
    if hashlib.md5(archive_bytes).hexdigest() != CIFAR100_MD5:
        raise ValueError("Cached CIFAR100 archive checksum mismatch")
    decoded, member_hashes = {}, {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        for name, count in [("train", 50000), ("test", 10000)]:
            member_name = f"cifar-100-binary/{name}.bin"
            source = tar.extractfile(member_name)
            if source is None:
                raise ValueError(f"Missing archive member {member_name}")
            raw = source.read()
            if len(raw) != count*3074:
                raise ValueError("Unexpected CIFAR100 binary row count")
            rows = np.frombuffer(raw, dtype=np.uint8).reshape(count, 3074)
            labels = rows[:, 1].astype(np.int64)
            images = rows[:, 2:].copy().reshape(count, 3, 32, 32)
            assert np.array_equal(np.bincount(labels, minlength=100), np.full(100, count//100))
            decoded[name] = (images, labels)
            member_hashes[name] = _sha256(raw)
    train_ids, val_ids = _stratified_split(decoded["train"][1], val_per_class, split_seed)
    result = {}
    for name, ids in [("train", train_ids), ("validation", val_ids)]:
        x, y = decoded["train"]
        result[name] = (torch.from_numpy(x[ids].copy()), torch.from_numpy(y[ids].copy()))
    result["test"] = tuple(torch.from_numpy(a.copy()) for a in decoded["test"])
    result["metadata"] = {
        "dataset": "CIFAR-100 fine-label classification, official binary archive",
        "source": "https://www.cs.toronto.edu/~kriz/cifar.html", "download_url": CIFAR100_URL,
        "archive_md5": CIFAR100_MD5, "archive_sha256": _sha256(archive_bytes), "member_sha256": member_hashes,
        "split_seed": split_seed, "val_per_class": val_per_class,
        "train_count": len(train_ids), "validation_count": len(val_ids), "test_count": 10000,
        "train_indices": train_ids.tolist(), "validation_indices": val_ids.tolist(),
        "train_indices_sha256": _sha256(train_ids.astype("<i8").tobytes()),
        "validation_indices_sha256": _sha256(val_ids.astype("<i8").tobytes()),
        "test_policy": "all10000 official test rows in original order; never part of splitting or fitting",
        "normalization": NORMALIZATION, "normalization_fit": "none; fixed affine mapping from[0,255] to[-1,1]",
        "augmentation": "training only: reflection pad4, uniform32x32 crop, independent horizontal flip p.5; supplied separate torch.Generator",
    }
    return result


def prepare_images(images, *, training=False, generator=None, image_size=32):
    """Transform raw uint8 NCHW images, with CPU or GPU augmentation RNG.

    Keep this generator independent of the data sampler and mask generator.
    Reset it to the same seed for each paired treatment. No randomness is used
    in evaluation. Returned dtype is float32 on the input device. Optional
    deterministic bilinear resizing is applied after cropping/flipping; this
    adds spatial samples, not new image information.
    """
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("prepare_images expects raw uint8 [B,3,32,32]")
    if image_size < 32 or image_size % 32:
        raise ValueError("image_size must be a positive multiple of32")
    x = images.float().div_(255.)
    if training:
        if generator is None:
            raise ValueError("training augmentation requires an explicit independent generator")
        b, c, h, w = x.shape
        rng_device = generator.device
        top = torch.randint(9, (b,), generator=generator, device=rng_device).to(x.device)
        left = torch.randint(9, (b,), generator=generator, device=rng_device).to(x.device)
        flip = (torch.rand((b,), generator=generator, device=rng_device) < .5).to(x.device)
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
        rows = top[:, None] + torch.arange(h, device=x.device)[None]
        cols = left[:, None] + torch.arange(w, device=x.device)[None]
        x = x.gather(2, rows[:, None, :, None].expand(b, c, h, w+8))
        x = x.gather(3, cols[:, None, None, :].expand(b, c, h, w))
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    if image_size != 32:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x.sub_(.5).div_(.5)


def self_test():
    """Tiny local CPU tests; no download, paid work, or full-size training."""
    torch.manual_seed(441)
    checks = {}
    tiny_models = {
        "vit": VisionTransformer(depth=3, width=24, heads=3, mlp_width=48),
        "convnext": ConvNeXtTiny(depths=(1, 1, 2, 1), widths=(8, 16, 24, 32), layer_scale_init=.1),
    }
    for name, model in tiny_models.items():
        model.double().eval()
        x = torch.randn(2, 3, 32, 32, dtype=torch.float64, requires_grad=True)
        full = model(x)
        assert torch.equal(full, model(x)), f"{name}: nondeterministic evaluation"
        assert torch.equal(full, model(x, keep=tuple(range(model.prunable_count))))
        assert torch.equal(full, model(x, scales=torch.ones(2, model.prunable_count, dtype=x.dtype)))
        keep = tuple(range(0, model.prunable_count, 2))
        scales = torch.zeros(2, model.prunable_count, dtype=x.dtype)
        scales[:, keep] = 1
        assert torch.equal(model(x, keep=keep), model(x, scales=scales))
        assert torch.equal(model(x, keep=()), model(x, scales=torch.zeros_like(scales)))
        assert torch.equal(model(x, keep=tuple(reversed(keep))), model(x, keep=keep))
        model(x, scales=scales).square().mean().backward()
        assert torch.isfinite(x.grad).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert not any(isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout)) for m in model.modules())
        # Per-example scaling is equivalent to evaluating each example alone.
        mixed = torch.tensor([[1.]*model.prunable_count, [0.]*model.prunable_count], dtype=x.dtype)
        expected = torch.cat([model(x[:1]), model(x[1:], keep=())])
        torch.testing.assert_close(model(x, scales=mixed), expected, rtol=1e-10, atol=1e-10)
        checks[name] = {"allkeep_equal": True, "zero_scales_equal_skip": True, "per_example_masks": True,
                        "finite_gradients": True, "deterministic_eval": True, "no_batchnorm": True,
                        "description": describe_model(model)}
    # Shared-mask ViT semantics: with scale2, both branches must use scale2;
    # the second branch consumes the already-scaled first update.
    block = ViTBlock(12, 3, 24).double()
    z = torch.randn(2, 4, 12, dtype=torch.float64)
    scale = torch.tensor([0., 2.], dtype=z.dtype)
    qkv = block.qkv(block.attn_norm(z)).reshape(2, 4, 3, 3, 4).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(*qkv.unbind(0), dropout_p=0., is_causal=False)
    h = z + block.attn_out(a.transpose(1, 2).reshape(2, 4, 12))*scale[:, None, None]
    expected = h + block.mlp(block.mlp_norm(h))*scale[:, None, None]
    assert torch.equal(block(z, scale), expected)
    assert torch.equal(block(z, scale)[0], z[0])
    # The augmentation stream remains unchanged by arbitrary mask RNG draws.
    raw = torch.arange(2*3*32*32).remainder(256).to(torch.uint8).reshape(2, 3, 32, 32)
    gen1 = torch.Generator().manual_seed(81)
    gen2 = torch.Generator().manual_seed(81)
    a1 = prepare_images(raw, training=True, generator=gen1)
    _ = torch.rand(317, generator=torch.Generator().manual_seed(99))
    a2 = prepare_images(raw, training=True, generator=gen2)
    assert torch.equal(a1, a2)
    assert torch.equal(prepare_images(raw), raw.float()/127.5-1.)
    resized = prepare_images(raw, image_size=64)
    assert resized.shape == (2, 3, 64, 64)
    assert torch.equal(resized, prepare_images(raw, image_size=64))
    assert a1.shape == raw.shape and torch.isfinite(a1).all() and a1.min() >= -1 and a1.max() <= 1
    labels = np.repeat(np.arange(100), 500)
    train, val = _stratified_split(labels)
    assert len(train) == 45000 and len(val) == 5000
    assert np.array_equal(np.bincount(labels[val]), np.full(100, 50))
    assert np.array_equal(val, _stratified_split(labels)[1])
    checks["shared_vit_residual_scale"] = True
    checks["augmentation_rng_independent"] = True
    checks["split_stratified_disjoint_reproducible"] = True
    return checks


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(self_test(), indent=2))
