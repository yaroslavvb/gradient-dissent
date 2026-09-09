"""Ciresan-width MNIST controls, with explicitly separate historical/adapted modes.

Reference: yaroslavvb/stuff/autotune/util.py, SimpleFullyConnected2, TinyMNIST.
Default nn.Linear initialization is preserved, including bias. Historical flags
are input_scale=1, output_relu=True; defaults below are a controlled adaptation.
Residual middle transitions are NEW architecture: fixed prefix crop, no learned
projection. A dropped transition still crops and applies ReLU; it is not an
identity on the original wider vector. SD gates are shared by the minibatch.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WIDTHS = (784, 2500, 2000, 1500, 1000, 500, 10)
SPLIT_SEED = 20260909
MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _cpu_list(values, name):
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(f"{name} must be a Python sequence or CPU tensor; no device synchronization")
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return values.tolist()
    return list(values)


def _probabilities(values, count):
    result = tuple(float(p) for p in _cpu_list(values, "drop_probs"))
    if len(result) != count or any(not math.isfinite(p) or not 0 <= p <= 1 for p in result):
        raise ValueError(f"drop_probs must contain {count} finite probabilities in [0,1]")
    return result


class CiresanMLP(nn.Module):
    def __init__(self, recipe="plain", pmax=0.0, input_scale=1/255,
                 output_relu=False, widths=WIDTHS, bias=True):
        super().__init__()
        if recipe not in {"plain", "residual", "unit_dropout", "residual_unit_dropout", "sd_constant", "sd_annealed"}:
            raise ValueError(f"Unknown recipe: {recipe}")
        if not math.isfinite(pmax) or not 0 <= pmax < 1:
            raise ValueError("pmax must be finite and in [0,1)")
        if not math.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("input_scale must be finite and positive")
        self.widths = tuple(int(n) for n in widths)
        if len(self.widths) < 3 or any(n <= 0 for n in self.widths):
            raise ValueError("widths must describe at least a stem and head")
        self.recipe, self.pmax = recipe, float(pmax)
        self.input_scale, self.output_relu = float(input_scale), bool(output_relu)
        self.residual = recipe in {"residual", "residual_unit_dropout", "sd_constant", "sd_annealed"}
        self.stochastic_depth = recipe in {"sd_constant", "sd_annealed"}
        self.unit_dropout = recipe in {"unit_dropout", "residual_unit_dropout"}
        if self.residual and any(b > a for a, b in zip(self.widths[1:-2], self.widths[2:-1])):
            raise ValueError("Residual prefix-crop transitions require nonincreasing hidden widths")
        self.layers = nn.ModuleList(nn.Linear(a, b, bias=bias) for a, b in zip(self.widths[:-1], self.widths[1:]))
        self.prunable_count = len(self.layers) - 2 if self.residual else 0
        self.prunable_ids = tuple(f"linear_{i}" for i in range(1, len(self.layers)-1)) if self.residual else ()

    @property
    def default_drop_probs(self):
        n = self.prunable_count
        return tuple(self.pmax * (i+1) / n for i in range(n)) if self.stochastic_depth else (0.0,) * n

    def _unit_mask(self, h, generator):
        if not self.training or not self.unit_dropout or self.pmax == 0:
            return h
        if generator is None:
            return F.dropout(h, p=self.pmax, training=True)
        # The caller can supply a CUDA generator for GPU unit dropout. CPU RNG
        # is also supported and transfers only masks, without a device readback.
        noise = torch.rand(h.shape, device=generator.device, generator=generator)
        return h * (noise >= self.pmax).to(device=h.device, dtype=h.dtype) / (1-self.pmax)

    def forward(self, x, drop_probs=None, active=None, generator=None):
        """Training-only gates; eval always keeps every branch with scale one.

        active: boolean sequence of length prunable_count, shared by the batch.
        drop_probs: matching probabilities; SD annealing is owned by the trainer.
        An explicit retained gate is scaled by 1/(1-p), just like a sampled one.
        No device tensor is converted to Python to choose a branch.
        """
        h = x.reshape(-1, self.widths[0]) * self.input_scale
        gates, probs = (True,) * self.prunable_count, (0.0,) * self.prunable_count
        # Plain controls accept the runner's gate arguments but ignore them;
        # their ordinary unit-dropout probability is always self.pmax.
        if self.training and self.residual:
            if self.recipe == "sd_annealed" and drop_probs is None:
                raise ValueError("sd_annealed requires trainer-supplied drop_probs")
            probs = self.default_drop_probs if drop_probs is None else _probabilities(drop_probs, self.prunable_count)
            if not self.stochastic_depth and any(probs):
                raise ValueError("Only SD recipes accept positive residual drop probabilities")
            if active is not None:
                gates = tuple(_cpu_list(active, "active"))
                if len(gates) != self.prunable_count or any(type(g) is not bool for g in gates):
                    raise ValueError("active must contain one bool per residual transition")
            elif self.stochastic_depth and any(probs):
                if generator is not None and generator.device.type != "cpu":
                    raise ValueError("SD requires a CPU generator for synchronization-free branching")
                draws = torch.rand(self.prunable_count, device="cpu", generator=generator).tolist()
                gates = tuple(u >= p for u, p in zip(draws, probs))
            if any(g and p == 1 for g, p in zip(gates, probs)):
                raise ValueError("Cannot retain a branch with zero survival probability")
        h = self._unit_mask(F.relu(self.layers[0](h)), generator)
        for i, layer in enumerate(self.layers[1:-1]):
            if self.residual:
                skip = h[..., :layer.out_features]
                # Genuine bypass: a dropped branch never calls Linear.
                h = F.relu(skip + layer(h) / (1-probs[i])) if gates[i] else F.relu(skip)
            else:
                h = F.relu(layer(h))
            h = self._unit_mask(h, generator)
        h = self.layers[-1](h)
        return F.relu(h) if self.output_relu else h


def split_indices(total=60000, train_size=50000, val_size=10000, seed=SPLIT_SEED):
    """One fixed random split; smaller training subsets are nested prefixes."""
    if not (0 < train_size <= total-val_size and 0 < val_size < total):
        raise ValueError("Invalid train/validation sizes")
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    return order[:train_size], order[total-val_size:]


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _download(root, filename):
    path = root / filename
    if not path.exists():
        with urllib.request.urlopen(MNIST_BASE_URL+filename, timeout=120) as response:
            data = response.read()
        if hashlib.md5(data).hexdigest() != MNIST_FILES[filename]:
            raise ValueError(f"MNIST download MD5 mismatch: {filename}")
        with tempfile.NamedTemporaryFile(dir=root, prefix=filename+".", delete=False) as stream:
            tmp = stream.name
            stream.write(data)
        os.replace(tmp, path)
    data = path.read_bytes()
    if hashlib.md5(data).hexdigest() != MNIST_FILES[filename]:
        raise ValueError(f"MNIST cached MD5 mismatch: {filename}")
    return data, {"url": MNIST_BASE_URL+filename, "md5": MNIST_FILES[filename], "sha256": _sha256(data), "bytes": len(data)}


def _decode_idx(compressed, images, expected_count):
    try:
        raw = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise ValueError("Invalid MNIST gzip stream") from exc
    if images:
        if len(raw) < 16:
            raise ValueError("Truncated image header")
        magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
        if (magic, count, rows, cols) != (2051, expected_count, 28, 28) or len(raw) != 16+count*784:
            raise ValueError("Invalid MNIST image IDX header/length")
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=16).copy().reshape(count, 28, 28))
    if len(raw) < 8:
        raise ValueError("Truncated label header")
    magic, count = struct.unpack(">II", raw[:8])
    if (magic, count) != (2049, expected_count) or len(raw) != 8+count:
        raise ValueError("Invalid MNIST label IDX header/length")
    array = np.frombuffer(raw, dtype=np.uint8, offset=8).copy()
    if np.any(array > 9):
        raise ValueError("Invalid MNIST label")
    return torch.from_numpy(array).long()


def load_mnist(root, train_size=50000, include_test=False, seed=SPLIT_SEED):
    """Raw float32 [0,255] pixels; no preprocessing fitted on heldout data.

    include_test=False does not download, open, decode or inspect official test
    data. train_size=10000 uses a nested subset of the same 50k training pool.
    Split is seeded random, NOT stratified; class counts are recorded.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    def load(name, images, count):
        blob, info = _download(root, name)
        files[name] = info
        return _decode_idx(blob, images, count)
    x = load("train-images-idx3-ubyte.gz", True, 60000)
    y = load("train-labels-idx1-ubyte.gz", False, 60000)
    train, val = split_indices(train_size=train_size, seed=seed)
    result = {"train_x": x[train].float(), "train_y": y[train], "val_x": x[val].float(), "val_y": y[val]}
    train_digest = _sha256(train.numpy().astype("<i8").tobytes())
    val_digest = _sha256(val.numpy().astype("<i8").tobytes())
    metadata = {"dataset": "MNIST", "input_shape": [28, 28], "pixel_dtype": "float32", "pixel_range": [0,255],
                "normalization": "none; model input_scale is explicit", "split_method": "torch.randperm; nested train prefix; fixed final 10000 validation indices",
                "split_seed": int(seed), "train_size": len(train), "val_size": len(val),
                "train_indices_sha256": train_digest, "val_indices_sha256": val_digest,
                "split_sha256": _sha256(bytes.fromhex(train_digest)+bytes.fromhex(val_digest)),
                "train_class_counts": torch.bincount(y[train], minlength=10).tolist(),
                "val_class_counts": torch.bincount(y[val], minlength=10).tolist(),
                "train_pixel_mean": float(result["train_x"].mean()), "train_pixel_std": float(result["train_x"].std(unbiased=False)),
                "test_included": bool(include_test), "files": files}
    if include_test:
        result["test_x"] = load("t10k-images-idx3-ubyte.gz", True, 10000).float()
        result["test_y"] = load("t10k-labels-idx1-ubyte.gz", False, 10000)
        metadata["test_size"] = 10000
    metadata["training_data_sha256"] = _sha256(json.dumps({"files": {k:v for k,v in files.items() if k.startswith("train")}, "train_indices": train_digest, "val_indices": val_digest}, sort_keys=True).encode())
    result["metadata"] = metadata
    return result
