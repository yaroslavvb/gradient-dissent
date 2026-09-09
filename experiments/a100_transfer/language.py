"""GPT-2-small shape and pinned WikiText103 input for the A100 transfer study.

Model adapted from Andrej Karpathy's nanoGPT (MIT); see vendor/nanogpt/LICENSE
and provenance.json. No model weights are downloaded. Scales implement paper
Eq. 6 with shared per-sequence masks on both residual branches. The scale path
executes dense kernels; only static `keep` really bypasses blocks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=-1)
        q, k, v = [z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
                   for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()  # nanoGPT's exact GELU, not a new activation recipe.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x, scale=1.0):
        # The same scale is reused, but the FFN sees the scaled attention result.
        # This is intentionally not x + scale * (ordinary_block(x) - x).
        x = x + scale * self.attn(self.ln_1(x))
        return x + scale * self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig | None = None):
        super().__init__()
        self.config = config or GPTConfig()
        c = self.config
        if min(c.block_size, c.vocab_size, c.n_layer, c.n_head, c.n_embd) < 1:
            raise ValueError("GPT dimensions must be positive")
        if c.n_embd % c.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.prunable_count = c.n_layer
        self.depth = c.n_layer
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(c.vocab_size, c.n_embd),
            "wpe": nn.Embedding(c.block_size, c.n_embd),
            "h": nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            "ln_f": nn.LayerNorm(c.n_embd, eps=1e-5),
        })
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens, scales=None, keep=None):
        """Return all-position logits [B,T,V]. `scales` is [B,L].

        `keep` is a sequence of retained integer block indices (not a bool mask).
        Blocks always execute in original order, with no new head or fine-tuning.
        Pass scales=m/(1-p) externally; the model never resamples randomness.
        """
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.block_size:
            raise ValueError("tokens must have shape [batch, 1..block_size]")
        b, t = tokens.shape
        if scales is not None and tuple(scales.shape) != (b, self.prunable_count):
            raise ValueError("scales must have shape [batch, n_layer]")
        retained = None
        if keep is not None:
            kept = tuple(keep)
            if any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                   or not 0 <= i < self.prunable_count for i in kept):
                raise ValueError("keep must contain valid integer block indices")
            if len(set(kept)) != len(kept):
                raise ValueError("keep contains repeated block indices")
            retained = set(kept)
        position = torch.arange(t, device=tokens.device)
        x = self.transformer.wte(tokens) + self.transformer.wpe(position)
        for i, block in enumerate(self.transformer.h):
            if retained is not None and i not in retained:
                continue
            scale = 1.0 if scales is None else scales[:, i, None, None].to(x.dtype)
            x = block(x, scale)
        return self.lm_head(self.transformer.ln_f(x))


DATASET_REPO = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_VERSION = "0.11.0"
PREPARATION_VERSION = 1
SOURCE_FILES = {
    "train": [
        ("train-00000-of-00002.parquet", "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"),
        ("train-00001-of-00002.parquet", "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720"),
    ],
    "val": [("validation-00000-of-00001.parquet", "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c")],
    "test": [("test-00000-of-00001.parquet", "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91")],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_checked(url, path: Path, digest):
    if path.exists():
        if sha256_file(path) != digest:
            raise ValueError(f"Existing source file failed SHA256 verification: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    if sha256_file(temporary) != digest:
        raise ValueError(f"Downloaded source failed SHA256 verification: {url}")
    temporary.replace(path)


def iter_documents(rows: Iterable[str]):
    """Reconstruct articles from raw rows using top-level '= Title =' headings.

    Raw row text is concatenated exactly; subsection '= = Heading = =' lines
    do not start new articles. Blank-only preamble is excluded. EOT insertion is
    performed by prepare_language, once after each nonempty reconstructed article.
    """
    article = []
    top_heading = re.compile(r"^= [^=\n].*? =$", re.ASCII)
    for text in rows:
        if top_heading.fullmatch(text.strip()) and article:
            previous = "".join(article)
            if previous.strip():
                yield previous
            article = []
        article.append(text)
    if article:
        previous = "".join(article)
        if previous.strip():
            yield previous


def _parquet_rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["text"]):
            yield from batch.column(0).to_pylist()


def prepare_language(root) -> dict:
    """Prepare all official splits; intended for remote CPU, never starts GPU work.

    Downloads ~315 MB of verified Parquet; writes <u2 bins, int64 document offsets,
    and manifest.json. Existing complete artifacts are checked before reuse.
    The preparation needs tiktoken==0.11.0 and pyarrow (tested pin 21.0.0).
    """
    import tiktoken
    if importlib.metadata.version("tiktoken") != TOKENIZER_VERSION:
        raise RuntimeError(f"Use tiktoken=={TOKENIZER_VERSION} for this frozen tokenizer protocol")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["preparation_version"] != PREPARATION_VERSION
                or manifest["dataset_revision"] != DATASET_REVISION):
            raise ValueError("Prepared dataset protocol differs; use a new output directory")
        for part in manifest["splits"].values():
            for artifact in ("tokens", "document_offsets"):
                item = part[artifact]
                if sha256_file(root / item["file"]) != item["sha256"]:
                    raise ValueError(f"Prepared {artifact} failed checksum verification")
        return manifest
    encoder = tiktoken.get_encoding("gpt2")
    if encoder.n_vocab != 50257 or encoder.eot_token != 50256:
        raise ValueError("Unexpected GPT-2 tokenizer identity")
    # Fingerprint the actual rank table, special-token table and regex, not only a name.
    vocabulary_digest = hashlib.sha256()
    for raw, rank in sorted(encoder._mergeable_ranks.items(), key=lambda item: item[1]):
        vocabulary_digest.update(rank.to_bytes(4, "little"))
        vocabulary_digest.update(len(raw).to_bytes(4, "little"))
        vocabulary_digest.update(raw)
    vocabulary_digest.update(json.dumps(encoder._special_tokens, sort_keys=True).encode())
    vocabulary_digest.update(encoder._pat_str.encode())
    manifest = {
        "preparation_version": PREPARATION_VERSION,
        "dataset_repository": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "tokenizer": {"library": "tiktoken", "version": TOKENIZER_VERSION,
                      "encoding": "gpt2", "vocab_size": 50257, "eot_id": 50256,
                      "rank_table_specials_regex_sha256": vocabulary_digest.hexdigest()},
        "dtype": "little-endian uint16 (<u2)",
        "document_policy": "Exact raw-row concatenation; a top-level '= Title =' heading starts an article; ignore blank-only preamble; encode_ordinary(article), then one EOT. Subsections remain in their article.",
        "split_policy": "Official train/validation/test, never concatenate across splits. Local name val denotes official validation.",
        "licensing": "Dataset card metadata lists CC-BY-SA-3.0 and GFDL; its prose links CC-BY-SA-4.0. Preserve source attribution; binaries are local experiment artifacts, not relicensed under nanoGPT MIT.",
        "dataset_card": f"https://huggingface.co/datasets/{DATASET_REPO}/blob/{DATASET_REVISION}/README.md",
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "pyarrow", "tiktoken")},
        "splits": {},
    }
    source_dir = root / "sources"
    source_dir.mkdir(exist_ok=True)
    for split, files in SOURCE_FILES.items():
        sources, paths = [], []
        for filename, digest in files:
            url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{DATASET_CONFIG}/{filename}"
            path = source_dir / filename
            _download_checked(url, path, digest)
            paths.append(path)
            sources.append({"url": url, "sha256": digest, "bytes": path.stat().st_size})
        token_path = root / f"{split}.bin"
        temporary = root / f"{split}.bin.part"
        offsets = [0]
        text_bytes = 0
        with temporary.open("wb") as stream:
            for document in iter_documents(_parquet_rows(paths)):
                token_ids = encoder.encode_ordinary(document)
                token_ids.append(encoder.eot_token)
                np.asarray(token_ids, dtype="<u2").tofile(stream)
                offsets.append(offsets[-1] + len(token_ids))
                text_bytes += len(document.encode("utf-8"))
        temporary.replace(token_path)
        offsets_path = root / f"{split}.doc_offsets.npy"
        np.save(offsets_path, np.asarray(offsets, dtype="<i8"))
        manifest["splits"][split] = {
            "official_split": "validation" if split == "val" else split,
            "source_files": sources,
            "documents": len(offsets) - 1, "text_utf8_bytes": text_bytes,
            "token_count": offsets[-1],
            "tokens": {"file": token_path.name, "sha256": sha256_file(token_path)},
            "document_offsets": {"file": offsets_path.name, "sha256": sha256_file(offsets_path)},
        }
        print(f"Prepared {split}: {offsets[-1]:,} tokens in {len(offsets)-1:,} articles", flush=True)
    temporary_manifest = root / "manifest.json.part"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


class TokenWindowLoader:
    """Memory-mapped, document-contained next-token windows.

    Train uses uniform random *valid starts* with replacement, never val/test.
    Held-out windows are fixed, stride context+1, so even x+y spans are disjoint.
    Short documents and per-document tails cannot fill a fixed window and are
    omitted. This evaluation is not the canonical full-corpus WikiText perplexity.
    """
    def __init__(self, root, split="train", context=512, seed=1234):
        self.root = Path(root)
        self.split = "val" if split == "validation" else split
        if self.split not in SOURCE_FILES or context < 1:
            raise ValueError("Use split train/val/test and a positive context")
        self.context = context
        self.tokens = np.memmap(self.root / f"{self.split}.bin", mode="r", dtype="<u2")
        self.offsets = np.load(self.root / f"{self.split}.doc_offsets.npy", allow_pickle=False)
        if (self.offsets.ndim != 1 or self.offsets[0] != 0
                or self.offsets[-1] != len(self.tokens) or np.any(np.diff(self.offsets) <= 0)):
            raise ValueError("Invalid document offsets")
        self.valid_counts = np.maximum(np.diff(self.offsets) - context, 0)
        self.cumulative = np.cumsum(self.valid_counts)
        if self.cumulative[-1] <= 0:
            raise ValueError("No document can fill a context+1 token window")
        self.rng = np.random.default_rng(seed)
        self.eval_starts = np.concatenate([
            np.arange(int(a), int(b) - context, context + 1, dtype=np.int64)
            for a, b in zip(self.offsets[:-1], self.offsets[1:])
            if b - a > context
        ])

    def _batch(self, starts, device):
        chunks = np.stack([np.asarray(self.tokens[s:s+self.context+1], dtype=np.int64)
                           for s in starts])
        x = torch.from_numpy(chunks[:, :-1].copy()).to(device)
        y = torch.from_numpy(chunks[:, 1:].copy()).to(device)
        return x, y

    def sample(self, batch_size, device="cpu"):
        if self.split != "train":
            raise ValueError("Random sampling is restricted to the official training split")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        choices = self.rng.integers(0, int(self.cumulative[-1]), size=batch_size)
        documents = np.searchsorted(self.cumulative, choices, side="right")
        preceding = np.where(documents > 0, self.cumulative[np.maximum(documents-1, 0)], 0)
        starts = self.offsets[documents] + choices - preceding
        return self._batch(starts, device)

    def fixed_batches(self, batch_size, device="cpu", max_windows=None):
        if self.split == "train":
            raise ValueError("Fixed evaluation batches must use val/test")
        if batch_size < 1 or (max_windows is not None and max_windows < 1):
            raise ValueError("batch_size and max_windows must be positive")
        starts = self.eval_starts
        if max_windows is not None and max_windows < len(starts):
            starts = starts[np.linspace(0, len(starts)-1, max_windows, dtype=np.int64)]
        for begin in range(0, len(starts), batch_size):
            yield self._batch(starts[begin:begin+batch_size], device)

    def evaluation_manifest(self, max_windows=None):
        starts = self.eval_starts
        if max_windows is not None:
            if max_windows < 1:
                raise ValueError("max_windows must be positive")
            if max_windows < len(starts):
                starts = starts[np.linspace(0, len(starts)-1, max_windows, dtype=np.int64)]
        return {"split": self.split, "context": self.context,
                "windows": len(starts), "target_tokens": len(starts)*self.context,
                "all_available_windows": len(self.eval_starts),
                "starts": starts.tolist(),
                "policy": "Document-contained disjoint context+1 spans, stride context+1; optional evenly spaced deterministic subset."}


def selfcheck():
    """Fast CPU checks with a tiny shape; no data/network/GPU required."""
    import importlib.util
    import sys
    import tempfile
    torch.manual_seed(20260909)
    c = GPTConfig(block_size=9, vocab_size=37, n_layer=3, n_head=2, n_embd=16)
    model = GPT(c).eval()
    tokens = torch.randint(c.vocab_size, (2, c.block_size))
    targets = torch.randint(c.vocab_size, tokens.shape)
    dense = model(tokens)
    assert model.transformer.wte.weight is model.lm_head.weight
    torch.testing.assert_close(dense, model(tokens, scales=torch.ones(2, 3)), rtol=0, atol=0)
    torch.testing.assert_close(dense, model(tokens, keep=range(3)), rtol=0, atol=0)
    mask = torch.tensor([[1., 0., 1.], [1., 0., 1.]])
    torch.testing.assert_close(model(tokens, scales=mask), model(tokens, keep=(0, 2)), rtol=0, atol=0)
    torch.testing.assert_close(model(tokens, scales=torch.zeros(2, 3)), model(tokens, keep=()), rtol=0, atol=0)
    future_changed = tokens.clone()
    future_changed[:, 5:] = (future_changed[:, 5:] + 1) % c.vocab_size
    torch.testing.assert_close(dense[:, :5], model(future_changed)[:, :5], rtol=0, atol=0)
    F.cross_entropy(model(tokens, scales=mask).flatten(0, 1), targets.flatten()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0
               for p in model.transformer.h[1].parameters())
    # Match dense logits/gradients to the unmodified, pinned nanoGPT reference.
    reference_path = Path(__file__).parent / "vendor/nanogpt/model.py"
    spec = importlib.util.spec_from_file_location("_pinned_nanogpt_reference", reference_path)
    reference_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference_module
    spec.loader.exec_module(reference_module)
    reference = reference_module.GPT(reference_module.GPTConfig(**asdict(c))).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    reference_logits, reference_loss = reference(tokens, targets)
    torch.testing.assert_close(dense, reference_logits, rtol=1e-6, atol=1e-7)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(tokens).flatten(0, 1), targets.flatten()).backward()
    reference_loss.backward()
    for (name, p), (other_name, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
    # Real parser convention, synthetic tokens: do not turn subsections into docs.
    assert list(iter_documents(["\n", " = First = \n", "A\n", " = = Sub = = \n", "B\n", " = Second = \n", "C\n"])) == [
        " = First = \nA\n = = Sub = = \nB\n", " = Second = \nC\n"]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for split in SOURCE_FILES:
            np.arange(40, dtype="<u2").tofile(directory / f"{split}.bin")
            np.save(directory / f"{split}.doc_offsets.npy", np.array([0, 3, 20, 40], dtype="<i8"))
        train = TokenWindowLoader(directory, "train", context=5)
        x, y = train.sample(100)
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert not any(a < 3 or (a < 20 and a+5 >= 20) or a+5 >= 40 for a in x[:, 0].tolist())
        heldout = TokenWindowLoader(directory, "val", context=5)
        starts = heldout.evaluation_manifest()["starts"]
        assert all(b-a >= 6 for a, b in zip(starts, starts[1:]))
        assert len(list(heldout.fixed_batches(2))) == math.ceil(len(starts)/2)
        try:
            heldout.sample(1)
        except ValueError:
            pass
        else:
            raise AssertionError("Held-out random sampling was not rejected")
    return {"passed": True, "tiny_config": asdict(c), "tiny_parameters": model.get_num_params(),
            "checks": ["weight tying", "unit masks", "all/static masks", "causality", "finite gradients",
                       "zero dropped-block gradients", "pinned nanoGPT logits/gradient parity",
                       "article boundaries", "train-only random windows", "disjoint heldout windows"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_language(args.prepare), indent=2))
    elif args.selfcheck:
        print(json.dumps(selfcheck(), indent=2))
    else:
        parser.error("Choose --prepare DIRECTORY or --selfcheck")
