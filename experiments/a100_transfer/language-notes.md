# GPT baseline and language-data protocol

## API for the runner

```python
from language import GPT, GPTConfig, TokenWindowLoader, prepare_language, selfcheck

manifest = prepare_language('/data/wikitext103-gpt2-v1')  # remote CPU preparation
model = GPT(GPTConfig()).cuda()  # random initialization, no downloaded weights
train = TokenWindowLoader('/data/wikitext103-gpt2-v1', 'train', context=512, seed=123)
validation = TokenWindowLoader('/data/wikitext103-gpt2-v1', 'val', context=512)
x, y = train.sample(batch_size=32, device='cuda')
logits = model(x, scales=None, keep=None)  # [B, T, 50257]
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
for vx, vy in validation.fixed_batches(8, device='cuda', max_windows=128):
    # Aggregate summed token loss and total target count, not batch means.
    pass
```

`GPT.prunable_count` and `GPT.depth` equal 12 by default. `scales` has shape `[B, L]` and contains the caller's `m/(1-p)` values; the model does not sample masks. The same scale multiplies attention and MLP residual branches separately, with the MLP receiving the attention-updated state, matching the paper's Eq. 6. Scales execute full dense kernels and must **not** be reported as physical compute savings. `keep` is a sequence of integer indices such as `(0,2,4,6,8,10)`; omitted blocks really are bypassed, in their original order. It is not a Boolean mask. There is one final LayerNorm and one tied head; no auxiliary exit heads or adapters.

The loss above allocates full vocabulary logits. Keep the measured peak VRAM in the result record and choose microbatch size after the diagnostic. Gradient accumulation changes effective batch size without requiring a different model API. Use a separate RNG for masks and the loader's RNG for data so paired treatments can replay the same batches. Save/restore `train.rng.bit_generator.state` if resuming a run.

## Architecture and initialization

The default is a nanoGPT/GPT-2-small-style decoder: 12 blocks, width 768, 12 attention heads, learned absolute positions, vocabulary 50,257, a 4× GELU MLP, biased LayerNorm/linear projections, tied token embedding and LM head. Context 512 gives **124,046,592 unique parameters**. Using context 1024 instead gives the familiar additional positional-embedding parameters, so the 512-context model is not claimed to be a byte-for-byte pretrained GPT-2 configuration.

The implementation uses PyTorch causal SDPA, zero attention/activation dropout, zero biases, normal initialization with standard deviation 0.02, and residual projection standard deviation `0.02/sqrt(2*L)`. Exact GELU follows the pinned nanoGPT implementation. It uses ordinary parameterization and initialization; no CompleteP/µP transfer rule is claimed. Thus this is an architectural transfer check against the paper, not a reproduction of its Celerity architecture, tokenizer, corpus, optimizer, or scale.

Source provenance: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291), immutable commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. `vendor/nanogpt/model.py` is unmodified reference code; `vendor/nanogpt/provenance.json` records its SHA256. The adaptation is in `language.py`. The upstream MIT license, copyright Andrej Karpathy 2022, is preserved at `vendor/nanogpt/LICENSE`. Training loads neither pretrained weights nor executable remote code.

## Data identity and preparation

Use [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1), config `wikitext-103-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. This is the raw, open-vocabulary variant, not the word-level variant with unknown-token substitutions. Both official training shards and the official validation/test shards are explicitly named and SHA256 checked against the Hugging Face LFS object identities. No new train/test split is manufactured.

Preparation downloads approximately 315 MB of Parquet on a CPU worker. It requires `tiktoken==0.11.0` and `pyarrow==21.0.0` in addition to NumPy; their versions are recorded. GPT-2 encoding has vocabulary 50,257 and EOT 50,256. The manifest fingerprints the actual merge ranks, special tokens, and tokenizer regex, and records the SHA256 of every source file and output bin. Tokenization uses `encode_ordinary` so literal special-token strings in source text remain ordinary text.

WikiText's rows are lines/paragraphs, not independent documents. The parser reconstructs an article from a top-level `= Title =` heading to the next top-level heading, preserving raw row text by concatenation; subsection headings stay inside the article. Blank-only preamble is omitted. One EOT is appended per reconstructed article. The output is little-endian uint16 `train.bin`, `val.bin`, `test.bin`, plus `*.doc_offsets.npy` and `manifest.json`. `val` is only a local alias for the official `validation` split. Prepared token counts are computed, not guessed from WikiText's word-token statistics.

Train samples uniformly among all valid context+1 starts within training documents, with replacement. A sampled sequence never crosses an article or official split boundary. Repeated sampling therefore is **not** a single-epoch experiment, and exposure relative to the training token count should be reported.

Held-out evaluation uses fixed context+1 spans with stride context+1 within each article. Neither inputs nor targets overlap between evaluation windows. A `max_windows` cap selects a deterministic evenly spaced subset of available windows; save `evaluation_manifest(max_windows)` with the results. Short articles and incomplete tails are omitted, so the result is a **fixed-window GPT-2-token evaluation on WikiText103**, not the canonical word-level full-corpus WikiText perplexity. Only validation results may select settings; test results are reserved for final evaluation.

Dataset licensing remains separate from source-code licensing. The [pinned dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md) metadata lists CC-BY-SA-3.0 and GFDL, while its prose links CC-BY-SA-4.0. Preserve that discrepancy in provenance rather than silently relicensing the corpus. Publish the preparation code, manifests, and measurements; raw/tokenized corpus files are not added to this repository.

## Verification performed

On the local CPU with `experiments/.venv/bin/python`, `python language.py --selfcheck` passed:

- Tied weight identity; dense versus all-one scaling and all-block `keep`.
- Static pruning versus equivalent zero/unit scaling, including zero retained blocks.
- Causality: changing future tokens leaves earlier logits unchanged.
- Finite gradients and zero gradients for a zero-scaled block.
- Dense logits and all parameter gradients match the unmodified pinned nanoGPT reference after loading identical weights.
- Article/subsection parsing, train-only random sampling, shifted targets, document-contained windows, and disjoint held-out spans.

The tiny check uses 3 blocks, width 16, 2 heads, 37 vocabulary entries, context 9; it does not allocate the default model. A meta-device construction independently verified the default parameter count and 12 prunable blocks. Full corpus preparation and A100 validation are runner steps; no paid GPU jobs were launched by this module or its authoring process.
