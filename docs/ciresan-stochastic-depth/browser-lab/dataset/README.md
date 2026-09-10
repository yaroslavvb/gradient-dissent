# Browser MNIST subset

The default dataset contains **10,000 training, 2,000 validation, and 2,000 test images**. It is **11,046,096 bytes raw or 2,296,866 bytes gzip**. An optional full official test set is 1,622,145 bytes gzip. The deployed page serves only `.bin.gz`; the producer also writes the raw `.bin` form for verification.

`manifest.json` records hashes, sizes, offsets, source hashes, original dataset indices, the existing experiment's parent split, and attribution. `preview.json` contains two public training examples per digit as base64 uint8 arrays; these can be drawn on a canvas without downloading a full split.

## Binary decoding

Every `.bin` starts with a 32-byte header. The first eight bytes are ASCII `GDMNIST1`. All following header fields are unsigned little-endian 32-bit integers:

| Byte offset | Value |
| --- | --- |
| 8 | Image count |
| 12 | Rows (28) |
| 16 | Columns (28) |
| 20 | Image-data offset (32) |
| 24 | Label-data offset |
| 28 | Original-index offset |

Image data consists of `count × 784` row-major uint8 values. Labels are `count` uint8 values. Original indices are `count` little-endian uint32 values. There may be up to three zero bytes between labels and indices for alignment. Pixels are unchanged grayscale MNIST pixels: 0 is black, 255 is white. For training, convert to float32 and divide by 255.

For `.gz`, decompress the fetched bytes with `new DecompressionStream('gzip')`; the decompressed buffer has exactly the `.bin` format and hash. Check size, hash, magic, offsets, and label range before using it. If a server has already applied HTTP content decoding, detect the `GDMNIST1` magic before decompressing again.

## Selection and evaluation

The train and validation subsets come from the **same disjoint 50,000/10,000 pools as the A100 experiments**, whose seeded permutation and hashes are verified in the producer. The browser subset takes the first 1,000/200 examples per digit within the respective parent pool. Within each digit, the parent order is retained. Classes are then interleaved 0–9, so any prefix whose size is a multiple of 10 is balanced. **Shuffle training batches each epoch.**

The default test subset takes the first 200 examples of each digit in official test order. Thus default test accuracy equally weights digits; it is not the exact naturally weighted 10,000-example A100 test. Optional `test-full.bin.gz` retains all 10,000 test examples in official order. Train and validation selections were complete before test data was opened. Test labels were used only for the declared test stratification. Use validation for choices such as epoch selection; repeatedly viewing test predictions makes this an exploratory evaluation.

The embedded original indices are zero-based indices into either official `train-images-idx3-ubyte.gz` (train and validation) or official `t10k-images-idx3-ubyte.gz` (test and test-full). Distinct source collections can share the same index number; this does not indicate overlap.

## Source and attribution

MNIST is by **Yann LeCun, Corinna Cortes, and Christopher J.C. Burges**. Its images were selected and repacked, with their pixel values preserved. Credit: [The MNIST Database of Handwritten Digits](https://yann.lecun.org/exdb/mnist/index.html). Reference: LeCun, Bottou, Bengio, and Haffner, *Gradient-Based Learning Applied to Document Recognition*, Proceedings of the IEEE 86(11), 1998, [DOI:10.1109/5.726791](https://doi.org/10.1109/5.726791).

The dataset assets are **[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)**, as documented by [Keras's official MNIST documentation](https://keras.io/api/datasets/mnist/). This license statement applies to MNIST dataset assets, not unrelated application code. [CVDF's MNIST mirror](https://github.com/cvdfoundation/mnist) records permission to mirror the dataset and describes the original IDX files. The original compressed files used here were already cached locally and were checked against the earlier experiment's MD5, SHA256, and byte counts.

## Rebuild

From the repository's uv environment:

```sh
uv run python scripts/browser-lab/build_dataset.py --repo /path/to/gradient-dissent --cache /path/to/mnist-cache --output /path/to/output
```

The cache must contain the four original `.gz` IDX files. The existing `hypothesis-audit-s101-v1/data.npz` and `result.json` archives provide independent provenance and pixel/label equality checks. No downloads or training occur in this script.
