# Pinned TensorFlow.js browser dependencies

Both browser bundles are served locally; the application does not fetch scripts from a CDN.

| Deployed file | npm package | Exact version | Package source |
| --- | --- | --- | --- |
| `tf.min.js` | `@tensorflow/tfjs` | 4.22.0 | `dist/tf.min.js` |
| `tf-backend-webgpu.min.js` | `@tensorflow/tfjs-backend-webgpu` | 4.22.0 | `dist/tf-backend-webgpu.min.js` |

`package-lock.json` pins npm tarball integrity. `npm run build:browser` verifies package versions and copies these bundles, then records SHA256 values in the lab's `manifest.json`. The TensorFlow.js repository is [tensorflow/tfjs](https://github.com/tensorflow/tfjs/tree/tfjs-v4.22.0). Its [Apache 2.0 license at the same release](https://github.com/tensorflow/tfjs/blob/tfjs-v4.22.0/LICENSE) is preserved as `LICENSE-tensorflow.txt`. The bundles retain their embedded license headers and are unmodified.

The optional `webgpu` 0.6.0 Node dependency is **only for native test execution**, is not sent to the browser, and is not needed to serve the page. No dependency lifecycle scripts are required for this project build.
