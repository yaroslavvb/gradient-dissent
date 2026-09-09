(function (root) {
  'use strict';
  const LAYERS = 12, SEQUENCES = 4;
  function probability(layer, progress, maximum, layers = LAYERS) {
    return maximum * layer / (layers - 1) * (1 - progress);
  }
  function expectedDepth(progress, maximum, layers = LAYERS) {
    return layers * (1 - maximum * (1 - progress) / 2);
  }
  function seededRandom(seed) {
    return () => {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function batch({mode, progress, maximum, retained = 8, layers = LAYERS, sequences = SEQUENCES}, random) {
    return Array.from({length: sequences}, () => Array.from({length: layers}, (_, layer) => {
      const p = mode === 'train' ? probability(layer, progress, maximum, layers) : 0;
      const keep = mode === 'train' ? random() >= p : mode === 'dense' || layer < retained;
      return {p, survival: 1 - p, keep, scale: keep ? (mode === 'train' ? 1 / (1 - p) : 1) : 0};
    }));
  }
  function scalarBlock(x, a, b, scale) {
    const z = x + scale * a * x;
    return {z, y: z + scale * b * z};
  }
  const api = {LAYERS, SEQUENCES, probability, expectedDepth, seededRandom, batch, scalarBlock};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.DropoutMath = api;
})(typeof window !== 'undefined' ? window : globalThis);
