// Six-affine page integration checks. No browser, cloud calls or screenshots.
// Run after the verified exporter and HTML builder have completed:
//   node experiments/ciresan_stochastic_depth/layerdrop_hypotheses/test_full_depth_page.cjs
// An optional first argument points to a separate directory for fixture checks.
'use strict';
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const {JSDOM, VirtualConsole} = require('jsdom');

const dir = path.resolve(process.argv[2] || path.join(__dirname, '../../../docs/ciresan-stochastic-depth/hypotheses'));
for (const name of ['index.html', 'data.js', 'app.js', 'extended-data.js', 'full-depth.js']) {
  assert(fs.existsSync(path.join(dir, name)), `Missing ${name}; build verified extended data and the page before this check.`);
}
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', error => errors.push(error));
const dom = new JSDOM(fs.readFileSync(path.join(dir, 'index.html'), 'utf8'), {
  runScripts: 'outside-only', virtualConsole,
  url: 'https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/hypotheses/',
});
const w = dom.window;
const $ = id => {
  const element = w.document.getElementById(id);
  assert(element, `Missing page element #${id}`);
  return element;
};
w.HTMLCanvasElement.prototype.getContext = function () {
  return {
    createImageData: (x, y) => ({data: new Uint8ClampedArray(x * y * 4)}),
    putImageData() {}, drawImage() {},
  };
};
const pct = value => value == null ? '—' : (100 * value).toFixed(2) + '%';
const close = (a, b, context, tolerance = 2e-12) => {
  assert(Number.isFinite(a) && Number.isFinite(b), `${context}: nonfinite ${a}, ${b}`);
  assert(Math.abs(a - b) <= tolerance, `${context}: ${a} != ${b}`);
};
const change = (id, value, event = 'change') => {
  $(id).value = String(value);
  $(id).dispatchEvent(new w.Event(event, {bubbles: true}));
};
const chooseModel = model => {
  for (const [id, value] of [['recipe', model.recipe], ['state', model.state], ['seed', model.seed]]) {
    $(id).value = String(value);
  }
  $('recipe').dispatchEvent(new w.Event('change', {bubbles: true}));
  assert.equal(w.fullDepthExplorer.getModel().id, model.id);
  assert.equal(w.hypothesisExplorer.getModel().id, model.id, 'Both explorers must use the same checkpoint');
};
function finiteDisplay() {
  for (const id of ['full-correct', 'full-coverage', 'full-accuracy', 'full-cost', 'full-class-table', 'full-mask-table']) {
    const element = $(id), numericText = element.querySelector('tbody')?.textContent ?? element.textContent;
    assert(!/\b(?:NaN|Infinity|null|undefined)\b/.test(numericText), `Invalid numeric display in ${id}`);
  }
  // Explanatory prose legitimately says CE is undefined. SVG coordinates do not.
  for (const id of ['full-network', 'full-chart']) {
    assert(!/NaN|Infinity|null|undefined/.test($(id).innerHTML), `Invalid SVG in ${id}`);
    assert($(id).getAttribute('aria-label'), `Missing chart description: ${id}`);
  }
}
function assertScores(row) {
  assert.equal($('full-correct').textContent, pct(row.correct_output_rate));
  assert.equal($('full-coverage').textContent, pct(row.coverage));
  assert.equal($('full-accuracy').textContent, pct(row.accuracy));
  assert.equal($('full-cost').textContent, pct(row.cost));
}
function expectedAggregate(masks, weights, digit = null) {
  const covered = masks.filter((m, i) => (digit == null ? m : m.per_class[digit]).coverage && weights[i]);
  const coverage = covered.reduce((sum, m) => sum + weights[m.id], 0);
  const correct = masks.reduce((sum, m) => sum + weights[m.id] * (digit == null ? m : m.per_class[digit]).correct_output_rate, 0);
  return {
    coverage, correct_output_rate: correct,
    accuracy: coverage ? correct / coverage : null,
    ce: coverage ? covered.reduce((sum, m) => sum + weights[m.id] * (digit == null ? m : m.per_class[digit]).ce, 0) / coverage : null,
    cost: masks.reduce((sum, m) => sum + weights[m.id] * m.cost, 0),
  };
}
function assertAggregate(actual, expected, context) {
  for (const key of ['coverage', 'correct_output_rate', 'accuracy', 'ce', 'cost']) {
    if (expected[key] == null) assert.equal(actual[key], null, `${context}/${key}`);
    else close(actual[key], expected[key], `${context}/${key}`);
  }
}

try {
  for (const name of ['data.js', 'app.js', 'extended-data.js', 'full-depth.js']) {
    w.eval(fs.readFileSync(path.join(dir, name), 'utf8'));
  }
  assert(w.fullDepthExplorer, 'Extended explorer did not initialize');
  const api = w.fullDepthExplorer;
  const data = w.HYPOTHESIS_EXTENDED;
  const original = w.HYPOTHESIS_DATA;
  assert.equal(data.models.length, 30);
  assert.equal(new Set(data.models.map(m => m.id)).size, 30);
  assert.equal(original.models.length, 30);
  const scripts = Array.from(w.document.querySelectorAll('script[src]'), s => s.getAttribute('src'));
  for (const file of ['data.js', 'app.js', 'extended-data.js', 'full-depth.js']) {
    assert(scripts.some(src => path.posix.basename(src.split('?')[0]) === file), `HTML does not load ${file}`);
  }

  let maskChecks = 0;
  for (const model of data.models) {
    chooseModel(model);
    const old = original.models.find(m => m.id === model.id);
    assert(old, `No original model ${model.id}`);
    assert.equal(model.masks.length, 64);
    assert.equal(model.n, 10000);
    assert.equal($('full-mask-table').querySelectorAll('tbody tr').length, 64);
    assert.deepEqual(Array.from(model.examples, e => e.index), Array.from(old.examples, e => e.index));
    assert(model.examples.length > 0, `No frozen gallery for ${model.id}`);

    // Match every old body mask, including full63=old15 and shallow33=old0.
    for (let prior = 0; prior < 16; prior++) {
      const current = model.masks[33 + 2 * prior], archived = old.masks[prior];
      assert.equal(current.accuracy, archived.accuracy, `${model.id}/old${prior} accuracy`);
      close(current.ce, archived.ce, `${model.id}/old${prior} CE`, 3e-5);
      close(current.cost, archived.cost, `${model.id}/old${prior} cost`);
      for (const e of model.examples) {
        assert.equal(e.pred[33 + 2 * prior], old.examples.find(o => o.index === e.index).pred[prior]);
      }
    }

    for (let mask = 0; mask < 64; mask++) {
      const row = model.masks[mask];
      assert.equal(row.id, mask);
      api.setMask(mask);
      assert.equal($('layer-scope').value, 'full');
      assert.equal($('inference-mode').value, 'fixed');
      assert.equal($('full-explorer').hidden, false);
      assert.equal($('body-explorer').hidden, true);
      assertScores(row);
      const inputs = $('full-mask-controls').querySelectorAll('input[data-full-layer]');
      assert.equal(inputs.length, 6);
      inputs.forEach((input, i) => {
        assert.equal(input.checked, !!(mask & (1 << i)));
        assert.equal(input.disabled, false);
      });
      assert.equal($('full-class-table').querySelectorAll('tbody tr').length, 10);
      assert.equal($('full-chart').querySelectorAll('[data-point-mask]').length, 33);
      assert.equal($('full-chart').querySelectorAll('[data-point-mask="32"]').length, 1, 'Classifier-only measurement must be plotted');
      assert($('full-mask-status').textContent.includes(`${row.executed_affines}/6`));
      const classRows = $('full-class-table').querySelectorAll('tbody tr');
      classRows.forEach((tr, digit) => {
        const cells = tr.querySelectorAll('td'), cls = row.per_class[digit];
        assert.equal(cells[0].textContent, pct(cls.correct_output_rate));
        assert.equal(cells[1].textContent, pct(cls.coverage));
        assert.equal(cells[2].textContent, pct(cls.accuracy));
        assert.equal(cells[3].textContent, cls.ce == null ? '—' : cls.ce.toFixed(3));
      });
      const example = model.examples.find(e => e.index === Number($('full-example-select').value));
      assert(example);
      if (mask < 32) {
        assert.equal(row.coverage, 0); assert.equal(row.correct_output_rate, 0);
        assert.equal(row.accuracy, null); assert.equal(row.ce, null);
        assert.equal(row.raw_macs, 0); assert.equal(row.executed_affines, 0);
        assert.equal(example.pred[mask], -1); assert.equal(example.ce[mask], null);
        assert.match($('full-prediction').textContent, /^No prediction/);
        assert.match($('full-example-note').textContent, /undefined/);
        assert.match($('full-network').getAttribute('aria-label'), /abstain/i);
      } else {
        assert.equal(row.coverage, 1); assert.equal(row.accuracy, row.correct_output_rate);
        assert(Number.isFinite(row.ce));
        assert.match($('full-prediction').textContent, new RegExp(`^Predicts ${example.pred[mask]} \\u00b7`));
      }
      finiteDisplay();
      maskChecks++;
    }
    assert.equal(model.masks[32].raw_macs, 5000);
    assert.equal(model.masks[32].executed_affines, 1);
    assert.equal(model.masks[33].raw_macs, 1965000);

    // Independent mixture checks: endpoints, normalization and exact support.
    for (const candidates of [0, 1, 30, 32, 63]) for (const p of [0, 0.5, 1]) {
      const ws = Array.from(api.weights(p, candidates));
      close(ws.reduce((a, b) => a + b, 0), 1, 'Weight normalization');
      const free = candidates.toString(2).replace(/0/g, '').length;
      const expectedMask = p === 1 ? 63 ^ candidates : 63;
      ws.forEach((weight, mask) => {
        assert(weight >= 0 && weight <= 1);
        const allowed = (mask & (63 ^ candidates)) === (63 ^ candidates);
        const expected = p === 0.5 ? (allowed ? 2 ** -free : 0) : +(mask === expectedMask);
        close(weight, expected, 'Mask weight');
      });
      for (const digit of [null, 1, 4, 9]) {
        const actual = api.aggregate(model.masks, ws, digit);
        assertAggregate(actual, expectedAggregate(model.masks, ws, digit), `${model.id}/p${p}/eligible${candidates}`);
        const coverage = candidates & 32 ? 1 - p : 1;
        close(actual.coverage, coverage, 'Analytical coverage');
        if (actual.accuracy == null) assert.equal(actual.correct_output_rate, 0);
        else close(actual.correct_output_rate, coverage * actual.accuracy, 'Correct-output/conditional-accuracy identity');
      }
    }
  }
  assert.equal(maskChecks, 1920);

  // Test real controls and event handlers rather than only the exposed API.
  chooseModel(data.models[0]);
  api.setMask(63);
  $('full-head-only').click();
  assertScores(api.getModel().masks[32]);
  assert.match($('full-explanation').textContent, /pixels.*zeros/i);
  $('full-keep').click(); assertScores(api.getModel().masks[63]);
  let checkbox = $('full-mask-controls').querySelector('[data-full-layer="5"]');
  checkbox.click(); assertScores(api.getModel().masks[31]);
  assert.match($('full-prediction').textContent, /No prediction/);
  $('full-mask-table').querySelector('[data-full-mask="33"]').click();
  assertScores(api.getModel().masks[33]);
  $('full-chart').querySelector('[data-point-mask="32"]').dispatchEvent(new w.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
  assertScores(api.getModel().masks[32]);
  $('full-drop').click(); assertScores(api.getModel().masks[0]);
  $('full-play').click();
  assert.equal($('full-play').getAttribute('aria-pressed'), 'true');
  assertScores(api.getModel().masks[63]);
  api.stop(); assert.equal($('full-play').getAttribute('aria-pressed'), 'false');

  change('layer-scope', 'body'); change('inference-mode', 'fixed');
  assert.equal($('body-explorer').hidden, false);
  assert.equal($('full-explorer').hidden, true);
  assert.equal(w.document.querySelectorAll('[data-branch]').length, 4);
  change('inference-mode', 'random');
  assert.equal($('body-explorer').hidden, true);
  assert.equal($('full-explorer').hidden, false);
  $('full-drop').click();
  change('drop-probability', 100, 'input');
  assert.equal(api.getPreviewMask(), 33);
  assertScores(api.getModel().masks[33]);
  for (const i of [0, 5]) assert($('full-mask-controls').querySelector(`[data-full-layer="${i}"]`).disabled);
  assert.equal($('random-controls').hidden, false);
  assert.equal($('full-play').hidden, true);
  finiteDisplay();

  change('layer-scope', 'full'); $('full-drop').click();
  for (const p of [0, 50, 100]) {
    change('drop-probability', p, 'input');
    assertScores(api.aggregate(api.getModel().masks, api.weights(p / 100, 63)));
    assert.equal($('drop-probability-value').textContent, `${p}%`);
    assert.equal($('full-chart').querySelectorAll('circle').length, 0);
    finiteDisplay();
  }
  assert.equal(api.getPreviewMask(), 0);
  assert.match($('full-prediction').textContent, /No prediction/);
  assert.equal($('full-accuracy').textContent, '—');
  $('full-keep').click();
  assert.equal(api.getPreviewMask(), 63);
  assertScores(api.getModel().masks[63]);
  assert(Array.from($('full-mask-controls').querySelectorAll('input')).every(i => !i.checked));

  // In random mode, checked means eligible; unchecking the head keeps coverage 1.
  $('full-drop').click();
  $('full-mask-controls').querySelector('[data-full-layer="5"]').click();
  assert.equal(api.getPreviewMask(), 32);
  assertScores(api.getModel().masks[32]);
  change('drop-probability', 50, 'input');
  assertScores(api.aggregate(api.getModel().masks, api.weights(0.5, 31)));
  assert.equal($('full-coverage').textContent, '100.00%');
  const originalRandom = w.Math.random;
  try {
    w.Math.random = () => 0.25; $('full-draw').click();
    assert.equal(api.getPreviewMask(), 32);
    w.Math.random = () => 0.75; $('full-draw').click();
    assert.equal(api.getPreviewMask(), 63);
    // The sampled diagram changes; the exact mixture scores do not.
    assertScores(api.aggregate(api.getModel().masks, api.weights(0.5, 31)));
  } finally { w.Math.random = originalRandom; }
  const lastExample = api.getModel().examples.at(-1);
  change('full-example-select', lastExample.index);
  assert.match($('full-digit').getAttribute('aria-label'), new RegExp(`image ${lastExample.index},`));
  finiteDisplay();
  assert.equal(errors.length, 0, errors.map(e => e.message).join('\n'));
  console.log(`PASS: ${data.models.length} checkpoints ×64 masks (${maskChecks} DOM states), original16 parity, abstention, exact random expectations, scope/checkbox/draw/animation controls and finite SVGs.`);
} finally {
  if (w.fullDepthExplorer) w.fullDepthExplorer.stop();
  if (w.hypothesisExplorer) w.hypothesisExplorer.stop();
  w.close();
}
