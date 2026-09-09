const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');
const dir = path.resolve(__dirname, '../docs/dropout-animation');
const M = require(path.join(dir, 'math.js'));
const close = (a, b, eps = 1e-10) => assert(Math.abs(a - b) < eps, `${a} != ${b}`);

// Enumerate a finite depth/time grid: the advertised formula must match its mean.
for (const layers of [2, 3, 12, 23]) for (const maximum of [0, .2, .8, .99]) {
  let total = 0;
  for (let t = 0; t <= 100; t++) {
    const ps = Array.from({length: layers}, (_, l) => M.probability(l, t / 100, maximum, layers));
    assert.equal(ps[0], 0);
    if (t === 100) assert(ps.every(p => p === 0));
    close(ps[layers - 1], maximum * (1 - t / 100));
    close(layers - ps.reduce((a, b) => a + b), M.expectedDepth(t / 100, maximum, layers));
    total += ps.reduce((a, b) => a + b);
  }
  close(total / (101 * layers), maximum / 4);
}
const settings = {mode: 'train', progress: 0, maximum: .8};
const batch = M.batch(settings, M.seededRandom(1));
assert.equal(batch.length, 4);
assert(batch.every(row => row.length === 12 && row[0].keep));
assert(new Set(batch.map(row => row.map(c => Number(c.keep)).join(''))).size > 1);
for (const c of batch.flat()) {
  close(c.scale, c.keep ? 1 / c.survival : 0);
  if (!c.keep) assert.deepEqual(M.scalarBlock(7, 2, 3, c.scale), {z: 7, y: 7});
}
// The same mask scales both residual updates; explicitly check the paper's subtle caveat.
assert.deepEqual(M.scalarBlock(1, 1, 1, 0), {z: 1, y: 1});
assert.deepEqual(M.scalarBlock(1, 1, 1, 2), {z: 3, y: 9});
close((1 + 9) / 2, 5);
assert.deepEqual(M.scalarBlock(1, 1, 1, 1), {z: 2, y: 4});
for (const state of [{...settings, maximum: 0}, {...settings, progress: 1}, {...settings, mode: 'dense'}]) {
  assert(M.batch(state, M.seededRandom(1)).flat().every(c => c.keep && c.scale === 1));
}
for (const retained of [1, 8, 12]) for (const row of M.batch({...settings, mode: 'exit', retained}, () => { throw Error('Inference must not draw masks'); })) {
  row.forEach((c, l) => assert.equal(c.scale, l < retained ? 1 : 0));
}
close(M.batch({...settings, maximum: .99}, () => .999)[0][11].scale, 100);

function harness(reduced = false) {
  const dom = new JSDOM(fs.readFileSync(path.join(dir, 'index.html'), 'utf8'), {runScripts: 'outside-only', url: 'https://yaroslavvb.github.io/gradient-dissent/dropout-animation/'});
  const w = dom.window, d = w.document, callbacks = new Map();
  let id = 0, time = 0, motionChanged;
  w.matchMedia = () => ({matches: reduced, addEventListener: (_, callback) => { motionChanged = callback; }});
  w.requestAnimationFrame = callback => { callbacks.set(++id, callback); return id; };
  w.cancelAnimationFrame = handle => callbacks.delete(handle);
  w.eval(fs.readFileSync(path.join(dir, 'math.js'), 'utf8'));
  w.eval(fs.readFileSync(path.join(dir, 'app.js'), 'utf8'));
  return {w, d, callbacks, dom, step(ms = 0) { time += ms; const current = [...callbacks.values()]; callbacks.clear(); current.forEach(fn => fn(time)); }, change(id, value) { const el = d.getElementById(id); el.value = value; el.dispatchEvent(new w.Event('input')); }, motionChanged};
}
const {w, d, callbacks, dom, step, change} = harness();
const el = id => d.getElementById(id);
const click = selector => d.querySelector(selector).dispatchEvent(new w.MouseEvent('click', {bubbles: true}));
const cells = () => [...d.querySelectorAll('.cell')];
assert.equal(cells().length, 48);
assert.equal(d.querySelectorAll('.cell[tabindex="0"]').length, 1);
assert.equal(el('expected').textContent, '7.2 / 12');
assert.equal(el('average').textContent, '20%');
assert.equal(callbacks.size, 0, 'Motion starts only on request');
for (const maximum of [0, 20, 80, 99]) for (const progress of [0, 25, 50, 100]) {
  change('maximum', maximum); change('progress', progress);
  assert.equal(cells().length, 48);
  assert(cells().every(c => !/NaN|Infinity/.test(c.getAttribute('aria-label'))));
  assert(cells().filter(c => c.dataset.layer === '0').every(c => c.getAttribute('aria-label').includes('kept')));
  const kept = cells().filter(c => c.getAttribute('aria-label').includes(': kept')).length;
  assert.equal(el('actual').textContent, `${kept} / 48`);
  if (progress === 100 || maximum === 0) assert.equal(kept, 48);
}
change('maximum', 80); change('progress', 0);
const original = cells().map(c => c.getAttribute('aria-label')).join();
click('#sample');
assert.notEqual(cells().map(c => c.getAttribute('aria-label')).join(), original);
click('.cell[data-sequence="2"][data-layer="6"]');
assert.equal(el('inspector-title').textContent, 'Sequence C · Block 7');
d.querySelector('.cell.selected').dispatchEvent(new w.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}));
assert.equal(el('inspector-title').textContent, 'Sequence C · Block 8');
assert.equal(d.querySelectorAll('.cell[tabindex="0"]').length, 1);
click('[data-mode="dense"]');
assert.equal(el('actual').textContent, '48 / 48');
assert(el('training-controls').hidden && el('sample').disabled);
assert.equal(el('selected-scale').textContent, '1×');
click('[data-mode="exit"]');
for (const retained of [1, 8, 12]) {
  change('retained', retained);
  assert.equal(el('actual').textContent, `${retained * 4} / 48`);
  cells().forEach(c => assert.equal(c.getAttribute('aria-label').includes(': kept'), +c.dataset.layer < retained));
}
click('[data-mode="train"]');
click('#animate'); assert.equal(callbacks.size, 1); step();
assert.equal(el('tokens-0').getAttribute('transform'), 'translate(113,143)');
step(1800);
assert.equal(callbacks.size, 1);
assert.equal(el('tokens-0').style.visibility, 'visible');
step(1800);
assert.equal(el('tokens-0').getAttribute('transform'), 'translate(1006,143)', 'Activations reach output head');
click('#animate'); assert.equal(callbacks.size, 0);
assert.equal(el('animate').getAttribute('aria-pressed'), 'false');
click('#animate'); step();
change('progress', 50); assert.equal(callbacks.size, 0, 'Editing controls stops the old animation');
change('progress', 0); click('#schedule'); step();
for (let i = 0; i < 11; i++) { step(3600); assert(callbacks.size <= 1); }
assert.equal(el('progress-value').textContent, '100%');
assert.equal(el('actual').textContent, '48 / 48');
assert.equal(callbacks.size, 0);
click('#schedule'); assert.equal(el('progress-value').textContent, '0%');
click('[data-mode="dense"]'); assert.equal(callbacks.size, 0, 'Mode switch cancels schedule');
click('#animate'); Object.defineProperty(d, 'hidden', {value: true, configurable: true});
d.dispatchEvent(new w.Event('visibilitychange')); assert.equal(callbacks.size, 0);
for (const node of d.querySelectorAll('a[href],script[src],link[href]')) {
  const ref = node.getAttribute('href') || node.getAttribute('src');
  if (/^https?:/.test(ref)) continue;
  const [pathname, fragment] = ref.split('#');
  if (!pathname) { assert(d.getElementById(fragment), ref); continue; }
  let target = path.resolve(dir, pathname);
  if (pathname.endsWith('/')) target = path.join(target, 'index.html');
  assert(fs.existsSync(target), ref);
}
assert(!/NaN|Infinity|undefined/.test(el('network').innerHTML + el('block-detail').innerHTML + el('schedule-chart').innerHTML));
dom.window.close();
const rm = harness(true);
assert(rm.d.getElementById('animate').disabled);
assert.equal(rm.callbacks.size, 0);
rm.d.getElementById('schedule').click(); rm.step(); rm.step(3600);
assert.equal(rm.d.getElementById('progress-value').textContent, '10%');
assert.equal(rm.d.getElementById('tokens-0').style.visibility, 'hidden');
rm.motionChanged(); assert.equal(rm.callbacks.size, 0);
rm.dom.window.close();
console.log('Dropout animation: exact depth/time averages, shared-mask scaling, inference, 16 slider combinations, keyboard selection, sampling, complete path animation, schedule completion, cancellation, reduced motion, and local links verified.');
