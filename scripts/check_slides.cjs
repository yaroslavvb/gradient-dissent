// Check the slide deck: formulas, navigation, every control, and local links, without a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM, VirtualConsole} = require('jsdom');
const dir = path.resolve(__dirname, '../docs/slides');
const close = (a, b, eps = 1e-9) => assert(Math.abs(a - b) < eps, `${a} != ${b}`);
const errors = [];
const vc = new VirtualConsole(); vc.on('jsdomError', e => errors.push(e));
const dom = new JSDOM(fs.readFileSync(path.join(dir, 'index.html'), 'utf8'), {runScripts: 'outside-only', url: 'https://yaroslavvb.github.io/gradient-dissent/slides/', pretendToBeVisual: true, virtualConsole: vc});
const w = dom.window, d = w.document;
w.scrollTo = () => {};
w.eval(fs.readFileSync(path.join(dir, 'app.js'), 'utf8'));
const M = w.SlidesMath, el = id => d.getElementById(id);
const change = (id, value) => { el(id).value = value; el(id).dispatchEvent(new w.Event('input')); };
const click = sel => d.querySelector(sel).dispatchEvent(new w.MouseEvent('click', {bubbles: true}));
const key = k => d.dispatchEvent(new w.KeyboardEvent('keydown', {key: k, bubbles: true}));

// Formulas: every distribution has the requested mean; the recipe averages to pmax/4; scaling is 1/p.
for (const L of [8, 12, 16, 32]) for (const s of [0, .05, .2, .45]) for (const kind of ['uniform', 'increasing', 'alternating']) {
  const ps = M.distribution(kind, s, L);
  assert.equal(ps.length, L); close(ps.reduce((a, b) => a + b, 0) / L, s, 1e-9);
  if (kind !== 'uniform') assert.equal(ps[0], 0);
  assert(ps.every(p => p >= 0 && p <= 1));
}
for (const kind of ['constant', 'increasing', 'decreasing']) for (const pmax of [0, .6, .8, .99]) {
  let total = 0, n = 0; const L = 12;
  for (let t = 0; t <= 1.00001; t += .01) for (let l = 0; l < L; l++) { total += M.recipe(kind, l, t, pmax, L); n++; }
  close(total / n, M.meanSaved(kind, pmax), 1e-2);
  close(M.recipe(kind, L - 1, 0, pmax, L), M.schedule(kind, 0, pmax));
  assert.equal(M.recipe(kind, 0, .5, pmax, L), 0);
}
close(M.survivorScale(.6), 1 / .6); close(M.effectiveDepth(.6, 32), 19.2);
close(M.weightsRead('batch', .5, 8), .5); close(M.weightsRead('sequence', .5, 8), 1 - 1 / 256); close(M.weightsRead('sequence', 0, 8), 1);
const batch = M.masks('batch', Array(12).fill(.5), 8, M.seeded(1));
assert(batch.every(row => row.join() === batch[0].join()), 'Per-batch masks are shared by every sequence');
const seq = M.masks('sequence', Array(12).fill(.5), 8, M.seeded(1));
assert(new Set(seq.map(r => r.join())).size > 1, 'Per-sequence masks differ');

// Navigation.
const N = d.querySelectorAll('.slide').length;
assert.equal(N, 13); assert.equal(d.querySelectorAll('#toc a').length, N);
assert.equal(el('slide-counter').textContent, `1 / ${N}`); assert(d.querySelectorAll('.slide')[0].classList.contains('current'));
click('#next-slide'); assert.equal(el('slide-counter').textContent, `2 / ${N}`); assert.equal(w.location.hash, '#/2');
key('ArrowRight'); assert.equal(el('slide-counter').textContent, `3 / ${N}`);
key('End'); assert.equal(el('slide-counter').textContent, `${N} / ${N}`); assert(el('next-slide').disabled);
key('ArrowRight'); assert.equal(el('slide-counter').textContent, `${N} / ${N}`);
key('Home'); assert.equal(el('slide-counter').textContent, `1 / ${N}`); assert(el('prev-slide').disabled);
w.location.hash = '#/7'; w.dispatchEvent(new w.Event('hashchange')); assert.equal(el('slide-counter').textContent, `7 / ${N}`);
assert(d.title.startsWith('7. Knob 4'));
assert(el('toc').hidden); key('o'); assert(!el('toc').hidden); assert.equal(el('toc-toggle').getAttribute('aria-expanded'), 'true'); key('Escape'); assert(el('toc').hidden);

// Controls: every slider at its ends, every toggle, no NaN anywhere.
for (const input of d.querySelectorAll('input[type=range]')) for (const v of [input.min, input.max, input.value]) change(input.id, v);
change('keep', 60); change('depth', 32); assert(el('scale-formula').textContent.includes('1.67×')); assert(el('scale-formula').textContent.includes('19.2'));
click('[data-gran="batch"]'); assert.equal(d.querySelector('[data-gran="batch"]').getAttribute('aria-pressed'), 'true');
change('gran-rate', 30); assert(el('gran-summary').textContent.includes('same'));
const before = el('gran-chart').innerHTML; click('#gran-sample'); assert.notEqual(el('gran-chart').innerHTML, before);
click('[data-gran="sequence"]'); assert(el('gran-summary').textContent.includes('independent masks'));
assert.equal(d.querySelectorAll('#gran-chart .cell').length, 96);
change('savings', 20); change('dist-depth', 16); assert.equal(d.querySelectorAll('#dist-chart .bar').length, 48); assert(el('dist-summary').textContent.includes('40% at layer 16'));
change('savings', 45); assert(!el('dist-summary').textContent.includes('capped')); change('savings', 20);
click('[data-sched="increasing"]'); assert(el('sched-summary').textContent.includes('worst')); change('progress-t', 100); change('pmax', 80); assert(el('sched-summary').textContent.includes('skipped 80%'));
click('[data-sched="decreasing"]'); assert(el('sched-summary').textContent.includes('always present')); change('progress-t', 0); assert(el('sched-summary').textContent.includes('skipped 80%')); assert(el('sched-summary').textContent.includes('7.2 of 12'));
assert.equal(d.querySelectorAll('#sched-chart .cell').length, 240);
change('hero-pmax', 99); assert.equal(el('hero-saved').textContent, '24.75%'); assert.equal(el('hero-last').textContent, '99%');
change('bytes-rate', 50); change('bytes-seqs', 8); assert(el('bytes-chart').textContent.includes('99.6%')); assert(el('bytes-chart').textContent.includes('50.0%'));
assert.equal(d.querySelectorAll('#elastic-chart .bar').length, 7);
click('#block-sample'); assert(/kept|skipped/.test(el('block-figure').textContent));
for (const svg of d.querySelectorAll('svg')) assert(!/NaN|Infinity|undefined/.test(svg.outerHTML), `Invalid chart ${svg.id}`);
for (const id of ['scale-formula', 'gran-summary', 'dist-summary', 'sched-summary', 'hero-note']) assert(!/NaN|undefined/.test(el(id).textContent), id);

// Local links and IDs.
const ids = [...d.querySelectorAll('[id]')].map(e => e.id); assert.equal(ids.length, new Set(ids).size, 'duplicate IDs');
for (const node of d.querySelectorAll('a[href],script[src],link[href]')) {
  const ref = node.getAttribute('href') || node.getAttribute('src');
  if (/^https?:/.test(ref)) continue;
  if (ref.startsWith('#/')) { assert(+ref.slice(2) >= 1 && +ref.slice(2) <= N, ref); continue; }
  if (ref.startsWith('#')) { assert(d.getElementById(ref.slice(1)), ref); continue; }
  let target = path.resolve(dir, ref); if (ref.endsWith('/')) target = path.join(target, 'index.html');
  assert(fs.existsSync(target), ref);
}
assert.equal(errors.length, 0, errors.map(e => e.message).join('\n'));
dom.window.close();
console.log(`Slides: ${N} slides, matched-mean distributions, recipe average pmax/4, 1/p scaling, per-batch vs per-sequence masks, weight-read formula, hash/keyboard/button navigation, contents, every control at both ends, measured-loss chart, no NaN, local links verified.`);
