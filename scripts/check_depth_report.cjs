// Check published data, local links and every interactive report state without a browser.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require('jsdom');
const root = path.resolve(__dirname, '../docs/depth-robustness');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const data = JSON.parse(fs.readFileSync(path.join(root, 'data.json')));
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json')));
assert(!html.includes('{{'), 'Unfilled template');
assert.equal(manifest.cloud_spend_usd, 0);
assert.equal(manifest.final_tuning_runs, 90);
assert.equal(manifest.final_evaluation_runs, 45);
assert.equal(manifest.subset_evaluations, 2315);
const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push(e.message));
vc.on('error', e => errors.push(String(e)));
const dom = new JSDOM(html, {url: 'https://yaroslavvb.github.io/gradient-dissent/depth-robustness/', runScripts: 'outside-only', virtualConsole: vc});
const w = dom.window, d = w.document;
const ids = [...d.querySelectorAll('[id]')].map(e => e.id);
assert.equal(ids.length, new Set(ids).size, 'Duplicate IDs');
for (const el of d.querySelectorAll('[href], [src]')) {
  const value = el.getAttribute('href') || el.getAttribute('src');
  if (/^(https?:|mailto:|data:)/.test(value)) continue;
  const [local, hash] = value.split('#');
  let file = local ? path.resolve(root, local) : path.join(root, 'index.html');
  assert(fs.existsSync(file), `Missing local target: ${value}`);
  if (fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
  assert(fs.existsSync(file), `Missing directory index: ${value}`);
  if (hash) {
    const target = local ? new JSDOM(fs.readFileSync(file, 'utf8')).window.document : d;
    assert(target.getElementById(hash), `Missing anchor: ${value}`);
  }
}
for (const [taskName, task] of Object.entries(data.tasks)) {
  for (const row of task.summaries) {
    for (const metric of ['ce', 'excess_ce', 'accuracy']) {
      const s = row[metric];
      assert.equal(s.n, 5);
      assert([s.mean, s.ci95_low, s.ci95_high].every(Number.isFinite));
      assert(s.ci95_low <= s.mean && s.mean <= s.ci95_high);
    }
  }
  const count = task.mask_means.reduce((s, r) => s + r.seeds, 0);
  assert.equal(count, taskName === 'lm' ? 1335 : 980);
  for (const config of ['constant_ild', 'decreasing_ild']) {
    const p = task.paired.find(r => r.config === config && r.retained === 4 && r.mode === 'prefix');
    assert(p.excess_ce_difference_vs_dense6.ci95_high < 0);
  }
}
w.fetch = async url => {
  assert.equal(url, 'data.json');
  return {ok: true, json: async () => data};
};
w.eval(fs.readFileSync(path.join(root, 'app.js'), 'utf8'));
const get = id => d.getElementById(id);
const change = (id, value) => {
  get(id).value = value;
  get(id).dispatchEvent(new w.Event('change'));
};
(async () => {
  await new Promise(resolve => setTimeout(resolve, 20));
  let views = 0, subsetViews = 0;
  for (const taskName of ['lm', 'digits']) {
    change('task', taskName);
    const task = data.tasks[taskName];
    for (const mode of ['prefix', 'keep_first', 'all_subsets']) {
      for (const metric of ['ce', 'excess_ce', 'accuracy']) {
        change('mask-mode', mode);
        change('metric', metric);
        assert.equal(get('curve-table').querySelectorAll('tbody tr').length, task.config_order.length);
        assert(!/NaN|Infinity|undefined/.test(get('curve').outerHTML));
        const expected = task.summaries.filter(r => r.mode === mode);
        assert.equal(get('curve').querySelectorAll('circle').length, expected.length);
        for (const row of get('curve-table').querySelectorAll('tbody tr')) assert.equal(row.children.length, 7);
        assert(get('task-context').textContent.includes(taskName === 'lm' ? '312,448' : '1,797'));
        views++;
      }
    }
    const options = [...get('mask-config').options].map(o => o.value);
    assert(options.every(c => task.depths[c] === 6));
    for (const config of options) {
      change('mask-config', config);
      for (let k = 1; k <= 6; k++) {
        change('keep-count', k);
        const n = [0, 6, 15, 20, 15, 6, 1][k];
        assert(get('mask-note').textContent.startsWith(`${n} exact subsets`));
        assert.equal(get('mask-viz').querySelectorAll('rect').length, n * 7);
        assert(!/NaN|Infinity|undefined/.test(get('mask-viz').outerHTML));
        subsetViews++;
      }
    }
  }
  assert.equal(errors.length, 0, errors.join('\n'));
  dom.window.close();
  console.log(`PASS: local assets and links, ${views} curve states, ${subsetViews} subset states, saved counts, seed intervals and primary comparisons.`);
})().catch(error => { console.error(error); dom.window.close(); process.exitCode = 1; });
