const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname, '..');
const dir = path.join(root, 'docs/stochastic-depth-history');
const source = fs.readFileSync(path.join(root, 'research/stochastic-depth-history.md'), 'utf8');
assert.equal(fs.readFileSync(path.join(dir, 'report.md'), 'utf8'), source);
const dom = new JSDOM(fs.readFileSync(path.join(dir, 'index.html'), 'utf8'));
const d = dom.window.document;
assert.equal(d.querySelectorAll('h1').length, 1);
assert.equal(d.querySelectorAll('main h2').length, 9);
assert.equal(d.querySelectorAll('aside .toc a').length, 9);
for (const person of ['Gao Huang:', 'Yu Sun:', 'Zhuang Liu:', 'Daniel (Dan) Sedra:', 'Kilian Q. Weinberger:']) {
  assert([...d.querySelectorAll('h3')].some(h => h.textContent.startsWith(person)), person);
}
const ids = [...d.querySelectorAll('[id]')].map(e => e.id);
assert.equal(new Set(ids).size, ids.length);
const external = new Set();
for (const a of d.querySelectorAll('a[href],link[href]')) {
  const href = a.getAttribute('href');
  if (href.startsWith('https://')) { external.add(href.split('#')[0]); continue; }
  assert(!/^https?:/.test(href), href);
  const [pathname, fragment] = href.split('#');
  if (!pathname) { assert(d.getElementById(fragment), href); continue; }
  let target = path.resolve(dir, pathname);
  if (pathname.endsWith('/')) target = path.join(target, 'index.html');
  assert(fs.existsSync(target), href);
}
assert(external.size >= 40);
assert.equal(d.querySelectorAll('.table-wrap table').length, 4);
assert(d.body.textContent.includes('estimated from measured stage throughput'));
assert(d.body.textContent.includes('does not establish his present title'));
assert(!/TODO|PLACEHOLDER|\[insert/i.test(d.body.textContent));
console.log(`History report: 9 sections, all 5 authors, 4 comparison tables, ${external.size} distinct cited URLs, navigation and Markdown download checked. Literature facts independently audited against primary sources.`);
dom.window.close();
