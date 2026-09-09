// Static measured-data and interaction checks; never starts a browser or compute.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const assert=require('node:assert/strict');
const {JSDOM,VirtualConsole}=require('jsdom');
const root=path.resolve(__dirname,'../docs/ciresan-stochastic-depth/optimization');
const exp=path.resolve(__dirname,'../experiments/ciresan_stochastic_depth');
const read=f=>fs.readFileSync(path.join(root,f),'utf8');
const data=JSON.parse(read('data.json'));
assert.equal(data.runs.length,34);
for(const r of data.runs){
  const raw=fs.readFileSync(path.join(exp,'results',r.run_id+'.json'));
  assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),r.sha256);
  const first=r.history.find(p=>p.test.errors<=137);
  assert.equal(Boolean(r.threshold),Boolean(first));
  if(first){assert.equal(first.epoch,r.threshold.epoch);assert(first.training_seconds>0);assert(first.run_wall_seconds>=first.training_seconds);}
}
const errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e.message));
const dom=new JSDOM(read('index.html'),{runScripts:'outside-only',virtualConsole:vc});
const w=dom.window,d=w.document;
w.eval(read('data.js'));w.eval(read('app.js'));
for(const [g,n] of [['recommended',3],['kernels',3],['fragile',5],['raw',3]])for(const x of ['run_wall_seconds','training_seconds','epoch'])for(const z of ['detail','full']){
  for(const [id,value] of [['group',g],['clock',x],['zoom',z]]){d.getElementById(id).value=value;d.getElementById(id).dispatchEvent(new w.Event('change'));}
  assert.equal(d.querySelectorAll('.chart-path').length,n);
  for(const p of d.querySelectorAll('.chart-path'))assert(!/NaN|undefined|Infinity/.test(p.getAttribute('d')));
  assert.equal(d.getElementById('legend').children.length,n);
  assert(d.querySelectorAll('#chart-data tbody tr').length>n);
}
for(const e of d.querySelectorAll('[href],[src]')){
  const value=e.getAttribute('href')||e.getAttribute('src');
  if(/^(https?:|mailto:|data:|#)/.test(value))continue;
  const target=path.resolve(root,value.split('#')[0]);assert(fs.existsSync(target),'Missing local link '+value);
}
assert.deepEqual(errors,[]);w.close();
console.log('Optimization report: 34 raw hashes, first-target checks, 24 chart states, local links passed.');
