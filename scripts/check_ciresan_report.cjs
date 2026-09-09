#!/usr/bin/env node
// Programmatic checks only: no browser, service, training, or paid calls.
const fs=require('node:fs'), path=require('node:path'), crypto=require('node:crypto');
const assert=require('node:assert/strict');
const {JSDOM,VirtualConsole}=require('jsdom');
const root=path.resolve(__dirname,'../docs/ciresan-stochastic-depth');
const exp=path.resolve(__dirname,'../experiments/ciresan_stochastic_depth');
const read=name=>fs.readFileSync(path.join(root,name));
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const data=JSON.parse(read('data.json')), manifest=JSON.parse(read('manifest.json'));
const pilot=process.argv.includes('--pilot');
assert.equal(data.final,!pilot,'Final report required unless --pilot explicitly supplied');
assert.equal(hash(read('data.json')),manifest.data_sha256);
assert.equal(data.runs.length,pilot?6:30);
assert.equal(new Set(data.runs.map(r=>r.run_id)).size,data.runs.length);
for(const file of manifest.raw_files){assert.equal(hash(fs.readFileSync(path.join(exp,'results',file.run_id+'.json'))),file.sha256);}
for(const run of data.runs){
  assert.equal(run.parameter_count,11972510);
  assert.equal(run.steps_per_epoch,781);
  assert.equal(run.history.at(-1).epoch,run.spec.epochs);
  assert.equal(run.spec.epochs,run.spec.stage==='pilot'?5:100);
  assert(run.training_seconds>0&&run.total_run_seconds>=run.training_seconds);
  for(const row of run.history){
    assert(!row.diverged);assert(Number.isFinite(row.validation.loss));
    assert(row.validation.accuracy>=0&&row.validation.accuracy<=1);
    assert(row.drop_probabilities.every(v=>v>=0&&v<1));
  }
  if(run.spec.stage!=='evaluate')assert(!run.final_test&&!run.selected_test);
  else for(const key of ['final_test','selected_test']){
    const r=run[key];assert.equal(r.n,10000);assert.equal(r.errors,r.wrong_indices.length);
    assert(Math.abs(r.accuracy-(1-r.errors/r.n))<1e-12);
    assert.equal(new Set(r.wrong_indices).size,r.errors);
  }
}
const errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e.message));
const dom=new JSDOM(read('index.html').toString(),{runScripts:'outside-only',url:'https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/',virtualConsole:vc});
const w=dom.window,d=w.document,$=id=>{const e=d.getElementById(id);assert(e,id);return e;};
w.eval(read('data.js').toString());w.eval(read('app.js').toString());
const ids=[...d.querySelectorAll('[id]')].map(n=>n.id);assert.equal(ids.length,new Set(ids).size);
const change=(id,value)=>{$(id).value=value;$(id).dispatchEvent(new w.Event('change'));};
for(const cohort of pilot?['pilot']:['main','fidelity','pilot']){
  change('cohort',cohort);
  const expected={main:5,fidelity:3,pilot:6}[cohort];
  for(const metric of ['validation_accuracy','validation_loss','train_accuracy','train_loss','stochastic_loss'])for(const x of ['epoch','seconds']){
    change('metric',metric);change('xaxis',x);
    assert.equal($('legend').children.length,expected);
    assert.equal($('curve-table').querySelectorAll('tbody tr').length,expected);
    assert($('curves').querySelectorAll('path').length>=expected);
    for(const p of $('curves').querySelectorAll('path'))assert(!/NaN|Infinity|undefined/.test(p.getAttribute('d')));
    assert($('svg-title').textContent.length>5);assert($('curve-note').textContent.length>60);
  }
  $('individual').checked=false;$('individual').dispatchEvent(new w.Event('change'));
  assert.equal($('curves').querySelectorAll('path').length,expected);
  $('individual').checked=true;$('individual').dispatchEvent(new w.Event('change'));
}
for(const element of d.querySelectorAll('[href],[src]')){
  const value=element.getAttribute('href')||element.getAttribute('src');
  if(/^(https?:|mailto:|data:)/.test(value))continue;
  const [local,anchor]=value.split('#');let target=path.resolve(root,local||'index.html');
  assert(fs.existsSync(target),'Missing '+value);
  if(fs.statSync(target).isDirectory())target=path.join(target,'index.html');
  assert(fs.existsSync(target),'Missing index '+value);
  if(anchor){const doc=local?new JSDOM(fs.readFileSync(target,'utf8')).window.document:d;assert(doc.getElementById(anchor),'Missing anchor '+value);}
}
assert.deepEqual(errors,[]);dom.window.close();
console.log('Ciresan report passed: raw-data hashes, endpoint counts, '+(pilot?10:30)+' chart combinations, readable tables and local links.');
