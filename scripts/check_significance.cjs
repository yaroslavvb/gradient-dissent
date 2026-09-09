const fs=require('fs'),path=require('path'),assert=require('assert/strict'),{JSDOM}=require('jsdom');
const root=path.resolve(__dirname,'..'),dir=path.join(root,'docs/significance');
const dom=new JSDOM(fs.readFileSync(path.join(dir,'index.html'),'utf8'),{runScripts:'outside-only',url:'https://yaroslavvb.github.io/gradient-dissent/significance/'});
const w=dom.window,d=w.document;w.eval(fs.readFileSync(path.join(dir,'data.js'),'utf8'));w.eval(fs.readFileSync(path.join(dir,'app.js'),'utf8'));
const raw=JSON.parse(fs.readFileSync(path.join(root,'experiments/a100_transfer/results/summary.json')));assert.deepEqual(JSON.parse(JSON.stringify(w.SIGNIFICANCE_DATA.curves)),raw.curves);
let states=0;for(const t of ['vit_cifar100','convnext_cifar100','gpt_wikitext103'])for(const m of ['ce','accuracy'])for(const mask of ['primary_two_thirds','random_keeps_first_0','random_keeps_first_1','random_keeps_first_2','delete_first_only']){
 for(const [id,v]of [['family',t],['metric',m],['mask',mask]]){d.getElementById(id).value=v;d.getElementById(id).dispatchEvent(new w.Event('change'))}
 assert.equal(d.querySelectorAll('#score-table tr').length,3);let i=0;
 for(const r of ['dense','constant_ild','decreasing_ild']){const cells=d.querySelectorAll('#score-table tr')[i++].querySelectorAll('td');for(const [j,mk]of [[0,'full'],[1,mask]]){const v=raw.curves.find(c=>c.task===t&&c.recipe===r&&c.mask_name===mk)[m].mean*(m==='accuracy'?100:1);assert.equal(cells[j].textContent,v.toFixed(m==='accuracy'?2:3)+(m==='accuracy'?'%':''))}}
 assert(!d.querySelector('#quality-chart').innerHTML.includes('NaN'));assert(d.getElementById('insight').textContent.length>80);states++;
}
for(const p of [0,50,100]){d.getElementById('progress').value=p;d.getElementById('progress').dispatchEvent(new w.Event('input'));assert.equal(d.querySelectorAll('.block').length,12);assert(d.getElementById('schedule-text').textContent.includes((12*(1-.4*(1-p/100))).toFixed(1)))}
for(const a of d.querySelectorAll('a[href],script[src],link[href]')){let v=a.getAttribute('href')||a.getAttribute('src');if(v.startsWith('#')){assert(d.getElementById(v.slice(1)),v);continue}if(/^(https?:|mailto:)/.test(v))continue;let local=path.resolve(dir,v.split('#')[0]);if(v.endsWith('/'))local=path.join(local,'index.html');assert(fs.existsSync(local),v)}
assert.equal(d.querySelectorAll('#effect-table tr').length,3);console.log(`Significance report: ${states} chart states, 180 endpoint values, schedule and local links verified.`);
