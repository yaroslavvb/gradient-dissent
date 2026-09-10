const fs=require('fs'),path=require('path'),assert=require('assert'),{JSDOM}=require('jsdom');
const page=path.resolve(__dirname,'../../../docs/ciresan-stochastic-depth/conclusions');
const w=new JSDOM(fs.readFileSync(path.join(page,'index.html'),'utf8'),{runScripts:'outside-only'}).window,d=w.document;
w.eval(fs.readFileSync(path.join(page,'data.js'),'utf8'));w.eval(fs.readFileSync(path.join(page,'app.js'),'utf8'));
const data=w.CIRESAN_CONCLUSIONS;let count=0;
function change(id,v){d.getElementById(id).value=v;d.getElementById(id).dispatchEvent(new w.Event('change'));}
for(const endpoint of ['selected','final']){
 change('endpoint',endpoint);
 for(const id of ['quality-chart','digit-chart']){const s=d.getElementById(id);assert.equal(s.querySelectorAll('.point').length,6);assert(!/NaN|Infinity|undefined/.test(s.innerHTML));count++;}
 for(const id of ['quality-table','digit-table']){assert.equal(d.getElementById(id).querySelectorAll('tbody tr').length,3);count++;}
 const selected=data.aggregates.filter(a=>a.endpoint===endpoint);
 for(const a of selected){assert(d.getElementById('quality-table').textContent.includes((100*a.metrics.primary_mask_accuracy.mean).toFixed(3)));assert(d.getElementById('digit-table').textContent.includes((100*a.metrics.adjusted_digit1_minus_mean4_9_resilience.mean).toFixed(3)));count+=2;}
}
for(const [filter,n]of [['all',18],['sd',6],['controls',9],['baseline',3]]){change('run-filter',filter);assert.equal(d.getElementById('run-table').querySelectorAll('tbody tr').length,n);assert(!/NaN|undefined/.test(d.getElementById('run-table').innerHTML));count++;}
change('run-filter','all');const links=[...d.getElementById('run-table').querySelectorAll('a')];assert.equal(links.filter(a=>a.href.includes('github.com')).length,18);
assert.equal(links.filter(a=>a.href.includes('wandb.ai')).length,data.wandb.verified_run_count);
if(!data.wandb.complete)assert(d.getElementById('wandb-status').textContent.includes('sign-in'));
assert.equal(d.querySelectorAll('h1').length,1);assert(d.getElementById('details').textContent.includes('necessary'));assert(!d.querySelector('a[href="#"]'));
console.log(JSON.stringify({checks:count+6,runs:18,verified_wandb_links:data.wandb.verified_run_count,status:'passed'}));w.close();
