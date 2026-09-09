// Check report links, data bindings, math and interactions without a browser dependency.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {JSDOM,VirtualConsole}=require('jsdom');
const root=path.join(__dirname,'../docs');const data=JSON.parse(fs.readFileSync(path.join(root,'data/experiment-results.json')));
for(const file of ['index.html','review.html']){
 const dom=new JSDOM(fs.readFileSync(path.join(root,file),'utf8'));const doc=dom.window.document;
 const ids=[...doc.querySelectorAll('[id]')].map(e=>e.id);assert.equal(ids.length,new Set(ids).size,`${file}: duplicate IDs`);
 for(const a of doc.querySelectorAll('[href]')){let href=a.getAttribute('href');if(/^(https?:|mailto:)/.test(href))continue;let [local,hash]=href.split('#');const target=local?path.resolve(root,local):path.join(root,file);assert(fs.existsSync(target),`${file}: missing ${href}`);if(hash){const td=local?new JSDOM(fs.readFileSync(target,'utf8')).window.document:doc;assert(td.getElementById(hash),`${file}: bad anchor ${href}`);}}
 if(file==='review.html'){assert(doc.querySelectorAll('.katex').length===15,'Expected 15 rendered math expressions');assert(!doc.body.textContent.includes('MATHPLACEHOLDER'));}
}
const errors=[];const vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e));
const dom=new JSDOM(fs.readFileSync(path.join(root,'index.html'),'utf8'),{url:'https://yaroslavvb.github.io/gradient-dissent/',runScripts:'outside-only',pretendToBeVisual:true,virtualConsole:vc});
const w=dom.window,d=w.document;
w.IntersectionObserver=class{observe(){}};w.matchMedia=()=>({matches:true});w.scrollTo=()=>{};w.HTMLElement.prototype.scrollIntoView=function(){};w.fetch=async()=>({ok:true,json:async()=>data});
w.eval(fs.readFileSync(path.join(root,'app.js'),'utf8'));
(async()=>{
 await new Promise(r=>setTimeout(r,30));const get=id=>d.getElementById(id);const change=(id,value)=>{get(id).value=value;get(id).dispatchEvent(new w.Event('input'));get(id).dispatchEvent(new w.Event('change'));};
 assert.equal(d.querySelectorAll('#toc a').length,16);assert.equal(d.querySelectorAll('#training-table tbody tr').length,5);assert.equal(d.querySelectorAll('#runtime-table tbody tr').length,8);
 assert(get('runtime-table').textContent.includes('1.545'));assert(get('training-table').textContent.includes('0.01563'));
 change('train-view','early_exit_depth4_test_mse');assert(get('training-summary').textContent.includes('four-block'));assert(get('training-table').textContent.includes('0.02168'));
 change('train-view','test_mse');change('train-regime','matched_steps');assert(get('training-summary').textContent.includes('At equal steps'));assert(get('training-table').textContent.includes('0.02674'));
 change('schedule-kind','constant');assert(get('schedule-result').textContent.includes('40.00%'));assert(!get('schedule-formula').textContent.includes('T − 1'));change('schedule-kind','increasing');assert(get('schedule-formula').textContent.includes('t/(T − 1)'));change('schedule-kind','decreasing');
 change('bias-drop',40);assert(get('bias-result').textContent.includes('2.233'));assert(get('bias-result').textContent.includes('2.100'));change('bias-drop',0);assert(get('bias-result').textContent.includes('Bias: +0.000'));
 for(const input of d.querySelectorAll('input[type=range]')){for(const val of [input.min,input.max])change(input.id,val);}
 for(const svg of d.querySelectorAll('svg'))assert(!/NaN|Infinity|undefined/.test(svg.outerHTML),`Invalid chart ${svg.id}`);
 get('deck-toggle').click();assert(d.body.classList.contains('deck'));get('next-slide').click();assert.equal(get('slide-counter').textContent,'2 / 16');get('prev-slide').click();assert.equal(get('slide-counter').textContent,'1 / 16');get('deck-toggle').click();assert(!d.body.classList.contains('deck'));
 assert.equal(errors.length,0,errors.map(e=>e.message).join('\n'));
 dom.window.close();console.log('PASS: 16 chapters, all local links, 15 rendered equations, data tables, budget/depth controls, schedule formulas, exact counterexample, sliders and slide navigation.');
})().catch(e=>{console.error(e);dom.window.close();process.exitCode=1;});
