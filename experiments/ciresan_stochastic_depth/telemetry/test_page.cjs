const fs=require('fs'),path=require('path'),assert=require('assert'),{JSDOM}=require('jsdom');
const page=path.resolve(__dirname,'../../../docs/ciresan-stochastic-depth/telemetry');
const dom=new JSDOM(fs.readFileSync(path.join(page,'index.html'),'utf8'),{runScripts:'outside-only',url:'https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/telemetry/'});
const w=dom.window,d=w.document;
w.eval(fs.readFileSync(path.join(page,'data.js'),'utf8'));w.eval(fs.readFileSync(path.join(page,'app.js'),'utf8'));
function change(id,value){d.getElementById(id).value=value;d.getElementById(id).dispatchEvent(new w.Event('change'));}
function healthy(id){const svg=d.getElementById(id);assert(!/NaN|Infinity|undefined/.test(svg.innerHTML),id+' contains invalid SVG');return svg.querySelectorAll('.point').length;}
let cases=0;assert(healthy('accuracy')>0);assert(healthy('history-chart')===91,'historical curve must include all91 observations');
for(const cohort of ['controlled','baseline'])for(const axis of ['epoch','training_seconds','run_elapsed_seconds'])for(const split of ['validation','test','train']){
 change('cohort',cohort);change('axis',axis);change('split',split);healthy('accuracy');healthy('loss');healthy('class-chart');cases++;
}
change('cohort','controlled');change('axis','epoch');change('seed','101');
for(const key of w.telemetryUI.metricKeys){change('metric',key);healthy('metric-chart');cases++;}
d.getElementById('metric-search').value='no-such-metric-key';d.getElementById('metric-search').dispatchEvent(new w.Event('input'));assert(d.getElementById('metric-note').textContent.includes('No metric'));healthy('metric-chart');
d.getElementById('metric-search').value='';d.getElementById('metric-search').dispatchEvent(new w.Event('input'));
for(let layer=0;layer<6;layer++)for(const family of ['ce_ggn','empirical_fisher','jacobian_gram','activation_factor','gradient_moments','output_mismatch']){
 change('curv-layer',String(layer));change('curv-family',family);healthy('curvature');healthy('spectrum');cases++;
}
for(const id of ['accuracy','loss','metric-chart','class-chart','curvature','spectrum','history-chart'])assert(d.getElementById(id).getAttribute('aria-label'));
assert(!d.querySelector('img:not([alt])'));assert(!d.querySelector('a[href="#"]'));
console.log(JSON.stringify({cases,scalar_metrics:w.telemetryUI.metricKeys.length,history_points:healthy('history-chart'),status:'passed'}));dom.window.close();
