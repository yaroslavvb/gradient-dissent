'use strict';
const D=window.TELEMETRY_DATA;
const $=id=>document.getElementById(id);
const names={plain:'Plain MLP',residual:'Residual',sd_constant:'Constant stochastic depth',sd_annealed:'Decreasing stochastic depth',residual_unit_dropout:'Residual + unit dropout',unit_dropout:'Optimized unit dropout'};
const colors={plain:'#72818a',residual:'#1d647f',sd_constant:'#c36b13',sd_annealed:'#864d9b',residual_unit_dropout:'#18845c',unit_dropout:'#1d647f'};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const fmt=x=>!finite(x)?'—':Math.abs(x)>=1e5||Math.abs(x)>0&&Math.abs(x)<.001?x.toExponential(2):Number(x.toPrecision(5)).toString();
const selected=()=>D.runs.filter(r=>r.runner===$('cohort').value&&($('seed').value==='all'||r.seed===Number($('seed').value)));
const tx=axis=>({epoch:'Epoch',training_seconds:'Synchronized training seconds',run_elapsed_seconds:'Run elapsed seconds'}[axis]);
const trace=(r,points)=>({name:(names[r.recipe]||r.recipe)+' · '+r.seed,color:colors[r.recipe]||'#4b7780',seed:r.seed,points});
let downloadRows=[];
function chart(id,series,{xlabel='Epoch',ylabel='',log=false,range=null,target=null}={}){
 const svg=$(id),vb=svg.viewBox?.baseVal,width=Number(svg.getAttribute('viewBox').split(' ')[2]),height=Number(svg.getAttribute('viewBox').split(' ')[3]);
 const pad={l:76,r:20,t:24,b:55},w=width-pad.l-pad.r,h=height-pad.t-pad.b;
 const good=series.flatMap(s=>s.points).filter(p=>finite(p.x)&&finite(p.y)&&(!log||p.y>0));
 if(!good.length){svg.innerHTML='<text x="76" y="70" class="empty">No measurements for this selection.</text>';return;}
 const transform=y=>log?Math.log10(y):y;
 let xmin=Math.min(...good.map(p=>p.x)),xmax=Math.max(...good.map(p=>p.x));if(xmin===xmax)xmax=xmin+1;
 let ymin=range?range[0]:Math.min(...good.map(p=>transform(p.y))),ymax=range?range[1]:Math.max(...good.map(p=>transform(p.y)));
 if(ymin===ymax){const delta=Math.max(1,Math.abs(ymin)*.1);ymin-=delta;ymax+=delta;}else if(!range){let d=(ymax-ymin)*.07;ymin-=d;ymax+=d;}
 const X=x=>pad.l+(x-xmin)/(xmax-xmin)*w,Y=y=>pad.t+(ymax-transform(y))/(ymax-ymin)*h;
 let html=`<defs><clipPath id="clip-${id}"><rect x="${pad.l}" y="${pad.t}" width="${w}" height="${h}"/></clipPath></defs>`;
 for(let i=0;i<=4;i++){let v=ymin+(ymax-ymin)*i/4,y=pad.t+h-i*h/4;html+=`<path class="grid" d="M${pad.l},${y}H${width-pad.r}"/><text x="${pad.l-11}" y="${y+5}" text-anchor="end">${esc(fmt(log?10**v:v))}</text>`;}
 for(let i=0;i<=5;i++){let v=xmin+(xmax-xmin)*i/5;html+=`<text x="${X(v)}" y="${height-28}" text-anchor="middle">${esc(fmt(v))}</text>`;}
 html+=`<text x="${pad.l}" y="16">${esc(ylabel.length>(width<800?58:105)?ylabel.slice(0,width<800?55:102)+'…':ylabel)}${log?' · log scale':''}</text><text x="${pad.l+w/2}" y="${height-5}" text-anchor="middle">${esc(xlabel)}</text>`;
 html+=`<g clip-path="url(#clip-${id})">`;
 if(target!==null&&target>=ymin&&target<=ymax)html+=`<path d="M${pad.l},${Y(target)}H${width-pad.r}" stroke="#526570" stroke-dasharray="3 5"/><text x="${width-pad.r-4}" y="${Y(target)-6}" text-anchor="end">${target}% target</text>`;
 for(const s of series){let d='',open=false;const dash={101:'',102:'8 4',103:'2 4'}[s.seed]||'';
 for(const p of s.points){if(!finite(p.x)||!finite(p.y)||(log&&p.y<=0)){open=false;continue;}d+=(open?'L':'M')+X(p.x).toFixed(2)+','+Y(p.y).toFixed(2);open=true;}
 html+=`<path class="trace" d="${d}" stroke="${s.color}" stroke-dasharray="${dash}"/>`;
 for(const p of s.points){if(!finite(p.x)||!finite(p.y)||(log&&p.y<=0))continue;html+=`<circle class="point" cx="${X(p.x)}" cy="${Y(p.y)}" r="2.2" fill="${s.color}" tabindex="0"><title>${esc(s.name)}; ${esc(xlabel)} ${fmt(p.x)}; ${esc(ylabel)} ${fmt(p.y)}${p.epoch!==undefined?'; epoch '+p.epoch:''}</title></circle>`;}}
 svg.innerHTML=html+'</g>';
}
function curve(r,split,metric){
 const rows=r.telemetry_rows||[],map=new Map();
 for(const t of rows){const key=split+'/'+(metric==='accuracy'?'accuracy_pct':'loss'),value=t.metrics?.[key];if(finite(value))map.set(t.epoch,{epoch:t.epoch,x:t[$('axis').value],y:value,source:'telemetry'});}
 // Primary split histories retain the original exact evaluator and cadence.
 const key=split==='validation'?'validation':split==='test'?'test':null;
 if(key)for(const h of r.history||[]){if(!h[key])continue;const t=rows.find(t=>t.epoch===h.epoch);let x=$('axis').value==='epoch'?h.epoch:$('axis').value==='training_seconds'?h.training_seconds:t?.run_elapsed_seconds;if(finite(x))map.set(h.epoch,{epoch:h.epoch,x,y:h[key][metric]*(metric==='accuracy'?100:1),source:'original evaluator'});}
 return [...map.values()].sort((a,b)=>a.epoch-b.epoch);
}
function renderCurves(){
 let baseline=$('cohort').value==='baseline';if(baseline&&$('split').value==='validation')$('split').value='test';
 $('split').querySelector('[value=validation]').disabled=baseline;
 let runs=selected(),split=$('split').value;
 let acc=runs.map(r=>trace(r,curve(r,split,'accuracy'))),loss=runs.map(r=>trace(r,curve(r,split,'loss')));
 chart('accuracy',acc,{xlabel:tx($('axis').value),ylabel:'Accuracy (%)',range:$('zoom').checked?[95,100]:[0,100],target:baseline&&split==='test'?98.63:null});
 chart('loss',loss,{xlabel:tx($('axis').value),ylabel:'Cross-entropy (nats)',log:$('loss-log').checked});
 $('legend').innerHTML=[...new Set(runs.map(r=>r.recipe))].map(r=>`<span><i style="background:${colors[r]}"></i>${esc(names[r])}</span>`).join('')+'<span class="muted">101 solid · 102 dashed · 103 dotted</span>';
 $('curve-note').textContent=baseline?'Baseline: 60,000 training examples, BF16 graph training, batch 256. Official-test evaluation every epoch; stop on the first ≤137 test errors. Test-based recipe selection and stopping make this a time-to-target replay.':'Controlled comparison: fixed 50,000/10,000 training/validation split, FP32/TF32 graph training, batch 64, 100 epochs. Validation loss selects the checkpoint. Test curves are repeated diagnostics; no recipe was changed from those outcomes.';
 downloadRows=[];for(let i=0;i<runs.length;i++)for(const p of acc[i].points){const l=loss[i].points.find(q=>q.epoch===p.epoch);downloadRows.push({run:runs[i].id,split,axis:$('axis').value,epoch:p.epoch,x:p.x,accuracy_pct:p.y,loss:l?.y??'',source:p.source});}
 $('curve-table').innerHTML='<thead><tr><th>Run</th><th>Epoch</th><th>'+esc(tx($('axis').value))+'</th><th>Accuracy %</th><th>CE</th></tr></thead><tbody>'+downloadRows.map(r=>`<tr><td>${esc(r.run)}</td><td>${r.epoch}</td><td>${fmt(r.x)}</td><td>${fmt(r.accuracy_pct)}</td><td>${fmt(r.loss)}</td></tr>`).join('')+'</tbody>';
 renderMetric();renderClass();
}
let metricKeys=[...new Set(D.runs.flatMap(r=>(r.telemetry_rows||[]).flatMap(t=>Object.keys(t.metrics))))].sort();
function fillMetrics(){const old=$('metric').value,query=$('metric-search').value.toLowerCase(),keys=metricKeys.filter(k=>k.toLowerCase().includes(query));$('metric').innerHTML=keys.map(k=>`<option value="${esc(k)}">${esc(k)}</option>`).join('');const preferred='layer-1/probe/weight_grad_diversity';if(keys.includes(old))$('metric').value=old;else if(keys.includes(preferred))$('metric').value=preferred;renderMetric();}
function renderMetric(){const key=$('metric').value,series=selected().map(r=>trace(r,(r.telemetry_rows||[]).map(t=>({x:t[$('axis').value],y:t.metrics[key],epoch:t.epoch}))));chart('metric-chart',series,{xlabel:tx($('axis').value),ylabel:key,log:$('metric-log').checked});const layer=key.match(/^layer-(\d)/)?.[1]||'1',normkey=`layer-${layer}/probe/grad_l2`;$('norm-heading').textContent=`Layer ${layer} · mean gradient magnitude`;chart('gradient-norm',selected().map(r=>trace(r,(r.telemetry_rows||[]).map(t=>({x:t[$('axis').value],y:t.metrics[normkey],epoch:t.epoch})))),{xlabel:tx($('axis').value),ylabel:'Mean probe weight-gradient L2 norm',log:true});$('metric-note').textContent=key?`${key}. ${metricKeys.length} recorded scalar keys available. Missing values break the line; zero or negative values are omitted on a log axis. Probe gradient moments use ordinary FP32 autograd; the later curvature pass uses stabilized float64 derivatives.`:'No metric matches this search.';}
function renderClass(){let split=$('class-split').value,d=$('digit').value;chart('class-chart',selected().map(r=>trace(r,(r.telemetry_rows||[]).filter(t=>finite(t.metrics[`${split}/class-${d}/accuracy`])).map(t=>({x:t[$('axis').value],y:100*t.metrics[`${split}/class-${d}/accuracy`],epoch:t.epoch})))),{xlabel:tx($('axis').value),ylabel:`Digit ${d} · ${split} accuracy (%)`,range:[80,100]});}
function flat(obj,p='',out={}){if(obj&&typeof obj==='object'&&!Array.isArray(obj))for(const [k,v]of Object.entries(obj)){let key=p?p+'/'+k:k;if(finite(v))out[key]=v;else if(v&&typeof v==='object'&&!Array.isArray(v))flat(v,key,out);}return out;}
function curvRuns(){return (D.curvature||[]).map(c=>c.result||c);}
function curvRows(c){return c.snapshots||c.checkpoints||[];}
function curvResult(row){return row.curvature||row.diagnostics||row.result||row;}
function curvObject(t,layer,family){const q=curvResult(t);return family==='output_mismatch'?q.output_mismatch:family==='activation_factor'?q.layers?.[layer]?.activation_factor:family==='gradient_moments'?q.layers?.[layer]?.gradient_moments:q.layers?.[layer]?.families?.[family];}
function fillCurv(){const previous=$('curv-metric').value,layer=+$('curv-layer').value,family=$('curv-family').value,keys=[...new Set(curvRuns().flatMap(c=>curvRows(c).flatMap(t=>Object.keys(flat(curvObject(t,layer,family)||{})))))].sort();$('curv-metric').innerHTML=keys.map(k=>`<option>${esc(k)}</option>`).join('');if(keys.includes(previous))$('curv-metric').value=previous;else{const preferred=keys.find(k=>k==='kfac/trace')||keys.find(k=>/trace/.test(k));if(preferred)$('curv-metric').value=preferred;}renderCurv();}
function renderCurv(){const layer=+$('curv-layer').value,family=$('curv-family').value,key=$('curv-metric').value;const series=curvRuns().map(c=>{const recipe=c.recipe||c.spec?.recipe||c.source_spec?.recipe||c.source?.recipe;return trace({recipe,seed:101},curvRows(c).map(t=>({x:t.epoch,epoch:t.epoch,y:flat(curvObject(t,layer,family)||{})[key]})));});chart('curvature',series,{xlabel:'Epoch · offline snapshots',ylabel:key||'Curvature',log:$('curv-log').checked});$('curv-note').textContent=`Layer ${layer}; ${family}; ${key||'no completed measurements'}. Exact output directions on N=128; KFAC factorization remains approximate. Seed 101 only; no multi-seed uncertainty estimate.`;renderSpectrum();}
function renderSpectrum(){const layer=+$('curv-layer').value,family=$('curv-family').value,epoch=+$('spectrum-epoch').value,factor=$('spectrum-factor').value;const series=curvRuns().map(c=>{const row=curvRows(c).find(r=>r.epoch===epoch),q=row?curvResult(row):{},l=q.layers?.[layer],values=factor==='activation'?l?.activation_factor?.eigenvalues:l?.families?.[family]?.backprop_factor?.eigenvalues;return trace({recipe:c.recipe||c.source_spec?.recipe||c.spec?.recipe,seed:101},(values||[]).map((y,i)=>({x:i+1,y,epoch})));});chart('spectrum',series,{xlabel:'Eigenvalue rank · descending',ylabel:factor==='activation'?'Activation-factor eigenvalue':'Backprop-factor eigenvalue',log:true});}
function tables(){const q=D.qualification;let pairs=q.pairs||[];$('qualification-note').innerHTML='<p><span class="pill">Weights unchanged</span> The adopted cadence passed all three same-A100 on/off checks: final parameters and original scientific histories are bitwise identical. The first, denser telemetry attempt is retained in the raw data; its cold controlled run exceeded the 10% target.</p>';
 $('overhead-table').innerHTML='<caption>Adopted telemetry: every 10 epochs, 128-example probe. No sparse snapshot writes in qualification.</caption><thead><tr><th>Pair</th><th>Telemetry on: training</th><th>Telemetry off: training</th><th>Diagnostics</th><th>Diagnostics / training</th><th>Final weights</th></tr></thead><tbody>'+pairs.map(p=>`<tr><td>${esc(p.name||p.pair)}</td><td>${fmt(p.on_training_seconds)} s</td><td>${fmt(p.off_training_seconds)} s</td><td>${fmt(p.telemetry_seconds)} s</td><td>${fmt(100*p.telemetry_over_training_fraction)}%</td><td>${p.parameters_bitwise_equal?'Identical':'See audit'}</td></tr>`).join('')+'</tbody>';
 let runs=D.runs.filter(r=>r.runner==='baseline');$('baseline-table').innerHTML='<caption>First observed ≥98.63% official-test accuracy in the instrumented optimized reruns</caption><thead><tr><th>Seed</th><th>Epoch</th><th>Accuracy</th><th>Training</th><th>Run elapsed at target</th><th>Direct telemetry</th></tr></thead><tbody>'+runs.map(r=>{const t=r.threshold||r.endpoints?.threshold||{};return `<tr><td>${r.seed}</td><td>${t.epoch??'—'}</td><td>${fmt(t.test?.accuracy*100)}%</td><td>${fmt(t.training_seconds)} s</td><td>${fmt(t.run_wall_seconds_with_telemetry)} s</td><td>${fmt(r.telemetry?.seconds)} s</td></tr>`;}).join('')+'</tbody>';
 $('baseline-note').textContent='Cold baseline telemetry totals 0.80–1.12 seconds (14.1–18.5% of the short training loop), exceeding the 10% target in these final runs. Epoch-zero setup accounts for most of it; later diagnostics cost 0.19–0.26 seconds (3.3–4.3%). These are measured costs, not a claim that every cold run passes the qualification overhead target.';
 const hist=D.historical.rows||[];chart('history-chart',[{name:'Historical ts4k9n55 official test',color:'#697a84',seed:101,points:hist.map(r=>({x:r.wandb_runtime_seconds??r._runtime,y:r.metrics?.['test/accuracy_pct']??r.val_accuracy,epoch:r.epoch}))}],{xlabel:'Historical W&B run elapsed seconds',ylabel:'Official-test accuracy (%)',range:[90,100],target:98.63});
}
$('download-curves').addEventListener('click',()=>{const keys=['run','split','axis','epoch','x','accuracy_pct','loss','source'],csv=[keys.join(','),...downloadRows.map(r=>keys.map(k=>'"'+String(r[k]).replace(/"/g,'""')+'"').join(','))].join('\n'),u=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=u;a.download='ciresan-learning-curves.csv';a.click();URL.revokeObjectURL(u);});
for(const id of ['cohort','axis','seed','split','zoom','loss-log'])$(id).addEventListener('change',renderCurves);
$('metric-search').addEventListener('input',fillMetrics);for(const id of ['metric','metric-log'])$(id).addEventListener('change',renderMetric);for(const id of ['digit','class-split'])$(id).addEventListener('change',renderClass);for(const id of ['curv-layer','curv-family'])$(id).addEventListener('change',fillCurv);for(const id of ['curv-metric','curv-log'])$(id).addEventListener('change',renderCurv);
for(const id of ['spectrum-epoch','spectrum-factor'])$(id).addEventListener('change',renderSpectrum);
fillMetrics();renderCurves();fillCurv();tables();
window.telemetryUI={chart,curve,selected,renderCurves,renderMetric,renderClass,fillCurv,metricKeys};
