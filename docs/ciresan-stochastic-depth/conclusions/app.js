'use strict';
const D=window.CIRESAN_CONCLUSIONS,$=id=>document.getElementById(id);
const recipes=['residual','sd_constant','sd_annealed'],colors={residual:'#1d647f',sd_constant:'#b56319',sd_annealed:'#854d9b'};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=3)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'—';
function estimate(m,scale=100){return {mean:m.mean*scale,low:m.ci95_low*scale,high:m.ci95_high*scale};}
function interval(v){return `${fmt(v.mean)} [${fmt(v.low)}, ${fmt(v.high)}]`;}
function plot(id,rows,{min,max,label,first,second,zero=false}){
 const left=225,right=950,top=62,step=82,bottom=290,X=x=>left+(x-min)/(max-min)*(right-left);let html='';
 for(let k=0;k<=5;k++){const v=min+(max-min)*k/5;html+=`<path class="grid" d="M${X(v)},28V${bottom}"/><text x="${X(v)}" y="318" text-anchor="middle">${fmt(v,1)}</text>`;}
 if(zero)html+=`<path d="M${X(0)},24V${bottom}" stroke="#617780" stroke-dasharray="4 4"/><text x="${X(0)+7}" y="18">No digit-1 advantage</text>`;
 rows.forEach((r,i)=>{let y=top+i*step;html+=`<text x="0" y="${y+5}" fill="${colors[r.recipe]}">${esc(D.names[r.recipe])}</text>`;for(let j=0;j<2;j++){const v=r[j===0?'a':'b'],yy=y+(j===0?-12:15),color=j===0?'#294653':colors[r.recipe];
 html+=`<path class="ci" stroke="${color}" d="M${X(v.low)},${yy}H${X(v.high)}M${X(v.low)},${yy-5}V${yy+5}M${X(v.high)},${yy-5}V${yy+5}"/>`;
 const tip=`${D.names[r.recipe]}, ${j===0?first:second}: ${interval(v)}`;
 html+=j===0?`<circle class="point" cx="${X(v.mean)}" cy="${yy}" r="5" fill="${color}" tabindex="0"><title>${esc(tip)}</title></circle>`:`<path class="point" d="M${X(v.mean)},${yy-7}l7,7 -7,7 -7,-7Z" fill="${color}" tabindex="0"><title>${esc(tip)}</title></path>`;
 }});html+=`<text x="${(left+right)/2}" y="346" text-anchor="middle">${esc(label)}</text>`;$(id).innerHTML=html;
}
function render(){const endpoint=$('endpoint').value,groups=recipes.map(recipe=>D.aggregates.find(a=>a.recipe===recipe&&a.endpoint===endpoint));
 const quality=groups.map(a=>({recipe:a.recipe,a:estimate(a.metrics.dense_accuracy),b:estimate(a.metrics.primary_mask_accuracy)}));
 plot('quality-chart',quality,{min:95,max:100,label:'Official-test accuracy (%) · three seeds',first:'Full network',second:'Two branches kept'});
 $('quality-table').innerHTML='<thead><tr><th>Method</th><th>Full accuracy %</th><th>Two-branch accuracy %</th><th>Deletion CE increase</th></tr></thead><tbody>'+groups.map((a,i)=>`<tr><td>${esc(D.names[a.recipe])}</td><td>${fmt(quality[i].a.mean)}</td><td>${fmt(quality[i].b.mean)}</td><td>${fmt(a.metrics.primary_excess_ce.mean,4)} nats</td></tr>`).join('')+'</tbody>';
 $('quality-note').textContent=endpoint==='selected'?'Primary endpoint: decreasing SD has the smallest mean deletion CE increase, but its full-model accuracy starts lower. The paired deletion-damage improvement is stronger evidence than comparing these absolute accuracies alone.':'Secondary endpoint: the SD models keep roughly 98.34–98.35% accuracy with two branches. Their full-model CE is worse than residual dense, so the outcome depends on which quality metric matters.';
 const digits=groups.map(a=>({recipe:a.recipe,a:estimate(a.metrics.raw_digit1_minus_mean4_9_resilience),b:estimate(a.metrics.adjusted_digit1_minus_mean4_9_resilience)}));
 plot('digit-chart',digits,{min:-1.5,max:3,label:'Digit 1 minus mean(4,9) resilience · percentage points',first:'Raw contrast',second:'Margin standardized',zero:true});
 $('digit-table').innerHTML='<thead><tr><th>Method</th><th>Raw contrast [95% CI]</th><th>Margin-standardized [95% CI]</th></tr></thead><tbody>'+digits.map(r=>`<tr><td>${esc(D.names[r.recipe])}</td><td>${interval(r.a)}</td><td>${interval(r.b)}</td></tr>`).join('')+'</tbody>';
 $('digit-note').textContent=endpoint==='selected'?'All three primary adjusted intervals cross zero. This does not prove equal resilience; it does not establish a dependable digit-1 routing rule either.':'At epoch 100, decreasing SD reverses sign after adjustment: −0.119 pp [−0.185, −0.054]. Residual dense retains a positive adjusted association. A universal class ordering is not supported.';
}
function renderRuns(){const which=$('run-filter').value,rows=D.runs.filter(r=>which==='all'||which==='sd'&&r.runner==='controlled'&&r.recipe.startsWith('sd_')||which==='controls'&&r.runner==='controlled'&&!r.recipe.startsWith('sd_')||which==='baseline'&&r.runner==='baseline');
 $('wandb-status').innerHTML=D.wandb.complete?`<span class="state verified">${D.wandb.verified_run_count} / 18 W&amp;B runs uploaded and verified</span>`:`<span class="state pending">Training complete · ${D.wandb.verified_run_count} / 18 W&amp;B uploads verified</span> W&amp;B sign-in is still required for new run links. Every raw result is available below; no placeholder run URLs are presented.`;
 $('run-table').innerHTML='<thead><tr><th>Method</th><th>Seed</th><th>Selected / target epoch</th><th>Selected / target accuracy</th><th>Final accuracy</th><th>Training s</th><th>W&amp;B</th><th>Source</th></tr></thead><tbody>'+rows.map(r=>{const b=r.runner==='baseline',e=r.endpoints,t=e.threshold;const epoch=b?t?.epoch:e.best_epoch,selected=b?t?.test?.accuracy:e.selected_test?.accuracy;
 return `<tr><td>${esc(r.name)}</td><td>${r.seed}</td><td>${epoch??'—'}</td><td>${fmt(selected*100)}%</td><td>${fmt(e.final_test?.accuracy*100)}%</td><td>${fmt(r.training_seconds,2)}</td><td>${r.wandb_url?`<a href="${esc(r.wandb_url)}">Open run ↗</a>`:'Pending sign-in'}</td><td><a href="${esc(r.raw_url)}">JSON ↗</a></td></tr>`;}).join('')+'</tbody>';
}
$('endpoint').addEventListener('change',render);$('run-filter').addEventListener('change',renderRuns);render();renderRuns();
window.practicalReportUI={render,renderRuns};
