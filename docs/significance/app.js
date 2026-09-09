'use strict';
const D=window.SIGNIFICANCE_DATA;
const $=id=>document.getElementById(id);
const recipes=['dense','constant_ild','decreasing_ild'];
const names=['Dense training','Constant ILD','Decreasing ILD'];
const colors=['#546478','#216aab','#bb5425'];
const curve=(t,r,m)=>D.curves.find(x=>x.task===t&&x.recipe===r&&x.mask_name===m);
function schedule(){const t=+$('progress').value/100;$('progress-value').textContent=Math.round(t*100)+'%';$('blocks').innerHTML=Array.from({length:12},(_,i)=>`<div class="block" style="background:rgba(8,127,120,${1-.8*i/11*(1-t)})"><span>${i+1}</span></div>`).join('');$('schedule-text').textContent=`Expected active blocks: ${(12*(1-.4*(1-t))).toFixed(1)} of 12`}
function render(){
 const task=$('family').value,metric=$('metric').value,mask=$('mask').value,acc=metric==='accuracy',factor=acc?100:1;
 const rows=recipes.map((r,i)=>({name:names[i],color:colors[i],full:curve(task,r,'full')[metric].mean*factor,pruned:curve(task,r,mask)[metric].mean*factor}));
 const vals=rows.flatMap(r=>[r.full,r.pruned]);let lo=Math.floor(Math.min(...vals)*.9*10)/10,hi=Math.max(...vals)*1.05;if(acc){lo=0;hi=Math.min(100,Math.ceil(hi/10)*10)}
 const y=v=>260-(v-lo)/(hi-lo)*205;let svg=`<title>${acc?'Accuracy (higher is better)':'Cross-entropy (lower is better)'}</title>`;
 for(let k=0;k<5;k++){let v=lo+(hi-lo)*k/4;svg+=`<line x1="110" x2="725" y1="${y(v)}" y2="${y(v)}" stroke="#e2e8e5"/><text x="98" y="${y(v)+4}" text-anchor="end" fill="#52646b" font-size="12">${v.toFixed(acc?0:2)}${acc?'%':''}</text>`}
 svg+='<text x="255" y="305" text-anchor="middle" fill="#172a34" font-size="15">Full model</text><text x="595" y="305" text-anchor="middle" fill="#172a34" font-size="15">Pruned model</text>';
 rows.forEach((r,i)=>{const x0=240+i*15,x1=580+i*15;svg+=`<line x1="${x0}" y1="${y(r.full)}" x2="${x1}" y2="${y(r.pruned)}" stroke="${r.color}" stroke-width="3" opacity=".8"/><circle cx="${x0}" cy="${y(r.full)}" r="6" fill="${r.color}"/><circle cx="${x1}" cy="${y(r.pruned)}" r="6" fill="${r.color}"/>`});$('quality-chart').innerHTML=svg;
 const f=v=>v.toFixed(acc?2:3)+(acc?'%':'');$('score-table').innerHTML=rows.map(r=>`<tr><th>${r.name}</th><td>${f(r.full)}</td><td>${f(r.pruned)}</td><td>${(r.pruned-r.full)>=0?'+':''}${(r.pruned-r.full).toFixed(3)}${acc?' pp':' nats'}</td></tr>`).join('');
 const fam=D.families.find(f=>f.task===task),kept=fam.mask_panel[mask];$('chart-context').textContent=`${fam.label}. Keeping ${kept.length} of ${fam.prunable_count} residual blocks (zero-based indices: ${kept.join(', ')}). ${task.startsWith('convnext')?'Stem and stage transitions always remain.':''}`;
 let text='';if(mask==='delete_first_only')text='The first block was never dropped in ILD training. This out-of-support deletion shows why resilience to one mask family does not mean resilience to arbitrary missing layers.';
 else if(task.startsWith('vit')&&mask==='primary_two_thirds')text=acc?'At the primary exit, both ILD recipes have higher mean accuracy than dense training. Accuracy is a secondary outcome; the relative-CE primary endpoint gives a different ranking.':'Dense improves more relative to its own full model, but constant ILD still has lower mean pruned CE. The starting point changes the relative ranking. Pruned-CE paired intervals include zero.';
 else if(task.startsWith('vit'))text='The same number of retained blocks, at different original positions: these fixed intermediate-deletion masks strongly reduce dense accuracy. Both ILD recipes also improve relative CE change here (secondary comparisons).';
 else if(task.startsWith('gpt'))text='ILD preserves substantially better pruned predictors here, while its intact GPT is worse at this short training budget. Robust subnetworks and best full-model quality are separate objectives.';
 else text='ILD preserves much stronger pruned image predictors for these supported masks. At the primary mask, the relative-CE effects are large but their three-seed intervals include zero.';
 $('insight').textContent=text;
}
$('effect-table').innerHTML=['gpt_wikitext103','convnext_cifar100','vit_cifar100'].map(t=>`<tr><th>${D.families.find(f=>f.task===t).label.split(' / ')[0]}</th>${recipes.slice(1).map(r=>{const v=D.primary_comparisons.find(c=>c.task===t&&c.recipe===r);return `<td>${v.mean.toFixed(3)} [${v.ci95_low.toFixed(3)}, ${v.ci95_high.toFixed(3)}]</td>`}).join('')}</tr>`).join('');
['family','metric','mask'].forEach(id=>$(id).addEventListener('change',render));$('progress').addEventListener('input',schedule);schedule();render();
