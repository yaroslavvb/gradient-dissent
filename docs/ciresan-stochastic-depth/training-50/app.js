'use strict';
(() => {
  const D = window.HALFDROP_DATA, $ = id => document.getElementById(id);
  if (!D || D.models.length !== 32) throw new Error('The complete half-drop experiment data is required.');
  const pct = x => (100*x).toFixed(2)+'%';
  const pp = x => (x>=0?'+':'')+x.toFixed(3)+' pp';
  const maskText = mask => Array.from({length:4},(_,i)=>mask&(1<<i)?i+1:null).filter(x=>x!==null).join(', ') || 'none';
  const size = mask => Array.from({length:4},(_,i)=>+(!!(mask&(1<<i)))).reduce((a,b)=>a+b,0);
  const endpoint = () => $('endpoint').value;
  const value = interval => interval.mean;
  const CI = (interval, unit='pp') => '['+[interval.ci95_low,interval.ci95_high].map(x=>unit==='pct'?pct(x):x.toFixed(3)).join(', ')+']'+(unit==='pp'?' pp':'');
  let chosenMask=0, timer=null, galleryData=null, galleryToken=0, focusIndex=null;
  const cache = new Map();
  const model = (mask=chosenMask) => D.models.find(m=>m.endpoint===endpoint()&&m.drop_mask===mask);
  const pathForOrder = order => D.paths.find(p=>p.order.join(',')===order.join(','));
  function chosenPath() {
    const selection=$('order').value;
    if(selection==='validation')return D.paths.find(p=>p.validation_selected);
    if(selection==='shallow')return pathForOrder([0,1,2,3]);
    if(selection==='deep')return pathForOrder([3,2,1,0]);
    return pathForOrder(selection.split(',').map(Number));
  }
  $('order').insertAdjacentHTML('beforeend', D.paths.map(p=>`<option value="${p.order.join(',')}">${p.order.map(i=>i+1).join(' → ')}</option>`).join(''));
  function stop() {if(timer)clearInterval(timer);timer=null;$('animate').textContent='Follow this order';$('animate').setAttribute('aria-pressed','false');}
  function setMask(mask,halt=true) {
    if(!Number.isInteger(mask)||mask<0||mask>15)throw new Error('Training eligibility mask must be 0–15.');
    if(halt)stop();chosenMask=mask;render();
  }
  function chart() {
    const models=D.models.filter(m=>m.endpoint===endpoint()),path=chosenPath(),selected=path.masks.map(model);
    const means=models.map(m=>m.metrics.accuracy.mean);
    let lo=0,hi=1;
    if($('scale').value==='zoom') {
      lo=Math.floor(1000*(Math.min(...means,...selected.map(m=>m.metrics.accuracy.ci95_low))-.001))/1000;
      hi=Math.ceil(1000*(Math.max(...means,...selected.map(m=>m.metrics.accuracy.ci95_high))+.001))/1000;
      if(hi-lo<.006){const mid=(hi+lo)/2;lo=mid-.003;hi=mid+.003;}
    }
    const x=k=>90+k*210,y=a=>305-(a-lo)/(hi-lo)*245,parts=[];
    for(let i=0;i<=5;i++){const a=lo+(hi-lo)*i/5;parts.push(`<path d="M90 ${y(a)} H930" stroke="#e0e8e4"/><text x="77" y="${y(a)+4}" text-anchor="end">${(100*a).toFixed($('scale').value==='zoom'?2:0)}%</text>`);}
    for(let k=0;k<=4;k++)parts.push(`<text x="${x(k)}" y="333" text-anchor="middle">${k}</text>`);
    parts.push('<text x="90" y="27">Full-network test accuracy · every layer active at testing</text><text x="500" y="364" text-anchor="middle">Branches eligible for 50% dropout during training</text>');
    for(const m of models)parts.push(`<circle cx="${x(m.k)}" cy="${y(value(m.metrics.accuracy))}" r="5" fill="#aec2bc" data-mask="${m.drop_mask}" tabindex="0" role="button" aria-label="Train dropout on branches ${maskText(m.drop_mask)}, accuracy ${pct(value(m.metrics.accuracy))}"><title>Branches ${maskText(m.drop_mask)}: ${pct(value(m.metrics.accuracy))}</title></circle>`);
    const byK=D.by_k.filter(m=>m.endpoint===endpoint()).sort((a,b)=>a.k-b.k);
    parts.push(`<path d="${byK.map((m,i)=>(i?'L':'M')+x(m.k)+' '+y(m.metrics.accuracy.mean)).join(' ')}" stroke="#08776f" stroke-width="3" fill="none"/>`);
    parts.push(`<path d="${selected.map((m,i)=>(i?'L':'M')+x(i)+' '+y(m.metrics.accuracy.mean)).join(' ')}" stroke="#c47b29" stroke-width="3" fill="none"/>`);
    selected.forEach((m,k)=>{
      const a=m.metrics.accuracy;
      // Full-scale view clips drawn intervals at the view boundary only;
      // complete, untruncated statistical intervals remain in the table.
      const lowY=y(Math.max(lo,a.ci95_low)),highY=y(Math.min(hi,a.ci95_high));
      parts.push(`<path d="M${x(k)} ${highY} V${lowY} M${x(k)-6} ${highY} H${x(k)+6} M${x(k)-6} ${lowY} H${x(k)+6}" stroke="#c47b29" stroke-width="1.5"/>`);
    });
    selected.forEach((m,k)=>parts.push(`<circle cx="${x(k)}" cy="${y(m.metrics.accuracy.mean)}" r="${m.drop_mask===chosenMask?9:6}" fill="#c47b29" stroke="white" stroke-width="2" data-mask="${m.drop_mask}" tabindex="0" role="button" aria-label="Select dropout on branches ${maskText(m.drop_mask)}"><title>${maskText(m.drop_mask)}: ${pct(m.metrics.accuracy.mean)}</title></circle>`));
    if(!path.masks.includes(chosenMask)){const m=model();parts.push(`<circle cx="${x(m.k)}" cy="${y(m.metrics.accuracy.mean)}" r="10" fill="none" stroke="#173942" stroke-width="3"/>`);}
    $('accuracy-chart').innerHTML=parts.join('');
    $('accuracy-chart').querySelectorAll('[data-mask]').forEach(point=>{point.onclick=()=>setMask(Number(point.dataset.mask));point.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();point.onclick();}};});
  }
  function subsetTable() {
    const links=D.wandb_runs||{};
    $('subset-table').innerHTML='<caption>Three-seed means. Intervals are unadjusted Student t intervals; all inference uses the full network.</caption><thead><tr><th>50% dropout branches</th><th>Test accuracy</th><th>Change vs none</th><th>Newly wrong</th><th>Newly correct</th><th>W&amp;B runs</th></tr></thead><tbody>'+D.models.filter(m=>m.endpoint===endpoint()).sort((a,b)=>a.k-b.k||a.drop_mask-b.drop_mask).map(m=>`<tr class="${m.drop_mask===chosenMask?'active':''}"><td><button type="button" data-subset="${m.drop_mask}">${maskText(m.drop_mask)}</button></td><td>${pct(m.metrics.accuracy.mean)}<br><small>${CI(m.metrics.accuracy,'pct')}</small></td><td>${pp(m.paired_vs_none.accuracy_pp.mean)}<br><small>${CI(m.paired_vs_none.accuracy_pp)}</small></td><td>${pct(m.paired_vs_none.harm_rate.mean)}</td><td>${pct(m.paired_vs_none.repair_rate.mean)}</td><td>${m.seeds.map(r=>links[r.run_id]?`<a href="${links[r.run_id]}">${r.seed}</a>`:String(r.seed)).join(' · ')}</td></tr>`).join('')+'</tbody>';
    $('subset-table').querySelectorAll('button').forEach(b=>b.onclick=()=>setMask(Number(b.dataset.subset)));
  }
  function classTable() {
    $('class-table').innerHTML='<caption>Means across three seed-matched comparisons, using all test examples of each true digit.</caption><thead><tr><th>Digit</th><th>Accuracy</th><th>Change vs none</th><th>Newly wrong / class</th><th>Newly correct / class</th></tr></thead><tbody>'+model().per_class.map(r=>`<tr><th>${r.digit}</th><td>${pct(r.accuracy.mean)}</td><td>${pp(r.accuracy_pp.mean)}</td><td>${pct(r.harm_rate.mean)}</td><td>${pct(r.repair_rate.mean)}</td></tr>`).join('')+'</tbody>';
  }
  function marginalTable() {
    $('marginal-table').innerHTML='<caption>Effect of adding one more droppable branch. Context range describes eight distinct choices of the other branches.</caption><thead><tr><th>Branch</th><th>Mean accuracy change</th><th>Three-seed 95% interval</th><th>Range across contexts</th></tr></thead><tbody>'+D.marginals.filter(r=>r.endpoint===endpoint()&&r.context_size==='all').map(r=>{
      const effects=D.edges.filter(e=>e.endpoint===endpoint()&&e.branch===r.layer).map(e=>e.metrics.accuracy_pp.mean);
      return `<tr><th>${r.layer+1}</th><td>${pp(r.accuracy_pp.mean)}</td><td>${CI(r.accuracy_pp)}</td><td>${pp(Math.min(...effects))} to ${pp(Math.max(...effects))}</td></tr>`;
    }).join('')+'</tbody>';
  }
  function drawDigit(canvas,index) {
    const encoded=galleryData&&galleryData.images[String(index)],ctx=canvas.getContext('2d');if(!encoded||!ctx)return;
    const bytes=atob(encoded),tiny=document.createElement('canvas');tiny.width=tiny.height=28;
    const c=tiny.getContext('2d'),pixels=c.createImageData(28,28);
    for(let i=0;i<784;i++){const v=bytes.charCodeAt(i);pixels.data.set([Math.round(20+v*.92),Math.round(44+v*.83),Math.round(52+v*.79),255],i*4);}
    c.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(tiny,0,0,canvas.width,canvas.height);
  }
  function filteredExamples() {
    const g=galleryData&&galleryData.models.find(m=>m.drop_mask===chosenMask);if(!g)return[];
    let examples=g.examples;
    if($('comparison').value==='previous') {
      const mask=referenceMask();if(mask===null)return[];
      const reference=galleryData.models.find(m=>m.drop_mask===mask);
      const byIndex=new Map(reference.examples.map(e=>[e.index,e]));
      examples=examples.filter(e=>e.selection.includes('fixed')).map(e=>{
        const r=byIndex.get(e.index);
        if(!r)throw new Error('Missing matched fixed-panel prediction.');
        const category=r.pred===r.label&&e.pred!==e.label?'harmed':r.pred!==r.label&&e.pred===e.label?'repaired':'unchanged';
        return {...e,baseline_pred:r.pred,baseline_probs:r.probs,baseline_ce:r.ce,category,selection:['fixed']};
      });
    }
    return examples.filter(e=>($('category').value==='all'||e.category===$('category').value)&&($('digit-class').value==='all'||e.label===Number($('digit-class').value)));
  }
  function referenceMask() {
    if($('comparison').value!=='previous')return 0;
    const k=chosenPath().masks.indexOf(chosenMask);
    return k<0?null:chosenPath().masks[Math.max(0,k-1)];
  }
  function focus() {
    const examples=filteredExamples(),e=examples.find(e=>e.index===focusIndex)||examples[0];
    if(!e){$('digit-image').hidden=true;$('image-id').textContent='No example in this selection';$('baseline-pred').textContent=$('treatment-pred').textContent='—';$('example-note').textContent='Try another digit or outcome filter. Empty categories are not filled with substitute examples.';$('probability-chart').innerHTML='';$('run-links').textContent='';return;}
    focusIndex=e.index;$('digit-image').hidden=false;drawDigit($('digit-image'),e.index);
    $('image-id').textContent=`TRUE DIGIT ${e.label} · TEST #${e.index}`;
    $('baseline-pred').textContent=`${e.baseline_pred} · ${e.baseline_pred===e.label?'correct':'wrong'}`;
    $('treatment-pred').textContent=`${e.pred} · ${e.pred===e.label?'correct':'wrong'}`;
    $('treatment-pred').style.color=e.pred===e.label?'#08776f':'#b24f35';
    const selection=e.selection.includes('fixed')?'Fixed reference-panel example.':'Outcome-selected illustration.';
    $('example-note').textContent=`${selection} Cross-entropy ${e.baseline_ce.toFixed(3)} → ${e.ce.toFixed(3)} nats. Every layer is active in both predictions.`;
    const links=D.wandb_runs||{},seed=Number($('seed').value),base=model(referenceMask()).seeds.find(r=>r.seed===seed),current=model().seeds.find(r=>r.seed===seed);
    $('run-links').innerHTML=[['Reference run',base],['Chosen training run',current]].map(([label,r])=>links[r.run_id]?`<a href="${links[r.run_id]}">${label}</a>`:'').filter(Boolean).join(' · ');
    const x=d=>83+d*94,y=p=>195-p*155,parts=[];
    for(const v of [0,.5,1])parts.push(`<path d="M55 ${y(v)} H980" stroke="#e3eae7"/><text x="45" y="${y(v)+4}" text-anchor="end">${100*v}%</text>`);
    for(let d=0;d<10;d++){parts.push(`<rect x="${x(d)-24}" y="${y(e.baseline_probs[d])}" width="21" height="${155*e.baseline_probs[d]}" fill="#778e9a"><title>Reference model: digit ${d}, ${pct(e.baseline_probs[d])}</title></rect><rect x="${x(d)+3}" y="${y(e.probs[d])}" width="21" height="${155*e.probs[d]}" fill="#08776f"><title>Chosen training subset: digit ${d}, ${pct(e.probs[d])}</title></rect><text x="${x(d)}" y="218" text-anchor="middle" style="font-weight:${d===e.label?750:400}">${d}${d===e.label?' ✓':''}</text>`);}
    parts.push('<text x="55" y="23">Reported probability for each digit</text>');$('probability-chart').innerHTML=parts.join('');
    $('gallery').querySelectorAll('button').forEach(b=>b.classList.toggle('selected',Number(b.dataset.index)===focusIndex));
  }
  function gallery() {
    const previous=$('comparison').value==='previous',reference=referenceMask();
    const label=previous?(reference===null?'Previous step unavailable':`Previous step · dropout on ${maskText(reference)}`):'No training dropout';
    $('reference-label').textContent=label;$('reference-legend').textContent=label;
    $('comparison-note').textContent=previous?(reference===null?'This custom subset is outside the chosen order. Select a point on its orange path or compare with the no-dropout baseline.':`Only the fixed 100-image reference panel is shown in this mode. Outcomes compare branches ${maskText(reference)} → ${maskText(chosenMask)}; the summary rates and digit table still compare with no dropout using all 10,000 images.`):'This viewer compares with no training dropout. It includes the fixed reference panel and labeled outcome-selected illustrations.';
    const edge=previous&&reference!==null&&reference!==chosenMask?D.edges.find(e=>e.endpoint===endpoint()&&e.from_mask===reference&&e.to_mask===chosenMask):null;
    $('edge-effect').hidden=!edge;
    $('edge-effect').textContent=edge?`Making branch ${edge.branch+1} additionally droppable: ${pp(edge.metrics.accuracy_pp.mean)} accuracy change (95% interval ${CI(edge.metrics.accuracy_pp)}); ${pct(edge.metrics.harm_rate.mean)} newly wrong and ${pct(edge.metrics.repair_rate.mean)} newly correct. These rates use all 10,000 test images and average the three matched seeds.`:'';
    const examples=filteredExamples();
    $('gallery').innerHTML=examples.map(e=>`<button type="button" data-index="${e.index}" aria-label="Digit ${e.label}, test image ${e.index}, ${e.category}"><canvas width="48" height="48" aria-hidden="true"></canvas><span>${e.baseline_pred} → ${e.pred}</span><small>#${e.index} · ${e.category}</small><small>${e.selection.includes('fixed')?'reference':'outcome-selected'}</small></button>`).join('');
    $('gallery').querySelectorAll('button').forEach(b=>{drawDigit(b.querySelector('canvas'),Number(b.dataset.index));b.onclick=()=>{focusIndex=Number(b.dataset.index);focus();};});focus();
  }
  async function loadGallery() {
    const key=$('seed').value+'-'+endpoint(),file=D.gallery_files[key],token=++galleryToken;
    galleryData=null;$('gallery-status').textContent='Loading the measured example predictions…';$('gallery').innerHTML='';focus();
    try {
      if(!cache.has(key))cache.set(key,fetch(file).then(r=>{if(!r.ok)throw new Error('Example data unavailable');return r.json();}).catch(error=>{cache.delete(key);throw error;}));
      const loaded=await cache.get(key);if(token!==galleryToken)return;
      galleryData=loaded;$('gallery-status').textContent=`Seed ${key.split('-')[0]} · paired trained models. Summary scores above average all three seeds.`;gallery();
    } catch(error){if(token===galleryToken)$('gallery-status').textContent='Example data could not load. Aggregate measurements and raw-data links remain available.';}
  }
  function render() {
    const m=model(),path=chosenPath(),k=size(chosenMask),onPath=path.masks[k]===chosenMask;
    $('count').value=k;$('count-label').textContent=`${k} of 4${onPath?'':' · custom subset'}`;
    $('order-note').textContent=`Order: ${path.order.map(i=>i+1).join(' → ')}. ${path.validation_selected?'Chosen using validation results before test analysis.':'A descriptive comparison; this order was not selected using test accuracy.'}`;
    $('branches').innerHTML=Array.from({length:4},(_,i)=>`<label><input type="checkbox" data-branch="${i}" ${chosenMask&(1<<i)?'checked':''}> Branch ${i+1}<small>${[2500,2000,1500,1000][i]} → ${[2000,1500,1000,500][i]} units<br>${chosenMask&(1<<i)?'Drop 50% in training':'Always keep in training'}<br>Always keep at testing</small></label>`).join('');
    $('branches').querySelectorAll('input').forEach(input=>input.onchange=()=>{const bit=1<<Number(input.dataset.branch);setMask(input.checked?chosenMask|bit:chosenMask&~bit);});
    $('subset-note').textContent=chosenMask===0?'Baseline: every branch is used during both training and testing.':`These models were trained with 50% dropout on branches ${maskText(chosenMask)}. Other branches were always kept. The test network uses all six learned layers at gain one.`;
    $('accuracy').textContent=pct(m.metrics.accuracy.mean);$('delta').textContent=pp(m.paired_vs_none.accuracy_pp.mean);$('harm').textContent=pct(m.paired_vs_none.harm_rate.mean);$('repair').textContent=pct(m.paired_vs_none.repair_rate.mean);
    $('uncertainty').textContent=`Paired three-seed accuracy change: ${CI(m.paired_vs_none.accuracy_pp)}. These are changes caused by different training configurations, not predictions made with missing inference layers.`;
    chart();subsetTable();classTable();marginalTable();gallery();
  }
  ['order','scale'].forEach(id=>$(id).addEventListener('change',()=>{stop();if(id==='order')chosenMask=chosenPath().masks[size(chosenMask)];render();}));
  $('endpoint').addEventListener('change',()=>{stop();render();loadGallery();});
  $('seed').addEventListener('change',loadGallery);
  ['category','digit-class','comparison'].forEach(id=>$(id).addEventListener('change',gallery));
  $('count').addEventListener('input',()=>setMask(chosenPath().masks[Number($('count').value)]));
  $('none').onclick=()=>setMask(0);$('all').onclick=()=>setMask(15);
  $('animate').onclick=()=>{if(timer){stop();return;}setMask(0);$('animate').textContent='Pause';$('animate').setAttribute('aria-pressed','true');let k=0;timer=setInterval(()=>{setMask(chosenPath().masks[++k],false);if(k===4)stop();},1600);};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  render();loadGallery();
  window.halfdropExplorer={setMask,getModel:model,stop,loadGallery,chosenPath};
})();
