(() => {
  'use strict';
  const data = window.CIRESAN_DATA;
  const $ = id => document.getElementById(id);
  const names = {plain:'Plain MLP',residual:'Residual control',sd_constant:'Constant SD',sd_annealed:'Decreasing SD',residual_unit_dropout:'Unit dropout',
    'source-plain':'Source · plain / ReLU head','source-linear':'Source · plain / linear head','source-residual':'Source · residual / ReLU head','source-sd':'Source · SD / ReLU head',
    'normalized-plain':'Normalized · plain','normalized-residual':'Normalized · residual','normalized-sd':'Normalized · decreasing SD'};
  const colors = ['#294257','#14846b','#166ac0','#bc5224','#8456ac','#be294b'];
  const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
  const ns = 'http://www.w3.org/2000/svg';
  function el(tag,attrs,text) {const node=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))node.setAttribute(k,v);if(text!==undefined)node.textContent=text;return node;}
  function metric(row,key) {
    if(key==='validation_accuracy')return 100*row.validation.accuracy;
    if(key==='validation_loss')return row.validation.loss;
    if(key==='train_accuracy')return 100*row.dense_train_probe.accuracy;
    if(key==='train_loss')return row.dense_train_probe.loss;
    return row.stochastic_training_loss;
  }
  function groups() {
    const cohort=$('cohort').value;
    const selected=data.runs.filter(r=>cohort==='pilot'?r.spec.stage==='pilot':r.spec.stage==='evaluate'&&r.spec.cohort===cohort);
    const map=new Map();
    for(const run of selected){const key=cohort==='pilot'?run.run_id.replace('pilot-',''):cohort==='fidelity'?run.spec.variant:run.spec.recipe;if(!map.has(key))map.set(key,[]);map.get(key).push(run);}
    return [...map.entries()].map(([key,runs],i)=>({key,name:names[key]||key,runs,color:colors[i%colors.length]}));
  }
  function render() {
    const series=groups(), key=$('metric').value, byTime=$('xaxis').value==='seconds';
    const svg=$('curves');while(svg.children.length>2)svg.lastChild.remove();
    $('legend').replaceChildren();
    const all=series.flatMap(g=>g.runs.flatMap(r=>r.history.filter(h=>h.validation).map(h=>({x:byTime?h.training_seconds:h.epoch,y:metric(h,key)}))));
    if(!all.length)return;
    const w=1000,h=410,left=72,right=24,top=20,bottom=53;
    let ymin=Math.min(...all.map(p=>p.y)), ymax=Math.max(...all.map(p=>p.y));
    const margin=Math.max((ymax-ymin)*.1,key.endsWith('accuracy')?.15:.002);
    ymin=Math.max(0,ymin-margin);ymax+=margin;if(key.endsWith('accuracy'))ymax=Math.min(100,ymax);
    const xmax=Math.max(...all.map(p=>p.x))*1.015;
    const X=x=>left+x/xmax*(w-left-right),Y=y=>h-bottom-(y-ymin)/(ymax-ymin)*(h-top-bottom);
    for(let i=0;i<=5;i++){
      const v=ymin+(ymax-ymin)*i/5,y=Y(v);
      svg.append(el('line',{x1:left,x2:w-right,y1:y,y2:y,stroke:'#dce3e7'}));
      svg.append(el('text',{x:left-10,y:y+5,'text-anchor':'end'},v.toFixed(key.endsWith('accuracy')?1:3)));
      const x=xmax*i/5;
      svg.append(el('text',{x:X(x),y:h-bottom+25,'text-anchor':'middle'},Math.round(x)));
    }
    svg.append(el('text',{x:(left+w-right)/2,y:h-5,'text-anchor':'middle'},byTime?'Synchronized training seconds':'Training epochs'));
    const path=points=>points.map((p,i)=>(i?'L':'M')+X(p.x).toFixed(2)+','+Y(p.y).toFixed(2)).join(' ');
    for(const g of series){
      const label=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=g.color;label.append(swatch,document.createTextNode(g.name));$('legend').append(label);
      const curves=g.runs.map(r=>r.history.filter(h=>h.validation).map(h=>({epoch:h.epoch,x:byTime?h.training_seconds:h.epoch,y:metric(h,key)})));
      if($('individual').checked&&g.runs.length>1)for(const curve of curves)svg.append(el('path',{d:path(curve),fill:'none',stroke:g.color,'stroke-width':1,opacity:.23}));
      const epochs=[...new Set(curves.flatMap(c=>c.map(p=>p.epoch)))].sort((a,b)=>a-b);
      const average=epochs.map(epoch=>{const points=curves.flatMap(c=>c.filter(p=>p.epoch===epoch));return{x:mean(points.map(p=>p.x)),y:mean(points.map(p=>p.y))};});
      svg.append(el('path',{d:path(average),fill:'none',stroke:g.color,'stroke-width':2.7,'stroke-linejoin':'round'}));
    }
    const label=$('metric').selectedOptions[0].textContent;
    $('svg-title').textContent=label+' by '+(byTime?'training time':'epoch');
    $('svg-desc').textContent=series.map(g=>g.name+': '+g.runs.length+' measured run(s)').join('. ')+'. Full final values appear in the accessible table below.';
    $('curve-note').textContent=(key==='stochastic_loss'?'Stochastic loss uses different masked networks across methods; compare dense training-probe loss for a common evaluation rule. ':'Training-probe scores use a fixed 10,000-image subset with dropout disabled. ')+(byTime?'Time includes the optimizer and sparse gradient diagnostics, but excludes evaluation and checkpointing. ':'Each epoch has 781 minibatches of 64 images; 16 shuffled training examples are omitted. ')+(series[0]?.runs.length>1?'Bold lines are seed means; faint lines are individual seeds.':'These pilots each use seed 1 and are exploratory.');
    const tbody=$('curve-table').querySelector('tbody');tbody.replaceChildren();
    for(const g of series){const last=g.runs.map(r=>r.history.filter(h=>h.validation).at(-1));const row=document.createElement('tr');for(const value of [g.name,g.runs.length,(100*mean(last.map(h=>h.validation.accuracy))).toFixed(2)+'%',mean(last.map(h=>h.validation.loss)).toFixed(4),mean(g.runs.map(r=>r.training_seconds)).toFixed(1)+' s']){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}tbody.append(row);}
  }
  if(!data.final){$('cohort').value='pilot';for(const o of $('cohort').options)if(o.value!=='pilot')o.disabled=true;$('headline').textContent='The pilots found a training failure in the raw-pixel residual variant. Baseline speed optimization is in progress before the multi-seed comparison resumes; the curves below show measured pilot results.';}
  else { $('headline').textContent='Stochastic depth cut 100-epoch training time by about 14.4%, but did not improve mean accuracy at the validation-selected checkpoint. Compare fitting, validation loss, and both reported test endpoints across three paired seeds.'; $('metric').value='validation_loss'; }
  for(const id of ['cohort','metric','xaxis','individual'])$(id).addEventListener('change',render);
  render();
})();
