'use strict';
(() => {
  const D=window.HYPOTHESIS_DATA,$=id=>document.getElementById(id);
  if(!D || !D.models.length) { $('model-note').textContent='The experiment data could not be loaded. Use the report or raw-data links below.';return; }
  const names={residual:'Residual, no dropout',sd_constant:'SD, fixed drop rates',sd_annealed:'SD, drop rates decay to zero',residual_unit_dropout:'Residual + unit dropout',plain:'Plain MLP, forced crop bypass'};
  const methods={
    sd_constant:'Training: skip branches 1–4 with probabilities 10%, 20%, 30%, 40% on each minibatch. These rates stay fixed across epochs: three of the four middle branches run on average.',
    sd_annealed:'Training: drop rates for branches 1–4 start at 20%, 40%, 60%, 80% and fall linearly to zero by epoch 100. The average number of active middle branches rises from two to four.',
    residual:'Training: every branch runs. Residual bypasses carry existing features alongside each learned transformation, but this model was never trained with branches skipped.',
    residual_unit_dropout:'Training: randomly zero 20% of individual hidden activations, rather than skipping whole branches. Every branch is computed; this unit dropout is switched off for the inference audit.',
    plain:'Training: an ordinary fully connected network, with no residual bypasses or branch dropout. Unchecking a box inserts a crop bypass that these weights were never trained to use.'
  };
  const pct=x=>(100*x).toFixed(2)+'%', bits=m=>[0,1,2,3].map(i=>(m>>i)&1).join('');
  let mask=15,focusIndex=null,timer=null,model=null;
  const cost=[5,3,1.5,.5];
  $('mask-controls').innerHTML=cost.map((c,i)=>`<label title="Unchecked: skip this learned transformation and crop to the first ${[2000,1500,1000,500][i]} features. The feature layer and classifier remain."><input type="checkbox" data-branch="${i}" checked> Keep branch ${i+1}<small>${c}M MACs · ${[2500,2000,1500,1000][i]} → ${[2000,1500,1000,500][i]}</small></label>`).join('');
  function drawDigit(canvas,index){
    const encoded=D.images[String(index)];if(!encoded)return;
    const bytes=atob(encoded),context=canvas.getContext('2d');
    if(!context)return;
    const tiny=document.createElement('canvas');tiny.width=tiny.height=28;
    const c=tiny.getContext('2d'),pixels=c.createImageData(28,28);
    for(let i=0;i<784;i++){const v=bytes.charCodeAt(i);pixels.data[i*4]=Math.round(20+v*.92);pixels.data[i*4+1]=Math.round(44+v*.83);pixels.data[i*4+2]=Math.round(52+v*.79);pixels.data[i*4+3]=255;}
    c.putImageData(pixels,0,0);context.imageSmoothingEnabled=false;context.drawImage(tiny,0,0,canvas.width,canvas.height);
  }
  function setMask(m){mask=m;document.querySelectorAll('[data-branch]').forEach(e=>{e.checked=!!(mask&(1<<Number(e.dataset.branch)));});renderMask();}
  function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='Play all 16 masks';$('play').setAttribute('aria-pressed','false');}
  function network(){
    const xs=[75,245,415,585,755,925],parts=[];
    parts.push('<path d="M75 130 H925" fill="none" stroke="#92aaa4" stroke-width="4"/>');
    for(let i=0;i<4;i++){
      const x=xs[i+1],on=!!(mask&(1<<i));
      if(model.recipe==='plain'&&on)parts.push(`<path d="M${x-69} 130 H${x+69}" stroke="white" stroke-width="9"/>`);
      parts.push(`<g class="branch"><path d="M${x-72} 130 Q${x-68} 52 ${x-36} 52 H${x+36} Q${x+69} 52 ${x+72} 130" fill="none" stroke="${on?'#098278':'#ccd5d2'}" stroke-width="${on?4:2}" ${on?'':'stroke-dasharray="6 5"'}/><rect x="${x-48}" y="31" width="96" height="43" rx="5" fill="${on?'#098278':'#edf1ef'}"/><text x="${x}" y="57" text-anchor="middle" style="fill:${on?'#fff':'#71847f'}">${on?'Affine '+(i+1):'Skipped'}</text><circle cx="${x+72}" cy="130" r="5" fill="#476e62"/><text x="${x}" y="161" text-anchor="middle">${model.recipe==='plain'&&on?'ReLU':'crop + ReLU'}</text><text x="${x}" y="183" text-anchor="middle" style="font-size:12px">${[2000,1500,1000,500][i]} units</text></g>`);
    }
    for(const [x,label,width] of [[75,'Features',2500],[925,'Classifier',10]])parts.push(`<g><title>Learned ${label.toLowerCase()} layer: always active, even with mask 0000</title><rect x="${x-60}" y="104" width="120" height="52" rx="5" fill="#173942"/><text x="${x}" y="135" text-anchor="middle" style="fill:#fff">${label}</text><text x="${x}" y="183" text-anchor="middle">${width} units</text><text x="${x}" y="206" text-anchor="middle" style="font-size:14px;font-weight:650;fill:#173942">always on</text></g>`);
    $('network').innerHTML=parts.join('');
    $('network').setAttribute('aria-label',`${names[model.recipe]}; mask ${bits(mask)}, first to last branch; learned feature layer and classifier always active.`);
  }
  function frontier(){
    const x=f=>75+f*880,y=a=>272-a*225,parts=[];
    for(const v of [0,.25,.5,.75,1])parts.push(`<path d="M75 ${y(v)} H955" stroke="#e2e9e6"/><text x="63" y="${y(v)+5}" text-anchor="end">${v*100}%</text>`);
    for(const v of [.2,.4,.6,.8,1])parts.push(`<text x="${x(v)}" y="298" text-anchor="middle">${v*100}%</text>`);
    parts.push('<text x="75" y="22">Test accuracy</text><text x="520" y="327" text-anchor="middle">Affine MACs relative to the full network</text>');
    for(const m of model.masks){parts.push(`<circle data-mask="${m.id}" tabindex="0" role="button" aria-label="Mask ${bits(m.id)}, ${pct(m.accuracy)} accuracy, ${pct(m.cost)} MACs" cx="${x(m.cost)}" cy="${y(m.accuracy)}" r="${m.id===mask?8:5}" fill="${m.id===mask?'#e18b37':'#087b73'}" stroke="white" stroke-width="2"><title>${bits(m.id)} · ${pct(m.accuracy)} · ${pct(m.cost)} MACs</title></circle>`);}
    const chosen=model.masks[mask];parts.push(`<text x="${Math.min(850,x(chosen.cost)+12)}" y="${Math.max(38,y(chosen.accuracy)-14)}" style="font-weight:650;fill:#965013">${bits(mask)} · ${pct(chosen.accuracy)}</text>`);
    $('frontier').innerHTML=parts.join('');
    $('frontier').querySelectorAll('[data-mask]').forEach(e=>{e.onclick=()=>{stop();setMask(Number(e.dataset.mask));};e.onkeydown=ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();e.onclick();}};});
  }
  function classTable(){
    let html='<caption>Test accuracy by true digit and exact layer mask. Bits read shallow → deep.</caption><thead><tr><th>Digit / n</th>'+model.masks.map(m=>`<th>${bits(m.id)}</th>`).join('')+'</tr></thead><tbody>';
    for(let digit=0;digit<10;digit++){
      const n=model.masks[15].per_class[digit].n;
      html+=`<tr><th>${digit} / ${n}</th>`+model.masks.map(m=>{const v=m.per_class[digit].accuracy;return `<td class="${m.id===mask?'active':''}" style="background:rgba(20,143,122,${.04+.36*v})" title="Digit ${digit}, mask ${bits(m.id)}, ${pct(v)}">${(100*v).toFixed(1)}</td>`;}).join('')+'</tr>';
    }
    $('class-table').innerHTML=html+'</tbody>';
  }
  function filteredExamples(){const digit=$('class').value,category=$('category').value;return model.examples.filter(e=>(digit==='all'||e.label===Number(digit))&&(category==='all'||e.category===category));}
  function renderFocus(){
    const examples=filteredExamples();
    const example=examples.find(e=>e.index===focusIndex)||examples[0];
    if(!example){focusIndex=null;$('digit').style.visibility='hidden';$('example-id').textContent='NO QUALIFYING EXAMPLE';$('prediction').textContent='This category is empty';$('example-note').textContent='Choose another digit or resilience category.';return;}
    $('digit').style.visibility='visible';focusIndex=example.index;
    drawDigit($('digit'),example.index);$('digit').setAttribute('aria-label',`MNIST test image ${example.index}, true digit ${example.label}`);
    $('example-id').textContent=`TEST #${example.index} · TRUE DIGIT ${example.label}`;
    const predicted=example.pred[mask];$('prediction').textContent=`Predicts ${predicted} · ${predicted===example.label?'correct':'incorrect'}`;
    $('prediction').style.color=predicted===example.label?'#08776f':'#b54b34';
    $('example-note').textContent=`${example.retained_two_correct_count} of the 6 two-branch masks preserve this digit. Current cross-entropy: ${example.ce[mask].toFixed(3)} nats.`;
    $('gallery').querySelectorAll('button').forEach(b=>b.classList.toggle('selected',Number(b.dataset.index)===focusIndex));
  }
  function gallery(){
    const category=$('category').value,examples=filteredExamples();
    $('gallery').innerHTML=examples.map(e=>`<button type="button" data-index="${e.index}" aria-label="Inspect true digit ${e.label}, test index ${e.index}, ${e.category}"><canvas width="56" height="56" aria-hidden="true"></canvas><span>${e.label} → ${e.pred[mask]}</span><small>#${e.index} · ${e.retained_two_correct_count}/6</small></button>`).join('')||(category==='all'?'No examples in this selection.':'No examples met this category’s fixed selection rule.');
    $('gallery').querySelectorAll('button').forEach(b=>{drawDigit(b.querySelector('canvas'),Number(b.dataset.index));b.onclick=()=>{focusIndex=Number(b.dataset.index);renderFocus();};});
  }
  function renderMask(){
    const m=model.masks[mask];$('mask-status').textContent=`Mask ${bits(mask)} · ${m.retained}/4 optional branches kept · ${m.retained+2}/6 learned layers still run`;
    $('remaining-note').hidden=mask!==0;
    $('remaining-note').textContent=mask===0
      ?`Two learned layers remain: 784 pixels → 2,500 ReLU features → crop to 500 → 10 digit scores. This trained shallow classifier uses ${pct(m.cost)} of counted arithmetic and scores ${pct(m.accuracy)}. Uniform random guessing averages 10%; these weights still classify digits.`
      :'';
    $('remaining-note').classList.toggle('all-skipped',mask===0);
    $('mask-accuracy').textContent=pct(m.accuracy);$('mask-macs').textContent=pct(m.cost);
    network();frontier();classTable();gallery();renderFocus();
  }
  function renderModel(){
    const id=`${$('recipe').value}-${$('state').value}-s${$('seed').value}`;model=D.models.find(m=>m.id===id);
    if(!model)throw new Error('Missing planned checkpoint: '+id);
    $('method-note').textContent=methods[model.recipe];
    $('model-note').textContent=`Saved weights: epoch ${model.epoch} · full six-layer model accuracy: ${pct(model.dense_accuracy)}. Kept branches are not rescaled, and unit dropout is off during inference.`;
    $('mask-table').innerHTML='<caption>All 16 interventions for the selected checkpoint</caption><thead><tr><th>Mask</th><th>Kept</th><th>MACs</th><th>Accuracy</th><th>CE</th><th>Harmed</th><th>Repaired</th></tr></thead><tbody>'+model.masks.map(m=>`<tr><td>${bits(m.id)}</td><td>${m.retained}</td><td>${pct(m.cost)}</td><td>${pct(m.accuracy)}</td><td>${m.ce.toFixed(4)}</td><td>${pct(m.harm)}</td><td>${pct(m.repair)}</td></tr>`).join('')+'</tbody>';
    renderMask();
  }
  ['recipe','state','seed'].forEach(id=>$(id).addEventListener('change',()=>{stop();renderModel();}));
  ['class','category'].forEach(id=>$(id).addEventListener('change',()=>{gallery();renderFocus();}));
  document.querySelectorAll('[data-branch]').forEach(e=>e.addEventListener('change',()=>{stop();const bit=1<<Number(e.dataset.branch);setMask(e.checked?mask|bit:mask&~bit);}));
  $('restore').onclick=()=>{stop();setMask(15);};
  $('play').onclick=()=>{if(timer){stop();return;}$('play').textContent='Pause masks';$('play').setAttribute('aria-pressed','true');timer=setInterval(()=>setMask((mask+1)%16),1600);};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  renderModel();
  window.hypothesisExplorer={setMask,getModel:()=>model,stop};
})();
