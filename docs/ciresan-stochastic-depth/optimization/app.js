'use strict';
(() => {
  const runs=window.OPTIMIZATION_DATA.runs;
  const colors=['#087b91','#d96d22','#6756ad','#369354','#b24576'];
  const $=id=>document.getElementById(id);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function selected(){
    const g=$('group').value;
    return runs.filter(r=>g==='recommended'?r.spec.confirmation_group==='normalized-dropout-step':g==='raw'?r.spec.confirmation_group==='raw-linear-dropout-step':g==='fragile'?r.spec.confirmation_group==='frozen-b256-bf16'&&r.spec.confirmation_type==='fresh-seed':/^opt-speed-(eager|graph|fused)-tf32-b64-s1$/.test(r.run_id));
  }
  function label(r){return $('group').value==='kernels'?(r.spec.mode==='eager'?'Eager':r.spec.fused?'CUDA Graph + fused SGD':'CUDA Graph'):`Seed ${r.spec.seed}`;}
  function render(){
    const group=selected(), key=$('clock').value, detail=$('zoom').value==='detail';
    const pts=group.flatMap(r=>r.history); const xMax=Math.max(...pts.map(p=>p[key]));
    const left=65,right=975,top=25,bottom=345,lo=detail?97:0,hi=detail?99:100;
    const x=v=>left+v/xMax*(right-left), y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
    let svg='<defs><clipPath id="plot-clip"><rect x="65" y="25" width="910" height="320"/></clipPath></defs>';
    for(let i=0;i<=4;i++){let v=lo+(hi-lo)*i/4;svg+=`<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#dce3e7"/><text x="53" y="${y(v)+5}" text-anchor="end">${v.toFixed(detail?1:0)}%</text>`;}
    for(let i=0;i<=5;i++){let v=xMax*i/5;svg+=`<text x="${x(v)}" y="371" text-anchor="middle">${v.toFixed(key==='epoch'?0:1)}</text>`;}
    svg+=`<line class="target" x1="${left}" x2="${right}" y1="${y(98.63)}" y2="${y(98.63)}"/><text x="975" y="${y(98.63)-9}" text-anchor="end">Target 98.63%</text>`;
    group.forEach((r,i)=>{const path=r.history.map((p,j)=>`${j?'L':'M'}${x(p[key]).toFixed(2)},${y(100*p.test.accuracy).toFixed(2)}`).join(' ');svg+=`<path class="chart-path" clip-path="url(#plot-clip)" stroke="${colors[i]}" d="${path}"><title>${esc(label(r))}: ${esc(r.hardware.gpu)}</title></path>`;});
    svg+=`<text x="520" y="397" text-anchor="middle">${key==='epoch'?'Epoch':key==='training_seconds'?'Synchronized training time (seconds)':'Invocation to score (seconds)'}</text>`;
    $('chart').innerHTML=svg;
    $('legend').innerHTML=group.map((r,i)=>`<span><i style="background:${colors[i]}"></i>${esc(label(r))}</span>`).join('');
    const reached=group.filter(r=>r.threshold).length;
    $('chart-note').textContent=`${reached} of ${group.length} runs reached the target. ${detail?'The detail view clips points below 97%; choose the full range to inspect early training or failed classes.':'Full range shows low-accuracy plateaus as well as early learning.'} ${$('group').value==='kernels'?'All three scored accuracy trajectories are identical; only the time axis changes.':'The report records each assigned GPU variant. Curves stop at the first qualifying epoch or the fixed horizon.'}`;
    $('chart-data').innerHTML='<caption>All measured points for the selected comparison</caption><thead><tr><th>Run</th><th>Epoch</th><th>Accuracy</th><th>Training s</th><th>Invocation s</th></tr></thead><tbody>'+group.flatMap(r=>r.history.map(p=>`<tr><td>${esc(label(r))}</td><td>${p.epoch}</td><td>${(100*p.test.accuracy).toFixed(2)}%</td><td>${p.training_seconds.toFixed(3)}</td><td>${p.run_wall_seconds.toFixed(3)}</td></tr>`)).join('')+'</tbody>';
  }
  ['group','clock','zoom'].forEach(id=>$(id).addEventListener('change',render));render();
})();
