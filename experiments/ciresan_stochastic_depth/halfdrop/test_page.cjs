// jsdom integration/data checks. --fixture uses ONLY in-memory synthetic data;
// default mode requires the complete actual data.js and all six local galleries.
// Neither mode makes network calls or writes report/experiment artifacts.
'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert');
const {JSDOM,VirtualConsole}=require('jsdom');
const page=path.resolve(__dirname,'../../../docs/ciresan-stochastic-depth/training-50');
const fixtureMode=process.argv.includes('--fixture');
const T=4.302652729911275,SEEDS=[201,202,203],ENDS=['selected','final'];
const pop=m=>m.toString(2).replace(/0/g,'').length;
const avg=a=>a.reduce((x,y)=>x+y,0)/a.length;
function interval(values){const mean=avg(values),sd=Math.sqrt(values.reduce((s,v)=>s+(v-mean)**2,0)/(values.length-1)),half=T*sd/Math.sqrt(values.length);return{values,mean,n:values.length,df:values.length-1,sample_sd:sd,ci95_low:mean-half,ci95_high:mean+half};}
function permutations(a){return a.length?a.flatMap((v,i)=>permutations(a.filter((_,j)=>i!==j)).map(p=>[v,...p])):[[]];}
function fixture(){
  const D={models:[],by_k:[],edges:[],marginals:[],paths:[],gallery_files:{},wandb_runs:{},verification:{expected_runs:48,complete_runs:48,endpoint_states:96}};
  const gallery={};
  for(const endpoint of ENDS){
    for(let mask=0;mask<16;mask++){
      const k=pop(mask),weight=mask+k*k,factor=endpoint==='selected'?1:.8;
      const seeds=SEEDS.map((seed,i)=>({seed,run_id:`halfdrop-m${String(mask).padStart(2,'0')}-s${seed}-v1`,accuracy:.98-.0001*weight*(1+.1*i)*factor+i*.0001,ce:.08+.001*k+i*.0001,paired_vs_none:{accuracy_pp:-.01*weight*(1+.1*i)*factor,harm_rate:.00015*weight*(1+.1*i)*factor,repair_rate:.00005*weight*(1+.1*i)*factor}}));
      for(const s of seeds)D.wandb_runs[s.run_id]=`https://wandb.ai/fixture/halfdrop/runs/${s.run_id}`;
      const m={endpoint,drop_mask:mask,k,seeds,metrics:{accuracy:interval(seeds.map(s=>s.accuracy)),ce:interval(seeds.map(s=>s.ce))},paired_vs_none:{},per_class:[]};
      for(const key of ['accuracy_pp','harm_rate','repair_rate'])m.paired_vs_none[key]=interval(seeds.map(s=>s.paired_vs_none[key]));
      for(let digit=0;digit<10;digit++)m.per_class.push({digit,n:1000,accuracy:m.metrics.accuracy,...m.paired_vs_none});
      D.models.push(m);
    }
    for(let k=0;k<5;k++){const models=D.models.filter(m=>m.endpoint===endpoint&&m.k===k);D.by_k.push({endpoint,k,masks:models.map(m=>m.drop_mask),metrics:{accuracy:interval(SEEDS.map((_,i)=>avg(models.map(m=>m.seeds[i].accuracy))))}});}
    for(let branch=0;branch<4;branch++){
      for(let mask=0;mask<16;mask++)if(!(mask&(1<<branch))){
        const next=mask|(1<<branch),base=D.models.find(m=>m.endpoint===endpoint&&m.drop_mask===mask),target=D.models.find(m=>m.endpoint===endpoint&&m.drop_mask===next);
        const seeds=SEEDS.map((seed,i)=>{const accuracy_pp=100*(target.seeds[i].accuracy-base.seeds[i].accuracy),repair_rate=.0001*(branch+1)*(i+1);return{seed,accuracy_pp,repair_rate,harm_rate:repair_rate-accuracy_pp/100};});
        const metrics={};for(const key of ['accuracy_pp','harm_rate','repair_rate'])metrics[key]=interval(seeds.map(s=>s[key]));
        D.edges.push({endpoint,branch,from_mask:mask,to_mask:next,seeds,metrics});
      }
      const edges=D.edges.filter(e=>e.endpoint===endpoint&&e.branch===branch);
      D.marginals.push({endpoint,layer:branch,branch,context_size:'all',accuracy_pp:interval(SEEDS.map((_,i)=>avg(edges.map(e=>e.seeds[i].accuracy_pp))))});
    }
    for(const seed of SEEDS){
      const file=`gallery-s${seed}-${endpoint}.json`,models=[],images={};D.gallery_files[`${seed}-${endpoint}`]=file;
      for(let mask=0;mask<16;mask++){
        const examples=[];
        for(let index=0;index<(mask?102:100);index++){
          const label=index<100?Math.floor(index/10):index%10;
          const baseline_pred=(index<100?index%17===0:index===101)?(label+1)%10:label;
          let pred=baseline_pred;if(mask&&(index%(7+mask)===0||index>=100))pred=baseline_pred===label?(label+1)%10:label;
          const probs=Array(10).fill(.01),baseline_probs=Array(10).fill(.01);probs[pred]=.91;baseline_probs[baseline_pred]=.91;
          const category=baseline_pred===label&&pred!==label?'harmed':baseline_pred!==label&&pred===label?'repaired':'unchanged';
          examples.push({index,label,pred,baseline_pred,probs,baseline_probs,ce:-Math.log(probs[label]),baseline_ce:-Math.log(baseline_probs[label]),category,selection:[index<100?'fixed':'first_'+category]});
          images[index]=Buffer.alloc(784,index).toString('base64');
        }
        models.push({drop_mask:mask,baseline_mask:0,examples});
      }
      gallery[file]={seed,endpoint,models,images,fixed_indices:Array.from({length:100},(_,i)=>i)};
    }
  }
  for(const order of permutations([0,1,2,3])){let mask=0;D.paths.push({order,masks:[0,...order.map(j=>(mask|=1<<j))],validation_selected:D.paths.length===0});}
  return{data:D,gallery};
}
function close(a,b,eps=1e-11){assert(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=eps,`${a} != ${b}`);}
function checkInterval(I,values){assert.equal(I.n,3);assert.equal(I.df,2);const expected=interval(values);for(const key of ['mean','sample_sd','ci95_low','ci95_high'])close(I[key],expected[key]);}
function checkGallery(g,D){
  assert(SEEDS.includes(g.seed)&&ENDS.includes(g.endpoint));assert.equal(g.models.length,16);
  assert.equal(new Set(g.fixed_indices).size,100);assert.equal(g.fixed_indices.length,100);
  const fixedCounts=Array(10).fill(0),base=g.models.find(m=>m.drop_mask===0),baseByIndex=new Map(base.examples.map(e=>[e.index,e]));
  for(const index of g.fixed_indices){assert(baseByIndex.has(index));fixedCounts[baseByIndex.get(index).label]++;}
  assert.deepEqual(fixedCounts,Array(10).fill(10));
  const modelMasks=new Set();let examples=0;
  for(const model of g.models){
    assert(!modelMasks.has(model.drop_mask));modelMasks.add(model.drop_mask);assert.equal(model.baseline_mask,0);
    assert(model.examples.length>=100&&model.examples.length<=140);const seen=new Set();
    for(const e of model.examples){
      examples++;assert(!seen.has(e.index));seen.add(e.index);assert(e.index>=0&&e.index<10000);
      assert(Number.isInteger(e.label)&&e.label>=0&&e.label<10);assert.equal(Buffer.from(g.images[e.index],'base64').length,784);
      assert(Array.isArray(e.selection)&&e.selection.length>0);
      for(const [key,pred,ce] of [['probs',e.pred,e.ce],['baseline_probs',e.baseline_pred,e.baseline_ce]]){
        const p=e[key];assert.equal(p.length,10);p.forEach(v=>assert(Number.isFinite(v)&&v>=0&&v<=1));close(avg(p)*10,1,1e-12);
        assert.equal(p.indexOf(Math.max(...p)),pred);assert(Number.isFinite(ce)&&ce>=0);
        if(p[e.label]>0)close(-Math.log(p[e.label]),ce,1e-8);
      }
      const category=e.baseline_pred===e.label&&e.pred!==e.label?'harmed':e.baseline_pred!==e.label&&e.pred===e.label?'repaired':'unchanged';
      assert.equal(e.category,category);
      if(g.fixed_indices.includes(e.index)){
        assert(e.selection.includes('fixed'));const b=baseByIndex.get(e.index);
        assert.equal(e.label,b.label);assert.equal(e.baseline_pred,b.pred);assert.deepEqual(e.baseline_probs,b.probs);close(e.baseline_ce,b.ce);
      }else assert(e.selection.includes('first_harmed')||e.selection.includes('first_repaired'));
      if(model.drop_mask===0){assert.equal(e.category,'unchanged');assert.equal(e.pred,e.baseline_pred);assert.deepEqual(e.probs,e.baseline_probs);}
    }
    g.fixed_indices.forEach(i=>assert(seen.has(i)));
  }
  assert.deepEqual([...modelMasks].sort((a,b)=>a-b),Array.from({length:16},(_,i)=>i));
  assert(D.gallery_files[`${g.seed}-${g.endpoint}`]);return examples;
}

async function main(){
  const errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e));
  const dom=new JSDOM(fs.readFileSync(path.join(page,'index.html'),'utf8'),{runScripts:'outside-only',url:'https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/training-50/',virtualConsole:vc});
  const w=dom.window,d=w.document,fixtures=fixtureMode?fixture():null,fetches=new Map(),galleryPayloads=new Map();
  w.HTMLCanvasElement.prototype.getContext=function(){return{createImageData:(x,y)=>({data:new Uint8ClampedArray(x*y*4)}),putImageData(){},drawImage(){}};};
  w.fetch=async file=>{
    assert.equal(typeof file,'string');assert(!file.includes('://')&&!file.includes('..'),'Gallery fetch must be local');
    fetches.set(file,(fetches.get(file)||0)+1);
    const payload=fixtureMode?fixtures.gallery[file]:JSON.parse(fs.readFileSync(path.join(page,file),'utf8'));
    assert(payload,'Unknown gallery fixture');galleryPayloads.set(file,payload);
    return{ok:true,json:async()=>JSON.parse(JSON.stringify(payload))};
  };
  if(fixtureMode)w.HALFDROP_DATA=JSON.parse(JSON.stringify(fixtures.data));
  else w.eval(fs.readFileSync(path.join(page,'data.js'),'utf8'));
  const D=w.HALFDROP_DATA;assert.equal(D.models.length,32);assert.equal(D.paths.length,24);
  assert.equal(D.verification.complete_runs,48);assert.equal(D.verification.endpoint_states,96);
  const keys=new Set(),runIds=new Set();
  for(const m of D.models){
    const key=`${m.endpoint}/${m.drop_mask}`;assert(!keys.has(key));keys.add(key);assert(ENDS.includes(m.endpoint));assert.equal(m.k,pop(m.drop_mask));
    assert.deepEqual(Array.from(m.seeds,r=>r.seed),SEEDS);m.seeds.forEach(r=>runIds.add(r.run_id));
    checkInterval(m.metrics.accuracy,Array.from(m.seeds,r=>r.accuracy));checkInterval(m.metrics.ce,Array.from(m.seeds,r=>r.ce));
    checkInterval(m.paired_vs_none.accuracy_pp,Array.from(m.seeds,r=>r.paired_vs_none.accuracy_pp));
    close(m.paired_vs_none.accuracy_pp.mean,100*(m.paired_vs_none.repair_rate.mean-m.paired_vs_none.harm_rate.mean),1e-10);
  }
  assert.equal(runIds.size,48);
  assert.equal(D.edges.length,64);
  for(const edge of D.edges){
    assert.equal(edge.to_mask,edge.from_mask|(1<<edge.branch));assert(!(edge.from_mask&(1<<edge.branch)));
    for(const key of ['accuracy_pp','harm_rate','repair_rate'])checkInterval(edge.metrics[key],Array.from(edge.seeds,s=>s[key]));
    close(edge.metrics.accuracy_pp.mean,100*(edge.metrics.repair_rate.mean-edge.metrics.harm_rate.mean),1e-10);
  }
  for(const by of D.by_k){const rows=D.models.filter(m=>m.endpoint===by.endpoint&&m.k===by.k);checkInterval(by.metrics.accuracy,SEEDS.map((_,i)=>avg(rows.map(r=>r.seeds[i].accuracy))));}
  assert.equal(new Set(D.paths.map(p=>p.order.join(','))).size,24);assert.equal(D.paths.filter(p=>p.validation_selected).length,1);
  for(const p of D.paths){assert.deepEqual(Array.from(p.order).sort(),[0,1,2,3]);let mask=0;assert.deepEqual(Array.from(p.masks),[0,...Array.from(p.order,j=>(mask|=1<<j))]);}
  w.eval(fs.readFileSync(path.join(page,'app.js'),'utf8'));assert(w.halfdropExplorer);
  const change=(id,value,event='change')=>{const e=d.getElementById(id);e.value=value;e.dispatchEvent(new w.Event(event));};
  const healthy=()=>{for(const id of ['accuracy-chart','subset-table','class-table','marginal-table','probability-chart','uncertainty','edge-effect'])assert(!/NaN|Infinity|undefined/.test(d.getElementById(id).innerHTML),`${id} contains a nonfinite/undefined value`);assert.equal(errors.length,0,errors.map(e=>e.message).join('\n'));};
  const summarySnapshot=()=>['accuracy','delta','harm','repair','uncertainty','class-table'].map(id=>d.getElementById(id).innerHTML);
  const verifyRunLinks=(referenceMask,currentMask,seed,endpoint)=>{
    const runId=mask=>D.models.find(m=>m.endpoint===endpoint&&m.drop_mask===mask).seeds.find(s=>s.seed===seed).run_id;
    const links=[runId(referenceMask),runId(currentMask)].map(id=>D.wandb_runs&&D.wandb_runs[id]).filter(Boolean);
    assert.deepEqual(Array.from(d.querySelectorAll('#run-links a'),a=>a.href),links,'Example links use a different seed/subset reference');
  };
  let cases=0,previousCases=0;
  for(const endpoint of ENDS){
    change('endpoint',endpoint);await w.halfdropExplorer.loadGallery();
    for(const p of D.paths){
      change('order',p.order.join(','));
      for(let k=0;k<5;k++){
        change('count',String(k),'input');assert.equal(w.halfdropExplorer.getModel().drop_mask,p.masks[k]);
        assert.equal(d.getElementById('count').value,String(k));assert.equal(d.querySelectorAll('#branches input:checked').length,k);
        assert.equal(d.getElementById('accuracy').textContent,(100*w.halfdropExplorer.getModel().metrics.accuracy.mean).toFixed(2)+'%');
        assert.equal(d.querySelectorAll('#subset-table tbody tr').length,16);assert.equal(d.querySelectorAll('#class-table tbody tr').length,10);
        assert.equal(d.querySelectorAll('#marginal-table tbody tr').length,4);healthy();cases++;
      }
    }
    for(let mask=0;mask<16;mask++){w.halfdropExplorer.setMask(mask);assert.equal(w.halfdropExplorer.getModel().drop_mask,mask);healthy();}
    d.getElementById('none').click();assert.equal(w.halfdropExplorer.getModel().drop_mask,0);
    const checkbox=d.querySelector('#branches input[data-branch="2"]');checkbox.checked=true;checkbox.dispatchEvent(new w.Event('change'));
    assert.equal(w.halfdropExplorer.getModel().drop_mask,4);assert.equal(d.querySelectorAll('#branches input:checked').length,1);
    assert(/Always keep at testing/.test(d.getElementById('branches').textContent));
    assert(/all six learned layers at gain one/i.test(d.getElementById('subset-note').textContent));
    change('scale','full');healthy();change('scale','zoom');healthy();
    for(const seed of SEEDS){
      change('seed',String(seed));await w.halfdropExplorer.loadGallery();
      const file=D.gallery_files[`${seed}-${endpoint}`],g=galleryPayloads.get(file);assert(g);checkGallery(g,D);
      for(let mask=0;mask<16;mask++){
        w.halfdropExplorer.setMask(mask);const gm=g.models.find(m=>m.drop_mask===mask);
        for(const category of ['all','harmed','repaired','unchanged']){
          change('category',category);change('digit-class','all');
          const examples=gm.examples.filter(e=>category==='all'||e.category===category);
          assert.equal(d.querySelectorAll('#gallery button').length,examples.length);
          assert.equal(d.getElementById('digit-image').hidden,examples.length===0);
          if(examples.length){
            const last=examples[examples.length-1];d.querySelector(`#gallery button[data-index="${last.index}"]`).click();
            assert(d.getElementById('image-id').textContent.includes(`TEST #${last.index}`));
            assert(d.getElementById('baseline-pred').textContent.startsWith(String(last.baseline_pred)+' ·'));
            assert(d.getElementById('treatment-pred').textContent.startsWith(String(last.pred)+' ·'));
            assert.equal(d.querySelectorAll('#probability-chart rect').length,20);
            const heights=Array.from(d.querySelectorAll('#probability-chart rect'),r=>Number(r.getAttribute('height')));
            for(let digit=0;digit<10;digit++){close(heights[2*digit],155*last.baseline_probs[digit],1e-10);close(heights[2*digit+1],155*last.probs[digit],1e-10);}
            verifyRunLinks(0,mask,seed,endpoint);
          }
          healthy();cases++;
        }
        change('category','all');change('digit-class','4');
        assert.equal(d.querySelectorAll('#gallery button').length,gm.examples.filter(e=>e.label===4).length);
        change('digit-class','all');
      }
      // Every previous-prefix comparison is reconstructed from matched fixed
      // examples, not from the existing no-dropout harm/repair category.
      change('category','all');change('digit-class','all');
      for(const p of D.paths){
        change('order',p.order.join(','));
        for(let k=0;k<5;k++){
          const mask=p.masks[k],reference=p.masks[Math.max(0,k-1)];
          change('comparison','baseline');w.halfdropExplorer.setMask(mask);
          assert(d.getElementById('edge-effect').hidden,'Edge effect must be hidden for baseline-reference examples');
          const priorSummary=summarySnapshot();change('comparison','previous');
          assert.deepEqual(summarySnapshot(),priorSummary,'Previous-example selector changed population/baseline summaries');
          const paragraph=d.getElementById('edge-effect');
          if(k===0){assert(paragraph.hidden);assert.equal(paragraph.textContent,'');}
          else{
            const edge=D.edges.find(e=>e.endpoint===endpoint&&e.from_mask===reference&&e.to_mask===mask);assert(edge);
            const pp=v=>(v>=0?'+':'')+v.toFixed(3)+' pp',pct=v=>(100*v).toFixed(2)+'%';
            const ci='['+[edge.metrics.accuracy_pp.ci95_low,edge.metrics.accuracy_pp.ci95_high].map(v=>v.toFixed(3)).join(', ')+'] pp';
            assert(!paragraph.hidden);assert(paragraph.textContent.includes(`Making branch ${edge.branch+1} additionally droppable`));
            assert(paragraph.textContent.includes(`${pp(edge.metrics.accuracy_pp.mean)} accuracy change (95% interval ${ci})`));
            assert(paragraph.textContent.includes(`${pct(edge.metrics.harm_rate.mean)} newly wrong and ${pct(edge.metrics.repair_rate.mean)} newly correct`));
            assert(paragraph.textContent.includes('all 10,000 test images and average the three matched seeds'));
          }
          const current=g.models.find(m=>m.drop_mask===mask),ref=g.models.find(m=>m.drop_mask===reference);
          const byIndex=new Map(ref.examples.map(e=>[e.index,e]));
          const expected=current.examples.filter(e=>e.selection.includes('fixed')).map(e=>{
            const r=byIndex.get(e.index);assert(r&&r.label===e.label);
            const category=r.pred===e.label&&e.pred!==e.label?'harmed':r.pred!==e.label&&e.pred===e.label?'repaired':'unchanged';
            return{...e,reference:r,category};
          });
          assert.equal(expected.length,100);assert.equal(d.querySelectorAll('#gallery button').length,100);
          assert(/Only the fixed 100-image reference panel/.test(d.getElementById('comparison-note').textContent));
          assert(/summary rates and digit table still compare with no dropout/.test(d.getElementById('comparison-note').textContent));
          for(const category of ['harmed','repaired','unchanged']){
            change('category',category);const examples=expected.filter(e=>e.category===category);
            assert.deepEqual(Array.from(d.querySelectorAll('#gallery button'),b=>Number(b.dataset.index)),examples.map(e=>e.index));
            assert.equal(d.getElementById('digit-image').hidden,examples.length===0);
            if(examples.length){
              const chosen=examples[examples.length-1];d.querySelector(`#gallery button[data-index="${chosen.index}"]`).click();
              assert(d.getElementById('baseline-pred').textContent.startsWith(String(chosen.reference.pred)+' ·'));
              assert(d.getElementById('treatment-pred').textContent.startsWith(String(chosen.pred)+' ·'));
              const bars=d.querySelectorAll('#probability-chart rect');assert.equal(bars.length,20);
              for(let digit=0;digit<10;digit++){close(Number(bars[digit*2].getAttribute('height')),155*chosen.reference.probs[digit],1e-10);close(Number(bars[digit*2+1].getAttribute('height')),155*chosen.probs[digit],1e-10);}
              verifyRunLinks(reference,mask,seed,endpoint);
              assert(d.getElementById('example-note').textContent.includes(`${chosen.reference.ce.toFixed(3)} → ${chosen.ce.toFixed(3)}`));
            }
            assert.deepEqual(summarySnapshot(),priorSummary);healthy();previousCases++;
          }
          change('category','all');
        }
        const custom=Array.from({length:16},(_,i)=>i).find(m=>!p.masks.includes(m));
        change('comparison','baseline');w.halfdropExplorer.setMask(custom);
        const baselineSummary=summarySnapshot();change('comparison','previous');
        assert.equal(d.querySelectorAll('#gallery button').length,0);assert(d.getElementById('digit-image').hidden);
        assert.equal(d.querySelectorAll('#probability-chart rect').length,0);assert.equal(d.querySelectorAll('#run-links a').length,0);
        assert(d.getElementById('edge-effect').hidden);assert.equal(d.getElementById('edge-effect').textContent,'');
        assert(/Previous step unavailable/.test(d.getElementById('reference-label').textContent));
        assert(/custom subset is outside the chosen order/.test(d.getElementById('comparison-note').textContent));
        assert.deepEqual(summarySnapshot(),baselineSummary);healthy();previousCases++;
        change('comparison','baseline');assert(d.querySelectorAll('#gallery button').length>=100);
      }
      change('comparison','baseline');
    }
  }
  assert.equal(fetches.size,6);for(const count of fetches.values())assert.equal(count,1,'Cached gallery fetched repeatedly');
  assert(/did not add dropout gradually within a training run/.test(d.body.textContent));
  assert(/All six learned layers run for every test prediction/.test(d.body.textContent));
  for(const id of ['accuracy-chart','probability-chart'])assert(d.getElementById(id).getAttribute('aria-label'));
  assert(!d.querySelector('a[href="#"]'));
  w.halfdropExplorer.stop();w.close();
  console.log(JSON.stringify({status:'passed',mode:fixtureMode?'in-memory synthetic fixtures only':'actual completed exports',runs:48,aggregate_models:32,endpoints:2,orders:24,order_steps:240,lazy_gallery_files:fetches.size,interaction_cases:cases,previous_step_cases:previousCases}));
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});
