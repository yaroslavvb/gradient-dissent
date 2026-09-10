import fs from 'node:fs';import assert from 'node:assert/strict';import {createRequire} from 'node:module';import {performance} from 'node:perf_hooks';
const require=createRequire(import.meta.url),Core=require('../../docs/ciresan-stochastic-depth/browser-lab/core.js'),fixture=JSON.parse(fs.readFileSync(new URL('./reference-fixture.json',import.meta.url)));
const backend=process.argv.includes('--gpu')?'webgpu':'cpu';
if(backend==='webgpu'){const {create,globals}=await import('webgpu');Object.assign(globalThis,globals);Object.defineProperty(globalThis,'navigator',{value:{gpu:create(['backend=metal'])},configurable:true});}
const tf=await import('@tensorflow/tfjs');if(backend==='webgpu')await import('@tensorflow/tfjs-backend-webgpu');await tf.setBackend(backend);await tf.ready();if(backend==='webgpu')tf.env().set('WEBGPU_CPU_FORWARD',false);
const E=require('../../docs/ciresan-stochastic-depth/browser-lab/tf-engine.js')(tf,Core),started=performance.now();let checked=0,worst=0;
function near(actual,expected,name,tolerance=3e-5){assert.equal(actual.length,expected.length,name);for(let i=0;i<actual.length;i++){const error=Math.abs(actual[i]-expected[i]);worst=Math.max(worst,error);assert(error<tolerance*(1+Math.abs(expected[i])),`${name}[${i}] ${actual[i]} != ${expected[i]}`);checked++;}}
const active=mask=>[0,1,2,3].map(j=>!!(mask&(1<<j))),probs=[.5,.5,.5,.5];
try{
 for(const c of fixture.training_cases){
  const model=new E.Model(fixture.architecture.widths,201,fixture.initial_parameters),xs=tf.tensor2d(fixture.inputs.values,fixture.inputs.shape),ys=tf.tidy(()=>tf.oneHot(tf.tensor1d(fixture.labels.values,'int32'),2));
  const logits=tf.tidy(()=>model.forward(xs,{training:true,active:active(c.active_mask),dropProbs:probs}));near(await logits.data(),c.logits.values,'logits mask'+c.active_mask);logits.dispose();
  const gradient=tf.tidy(()=>model.gradients(xs,ys,active(c.active_mask),probs));near(await gradient.value.data(),[c.mean_cross_entropy],'loss');
  for(let i=0;i<model.params.length;i++){const key=fixture.parameter_order[i],g=gradient.grads[model.params[i].name],expected=c.gradients[key];if(expected===null)assert.equal(g,undefined,'skipped gradient '+key);else near(await g.data(),expected.values,'gradient '+key);}
  tf.dispose(gradient);model.dispose();xs.dispose();ys.dispose();assert.equal(tf.memory().numTensors,0,'mask leaks');
 }
 const model=new E.Model(fixture.architecture.widths,201,fixture.initial_parameters),xs=tf.tensor2d(fixture.inputs.values,fixture.inputs.shape),ys=tf.tidy(()=>tf.oneHot(tf.tensor1d(fixture.labels.values,'int32'),2));
 let logits=tf.tidy(()=>model.forward(xs));near(await logits.data(),fixture.inference_case.logits.values,'inference gain1');logits.dispose();
 for(const step of fixture.momentum_sequence.steps){const loss=model.trainBatch(xs,ys,active(step.active_mask),probs,fixture.momentum_sequence.learning_rate,fixture.momentum_sequence.momentum);near(await loss.data(),[step.mean_cross_entropy_before_update],'step loss');loss.dispose();for(let i=0;i<model.params.length;i++){const name=fixture.parameter_order[i];near(await model.params[i].data(),step.parameters_after_update[name].values,'updated '+name);near(await model.velocities[i].data(),step.velocities_after_update[name].values,'velocity '+name);}}
 logits=tf.tidy(()=>model.forward(xs));near(await logits.data(),fixture.momentum_sequence.full_depth_logits_after_sequence.values,'final full inference');logits.dispose();
 const before=tf.memory().numTensors;for(let k=0;k<40;k++){const loss=model.trainBatch(xs,ys,active(k%16),probs,.005,.9);await loss.data();loss.dispose();}assert.equal(tf.memory().numTensors,before,'repeated updates leak tensors');model.dispose();xs.dispose();ys.dispose();assert.equal(tf.memory().numTensors,0);
 const a=new E.Model(Core.PRESETS.tiny.widths,201),b=new E.Model(Core.PRESETS.tiny.widths,201);for(let i=0;i<a.params.length;i++)assert.deepEqual(await a.params[i].data(),await b.params[i].data());a.dispose();b.dispose();
 const r1=Core.rng(123),r2=Core.rng(123);for(let k=0;k<100;k++)assert.deepEqual(Core.gates(0,.5,r1).draws,Core.gates(15,.5,r2).draws);
 const cfg={preset:'small',plan:'sweep',mask:15,trainN:1000,testN:2000,epochs:5,lr:.03,seed:201,dropRate:.5,batchSize:64,backend:'auto',order:[3,0,2,1]};assert.deepEqual(Core.queue(Core.validate(cfg)),[0,8,9,13,15]);assert.throws(()=>Core.validate({...cfg,epochs:101}));assert.throws(()=>Core.validate({...cfg,order:[0,0,1,2]}));
 // Allocation/clone failure must also release tensors already created.
 const broken=structuredClone(fixture.initial_parameters);broken['layer_3.weight'].values=[1];assert.throws(()=>new E.Model(fixture.architecture.widths,201,broken));assert.equal(tf.memory().numTensors,0,'constructor failure leaked');
 const partial=new E.Model(fixture.architecture.widths,201,fixture.initial_parameters),countBefore=tf.memory().numTensors,originalClone=partial.params[2].clone;partial.params[2].clone=()=>{throw Error('injected snapshot allocation failure');};assert.throws(()=>partial.snapshot());assert.equal(tf.memory().numTensors,countBefore,'partial snapshot leaked');partial.params[2].clone=originalClone;partial.dispose();
 const report={status:'passed',backend,tfjs:tf.version.tfjs,activeMasks:16,momentumSequence:fixture.momentum_sequence.steps.map(s=>s.active_mask),scalarComparisons:checked,maxAbsoluteDifference:worst,remainingTensors:tf.memory().numTensors,seconds:(performance.now()-started)/1000,nativeWebGPU:backend==='webgpu'?'Dawn Metal (kernel test, not browser UI test)':null};console.log(JSON.stringify(report));fs.writeFileSync(new URL(`./test-engine-${backend}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
 tf.backend().dispose();process.exit(0);
}catch(e){console.error(e.stack);process.exit(1);}
