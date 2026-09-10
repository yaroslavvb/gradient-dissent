/* All training runs in this worker. The UI never receives GPU tensors. */
'use strict';
importScripts('vendor/tf.min.js','vendor/tf-backend-webgpu.min.js','core.js','tf-engine.js','dataset.js');
const Core=MnistLabCore,Engine=createMnistEngine(tf,Core),Data=MnistLabData;
let busy=false,cancelled=false,live=null,taskId=null;
const send=(type,data={})=>postMessage({type,taskId,...data});
const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
const checkStop=()=>{if(cancelled)throw new Error('__STOPPED__');};
function releaseLive(){if(!live)return;live.model.dispose();Engine.disposeSnapshot(live.selectedSnapshot);Engine.disposeSnapshot(live.finalSnapshot);live.testX.dispose();live=null;}
function strip(score){const {predictions,probabilities,losses,...rest}=score;return rest;}
async function chooseBackend(config){
 releaseLive();const choices=config.backend==='auto'?['webgpu','webgl','cpu']:[config.backend],attempts=[];
 for(const name of choices){checkStop();try{
  if(name==='webgpu'&&!navigator.gpu)throw Error('WebGPU is unavailable in this worker.');
  if(name==='cpu'&&config.preset==='full')throw Error('Full Ciresan is too large for the CPU fallback. Choose a smaller network.');
  send('status',{message:`Checking ${name.toUpperCase()} forward, backward and parameter updates…`});
  if(!await tf.setBackend(name))throw Error('Backend did not initialize.');await tf.ready();
  if(name==='webgpu')tf.env().set('WEBGPU_CPU_FORWARD',false);
  const begin=performance.now();await Engine.qualify(config.widths,config.batchSize);attempts.push({backend:name,passed:true,seconds:(performance.now()-begin)/1000});
  return{backend:name,version:tf.version.tfjs,attempts,webglFloat32:name==='webgl'?tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE'):null};
 }catch(error){attempts.push({backend:name,passed:false,reason:String(error.message).slice(0,240)});send('status',{message:`${name.toUpperCase()} could not train this model. ${config.backend==='auto'?'Trying the next backend…':'Choose Auto or another backend.'}`});}}
 throw Error('No selected backend could train this network. '+attempts.map(a=>a.backend+': '+a.reason).join(' '));
}
async function runOne(config,position,total,qualification){
 checkStop();releaseLive();const runId=taskId+'-'+position+'-'+config.mask,started=performance.now();let model,train,val,bestSnapshot,finalSnapshot,testX,history=[],epochsCompleted=0,trainingSeconds=0,evalSeconds=0;
 const skipCounts=[0,0,0,0],stepCounts=[0,0,0,0],dropProbs=[0,1,2,3].map(j=>config.mask&(1<<j)?config.dropRate:0);
 send('runStart',{runId,config,position,total,qualification});
 try{
  send('status',{message:'Loading and verifying the separate training and validation data…'});
  const [trainSplit,valSplit]=await Promise.all([Data.load('train'),Data.load('validation')]);checkStop();
  const initStarted=performance.now();train=Engine.tensors(trainSplit,config.trainN);val=Engine.tensors(valSplit);model=new Engine.Model(config.widths,config.seed);
  // Synchronize initial allocation; qualification weights are never reused.
  await model.params[0].data();const initializationSeconds=(performance.now()-initStarted)/1000;
  const initBytes=await Promise.all(model.params.map(async p=>new Uint8Array((await p.data()).buffer)));
  const initialBuffer=new Uint8Array(initBytes.reduce((s,b)=>s+b.byteLength,0));let offset=0;for(const b of initBytes){initialBuffer.set(b,offset);offset+=b.length;}
  const initializationSHA256=await Data.sha(initialBuffer),orderRandom=Core.rng(config.seed+100000),gateRandom=Core.rng(config.seed+200000);let orderHash=2166136261,gateHash=2166136261;
  const digest=(h,v)=>Math.imul(h^(v>>>0),16777619)>>>0;
  let bestCE=Infinity,bestEpoch=0;
  send('status',{message:`Training run ${position+1}/${total}: dropout on branches ${Core.maskName(config.mask)}.`});
  for(let epoch=1;epoch<=config.epochs;epoch++){
   checkStop();const order=Core.shuffle(config.trainN,orderRandom);for(const i of order)orderHash=digest(orderHash,i);
   const epochStart=performance.now();let lossSum=tf.scalar(0),seen=0,steps=0,lastActive=[true,true,true,true];
   try{
    for(let i=0;i<config.trainN;i+=config.batchSize){
     checkStop();const batchIndices=order.subarray(i,Math.min(i+config.batchSize,config.trainN)),n=batchIndices.length,g=Core.gates(config.mask,config.dropRate,gateRandom);lastActive=g.active;
     for(let j=0;j<4;j++){gateHash=digest(gateHash,Math.floor(g.draws[j]*4294967296));stepCounts[j]++;skipCounts[j]+=+!g.active[j];}
     const next=tf.tidy(()=>{const ix=tf.tensor1d(batchIndices,'int32'),xs=tf.gather(train.x,ix),ys=tf.gather(train.y,ix),loss=model.trainBatch(xs,ys,g.active,dropProbs,config.lr,.9);return tf.add(lossSum,tf.mul(loss,n));});lossSum.dispose();lossSum=next;seen+=n;steps++;
     if(steps%8===0||seen===config.trainN){const value=(await lossSum.data())[0]/seen;if(!Number.isFinite(value))throw Error('Training diverged. Try a smaller learning rate.');send('batch',{runId,position,total,epoch,epochs:config.epochs,fraction:seen/config.trainN,active:g.active,stochasticLoss:value,trainingSeconds:trainingSeconds+(performance.now()-epochStart)/1000});await tick();}
    }
    const trainLoss=(await lossSum.data())[0]/seen;trainingSeconds+=(performance.now()-epochStart)/1000;checkStop();
    const ev=performance.now(),validation=await Engine.evaluate(model,val.x,valSplit.labels,undefined,async()=>{await tick();checkStop();});evalSeconds+=(performance.now()-ev)/1000;
    if(validation.ce<bestCE){bestCE=validation.ce;bestEpoch=epoch;Engine.disposeSnapshot(bestSnapshot);bestSnapshot=model.snapshot();}
    epochsCompleted=epoch;const row={epoch,stochasticLoss:trainLoss,validation:strip(validation),trainingSeconds,elapsedSeconds:(performance.now()-started)/1000};history.push(row);send('epoch',{runId,row,active:lastActive,skipCounts:[...skipCounts],steps:stepCounts[0]});
   }finally{lossSum.dispose();}
  }
  checkStop();finalSnapshot=model.snapshot();send('status',{message:'Training complete. Evaluating the held-out test set with every branch present…'});
  const testSplit=await Data.load(config.testN===10000?'test_full':'test');checkStop();testX=tf.tidy(()=>{const tensors=Engine.tensors(testSplit);tensors.y.dispose();return tensors.x;});
  const evaluationStart=performance.now(),final=await Engine.evaluate(model,testX,testSplit.labels,undefined,async()=>{await tick();checkStop();});model.restore(bestSnapshot);
  const selected=bestEpoch===config.epochs?final:await Engine.evaluate(model,testX,testSplit.labels,undefined,async()=>{await tick();checkStop();});evalSeconds+=(performance.now()-evaluationStart)/1000;
  const result={id:runId,status:'complete',config:{...config,actualBackend:qualification.backend},qualification,history,epochsCompleted,bestEpoch,endpoints:{selected,final},timing:{trainingSeconds,initializationSeconds,evaluationSeconds:evalSeconds,totalSeconds:(performance.now()-started)/1000},dropCounts:skipCounts,steps:stepCounts[0],provenance:{initializationSHA256,orderHash:orderHash.toString(16),gateDrawHash:gateHash.toString(16),trainSHA256:trainSplit.sha256,validationSHA256:valSplit.sha256,testSHA256:testSplit.sha256,testSource:config.testN===10000?'all official test images':'balanced official test subset',selection:'earliest minimum validation CE, evaluated each completed epoch',testAccess:'only after training and checkpoint selection'}};
  live={model,selectedSnapshot:bestSnapshot,finalSnapshot,testX,testSplit,runId};model=null;bestSnapshot=null;finalSnapshot=null;testX=null;
  send('result',{run:result});
 }catch(error){send('interrupted',{runId,config,history,epochsCompleted,trainingSeconds,status:cancelled?'stopped':'failed'});throw error;
 }finally{model?.dispose();Engine.disposeSnapshot(bestSnapshot);Engine.disposeSnapshot(finalSnapshot);testX?.dispose();train?.x.dispose();train?.y.dispose();val?.x.dispose();val?.y.dispose();}
}
self.onmessage=async event=>{
 const msg=event.data;
 if(msg.type==='stop'){cancelled=true;Data.abortPending();return;}
 if(busy){send('error',{message:'An operation is already running.'});return;}
 if(msg.type==='start'){
  busy=true;cancelled=false;taskId=msg.taskId;
  try{const config=Core.validate(msg.config),qualification=await chooseBackend(config),masks=Core.queue(config);send('backend',{qualification});for(let i=0;i<masks.length;i++){checkStop();await runOne({...config,mask:masks[i]},i,masks.length,qualification);}send('done',{message:`Completed ${masks.length} run${masks.length===1?'':'s'}. Test predictions use every branch.`});}
  catch(error){send(cancelled?'stopped':'error',{message:cancelled?'Stopped. Incomplete runs are not scored on the test set.':String(error.message||error).slice(0,650)});}
  finally{busy=false;cancelled=false;}
 }else if(msg.type==='evaluate'){
  if(!live){send('error',{message:'Train a model first.'});return;}busy=true;cancelled=false;
  try{if(!['selected','final'].includes(msg.endpoint)||!Array.isArray(msg.active)||msg.active.length!==4||msg.active.some(v=>typeof v!=='boolean'))throw Error('Invalid inference mask.');live.model.restore(msg.endpoint==='selected'?live.selectedSnapshot:live.finalSnapshot);const start=performance.now(),score=await Engine.evaluate(live.model,live.testX,live.testSplit.labels,msg.active,async()=>{await tick();checkStop();});send('inference',{runId:live.runId,endpoint:msg.endpoint,active:msg.active,score,seconds:(performance.now()-start)/1000});}
  catch(error){send(cancelled?'stopped':'error',{message:cancelled?'Evaluation stopped.':String(error.message)});}finally{busy=false;cancelled=false;}
 }
};
