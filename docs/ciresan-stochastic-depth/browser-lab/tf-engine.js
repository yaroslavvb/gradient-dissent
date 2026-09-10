/* TensorFlow.js primitives only; no DOM, data fetches or backend policy. */
(function(root,factory){if(typeof module==='object')module.exports=factory;root.createMnistEngine=factory;})(globalThis,(tf,Core)=>{
'use strict';let serial=0;
class Model{
 constructor(widths,seed=201,initial=null){
  this.widths=[...widths];this.params=[];this.velocities=[];this.layers=[];this.disposed=false;const random=Core.rng(seed),id=++serial;
  try{for(let l=0;l<widths.length-1;l++){
   const a=widths[l],b=widths[l+1],bound=1/Math.sqrt(a),w=initial?initial[`layer_${l}.weight`].values:Float32Array.from({length:a*b},()=> (random()*2-1)*bound),bias=initial?initial[`layer_${l}.bias`].values:Float32Array.from({length:b},()=> (random()*2-1)*bound);
   // Preserve PyTorch row-major [out,in] layout; transpose only in matmul.
   const weight=tf.tidy(()=>tf.variable(tf.tensor2d(w,[b,a]),true,`m${id}_l${l}_w`));this.params.push(weight);
   const biasVar=tf.tidy(()=>tf.variable(tf.tensor1d(bias),true,`m${id}_l${l}_b`));this.params.push(biasVar);this.layers.push({w:weight,b:biasVar});
  }
  for(let i=0;i<this.params.length;i++)this.velocities.push(tf.tidy(()=>tf.variable(tf.zerosLike(this.params[i]),false,`m${id}_v${i}`)));
  }catch(error){this.dispose();throw error;}
 }
 forward(x,{training=false,active=[true,true,true,true],dropProbs=[0,0,0,0]}={}){
  let h=tf.relu(tf.add(tf.matMul(x,this.layers[0].w,false,true),this.layers[0].b));
  for(let j=0;j<4;j++){
   const layer=this.layers[j+1],skip=tf.slice(h,[0,0],[-1,this.widths[j+2]]);
   if(active[j]){const gain=training?1/(1-dropProbs[j]):1;h=tf.relu(tf.add(skip,tf.mul(tf.add(tf.matMul(h,layer.w,false,true),layer.b),gain)));}
   else h=tf.relu(skip);
  }
  const last=this.layers[5];return tf.add(tf.matMul(h,last.w,false,true),last.b);
 }
 activeParams(active){return this.params.filter((_,i)=>{const layer=Math.floor(i/2);return layer===0||layer===5||active[layer-1];});}
 gradients(xs,ys,active,dropProbs){return tf.variableGrads(()=>tf.losses.softmaxCrossEntropy(ys,this.forward(xs,{training:true,active,dropProbs})),this.activeParams(active));}
 trainBatch(xs,ys,active,dropProbs,lr,momentum=.9){
  return tf.tidy(()=>{
   const {value,grads}=this.gradients(xs,ys,active,dropProbs);
   for(let i=0;i<this.params.length;i++){
    const p=this.params[i],g=grads[p.name];if(!g)continue;
    // Key by the fixed parameter index, never the position in a sparse gradient map.
    const v=this.velocities[i],next=tf.add(tf.mul(v,momentum),g);v.assign(next);p.assign(tf.sub(p,tf.mul(next,lr)));
   }
   return value;
  });
 }
 snapshot(){const copies=[];try{for(const p of this.params)copies.push(p.clone());return copies;}catch(error){disposeSnapshot(copies);throw error;}}
 restore(snapshot){if(snapshot.length!==this.params.length)throw Error('Checkpoint parameter count mismatch');this.params.forEach((p,i)=>p.assign(snapshot[i]));}
 dispose(){if(this.disposed)return;this.disposed=true;this.params.forEach(p=>p.dispose());this.velocities.forEach(v=>v.dispose());}
}
function disposeSnapshot(s){if(s)s.forEach(t=>t.dispose());}
function tensors(split,n=split.count){const x=new Float32Array(n*784);for(let i=0;i<x.length;i++)x[i]=split.pixels[i]/255;return tf.tidy(()=>({x:tf.tensor2d(x,[n,784]),y:tf.oneHot(tf.tensor1d(split.labels.subarray(0,n),'int32'),10)}));}
async function evaluate(model,x,labels,active=[true,true,true,true],onChunk=async()=>{}){
 const n=labels.length,classes=model.widths.at(-1),all=new Float32Array(n*classes);
 for(let i=0;i<n;i+=256){const count=Math.min(256,n-i),out=tf.tidy(()=>model.forward(tf.slice(x,[i,0],[count,-1]),{active}));try{all.set(await out.data(),i*classes);}finally{out.dispose();}await onChunk();}
 return Core.scores(all,labels,classes);
}
async function qualify(widths,batchSize=4){
 let model,xs,ys;const classes=widths.at(-1),random=Core.rng(19);
 try{model=new Model(widths,7);xs=tf.tensor2d(Float32Array.from({length:batchSize*widths[0]},()=>random()),[batchSize,widths[0]]);ys=tf.tidy(()=>tf.oneHot(tf.tensor1d(Array.from({length:batchSize},(_,i)=>i%classes),'int32'),classes));
 for(const m of [15,0,5,10]){const loss=model.trainBatch(xs,ys,[0,1,2,3].map(i=>!!(m&(1<<i))),[.5,.5,.5,.5],.01);try{if(!Number.isFinite((await loss.data())[0]))throw Error('Training qualification produced non-finite loss');}finally{loss.dispose();}}
 const logits=tf.tidy(()=>model.forward(xs));try{const values=await logits.data();if(!values.every(Number.isFinite))throw Error('Evaluation qualification produced non-finite values');}finally{logits.dispose();}
 }finally{model?.dispose();xs?.dispose();ys?.dispose();}
}
return{Model,disposeSnapshot,tensors,evaluate,qualify};
});
