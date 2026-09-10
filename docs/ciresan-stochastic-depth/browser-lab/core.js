/* Pure deterministic experiment logic, shared by the page, worker and tests. */
(function(root,factory){const api=factory();if(typeof module==='object')module.exports=api;root.MnistLabCore=api;})(globalThis,()=>{
'use strict';
const PRESETS={tiny:{name:'Tiny Ciresan',widths:[784,128,96,64,48,32,10]},small:{name:'Small Ciresan',widths:[784,256,192,128,96,64,10]},uniform:{name:'Equal-width residual MLP',widths:[784,128,128,128,128,128,10]},wide:{name:'Wide Ciresan',widths:[784,512,384,256,192,128,10]},full:{name:'Full Ciresan widths',widths:[784,2500,2000,1500,1000,500,10]}};
function rng(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
const popcount=m=>{let n=0;for(let i=0;i<4;i++)n+=(m>>i)&1;return n;};
const maskName=m=>[0,1,2,3].filter(i=>m&(1<<i)).map(i=>i+1).join(', ')||'none';
function permutations(a){return a.length?a.flatMap((v,i)=>permutations(a.filter((_,j)=>j!==i)).map(p=>[v,...p])):[[]];}
function prefixMasks(order){let m=0;return[0,...order.map(i=>(m|=1<<i))];}
function validate(c){
 if(!PRESETS[c.preset])throw Error('Choose a network preset.');
 for(const [key,min,max] of [['seed',0,2147483647],['epochs',1,100],['mask',0,15]])if(!Number.isInteger(c[key])||c[key]<min||c[key]>max)throw Error(`${key} must be an integer from ${min} to ${max}.`);
 if(![1000,4000,10000].includes(c.trainN)||![32,64,128].includes(c.batchSize)||![2000,10000].includes(c.testN))throw Error('Unsupported dataset or batch size.');
 if(![.25,.5,.75].includes(c.dropRate)||!Number.isFinite(c.lr)||c.lr<.0001||c.lr>.3)throw Error('Choose a valid dropout rate and learning rate.');
 if(!['single','paired','sweep'].includes(c.plan)||!['auto','webgpu','webgl','cpu'].includes(c.backend))throw Error('Unsupported run plan or backend.');
 if(!Array.isArray(c.order)||c.order.length!==4||[...c.order].sort().join(',')!=='0,1,2,3')throw Error('The order must contain each branch once.');
 return {...c,momentum:.9,widths:[...PRESETS[c.preset].widths]};
}
function queue(c){return c.plan==='single'?[c.mask]:c.plan==='paired'?[...new Set([0,c.mask])]:prefixMasks(c.order);}
function shuffle(n,random){const a=Int32Array.from({length:n},(_,i)=>i);for(let i=n-1;i>0;i--){const j=Math.floor(random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function gates(mask,p,random){const draws=[random(),random(),random(),random()];return{draws,active:draws.map((r,i)=>!(mask&(1<<i))||r>=p)};}
function parameterCount(widths){return widths.slice(1).reduce((n,b,i)=>n+(widths[i]+1)*b,0);}
function signature(c){return JSON.stringify([c.preset,c.trainN,c.testN,c.batchSize,c.epochs,c.lr,c.seed,c.dropRate,c.actualBackend||c.backend]);}
function scores(logits,labels,classes=10){
 const n=labels.length,predictions=new Uint8Array(n),probabilities=new Float32Array(n*classes),losses=new Float32Array(n),confusion=Array.from({length:classes},()=>Array(classes).fill(0));let correct=0,ce=0;
 for(let i=0;i<n;i++){const start=i*classes;let top=0,max=-Infinity;for(let j=0;j<classes;j++)if(logits[start+j]>max){max=logits[start+j];top=j;}let sum=0;for(let j=0;j<classes;j++)sum+=Math.exp(logits[start+j]-max);const loss=Math.log(sum)+max-logits[start+labels[i]];if(!Number.isFinite(loss))throw Error('Non-finite evaluation output. Try a smaller learning rate.');predictions[i]=top;losses[i]=loss;ce+=loss;correct+=+(top===labels[i]);confusion[labels[i]][top]++;for(let j=0;j<classes;j++)probabilities[start+j]=Math.exp(logits[start+j]-max)/sum;}
 const perClass=confusion.map((row,digit)=>({digit,n:row.reduce((a,b)=>a+b,0),correct:row[digit],accuracy:row.reduce((a,b)=>a+b,0)?row[digit]/row.reduce((a,b)=>a+b,0):null}));
 return{n,accuracy:correct/n,ce:ce/n,errors:n-correct,predictions,probabilities,losses,confusion,perClass};
}
function compare(a,b,labels){if(!a||!b||a.predictions.length!==labels.length||b.predictions.length!==labels.length)return null;let harmed=0,repaired=0;for(let i=0;i<labels.length;i++){harmed+=+(b.predictions[i]===labels[i]&&a.predictions[i]!==labels[i]);repaired+=+(b.predictions[i]!==labels[i]&&a.predictions[i]===labels[i]);}return{harmed,repaired,n:labels.length,deltaPP:100*(repaired-harmed)/labels.length};}
return{PRESETS,rng,popcount,maskName,permutations,prefixMasks,validate,queue,shuffle,gates,parameterCount,signature,scores,compare};
});
