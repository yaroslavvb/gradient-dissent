(function(root,factory){const api=factory();if(typeof module==='object')module.exports=api;root.MnistLabData=api;})(globalThis,()=>{
'use strict';const cache=new Map(),requests=new Set();let manifestPromise;
async function request(url,format){const controller=new AbortController();requests.add(controller);try{const r=await fetch(url,{signal:controller.signal});if(!r.ok)throw Error(`Could not download ${url} (${r.status}).`);return await r[format]();}finally{requests.delete(controller);}}
function abortPending(){for(const controller of requests)controller.abort();}
async function manifest(){if(!manifestPromise)manifestPromise=request('dataset/manifest.json','json').catch(e=>{manifestPromise=null;throw e;});return manifestPromise;}
function decode(buffer){
 const b=new Uint8Array(buffer),v=new DataView(buffer);if(b.length<32||String.fromCharCode(...b.subarray(0,8))!=='GDMNIST1')throw Error('Invalid MNIST asset header.');
 const count=v.getUint32(8,true),rows=v.getUint32(12,true),cols=v.getUint32(16,true),po=v.getUint32(20,true),lo=v.getUint32(24,true),io=v.getUint32(28,true);
 if(!count||count>10000||rows!==28||cols!==28||po!==32||lo!==32+count*784||io!==Math.ceil((lo+count)/4)*4||b.byteLength!==io+count*4)throw Error('Invalid MNIST asset dimensions.');
 const pixels=b.subarray(po,lo),labels=b.subarray(lo,lo+count),indices=Uint32Array.from({length:count},(_,i)=>v.getUint32(io+4*i,true));if(labels.some(y=>y>9))throw Error('Invalid MNIST labels.');return{count,pixels,labels,indices};
}
async function sha(bytes){const hash=await crypto.subtle.digest('SHA-256',bytes);return [...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('');}
async function load(name){
 if(!cache.has(name))cache.set(name,(async()=>{const m=await manifest(),spec=m.splits[name]||m.optional_splits[name];if(!spec)throw Error('Unknown dataset split.');let bytes=await request('dataset/'+spec.gzip_file,'arrayBuffer');const head=new Uint8Array(bytes,0,Math.min(bytes.byteLength,2));if(head[0]===31&&head[1]===139){if(typeof DecompressionStream==='undefined')throw Error('This browser needs support for gzip decompression. Try a current browser.');if(await sha(bytes)!==spec.gzip_sha256)throw Error('Compressed dataset hash mismatch.');bytes=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();}
 if(await sha(bytes)!==spec.sha256)throw Error('Dataset checksum mismatch. Reload to fetch a fresh copy.');const split=decode(bytes);if(split.count!==spec.count)throw Error('Dataset count mismatch.');return{...split,name,sha256:spec.sha256};})().catch(e=>{cache.delete(name);throw e;}));return cache.get(name);
}
return{manifest,load,decode,sha,abortPending};
});
