const fs=require('fs'),path=require('path'),assert=require('assert/strict'),crypto=require('crypto'),{JSDOM}=require('jsdom');
(async()=>{
const root=path.resolve(__dirname,'..'),dir=path.join(root,'docs/transcripts/research-session');
const data=JSON.parse(fs.readFileSync(path.join(dir,'transcript.json'))),manifest=JSON.parse(fs.readFileSync(path.join(dir,'export-manifest.json')));
assert.equal(data.messageCount,111);assert.equal(data.messages.length,data.messageCount);assert.equal(new Set(data.messages.map(m=>m.sourceItemId)).size,111);
assert.equal(data.messages.filter(m=>m.role==='user').length,5);assert.equal(data.messages.at(-1).text,manifest.cutoffRequest);assert.equal(data.messages.at(-1).role,'user');
for(const m of data.messages){assert(['user','assistant'].includes(m.role));assert(['request','commentary','final'].includes(m.phase));assert(Date.parse(m.timestamp)<=Date.parse(data.endedAt));assert(!/\/Users\/|<system\b|<developer\b|<environment_context\b|<in-app-browser-context\b|:codex-file-citation/.test(m.text));assert(!/<script\b|<iframe\b|<object\b|\son\w+=|href=["']javascript:/i.test(m.html));}
for(const [f,h]of Object.entries(manifest.files))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(dir,f))).digest('hex'),h);
const dom=new JSDOM(fs.readFileSync(path.join(dir,'index.html'),'utf8'),{url:'https://yaroslavvb.github.io/gradient-dissent/transcripts/research-session/',runScripts:'outside-only',pretendToBeVisual:true});const w=dom.window,d=w.document;
w.fetch=async()=>({ok:true,json:async()=>data});w.HTMLElement.prototype.scrollIntoView=()=>{};
w.eval(fs.readFileSync(path.join(dir,'transcript.js'),'utf8'));await new Promise(r=>setTimeout(r,50));
assert.equal(d.querySelectorAll('.message').length,111);assert.equal(d.getElementById('message-count').textContent,'111 messages');
const count=()=>[...d.querySelectorAll('.message')].filter(x=>!x.hidden).length;
d.getElementById('show-progress').checked=false;d.getElementById('show-progress').dispatchEvent(new w.Event('change'));assert.equal(count(),10);
d.getElementById('search').value='ConvNeXt';d.getElementById('search').dispatchEvent(new w.Event('input'));assert.equal(count(),1);
d.getElementById('show-progress').checked=true;d.getElementById('show-progress').dispatchEvent(new w.Event('change'));assert(count()>1);
d.getElementById('search').value='no_possible_match_012345';d.getElementById('search').dispatchEvent(new w.Event('input'));assert.equal(count(),0);assert.equal(d.getElementById('empty-results').hidden,false);
w.location.hash='#message-001';w.dispatchEvent(new w.HashChangeEvent('hashchange'));assert.equal(count(),1);assert.equal(d.querySelector('#message-001 .linked-note').hidden,false);
w.location.hash='';d.getElementById('clear-search').click();assert.equal(count(),111);assert.equal(d.getElementById('search').value,'');
let checked=0;for(const a of d.querySelectorAll('a[href],script[src],link[href]')){
 const raw=a.getAttribute('href')||a.getAttribute('src'),url=new URL(raw,w.location.href);
 if(url.origin!=='https://yaroslavvb.github.io'||!url.pathname.startsWith('/gradient-dissent/'))continue;
 const target=path.join(root,'docs',decodeURIComponent(url.pathname.slice('/gradient-dissent/'.length)));let file=target;
 if(url.pathname.endsWith('/'))file=path.join(file,'index.html');assert(fs.existsSync(file),raw);
 if(url.pathname===w.location.pathname&&url.hash)assert(d.getElementById(decodeURIComponent(url.hash.slice(1))),raw);checked++;
}
assert(d.querySelector('a[href="https://yaroslavvb.github.io/gradient-dissent/significance/depth-robustness-audit.pdf"]'));
console.log(`PASS: 111 messages; publication cutoff, visible-only schema, digests, search, progress toggle, permalinks, PDF and ${checked} internal links.`);w.close();
})().catch(e=>{console.error(e);process.exit(1)});
