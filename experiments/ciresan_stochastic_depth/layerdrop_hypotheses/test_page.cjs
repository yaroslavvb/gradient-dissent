// Static DOM/data integration checks; no browser or screenshot inspection.
const fs=require('fs'),path=require('path'),assert=require('assert');
const {JSDOM}=require('jsdom');
const dir=path.resolve(__dirname,'../../../docs/ciresan-stochastic-depth/hypotheses');
const html=fs.readFileSync(path.join(dir,'index.html'),'utf8');
const dom=new JSDOM(html,{runScripts:'outside-only',url:'https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/hypotheses/'});
const w=dom.window;
w.HTMLCanvasElement.prototype.getContext=function(){return {createImageData:(x,y)=>({data:new Uint8ClampedArray(x*y*4)}),putImageData(){},drawImage(){}};};
w.eval(fs.readFileSync(path.join(dir,'data.js'),'utf8'));w.eval(fs.readFileSync(path.join(dir,'app.js'),'utf8'));
const data=w.HYPOTHESIS_DATA;assert.equal(data.models.length,30);assert.equal(w.document.querySelectorAll('[data-branch]').length,4);
for(const m of data.models){
 for(const [id,value] of [['recipe',m.recipe],['state',m.state],['seed',String(m.seed)]]){w.document.getElementById(id).value=value;}
 w.document.getElementById('recipe').dispatchEvent(new w.Event('change'));
 assert.equal(w.hypothesisExplorer.getModel().id,m.id);
 for(let mask=0;mask<16;mask++){
  w.hypothesisExplorer.setMask(mask);
  assert.equal(w.document.querySelectorAll('#frontier circle').length,16);
  assert.equal(w.document.querySelectorAll('#class-table tbody tr').length,10);
  const shown=w.document.getElementById('mask-accuracy').textContent;
  assert.equal(shown,(m.masks[mask].accuracy*100).toFixed(2)+'%');
 }
 assert.equal(w.document.querySelectorAll('#mask-table tbody tr').length,16);
}
const select=(id,val)=>{w.document.getElementById(id).value=val;w.document.getElementById(id).dispatchEvent(new w.Event('change'));};
select('recipe','plain');select('state','selected');select('seed','101');select('class','1');select('category','preserved');
const candidates=w.hypothesisExplorer.getModel().examples.filter(e=>e.label===1&&e.category==='preserved');
if(!candidates.length)assert.equal(w.document.getElementById('digit').style.visibility,'hidden');
select('category','all');assert.equal(w.document.getElementById('digit').style.visibility,'visible');assert(w.document.getElementById('example-id').textContent.includes('TRUE DIGIT 1'));
w.hypothesisExplorer.stop();w.close();console.log('PASS:30 checkpoint controls ×16 masks, tables, chart, image categories and empty-filter behavior.');
