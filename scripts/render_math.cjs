const fs=require('node:fs');
const path=require('node:path');
const katex=require('katex');
const {JSDOM}=require('jsdom');
const file=path.join(__dirname,'../docs/review.html');
const dom=new JSDOM(fs.readFileSync(file,'utf8'));
const equations=dom.window.document.querySelectorAll('[data-tex]');
for(const e of equations){e.innerHTML=katex.renderToString(e.getAttribute('data-tex'),{displayMode:e.classList.contains('tex-display'),throwOnError:true,output:'htmlAndMathml'});e.removeAttribute('data-tex');}
fs.writeFileSync(file,dom.serialize());
const vendor=path.join(__dirname,'../docs/vendor/katex');fs.mkdirSync(vendor,{recursive:true});
const dist=path.join(path.dirname(require.resolve('katex')));
fs.copyFileSync(path.join(dist,'katex.min.css'),path.join(vendor,'katex.min.css'));
fs.cpSync(path.join(dist,'fonts'),path.join(vendor,'fonts'),{recursive:true});
fs.copyFileSync(path.join(dist,'../LICENSE'),path.join(vendor,'LICENSE'));
console.log(`Rendered ${equations.length} equations to accessible HTML/MathML; vendored CSS and fonts.`);
