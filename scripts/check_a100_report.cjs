#!/usr/bin/env node
// Static provenance and DOM checks. Never starts a browser, training or a service.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require('jsdom');
const args = process.argv.slice(2);
const allowSynthetic = args.includes('--allow-synthetic');
assert(args.filter(a => !a.startsWith('--')).length <= 1, 'Expected at most one output directory');
assert(args.filter(a => a.startsWith('--')).every(a => a === '--allow-synthetic'), 'Unknown flag');
const repo = path.resolve(__dirname, '..');
const root = path.resolve(args.find(a => !a.startsWith('--')) || path.join(repo, 'docs/a100-transfer'));
const read = file => fs.readFileSync(path.join(root, file));
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const html = read('index.html').toString();
const data = JSON.parse(read('data.json'));
const manifest = JSON.parse(read('manifest.json'));
const synthetic = Boolean(data.synthetic_fixture);
assert.equal(synthetic, allowSynthetic, 'Synthetic data requires explicit --allow-synthetic; measured data must omit it');
assert(!synthetic || !root.startsWith(path.join(repo, 'docs') + path.sep), 'Synthetic fixture under publish directory');
assert.equal(manifest.synthetic_fixture, synthetic);
assert.equal(manifest.summary_sha256, hash(read('data.json')));
const canonical = value => Array.isArray(value)?'['+value.map(canonical).join(',')+']':
  value!==null&&typeof value==='object'?'{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+canonical(value[k])).join(',')+'}':JSON.stringify(value);
const scientificPayload = {
  raw_files:data.verification.raw_files.map(r=>({run_id:r.run_id,sha256:r.sha256})).sort((a,b)=>a.run_id<b.run_id?-1:a.run_id>b.run_id?1:0),
  evaluation_manifest_sha256:data.verification.evaluation_manifest_sha256,
  executed_core_source_sha256:data.verification.executed_core_source_sha256,
  tuning_manifest:data.verification.tuning_manifest,
};
assert.equal(manifest.scientific_input_sha256,hash(Buffer.from(canonical(scientificPayload))));
assert.equal(manifest.complete_final_runs, 27);
assert.deepEqual(manifest.initialization_numerical_audit,data.verification.initialization_numerical_audit);
assert.deepEqual(manifest.vit_initialization_numerical_audit,data.verification.vit_initialization_numerical_audit??null);
assert.equal(data.verification.complete_final_runs, 27);
assert.equal(data.verification.passed, true);
assert.equal(data.curves.length, 81);
assert.equal(data.paired_comparisons.length, 216);
assert.equal(data.primary_comparisons.length, 6);
assert.equal(data.verification.raw_files.length, 27);
assert.equal(Object.keys(manifest.figures).length, 26);
assert(!html.includes('@@'), 'Unfilled template');
assert.equal(html.includes('SYNTHETIC TEST FIXTURE'), synthetic);
const approximately = (a,b, label) => assert(Math.abs(a-b) <= 1e-8 * Math.max(1,Math.abs(a),Math.abs(b)), `${label}: ${a} != ${b}`);
for (const row of [...data.curves.flatMap(r => ['ce','excess_ce','accuracy','accuracy_change'].map(k => r[k])), ...data.paired_comparisons]) {
  assert.equal(row.n, 3); assert.equal(row.df, 2); assert.equal(row.values.length, 3);
  assert(row.values.every(Number.isFinite));
  const mean = row.values.reduce((a,b) => a+b,0)/3;
  const sd = Math.sqrt(row.values.reduce((s,v) => s+(v-mean)**2,0)/2);
  const radius = 4.3026527299*sd/Math.sqrt(3);
  approximately(row.mean, mean, 'Mean');
  approximately(row.std, sd, 'Standard deviation');
  approximately(row.ci95_low, mean-radius, 'Lower CI');
  approximately(row.ci95_high, mean+radius, 'Upper CI');
}
for (const [file, record] of Object.entries(manifest.figures)) {
  assert.equal(hash(read(file)), record.sha256, `Figure digest ${file}`);
  assert.equal(record.backing_data, 'data.json');
  if (file.endsWith('.png')) {
    assert.equal(read(file).subarray(1,4).toString(), 'PNG');
    assert(read(file).readUInt32BE(16) >= 1800, 'Scientific PNG resolution');
  } else assert(read(file).toString().includes('<svg'), 'SVG export');
}

const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', error => errors.push(error.message));
virtualConsole.on('error', error => errors.push(String(error)));
const dom = new JSDOM(html, {url:'https://yaroslavvb.github.io/gradient-dissent/a100-transfer/', runScripts:'outside-only', virtualConsole});
const w = dom.window, d = w.document;
const get = id => {const value=d.getElementById(id); assert(value, `Missing ID ${id}`); return value;};
const ids = [...d.querySelectorAll('[id]')].map(e => e.id);
assert.equal(ids.length, new Set(ids).size, 'Duplicate IDs');
function checkLinks() {
  for (const element of d.querySelectorAll('[href], [src]')) {
    const value = element.getAttribute('href') || element.getAttribute('src');
    if (/^(https?:|mailto:|data:)/.test(value)) continue;
    const [local, anchor] = value.split('#');
    let target = path.resolve(root, local || 'index.html');
    // The fixture is deliberately outside docs; its two parent navigation links
    // still refer to existing real site pages, never to experimental fixture data.
    if (synthetic && local.startsWith('../')) target = path.resolve(repo, 'docs/a100-transfer', local);
    assert(fs.existsSync(target), `Missing local link ${value}`);
    if (fs.statSync(target).isDirectory()) target = path.join(target, 'index.html');
    assert(fs.existsSync(target), `Missing local index ${value}`);
    if (anchor) {
      const targetDocument = local ? new JSDOM(fs.readFileSync(target, 'utf8')).window.document : d;
      assert(targetDocument.getElementById(anchor), `Missing anchor ${value}`);
    }
  }
}
const timers = new Map(); let nextTimer=0;
w.setInterval = callback => {const id=++nextTimer; timers.set(id, callback); return id;};
w.clearInterval = id => timers.delete(id);
const change = (id,value) => {get(id).value=value; get(id).dispatchEvent(new w.Event('change'));};
const recipes = ['dense','constant_ild','decreasing_ild'];
const names = {dense:'Dense',constant_ild:'Constant ILD',decreasing_ild:'Decreasing ILD'};
const signed = (x,n=4) => (x>=0?'+':'')+Number(x).toFixed(n);
const interval = (row,factor=1) => `${signed(row.mean*factor)} [${signed(row.ci95_low*factor)}, ${signed(row.ci95_high*factor)}]`;
const curve = (task,recipe,mask) => data.curves.find(r => r.task===task && r.recipe===recipe && r.mask_name===mask);
// The downloadable report must carry the absolute quantities, exposure and cost
// itself, rather than depending on the interactive page for its interpretation.
const markdown = read('report.md').toString();
assert.equal(markdown.includes('SYNTHETIC RENDERER FIXTURE'), synthetic);
if(manifest.scientific_findings) {
  assert(!synthetic,'Synthetic renderer fixtures must remain independent of real findings');
  const findingBytes=read(manifest.scientific_findings.file),findings=JSON.parse(findingBytes);
  assert.equal(hash(findingBytes),manifest.scientific_findings.sha256);
  assert.equal(findings.scientific_input_sha256,manifest.scientific_input_sha256);
  const paragraphs=[...get('scientific-interpretation').querySelectorAll('p')].slice(0,-1).map(p=>p.textContent);
  assert.deepEqual(paragraphs,findings.paragraphs);
  for(const paragraph of findings.paragraphs) assert(markdown.includes(paragraph),'Markdown interpretation differs');
  assert(markdown.includes(findings.scientific_input_sha256));
} else {
  assert(!d.getElementById('scientific-interpretation'),'Unprovenanced interpretation present');
  assert(!fs.existsSync(path.join(root,'findings.json')),'Stale interpretation artifact retained');
}
const table = heading => {
  const section = markdown.split(`## ${heading}\n`)[1];
  assert(section, `Missing standalone Markdown section: ${heading}`);
  return section.split('\n## ')[0].split('\n').filter(line => line.startsWith('| ')).slice(1)
    .map(line => line.split('|').slice(1,-1).map(cell => cell.trim()));
};
const qualityRows = table('Absolute predictive quality');
const exposureRows = table('Model size and training exposure');
const tuningRows = table('Recipe selection');
const candidateRows = table('Candidate validation measurements');
const initializations = [data.verification.initialization_numerical_audit,data.verification.vit_initialization_numerical_audit].filter(Boolean);
const initializationText=get('initialization-audit').textContent;
for (const text of [initializationText,markdown]) {
  assert(text.includes('GPT retains byte-identical same-seed initial states'));
  assert(!text.includes('GPT and ViT retain byte-identical'));
  assert(text.includes('same-seed and numerically close, not byte-identical'));
  assert(text.includes('posthoc verification adjustment made independently of test outcomes'));
  assert(text.includes('does not change the frozen executed training code'));
  assert(text.includes('first observed difference at CPU erfinv after identical uniform draws'));
  assert(text.includes('do not establish the exact dispatch path'));
}
for(const initialization of initializations) {
  assert.equal(initialization.passed,true);
  assert.equal(initialization.posthoc_verification_adjustment,true);
  const name=initialization.task==='convnext_cifar100'?'ConvNeXt':'ViT';
  const text=[...get('initialization-audit').querySelectorAll('p')].find(p=>p.textContent.startsWith(name+':')).textContent;
  assert(markdown.includes(text));
  assert(text.includes(initialization.seed_audits[0].parameters.toLocaleString('en-US')));
  const absolute=Number(text.match(/worst absolute difference was ([\d.]+(?:e[+-]?\d+)?)/i)[1]);
  const relative=Number(text.match(/worst relative L2 difference was ([\d.]+(?:e[+-]?\d+)?)/i)[1]);
  assert(Math.abs(absolute-initialization.max_absolute_difference)<=initialization.max_absolute_difference*.005);
  assert(Math.abs(relative-initialization.max_relative_l2_difference)<=initialization.max_relative_l2_difference*.005);
  for (const file of initialization.evidence_files) {
    assert(markdown.includes(`/results/${path.basename(file.file)}`));
    assert([...get('initialization-audit').querySelectorAll('a')].some(a => a.href.endsWith(`/results/${path.basename(file.file)}`)));
  }
}
assert.equal(qualityRows.length, 9);
assert.equal(exposureRows.length, 3);
assert.equal(tuningRows.length, 9);
assert.equal(candidateRows.length, 27);
assert.equal(get('tuning-validation').querySelectorAll('tbody tr').length,27);
const lrMetric = (row,key,lr) => Object.entries(row[key]).find(([rate]) => Number(rate)===lr)[1];
const familyLabels = {vit_cifar100:'ViT · CIFAR-100',convnext_cifar100:'ConvNeXt · CIFAR-100',gpt_wikitext103:'GPT · WikiText-103'};
const absolute = (row,factor=1,n=4) => `${(row.mean*factor).toFixed(n)} [${(row.ci95_low*factor).toFixed(n)}, ${(row.ci95_high*factor).toFixed(n)}]`;
for (const family of data.families) {
  const exposure = exposureRows.find(row => row[0]===familyLabels[family.task]);
  assert(exposure, 'Missing Markdown family exposure');
  assert.equal(Number(exposure[1].replaceAll(',','')),family.parameters);
  assert.equal(Number(exposure[2]),family.prunable_count);
  assert.equal(exposure[3],`${family.steps.toLocaleString('en-US')} × ${family.batch_size.toLocaleString('en-US')}`);
  assert(exposure[5].startsWith(family.training_targets_or_images.toLocaleString('en-US')+' '));
  assert(exposure[6].startsWith(family.task==='gpt_wikitext103'?family.tokens_per_parameter.toFixed(3):family.image_epochs.toFixed(2)));
  if (family.task==='gpt_wikitext103') {
    assert.equal(family.test_panel.target_tokens,family.test_panel.windows*family.context);
    assert.equal(exposure[7],`${family.test_panel.target_tokens.toLocaleString('en-US')} target tokens in ${family.test_panel.windows.toLocaleString('en-US')} actual windows`);
  } else assert(exposure[6].endsWith('equivalent passes with replacement'));
  for (const recipe of recipes) {
    const row = qualityRows.find(row => row[0]===familyLabels[family.task] && row[1]===names[recipe]);
    assert(row, 'Missing Markdown quality cell');
    const full=curve(family.task,recipe,'full'),pruned=curve(family.task,recipe,'primary_two_thirds');
    assert.deepEqual(row.slice(2),[absolute(full.ce),absolute(pruned.ce),absolute(full.accuracy,100,2),absolute(pruned.accuracy,100,2)]);
    const tuning = data.tuning.find(r => r.task===family.task && r.recipe===recipe);
    const selection = tuningRows.find(row => row[0]===familyLabels[family.task] && row[1]===names[recipe]);
    assert(selection, 'Missing Markdown tuning cell');
    assert.deepEqual(selection[2].split(', ').map(Number),tuning.learning_rates);
    assert.equal(Number(selection[3]),tuning.selected_lr);
    assert.equal(selection[4],tuning.selected_at_boundary?'yes':'no');
    assert.equal(selection[5],tuning.tuning_seeds.join(', '));
    const ceWinner=[...tuning.learning_rates].sort((a,b) => lrMetric(tuning,'mean_validation_ce',a)-lrMetric(tuning,'mean_validation_ce',b)||a-b)[0];
    assert.equal(tuning.selected_lr,ceWinner,'Candidate selected by validation CE');
    for (const lr of tuning.learning_rates) {
      const candidate=candidateRows.find(row => row[0]===familyLabels[family.task] && row[1]===names[recipe] && Number(row[2])===lr);
      const ce=lrMetric(tuning,'mean_validation_ce',lr),accuracy=lrMetric(tuning,'mean_validation_accuracy',lr)*100;
      assert(candidate,'Missing Markdown candidate score');
      assert.equal(candidate[3],ce.toFixed(4));
      assert.equal(candidate[4],accuracy.toFixed(2));
      assert.equal(candidate[5],lr===tuning.selected_lr?'yes — lowest CE':'no');
      const htmlCandidate=[...get('tuning-validation').querySelectorAll('tbody tr')].find(row => row.dataset.task===family.task && row.dataset.recipe===recipe && Number(row.dataset.lr)===lr);
      assert(htmlCandidate,'Missing HTML candidate score');
      assert.equal(htmlCandidate.children[3].textContent,ce.toFixed(4));
      assert.equal(htmlCandidate.children[4].textContent,accuracy.toFixed(2)+'%');
      assert.equal(htmlCandidate.children[5].textContent,candidate[5]);
    }
  }
}
assert(markdown.includes(data.generated_utc), 'Missing summary timestamp');
for (const text of [d.body.textContent,markdown]) {
  assert(!text.includes('equivalent epochs')&&!text.includes('Equivalent epochs'),'Unqualified epoch terminology');
  assert(text.includes('20% expected masking rate'),'Expected masking rate must be explicit');
  assert(text.includes('not saved execution'),'Masking must not imply skipped execution');
  assert(text.includes('a cap'),'Requested evaluation maximum must be distinguished from actual windows');
}
assert(markdown.includes(`**$${data.budget.reserved_upper_usd.toFixed(2)}**`), 'Missing reservation total');
assert(markdown.includes(`**${data.budget.reservation_count} invocation reservations**`));
assert(markdown.includes(`**$${data.budget.cap_usd.toFixed(2)}**`), 'Missing authorized cap');
if (data.budget.latest_metered_usage) {
  const billing=data.budget.latest_metered_usage;
  const queried=new Date(billing.queried_unix*1000).toISOString().slice(0,16).replace('T',' ')+' UTC';
  assert(markdown.includes(`**$${billing.metered_cost_usd.toFixed(2)}**`), 'Missing metered amount');
  assert(markdown.includes(queried), 'Missing metered-query timestamp');
} else assert(markdown.includes('No metered-usage snapshot was available'));
let curveStates=0, maskStates=0, checkedNumbers=0;
try {
  w.eval(read('data.js').toString());
  w.eval(read('app.js').toString());
  for (const family of data.families) {
    change('task', family.task);
    assert(get('task-context').textContent.includes(family.parameters.toLocaleString('en-US')));
    if(family.task==='gpt_wikitext103') {
      assert(get('task-context').textContent.includes(family.test_panel.windows.toLocaleString('en-US')+' actual test windows'));
      assert(get('task-context').textContent.includes(family.test_panel.target_tokens.toLocaleString('en-US')+' target tokens'));
    }
    assert.equal(get('primary').querySelectorAll('[data-primary-point]').length, 2);
    assert.equal(get('primary').querySelectorAll('[data-seed]').length, 6);
    const primaryRows = data.primary_comparisons.filter(r => r.task === family.task);
    const statements = [...get('primary-narrative').querySelectorAll('p')];
    primaryRows.forEach(row => {
      const statement=statements.find(p => p.textContent.startsWith(names[row.recipe]+':'));
      assert(statement.textContent.includes(interval(row)));
      assert.equal(statement.dataset.direction, row.ci95_high<0?'less':row.ci95_low>0?'more':'unresolved');
      if(curve(family.task,'dense','primary_two_thirds').excess_ce.mean<0 && curve(family.task,row.recipe,'primary_two_thirds').excess_ce.mean<0) {
        assert(statement.textContent.includes('Pruning lowers CE in both recipes'));
        assert(statement.textContent.includes(row.mean>0?'smaller CE improvement after dropout':'larger CE improvement after dropout'));
      }
    });
    const quality = [...get('quality-table').querySelectorAll('tbody tr')];
    assert.equal(quality.length, 3);
    recipes.forEach((recipe,i) => {
      assert.equal(quality[i].children[1].textContent, curve(family.task,recipe,'full').ce.mean.toFixed(4));
      assert.equal(quality[i].children[2].textContent, curve(family.task,recipe,'primary_two_thirds').ce.mean.toFixed(4));
      assert.equal(quality[i].children[3].textContent, (curve(family.task,recipe,'full').accuracy.mean*100).toFixed(2)+'%');
      assert.equal(quality[i].children[4].textContent, (curve(family.task,recipe,'primary_two_thirds').accuracy.mean*100).toFixed(2)+'%');
    });
    const order = [...w.A100_REPORT.maskOrder()];
    assert.equal(order.length, 9);
    assert.equal(new Set(order).size, 9);
    assert.equal(order[2], 'primary_two_thirds');
    for (const metric of ['ce','excess_ce','accuracy','accuracy_change']) {
      for (const visibleBits of [7,1,2,4,3,5,6]) {
        recipes.forEach((recipe,i) => {get('recipe-'+recipe).checked=Boolean(visibleBits & (1<<i));});
        change('metric', metric);
        const active=recipes.filter((_,i) => visibleBits & (1<<i));
        const factor=metric.startsWith('accuracy')?100:1;
        const points=[...get('curve').querySelectorAll('[data-point]')];
        const numericRows=[...get('curve-table').querySelectorAll('tbody tr')];
        assert.equal(points.length, 9*active.length);
        assert.equal(numericRows.length, points.length);
        assert(!/NaN|Infinity|undefined/.test(get('curve').outerHTML));
        let index=0;
        for (const mask of order) for (const recipe of active) {
          const source=curve(family.task,recipe,mask)[metric];
          const row=numericRows[index++];
          assert.equal(row.children[1].textContent, names[recipe]);
          assert.equal(row.children[3].textContent, interval(source,factor));
          assert.equal(row.children[4].textContent, source.values.map(v => (v*factor).toFixed(4)).join(', '));
          const point=points.find(p => p.dataset.point===`${recipe}/${mask}`);
          assert(point.getAttribute('aria-label').includes(interval(source,factor)));
          checkedNumbers++;
        }
        for (const extension of ['png','svg']) assert.equal(get('download-'+extension).getAttribute('href'), `${family.task}-${metric}.${extension}`);
        checkLinks();
        curveStates++;
      }
    }
    // A zero-recipe selection is corrected without producing an empty range.
    recipes.forEach(recipe => {get('recipe-'+recipe).checked=false;});
    get('recipe-dense').dispatchEvent(new w.Event('change'));
    assert.equal(get('recipe-dense').checked,true);
    assert.equal(get('curve').querySelectorAll('[data-point]').length,9);
    for (const mask of order) {
      change('mask',mask);
      const blocks=[...get('mask-viz').querySelectorAll('[data-block]')];
      assert.equal(blocks.length,family.prunable_count);
      assert.deepEqual(blocks.filter(b => b.dataset.kept==='true').map(b => Number(b.dataset.block)), family.mask_panel[mask]);
      assert.equal(get('mask-table').querySelectorAll('tbody tr').length,3);
      if(mask==='delete_first_only') assert(get('mask-description').textContent.includes('outside'));
      assert(!/NaN|Infinity|undefined/.test(get('mask-viz').outerHTML));
      maskStates++;
    }
    const point=get('curve').querySelector('[data-point="dense/primary_two_thirds"]');
    point.dispatchEvent(new w.MouseEvent('click'));
    assert.equal(get('mask').value,'primary_two_thirds');
    change('mask','full');
    point.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
    assert.equal(get('mask').value,'primary_two_thirds');
    const initialMask=get('mask').value;
    get('play-masks').click();
    assert.equal(get('play-masks').getAttribute('aria-pressed'),'true');
    assert.equal(timers.size,1);
    [...timers.values()][0]();
    assert.notEqual(get('mask').value,initialMask);
    get('play-masks').click();
    assert.equal(timers.size,0);
    get('play-masks').click();
    w.dispatchEvent(new w.Event('pagehide'));
    assert.equal(timers.size,0);
    assert.equal(get('play-masks').getAttribute('aria-pressed'),'false');
  }
  assert.equal(errors.length,0,errors.join('\n'));
  checkLinks();
  console.log(`PASS${synthetic?' (SYNTHETIC RENDERER FIXTURE ONLY)':''}: summary and 26 figure hashes, 3-seed interval arithmetic, standalone Markdown quality/exposure/tuning/cost, ${curveStates} curve states, ${maskStates} mask states, ${checkedNumbers} point/table values, keyboard selection, explicit animation controls, local links.`);
} finally {
  if (w.A100_REPORT) w.A100_REPORT.stop();
  dom.window.close();
}
