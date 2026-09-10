'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const data = window.HYPOTHESIS_EXTENDED;
  const original = window.HYPOTHESIS_DATA;
  if (!data || data.models.length !== 30) {
    $('extended-status').textContent = 'Extended inference data could not load. The original four-branch explorer remains available below.';
    $('layer-scope').value = 'body';
    $('layer-scope').disabled = $('inference-mode').disabled = true;
    return;
  }
  const names = ['Features (stem)', 'Branch 1', 'Branch 2', 'Branch 3', 'Branch 4', 'Classifier (head)'];
  const widths = ['784 → 2500', '2500 → 2000', '2000 → 1500', '1500 → 1000', '1000 → 500', '500 → 10'];
  const macs = ['1.96M', '5M', '3M', '1.5M', '0.5M', '0.005M'];
  const pct = v => v == null ? '—' : (100 * v).toFixed(2) + '%';
  const bits = m => Array.from({length: 6}, (_, i) => (m >> i) & 1).join('');
  const count = m => bits(m).split('1').length - 1;
  let model, fixedMask = 63, eligibleMask = 63, previewMask = 63, timer = null;
  const randomMode = () => $('inference-mode').value === 'random';
  const fullScope = () => $('layer-scope').value === 'full';
  const probability = () => Number($('drop-probability').value) / 100;
  const eligible = () => eligibleMask & (fullScope() ? 63 : 30);

  // Exact expectation of an independent Bernoulli mask for one prediction.
  // This never averages logits/probabilities across separately run networks.
  function weights(p, candidates) {
    return Array.from({length: 64}, (_, mask) => {
      let w = 1;
      for (let i = 0; i < 6; i++) {
        const on = !!(mask & (1 << i));
        w *= candidates & (1 << i) ? (on ? 1 - p : p) : (on ? 1 : 0);
      }
      return w;
    });
  }
  function aggregate(masks, ws, digit = null) {
    let coverage = 0, correct = 0, cost = 0, ce = 0;
    masks.forEach((m, i) => {
      const row = digit == null ? m : m.per_class[digit];
      coverage += ws[i] * row.coverage;
      correct += ws[i] * row.correct_output_rate;
      cost += ws[i] * m.cost;
      if (row.ce != null) ce += ws[i] * row.coverage * row.ce;
    });
    return {coverage, correct_output_rate: correct, cost,
      accuracy: coverage > 0 ? correct / coverage : null,
      ce: coverage > 0 ? ce / coverage : null};
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    $('full-play').textContent = 'Animate removing layers';
    $('full-play').setAttribute('aria-pressed', 'false');
  }
  function draw() {
    previewMask = 63;
    for (let i = 0; i < 6; i++) {
      if ((eligible() & (1 << i)) && Math.random() < probability()) previewMask &= ~(1 << i);
    }
  }
  function selectedMask() { return randomMode() ? previewMask : fixedMask; }
  function renderControls() {
    const random = randomMode();
    $('full-mask-controls').innerHTML = names.map((name, i) => {
      const bit = 1 << i, mandatory = !fullScope() && (i === 0 || i === 5);
      const checked = mandatory || (random ? !!(eligible() & bit) : !!(fixedMask & bit));
      return `<label><input type="checkbox" data-full-layer="${i}" ${checked ? 'checked' : ''} ${mandatory ? 'disabled' : ''}> ${mandatory ? 'Always keep' : random ? 'May drop' : 'Keep'} ${name}<small>${widths[i]} · ${macs[i]} MACs</small></label>`;
    }).join('');
    $('full-mask-controls').querySelectorAll('input').forEach(input => input.addEventListener('change', () => {
      stop();
      const bit = 1 << Number(input.dataset.fullLayer);
      if (random) { eligibleMask = input.checked ? eligibleMask | bit : eligibleMask & ~bit; draw(); }
      else fixedMask = input.checked ? fixedMask | bit : fixedMask & ~bit;
      render();
    }));
    $('full-help').textContent = random
      ? 'Checked = eligible for random dropout. Unchecked = always kept. The slider sets each eligible layer’s drop chance during inference; no weights are updated.'
      : 'Checked = keep this learned layer. Unchecked = replace it with the stated bypass. The same fixed selection is applied to every test image; no weights are updated.';
    $('random-controls').hidden = !random;
    $('full-draw').hidden = !random;
    $('full-head-only').hidden = random || !fullScope();
    $('full-play').hidden = random;
    $('full-keep').textContent = random ? 'Make none eligible' : 'Keep all 6';
    $('full-drop').textContent = random ? `Make all ${fullScope() ? 6 : 4} eligible` : 'Drop all 6';
    $('drop-probability-value').textContent = Math.round(probability() * 100) + '%';
    $('drop-probability').setAttribute('aria-valuetext', Math.round(probability() * 100) + '% chance per eligible layer');
  }
  function network(mask) {
    const head = !!(mask & 32), parts = [];
    parts.push(`<path d="M75 80 H925" fill="none" stroke="${head ? '#96b5ad' : '#d9dee0'}" stroke-width="3"/>`);
    names.forEach((name, i) => {
      const x = 75 + i * 170, present = !!(mask & (1 << i));
      const runs = head && present;
      const status = !head ? (i === 5 ? 'No prediction' : 'Not executed')
        : present ? 'Learned affine' : i === 0 ? 'Pad pixels with zeros' : 'Crop + ReLU';
      parts.push(`<g><title>${name}: ${status}</title><rect x="${x-72}" y="49" width="144" height="62" rx="6" fill="${runs ? '#087b73' : '#eff2f1'}" stroke="${runs ? '#087b73' : '#aebdb7'}" ${runs ? '' : 'stroke-dasharray="5 4"'}/><text x="${x}" y="76" text-anchor="middle" style="fill:${runs ? 'white' : '#536a63'}">${i === 0 ? 'Features' : i === 5 ? 'Classifier' : 'Branch ' + i}</text><text x="${x}" y="96" text-anchor="middle" style="font-size:12px;fill:${runs ? '#e7f3ed' : '#536a63'}">${present ? 'selected' : 'dropped'}</text><text x="${x}" y="141" text-anchor="middle" style="font-size:12px">${status}</text><text x="${x}" y="164" text-anchor="middle" style="font-size:12px">${widths[i]}</text></g>`);
    });
    $('full-network').innerHTML = parts.join('');
    $('full-network').setAttribute('aria-label', `Mask ${bits(mask)}. ${head ? model.masks[mask].executed_affines + ' learned affine layers execute.' : 'Classifier absent: abstain before all affine computation.'}`);
  }
  function drawDigit(index) {
    const encoded = original.images[String(index)], ctx = $('full-digit').getContext('2d');
    if (!encoded || !ctx) return;
    const tiny = document.createElement('canvas'); tiny.width = tiny.height = 28;
    const c = tiny.getContext('2d'), pixels = c.createImageData(28, 28), bytes = atob(encoded);
    for (let i = 0; i < 784; i++) {
      const v = bytes.charCodeAt(i);
      pixels.data.set([Math.round(20+v*.92), Math.round(44+v*.83), Math.round(52+v*.79), 255], i*4);
    }
    c.putImageData(pixels, 0, 0); ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tiny, 0, 0, 140, 140);
  }
  function example() {
    const row = model.examples.find(e => e.index === Number($('full-example-select').value));
    if (!row) return;
    const mask = selectedMask(), pred = row.pred[mask];
    drawDigit(row.index);
    $('full-digit').setAttribute('aria-label', `MNIST test image ${row.index}, true digit ${row.label}`);
    $('full-prediction').textContent = pred === -1 ? 'No prediction · abstains' : `Predicts ${pred} · ${pred === row.label ? 'correct' : 'incorrect'}`;
    $('full-prediction').style.color = pred === -1 ? '#705421' : pred === row.label ? '#08776f' : '#b54b34';
    $('full-example-note').textContent = `${randomMode() ? 'One random draw; aggregate scores above average all possible draws. ' : ''}Mask ${bits(mask)}. ${pred === -1 ? 'No digit scores exist, so cross-entropy is undefined.' : 'Cross-entropy: ' + row.ce[mask].toFixed(3) + ' nats.'}`;
  }
  function tables(ws) {
    $('full-class-table').innerHTML = '<caption>One true digit per row. Coverage = any prediction; accuracy = correct among predictions.</caption><thead><tr><th>Digit / n</th><th>Correct / all</th><th>Coverage</th><th>Accuracy</th><th>CE (nats)</th></tr></thead><tbody>' + Array.from({length: 10}, (_, digit) => {
      const row = aggregate(model.masks, ws, digit);
      return `<tr><th>${digit} / ${model.masks[63].per_class[digit].n}</th><td>${pct(row.correct_output_rate)}</td><td>${pct(row.coverage)}</td><td>${pct(row.accuracy)}</td><td>${row.ce == null ? '—' : row.ce.toFixed(3)}</td></tr>`;
    }).join('') + '</tbody>';
  }
  function fixedTable() {
    $('full-mask-table').innerHTML = '<caption>Bits read stem → four middle branches → classifier. 1 = selected. Without the classifier, selected affines are not executed. Accuracy and CE are undefined for abstention.</caption><thead><tr><th>Mask</th><th>Selected / run</th><th>MACs</th><th>Correct / all</th><th>Coverage</th><th>Accuracy</th><th>CE</th></tr></thead><tbody>' + model.masks.map(m => `<tr><td><button type="button" data-full-mask="${m.id}" aria-label="Inspect mask ${bits(m.id)}">${bits(m.id)}</button></td><td>${m.selected_affines} / ${m.executed_affines}</td><td>${pct(m.cost)}</td><td>${pct(m.correct_output_rate)}</td><td>${pct(m.coverage)}</td><td>${pct(m.accuracy)}</td><td>${m.ce == null ? '—' : m.ce.toFixed(3)}</td></tr>`).join('') + '</tbody>';
    $('full-mask-table').querySelectorAll('button').forEach(button => button.onclick = () => setMask(Number(button.dataset.fullMask)));
  }
  function chart() {
    const random = randomMode(), x = v => 78 + 850 * v, y = v => 282 - 238 * v, parts = [];
    for (const v of [0,.25,.5,.75,1]) parts.push(`<path d="M78 ${y(v)} H928" stroke="#e2e9e6"/><text x="66" y="${y(v)+5}" text-anchor="end">${v*100}%</text>`);
    if (random) {
      $('full-chart-title').textContent = 'What happens as inference dropout increases?';
      $('full-chart-subtitle').textContent = 'Exact expectation over independent mask draws';
      for (const v of [0,.25,.5,.75,1]) parts.push(`<text x="${x(v)}" y="309" text-anchor="middle">${v*100}%</text>`);
      const curve = Array.from({length: 101}, (_, i) => aggregate(model.masks, weights(i/100, eligible())));
      const series = [['coverage','#8899a1','6 4'], ['accuracy','#b76a15','3 4'], ['correct_output_rate','#087b73','']];
      for (const [key, color, dash] of series) {
        let path = '', active = false;
        curve.forEach((row, i) => {
          if (row[key] == null) { active = false; return; }
          path += `${active ? 'L' : 'M'}${x(i/100).toFixed(2)} ${y(row[key]).toFixed(2)} `;
          active = true;
        });
        parts.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="3" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`);
      }
      parts.push(`<path d="M${x(probability())} 38 V282" stroke="#324d60" stroke-dasharray="3 4"/><text x="500" y="338" text-anchor="middle">Drop chance per eligible layer during inference</text>`);
      $('full-chart-legend').innerHTML = '<span><i style="background:#087b73"></i>Correct outputs / all inputs</span><span><i style="background:#b76a15"></i>Accuracy among predictions (dotted)</span><span><i style="background:#8899a1"></i>Coverage (dashed)</span>';
      $('full-chart-note').textContent = eligible() & 32
        ? 'Dropping the classifier with probability p gives coverage 1−p. At 100%, correct outputs and coverage reach zero; conditional accuracy has no endpoint there. These are expected single-prediction outcomes, not an ensemble. The sample diagram can change while the curves stay fixed.'
        : 'The classifier is always kept, so coverage stays 100%. Correct-output rate and accuracy coincide. At 100% dropout, every eligible layer is skipped, but any ineligible layers remain.';
    } else {
      $('full-chart-title').textContent = 'From six learned layers to none';
      $('full-chart-subtitle').textContent = 'Click a dot to inspect that fixed mask';
      for (let k = 0; k <= 6; k++) parts.push(`<text x="${x(k/6)}" y="309" text-anchor="middle">${k}</text>`);
      // The 32 head-absent settings implement one identical abstention policy.
      for (const m of model.masks.filter(m => m.coverage === 1)) {
        const cx = x(m.executed_affines / 6), cy = y(m.correct_output_rate);
        parts.push(`<circle data-point-mask="${m.id}" tabindex="0" role="button" aria-label="Mask ${bits(m.id)}, ${pct(m.accuracy)} accuracy" cx="${cx}" cy="${cy}" r="${m.id === fixedMask ? 8 : 5}" fill="${m.id === fixedMask ? '#dc8a32' : '#087b73'}" stroke="white"><title>${bits(m.id)} · ${pct(m.accuracy)} · ${pct(m.cost)} MACs</title></circle>`);
      }
      parts.push(`<circle data-point-mask="0" tabindex="0" role="button" aria-label="Drop all layers: abstain, 0% correct outputs, no accuracy" cx="${x(0)}" cy="${y(0)}" r="${fixedMask & 32 ? 6 : 9}" fill="#b76a15"/><text x="${x(0)+13}" y="${y(0)-12}" style="font-size:12px">Abstain (32 masks)</text><text x="500" y="338" text-anchor="middle">Learned affine layers actually executed (out of 6)</text>`);
      $('full-chart-legend').innerHTML = '<span><i style="background:#087b73"></i>Classifier present: accuracy = correct-output rate</span><span><i style="background:#b76a15"></i>Classifier absent: no prediction</span>';
      $('full-chart-note').textContent = 'Vertical axis: correct outputs / all inputs. All 32 classifier-present networks were measured on 10,000 images. The other 32 settings all abstain and overlap at zero. Points with the same depth can perform very differently; layer count is not a guarantee of quality or speed.';
    }
    parts.unshift('<text x="78" y="24">Fraction of test images / predictions</text>');
    $('full-chart').innerHTML = parts.join('');
    $('full-chart').setAttribute('aria-label', random ? 'Expected correct-output rate, prediction coverage and conditional accuracy versus inference drop probability' : 'All six-layer masks: correct-output rate versus the number of affine layers actually executed');
    $('full-chart').querySelectorAll('[data-point-mask]').forEach(dot => {
      dot.onclick = () => setMask(Number(dot.dataset.pointMask));
      dot.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); dot.onclick(); } };
    });
  }
  function render() {
    const visible = fullScope() || randomMode();
    $('body-explorer').hidden = visible;
    $('full-explorer').hidden = !visible;
    $('extended-status').textContent = visible ? 'Extended audit: 30 frozen checkpoint states, 64 settings each. The original 16 masks reproduce exactly.' : 'Original audit: four optional middle branches. The trained feature layer and classifier always remain.';
    if (!visible) return;
    window.hypothesisExplorer.stop();
    renderControls();
    const random = randomMode(), mask = selectedMask(), m = model.masks[mask];
    const ws = random ? weights(probability(), eligible()) : Array.from({length: 64}, (_, i) => +(i === mask));
    const score = aggregate(model.masks, ws);
    $('full-correct').textContent = pct(score.correct_output_rate);
    $('full-coverage').textContent = pct(score.coverage);
    $('full-accuracy').textContent = pct(score.accuracy);
    $('full-cost').textContent = pct(score.cost);
    $('full-mask-status').textContent = `${random ? 'Preview draw' : 'Fixed mask'} ${bits(mask)} · ${m.selected_affines}/6 selected · ${m.executed_affines}/6 learned layers execute${random ? ' · scores above/below average every possible draw' : ''}`;
    $('full-explanation').textContent = !(mask & 32)
      ? `${random ? 'This preview draw' : 'This mask'} has no classifier: it produces no digit and short-circuits all affine work. ${random ? 'The four summary scores describe the whole random policy, including draws that do predict.' : 'Correct outputs = 0%; coverage = 0%; classification accuracy and cross-entropy are undefined.'}`
      : !(mask & 1)
        ? 'The learned feature layer is absent. Normalized pixels are padded with zeros to width 2,500, then pass through any kept branches and the trained classifier. This untrained replacement tests dependence on learned features.'
        : mask === 33
          ? `All four middle branches are absent, but trained features and the classifier remain. This shallow path still achieves ${pct(m.accuracy)} accuracy.`
          : 'The learned feature layer and classifier remain. A dropped middle branch uses crop + ReLU; kept branches use the saved weights with gain one.';
    network(mask); example(); tables(ws); chart();
  }
  function loadModel() {
    model = data.models.find(m => m.id === `${$('recipe').value}-${$('state').value}-s${$('seed').value}`);
    if (!model) throw new Error('Missing extended checkpoint state');
    const previous = Number($('full-example-select').value);
    $('full-example-select').innerHTML = model.examples.map(e => `<option value="${e.index}">True digit ${e.label} · test #${e.index}</option>`).join('');
    if (model.examples.some(e => e.index === previous)) $('full-example-select').value = String(previous);
    fixedTable(); render();
  }
  function setMask(mask, halt = true) {
    if (!Number.isInteger(mask) || mask < 0 || mask > 63) throw new Error('Mask must be 0–63');
    if (halt) stop();
    $('layer-scope').value = 'full'; $('inference-mode').value = 'fixed';
    fixedMask = mask; render();
  }
  ['recipe','state','seed'].forEach(id => $(id).addEventListener('change', () => { stop(); loadModel(); }));
  ['layer-scope','inference-mode'].forEach(id => $(id).addEventListener('change', () => { stop(); draw(); render(); }));
  $('drop-probability').addEventListener('input', () => { draw(); render(); });
  $('full-example-select').addEventListener('change', example);
  $('full-keep').onclick = () => { stop(); if (randomMode()) { eligibleMask = 0; draw(); render(); } else setMask(63); };
  $('full-drop').onclick = () => { stop(); if (randomMode()) { eligibleMask = fullScope() ? 63 : 30; draw(); render(); } else setMask(0); };
  $('full-head-only').onclick = () => setMask(32);
  $('full-draw').onclick = () => { draw(); render(); };
  $('full-play').onclick = () => {
    if (timer) { stop(); return; }
    // Remove body branches first, then stem, then classifier. One illustrative
    // order through measured masks, not a claim of monotonic accuracy decay.
    const path = [63, 47, 39, 35, 33, 32, 0];
    let step = 0; setMask(path[0]);
    $('full-play').textContent = 'Pause removal'; $('full-play').setAttribute('aria-pressed', 'true');
    timer = setInterval(() => {
      step++; setMask(path[step], false);
      if (step === path.length - 1) stop();
    }, 1500);
  };
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  loadModel();
  window.fullDepthExplorer = {setMask, weights, aggregate, getModel: () => model, stop,
    getPreviewMask: () => previewMask};
})();
