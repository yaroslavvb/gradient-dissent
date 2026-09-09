(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const M = window.DropoutMath;
  const state = {mode: 'train', maximum: .8, progress: 0, retained: 8, selected: [0, 11], draws: 1};
  const random = M.seededRandom(260905275);
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let masks, playing = null, frame = null, cycleStart = null, phase = 0;
  const origin = 145, width = 68, rowY = s => 143 + s * 80;
  const percent = n => `${(100 * n).toFixed(1).replace(/\.0$/, '')}%`;
  const fixed = n => n.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
  const names = ['A', 'B', 'C', 'D'];
  const modeNames = {train: 'TRAINING', dense: 'FULL-DEPTH INFERENCE', exit: 'EARLY-EXIT INFERENCE'};

  function sample() { masks = M.batch(state, random); }

  function renderNetwork() {
    let svg = '<title id="network-title">Per-sequence paths through twelve shared transformer blocks</title><desc id="network-desc">Each row is one sequence; columns share model weights. Filled cells execute attention and FFN together. Dashed cells are identity bypasses. Use arrow keys to move the selected block.</desc>';
    svg += `<text x="10" y="29" fill="#64736b" font-size="10">${state.mode === 'train' ? 'DROP PROBABILITY' : 'DROPOUT OFF'}</text>`;
    svg += '<text x="10" y="64" fill="#64736b" font-size="10">SHARED BLOCK</text>';
    for (let l = 0; l < M.LAYERS; l++) {
      const x = origin + l * width, p = masks[0][l].p;
      svg += `<text x="${x + 30}" y="29" text-anchor="middle" font-size="10" fill="#64736b">${percent(p)}</text><rect x="${x + 5}" y="37" width="50" height="4" rx="2" fill="#eff1e9"/><rect x="${x + 5}" y="37" width="${50 * p}" height="4" rx="2" fill="#bb9164"/><text x="${x + 30}" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="#263f32">${l + 1}</text><path d="M${x + 30} 76 V414" stroke="#edf0e8" stroke-dasharray="2 5"/>`;
    }
    svg += '<text x="1007" y="64" text-anchor="middle" font-size="10" fill="#64736b">HEAD</text>';
    for (let s = 0; s < M.SEQUENCES; s++) {
      const y = rowY(s), count = masks[s].filter(c => c.keep).length;
      svg += `<text x="10" y="${y - 10}" font-size="12" font-weight="700" fill="#263f32">Sequence ${names[s]}</text><text x="10" y="${y + 9}" font-size="10" fill="#64736b">${count} / 12 active</text><path d="M112 ${y} H994" stroke="#b6c7b8" stroke-width="1.5"/><circle cx="113" cy="${y}" r="3" fill="#147a64"/><circle cx="122" cy="${y}" r="3" fill="#147a64"/><circle cx="131" cy="${y}" r="3" fill="#147a64"/>`;
      for (let l = 0; l < M.LAYERS; l++) {
        const x = origin + l * width, c = masks[s][l], selected = state.selected[0] === s && state.selected[1] === l;
        const route = c.keep ? 'M0 0 H6 V-14 H27 V0 H34 V-14 H55 V0 H64' : 'M0 0 H64';
        svg += `<g class="cell${selected ? ' selected' : ''}" data-sequence="${s}" data-layer="${l}" role="button" tabindex="${selected ? 0 : -1}" aria-pressed="${selected}" aria-label="Sequence ${names[s]}, block ${l + 1}: ${c.keep ? 'kept' : 'bypassed'}, drop probability ${percent(c.p)}, branch multiplier ${fixed(c.scale)}" transform="translate(${x},${y})"><rect class="cell-frame" x="0" y="-29" width="64" height="53" rx="7" fill="${c.keep ? '#e7f3e9' : '#fcf8ef'}" stroke="${c.keep ? '#9bc5a9' : '#c7ac86'}" ${c.keep ? '' : 'stroke-dasharray="3 3"'}/><path d="${route}" fill="none" stroke="${c.keep ? '#147a64' : '#b4956e'}" stroke-width="1.5"/><text x="32" y="16" text-anchor="middle" font-size="8" font-weight="600" fill="${c.keep ? '#147a64' : '#896b43'}">${c.keep ? 'KEEP' : 'BYPASS'}</text>${c.keep ? '<rect x="9" y="-22" width="15" height="15" rx="3" fill="#147a64"/><text x="16.5" y="-11" text-anchor="middle" fill="white" font-size="9">A</text><rect x="37" y="-22" width="15" height="15" rx="3" fill="#147a64"/><text x="44.5" y="-11" text-anchor="middle" fill="white" font-size="9">F</text>' : ''}</g>`;
      }
      svg += `<rect x="989" y="${y - 14}" width="34" height="28" rx="5" fill="#eff2e9" stroke="#c5d2bf"/><text x="1006" y="${y + 4}" text-anchor="middle" font-size="10" fill="#53634d">out</text><g class="token" id="tokens-${s}" aria-hidden="true"><circle cx="-8" r="3" fill="#253e32" stroke="white" stroke-width="1.2"/><circle cx="0" r="3" fill="#253e32" stroke="white" stroke-width="1.2"/><circle cx="8" r="3" fill="#253e32" stroke="white" stroke-width="1.2"/></g>`;
    }
    $('network').innerHTML = svg;
    moveTokens(phase);
  }

  function renderInspector() {
    const [s, l] = state.selected, c = masks[s][l], training = state.mode === 'train';
    $('inspector-title').textContent = `Sequence ${names[s]} · Block ${l + 1}`;
    $('selection-state').textContent = c.keep ? 'EXECUTE BOTH' : 'IDENTITY BYPASS';
    $('selection-state').style.color = c.keep ? '#147a64' : '#896b43';
    $('selection-state').style.background = c.keep ? '#e4f2e9' : '#faf2e4';
    $('selected-p').textContent = training ? percent(c.p) : 'Off';
    $('selected-survival').textContent = training ? percent(c.survival) : '—';
    $('selected-mask').textContent = training ? (c.keep ? '1' : '0') : 'Fixed';
    $('selected-scale').textContent = `${fixed(c.scale)}×`;
    const color = c.keep ? '#147a64' : '#b8bdb3', fill = c.keep ? '#e4f2e9' : '#f3f4ef';
    const label = training ? `One shared mask: M = ${c.keep ? 1 : 0}` : 'Fixed inference path · no random mask';
    $('block-detail').innerHTML = `<title>${c.keep ? 'Both branches execute' : 'Both branches bypassed'} for sequence ${names[s]}, block ${l + 1}</title><path d="M143 35 V23 H463 V35" fill="none" stroke="${color}" stroke-dasharray="3 3"/><rect x="204" y="11" width="206" height="21" fill="white"/><text x="307" y="26" text-anchor="middle" font-size="11" fill="#64736b">${label}</text><path d="M27 118 H585" stroke="#263e31" stroke-width="2" fill="none"/><path d="M64 118 V72 H263 V118 M335 118 V72 H535 V118" stroke="${color}" stroke-width="2" fill="none" ${c.keep ? '' : 'stroke-dasharray="5 4"'}/><rect x="87" y="48" width="143" height="48" rx="7" fill="${fill}" stroke="${color}"/><rect x="359" y="48" width="143" height="48" rx="7" fill="${fill}" stroke="${color}"/><text x="158" y="67" text-anchor="middle" font-size="12" fill="${color}">Attention(h)</text><text x="158" y="84" text-anchor="middle" font-size="10" fill="${color}">× ${fixed(c.scale)}</text><text x="430" y="67" text-anchor="middle" font-size="12" fill="${color}">FFN(z)</text><text x="430" y="84" text-anchor="middle" font-size="10" fill="${color}">× ${fixed(c.scale)}</text><circle cx="263" cy="118" r="12" fill="white" stroke="#263e31"/><circle cx="535" cy="118" r="12" fill="white" stroke="#263e31"/><text x="263" y="122" text-anchor="middle" fill="#263e31" font-size="15">+</text><text x="535" y="122" text-anchor="middle" fill="#263e31" font-size="15">+</text><text x="28" y="141" fill="#263e31" font-size="12">h</text><text x="302" y="141" fill="#263e31" font-size="12">z</text><text x="580" y="141" fill="#263e31" font-size="12">h′</text><text x="158" y="158" text-anchor="middle" font-size="10" fill="#64736b">identity stream · never scaled</text><text x="430" y="158" text-anchor="middle" font-size="10" fill="#64736b">identity stream · never scaled</text>`;
    $('factor-a').textContent = $('factor-f').textContent = training ? 'M / ρ' : c.keep ? '1' : '0';
    $('block-description').textContent = c.keep
      ? training ? `Mask 1: keep attention and FFN. Each residual update gets a ${fixed(c.scale)}× dropout multiplier. The FFN receives z, after the attention update.` : 'This retained inference block executes both branches with multiplier 1. The FFN receives the post-attention activation z.'
      : training ? 'Mask 0: bypass attention and FFN together. The activation is carried forward unchanged: h′ = h. It is not replaced by zero.' : 'This block is beyond the chosen exit depth. Its two residual updates are omitted; the retained prefix feeds the output head.';
  }

  function renderChart() {
    const x = t => 48 + t * 378, y = d => 158 - (d - 6) / 6 * 120;
    const initial = M.expectedDepth(0, state.maximum), depth = M.expectedDepth(state.progress, state.maximum);
    let chart = '<text x="48" y="15" font-size="10" fill="#64736b">EXPECTED ACTIVE BLOCKS / SEQUENCE</text>';
    for (const d of [6, 9, 12]) chart += `<path d="M48 ${y(d)} H426" stroke="#e0e7dc"/><text x="32" y="${y(d) + 4}" text-anchor="end" font-size="10" fill="#64736b">${d}</text>`;
    chart += `<path d="M48 ${y(initial)} L426 ${y(12)} V158 H48 Z" fill="#edf5ec"/><path d="M48 ${y(initial)} L426 ${y(12)}" fill="none" stroke="#147a64" stroke-width="2.5"/><path d="M${x(state.progress)} 30 V166" stroke="#77967d" stroke-dasharray="4 4"/><circle cx="${x(state.progress)}" cy="${y(depth)}" r="5" fill="#147a64" stroke="white" stroke-width="2"/><text x="${Math.max(67, Math.min(405, x(state.progress)))}" y="${y(depth) - 12}" text-anchor="middle" font-size="12" font-weight="700" fill="#147a64">${fixed(depth)}</text><text x="48" y="183" font-size="10" fill="#64736b">0%</text><text x="237" y="183" text-anchor="middle" font-size="10" fill="#64736b">50%</text><text x="426" y="183" text-anchor="end" font-size="10" fill="#64736b">100%</text><text x="237" y="202" text-anchor="middle" font-size="10" fill="#64736b">TRAINING PROGRESS</text>`;
    $('schedule-chart').innerHTML = chart;
    $('schedule-takeaway').textContent = `At ${percent(state.maximum)} maximum dropout, expected depth grows from ${fixed(initial)} to 12 blocks. Averaged across all blocks and training steps, ${percent(state.maximum / 4)} of sequence–block masks are off.`;
  }

  function renderCounters() {
    const training = state.mode === 'train', active = masks.flat().filter(c => c.keep).length;
    $('actual').textContent = `${active} / 48`;
    $('actual-note').textContent = training ? `Sampled batch ${state.draws} · can fluctuate` : 'Fixed for every sequence';
    $('expected-label').textContent = training ? 'Expected active depth / sequence' : 'Active depth / sequence';
    $('expected').textContent = `${fixed(training ? M.expectedDepth(state.progress, state.maximum) : active / 4)} / 12`;
    $('expected-note').textContent = training ? 'An expectation, not a fixed count' : 'A fixed inference path';
    $('average').textContent = training ? percent(state.maximum / 4) : '1×';
    $('average-label').textContent = training ? 'Average masks omitted over training' : 'Retained branch multiplier';
    $('average-note').textContent = training ? `${percent(state.maximum)} ÷ 2 across depth ÷ 2 across time` : 'No inverse-survival amplification';
    $('network-caption').textContent = training ? 'Each row gets its own mask. All tokens within that sequence share it. Columns use the same model weights. A = attention; F = FFN.' : state.mode === 'dense' ? 'Every sequence uses all twelve blocks. The learned weights are unchanged; random dropping and inverse-survival amplification are switched off.' : 'Every sequence uses the same selected prefix, in the original order, then the normal output head. Dashed cells show omitted blocks, not random dropout.';
  }

  function render() {
    $('maximum-value').textContent = percent(state.maximum);
    $('progress-value').textContent = percent(state.progress);
    $('retained-value').textContent = `${state.retained} / 12`;
    $('progress').value = Math.round(state.progress * 100);
    $('mode-tag').textContent = modeNames[state.mode];
    $('training-controls').hidden = state.mode !== 'train';
    $('exit-controls').hidden = state.mode !== 'exit';
    $('dense-explanation').hidden = state.mode !== 'dense';
    $('sample').disabled = state.mode !== 'train';
    document.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode)));
    renderNetwork(); renderInspector(); renderChart(); renderCounters(); renderPlayButtons();
  }

  function select(s, l, focus = false) {
    state.selected = [s, l];
    document.querySelectorAll('.cell').forEach(cell => {
      const selected = +cell.dataset.sequence === s && +cell.dataset.layer === l;
      cell.classList.toggle('selected', selected);
      cell.setAttribute('aria-pressed', String(selected));
      cell.setAttribute('tabindex', selected ? '0' : '-1');
      if (focus && selected) cell.focus();
    });
    renderInspector();
  }

  function pointOnRoute(q, keep) {
    if (!keep) return [q * 64, 0];
    const points = [[0, 0], [6, 0], [6, -14], [27, -14], [27, 0], [34, 0], [34, -14], [55, -14], [55, 0], [64, 0]];
    let remaining = q * 120;
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1], b = points[i], length = Math.abs(b[0] - a[0]) + Math.abs(b[1] - a[1]);
      if (remaining <= length) return [a[0] + (b[0] - a[0]) * remaining / length, a[1] + (b[1] - a[1]) * remaining / length];
      remaining -= length;
    }
    return [64, 0];
  }

  function moveTokens(p) {
    const position = p * 14 - 1, l = Math.min(11, Math.max(0, Math.floor(position))), q = Math.min(1, Math.max(0, position - l));
    for (let s = 0; s < 4; s++) {
      let [x, y] = pointOnRoute(q, masks[s][l].keep);
      x += origin + l * width;
      if (position < 0) { x = 113 + (position + 1) * (origin - 113); y = 0; }
      if (position >= 12) { x = 957 + (position - 12) * (1006 - 957); y = 0; }
      const tokens = $(`tokens-${s}`);
      if (tokens) {
        tokens.setAttribute('transform', `translate(${x},${rowY(s) + y})`);
        tokens.style.visibility = playing && !reducedMotion.matches ? 'visible' : 'hidden';
      }
    }
  }

  function renderPlayButtons() {
    $('animate').textContent = playing === 'paths' ? '■ Stop paths' : '▶ Animate paths';
    $('animate').setAttribute('aria-pressed', String(playing === 'paths'));
    $('animate').disabled = reducedMotion.matches;
    $('schedule').textContent = playing === 'schedule' ? '■ Stop training schedule' : '▶ Run training schedule';
    $('schedule').setAttribute('aria-pressed', String(playing === 'schedule'));
    $('motion-help').textContent = reducedMotion.matches ? 'Reduced motion is enabled. Use New masks or the sliders to explore still frames. The schedule advances in discrete steps.' : playing === 'schedule' ? 'Each pass advances training by 10 percentage points and samples a new batch. Motion shows logical paths, not elapsed compute time.' : state.mode === 'train' ? 'Animation replays this batch. New masks draws another batch at the same training progress.' : 'Animation replays the fixed inference path. The path length does not represent measured latency.';
  }

  function stop() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null; playing = null; cycleStart = null; phase = 0;
    renderPlayButtons(); moveTokens(0);
  }

  function tick(now) {
    if (!playing) return;
    if (cycleStart === null) cycleStart = now;
    phase = Math.min(1, (now - cycleStart) / 3600);
    moveTokens(phase);
    if (phase >= 1) {
      if (playing === 'schedule') {
        if (state.progress >= 1) { stop(); announce('Training ends with every block active.'); return; }
        state.progress = Math.min(1, Math.round((state.progress + .1) * 100) / 100);
        state.draws++; sample(); render();
      }
      cycleStart = now; phase = 0;
    }
    frame = requestAnimationFrame(tick);
  }

  function play(kind) {
    if (playing === kind) { stop(); return; }
    stop();
    if (kind === 'paths' && reducedMotion.matches) return;
    if (kind === 'schedule' && state.progress >= 1) { state.progress = 0; state.draws++; sample(); render(); }
    playing = kind; renderPlayButtons(); frame = requestAnimationFrame(tick);
  }

  function announce(prefix) {
    $('status').textContent = `${prefix} ${masks.flat().filter(c => c.keep).length} of 48 sequence–block executions active.`;
  }

  document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
    stop(); state.mode = button.dataset.mode; sample(); render(); announce(modeNames[state.mode]);
  }));
  for (const [id, key, divisor] of [['maximum', 'maximum', 100], ['progress', 'progress', 100], ['retained', 'retained', 1]]) {
    $(id).addEventListener('input', () => { stop(); state[key] = Number($(id).value) / divisor; state.draws++; sample(); render(); });
    $(id).addEventListener('change', () => announce('Settings updated.'));
  }
  $('sample').addEventListener('click', () => { stop(); state.draws++; sample(); render(); announce('New masks sampled.'); });
  $('animate').addEventListener('click', () => play('paths'));
  $('schedule').addEventListener('click', () => play('schedule'));
  $('network').addEventListener('click', event => {
    const cell = event.target.closest('.cell');
    if (cell) select(+cell.dataset.sequence, +cell.dataset.layer);
  });
  $('network').addEventListener('keydown', event => {
    const cell = event.target.closest('.cell');
    if (!cell) return;
    let s = +cell.dataset.sequence, l = +cell.dataset.layer;
    if (event.key === 'ArrowLeft') l = Math.max(0, l - 1);
    else if (event.key === 'ArrowRight') l = Math.min(11, l + 1);
    else if (event.key === 'ArrowUp') s = Math.max(0, s - 1);
    else if (event.key === 'ArrowDown') s = Math.min(3, s + 1);
    else if (event.key === 'Home') l = 0;
    else if (event.key === 'End') l = 11;
    else if (![' ', 'Enter'].includes(event.key)) return;
    event.preventDefault(); select(s, l, true);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  reducedMotion.addEventListener('change', () => { stop(); });
  sample(); render();
})();
