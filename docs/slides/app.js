(() => {
  'use strict';
  // ---- Math: the paper's stated formulas, exposed for the check script. ----
  const M = {
    survivorScale: p => 1 / p,
    effectiveDepth: (p, L) => p * L,
    // Per-layer drop probabilities with mean `s` (= expected block-FLOPs saved). Layer 0 is never dropped by the non-uniform shapes.
    distribution(kind, s, L) {
      const ps = Array.from({length: L}, (_, l) => {
        if (kind === 'uniform') return s;
        if (kind === 'increasing') return 2 * s * l / (L - 1);
        return l % 2 === 1 ? 2 * s * L / Math.floor(L / 2) / 2 : 0; // alternating: odd layers carry the whole budget
      });
      return ps.map(p => Math.min(1, p));
    },
    feasible: (kind, s) => kind === 'uniform' ? s <= 1 : s <= .5,
    schedule(kind, t, pmax) { return kind === 'constant' ? pmax : kind === 'increasing' ? pmax * t : pmax * (1 - t); },
    // The recipe: increasing in depth × the chosen time schedule. Layer l of L at progress t.
    recipe: (kind, l, t, pmax, L) => M.schedule(kind, t, pmax) * l / (L - 1),
    meanSaved: (kind, pmax) => kind === 'constant' ? pmax / 2 : pmax / 4,
    // Fraction of a layer's weights that must be read in one step: per batch, the layer is read unless dropped for the whole step; per sequence, unless every sequence drops it.
    weightsRead: (gran, r, S) => gran === 'batch' ? 1 - r : 1 - Math.pow(r, S),
    masks(gran, ps, S, rng) {
      if (gran === 'batch') { const step = ps.map(p => rng() >= p); return Array.from({length: S}, () => step.slice()); }
      return Array.from({length: S}, () => ps.map(p => rng() >= p));
    },
    seeded(seed) { let x = seed >>> 0; return () => { x = (x * 1664525 + 1013904223) >>> 0; return x / 4294967296; }; }
  };
  window.SlidesMath = M;

  const w = window;
  const $ = id => document.getElementById(id);
  const on = (id, ev, fn) => $(id).addEventListener(ev, fn);
  const pct = (n, d = 0) => `${(100 * n).toFixed(d)}%`;
  const fx = (n, d = 2) => n.toFixed(d);
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const C = {blue: 'var(--blue)', orange: 'var(--orange)', green: 'var(--green)', muted: 'var(--muted)', line: 'var(--line)', soft: 'var(--soft)'};

  // ---- Tooltip: one element, fed by data-tip on any mark. ----
  const tip = $('tooltip');
  document.addEventListener('mousemove', e => {
    const t = e.target.closest && e.target.closest('[data-tip]');
    if (!t) { tip.hidden = true; return; }
    tip.textContent = t.dataset.tip; tip.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = `${x}px`; tip.style.top = `${e.clientY + 16}px`;
  });

  // ---- Deck navigation: hash-routed (#/N), keyboard, buttons, swipe, contents. ----
  const slides = [...document.querySelectorAll('.slide')];
  const N = slides.length;
  let index = 0;
  $('toc-list').innerHTML = slides.map((s, i) => `<li><a href="#/${i + 1}"><span>${String(i + 1).padStart(2, '0')}</span>${esc(s.dataset.title)}</a></li>`).join('');
  function show(i, push = true) {
    index = Math.max(0, Math.min(N - 1, i));
    slides.forEach((s, j) => s.classList.toggle('current', j === index));
    $('slide-counter').textContent = `${index + 1} / ${N}`;
    $('progress').style.width = `${100 * (index + 1) / N}%`;
    $('prev-slide').disabled = index === 0; $('next-slide').disabled = index === N - 1;
    document.querySelectorAll('#toc a').forEach((a, j) => a.classList.toggle('active', j === index));
    document.title = `${index + 1}. ${slides[index].dataset.title} · Don't Drop Dropout · Gradient dissent`;
    if (push && location.hash !== `#/${index + 1}`) history.replaceState(null, '', `#/${index + 1}`);
    window.scrollTo(0, 0);
  }
  const fromHash = () => { const m = /^#\/(\d+)$/.exec(location.hash); show(m ? +m[1] - 1 : 0, false); };
  window.addEventListener('hashchange', fromHash);
  on('prev-slide', 'click', () => show(index - 1)); on('next-slide', 'click', () => show(index + 1));
  on('toc-toggle', 'click', () => { const open = $('toc').hidden; $('toc').hidden = !open; $('toc-toggle').setAttribute('aria-expanded', String(open)); });
  $('toc').addEventListener('click', e => { if (e.target.closest('a')) { $('toc').hidden = true; $('toc-toggle').setAttribute('aria-expanded', 'false'); } });
  document.addEventListener('keydown', e => {
    const t = e.target instanceof w.Element ? e.target : null;
    if (t && t.matches('input,textarea') && ['ArrowLeft', 'ArrowRight', ' ', 'Home', 'End'].includes(e.key)) return; // sliders own their arrow keys
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (['ArrowRight', 'PageDown', ' ', 'j'].includes(e.key)) { e.preventDefault(); show(index + 1); }
    else if (['ArrowLeft', 'PageUp', 'k'].includes(e.key)) { e.preventDefault(); show(index - 1); }
    else if (e.key === 'Home') show(0); else if (e.key === 'End') show(N - 1);
    else if (e.key === 'o') $('toc-toggle').click();
    else if (e.key === 'f' && document.documentElement.requestFullscreen) { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
    else if (e.key === 'Escape' && !$('toc').hidden) $('toc-toggle').click();
  });
  let touchX = null;
  document.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, {passive: true});
  document.addEventListener('touchend', e => { if (touchX === null || e.target.closest('input,button,a,.figure')) return; const dx = e.changedTouches[0].clientX - touchX; if (Math.abs(dx) > 60) show(index + (dx < 0 ? 1 : -1)); touchX = null; }, {passive: true});

  // ---- Chart helpers (inline SVG strings). ----
  const line = (x1, y1, x2, y2, cls = 'gridline') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}"/>`;
  const text = (x, y, s, cls = 'lbl', extra = '') => `<text x="${x}" y="${y}" class="${cls}" ${extra}>${esc(s)}</text>`;
  const legend = (x, y, items, step = 150) => items.map((it, i) => `<rect x="${x + i * step}" y="${y - 9}" width="10" height="10" rx="2" fill="${it.color}"/>` + text(x + 15 + i * step, y, it.label)).join('');

  // ---- Slide 2: one residual block. ----
  const blockRng = M.seeded(1603);
  let blockKept = true;
  function renderBlock() {
    const p = .4, scale = M.survivorScale(1 - p);
    const stroke = blockKept ? C.blue : C.line, dash = blockKept ? '' : 'stroke-dasharray="5 5"';
    $('block-figure').innerHTML = `
      ${text(20, 30, 'x', 'val')}${text(380, 30, blockKept ? 'x + branch(x) / 0.6' : 'x  (identity)', 'val', 'text-anchor="end"')}
      <path d="M30 60 H390" stroke="${C.muted}" stroke-width="2" fill="none"/>
      ${text(210, 52, 'residual stream', 'lbl', 'text-anchor="middle"')}
      <path d="M120 60 V130 H300 V60" stroke="${stroke}" stroke-width="2" fill="none" ${dash}/>
      <rect x="150" y="105" width="120" height="70" rx="3" fill="${blockKept ? C.soft : 'none'}" stroke="${stroke}" stroke-width="1.5" ${dash}/>
      ${text(210, 133, 'attention', 'val', 'text-anchor="middle"')}${text(210, 155, '+ feed-forward', 'val', 'text-anchor="middle"')}
      ${text(210, 200, blockKept ? 'kept · output scaled by 1 / 0.6 = 1.67' : 'skipped · branch not computed', blockKept ? 'val' : 'lbl', 'text-anchor="middle"')}
      ${text(210, 232, `drop probability ${pct(p)} · keep ${pct(1 - p)}`, 'lbl', 'text-anchor="middle"')}`;
    $('block-status').textContent = blockKept ? `Ran the branch (scale ${fx(scale)}×).` : 'Skipped: the residual carried the stream.';
  }
  on('block-sample', 'click', () => { blockKept = blockRng() >= .4; renderBlock(); });
  renderBlock();

  // ---- Slide 4: scaling factor. ----
  function renderScale() {
    const p = +$('keep').value / 100, L = +$('depth').value;
    $('keep-value').textContent = fx(p); $('depth-value').textContent = L;
    const eff = M.effectiveDepth(p, L), scale = M.survivorScale(p);
    $('scale-formula').innerHTML = `CompleteP branch scale 1/L = 1/${L}  →  with dropout: effective depth p·L = ${fx(eff, 1)},  survivor scale 1/p = <strong>${fx(scale)}×</strong>,  branch scale (1/p)·(1/L) = 1/${fx(eff, 1)}`;
    const W = 460, H = 280, x0 = 50, x1 = 420, y0 = 210, y1 = 30;
    const X = q => x0 + (x1 - x0) * (q - .1) / .9;
    const Yd = d => y0 - (y0 - y1) * d / L, Ys = s => y0 - (y0 - y1) * Math.min(s, 10) / 10;
    let svg = '';
    for (let g = 0; g <= 4; g++) { const y = y0 - (y0 - y1) * g / 4; svg += line(x0, y, x1, y) + text(x0 - 6, y + 4, `${(L * g / 4).toFixed(0)}`, 'lbl', 'text-anchor="end"'); }
    svg += line(x0, y0, x1, y0, 'axis') + text((x0 + x1) / 2, y0 + 34, 'keep probability p', 'lbl', 'text-anchor="middle"') + text(x0 - 6, y1 - 12, 'layers', 'lbl', 'text-anchor="end"') + text(x1 + 6, y1 - 12, '×', 'lbl');
    for (let g = 0; g <= 4; g++) svg += text(x1 + 6, y0 - (y0 - y1) * g / 4 + 4, `${(10 * g / 4).toFixed(0)}`, 'lbl');
    for (const q of [.1, .25, .5, .75, 1]) svg += text(X(q), y0 + 18, fx(q), 'lbl', 'text-anchor="middle"');
    const pts = []; for (let q = .1; q <= 1.0001; q += .025) pts.push([X(q), Yd(M.effectiveDepth(q, L))]);
    svg += `<path d="M${pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L')}" fill="none" stroke="${C.blue}" stroke-width="2"/>`;
    const sp = []; for (let q = .1; q <= 1.0001; q += .01) sp.push([X(q), Ys(M.survivorScale(q))]);
    svg += `<path d="M${sp.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L')}" fill="none" stroke="${C.orange}" stroke-width="2"/>`;
    svg += `<line x1="${X(p)}" y1="${y1}" x2="${X(p)}" y2="${y0}" stroke="${C.muted}" stroke-dasharray="3 4"/>`;
    svg += `<circle cx="${X(p)}" cy="${Yd(eff)}" r="5" fill="${C.blue}" stroke="var(--paper)" stroke-width="2"/><circle cx="${X(p)}" cy="${Ys(scale)}" r="5" fill="${C.orange}" stroke="var(--paper)" stroke-width="2"/>`;
    svg += text(X(p) + 8, Yd(eff) - 8, `p·L = ${fx(eff, 1)}`, 'val') + text(X(p) + 8, Ys(scale) + (scale > 8 ? 16 : -8), `1/p = ${fx(scale)}×`, 'val');
    svg += legend(x0 + 120, H - 4, [{color: C.blue, label: 'effective depth (left)'}, {color: C.orange, label: 'survivor scale (right)'}], 160);
    $('scale-chart').innerHTML = `<title id="scale-title">Effective depth and survivor scale against keep probability</title>${svg}`;
  }
  on('keep', 'input', renderScale); on('depth', 'input', renderScale); renderScale();

  // ---- Slide 5: granularity. ----
  const gran = {kind: 'sequence', seed: 7};
  const granRng = M.seeded(2609);
  function renderGran() {
    const r = +$('gran-rate').value / 100, S = 8, L = 12;
    $('gran-rate-value').textContent = pct(r);
    const rng = M.seeded(gran.seed * 7919 + Math.round(r * 1000) * 31 + (gran.kind === 'batch' ? 1 : 0));
    const masks = M.masks(gran.kind, Array(L).fill(r), S, rng);
    const depths = masks.map(m => m.filter(Boolean).length);
    const mean = depths.reduce((a, b) => a + b, 0) / S, sd = Math.sqrt(depths.reduce((a, d) => a + (d - mean) ** 2, 0) / S);
    const x0 = 24, cw = 30, ch = 26, y0 = 40;
    let svg = '';
    for (let l = 0; l < L; l++) svg += text(x0 + l * cw + cw / 2, 28, `${l + 1}`, 'lbl', 'text-anchor="middle"');
    svg += text(x0 + L * cw + 14, 28, 'depth', 'lbl');
    masks.forEach((row, s) => {
      const y = y0 + s * ch;
      svg += text(x0 - 6, y + 17, `${s + 1}`, 'lbl', 'text-anchor="end"');
      row.forEach((keep, l) => { svg += `<rect class="cell" x="${x0 + l * cw + 2}" y="${y + 2}" width="${cw - 4}" height="${ch - 4}" rx="3" fill="${keep ? C.blue : 'none'}" stroke="${keep ? C.blue : C.line}" stroke-dasharray="${keep ? '' : '3 3'}" data-tip="sequence ${s + 1}, block ${l + 1}: ${keep ? 'runs' : 'identity bypass'}"/>`; });
      svg += text(x0 + L * cw + 14, y + 17, `${depths[s]} / ${L}`, 'val');
    });
    svg += text(x0, y0 + S * ch + 22, `${gran.kind === 'batch' ? 'Per batch: one draw per layer, shared by all sequences.' : 'Per sequence: every sequence draws its own mask.'}`, 'lbl');
    svg += text(x0, y0 + S * ch + 40, `Active depth: mean ${fx(mean, 1)}, spread ±${fx(sd, 1)} across sequences · step depth ${gran.kind === 'batch' ? 'all-or-nothing' : 'averages out'}`, 'lbl');
    $('gran-chart').innerHTML = `<title id="gran-title">Which of eight sequences run which of twelve blocks in one training step</title>${svg}`;
    $('gran-summary').textContent = gran.kind === 'batch'
      ? `This step every sequence ran the same ${depths[0]} of ${L} blocks — the whole batch sees one depth, and the next step may see a very different one.`
      : `Sequences saw ${Math.min(...depths)}–${Math.max(...depths)} of ${L} blocks; the batch as a whole averages over ${S} independent masks, so the gradient is far less noisy.`;
  }
  document.querySelectorAll('[data-gran]').forEach(b => b.addEventListener('click', () => { gran.kind = b.dataset.gran; document.querySelectorAll('[data-gran]').forEach(o => o.setAttribute('aria-pressed', String(o === b))); renderGran(); }));
  on('gran-rate', 'input', renderGran); on('gran-sample', 'click', () => { gran.seed = Math.floor(granRng() * 1e6); renderGran(); }); renderGran();

  // ---- Slide 6: distribution across depth at matched mean. ----
  function renderDist() {
    const s = +$('savings').value / 100, L = +$('dist-depth').value;
    $('savings-value').textContent = pct(s); $('dist-depth-value').textContent = L;
    const kinds = [['uniform', C.muted, 'Uniform'], ['increasing', C.blue, 'Increasing'], ['alternating', C.orange, 'Alternating']];
    const x0 = 44, x1 = 440, y0 = 240, y1 = 34, bw = (x1 - x0) / L;
    let svg = '';
    for (let g = 0; g <= 4; g++) { const y = y0 - (y0 - y1) * g / 4; svg += line(x0, y, x1, y) + text(x0 - 6, y + 4, pct(g / 4), 'lbl', 'text-anchor="end"'); }
    svg += line(x0, y0, x1, y0, 'axis') + text(x1, y0 + 32, 'layer →', 'lbl', 'text-anchor="end"');
    kinds.forEach(([kind, color], k) => {
      const ps = M.distribution(kind, s, L);
      ps.forEach((p, l) => { const h = (y0 - y1) * p, x = x0 + l * bw + 2 + k * ((bw - 4) / 3); svg += `<rect class="bar" x="${x.toFixed(1)}" y="${(y0 - h).toFixed(1)}" width="${((bw - 4) / 3 - 1).toFixed(1)}" height="${h.toFixed(1)}" fill="${color}" data-tip="${kind}, layer ${l + 1}: drop ${pct(p, 1)}"/>`; });
    });
    for (let l = 0; l < L; l += L > 16 ? 4 : 2) svg += text(x0 + l * bw + bw / 2, y0 + 16, `${l + 1}`, 'lbl', 'text-anchor="middle"');
    svg += legend(x0, 290, kinds.map(([, color, label]) => ({color, label})));
    const infeasible = kinds.filter(([kind]) => !M.feasible(kind, s)).map(([, , label]) => label);
    svg += text(x1, y1 - 14, `mean drop rate = ${pct(s)} for every shape`, 'val', 'text-anchor="end"');
    $('dist-chart').innerHTML = `<title id="dist-title">Per-layer drop probability for uniform, increasing and alternating distributions at matched mean</title>${svg}`;
    const inc = M.distribution('increasing', s, L), alt = M.distribution('alternating', s, L);
    $('dist-summary').textContent = `At ${pct(s)} saved: uniform drops every layer ${pct(s)} of the time; increasing runs from 0% at layer 1 to ${pct(inc[L - 1], 0)} at layer ${L}; alternating never touches the odd layers and drops the even ones ${pct(alt[1], 0)} of the time.` + (infeasible.length ? ` Above 50% savings ${infeasible.join(' and ')} would need rates over 100% — capped.` : '');
  }
  on('savings', 'input', renderDist); on('dist-depth', 'input', renderDist); renderDist();

  // ---- Slide 7: schedule over time + recipe grid. ----
  const sched = {kind: 'decreasing'};
  function renderSched() {
    const pmax = +$('pmax').value / 100, t = +$('progress-t').value / 100, L = 12;
    $('pmax-value').textContent = fx(pmax); $('progress-t-value').textContent = pct(t);
    const x0 = 44, x1 = 380, yT0 = 120, yT1 = 24;
    let svg = '';
    for (let g = 0; g <= 2; g++) { const y = yT0 - (yT0 - yT1) * g / 2; svg += line(x0, y, x1, y) + text(x0 - 6, y + 4, fx(g / 2, 1), 'lbl', 'text-anchor="end"'); }
    svg += line(x0, yT0, x1, yT0, 'axis') + text((x0 + x1) / 2, yT0 + 14, 'training progress →', 'lbl', 'text-anchor="middle"') + text(x0, yT1 - 8, 'last-layer drop rate', 'ttl');
    const kinds = [['constant', C.muted], ['increasing', C.orange], ['decreasing', C.blue]];
    const ends = kinds.map(([kind, color]) => ({kind, color, a: M.schedule(kind, 0, pmax), b: M.schedule(kind, 1, pmax), active: kind === sched.kind}));
    ends.forEach(e => { e.y = yT0 - (yT0 - yT1) * e.b; e.ly = e.y + 4; });
    ends.slice().sort((p, q) => p.y - q.y).forEach((e, i, arr) => { if (i > 0 && e.ly - arr[i - 1].ly < 11) e.ly = arr[i - 1].ly + 11; }); // stagger labels that share an endpoint
    ends.forEach(e => {
      svg += `<line x1="${x0}" y1="${yT0 - (yT0 - yT1) * e.a}" x2="${x1}" y2="${e.y}" stroke="${e.color}" stroke-width="${e.active ? 2.5 : 1.2}" opacity="${e.active ? 1 : .45}" ${e.active ? '' : 'stroke-dasharray="4 4"'}/>`;
      svg += text(x1 + 6, e.ly, e.kind, e.active ? 'val' : 'lbl', 'text-anchor="start" font-size="10"');
    });
    const cur = M.schedule(sched.kind, t, pmax), xt = x0 + (x1 - x0) * t;
    svg += `<line x1="${xt}" y1="${yT1}" x2="${xt}" y2="${yT0}" stroke="${C.muted}" stroke-dasharray="3 4"/><circle cx="${xt}" cy="${yT0 - (yT0 - yT1) * cur}" r="5" fill="${C.blue}" stroke="var(--paper)" stroke-width="2"/>`;
    // Recipe grid: depth × time, increasing in depth × chosen schedule.
    const gy0 = 160, gh = 130, T = 20, gx1 = 440, cw = (gx1 - x0) / T, ch = gh / L;
    svg += text(x0, gy0 - 8, 'recipe grid · rows = layers 1…12 (top = first), columns = training time', 'ttl');
    for (let l = 0; l < L; l++) for (let c = 0; c < T; c++) {
      const tt = c / (T - 1), p = M.recipe(sched.kind, l, tt, pmax, L);
      svg += `<rect class="cell" x="${(x0 + c * cw).toFixed(1)}" y="${(gy0 + l * ch).toFixed(1)}" width="${(cw - 1).toFixed(1)}" height="${(ch - 1).toFixed(1)}" fill="${C.blue}" fill-opacity="${(0.06 + 0.94 * p).toFixed(2)}" data-tip="layer ${l + 1} at ${pct(tt)} of training: drop ${pct(p, 1)}"/>`;
    }
    svg += `<rect x="${(x0 + Math.round(t * (T - 1)) * cw).toFixed(1)}" y="${gy0}" width="${(cw - 1).toFixed(1)}" height="${gh}" fill="none" stroke="${C.orange}" stroke-width="1.5"/>`;
    svg += text(x0, gy0 + gh + 16, `average over depth and time: ${pct(M.meanSaved(sched.kind, pmax), 1)} of block FLOPs skipped`, 'lbl');
    $('sched-chart').innerHTML = `<title id="sched-title">Drop rate of the last layer over training, and the depth-by-time grid of the recipe</title>${svg}`;
    const eff = L - Array.from({length: L}, (_, l) => M.recipe(sched.kind, l, t, pmax, L)).reduce((a, b) => a + b, 0);
    $('sched-summary').textContent = `${sched.kind[0].toUpperCase() + sched.kind.slice(1)} schedule, pmax ${fx(pmax)}: at ${pct(t)} of training the last layer is skipped ${pct(cur, 0)} of the time and the expected active depth is ${fx(eff, 1)} of ${L} blocks.` + (sched.kind === 'decreasing' ? ' By the end every block is always present — the model has "grown" to full depth.' : sched.kind === 'increasing' ? ' The model is densest at the start and thinnest at the end — the LayerSkip / progressive-layer-dropping direction, which this paper finds worst.' : '');
  }
  document.querySelectorAll('[data-sched]').forEach(b => b.addEventListener('click', () => { sched.kind = b.dataset.sched; document.querySelectorAll('[data-sched]').forEach(o => o.setAttribute('aria-pressed', String(o === b))); renderSched(); }));
  on('pmax', 'input', renderSched); on('progress-t', 'input', renderSched); renderSched();

  // ---- Slide 8: measured elastic-depth losses (Table 5). ----
  (function renderElastic() {
    const groups = [
      {label: '3.9B · full depth', dense: 1.732, dropout: 1.745},
      {label: '3.9B · alternate layers skipped', dense: 6.4, dropout: 2.129},
      {label: '3.9B · 75% depth (early exit)', dense: null, dropout: 2.143},
      {label: '8.2B · full depth', dense: null, dropout: 1.663},
      {label: '8.2B · 75% depth (early exit)', dense: null, dropout: 1.777}
    ];
    const x0 = 190, x1 = 440, rowH = 44, y0 = 48, max = 7;
    const X = v => x0 + (x1 - x0) * v / max;
    let svg = '';
    for (const g of [0, 2, 4, 6]) svg += line(X(g), y0 - 10, X(g), y0 + groups.length * rowH - 6) + text(X(g), y0 - 16, `${g}`, 'lbl', 'text-anchor="middle"');
    svg += text(x1, y0 - 32, 'validation loss (nats)', 'lbl', 'text-anchor="end"');
    groups.forEach((g, i) => {
      const y = y0 + i * rowH;
      svg += text(x0 - 8, y + 20, g.label, 'lbl', 'text-anchor="end"');
      if (g.dense !== null) svg += `<rect class="bar" x="${x0}" y="${y}" width="${(X(g.dense) - x0).toFixed(1)}" height="14" rx="2" fill="${C.muted}" data-tip="dense: ${g.dense}"/>` + text(X(g.dense) + 5, y + 11, `${g.dense}`, 'val');
      else svg += text(x0 + 4, y + 11, 'no dense number reported', 'lbl');
      svg += `<rect class="bar" x="${x0}" y="${y + 17}" width="${(X(g.dropout) - x0).toFixed(1)}" height="14" rx="2" fill="${C.blue}" data-tip="dropout-trained: ${g.dropout}"/>` + text(X(g.dropout) + 5, y + 28, `${g.dropout}`, 'val');
    });
    svg += legend(x0, y0 + groups.length * rowH + 14, [{color: C.muted, label: 'dense'}, {color: C.blue, label: 'dropout-trained'}]);
    svg += text(x0, y0 + groups.length * rowH + 34, 'Skipping half the 3.9B layers costs 0.38 nats;', 'lbl') + text(x0, y0 + groups.length * rowH + 50, 'a 500M dense model lands near 2.1.', 'lbl');
    $('elastic-chart').innerHTML = `<title id="elastic-title">Measured losses at reduced depth for the 3.9B and 8.2B models</title>${svg}`;
  })();

  // ---- Slide 9: hero p_max. ----
  function renderHero() {
    const pmax = +$('hero-pmax').value / 100;
    $('hero-pmax-value').textContent = fx(pmax); $('hero-last').textContent = pct(pmax, 0); $('hero-saved').textContent = pct(M.meanSaved('decreasing', pmax), 2);
    $('hero-note').textContent = `Under the recipe the average active depth at step 0 is ${pct(1 - pmax / 2, 1)} of full depth; the last layer survives ${pct(1 - pmax, 0)} of sequences. Averaged over the whole run, ${pct(pmax / 4, 2)} of block FLOPs are skipped — the "25%" is p_max/4 with p_max = 0.99, not 99% of anything.`;
  }
  on('hero-pmax', 'input', renderHero); renderHero();

  // ---- Slide 12: FLOPs versus bytes. ----
  function renderBytes() {
    const r = +$('bytes-rate').value / 100, S = +$('bytes-seqs').value;
    $('bytes-rate-value').textContent = pct(r); $('bytes-seqs-value').textContent = S;
    const rows = [
      {label: 'FLOPs for this layer (either granularity)', v: 1 - r, color: C.muted},
      {label: 'weights read · per batch', v: M.weightsRead('batch', r, S), color: C.orange},
      {label: 'weights read · per sequence', v: M.weightsRead('sequence', r, S), color: C.blue}
    ];
    const x0 = 20, x1 = 440, y0 = 30, rowH = 62;
    let svg = text(x0, 18, 'fraction of the dense cost, expected per training step', 'ttl');
    rows.forEach((row, i) => {
      const y = y0 + i * rowH;
      svg += text(x0, y + 12, row.label, 'lbl') + `<rect x="${x0}" y="${y + 20}" width="${x1 - x0}" height="16" rx="2" fill="${C.soft}"/><rect class="bar" x="${x0}" y="${y + 20}" width="${((x1 - x0) * row.v).toFixed(1)}" height="16" rx="2" fill="${row.color}" data-tip="${row.label}: ${pct(row.v, 1)}"/>` + text(x0 + Math.max(4, (x1 - x0) * row.v - 40), y + 32, pct(row.v, 1), 'val', `fill="${row.v > .12 ? 'var(--paper)' : 'var(--ink)'}"`);
    });
    svg += text(x0, y0 + rows.length * rowH + 8, `P(all ${S} sequences skip) = ${pct(Math.pow(r, S), 2)} — the only case where per-sequence saves a read.`, 'lbl');
    $('bytes-chart').innerHTML = `<title id="bytes-title">Fraction of a layer's weights read per step: FLOPs versus bytes</title>${svg}`;
  }
  on('bytes-rate', 'input', renderBytes); on('bytes-seqs', 'input', renderBytes); renderBytes();

  fromHash();
})();
