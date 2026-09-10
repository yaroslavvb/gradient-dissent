"""Standalone scientific figure from verified half-drop analysis; no cloud calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent/'results/halfdrop'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def figure(data, synthetic=False):
    if data.get('status') != 'complete' or len(data.get('models', [])) != 32:
        raise ValueError('Require complete32 endpoint/subset means')
    chosen = [p for p in data['paths'] if p['validation_selected']]
    if len(chosen) != 1:
        raise ValueError('Require exactly one validation-selected static path')
    path = chosen[0]
    lookup = {(m['endpoint'], m['drop_mask']): m for m in data['models']}
    by_k = {(m['endpoint'], m['k']): m for m in data['by_k']}
    styles = {'gray': '#8e999b', 'green': '#08776f', 'orange': '#bc621b', 'text': '#233e43'}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': styles['text'],
                         'text.color': styles['text'], 'xtick.color': styles['text'],
                         'ytick.color': styles['text'], 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.2), sharey=True)
        fig.subplots_adjust(left=.077, right=.98, bottom=.255, top=.76, wspace=.10)
        all_limits = []
        for ax, endpoint, title in zip(axes, ('selected', 'final'),
                                       ('A  Validation-CE-selected checkpoint · primary', 'B  Epoch 100 checkpoint · secondary')):
            rows = [lookup[(endpoint, mask)] for mask in range(16)]
            ax.scatter([r['k'] for r in rows], [100*r['metrics']['accuracy']['mean'] for r in rows],
                       color=styles['gray'], alpha=.75, s=30, linewidths=.5, edgecolors='white', zorder=2)
            for selected_rows, color, marker in [([by_k[(endpoint, k)] for k in range(5)], styles['green'], 's'),
                                                ([lookup[(endpoint, mask)] for mask in path['masks']], styles['orange'], 'o')]:
                stats = [r['metrics']['accuracy'] for r in selected_rows]
                if any(r['n'] != 3 or r['ci95_low'] is None or r['ci95_high'] is None for r in stats):
                    raise ValueError('Every plotted interval must contain three seed observations')
                means = 100*np.array([r['mean'] for r in stats])
                lows = 100*np.array([r['ci95_low'] for r in stats])
                highs = 100*np.array([r['ci95_high'] for r in stats])
                all_limits.extend(lows.tolist()+highs.tolist())
                # These are seed CIs, not variation across subsets or test images.
                ax.errorbar(range(5), means, yerr=np.stack([means-lows, highs-means]),
                            color=color, marker=marker, linewidth=1.9, markersize=5.5,
                            markerfacecolor='white' if marker == 'o' else color,
                            markeredgewidth=1.3, capsize=3.5, elinewidth=1.05, zorder=4 if marker == 'o' else 3)
            all_limits.extend(100*r['metrics']['accuracy']['mean'] for r in rows)
            ax.set_title(title, loc='left', fontsize=10.3, pad=13, fontweight='semibold')
            ax.set_xticks(range(5))
            ax.set_xlim(-.20, 4.20)
            ax.set_xlabel('Middle branches eligible for 50% training dropout', labelpad=10)
            ax.grid(axis='y', color='#dde5e3', linewidth=.75)
            ax.set_axisbelow(True)
            ax.spines['left'].set_color('#9dadaa'); ax.spines['bottom'].set_color('#9dadaa')
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        low, high = min(all_limits), max(all_limits)
        pad = max(.12, .10*(high-low))
        axes[0].set_ylim(low-pad, high+pad)
        axes[0].set_ylabel('Full-depth test accuracy (%)', labelpad=10)
        heading = 'SYNTHETIC FIXTURE — NOT RESULTS' if synthetic else 'Training with 50% branch dropout'
        fig.text(.077, .963, heading, fontsize=19, weight='bold', va='top')
        fig.text(.077, .903, 'Every test prediction retains all six learned affines at gain one.  Three training seeds: 201, 202, 203.',
                 fontsize=10.3, va='top')
        handles = [Line2D([], [], linestyle='none', marker='o', color=styles['gray'], markersize=5, label='Every eligibility subset: seed mean'),
                   Line2D([], [], color=styles['green'], marker='s', markersize=5, label='Uniform mean over subsets at each count'),
                   Line2D([], [], color=styles['orange'], marker='o', markerfacecolor='white', markersize=5, label='Validation-selected prefix path')]
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.067, .865), ncol=3, frameon=False,
                   handlelength=2.1, columnspacing=1.8, fontsize=9.5)
        order = ' → '.join(str(j+1) for j in path['order'])
        fig.text(.077, .139, f'Frozen layer order: {order}. It minimizes validation error (then CE, then lexical ties), not test error.', fontsize=9.2)
        fig.text(.077, .100, 'Whiskers: 95% t intervals over three seeds (df=2), unadjusted for multiple comparisons and conditional on validation selection.', fontsize=9.2)
        fig.text(.077, .061, 'The green mean averages subsets within each seed first. Paths connect independently trained subsets; they are not temporal curricula.', fontsize=9.2)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    source = args.results_dir/'analysis.json'
    data = json.loads(source.read_text())
    order = Path(data['provenance']['order_manifest']['path'])
    if not order.is_absolute():
        order = HERE.parents[2]/order
    if sha(order) != data['provenance']['order_manifest']['sha256']:
        raise ValueError('Ordering manifest changed after analysis')
    fig = figure(data)
    outputs = {}
    for suffix in ('svg', 'png'):
        path = args.results_dir/('accuracy-vs-drop-count.'+suffix)
        fig.savefig(path, dpi=250, facecolor='white')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        outputs[suffix] = {'file': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size}
    plt.close(fig)
    manifest = {'analysis_sha256': sha(source), 'plot_source_sha256': sha(__file__), 'outputs': outputs,
                'confidence_intervals': 'Three seed t intervals, df2; no multiple-comparison or validation-selection correction',
                'inference': 'All six affines at gain1; x axis counts training eligibility at p=.5'}
    (args.results_dir/'figure-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'status': 'complete', 'files': [v['file'] for v in outputs.values()]}))


if __name__ == '__main__':
    main()
