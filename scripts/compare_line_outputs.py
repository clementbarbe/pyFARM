"""Compare baseline FARM outputs and explicit residual-line outputs, read-only."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MNE_DONTWRITE_HOME', 'true')

import mne
import numpy as np
from farm.preprocessing.filters import apply_bandpass
from farm.visualization.spectra import _windowed_fft_db


def compare(root, output, validation_root=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    (output/'figures').mkdir(exist_ok=True)
    rows, links = [], []
    for path in sorted(root.rglob('*_denoise.json')):
        extra = json.loads(path.read_text())
        if 'line_cleanup_input_vhdr' not in extra:
            continue
        base = Path(extra['line_cleanup_input_vhdr'])
        bm = json.loads(base.with_name(base.stem+'_denoise.json').read_text())
        names = extra['selected_channels']
        start, stop = bm['scan_start_sample'], bm['scan_stop_sample_exclusive']
        after_path = path.with_name(path.name.replace('_denoise.json', '.vhdr'))
        with mne.io.read_raw_brainvision(base, preload=False, verbose=False) as b, mne.io.read_raw_brainvision(after_path, preload=False, verbose=False) as a:
            if b.n_times != a.n_times or b.info['sfreq'] != a.info['sfreq']:
                raise ValueError(f'Incompatible samples: {path}')
            fs = float(b.info['sfreq'])
            if not np.array_equal(np.rint(b.annotations.onset*fs),np.rint(a.annotations.onset*fs)):
                raise ValueError(f'Marker sample mismatch: {path}')
            before, after = b.get_data(picks=names), a.get_data(picks=names)
            if not all(np.allclose(before[:,s:e],after[:,s:e],rtol=3e-7,atol=1e-12)
                       for s,e in [(0,start),(stop,b.n_times)]):
                raise ValueError(f'Changed samples outside scan crop: {path}')
        for i, name in enumerate(names):
            report = extra['line_cleanup']['channels'][i]
            x,y = (apply_bandpass(v[i,start:stop],fs,bm['qc_bandpass_hz']) for v in (before,after))
            f = np.fft.rfftfreq(len(x),1/fs)
            pb,pa = [10**(_windowed_fft_db(v)/10) for v in (x,y)]
            selected = [c for c in report['candidates'] if c['selected']]
            peak_b,peak_a = 0.,0.
            for c in selected:
                mask = abs(f-c['frequency_hz']) <= .05
                peak_b += float(pb[mask].max())
                peak_a += float(pa[mask].max())
            lb = next(r for r in bm['scanner_locked_reduction'] if r['ch_name']==name)
            la = next(r for r in extra['scanner_locked_reduction'] if r['ch_name']==name)
            row = dict(run=Path(bm['input_vhdr']).stem,channel=name,selected_lines=len(selected),
                       frequencies_hz=';'.join(f"{c['frequency_hz']:.4f}" for c in selected),
                       selected_fft_peak_reduction_db=float(10*np.log10(peak_b/peak_a)) if peak_b>0 and peak_a>0 else None,
                       median_selected_region_reduction_db=float(np.median([c['reduction_db'] for c in selected])) if selected else None,
                       rms_farm_uv=float(np.sqrt(np.mean(x.astype(float)**2))*1e6),
                       rms_extra_uv=float(np.sqrt(np.mean(y.astype(float)**2))*1e6),
                       locked_farm_uv=lb['locked_rms_after']*1e6,locked_extra_uv=la['locked_rms_after']*1e6,
                       envelope_change_percent=100*report.get('metrics',{}).get('band_relative_envelope_change',0),
                       off_line_power_ratio=report.get('metrics',{}).get('off_line_power_ratio'),
                       output_vhdr=str(after_path))
            rows.append(row)
            fig,axes=plt.subplots(1,2,figsize=(15,4),constrained_layout=True)
            mask=f<=375
            for ax in axes:
                for power,label,color in [(pb,'FARM','#c45d5d'),(pa,'FARM + regression','#25854f')]:
                    ax.plot(f[mask],power[mask] if ax is axes[0] else 10*np.log10(np.maximum(power[mask],1e-40)),
                            color=color,lw=.5,alpha=.8,label=label)
                ax.set(xlim=(0,375),xlabel='Hz')
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
            axes[0].set(ylabel='Squared FFT amplitude (V²)',title='Same linear scale')
            axes[1].set(ylabel='dB re 1 V²',title='Same dB scale')
            fig.suptitle(f"{row['run']} — {name} — {len(selected)} lines corrected; identical QC band")
            image=output/'figures'/f"{row['run']}_{name}.png"
            fig.savefig(image,dpi=130);plt.close(fig)
            links.append((row,image.relative_to(output).as_posix()))
    if not rows:
        raise ValueError('No evaluated residual-line outputs found')
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(runs=len({r['run'] for r in rows}),channels=len(rows),
                 channels_with_selected_lines=sum(r['selected_lines']>0 for r in rows),
                 note='Artifact reduction and waveform change, not physiological validation',
                 median_selected_fft_peak_reduction_db=float(np.median([r['selected_fft_peak_reduction_db'] for r in rows if r['selected_lines']])))
    validation = []
    if validation_root is not None:
        for path in sorted(validation_root.glob('*.json')):
            record = json.loads(path.read_text())
            if 'injection_rms_uv' not in record:
                continue
            for channel in record['channels']:
                old = channel['without_line_cleanup']
                validation.append(dict(run=Path(record['input_vhdr']).stem, channel=channel['channel'],
                    injection_rms_uv=record['injection_rms_uv'], gain_farm=old['gain'], gain_extra=channel['gain'],
                    envelope_error_farm=old['relative_envelope_error'], envelope_error_extra=channel['relative_envelope_error']))
        if validation:
            summary['injection_channel_cases'] = len(validation)
            summary['max_absolute_injection_gain_change'] = max(abs(v['gain_extra']-v['gain_farm']) for v in validation)
            summary['max_injection_envelope_error_increase'] = max(v['envelope_error_extra']-v['envelope_error_farm'] for v in validation)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    esc=lambda v:html.escape(str(v))
    parts=['<!doctype html><html lang="fr"><meta charset="utf-8"><title>Comparaison débruitage</title>',
           '<style>body{font:15px system-ui;margin:2em}img{max-width:100%}td,th{padding:.4em;border:1px solid #ccc}table{border-collapse:collapse}</style>',
           '<h1>FARM et régression optionnelle des raies</h1>',
           f'<p>{summary["runs"]} runs ; {len(rows)} canaux. <a href="comparison.csv">Tableau CSV</a>. Les courbes superposées partagent la même échelle et le même filtrage QC. La réduction spectrale ne prouve pas la préservation physiologique.</p>',
           '<p>Les fréquences sans support suffisant restent inchangées. Les métriques évaluent une correction supplémentaire, sans certifier l’origine physiologique du signal restant.</p>',
           '<table><tr><th>Run</th><th>Canal</th><th>Raies</th><th>Réduction pics FFT sélectionnés (dB)</th><th>RMS scanner avant/après (µV)</th></tr>']
    for r,image in links:
        value=r['selected_fft_peak_reduction_db']
        parts.append(f'<tr><td>{esc(r["run"])}</td><td>{esc(r["channel"])}</td><td>{r["selected_lines"]}</td><td>{value:.1f}</td><td>{r["locked_farm_uv"]:.2f} / {r["locked_extra_uv"]:.2f}</td></tr>' if value is not None else f'<tr><td>{esc(r["run"])}</td><td>{esc(r["channel"])}</td><td>0</td><td>Sans correction supplémentaire</td><td>{r["locked_farm_uv"]:.2f} / {r["locked_extra_uv"]:.2f}</td></tr>')
    parts.append('</table>')
    if validation:
        parts.append('<h2>Contrôles par injections indépendantes</h2><p>Quatre extraits de 24 volumes, amplitudes 5 et 20 µV, timing fixé à 82 ms, bande injectée 45–220 Hz. Comparaison D(brut + injection) − D(brut). Ces extraits ne valident pas tous les runs ni l’activité scanner-synchronisée. Des erreurs importantes existent déjà dans le cœur FARM pour certains signaux faibles ; une faible différence entre les deux colonnes ne les résout pas.</p>')
        parts.append('<table><tr><th>Run / canal</th><th>Injection (µV)</th><th>Gain FARM / extra</th><th>Erreur enveloppe FARM / extra</th></tr>')
        for v in validation:
            parts.append(f'<tr><td>{esc(v["run"])} {esc(v["channel"])}</td><td>{v["injection_rms_uv"]:g}</td><td>{v["gain_farm"]:.3f} / {v["gain_extra"]:.3f}</td><td>{v["envelope_error_farm"]:.3f} / {v["envelope_error_extra"]:.3f}</td></tr>')
        parts.append('</table>')
    for r,image in links:
        parts.append(f'<h2>{esc(r["run"])} — {esc(r["channel"])}</h2><img loading="lazy" src="{image}">')
    parts.append('</html>')
    (output/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('extra_root',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--validation-dir',type=Path)
    a=p.parse_args()
    compare(a.extra_root,a.output_dir,a.validation_dir)
