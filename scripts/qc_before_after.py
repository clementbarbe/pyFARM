"""Read-only, paired BrainVision QC; run directly from any directory."""

import argparse
import csv
import html
import json
import os
from pathlib import Path
import sys

# Also works without an editable installation of pyFARM.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np

from farm import __version__
from farm.diagnostics.acquisition import assess_adc_rails
from farm.diagnostics.metrics import compute_gradient_harmonic_reduction
from farm.export.metadata import _jsonable
from farm.preprocessing.filters import apply_bandpass


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def db_ratio(numerator, denominator, power=False):
    # Undefined ratios remain missing, rather than being turned into huge scores.
    if numerator <= 0 or denominator <= 0:
        return None
    return float((10 if power else 20) * np.log10(numerator / denominator))


def locked_metrics(x, fs, onsets, tr):
    """Separate a median TR-locked waveform and its asynchronous residual.

    The latter includes physiological activity AND asynchronous noise. Their
    ratio is a reproducible QC proxy, never a physiological SNR estimate.
    """
    width = int(round(fs * tr))
    if width < 2:
        return dict(locked_epochs=0, locked_rms_uv=None,
                    asynchronous_rms_uv=None, snr_proxy_db=None)
    epochs = np.asarray([x[int(s):int(s)+width] for s in onsets
                         if s >= 0 and int(s)+width <= len(x)], dtype=np.float32)
    if len(epochs) < 3:
        return dict(locked_epochs=len(epochs), locked_rms_uv=None,
                    asynchronous_rms_uv=None, snr_proxy_db=None)
    template = np.median(epochs, axis=0)
    locked = rms(template)
    epochs -= template
    asynchronous = rms(epochs)
    return dict(locked_epochs=len(epochs), locked_rms_uv=locked*1e6,
                asynchronous_rms_uv=asynchronous*1e6,
                snr_proxy_db=db_ratio(asynchronous, locked))


def channel_metrics(x, fs, onsets, tr, band):
    if len(x) < 64 or not np.isfinite(x).all():
        raise ValueError("Signal too short (<64 samples) or contains NaN/Inf")
    filtered = apply_bandpass(x, fs, band)
    result = dict(min_uv=float(x.min())*1e6, max_uv=float(x.max())*1e6,
                  rms_uv=rms(x)*1e6,
                  band_min_uv=float(filtered.min())*1e6,
                  band_max_uv=float(filtered.max())*1e6,
                  band_rms_uv=rms(filtered)*1e6,
                  zero_fraction=float(np.mean(x == 0)))
    result.update(locked_metrics(filtered, fs, onsets, tr))
    return result


def resolve_input(metadata, raw_index):
    original = Path(metadata["input_vhdr"])
    if raw_index is not None:
        candidates = raw_index.get(original.name, [])
        if len(candidates) != 1:
            raise ValueError(f"--raw-root: {len(candidates)} candidates for {original.name}")
        return candidates[0]
    return original


def read_pair(path, band, raw_index=None):
    m = json.loads(path.read_text())
    source = resolve_input(m, raw_index)
    # Prefer the file next to its sidecar, so moved output directories work.
    cleaned = path.with_name(path.name.removesuffix("_denoise.json") + ".vhdr")
    if not cleaned.exists():
        cleaned = Path(m["output_vhdr"])
    before = mne.io.read_raw_brainvision(source, preload=False, verbose=False)
    after = mne.io.read_raw_brainvision(cleaned, preload=False, verbose=False)
    try:
        fs = float(before.info["sfreq"])
        if fs != float(after.info["sfreq"]):
            raise ValueError("Before/after sampling rates differ; cannot align samples")
        if before.n_times != after.n_times:
            raise ValueError(f"Before/after file lengths differ: {before.n_times}/{after.n_times}")
        start, stop = int(m["scan_start_sample"]), int(m["scan_stop_sample_exclusive"])
        if not 0 <= start < stop <= min(before.n_times, after.n_times):
            raise ValueError("Scan crop outside before/after data")
        if not 0 < band[0] < band[1] < fs / 2:
            raise ValueError("QC band must be between 0 and Nyquist")
        onsets = np.asarray(m["volume_onsets_samples_crop"], dtype=np.int64)
        tr = float(m["tr_seconds"])
        rows, errors = [], []
        for name in m["selected_channels"]:
            try:
                ib, ia = before.ch_names.index(name), after.ch_names.index(name)
                xb = before.get_data(picks=[ib], start=start, stop=stop)[0]
                xa = after.get_data(picks=[ia], start=start, stop=stop)[0]
                b = channel_metrics(xb, fs, onsets, tr, band)
                a = channel_metrics(xa, fs, onsets, tr, band)
                rail = assess_adc_rails(xb[None], fs, [before.info["chs"][ib]],
                                        [name], onsets, source)[0]
                harmonic = compute_gradient_harmonic_reduction(
                    xb[None], xa[None], fs, [name], float(m["sdur_seconds"]), band)[0]
                flags = []
                if rail.get("rail_samples", 0):
                    flags.append("source_at_ADC_rails")
                if not rail["assessed"]:
                    flags.append("ADC_limits_unassessed")
                if a["band_rms_uv"] == 0:
                    flags.append("zero_output")
                if a["band_rms_uv"] > b["band_rms_uv"]:
                    flags.append("band_RMS_increased")
                if (a["locked_rms_uv"] is not None and b["locked_rms_uv"] is not None
                        and a["locked_rms_uv"] >= b["locked_rms_uv"]):
                    flags.append("scanner_locked_RMS_not_reduced")
                if a["locked_epochs"] < 3:
                    flags.append("insufficient_complete_volumes")
                if m.get("masked_fraction_of_scan", 0) > 0:
                    flags.append("intentional_zero_mask_present")
                if m.get("timing_consensus_quality", {}).get("unresolved_disagreement"):
                    flags.append("timing_disagreement")
                row = dict(run=source.stem, channel=name, source_vhdr=str(source),
                           output_vhdr=str(cleaned), sidecar=str(path),
                           software_version=m.get("software_version", "unknown"),
                           science_mode=m.get("science_mode", "unknown"),
                           timing_mode=m.get("timing_mode", "unknown"),
                           file_duration_before_s=before.n_times/fs,
                           file_duration_after_s=after.n_times/fs,
                           analyzed_duration_s=(stop-start)/fs,
                           complete_volumes=a["locked_epochs"],
                           qc_low_hz=band[0], qc_high_hz=band[1],
                           source_rail_percent=(100*rail["rail_fraction"]
                                                if rail["assessed"] else None),
                           source_longest_rail_s=rail.get("longest_rail_seconds"),
                           harmonic_reduction_db=db_ratio(harmonic["excess_power_before"],
                                                         harmonic["excess_power_after"], power=True),
                           flags="; ".join(flags))
                row.update({f"{k}_before": v for k, v in b.items()})
                row.update({f"{k}_after": v for k, v in a.items()})
                row["snr_proxy_gain_db"] = (a["snr_proxy_db"]-b["snr_proxy_db"]
                                             if a["snr_proxy_db"] is not None
                                             and b["snr_proxy_db"] is not None else None)
                row["locked_reduction_db"] = (db_ratio(b["locked_rms_uv"], a["locked_rms_uv"])
                                               if a["locked_rms_uv"] is not None
                                               and b["locked_rms_uv"] is not None else None)
                rows.append(row)
            except Exception as exc:
                errors.append(dict(sidecar=str(path), channel=name,
                                   error=f"{type(exc).__name__}: {exc}"))
        return rows, errors
    finally:
        before.close()
        after.close()


def plot_pages(rows, output_dir, page_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = []
    for channel in sorted({r["channel"] for r in rows}):
        selected = [r for r in rows if r["channel"] == channel]
        for offset in range(0, len(selected), page_size):
            batch = selected[offset:offset+page_size]
            x = np.arange(len(batch))
            fig, axes = plt.subplots(3, 2, figsize=(max(13, len(batch)*.35), 12),
                                     constrained_layout=True)
            band = f'{batch[0]["qc_low_hz"]:g}–{batch[0]["qc_high_hz"]:g} Hz'
            def values(key):
                return np.asarray([np.nan if r.get(key) is None else r[key] for r in batch])
            def paired(ax, key, title, unit, log=False):
                ax.plot(x, values(key+"_before"), "o-", ms=3, color="#cf5c5c", label="Avant")
                ax.plot(x, values(key+"_after"), "o-", ms=3, color="#238b45", label="Après")
                ax.set(title=title, ylabel=unit)
                if log and np.any(np.r_[values(key+"_before"), values(key+"_after")] > 0):
                    ax.set_yscale("log")
                ax.legend(fontsize=8)
            ax = axes[0, 0]
            ax.plot(x, values("file_duration_before_s"), "o", label="Fichier avant", color="#cf5c5c")
            ax.plot(x, values("file_duration_after_s"), "+", label="Fichier après", color="#238b45")
            ax.plot(x, values("analyzed_duration_s"), ".--", label="Portion analysée", color="#555555")
            ax.set(title="Durées", ylabel="s")
            ax.legend(fontsize=8)
            paired(axes[0, 1], "snr_proxy_db", "Proxy SNR : asynchrone / synchronisé scanner", "dB")
            paired(axes[1, 0], "band_rms_uv", f"RMS ({band})", "µV", log=True)
            ax = axes[1, 1]
            for stage, color, label in [("before", "#cf5c5c", "Avant"), ("after", "#238b45", "Après")]:
                ax.plot(x, values("band_max_uv_"+stage), "o-", ms=3, color=color, label=label+" max")
                ax.plot(x, values("band_min_uv_"+stage), "v--", ms=3, color=color, label=label+" min")
            ax.set(title=f"Minimum / maximum ({band})", ylabel="µV")
            ax.legend(fontsize=8, ncol=2)
            paired(axes[2, 0], "locked_rms_uv", f"Résidu synchronisé au scanner ({band})", "µV", log=True)
            ax = axes[2, 1]
            ax.bar(x, values("source_rail_percent"), color="#d97a22")
            ax.set(title="Échantillons bruts aux limites ADC", ylabel="%")
            ax.set_ylim(bottom=0, top=max(1, np.nanmax(np.r_[0., values("source_rail_percent")])*1.1))
            for ax in axes.flat:
                ax.set_xticks(x, [r["run"] for r in batch], rotation=90, fontsize=7)
                ax.grid(alpha=.2)
                for i, row in enumerate(batch):
                    if row["flags"]:
                        ax.get_xticklabels()[i].set_color("#ad2222")
            fig.suptitle(f"QC avant / après — {channel} — page {offset//page_size+1}\n"
                         "Étiquette rouge : alerte à examiner ; absence d’alerte ≠ EMG validé", fontsize=13)
            # Do not use arbitrary channel names as filesystem paths.
            filename = f"qc_channel_{sorted({r['channel'] for r in rows}).index(channel)+1}_{offset//page_size+1:02d}.png"
            fig.savefig(output_dir/filename, dpi=130)
            plt.close(fig)
            plots.append((channel, filename))
    return plots


def write_report(rows, errors, paths, output_dir, band, page_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _jsonable(rows)
    with (output_dir/"metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["run", "channel", "flags"])
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(qc_software_version=__version__, sidecars_found=len(paths),
                   channels_analyzed=len(rows), channels_with_alerts=sum(bool(r["flags"]) for r in rows),
                   qc_band_hz=list(band), errors=errors,
                   snr_definition="20 log10(RMS asynchronous residual / RMS median TR-locked waveform); not physiological SNR",
                   interpretation="No automatic acceptance. Less scanner-locked energy does not prove EMG preservation.")
    (output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    plots = plot_pages(rows, output_dir, page_size)
    esc = lambda value: html.escape(str(value))
    parts = ["<!doctype html><html lang='fr'><meta charset='utf-8'><title>QC EMG avant / après</title>",
             "<style>body{font:15px system-ui;margin:2em}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.4em}th{background:#eee}.alert{background:#fff0e5}</style>",
             "<h1>QC EMG avant / après</h1>",
             f"<p>{len(paths)} sidecars ; {len(rows)} canaux analysés ; {len(errors)} erreurs. "
             "<a href='metrics.csv'>Tableau CSV complet</a> · <a href='summary.json'>Résumé et erreurs</a></p>",
             f"<p>Comparaison sur les mêmes échantillons, même bande {band[0]:g}–{band[1]:g} Hz pour RMS, extrema des figures et métriques scanner. "
             "Les extrema sans filtre sont aussi dans le CSV (min_uv / max_uv). Les durées utilisent N / fréquence d’échantillonnage.</p>",
             "<p><b>Proxy SNR</b> = 20 log10(RMS de la composante asynchrone / RMS de la médiane synchronisée aux volumes). "
             "Une valeur plus haute indique moins de contamination synchronisée relativement au reste du signal. "
             "La composante asynchrone contient aussi du bruit ; une réponse musculaire synchronisée peut contribuer à la médiane. "
             "Ce score ne mesure ni le vrai SNR physiologique ni la préservation de l’EMG. Moins de trois volumes ou un dénominateur nul : valeur absente.</p>",
             "<p>Les alertes signalent des observations à examiner, sans seuil automatique d’exclusion : limites ADC dans le brut, "
             "RMS augmentée, composante scanner non réduite, sortie nulle, masque de zéros ou désaccord de timing. "
             "Les limites ADC inconnues restent signalées ; leur absence ne prouve pas l’absence de saturation analogique. "
             "Les échantillons masqués par le pipeline sont inclus et peuvent améliorer artificiellement les métriques. "
             "Aucun canal n’est déclaré « bon » automatiquement.</p>"]
    for channel, filename in plots:
        parts.append(f"<h2>{esc(channel)}</h2><a href='{filename}'><img src='{filename}'></a>")
    parts.append("<h2>Observations par run et canal</h2>"
                 "<p><input id='search' placeholder='Chercher un sujet, run ou canal'> "
                 "<label><input type='checkbox' id='alerts'> Afficher seulement les alertes</label></p>"
                 "<table><thead><tr><th>Run</th><th>Canal</th><th>Durée analysée (s)</th>"
                 "<th>Proxy SNR avant (dB)</th><th>Proxy SNR après (dB)</th>"
                 "<th>Résidu scanner après (µV)</th><th>Limites ADC (%)</th>"
                 "<th>Observations</th></tr></thead><tbody id='metrics'>")
    for r in rows:
        cls = " class='alert'" if r["flags"] else ""
        cells = "".join(f"<td>{r[k]:.2f}</td>" if r[k] is not None else "<td>—</td>"
                        for k in ["analyzed_duration_s", "snr_proxy_db_before", "snr_proxy_db_after",
                                  "locked_rms_uv_after", "source_rail_percent"])
        parts.append(f"<tr{cls}><td>{esc(r['run'])}</td><td>{esc(r['channel'])}</td>"
                     f"{cells}<td>{esc(r['flags'] or 'Pas d’alerte automatique — à valider')}</td></tr>")
    parts.append("</tbody></table><h2>Erreurs de lecture ou de comparaison</h2><ul>")
    for error in errors:
        parts.append(f"<li>{esc(error)}</li>")
    parts.append("""</ul><script>
const search = document.getElementById('search');
const alerts = document.getElementById('alerts');
function filterRows() {
  for (const row of document.querySelectorAll('#metrics tr')) {
    row.hidden = !row.textContent.toLowerCase().includes(search.value.toLowerCase())
      || (alerts.checked && !row.classList.contains('alert'));
  }
}
search.addEventListener('input', filterRows);
alerts.addEventListener('change', filterRows);
</script></html>""")
    (output_dir/"report.html").write_text("\n".join(parts), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Dossier de sorties FARM, parcouru récursivement")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, help="Retrouver les bruts déplacés par nom unique de .vhdr")
    parser.add_argument("--band", type=float, nargs=2, default=(30., 250.), metavar=("LOW", "HIGH"))
    parser.add_argument("--page-size", type=int, default=30, help="Runs par figure (défaut 30)")
    parser.add_argument("--limit", type=int, help="Limiter le nombre de runs pour un essai")
    args = parser.parse_args()
    if args.page_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("page-size et limit doivent être positifs")
    if not 0 < args.band[0] < args.band[1]:
        parser.error("La bande doit vérifier 0 < LOW < HIGH")
    paths = sorted(args.root.resolve().rglob("*_denoise.json"))
    if not paths:
        parser.error("Aucun fichier *_denoise.json trouvé")
    if args.limit:
        paths = paths[:args.limit]
    raw_index = None
    if args.raw_root is not None:
        raw_index = {}
        for path in args.raw_root.resolve().rglob("*.vhdr"):
            raw_index.setdefault(path.name, []).append(path)
    rows, errors = [], []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name}", flush=True)
        try:
            new_rows, new_errors = read_pair(path, tuple(args.band), raw_index)
            rows.extend(new_rows)
            errors.extend(new_errors)
            for error in new_errors:
                print(f"  ERREUR: {error}", flush=True)
        except Exception as exc:
            error = dict(sidecar=str(path), error=f"{type(exc).__name__}: {exc}")
            errors.append(error)
            print(f"  ERREUR: {error}", flush=True)
    summary = write_report(rows, errors, paths, args.output_dir.resolve(), tuple(args.band), args.page_size)
    print(json.dumps(summary, indent=2))
    print(f"Rapport : {args.output_dir.resolve()/'report.html'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
