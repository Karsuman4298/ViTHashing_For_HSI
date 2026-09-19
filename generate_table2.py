#!/usr/bin/env python3
"""CLS-only versus all-features retrieval mAP (%) table. See table2/README.md."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np

DATASETS = ['Houston2013', 'Houston2018', 'Trento', 'NiliFossae']
MODELS = ['SSFTT', 'Mamba', 'MoE-Mamba', 'SSRN', 'A2S2KResNet',
          'ContextualNet', 'CNN-2D', 'CNN-3D', 'HybridSN', 'MorphFormer', 'SpectralFormer']
LOSSES = ['CSQ', 'DPN', 'DSH', 'GreedyHash', 'HashNet', 'IDHN', 'OrthoHash', 'DSPCH', 'DHNN']
BITS = [16, 32, 64]
MODES = ['cls', 'all']
KEYS = ['dataset', 'backbone', 'loss', 'bits', 'features']
FIELDS = KEYS + ['applicable', 'feature_definition', 'protocol', 'map_pct', 'codes_file', 'source']


def write_csv(path, rows, fields=FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_grid(datasets, models, losses):
    for dataset in datasets:
        for model in models:
            for loss in losses:
                for bits in BITS:
                    for mode in MODES:
                        row = dict.fromkeys(FIELDS, '')
                        row.update(dataset=dataset, backbone=model, loss=loss,
                                   bits=str(bits), features=mode, applicable='yes')
                        row['codes_file'] = f'codes/{dataset}/{model}/{loss}/{bits}/{mode}.npz'
                        yield row


def read_manifest(path):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest must contain columns: {FIELDS}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty manifest')
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        if key in seen:
            raise ValueError(f'Duplicate experiment: {key}')
        seen.add(key)
        if (any(not row[k] for k in KEYS) or row['bits'] not in {'16', '32', '64'}
                or row['features'] not in MODES or row['applicable'] not in {'yes', 'no'}):
            raise ValueError(f'Invalid experiment: {key}')
    return rows


def retrieval_map(query_hash, database_hash, query_labels, database_labels):
    """Full-database mAP; stable Hamming ties; queries with no relevant item score 0.

    Labels are 1D class IDs or 2D binary multi-hot arrays (any shared label is
    relevant). Codes must be -1/+1. Query/database sets must already be disjoint.
    """
    q, d, ql, dl = map(np.asarray, (query_hash, database_hash, query_labels, database_labels))
    if (q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1]
            or min(q.shape) == 0 or len(d) == 0):
        raise ValueError('Codes must have nonempty shapes [queries, bits] and [database, bits]')
    if not np.isin(q, [-1, 1]).all() or not np.isin(d, [-1, 1]).all():
        raise ValueError('Export binary codes in {-1,+1}; use where(logits >= 0, 1, -1)')
    if ql.ndim not in (1, 2) or dl.ndim != ql.ndim or len(ql) != len(q) or len(dl) != len(d):
        raise ValueError('Label dimensions/counts do not match codes')
    if not (np.issubdtype(ql.dtype, np.number) and np.issubdtype(dl.dtype, np.number)
            and np.isfinite(ql).all() and np.isfinite(dl).all()):
        raise ValueError('Labels must be finite numeric values')
    if ql.ndim == 2 and (ql.shape[1] != dl.shape[1] or ql.shape[1] == 0
                         or not np.isin(ql, [0, 1]).all() or not np.isin(dl, [0, 1]).all()):
        raise ValueError('Multi-label arrays must be binary and have the same class dimension')
    aps = []
    for code, label in zip(q, ql):
        # count_nonzero avoids int8 dot-product overflow and only uses O(database) memory.
        order = np.argsort(np.count_nonzero(d != code, axis=1), kind='stable')
        relevant = dl == label if ql.ndim == 1 else np.any((dl > 0) & (label > 0), axis=1)
        hits = np.flatnonzero(relevant[order]) + 1
        aps.append(float(np.mean(np.arange(1, len(hits) + 1) / hits)) if len(hits) else 0.0)
    return float(np.mean(aps) * 100.0)


def score_codes(path, bits):
    with np.load(path, allow_pickle=False) as data:
        names = ['query_hash', 'database_hash', 'query_labels', 'database_labels',
                 'query_ids', 'database_ids']
        if not set(names).issubset(data.files):
            raise ValueError(f'{path}: NPZ requires {names}')
        qid, did = data['query_ids'], data['database_ids']
        for ids, hashes in [(qid, data['query_hash']), (did, data['database_hash'])]:
            if ids.ndim != 1 or len(ids) != len(hashes) or len(np.unique(ids)) != len(ids):
                raise ValueError('Sample IDs must be unique 1D arrays matching code counts')
        if np.intersect1d(qid, did).size:
            raise ValueError('Query/database overlap: export disjoint retrieval sets')
        if data['query_hash'].ndim != 2 or data['query_hash'].shape[1] != bits:
            raise ValueError('Hash length differs from manifest')
        score = retrieval_map(*(data[name] for name in names[:4]))
        # This fingerprint also checks label/order consistency across compared experiments.
        import hashlib
        digest = hashlib.sha256()
        for name in ['query_ids', 'database_ids', 'query_labels', 'database_labels']:
            digest.update(json.dumps(data[name].tolist(), separators=(',', ':')).encode())
        return score, digest.hexdigest()


def numeric_score(value):
    if value == '':
        return None
    score = float(value)
    if not np.isfinite(score) or not 0 <= score <= 100:
        raise ValueError(f'map_pct must be a finite percentage in [0,100]: {value}')
    return score


def escape(s):
    special = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(special.get(c, c) for c in str(s))


def dimensions(rows):
    return [list(dict.fromkeys(r[k] for r in rows)) for k in KEYS[:3]]


def render(rows, outdir):
    datasets, models, losses = dimensions(rows)
    idx = {tuple(r[k] for k in KEYS): r for r in rows}
    caption = ('Retrieval mAP (\\%) using CLS-only and all features at 16, 32, and 64 bits. '
               'Bold indicates the better feature mode for the same dataset, backbone, loss and bit length '
               '(ties at the displayed precision are bold). -- denotes an unmeasured result; '
               'N/A denotes an explicitly inapplicable mode.')
    tex = [r'\small', r'\setlength{\tabcolsep}{6pt}', r'\begin{longtable}{llrrrrrr}',
           r'\caption{' + caption + r'}\label{tab:cls-all-hashing}\\', r'\toprule',
           r' & & \multicolumn{3}{c}{CLS-only features} & \multicolumn{3}{c}{All features}\\',
           r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
           r'Backbone & Loss & 16 bits & 32 bits & 64 bits & 16 bits & 32 bits & 64 bits\\',
           r'\midrule', r'\endfirsthead',
           r'\multicolumn{8}{c}{\tablename\ \thetable{} -- continued}\\', r'\toprule',
           r' & & \multicolumn{3}{c}{CLS-only features} & \multicolumn{3}{c}{All features}\\',
           r'Backbone & Loss & 16 bits & 32 bits & 64 bits & 16 bits & 32 bits & 64 bits\\',
           r'\midrule', r'\endhead', r'\midrule',
           r'\multicolumn{8}{r}{Continued on next page}\\', r'\endfoot',
           r'\bottomrule', r'\endlastfoot']
    md = ['# Table II: CLS-only versus all-features retrieval', '',
          'mAP (%). Bold compares the two feature modes at each bit length; ties are bold. '
          '`--` = unmeasured; `N/A` = explicitly inapplicable.', '']
    for dataset in datasets:
        tex += [r'\multicolumn{8}{c}{\textbf{' + escape(dataset) + r'}}\\*', r'\midrule']
        md += [f'## {dataset}', '', '| Backbone | Loss | CLS 16 | CLS 32 | CLS 64 | All 16 | All 32 | All 64 |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
        for model in models:
            for loss in losses:
                tc, mc = [escape(model), escape(loss)], [model, loss]
                for mode in MODES:
                    for bits in BITS:
                        row = idx.get((dataset, model, loss, str(bits), mode), {})
                        pair = [idx.get((dataset, model, loss, str(bits), m), {}) for m in MODES]
                        val = numeric_score(row.get('map_pct', ''))
                        both = all(p.get('applicable') == 'yes' and p.get('map_pct', '') != '' for p in pair)
                        bold = both and val is not None and round(val, 2) == max(round(float(p['map_pct']), 2) for p in pair)
                        display = 'N/A' if row.get('applicable') == 'no' else ('--' if val is None else f'{val:.2f}')
                        tc.append(r'\textbf{' + display + '}' if bold else display)
                        mc.append('**' + display + '**' if bold else display)
                tex.append(' & '.join(tc) + r' \\')
                md.append('| ' + ' | '.join(mc) + ' |')
            tex.append(r'\addlinespace')
        md.append('')
    tex.append(r'\end{longtable}')
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'table2.tex').write_text('\n'.join(tex) + '\n')
    (outdir / 'table2.md').write_text('\n'.join(md).rstrip() + '\n')
    (outdir / 'table2_standalone.tex').write_text('\n'.join([
        r'\documentclass{article}', r'\usepackage[a4paper,margin=15mm]{geometry}',
        r'\usepackage{booktabs,longtable}', r'\renewcommand{\thetable}{\Roman{table}}',
        r'\begin{document}', r'\setcounter{table}{1}', r'\input{table2.tex}', r'\end{document}', '']))


def generate(args):
    manifest = args.manifest.resolve()
    rows = read_manifest(manifest)
    runner = None
    if args.runner:
        module, function = args.runner.split(':', 1)
        runner = getattr(importlib.import_module(module), function)
    problems, protocols, fingerprints = [], {}, {}
    for row in rows:
        key = '/'.join(row[k] for k in KEYS)
        if row['applicable'] == 'no':
            if row['map_pct'] or not row['feature_definition']:
                raise ValueError(f'{key}: N/A requires an explanation and no score')
            continue
        path = manifest.parent / row['codes_file'] if row['codes_file'] else None
        score = numeric_score(row['map_pct'])
        if score is None and (path is None or not path.is_file()) and runner:
            print(f'Running {key}', flush=True)
            result = runner(dict(row), manifest.parent)
            if not isinstance(result, dict) or set(result) - {'map_pct', 'codes_file', 'source', 'protocol', 'feature_definition'}:
                raise ValueError('Runner must return a dict of result/provenance fields; see README')
            row.update(result)
            path = manifest.parent / row['codes_file'] if row['codes_file'] else None
            score = numeric_score(row['map_pct'])
        if path is not None and path.is_file():
            measured, fingerprint = score_codes(path, int(row['bits']))
            previous = fingerprints.setdefault(row['dataset'], fingerprint)
            if previous != fingerprint:
                raise ValueError(f'{key}: query/database IDs, order or labels differ across experiments')
            if score is not None and abs(score - measured) > 0.00501:
                raise ValueError(f'{key}: supplied mAP disagrees with saved codes')
            score = measured
            row['source'] = row['source'] or str(path)
        if score is not None:
            if not all(row[k] for k in ['protocol', 'source', 'feature_definition']):
                raise ValueError(f'{key}: a score requires protocol, source and feature_definition')
            previous = protocols.setdefault(row['dataset'], row['protocol'])
            if previous != row['protocol']:
                raise ValueError(f'{key}: mixed evaluation protocols within a dataset')
            row['map_pct'] = f'{score:.10f}'
        else:
            problems.append(f'{key}: missing measurement')
    # A removed row must not silently make a table appear complete.
    datasets, models, losses = dimensions(rows)
    present = {tuple(r[k] for k in KEYS) for r in rows}
    for row in make_grid(datasets, models, losses):
        if tuple(row[k] for k in KEYS) not in present:
            problems.append('/'.join(row[k] for k in KEYS) + ': absent manifest row')
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / 'table2_results.csv', rows)
    (args.outdir / 'missing_measurements.txt').write_text('\n'.join(problems) + '\n')
    render(rows, args.outdir)
    measured = sum(r['map_pct'] != '' for r in rows)
    print(f'Wrote {args.outdir}: {measured} measured scores; {len(problems)} missing experiments.')
    if args.strict and problems:
        raise SystemExit('Incomplete table: see missing_measurements.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='Create the full experiment manifest without inventing scores')
    init.add_argument('--manifest', type=Path, default=Path('table2/experiments.csv'))
    init.add_argument('--datasets', nargs='+', default=DATASETS)
    init.add_argument('--models', nargs='+', default=MODELS)
    init.add_argument('--losses', nargs='+', default=LOSSES)
    run = sub.add_parser('generate', help='Compute/import mAP and produce the complete table')
    run.add_argument('--manifest', type=Path, default=Path('table2/experiments.csv'))
    run.add_argument('--outdir', type=Path, default=Path('table2/output'))
    run.add_argument('--runner', help='Optional module:function that trains/evaluates each missing experiment')
    run.add_argument('--strict', action='store_true', help='Return an error if any applicable score is missing')
    args = parser.parse_args()
    if args.command == 'init':
        if args.manifest.exists():
            parser.error(f'Refusing to overwrite {args.manifest}')
        rows = list(make_grid(args.datasets, args.models, args.losses))
        write_csv(args.manifest, rows)
        print(f'Created {args.manifest}: {len(rows)} experiments')
    else:
        generate(args)


if __name__ == '__main__':
    main()
