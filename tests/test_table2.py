from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from generate_table2 import (make_grid, read_manifest, retrieval_map, score_codes,
                             write_csv, generate, DATASETS, MODELS, LOSSES)


def test_full_grid():
    rows = list(make_grid(DATASETS, MODELS, LOSSES))
    assert len(rows) == 2376
    assert len({r['codes_file'] for r in rows}) == 2376


def test_map_known_ranking_and_empty_relevance():
    # Stable order has relevant hits at ranks 1 and 3: AP=(1+2/3)/2.
    q = np.array([[1, 1], [1, 1]])
    d = np.array([[1, 1], [1, -1], [-1, 1]])
    assert retrieval_map(q, d, np.array([0, 9]), np.array([0, 1, 0])) == pytest.approx(125 / 3)


def test_multilabel_and_64bit_identical_codes():
    q = np.ones((1, 64), dtype=np.int8)
    d = np.ones((3, 64), dtype=np.int8)
    assert retrieval_map(q, d, np.array([[1, 0]]),
                         np.array([[1, 1], [0, 1], [1, 0]])) == pytest.approx(250 / 3)


def test_invalid_codes_and_shape():
    with pytest.raises(ValueError, match='binary codes'):
        retrieval_map([[0, 1]], [[1, 1]], [0], [0])
    with pytest.raises(ValueError, match='Label dimensions'):
        retrieval_map([[1, 1]], [[1, 1]], [0, 1], [0])


def test_overlap_and_wrong_bit_length(tmp_path):
    path = tmp_path / 'codes.npz'
    arrays = dict(query_hash=np.ones((1, 16)), database_hash=np.ones((1, 16)),
                  query_labels=np.array([0]), database_labels=np.array([0]),
                  query_ids=np.array([1]), database_ids=np.array([1]))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='overlap'):
        score_codes(path, 16)
    arrays['database_ids'] = np.array([2])
    np.savez(path, **arrays)
    assert score_codes(path, 16)[0] == 100
    with pytest.raises(ValueError, match='Hash length'):
        score_codes(path, 32)


def test_render_bold_missing_and_na(tmp_path):
    rows = list(make_grid(['D'], ['Model'], ['CSQ']))
    for r in rows:
        r.update(protocol='same split', source='synthetic unit-test fixture',
                 feature_definition='test mode definition', codes_file='')
    rows[0]['map_pct'] = '70'
    rows[1]['map_pct'] = '80'
    rows[2]['map_pct'] = '90'
    rows[3]['map_pct'] = '90'
    rows[5].update(applicable='no', feature_definition='not supported')
    path = tmp_path / 'manifest.csv'
    write_csv(path, rows)
    args = Namespace(manifest=path, outdir=tmp_path / 'out', runner=None, strict=False)
    generate(args)
    md = (args.outdir / 'table2.md').read_text()
    assert '| 70.00 | **90.00** | -- | **80.00** | **90.00** | N/A |' in md
    assert '\\textbf{80.00}' in (args.outdir / 'table2.tex').read_text()
    args.strict = True
    with pytest.raises(SystemExit, match='Incomplete'):
        generate(args)


def test_duplicate_rows_and_mixed_protocols(tmp_path):
    path = tmp_path / 'manifest.csv'
    rows = list(make_grid(['D'], ['Model'], ['CSQ']))
    write_csv(path, rows + [rows[0]])
    with pytest.raises(ValueError, match='Duplicate'):
        read_manifest(path)
    for i, r in enumerate(rows):
        r.update(map_pct='50', source='test', feature_definition='test',
                 protocol=str(i), codes_file='')
    write_csv(path, rows)
    with pytest.raises(ValueError, match='mixed evaluation protocols'):
        generate(Namespace(manifest=path, outdir=tmp_path / 'out', runner=None, strict=False))
