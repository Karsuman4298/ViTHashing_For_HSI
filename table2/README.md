# Table II: CLS-only versus all-features retrieval

This implements the **retrieval Table 3 caption quoted in the conversation**, not
the semantic-segmentation Table 3 in the supplied Agent Attention PDF.

Columns are:

| Backbone | Hashing loss | CLS 16 bits | CLS 32 bits | CLS 64 bits | All 16 bits | All 32 bits | All 64 bits |
|---|---|---|---|---|---|---|---|

There is a section for each dataset. Entries are mAP percentages. Bold compares
CLS versus all features **within the same dataset, backbone, loss and bit length**;
both are bold if their two-decimal displayed values tie. A lone measured value is
not bold until its comparison is available. Missing results are `--`, never zero.

Default coverage matches your Table I: Houston2013, Houston2018, Trento,
NiliFossae; SSFTT, Mamba, MoE-Mamba, SSRN, A2S2KResNet, ContextualNet, CNN-2D,
CNN-3D, HybridSN, MorphFormer, SpectralFormer; CSQ, DPN, DSH, GreedyHash,
HashNet, IDHN, OrthoHash, DSPCH, DHNN. This is 2,376 measurements, displayed as
396 rows with six scores each. No new numerical results are available in the
supplied paper, so the generated initial table is an **unmeasured template**.

## Generate the table

Run from the repository root. Only NumPy is needed for the generator:

```bash
python3 -m pip install numpy
# Already created in this delivery. Run init only for a NEW manifest:
python3 generate_table2.py init --manifest table2/new_experiments.csv

# Edit table2/experiments.csv with your measured results, then:
python3 generate_table2.py generate

# Before submission, require every applicable experiment to be present:
python3 generate_table2.py generate --strict
```

Outputs in `table2/output/`:

- `table2_results.csv`: all scores and their provenance, retaining numerical precision.
- `table2.md`: readable table with two-decimal scores and bold comparisons.
- `table2.tex`: complete multipage LaTeX table.
- `table2_standalone.tex`: wrapper that numbers the table **II**.
- `missing_measurements.txt`: every absent score/experiment.

To compile a PDF with an installed LaTeX distribution:

```bash
cd table2/output
pdflatex table2_standalone.tex
pdflatex table2_standalone.tex
```

The complete table is too long for one IEEE two-column float. The standalone
version uses `longtable`; place it in a one-column appendix/supplement or split
it into smaller floats for the final IEEE manuscript. Do not put `longtable`
inside `table*`. `table2.tex` requires `booktabs` and `longtable`.

For only VTS16/VTS32, or any other explicitly selected experiment suite:

```bash
python3 generate_table2.py init --manifest table2/vts.csv \
  --models VTS32 VTS16 --losses DSH HashNet GreedyHash IDHN CSQ DPN
python3 generate_table2.py generate --manifest table2/vts.csv --outdir table2/vts_output
```

Names select table rows; they do not implement an architecture. In VTS16/VTS32,
16/32 refer to the architecture's patch configuration, **not** its hash length.

## Option A: import measured mAP

Fill these fields in each CSV row:

- `map_pct`: percentage, e.g. `81.28` rather than `0.8128`.
- `feature_definition`: exact feature vector used by that trained hashing head.
- `protocol`: shared evaluation identifier within each dataset, documenting split,
  preprocessing, database/query membership/order, ranking cutoff, tie handling,
  checkpoint selection and training seed/aggregation rule.
- `source`: result file, experiment ID or checkpoint/log reference.

Leave `codes_file` blank if importing only an existing score. Do not reuse your
Table I numbers as CLS or all-features numbers unless the original experiments
explicitly establish the feature mode. Do not derive one mode's score from the other.

You can import mAP@K from an existing pipeline only if all compared results use
that same cutoff and its AP convention, clearly named in `protocol`. The built-in
code evaluator below implements **full-database mAP**, not mAP@K.

## Option B: calculate mAP from saved codes

The CSV already specifies one NPZ path for each experiment; paths are relative
to the manifest directory. After evaluating each independently trained model:

```python
from pathlib import Path
import numpy as np

path = Path('table2/codes/Houston2013/SSFTT/CSQ/16/cls.npz')
path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    path,
    query_hash=np.where(query_logits >= 0, 1, -1).astype(np.int8),
    database_hash=np.where(database_logits >= 0, 1, -1).astype(np.int8),
    query_labels=query_labels,       # [Nq] class IDs, or [Nq, C] binary multi-hot
    database_labels=database_labels, # same label format/class ordering
    query_ids=query_sample_ids,      # unique IDs in a shared dataset namespace
    database_ids=database_sample_ids,
)
```

Set `feature_definition` and `protocol` in the CSV. The script computes mAP,
uses the NPZ as the default source, and checks a supplied score against the codes
if both are present. All exported code experiments in a dataset must use identical
query/database IDs, ordering and labels. Query/database overlap is rejected.

The evaluator ranks by Hamming distance, uses database order to break equal-distance
ties deterministically, defines relevance as equal class (single-label) or any
shared positive class (multi-label), and averages precision at every relevant rank.
A query with no relevant database item contributes zero. Returned scores are
multiplied by 100. It never evaluates float logits as if they were binary codes.

This stable tie rule may differ from an older implementation's `np.argsort`
default. Re-evaluate both modes using the same rule; do not mix protocols.

## Training/evaluating the whole experiment grid

The repository does **not** contain the complete 11-backbone/nine-loss experiment
suite used by Table I. It contains a VTS HSI implementation and an HSI trainer for
the repository's CSQ-style and DPN-style losses. These must not be silently
substituted for SSFTT, Mamba, CNNs or the missing loss implementations.

The generator accepts an optional runner from your actual experiment suite:

```bash
python3 generate_table2.py generate --runner my_experiments:run_one --strict
```

The callable receives `(row: dict, manifest_directory: pathlib.Path)` and runs
one missing experiment. It must return a dict with any of `map_pct`, `codes_file`,
`source`, `protocol`, `feature_definition`; it must not change the experiment key.
For example, in a local `my_experiments.py`:

```python
def run_one(row, manifest_directory):
    # Use your REAL model factory and training/evaluation code here.
    # row keys: dataset, backbone, loss, bits, features ('cls' or 'all').
    # Train a separate hashing head/model for each feature mode and bit length.
    # Save NPZ arrays as above to manifest_directory / row['codes_file'].
    # Return provenance after training and evaluation have completed.
    raise NotImplementedError('Connect the experiment suite that produced Table I')
```

With that adapter, one command executes missing experiments and generates the
whole table. The provided generator itself does not implement or train missing
backbones/losses. It skips rows with an existing score or saved code file.

## Define feature modes before comparison

For a transformer returning `[batch, tokens, channels]` with token zero as CLS:

```python
cls_features = encoded[:, 0, :]
all_features = encoded.reshape(encoded.shape[0], -1)  # CLS plus every patch token
```

Use a separately trained compatible hash head for each mode. Changing the input
dimension at evaluation time is not a valid ablation. Keep splits, preprocessing,
training budget, checkpoint-selection rule and loss settings matched. The local
`VTSHSIModel` already uses these definitions through `use_all_tokens`.

For CNNs and models without a native CLS token, **do not call global pooling a CLS
token**. Either define and disclose a separate pooled-versus-all comparison with
appropriate column names in your manuscript, or set `applicable=no` and describe
why in `feature_definition`. N/A rows cannot have a numeric score. Leaving the
initial row as `yes` is only a request for a measurement, not evidence that that
backbone supports CLS. Verify each implementation before running it.

Full new scores require your actual paired experiment outputs or the complete
training suite. The attached Table I alone cannot provide those missing results.
