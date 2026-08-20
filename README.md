# slm-125m

A from-scratch, 125M-parameter legal/financial small language model: LLaMA
architecture (12L/768d/12h, 16K byte-level BPE vocab, 1024 context, tied
embeddings), pretrained on a cleaned, deduplicated, decontaminated corpus built
from US case law, SEC filings, and a small educational-web slice. Runs on
[Modal](https://modal.com); see `Replication Guide.md` for the full data-mix
rationale and the exact commands for each phase.

## Pipeline

| Phase | What | Where |
|---|---|---|
| 0 | Measure true token yield per source | `modal_app.py::measure` |
| 1 | Stream + clean (6-step deterministic chain) | `modal_app.py::clean`, `cleaning.py` |
| 2 | Dedup (MinHash near-dup + exact) + decontaminate vs. eval sets | `modal_app.py::dedup`, `dedup.py` |
| 3 | Train the 16K byte-level BPE tokenizer | `modal_app.py::tokenizer` |
| 4 | Tokenize + pack into 1024-token windows, 99/1 train/val split | `modal_app.py::tokenize` |
| 5 | Pretrain on GPU | `modal_app.py::pretrain_calibrate`, `::pretrain_run`, `train.py` |
| 6 | Push tokenizer + checkpoint to Hugging Face Hub | `modal_app.py::deploy_run` |

Phases 0-4 follow `Replication Guide.md` verbatim. Phases 5-6 are not covered
by the guide (it explicitly scopes pretraining out) and were added here,
sized for a single A10G GPU rather than the guide's 8xH100 assumption — see
`config.py`'s `TrainConfig` and the `micro_batch_size` override in
`modal_app.py::pretrain` for why.

`config.py` is the single source of truth; every other file imports from it.

## Setup

```bash
pip install modal
modal token new                      # or: modal token set --token-id ... --token-secret ...
```

Create `.env.local` (git-ignored) with `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`,
and `HUGGINGFACE_TOKEN`, then before every run:

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
```

Phase 6 additionally needs a Modal secret holding the HF token:

```bash
modal secret create huggingface-token HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"
```

## Running

```bash
modal run modal_app.py::measure                 # Phase 0
modal run modal_app.py::clean --fineweb-shards 5 # Phase 1
modal run modal_app.py::dedup                    # Phase 2
modal run modal_app.py::tokenizer                # Phase 3
modal run modal_app.py::tokenize                 # Phase 4
modal run modal_app.py::pretrain_calibrate        # Phase 5, measure throughput first
modal run --detach modal_app.py::pretrain_run     # Phase 5, full run (hours; --detach survives disconnects)
modal run modal_app.py::deploy_run                # Phase 6, after training finishes
```

Phases 0-4 are CPU-only and cost well under $1 total. Phase 5 cost/time
depends on the calibrated throughput and epoch count — `pretrain_calibrate`
prints an estimate before you commit to the full run.

## Tests

`cleaning.py`, `dedup.py`, and `train.py` are pure functions (no Modal/GPU
dependency), covered by `tests/`:

```bash
pip install -r requirements-dev.txt
pytest
```
