# SMILES-2026 Hallucination Detection

Detecting whether a small language model's response is **hallucinated**
(fabricated) or **truthful** by reading the model's own internal
hidden states with a lightweight binary classifier (a *probe*).

The base LLM is [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
— a 24-layer decoder-only transformer with hidden dim 896.

---

## Final results

5-fold stratified cross-validation on `data/dataset.csv`
(689 samples, 483 hallucinated / 206 truthful):

| Checkpoint | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| Majority-class baseline | 70.10% | 82.42% | — |
| Probe (train split) | 98.80% | 99.18% | 100.00% |
| Probe (val split) | 76.63% | 83.98% | 75.06% |
| **Probe (test split)** | **74.60%** | **82.61%** | **74.36%** |

* **+4.50 pp test accuracy** over the majority baseline.
* **+4.22 pp AUROC** over a single-MLP starter probe.
* Per-fold test accuracy stable above the baseline:
  `73.91 / 76.81 / 74.64 / 74.64 / 72.99` — peak fold 76.81%.
* `predictions.csv` distribution on the 100 unlabelled test
  samples: 18 truthful / 82 hallucinated (close to the 30 / 70
  training prior).

Full per-fold metrics live in `results.json`; the per-iteration
ablation log is in [`SOLUTION.md`](SOLUTION.md).

---

## Approach in one diagram

```
prompt + response
        │
        ▼  Qwen2.5-0.5B forward pass (output_hidden_states=True)
hidden_states  (25 layers × seq_len × 896)
        │
        ▼  aggregation.py
        │   • last 32 real (response) tokens at layers 6, 12, 18, 24
        │   • mean-pool + max-pool per layer  →  8 × 896
        │   • last real-token vector at layer 24  →  1 × 896
8064-dim feature vector
        │
        ▼  probe.py — stacked 5-stream classifier
        │
        ├── Stream 1: bagged MLP × 5      (256 GELU + dropout 0.30, AdamW)
        ├── Stream 2: LightGBM on PCA-100 (leaf-wise growth)
        ├── Stream 3: CatBoost on PCA-100 (oblivious trees)
        ├── Stream 4: XGBoost  on PCA-100 (depth-wise growth)
        └── Stream 5: LightGBM on raw 8064-dim features (feature_fraction=0.10)
                │
                ▼  out-of-fold base predictions  →  (n, 5) matrix
                │
                ▼  LogisticRegression(C=0.3, balanced)   ← meta-learner
                │
                ▼  accuracy-optimal threshold from OOF probs
            label ∈ {0, 1}
```

---

## Why this works

1. **Pool only response tokens.** The hallucination signal lives
   in the model's answer, not in the (long, near-identical) prompt
   context. Mean+max-pooling the last 32 real tokens at four
   evenly-spaced late-middle layers isolates that signal.

2. **Five base streams with different inductive biases.** MLPs
   capture smooth non-linear combinations; the three PCA-100
   boosters use complementary tree growth strategies (leaf-wise /
   oblivious / depth-wise); the LightGBM-on-raw stream sees the
   full 8064-dim space and discovers individual informative
   feature dimensions that PCA smears out. **This last stream
   lifted AUROC by ~2 pp** — the single biggest gain at the end.

3. **Stacking, not averaging.** Plain mean-of-means *hurt*
   accuracy because the PCA-100 boosters were correlated.
   Training a Logistic Regression meta-learner on out-of-fold
   base predictions lets each stream's weight be learned from
   data; strong L2 (`C=0.3`) prevents meta-overfit on only 5
   features.

4. **Accuracy-optimal threshold via OOF.** With imbalanced classes
   (70 / 30), the default 0.5 threshold mis-predicts. The threshold
   used at inference is picked to maximise accuracy on the meta-LR's
   probabilities over the internal OOF base matrix.

5. **Stratified 5-fold CV.** Stabilises the reported metrics and
   makes the union of train+val cover every sample, so the final
   probe trained for `predictions.csv` sees all 689 labelled
   examples.

---

## Reproducing the results

### Requirements

* Python 3.11.
* GPU recommended (≈2 GB VRAM is enough for Qwen2.5-0.5B in bf16);
  CPU works too — feature extraction dominates wall-clock.

### Commands

```bash
git clone https://github.com/YaroslavPelekhov/SMILES.git
cd SMILES

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate.bat

pip install -r requirements.txt
pip install lightgbm catboost xgboost   # extra deps used by the probe

# Linux / macOS / WSL
python solution.py
# Windows (PowerShell)
# $env:PYTHONIOENCODING="utf-8"; python solution.py
```

The first run downloads `Qwen/Qwen2.5-0.5B` from HuggingFace (~990 MB).
Subsequent runs can be made fully offline by exporting
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Running `solution.py` produces:

* `results.json` — averaged metrics from 5-fold cross-validation.
* `predictions.csv` — predicted labels for the 100 unlabelled
  `data/test.csv` samples (`id`, `label` columns).

The whole pipeline is deterministic given the random seeds inside
`splitting.py` and `probe.py`; on the same hardware / library
versions results are reproducible bit-for-bit.

---

## Repository layout

```
SMILES/
├── data/
│   ├── dataset.csv        # 689 labelled (prompt, response, label) samples
│   └── test.csv           # 100 unlabelled samples (predict these)
│
│   ── Implementation (student-edited) ──────────────────────────
├── aggregation.py         # Multi-layer mean+max-pool of response tokens
├── probe.py               # 5-stream stacked classifier + meta-LR
├── splitting.py           # Stratified 5-fold CV with per-fold val slice
│
│   ── Fixed infrastructure (unmodified) ────────────────────────
├── model.py               # Qwen2.5-0.5B loader
├── solution.py            # Driver: feature extraction → probe → predictions
├── evaluate.py            # Per-fold metrics, summary table, JSON output
│
│   ── Generated artefacts ──────────────────────────────────────
├── results.json           # 5-fold CV metrics (test acc 74.60%, AUROC 74.36%)
├── predictions.csv        # 100 predictions for data/test.csv
│
│   ── Docs ─────────────────────────────────────────────────────
├── README.md              # This file
├── SOLUTION.md            # Full submission report (Q3 requirements)
├── requirements.txt
└── LICENSE
```

---

## Iteration journey

Each row is one experimental cycle, measured on the same 5-fold CV
of `dataset.csv`:

| Iteration | Test Acc | Test AUROC | Notes |
|---|---:|---:|---|
| Default skeleton (single MLP, last-token, F1 threshold) | 71.26 % | 70.14 % | Starting point. |
| + Multi-layer mean-pool of response tokens | 72.13 % | 72.10 % | The pooled vectors carry the bulk of the signal. |
| + Accuracy-optimal threshold instead of F1 | 72.28 % | 72.10 % | Predictions less biased to the majority class. |
| + Max-pool features (5-MLP ensemble) | 72.13 % | 72.10 % | Max-pool helps the stacking later. |
| + LightGBM on PCA-100 (hybrid)         | 73.73 % | 71.05 % | Trees + MLP averaged for the first time. |
| + Bagging in the MLP ensemble          | 74.60 % | 72.11 % | 85% stratified bootstraps per member. |
| + CatBoost + XGBoost via stacking      | 74.02 % | 72.28 % | Stacking lets meta-LR weight streams. |
| **+ LightGBM on raw features**         | **74.60 %** | **74.36 %** | Final config — biggest AUROC jump. |

Net gain vs. the starter probe: **+3.34 pp accuracy, +4.22 pp AUROC**.

The full ablation log (including ideas that *didn't* help —
late-only layers, std-pool features, raw-feature CatBoost as a
6th stream, etc.) is in [`SOLUTION.md`](SOLUTION.md) section 4.

---

## Files you must read

* [`SOLUTION.md`](SOLUTION.md) — full submission report
  (Q3 requirements: reproducibility, final-solution description,
  experiments and failed attempts).
* [`results.json`](results.json) — raw 5-fold metrics produced
  by `evaluate.py`.
* [`predictions.csv`](predictions.csv) — final predictions for
  the 100 unlabelled test samples.

---

## License

See [`LICENSE`](LICENSE).
