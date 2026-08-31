# Architecture search — how to run it, and what to search over

Companion to issue #91. This document answers two questions: *which* search
procedure, and *which* parameters. Both answers come from the computer-vision
literature plus what this box has already measured.

## 1. Don't brute-force. The literature says why.

A full grid over the axes in §3 is ~600 configurations at 5–70 min per epoch. It is
also the wrong shape of search:

* **Random beats grid.** Bergstra & Bengio (JMLR 2012) show grid search wastes trials
  on dimensions that do not matter: search spaces are high-dimensional but their
  *effective* dimensionality is low — a few hyperparameters account for most of the
  variance. Random search finds equal or better models in a fraction of the compute,
  and parallelises trivially.
* **Don't train losers to completion.** Successive Halving / Hyperband / ASHA give
  every configuration a small budget, keep the top 1/η, multiply the budget by η, and
  repeat. Reported speedups are an order of magnitude over Bayesian optimisation and
  ≥10× over random search alone.

**Our own runs confirm that early ranking works here.** run 3 (kraken default) reached
val 0.7057 at **epoch 3**; run 2 was at 0.308 at epoch 7. The final ordering
(0.8226 vs 0.7809) was visible in the first three epochs, at ~4% of the compute
eventually spent.

The caveat is equally visible in our data: kraken+ vs run 2 differ by only 0.03 at
epoch 18. **Small gaps need high rungs; large gaps are settled at rung 1.** That is
exactly what successive halving does, and it is why a single fixed budget per config
is the wrong design.

### Proposed schedule (η = 3)

| rung | budget | configs | screening shard | wall clock |
|---|---|---|---|---|
| 0 | 3 epochs | 45 | 2,500 pages | ~4 min/epoch → ~9 h |
| 1 | 9 epochs | 15 | 5,000 pages | ~8 min/epoch → ~18 h |
| 2 | 27 epochs | 5 | `shard_00` (24,744) | ~20–70 min/epoch → ~3 d |
| 3 | to plateau | 2 | `shard_00` | until early stop |

Rungs 0–1 rank; rung 2 confirms; rung 3 produces a model. Only rung 3 output is ever
published. **Rank inversion between small and large data is the known failure mode** —
small sets favour small models — so promotion is never the final word.

## 2. Fairness rules (each one is a bug we already hit)

1. **Fixed optimizer-step budget, not fixed epochs.** run 3's epoch was 4× run 2's,
   partly because its OOM forced batch 64 and therefore 4× the steps.
2. **Fixed *effective* batch, micro-batch found per config.** run 3 OOMed at 256 where
   run 2 did not — activation memory, not parameters. Probe the largest micro-batch
   that fits, then set `--accumulate-grad-batches` to reach the common effective batch.
   The linear scaling rule (batch ↔ LR) means an uncontrolled batch silently changes
   the learning rate too.
3. **`--quit dumb` with a fixed epoch count in rungs 0–2.** `--epochs` only sizes the
   `1cycle` schedule; `--quit early` decides when to stop. run 2 spent ~18 of its 27
   hours training after the LR had annealed to zero.
4. **From scratch, or vary base models — never mixed.** `--spec` is ignored when
   `--load` is given.
5. **One data version per sweep**, recorded. `shard_00.arrow` predates #89/#90 and
   still contains lines those fixes now drop.

## 3. Parameters, ranked by expected payoff

### Tier 1 — the axes with direct evidence

**Input height** — 48 / 64 / 96 / 120 / 128.
Scene-text CRNNs use 32; historical HTR implementations cluster at 60–128. Our only
controlled comparison is 64 (run 2, 0.7809) vs 120 (run 3, 0.8226). Strong prior, and
cheap to vary.

**Horizontal downsampling / frames per character** — total width stride 2 / 4 / 8.
CTC cannot emit more labels than it has timesteps, and over-downsampling makes narrow
or compact characters unrepresentable; the literature warns explicitly against
reducing sequence length too far. Our eval audit measured **13.5 px per character
(p50 12.15)** and **60.9 characters per line**. At width stride 8 that is ~1.7 frames
per character — tight. **Add a diagnostic that reports frames-per-character for every
config before training starts**; anything under ~2 should be flagged, not silently
trained.

**Augmentation on/off** — `--augment`.
Random rotation and elastic distortion both beat the baseline on historical material
(≈3.6% relative CER in one ablation); affine + elastic are the two standard schemes.
Ströbel attributes his 1.6-point gap between kraken+ and HTR+ specifically to
pre-processing and augmentation. Neither of our runs used it. Highest-value single
switch we have not tried.

**Peak learning rate** — 3e-4 / 1e-3 / 3e-3, with warmup.
Measured here: 1e-4 from scratch under `1cycle` starts at 4e-6 and never escapes CTC
blank collapse; 1e-3 works. Warmup exists precisely because early parameters are far
from any solution and a large LR is unstable there.

### Tier 2 — plausible, cheap to include

**Recurrent width × depth** — `Lbx128/200/256/400` × 2/3/4 layers. The original CRNN
uses 256; implementations at higher horizontal resolution move to 512. Our two
architectures differ here (256×3 vs 200×3) but confounded with height, so it is
currently unmeasured.

**Dropout rate and placement** — 0.1 after conv blocks (kraken default) vs 0.5 after
each BLSTM (Ströbel/PyLaia). A factor of five apart, never compared on our data.

**Conv stack depth and widths** — 3 vs 4 blocks; 12/24/48/48 (PyLaia) vs 32/32/64/64
(kraken) vs 8/32/64 (kraken+).

**Output bottleneck** — the `Cr*,*,85` question from `docs/KRAKEN_PLUS.md`: 85 filters
before 102 classes is rank-limiting. Currently being measured.

### Tier 3 — worth one run each, not a sweep axis

* **Fine-tune vs from scratch.** Historically our largest single effect: 0.9838 → 0.3921
  on identical data (§9b). Any sweep result must be read against a fine-tuned baseline.
* **Label smoothing / cosine annealing.** Standard "bag of tricks" gains on
  classification; unverified for CTC.
* **Deformable convolutions**, reported to boost modern *and* historical HTR — not
  expressible in VGSL, so it would need a kraken fork. Park it.
* **Combining marks.** Neither model emits a single one of the 170 `Inherited`
  characters (nasal bars, superscript vowels). This is not an architecture-scale
  problem and will not be fixed by this sweep; it needs its own investigation.

## 4. What the sweep can realistically buy

Current best on our held-out medieval set: **CER 0.1335** (run 3). Published SOTA for
line-level CTC on IAM sits near 4.6–4.7% CER, TrOCR at 2.89%, and a multilingual
historical "supermodel" reports ~2.95% average — on cleaner, better-resourced material
than ours.

The gap is therefore not mostly architectural. Expect an architecture sweep to move
0.1335 into roughly the 0.09–0.11 range; expect augmentation, more data, and starting
from pretrained weights to matter more. The sweep is worth running because it is cheap
and mechanisable, not because it is where the remaining error lives.

## 5. Sources

* Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 13 (2012) —
  https://jmlr.org/papers/v13/bergstra12a.html
* Li et al., *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization* —
  https://arxiv.org/pdf/1603.06560
* He et al., *Bag of Tricks for Image Classification with CNNs*, CVPR 2019 —
  https://arxiv.org/pdf/1812.01187
* *Handwritten Text Recognition: A Survey* (2025) — https://arxiv.org/pdf/2502.08417
* Cascianelli et al., *Boosting Modern and Historical HTR with Deformable Convolutions* —
  https://arxiv.org/pdf/2208.08109
* *Handwriting Recognition of Historical Documents with few labeled data* —
  https://arxiv.org/pdf/1811.07768
* *2D-CTC for Scene Text Recognition* — https://arxiv.org/pdf/1907.09705
* Ströbel, dissertation, Kap. 3.6.1 — see `docs/KRAKEN_PLUS.md`
