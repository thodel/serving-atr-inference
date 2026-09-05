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

1. **Equal epochs *are* equal compute once rule 2 holds — do not "improve" on this.**
   The original wording was "fixed optimizer-step budget, not fixed epochs", motivated
   by run 3's epoch being 4× run 2's after an OOM forced batch 64. Implemented
   literally in the first height sweep, the budget was computed as
   `epochs × lines / micro_batch` — which counts **micro-batches, not optimizer
   steps**. Gradient accumulation was never divided out:

   | | micro-batches | accum | actual optimizer steps |
   |---|---:|---:|---:|
   | h48, 4 epochs | 12,996 | 1 | **12,996** |
   | h120, 1 epoch | 12,996 | 4 | **3,249** |

   The tall configurations ran on a quarter of the budget and returned val 0.27
   against 0.55–0.65 for the short ones — which reads as "tall is far worse" and means
   "undertrained". Rerun at four epochs each, that same h120 reaches **0.7154**, the
   best of the sweep.

   Rule 2 already fixes the *effective* batch, and then
   `optimizer steps = lines × epochs / effective_batch` — nothing else enters, so equal
   epochs are equal steps by construction. Fix a step budget only where the effective
   batch genuinely varies, and count it as `micro_batches / accumulation`, which is what
   Lightning's `global_step` records and what the checkpoints can be checked against.
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

**Horizontal downsampling / frames per character** — total width stride 2 / 4 / 8,
and input height, which cannot be separated from it.
CTC cannot emit more labels than it has timesteps, and the literature warns
explicitly against reducing sequence length too far. The quantity that decides this
is scale-free — `width / (height × characters)` — because kraken normalises every
crop to the spec's input height and scales the width with it. Measured on
`val_clean.arrow` (6,319 lines): crops are a median 91 px tall at 33.1 px per
character, **aspect_per_char 0.326, p10 0.246**. That puts the two trained
architectures at:

| spec | height | stride | frames/char (p10) |
|---|---|---|---|
| run 2 (kraken+) | 64 | 8 | **1.97** |
| run 3 (default) | 120 | 8 | **3.69** |

So run 3 sees nearly **twice the horizontal resolution** of run 2 from the same
pages — a measured mechanism for part of the 0.7809 → 0.8226 gap, and an argument
that height and stride should be searched as one axis rather than two. Implemented
as S10 (`training/vgsl_geometry.py`), which runs after `prepare` and refuses a spec
that leaves under 1.25 frames per character.

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


## 6. Results — height sweep, rung 0 (2026-09-02)

`shard_00` / `val_clean`, kraken default architecture, **only the input height varies**.
Effective batch 256 throughout (the micro-batch shrinks with height because 120 px OOMs
at 256), 4 epochs = 12,996 optimizer steps each, lrate 1e-3, seed 42, `--no-augment`.

| height | micro × accum | val accuracy | frames/char (S10) | wall |
|---:|---|---:|---:|---:|
| 48 | 256 × 1 | 0.5528 | 1.48 `warn` | 1 h 43 |
| 64 | 256 × 1 | 0.6475 | 1.97 `warn` | 2 h 34 |
| 96 | 128 × 2 | 0.6827 | 2.95 `ok` | 4 h 10 |
| 120 | 64 × 4 | **0.7154** | 3.69 `ok` | 5 h 28 |
| 128 | 64 × 4 | *running* | 3.94 `ok` | — |

**Monotone: every step up in height buys accuracy**, and the gain has not flattened
inside the range tested. The ordering matches the S10 geometry: the two heights flagged
`warn` for leaving under two CTC frames per character are the two worst, and the gap
between them (0.0947) is wider than between any adjacent `ok` pair.

It also explains run 3 against run 2 retrospectively — **0.8226 vs 0.7809 was the
height, not the architecture**. kraken+ (`Cr1,1,85`, height 64) landing between them
supports the same reading; see `docs/KRAKEN_PLUS.md`.

The cost: 120 px takes ~3× the wall time of 48 px for the same number of optimizer
steps. Height is both the most valuable knob found so far and the most expensive per
epoch — precisely the trade a rung ladder exists to manage.


## 7. Results — height vs. capacity (2026-09-05)

The height axis moves two things at once: `S1(1x0)1,3` folds the residual height into
channels, so h48 gives the LSTM stack 384 inputs and h256 gives it 2048. A pure height
sweep therefore cannot say whether resolution or capacity is doing the work. Arm A varies
the height; Arm B holds it at 128 and buys the same parameter count through LSTM width.
Pairs matched to ~1 % on parameters, measured with kraken's own VGSL builder.

| params | Arm A (height, Lbx200) | Arm B (h128, LSTM width) | Δ | wall A : B |
|---:|---|---|---:|---|
| ~5.7 M | **h256: 0.7515** | Lbx248: 0.7411 | **+0.0104** | 19.9 h : 6.3 h |
| ~4.9 M | **h192: 0.7494** | Lbx224: 0.7346 | **+0.0148** | 11.7 h : 6.3 h |

**Height wins both pairs at matched capacity**, so the gain is not capacity in disguise —
resolution contributes on its own.

**LSTM width, by contrast, is inert.** At fixed height 128: 4.1 M → 0.7355, 4.9 M →
0.7346, 5.7 M → 0.7411. Eight hundred thousand extra parameters buy 0.0009. It is not a
usable axis on this material.

**And height flattens.** 128 → 192 is +0.0139; 192 → 256 is +0.0021 for 70 % more wall
time. The knee is around 192.

*Caveat:* single runs, no seed repetition. Plateau fluctuation in earlier runs was
±0.005–0.01, so the pair differences sit at the edge of that band. Both pairs pointing the
same way while the capacity axis stays flat is what carries the finding; two seeds per
configuration (~52 GPU-hours) would settle it.

**For a production model: height 192, LSTM width 200.** And the ordering of the three
earlier runs is now fully explained — run 3 (h120) beat run 2 and kraken+ (both h64)
because of the height, not the architecture.
