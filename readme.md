# SIH26142 — Fidelity-First Super-Resolution Mapping (SRM) of Sentinel-2 Imagery
### Implementation Plan (v1.0 · prepared 8 Oct 2026)

**Problem:** Deep-learning super-resolution from medium-resolution satellite imagery (NTRO · Smart Education · Software)
**One-line strategy:** Build a *trustworthy* 10 m → 2.5 m Sentinel-2 super-resolver. The product is the **proof of fidelity** (hallucination, spectral, geometric and downstream tests, plus per-pixel uncertainty), not just sharper-looking pictures.

> **How to use this document.** Section 0 is the *contract*. Every later section obeys it. If you ever feel two sections disagree, Section 0 wins. Items marked **[VERIFY]** could not be confirmed during research and must be checked in Phase 0 before being relied on.

---

## 0. Conflict-Prevention Contract (read first)

Most implementation conflicts in SR projects come from silent mismatches: band order, reflectance scaling, patch sizes, degradation operators, data leakage, mixed environments and over-claiming. They are all frozen here.

### 0.1 Decision Register (frozen after Phase 0; changes need a written entry in `DECISIONS.md`)

| ID | Decision | Value (frozen) | Why |
|----|----------|----------------|-----|
| D1 | Scope | Single-image SR, Sentinel-2 **10 m RGB+NIR → 2.5 m (×4)** | Matches the brief ("Sentinel-2 at ten metres"), the benchmark (opensr-test) and the training data (SEN2NAIP). |
| D2 | Band order everywhere | **R, G, B, NIR = B04, B03, B02, B08** (channel 0..3) | opensr-test treats channels 0:3 as RGB. Mixing with B02-B03-B04 order silently swaps colours. Assert on every loader. **[VERIFY per loader]** |
| D3 | Radiometry | `float32` surface reflectance in **[0, 1] = DN/10000 after applying the product's own scale/offset metadata** | Newer L2A products carry a reflectance offset. Never hard-code; read STAC `raster:bands` scale/offset or product metadata. **[VERIFY on India tiles]** |
| D4 | Patch geometry | **LR 128×128 → SR 512×512** (center-crop SEN2NAIPv2's 130→128 by 1 px/side and 520→512 by 4 px/side, so alignment is preserved) | Same 4:1 ratio, same window the pretrained baselines use. |
| D5 | Degradation / consistency operator | *Training* uses a 4×4 average-pool (documented approximation). *Reported* consistency metrics use **opensr-test's default operator**. Both are reported side by side. | Avoids an unnoticed train/eval mismatch. |
| D6 | Splits | Use the dataset-provided train/val/test of SEN2NAIPv2. **opensr-test datasets are external test only. Never train or tune on them.** | Prevents leakage. |
| D7 | Model interface | Every model (baseline or ours) is a **black box that writes GeoTIFFs** to `outputs/sr/{model}/{tile_id}.tif`. The evaluator only reads files. | Lets incompatible dependencies live in separate environments. |
| D8 | Single evaluator | One `evaluate.py`; every model goes through the same code path and the same test tiles. | No cherry-picked comparisons. |
| D9 | Randomness | Fixed seeds; **3 seeds** for our model; diffusion baselines use K samples. Report mean ± std. | Honest variance. |
| D10 | Claims policy | See Section 14. | Prevents over-claiming. |
| D11 | Environments | At most **two** Python environments (`env-main`, `env-baselines`). Freeze each into a lock file after first successful install. | Stops dependency fights. |
| D12 | Evaluation tracks | **Track A** (HR reference exists, US/Spain tiles): full hallucination metrics. **Track B** (India AOI, no HR reference): reference-free checks only. | Prevents claiming "hallucination score" where no ground truth exists. |
| D13 | Out of scope | 20 m / 60 m bands, Landsat 30 m, ×8, multi-temporal fusion, SWIR | Keeps the build finishable. They go on a "future work" slide only. |

### 0.2 Hard rules
1. **No silent resampling.** Any resize must be a named function in `src/common/resample.py` with its kernel documented.
2. **No metric without its caveat.** PSNR/SSIM/LPIPS are always reported alongside hallucination metrics (Section 2.3 explains why).
3. **Test set is touched once** per model configuration, after the configuration is frozen.
4. **Every output carries provenance:** GeoTIFF tags `SR_MODEL`, `SR_VERSION`, `SYNTHETIC_ENHANCEMENT=true`, `NOT_GROUND_TRUTH=true`, source scene ID and date.
5. **No claim about India imagery that requires an HR reference.** Use Track B language only.

---

## 1. Problem Statement Analysis

### 1.1 What is really being asked
Free satellite images at 10-30 m cover large areas frequently but are too coarse to resolve small buildings, narrow roads or field boundaries. The task is a deep-learning model that sharpens such images into finer detail. The qualifier that matters is that it must **reconstruct genuine fine-scale features while keeping geographic registration and spectral values honest**. Prettier is not the goal.

### 1.2 Explicit requirements (from the brief)
| # | Requirement | How this plan satisfies it |
|---|-------------|---------------------------|
| R1 | Input: medium-res satellite imagery (e.g., Sentinel-2, 10 m) | D1; Sections 4-5 |
| R2 | Model: GAN / diffusion / deep network trained on paired medium/high-res imagery | Section 6: our constrained CNN family plus pretrained diffusion baseline (LDSR-S2) and GAN-style loss arm |
| R3 | **Preserve geographic registration** | Section 7.2 (alignment tests, affine-transform assertion) |
| R4 | **Preserve spectral consistency** | Spectral consistency loss + hard low-frequency constraint (6.3); spectral metrics (7.2) |
| R5 | **No plausible-but-false detail** | Hallucination/omission/improvement metrics, uncertainty maps, ablation of sharpness-vs-fidelity (7.2, 6.5) |
| R6 | Evaluate on standard image-quality metrics | PSNR/SSIM/LPIPS (7.1) |
| R7 | Evaluate whether downstream tasks (building/road extraction) actually improve | Section 8 |
| R8 | Smallest winning demo: real S2 tile, SR output next to true HR, road/field structure recovered, fidelity metrics reported | Section 9 |

### 1.3 Implicit requirements (what an NTRO-style judge will probe)
- *Can I trust this output for an intelligence decision?* → uncertainty maps, labelled synthetic provenance, documented failure modes.
- *Does it generalise outside the training region?* → external test sets (Spain), India AOI qualitative/Track B analysis.
- *Is your improvement real or a metric artefact?* → hallucination-aware metrics plus downstream tests, including honest reporting of negative results.
- *Is the data legal and reproducible?* → licence table (5.2), fixed pipeline, lock files.

### 1.4 Success criteria (definition of "done")
1. Reproducible pipeline: raw S2 tile in → georeferenced 2.5 m GeoTIFF + uncertainty/trust map out.
2. Benchmark table comparing **bicubic, pretrained SOTA baselines and our model** on one evaluator and one test set.
3. A **sharpness-vs-fidelity ablation** showing where hallucination begins to rise.
4. A **downstream study** (building and road extraction) with an honest conclusion, whatever it is.
5. A demo showing triptych (LR | SR | true HR) plus uncertainty overlay and metrics.

### 1.5 Ambiguities and how this plan resolves them
| Ambiguity | Resolution |
|-----------|-----------|
| "10-30 m" implies Landsat too | Out of scope (D13): no clean public paired HR data at that scale; naming it as future work is safer than half-supporting it. |
| "GAN, diffusion or deep network" | We use all three families in the comparison, but only **one** is our trained model (6.2). The rest are pretrained baselines. |
| "Paired training data" | Real cross-sensor pairs are scarce; we use SEN2NAIP (real + synthetic) and validate on external real pairs (5.1). |
| "Geographic fidelity" has no formula | We define it operationally: (a) affine/registration test, (b) shift-tolerant edge/structure fidelity, (c) road/building overlap against vector labels. |

---

## 2. Research Findings and What They Mean for the Plan

### 2.1 The landscape (verified in sources, Section 16)
| Finding | Implication |
|---------|-------------|
| SR is inherently ill-posed: many HR images are consistent with one LR input. | Report uncertainty; never present SR as ground truth. |
| Training pairs are the bottleneck. Early work used another sensor with similar bands as the HR reference. | Use SEN2NAIP and keep the cross-sensor caveats explicit. |
| **SEN2NAIP** (Scientific Data, 2024): real Sentinel-2/NAIP cross-sensor pairs plus a large synthetic set from a learned degradation model. | Primary training source. Counts differ between versions, so read the card of the version you download. **[VERIFY]** |
| **opensr-test** (IEEE GRSL 2024, MIT licence): metrics for consistency, synthesis and correctness (hallucination / omission / improvement), five datasets (NAIP, SPOT, Venµs, Spain crops, Spain urban). | Our evaluation backbone (Section 7). |
| **LDSR-S2 / opensr-model**: latent-diffusion SR for RGB+NIR, 10 m → 2.5 m, produces pixel-level uncertainty from sampling variability. | Pretrained diffusion baseline and uncertainty reference. |
| **SEN2SR**: CNN/Swin/Mamba SR with a *low-frequency hard constraint* for spectral consistency; bigger models (>15 M parameters) did not give significant gains; Mamba beat CNN but needs special CUDA setup. | Motivates a small model with a hard consistency constraint (6.3). Use the **Lite CNN** baseline, skip Mamba. |
| **DiffFuSR**: diffusion on RGB plus a fusion network for other bands, evaluated on OpenSR benchmark. | Cited as related work; not reproduced (D13). |
| **Downstream reality check** (TU Wien thesis, 2025, Austria): UNet building delineation on *interpolated* Sentinel-2 beat the SR outputs of the tested models; native HR orthophoto was best. Other studies found raster IoU/F1 barely moved while vector/object metrics improved for larger buildings. | **Do not assume SR helps downstream.** Pre-register the hypothesis and report honestly (Section 8). This is also a differentiator. |
| Sentinel-2 L2A absolute geolocation is better than ~6 m, i.e. more than two 2.5 m pixels. | Pixel-exact metrics against HR are unfair. Use shift-tolerant metrics (7.2). |
| A 2025 benchmark of hallucination metrics for image restoration found LPIPS/DISTS only marginally above chance at detecting hallucinations. | Never rely on LPIPS alone (D10, hard rule 2). |

### 2.2 Datasets and tools discovered
| Resource | Role | Notes |
|----------|------|-------|
| `isp-uv-es/SEN2NAIP` and `tacofoundation/SEN2NAIPv2*` (Hugging Face) | Training/validation/test | LR 130×130, HR 520×520 patches in v2. |
| `isp-uv-es/opensr-test` (HF) + `pip install opensr-test` | External test + metrics | NAIP ×4 (62 imgs), SPOT ×4 (9), Venµs ×2 (59), Spain Crops ×4 (28), Spain Urban ×4 (20). |
| `pip install opensr-model`, `opensr-utils`, `sen2sr` + `mlstac` | Pretrained baselines | SEN2SR full version needs `mamba-ssm` and recent CUDA → **avoid**, use Lite. |
| WorldStrat (Zenodo) | Optional extra pairs (SPOT 1.5 m pan / 6 m RGBN vs S2) | HR imagery is **CC-BY-NC**; non-commercial/demo use only; do not redistribute. |
| Google Open Buildings v3 | Building labels for **Track B** (India) | Covers Africa, South & South-East Asia, Latin America; **not the US**. Use ≥ 90 % precision threshold; CC-BY-4.0 / ODbL. |
| OpenStreetMap | Road/building labels for **Track A** (US/Spain) | ODbL. |
| Copernicus Data Space Ecosystem (CDSE) STAC (`stac.dataspace.copernicus.eu/v1`) and AWS Earth Search COGs | Fresh Sentinel-2 L2A | CDSE requires a free account and S3 keys; AWS COGs readable with no sign-in. |

### 2.3 Why standard metrics are not enough (the core argument)
PSNR/SSIM/LPIPS reward sharpness and tolerate plausible-looking fabrication, and they break under tiny shifts or brightness differences that are normal in real cross-sensor pairs. So the plan treats them as *necessary but insufficient* and adds consistency, hallucination and downstream measures.

---

## 3. Solution Overview

```
                        ┌────────────────────────────────────────────────┐
 CDSE / AWS COGs ──►    │ 1. INGEST   read S2 L2A window, RGBN, scale     │
 (any AOI, e.g. India)  │    → reflectance [0,1], D2/D3 asserts           │
                        └───────────────┬────────────────────────────────┘
                                        ▼
 SEN2NAIPv2 (train) ─►  ┌────────────────────────────────────────────────┐
                        │ 2. MODELS (black boxes → GeoTIFF, D7)           │
                        │   bicubic │ SEN2SR-Lite │ LDSR-S2 │ OURS(A1-A4) │
                        └───────────────┬────────────────────────────────┘
                                        ▼
                        ┌────────────────────────────────────────────────┐
                        │ 3. TRUST LAYER                                 │
                        │   uncertainty map · provenance tags ·          │
                        │   consistency check (SR↓ vs LR)                │
                        └───────────────┬────────────────────────────────┘
                                        ▼
 opensr-test sets ──►   ┌────────────────────────────────────────────────┐
 OSM / OpenBuildings    │ 4. EVALUATE (single evaluate.py, D8)            │
                        │   Tier1 PSNR/SSIM/LPIPS                         │
                        │   Tier2 consistency + HA/OM/IM + registration   │
                        │   Tier3 downstream buildings & roads            │
                        └───────────────┬────────────────────────────────┘
                                        ▼
                        ┌────────────────────────────────────────────────┐
                        │ 5. DEMO + REPORT  triptych · overlays · tables  │
                        └────────────────────────────────────────────────┘
```

---

## 4. Canonical Data Contract (implements D2-D4, D7)

| Item | Specification |
|------|---------------|
| Array layout | `float32`, shape `(4, H, W)`, channel order R,G,B,NIR |
| Value range | Reflectance in [0, 1]; clip to [0, 1] only at the I/O boundary and log how many pixels were clipped |
| LR patch | `(4, 128, 128)`, 10 m |
| SR/HR patch | `(4, 512, 512)`, 2.5 m |
| Alignment rule | LR pixel `(i, j)` covers SR pixels `4i..4i+3`, `4j..4j+3`. SR affine = LR affine with pixel size ÷ 4 and **identical origin**. A unit test asserts this for every output file. |
| CRS | Keep the source UTM CRS; never reproject LR before SR |
| File format | Cloud-optimised GeoTIFF, `float32` or `uint16` (DN/10000 scale stored in tags), LZW compression |
| Naming | `{dataset}_{tile_id}_{modelname}_{version}.tif` (lowercase, no spaces) |
| Large scenes | Sliding window 128 px with 16 px overlap, feathered blending, discard 4 px at window borders (opensr-utils uses a similar window/overlap approach) |

**Unit tests that must pass before any experiment (`tests/test_contract.py`):** band-order assertion on a vegetation tile (NIR mean > Red mean), affine alignment, value range, shape/dtype, no NaN.

---

## 5. Data Plan

### 5.1 Datasets and roles
| Role | Dataset | Use |
|------|---------|-----|
| Train / val / test (in-domain) | **SEN2NAIPv2** (real cross-sensor + synthetic, provided splits) | Training our model (synthetic + real mix), validation for early stopping, in-domain test (touched once) |
| External test (Track A) | **opensr-test**: NAIP, SPOT, Spain Crops, Spain Urban (Venµs is ×2, so run it as a separate secondary table, not mixed in) | Hallucination / consistency metrics on data never trained on |
| Geographic-structure labels (Track A) | OSM roads/buildings rasterised to 2.5 m for NAIP/Spain tiles | Road/building overlap and downstream training |
| Track B AOI | 2-3 Sentinel-2 L2A windows over **Punjab/Ludhiana region** (cities, villages, canals/roads, fields), cloud-free | Qualitative + reference-free evaluation, demo |
| Track B labels | Google Open Buildings v3 (≥ 90 % precision threshold) + OSM roads | Downstream check on India with caveat that labels are themselves model/volunteer-derived |
| Optional | WorldStrat (non-commercial) | Extra evaluation, only if time allows, kept out of the headline table |

### 5.2 Licence table (checked into `LICENSES.md`)
| Dataset/tool | Licence (as seen in sources) | Constraint for us |
|--------------|------------------------------|-------------------|
| SEN2NAIP v1 | CC-BY-4.0 | Attribute |
| SEN2NAIPv2-real | CC0 (per its STAC metadata) **[VERIFY on the card of the exact version]** | Attribute anyway |
| opensr-test code | MIT | Attribute |
| opensr-utils | Apache-2.0 | Attribute |
| opensr-model / SEN2SR | **[VERIFY licence in each repo before reuse of weights]** | Do not redistribute weights until verified |
| WorldStrat | HR imagery CC-BY-NC; S2 and labels CC-BY | Non-commercial only |
| Open Buildings v3 | CC-BY-4.0 and ODbL (choose one) | Attribute |
| OSM | ODbL | Attribute, share-alike for derived DBs |
| Sentinel-2 | Free, full and open (Copernicus) | Attribute "Contains modified Copernicus Sentinel data" |

### 5.3 Leakage and bias controls
- Use SEN2NAIPv2's own splits (D6). Compute footprint intersection between SEN2NAIPv2 train tiles and each opensr-test tile; **drop any overlapping test tile** and log how many were dropped. **[VERIFY]**
- The test set is looked at once per frozen configuration. Hyper-parameter tuning uses validation only.
- Report results **stratified by land cover**: urban, cropland, forest/bare. Averages hide failures.
- Document the geographic bias openly: training data is US-dominated (NAIP); India behaviour is therefore a stated limitation, not a hidden assumption.

### 5.4 Ingestion steps
1. Create CDSE account (needed for the official STAC/S3), and test AWS COG anonymous reads as a fallback.
2. Write `src/data/ingest_s2.py`: STAC search → pick low-cloud scene → windowed read of B04,B03,B02,B08 → apply scale/offset from metadata (D3) → write contract-conformant GeoTIFF.
3. Write `src/data/sen2naip_loader.py`: download from Hugging Face, center-crop to D4, assert D2 order, produce a PyTorch `Dataset`.
4. Write `src/data/rasterize_labels.py`: OSM/Open Buildings → 2.5 m masks aligned to HR tiles.
5. Run `tests/test_contract.py` on all sources.

---

## 6. Model Plan

### 6.1 Philosophy
Spend effort where the judges will look: **fidelity engineering and evaluation**. The architecture is not the differentiator. Research shows small models are enough, so we train a **small constrained model** and compare it with strong pretrained baselines.

### 6.2 Models in the comparison
| Name | Type | Source | Trained by us? |
|------|------|--------|----------------|
| `bicubic` | Interpolation (Pillow-based, anti-aliased) | `src/common/resample.py` | No |
| `sen2sr_lite` | CNN, 10 m → 2.5 m with low-frequency constraint | `sen2sr` package | No (pretrained) |
| `ldsr_s2` | Latent diffusion, with uncertainty | `opensr-model` | No (pretrained) |
| `ours_a1` … `ours_a4` | Small constrained CNN (see 6.3-6.5), ≤ ~5 M parameters | `src/models/` | **Yes** |

### 6.3 Our model: "consistency by construction"
**Generator:** a compact residual network (RRDB-lite or EDSR-style, ≤ ~5 M parameters) predicting a high-frequency residual on top of the bicubic upsample.

**Hard consistency constraint (core idea):**
```
Let U = bicubic ×4 upsample,  D = 4×4 average pool.
G(x)       = raw network output (4,512,512)
detail     = G(x) − U(D(G(x)))          # remove what D can see (the low-frequency part)
SR(x)      = U(x) + detail              # add only high-frequency detail to the upsampled input
```
This pushes the network to add detail that does not change the low-frequency content, so `D(SR)` stays close to the LR input. It is *approximately* consistent (the real sensor operator differs from D), so we **measure** this with opensr-test rather than assume it (D5). This idea mirrors the low-frequency hard constraint reported for SEN2SR and is independently implemented here.

**Loss (all weights are starting values, tuned on validation only):**
| Term | Purpose | Start weight |
|------|---------|--------------|
| L1 on reflectance | Pixel fidelity | 1.0 |
| Spectral-angle loss (per pixel) | Colour/spectral preservation | 0.1 |
| Consistency loss `L1(D(SR), LR)` | Soft version of the hard constraint | 1.0 |
| Edge/gradient loss (Sobel) | Sharp, correctly placed structure | 0.1 |
| LPIPS | Perceptual sharpness | **0.05** (small, because perceptual/adversarial terms encourage invented detail; ablated in 6.5) |
| Adversarial (PatchGAN) | Realistic texture | **0 by default**; only in arm A4 |

### 6.4 Training recipe
- Data: SEN2NAIPv2 train; mix synthetic and real-pair samples (start 80/20, tuned on val).
- Augmentation: flips/rot90 only, applied jointly to LR and HR. **No** brightness/contrast augmentation (breaks spectral claims).
- Optimiser: Adam, lr 1e-4, cosine decay, mixed precision, batch size set by GPU memory.
- Early stopping on validation consistency + edge fidelity, **not** PSNR alone.
- Hardware: single GPU (free-tier cloud GPUs are enough for a ≤ 5 M-parameter model). Record time per epoch in `RUNLOG.md` in Phase 2 to size the schedule.
- 3 seeds per arm (D9).

### 6.5 Ablation arms ("the fidelity dial")
| Arm | Description | Expected behaviour |
|-----|-------------|--------------------|
| A0 | Bicubic baseline | Blurry, perfect consistency, no hallucination, many omissions |
| A1 | L1 + hard constraint + spectral + consistency | Conservative; low hallucination |
| A2 | A1 + edge loss | Sharper edges; check hallucination rise |
| A3 | A2 + LPIPS | Sharper still; likely more hallucination |
| A4 | A3 + adversarial | Sharpest; highest hallucination risk |

Plot **hallucination vs. omission vs. improvement** (opensr-test) against sharpness for A0-A4 plus baselines. The expected finding, to be tested rather than assumed, is a frontier where extra sharpness is paid for with hallucination. **The recommended release model is the arm with the best fidelity at acceptable sharpness**, chosen on validation, not the prettiest arm.

### 6.6 Uncertainty / trust map
| Model | Uncertainty source |
|-------|--------------------|
| `ldsr_s2` | Built-in: variability over multiple diffusion samples (opensr-model provides this) |
| `ours_*` | **Seed ensemble disagreement** (std across 3 seeds' outputs) as the first version. Optionally add test-time augmentation (flip/rot) disagreement. |
| Validation | Check that high-uncertainty pixels correlate with high error vs HR on Track A (Spearman correlation reported). If correlation is weak, say so. |

---

## 7. Evaluation Plan

All tiers run through one `evaluate.py` (D8) with fixed random seeds and one list of test tiles.

### 7.1 Tier 1: standard image-quality metrics
PSNR, SSIM, LPIPS, computed after **shift-tolerant alignment** (search ±2 px at 2.5 m and take the best shift per tile; record the shift). Reported for completeness and labelled "insufficient alone".

### 7.2 Tier 2: fidelity metrics (our headline)
| Metric | Tool | What it proves |
|--------|------|----------------|
| Reflectance consistency | opensr-test `reflectance` | SR, when downsampled, still matches the original LR magnitudes |
| Spectral consistency | opensr-test `spectral` (spectral angle, degrees) | No colour/spectral drift |
| Spatial alignment | opensr-test `spatial` (phase correlation) + our affine unit test (Section 4) | Geographic registration preserved |
| Synthesis | opensr-test `synthesis` | Amount of new high-frequency detail added |
| **Hallucination** `ha` | opensr-test | Detail in SR that is in neither LR nor HR |
| **Omission** `om` | opensr-test | Detail in HR that SR failed to recover |
| **Improvement** `im` | opensr-test | Detail in SR that is genuinely in HR and not in LR |
| Distance variants | Report `ha/om/im` under at least the **LPIPS** and **normalized-difference** distances, as opensr-test does, because the choice of distance changes the numbers | Robustness of conclusions |
| Edge fidelity (custom) | Sobel edges, tolerance ±1 px: edge precision/recall/F1 of SR vs HR | Roads and field boundaries recovered rather than invented |
| Uncertainty calibration | Spearman(uncertainty, |SR−HR|) | Trust map is meaningful |

**Reference numbers for sanity-checking the harness (from the opensr-test README, LPIPS variant):** pretrained opensr-model shows reflectance ≈ 0.0076 and spectral ≈ 1.97°; SuperImage ≈ 0.0068 and 1.90°; SR4RS and Satlas are far worse on consistency. If our harness gives wildly different values for those models, the harness, not the model, is wrong. **[VERIFY by running the reference models in Phase 2]**

### 7.3 Pass/fail gates (initial, recalibrated once after the Phase-2 baseline run, then frozen)
| Gate | Criterion |
|------|-----------|
| G1 Contract | 100 % of outputs pass Section 4 unit tests |
| G2 Consistency | Our released arm's reflectance and spectral errors are in the **same order of magnitude** as the best pretrained baseline (initial target: reflectance ≤ 0.015, spectral ≤ 4°) |
| G3 Honesty | Hallucination of the released arm ≤ that of `bicubic`-plus-sharpen controls and clearly below the GAN arm A4 |
| G4 Value | Improvement `im` > bicubic and ≥ one pretrained baseline, **or** we report plainly that it is not |
| G5 Registration | Median measured shift ≤ 0.5 px (2.5 m) on Track A |
| G6 Generalisation | Results reported per land-cover class and for both external sets |

If a gate fails, the report says so and explains why. A failed gate that is reported is better than a hidden one.

---

## 8. Downstream Task Evaluation (building and road extraction)

### 8.1 Pre-registered hypothesis (write in `HYPOTHESES.md` before running)
- **H1:** Segmentation trained and tested on SR outputs scores higher than on bicubic-upsampled Sentinel-2 for buildings and roads.
- **H0 (live risk):** No raster-level improvement, as seen in prior studies on SR for buildings, with possible object-level gains for larger structures.

### 8.2 Setup
| Item | Choice |
|------|--------|
| Segmentation model | UNet with an SE-ResNeXt50 encoder via `segmentation_models_pytorch` (a configuration shown to work for this exact task in the literature) |
| Inputs compared | (a) bicubic 2.5 m, (b) `sen2sr_lite`, (c) `ldsr_s2`, (d) our released arm, (e) **native HR (NAIP)** as the upper bound |
| Labels | Track A: OSM buildings and roads rasterised to 2.5 m. Track B: Open Buildings (≥ 90 % precision) and OSM roads |
| Loss | Focal-Tversky + BCE (handles class imbalance) |
| Split | Spatial block split, never random pixels |
| Training | **Identical** hyper-parameters, seeds and data for every input type |
| Metrics | Raster: IoU, F1, precision, recall. **Object-level:** fraction of ground-truth buildings found (≥ 50 % overlap) and overlap ratio, **grouped by building size**. Roads: IoU on buffered centrelines plus connectivity (number of broken segments) |
| Reporting | All inputs in one table with confidence intervals across seeds |

### 8.3 Interpretation rules
- If SR does not beat bicubic, the headline becomes: *"SR improves visual and edge fidelity, but we did not find a downstream gain, and here is the evidence."* This is credible and is exactly what an evaluator probes.
- Label noise caveat (Track B): Open Buildings is itself a model output, so downstream scores on India measure agreement with that product, not with the truth. State this on the slide.

---

## 9. Demo Plan ("smallest thing that wins the room")

### 9.1 Demo script (3-4 minutes)
1. **Triptych on a Track A tile:** LR (10 m) | our SR (2.5 m) | true HR (NAIP/SPOT). Swipe slider.
2. **Roads and field edges:** overlay OSM roads and HR-derived edges on the SR; show recovered vs invented (hallucinated pixels in red, from opensr-test correctness map).
3. **Fidelity scoreboard:** one table: bicubic vs two baselines vs ours on consistency, `ha/om/im`, edge F1.
4. **The fidelity dial:** the A0-A4 plot showing the sharpness vs hallucination trade-off and where we chose to stop.
5. **Trust map:** uncertainty overlay on an **India (Punjab) tile**, with an explicit banner "No HR reference exists here; showing consistency and uncertainty only."
6. **Downstream slide:** building/road extraction table and the honest conclusion.

### 9.2 Implementation
- Single **Streamlit** app (one choice, avoids framework conflicts) reading **precomputed** GeoTIFFs and metric CSVs, so the demo never depends on a live GPU.
- Optional live mode: run the Lite model on CPU for a small user-clicked window.
- Every displayed SR image carries the banner **"AI-enhanced, not ground truth"** and the provenance tags (hard rule 4).
- Pre-render all figures to PNG as a fallback in case the app fails on stage.

---

## 10. Repository Layout and Environments

```
sih26142-srm/
├── DECISIONS.md          # Section 0 register + change log
├── HYPOTHESES.md         # Section 8.1, written before experiments
├── LICENSES.md           # Section 5.2
├── RUNLOG.md             # timings, gotchas, seeds
├── configs/              # one YAML per experiment (frozen, hashed)
├── src/
│   ├── common/           # resample.py, contract.py, io.py, seeds.py
│   ├── data/             # ingest_s2.py, sen2naip_loader.py, rasterize_labels.py
│   ├── models/           # ours_generator.py, losses.py, constraint.py
│   ├── baselines/        # run_bicubic.py, run_sen2sr.py, run_ldsr.py (write GeoTIFFs only)
│   ├── eval/             # evaluate.py, edge_fidelity.py, alignment.py, calibration.py
│   ├── downstream/       # train_seg.py, eval_seg.py
│   └── demo/             # app.py (Streamlit)
├── tests/                # test_contract.py, test_constraint.py, test_eval_sanity.py
├── outputs/
│   ├── sr/{model}/       # GeoTIFFs (D7)
│   ├── metrics/          # CSVs
│   └── figures/
├── env/
│   ├── env-main.lock     # training, evaluation, downstream, demo
│   └── env-baselines.lock# sen2sr / opensr-model / mlstac / cubo, kept apart if they conflict
└── README.md
```

**Environment rules (D11):**
1. Create `env-main` first and install: PyTorch, rasterio, geopandas, segmentation-models-pytorch, lpips, opensr-test, streamlit. Run `tests/` and freeze the versions to a lock file immediately.
2. Install baselines in `env-baselines`. If they install cleanly into `env-main`, merge them. If not, keep them separate. The file contract (D7) makes this safe.
3. Do **not** install the Mamba-based SEN2SR variant (extra CUDA build requirements) unless everything else is finished.
4. Never upgrade packages after the lock without re-running the full test-suite.

---

## 11. Phased Schedule with Exit Criteria

Durations are relative working days for a small team and should be scaled to the real deadline. **The cut line is the end of Phase 3.** Everything before it is the minimum viable winning submission.

| Phase | Days | Work | Exit criteria |
|-------|------|------|---------------|
| **0. Freeze** | 1 | Create repo, accounts (CDSE, Hugging Face), resolve all **[VERIFY]** items, write `DECISIONS.md`, `HYPOTHESES.md`, `LICENSES.md` | Decisions frozen; licence of every weight/data confirmed or flagged |
| **1. Data** | 3-4 | Ingest scripts, SEN2NAIPv2 loader, label rasterisation, contract tests, Punjab AOI tiles | `tests/test_contract.py` green on all sources; leakage report done |
| **2. Baselines + harness** | 4 | Run bicubic, SEN2SR-Lite, LDSR-S2 → GeoTIFFs; build `evaluate.py`; sanity-check against opensr-test reference numbers | Baseline table reproduced within reason; gates recalibrated and frozen; epoch timing logged |
| **3. Own model (A1-A2)** | 5-6 | Implement constrained generator and losses; train A1, A2 (3 seeds); uncertainty by seed ensemble | A1/A2 evaluated on val; one candidate released; **demo-able triptych exists** → *MVP reached* |
| **4. Ablations + test** | 3 | Train A3, A4; run the fidelity-dial plot; **single** test-set evaluation per frozen config | Full benchmark table + dial figure + per-land-cover breakdown |
| **5. Downstream** | 4 | Train segmentation for 5 input types on Track A; Track B check | Downstream table + written interpretation (H1 vs H0) |
| **6. Demo + report** | 3 | Streamlit app, pre-rendered fallbacks, slides, README, rehearsal with Q&A | Dry-run of the 3-4 min demo without errors |
| **Buffer** | 2-3 | Re-runs, fixes | n/a |

**If time is short:** drop A3/A4 (keep only A1/A2 plus baselines), run downstream on buildings only, and use one India tile. Never drop the evaluation harness or the honesty framing.

---

## 12. Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Owner trigger |
|---|------|------------|--------|-----------|---------------|
| 1 | Our model hallucinates | Medium | High | Hard constraint, conservative arms, hallucination metrics, say so openly | G3 fails |
| 2 | Paired-data mismatch teaches wrong mapping | Medium | High | Use curated SEN2NAIP, filter low-correlation pairs, validate on external real pairs | Large registration shift in val |
| 3 | Downstream shows no gain | **High** (literature precedent) | Medium | Pre-registered H0, honest framing, object-level and size-grouped metrics | H1 rejected |
| 4 | Dependency conflicts | Medium | Medium | D7 file contract, two environments, lock files | Install fails |
| 5 | Radiometry offset/scale error on fresh tiles | Medium | High | Read metadata, unit-test vegetation vs water reflectance ranges | Tile looks washed out |
| 6 | Band-order error | Low | High | Assertion test (D2) | Colours wrong |
| 7 | Train/eval operator mismatch | Medium | Medium | D5: report both operators | Consistency gap between operators |
| 8 | Test leakage | Low | High | Footprint-overlap check, test touched once | Overlap found |
| 9 | India behaviour worse than US | **High** | Medium | Track B labelling, uncertainty overlay, state as limitation | Visual artefacts on Punjab |
| 10 | Licence breach | Low | High | `LICENSES.md`; WorldStrat kept non-commercial | Redistribution attempted |
| 11 | Diffusion baseline too slow | Medium | Low | Run on subset/lower K, record time | Eval >1 day |
| 12 | Demo failure on stage | Medium | High | Precomputed outputs, PNG fallbacks | Rehearsal failure |
| 13 | Misuse concern (fabricated structure in an intelligence setting) | n/a | High | Provenance tags, uncertainty maps, explicit "not ground truth" banner, documented failure modes | n/a |

---

## 13. Judge Q&A Preparation

| Likely question | Prepared answer (backed by the plan) |
|-----------------|--------------------------------------|
| "How do you know you are not inventing roads?" | Hallucination/omission/improvement metrics against real HR on Track A; edge-fidelity F1; red hallucination overlay in the demo; uncertainty map. |
| "Why not just use a GAN?" | We tested one (A4) and show the sharpness vs hallucination trade-off; our released model is chosen on fidelity. |
| "Do your numbers hold outside your training data?" | External opensr-test sets never used in training; leakage check; per-land-cover breakdown. |
| "Does it work on Indian imagery?" | Qualitatively and with reference-free checks (Track B); no HR reference exists for those tiles, so we make no hallucination-rate claim there. |
| "Does it help real tasks?" | Downstream study with pre-registered hypothesis; we report the outcome, whatever it is. |
| "Is geographic registration preserved?" | Affine-alignment unit test on every output plus measured sub-pixel shift. |
| "Can I trust a pixel?" | Pixel-level uncertainty, checked for correlation with true error. |
| "Why are you not using the biggest model?" | Published evidence that models above ~15 M parameters did not give significant gains; we prioritise consistency over size. |

---

## 14. Claims Policy (what we may and may not say)

**We may say:**
- "Recovered X % of HR edge structure with Y hallucination on external real pairs (Track A)."
- "Spectral/reflectance drift is within the same order as the best published open models."
- "On India imagery, outputs satisfy the consistency checks, and uncertainty is shown; no HR reference exists to measure hallucination."

**We may not say:**
- "True 2.5 m imagery" or "ground truth".
- Any hallucination or accuracy number on India tiles that needs an HR reference.
- That SR improves downstream tasks unless Section 8 data supports it.
- That LPIPS/SSIM prove fidelity.

---

## 15. Pre-Flight Conflict Audit (tick all before starting Phase 3)

- [ ] D1-D13 recorded in `DECISIONS.md`, and every teammate has read Section 0.
- [ ] Band-order test passes for every data loader (D2).
- [ ] Reflectance scale/offset handled from metadata on SEN2NAIP and on a fresh CDSE/AWS tile (D3).
- [ ] Patch crop test: LR 130→128 (1 px/side) matches HR 520→512 (4 px/side).
- [ ] Output affine test passes on baseline outputs (Section 4).
- [ ] Same test tiles used for every model; list saved to `outputs/metrics/test_tiles.txt`.
- [ ] Leakage report between SEN2NAIPv2-train and opensr-test done.
- [ ] Two lock files exist; `pytest` green in each.
- [ ] Gates G1-G6 recalibrated once and then frozen.
- [ ] Every **[VERIFY]** item resolved or explicitly recorded as "unresolved".
- [ ] Licence table complete.

---

## 16. Sources (retrieved during research)

1. SEN2NAIP dataset card: https://huggingface.co/datasets/isp-uv-es/SEN2NAIP and v2 collection https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2
2. Aybar et al., *SEN2NAIP: A large-scale dataset for Sentinel-2 Image Super-Resolution*, Scientific Data (2024), https://doi.org/10.1038/s41597-024-04214-y
3. opensr-test benchmark and metric definitions: https://github.com/ESAOpenSR/opensr-test (paper: https://ieeexplore.ieee.org/document/10530998)
4. opensr-model (LDSR-S2) with uncertainty maps: https://github.com/ESAOpenSR/opensr-model ; opensr-utils: https://github.com/ESAopenSR/opensr-utils
5. SEN2SR: https://github.com/ESAOpenSR/sen2sr (PyPI: https://pypi.org/project/sen2sr/)
6. DiffFuSR: https://arxiv.org/html/2506.11764v2
7. WorldStrat: https://arxiv.org/abs/2207.06418 (data: https://zenodo.org/record/6810792)
8. Hollendonner (TU Wien, 2025), *Evaluating Sentinel-2 Super-Resolution Algorithms for Automated Building Delineation*: https://repositum.tuwien.at/ (record 20.500.12708/221077)
9. Galar et al., *Super-resolution for Sentinel-2 images*, ISPRS Archives (2019): https://isprs-archives.copernicus.org/articles/XLII-2-W16/95/2019/ ; and the 2020 ISPRS Annals follow-up: https://isprs-annals.copernicus.org/articles/V-1-2020/9/2020/
10. HalluGen (hallucination metric benchmark for image restoration): https://arxiv.org/pdf/2512.03345
11. On hallucinations in inverse problems (includes Sentinel-2 ×4 experiment): https://arxiv.org/pdf/2605.13146
12. Google Open Buildings: https://sites.research.google/open-buildings/ (coverage/licence mirror: https://source.coop/cholmes/google-open-buildings)
13. Copernicus Data Space STAC: https://stac.dataspace.copernicus.eu/v1 ; guide: https://dataspace.copernicus.eu/node/2298
14. AWS Sentinel-2 L2A COGs (no sign-in): https://registry.opendata.aws/sentinel-2-l2a-cogs/
15. ESA Living Planet Symposium 2025 DiffFuSR slides: https://lps25.esa.int/lps25-presentations/presentations/518/_518.pdf

### Items to verify first (Phase 0)
1. Exact temporal window, counts and licence of the SEN2NAIPv2 version actually downloaded (sources disagree between versions).
2. Licence terms of opensr-model and SEN2SR weights.
3. Channel order produced by each loader (D2) and the reflectance offset on current L2A products (D3).
4. Geographic overlap between SEN2NAIPv2 train tiles and opensr-test tiles.
5. CDSE free-tier quotas (found only in a third-party summary, not in official documentation).
6. Official SIH26142 statement wording on the SIH portal (the brief we analysed is a third-party summary).
