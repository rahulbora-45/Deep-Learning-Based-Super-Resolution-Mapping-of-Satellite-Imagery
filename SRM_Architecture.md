# Satellite Super-Resolution Mapping (SRM) — Architecture

---

## 1. The idea in plain words

Free satellite photos (Sentinel-2) are blurry: one pixel covers **10 m × 10 m** on the ground. Small roads, houses and field edges get lost.

Our project takes that blurry image and makes a sharper one, where **one pixel covers 2.5 m × 2.5 m** (4× sharper).

The hard part is that an AI can *invent* things that look real but are false, such as a road that doesn't exist. So the project has two jobs:

1. **Sharpen** the image.
2. **Prove** the sharper image is honest.

The proof is the most important part.

---

## 2. The big picture

```
 ┌───────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌──────────────┐
 │ 1. GET DATA   │ → │ 2. PREPARE     │ → │ 3. SUPER-      │ → │ 4. CHECK       │ → │ 5. SHOW      │
 │ Satellite     │   │ Clean, align,  │   │ RESOLVE        │   │ TRUST          │   │ Demo app +   │
 │ images        │   │ standard format│   │ (AI models)    │   │ Is it honest?  │   │ report       │
 └───────────────┘   └────────────────┘   └────────────────┘   └────────────────┘   └──────────────┘
```

There are five blocks, and each one hands its result to the next as a file. Models never talk to each other directly, which keeps the project simple and avoids software conflicts.

---

## 3. Block by block

### Block 1 — Get Data

| What | Where it comes from | Used for |
|------|--------------------|----------|
| Sentinel-2 images (10 m) | Copernicus Data Space or AWS (free) | The blurry input |
| SEN2NAIP pairs (blurry + sharp of the same place) | Hugging Face | Teaching the AI |
| opensr-test images | Hugging Face | Fair final testing (never used for teaching) |
| Road and building maps | OpenStreetMap, Google Open Buildings | Checking roads/buildings |
| Punjab / Ludhiana images | Copernicus | Showing it works on Indian land |

**Two kinds of testing, kept separate:**
- **Track A (US/Spain):** a real sharp photo exists, so we can truly measure mistakes.
- **Track B (India):** no sharp photo exists, so we only show consistency checks and an uncertainty map. We never claim an accuracy number here.

### Block 2 — Prepare

Everything is converted to **one standard format** so nothing mismatches:

| Rule | Value |
|------|-------|
| Colour channel order | Red, Green, Blue, Near-Infrared (always) |
| Pixel values | Reflectance from 0 to 1 |
| Input size | 128 × 128 pixels (10 m) |
| Output size | 512 × 512 pixels (2.5 m) |
| Alignment | Each blurry pixel exactly covers a 4×4 block of sharp pixels |
| File type | GeoTIFF (keeps location information) |

Tests run automatically to catch wrong colour order, wrong scale or shifted location.

### Block 3 — Super-Resolve (the AI models)

Four models are compared, all on the same test images:

| Model | What it is | Do we train it? |
|-------|-----------|----------------|
| **Bicubic** | Plain enlarging, no AI (the "do nothing smart" baseline) | No |
| **SEN2SR-Lite** | Ready-made small AI model | No |
| **LDSR-S2** | Ready-made diffusion model that also gives uncertainty | No |
| **Our model** | Small AI model with a built-in honesty rule | **Yes** |

**Our model, simply:**

```
 Blurry image ──► Enlarge normally ──────────────┐
      │                                          ├──► ADD ──► Sharp image
      └──► Small AI network ──► Fine detail ─────┘
                                (only the detail the blurry image
                                 could NOT already show)
```

- The AI only adds **fine detail**, not new colours or brightness. If you shrink the result back down, it should look like the original blurry image. We call this the **consistency rule**.
- It is a small network (about 5 million parameters), because research shows bigger is not much better here.

**What teaches the AI (loss terms):**

| Teacher | Purpose |
|---------|---------|
| Pixel difference | Stay close to the real sharp image |
| Colour/spectral match | Don't change colours |
| Consistency check | Shrunk result must match input |
| Edge sharpness | Put edges in the right place |
| Small "looks real" term | A bit more natural texture (kept small because it encourages invention) |
| Adversarial (GAN) term | Off by default, used only in one test version |

**Five versions are trained, called the "fidelity dial":**

| Version | What is added | Expected |
|---------|---------------|----------|
| A0 | Nothing (bicubic) | Blurry but honest |
| A1 | Basic rules | Careful, low invention |
| A2 | + edge sharpness | Sharper |
| A3 | + "looks real" term | Sharper, more invention risk |
| A4 | + GAN | Sharpest, highest invention risk |

We pick the best balance using validation data, not the prettiest picture.

### Block 4 — Check Trust (the most important block)

This block answers: **"Can I believe this image?"**

```
 Sharp image ──► Does shrinking it match the original?     (consistency)
             ──► Is every pixel in the right place?        (registration)
             ──► Is there detail that isn't real?          (hallucination)
             ──► Is real detail missing?                   (omission)
             ──► Did we gain real detail?                  (improvement)
             ──► How unsure is the AI at each pixel?       (uncertainty map)
```

**Three tiers of measuring, all from one evaluation script:**

| Tier | Measures | Note |
|------|----------|------|
| 1. Standard | PSNR, SSIM, LPIPS | Reported, but they reward sharpness even if fake, so they are never used alone |
| 2. Fidelity | Reflectance, spectral angle, alignment, hallucination, omission, improvement, edge match | Our main scoreboard (uses the opensr-test tool) |
| 3. Real use | Does building/road finding get better? | Honest answer, even if the answer is "no" |

**Uncertainty map:** a picture showing which pixels the AI is less sure about. Diffusion model: it comes from running several times. Our model: it comes from disagreement between 3 separately trained copies. We also check that "unsure" really means "more wrong".

**Tags on every output file:** model name, version, source image and date, plus the flags `SYNTHETIC_ENHANCEMENT=true` and `NOT_GROUND_TRUTH=true`.

### Block 5 — Show

| Part | What it does |
|------|--------------|
| **Streamlit demo app** | Reads pre-made results (no live GPU needed) |
| Triptych view | Blurry \| Our sharp \| True sharp, with a swipe slider |
| Overlays | Roads and field edges drawn on top; fake detail in red |
| Scoreboard | All models in one table |
| Fidelity dial chart | Sharpness vs invention for A0–A4 |
| Uncertainty view | Shown on a Punjab tile with a clear banner: *"No reference image exists here"* |
| Downstream slide | Building/road results and honest conclusion |
| Backup | All figures saved as PNG in case the app fails on stage |

---

## 4. Downstream test: do real tasks improve?

We train a standard building/road finder (UNet with SE-ResNeXt50) on five kinds of input and compare:

| Input | Role |
|-------|------|
| Bicubic | Baseline |
| SEN2SR-Lite | Competitor |
| LDSR-S2 | Competitor |
| Our model | Our result |
| Real sharp image (NAIP) | Upper limit |

Everything else is identical (same settings, same data, same seeds), so the comparison is fair. We write down our prediction **before** running it. Past studies found super-resolution sometimes did *not* beat plain enlarging, so we are ready to report that if it happens.

---

## 5. Folder structure

```
srm-project/
├── DECISIONS.md      the frozen rules (never change without a note)
├── HYPOTHESES.md     predictions written before experiments
├── LICENSES.md       data and tool licences
├── RUNLOG.md         timings, seeds, problems found
├── configs/          one settings file per experiment
├── src/
│   ├── common/       resize, standard format, seeds, file reading/writing
│   ├── data/         download, loaders, label maps
│   ├── models/       our network, losses, consistency rule
│   ├── baselines/    run bicubic, SEN2SR-Lite, LDSR-S2 → save files
│   ├── eval/         evaluation script, edge check, alignment, uncertainty check
│   ├── downstream/   building/road training and testing
│   └── demo/         Streamlit app
├── tests/            automatic safety checks
├── outputs/
│   ├── sr/           sharpened images per model
│   ├── metrics/      score tables (CSV)
│   └── figures/      charts and pictures
├── env/              locked software versions (two environments)
└── README.md
```

---

## 6. Software setup

| Environment | Contains |
|-------------|----------|
| `env-main` | PyTorch, rasterio, geopandas, segmentation-models-pytorch, lpips, opensr-test, Streamlit |
| `env-baselines` | sen2sr, opensr-model, mlstac (kept separate only if they clash) |

- Versions are frozen into lock files straight after the first working install.
- We skip the heavy Mamba version of SEN2SR (needs special CUDA setup).
- A single GPU is enough because our model is small.

---

## 7. Safety checks (automatic tests)

| Test | Catches |
|------|---------|
| Band order | Red and blue swapped |
| Value range | Wrong scale or washed-out colours |
| Shape and type | Wrong patch sizes, NaN values |
| Alignment | Output shifted compared to input |
| Cropping | SEN2NAIP crop mismatch (130→128 and 520→512) |
| Evaluation sanity | Our scoring giving odd numbers for known models |
| Same test list | Every model tested on identical images |
| Leakage check | Test images also used in training |

---

## 8. Pass/fail goals

| Goal | Meaning |
|------|---------|
| G1 | All output files pass the format tests |
| G2 | Colour and brightness errors are as small as the best open model |
| G3 | Our invention rate is clearly below the GAN version |
| G4 | We gain real detail over bicubic, or we say plainly that we did not |
| G5 | Typical location shift is half a pixel or less |
| G6 | Results reported separately for city, farmland and forest |

If a goal is missed, we report it. An honest miss is better than a hidden one.

---

## 9. What we may and may not say

**We may say**
- "We recovered X% of real edge structure with Y% invented detail on real test pairs (US/Spain)."
- "Colour and brightness stay within the range of the best open models."
- "On Indian tiles, consistency checks pass and uncertainty is shown. No reference exists to measure invention."

**We may not say**
- That the output is "real 2.5 m imagery" or "ground truth".
- Any accuracy number for India that needs a reference image.
- That super-resolution helps downstream tasks, unless our data shows it.
- That PSNR/SSIM/LPIPS alone prove the image is honest.

---

## 10. Build order

| Step | Work | Done when |
|------|------|-----------|
| 0 | Freeze rules, make accounts, check open questions | Rules written down |
| 1 | Data download, loaders, label maps, tests | All safety tests pass |
| 2 | Run the three baselines, build the evaluation script | Scores look like published numbers |
| 3 | Train our model (A1, A2), make the first triptych | **Minimum winning version exists** |
| 4 | Train A3 and A4, make the fidelity dial, do the final test | Full score table and charts |
| 5 | Building/road experiment | Honest conclusion written |
| 6 | Demo app, slides, practice run | Demo runs without errors |

If time is short, drop A3/A4, test buildings only, and use one India tile. Never drop the evaluation or the honesty message.

---

## 11. Main risks in one table

| Risk | What we do |
|------|-----------|
| AI invents false detail | Consistency rule, careful versions, invention metrics, say so openly |
| Training pairs don't match perfectly | Use curated data, filter bad pairs, test on outside data |
| No gain on real tasks | Prediction written in advance, honest reporting |
| Software conflicts | File-based hand-offs, two environments, lock files |
| Wrong colours or scale | Automatic tests |
| India worse than US | Clear Track B labels, uncertainty map, stated as a limitation |
| Licence issues | Licence table, WorldStrat kept for non-commercial use only |
| Demo fails on stage | Pre-made results and PNG backups |

---

## 12. One-minute summary for judges

> We take free 10 m Sentinel-2 images and sharpen them to 2.5 m using a small AI model that is built to only add fine detail, never change colours or locations. Then we prove it: we compare against real sharp images, count invented and missed detail, check locations and colours, show per-pixel uncertainty, and test whether real tasks like finding buildings and roads actually improve. Where we cannot prove something, as with Indian tiles that have no reference image, we say so.
