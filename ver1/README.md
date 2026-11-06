# Appendix B — code, data, logs and results

**Article title:** Detection Completeness and Throughput Screening of Three Deep Learning Detectors for Site-Specific Weed Control in Sugarcane

**Journal:** Artificial Intelligence in Agriculture

**Authors:** Nguyen Huynh Truong Gia, Nguyen Duc Khuyen, Vo Van Qui, Tran Quoc Khanh, Tran Thi Kim Nga

**Affiliation:** Faculty of Mechanical Engineering – Technology, Nong Lam University, Ho Chi Minh City, Vietnam

**Corresponding author: ttknga@hcmuaf.edu.vn**

Packaged 2026-11-04.

This archive contains everything needed to reproduce every number, table and
figure in the article without access to a GPU and without downloading the
images. The image files themselves are supplied by the corresponding author on
request; `MANIFEST.sha256` lists the SHA-256 checksum of every one of the
4451 of them, so a recipient can verify that the image
set received is the set used here.

## Contents

```
code/
  src/                 the full pipeline, 47 files: training,
                       evaluation, bootstrap, significance testing, figure and
                       table generation, and the three verification layers
  configs/paper.yaml   every hyper-parameter and evaluation setting; no value
                       used in the article is hard-coded anywhere else
  run_all.sh           end-to-end driver
  setup_env.sh         environment construction
data/
  annotations/         COCO-format annotations for both Roboflow exports,
                       train/valid/test, as used (6 files)
  dataset_stats.json   image and instance counts behind Table 1
  (images are not in this archive — see MANIFEST.sha256)
results/               raw measurements, one JSON per model plus the pooled
                       analyses (53 files). Every number printed
                       in the article is read from this directory
tables/                Tables 1-5 as CSV and Markdown
figures/               Figures 1-4 as PDF and PNG
logs/                  training and evaluation logs (90 files),
                       including per-seed console logs and the per-run
                       directories from which the training curves are drawn
manuscript/            the manuscript and the Supplementary Information as
                       Markdown, so that the automatic checks below can verify
                       the text against results/

MANIFEST.sha256        SHA-256 of every file in this archive and of every image
```

## Dataset

Images were captured at Tan Hung farm, Chau Thanh district, Tay Ninh province,
Vietnam. Two Roboflow exports of the same capture campaign were annotated at
two levels of granularity:

| Export | Annotation | Train | Valid | Test | Source images | Instances |
|---|---|---|---|---|---|---|
| bbox | bounding box | 2892 | 123 | 124 | 1211 | 34350 |
| polygon | polygon | 1212 | 50 | 50 | 504 | 33380 |

Union of the two exports: 1363 distinct source images, of which
352 carry annotations in both formats.

**Important.** The two exports are annotated at different granularity
(29.34 against 10.64 instances per image in the test
partition), so the bounding-box and polygon models cannot be ranked against
each other. The article reports them on separate test partitions for this
reason, and the point is set out under "Basis of comparison" in the Methods.

## Reproducing the article

Without a GPU and without the images, the computational chain can be checked
in full:

```bash
python3 code/src/selftest.py          # recomputes metrics from stored predictions
python3 code/src/audit.py             # 11 consistency checks over results/
python3 code/src/check_consistency.py --journal aiia \
       --docx manuscript/AIIA_manuscript.docx
```

With a GPU and the image set in place:

```bash
bash code/setup_env.sh                # torch cu118 + Detectron2 + Ultralytics
bash code/run_all.sh full             # all models, all seeds
```

No number in `tables/` or `figures/` is typed by hand; all are generated from
`results/*.json`, and the third check above verifies that every number printed
in the manuscript still matches the one in `results/`.

## Verification

```bash
sha256sum -c MANIFEST.sha256 --ignore-missing
```

## Note on language

The article, this README and all generated tables and figures are in English.
Comments inside the Python source are in Vietnamese, the working language of
the authors. Function and variable names, log output and all file formats are
in English, so the code can be followed without Vietnamese.

## Environment used

NVIDIA GeForce RTX 3080 Ti (12 GB) - CUDA 11.8 - PyTorch 2.1.2 -
Detectron2 0.6 - Ultralytics 8.3.253 - Python 3.10
