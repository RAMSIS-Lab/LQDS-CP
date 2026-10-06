# Reproducibility code

This repository reproduces the real-data benchmark reported in the manuscript. It contains clean implementations of LQDS and every reported baseline, but it contains no datasets, checkpoints, or saved result files. Those items are downloaded or generated locally.

The benchmark follows the data layout of the public [Conformalized Quantile Regression (CQR) repository](https://github.com/yromano/cqr). It uses ten datasets, thirteen procedures, thirty splits (seeds 2000--2029), and the training protocol stated in the paper. Concrete and Temperature are not used.

## Installation

The required Python environment, matching the environment used for the checkpoint-equivalence test, is:

- Python 3.9.12
- NumPy 1.22.4
- pandas 1.4.2
- SciPy 1.7.3
- scikit-learn 1.0.2
- PyTorch 2.8.0
- Matplotlib 3.5.1

Install it with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, create the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate lqds-reproducibility
```

The benchmark and automatic non-MEPS data downloader require no additional software. Preparing MEPS additionally requires R with the `foreign` package; the supplied R script installs that package if it is missing. This step was tested with R 4.5.2. Generating the final appendix PDFs requires a LaTeX installation with Computer Modern fonts and was tested with TeX Live 2019. These two system dependencies are only needed for their respective optional steps.

All paper runs use CPU execution and one thread per job.

## Data

Run the downloader from this directory:

```bash
python scripts/download_data.py
```

It downloads the non-MEPS data, places each file under `datasets/`, and verifies the exact paper input using `data_manifest.json`. The script obtains BlogFeedback and Facebook directly from UCI. The prepared CASP, STAR, Communities, and Bike benchmark files come from the CQR repository so that preprocessing is identical to the published benchmark.

The original data sources are:

- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), DOI 10.24432/C5W894. The exact prepared file is [`bike_train.csv` in CQR](https://github.com/yromano/cqr/blob/master/datasets/bike_train.csv).
- [UCI BlogFeedback](https://archive.ics.uci.edu/dataset/304/blogfeedback), DOI 10.24432/C58S3F.
- [UCI Facebook Comment Volume](https://archive.ics.uci.edu/dataset/363/facebook+comment+volume), DOI 10.24432/C5Q886. The benchmark uses training variants 1 and 2.
- [UCI Physicochemical Properties of Protein Tertiary Structure (CASP)](https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure), DOI 10.24432/C5QW3H.
- [UCI Communities and Crime](https://archive.ics.uci.edu/dataset/183/communities+and+crime), DOI 10.24432/C53W3X.
- [Tennessee STAR](https://doi.org/10.7910/DVN/SIWH9F), Harvard Dataverse.
- [MEPS HC-181](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181), panels 19 and 20.
- [MEPS HC-192](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192), panel 21.

For manual installation, download the files linked above and use this layout:

```text
datasets/
  CASP.csv
  STAR.csv
  bike_train.csv
  blogData_train.csv
  communities.data
  communities_attributes.csv
  facebook/
    Features_Variant_1.csv
    Features_Variant_2.csv
  meps_19_reg.csv
  meps_20_reg.csv
  meps_21_reg.csv
```

The exact non-MEPS prepared filenames are also available from the [CQR datasets directory](https://github.com/yromano/cqr/tree/master/datasets). After placing files manually, run:

```bash
python scripts/download_data.py --verify-only
```

### MEPS

MEPS requires acknowledgement of the AHRQ usage notice, so it is prepared separately. Install R and its `foreign` package, read the AHRQ data-use notice displayed by the script, then run:

```bash
python scripts/prepare_meps.py
```

This downloads HC-181 and HC-192 from AHRQ, creates `datasets/meps_19_reg.csv`, `datasets/meps_20_reg.csv`, and `datasets/meps_21_reg.csv`, and verifies all eleven benchmark files. The preparation code is the version distributed by CQR, with its license retained.

## One split

Training from scratch creates all model-family checkpoints needed by the selected methods:

```bash
python run.py --mode train --dataset star --seed 2000
```

The default is the paper protocol: 600 epochs with early stopping. A local checkpoint is written to `artifacts/checkpoints/star/2000/`, and split metrics are written to `artifacts/results/train/star_2000.json`.

Reuse the locally generated checkpoints without retraining. These evaluation-only records are kept separately under `artifacts/results/checkpoint/`, so they do not overwrite the training-time records:

```bash
python run.py --mode checkpoint --dataset star --seed 2000
```

Select a subset with `--methods`, for example:

```bash
python run.py --mode train --dataset star --seed 2000 --methods lqds_cp lqds_hpd cqr
```

Available procedures are `cti`, `spice_nd`, `lqds_cp`, `cir_nu`, `hpd_split`, `spice_hpd`, `lqds_hpd`, `cir_fast`, `cir_plus_fast`, `cqr`, `dcp`, `dcp_cqr`, and `dist_split`.

## Full benchmark

Generate every model-family checkpoint and every reported real-data result:

```bash
python scripts/run_all.py --mode train
```

The command covers all ten datasets and seeds 2000--2029. Interrupted runs can resume with `--skip-existing`. Once checkpoints exist, all results can be recomputed with:

```bash
python scripts/run_all.py --mode checkpoint
```

Aggregate the split files:

```bash
python scripts/summarize_results.py
```

This creates `artifacts/summary/benchmark_summary.csv` and `artifacts/summary/benchmark_splits.json`. All files below `artifacts/` are generated outputs and are excluded from the supplementary archive.

Generate the three appendix boxplots after aggregation:

```bash
python scripts/make_width_boxplots.py
```

The vector PDFs use the AISTATS text width (6.75 inches) and Computer Modern through LaTeX. A working LaTeX installation is therefore required for this final plotting step.


