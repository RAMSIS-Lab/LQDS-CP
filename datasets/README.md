# Datasets

No dataset files are distributed with this repository. From the repository root, run:

```bash
python scripts/download_data.py
```

This downloads the public non-MEPS benchmark files, extracts the two Facebook variants used in the paper, and verifies every file against the checksums in `data_manifest.json`.

MEPS requires acknowledgement of the AHRQ data-use notice. Follow the MEPS instructions in the root `README.md`, then run `python scripts/download_data.py --verify-only`.
