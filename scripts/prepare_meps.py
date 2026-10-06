#!/usr/bin/env python3
"""Download, preprocess, and verify MEPS 19--21."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "get_meps_data"


def main() -> None:
    subprocess.run(["Rscript", "download_data.R"], cwd=TOOLS, check=True)
    subprocess.run(
        [sys.executable, str(TOOLS / "main_clean_and_save_to_csv.py")],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "download_data.py"), "--verify-only"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
