#!/usr/bin/env python
"""GESTA 전략 §14: train split 의 ROI-pair bundle 크기 분포 -> outputs/stats/bundle_statistics.json"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atm_sc.data.bundle_statistics import main   # noqa: E402

if __name__ == "__main__":
    main()
