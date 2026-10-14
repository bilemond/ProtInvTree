#!/usr/bin/env python3
"""Run CATH evaluation from a checkout or an installed package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protinvtree.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
