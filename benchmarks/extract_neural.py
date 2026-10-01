#!/usr/bin/env python3
"""Pre-compute (and cache) language-model features for every (capped) corpus segment."""

import logging
import sys

from common import load_manifest, neural_rows

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = neural_rows(load_manifest())
    print(f"neural features available for {len(rows)} segments")
    sys.exit(0)
