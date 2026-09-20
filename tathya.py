#!/usr/bin/env python3
"""Tathya — uncover the truth of your dataset before you train.

Usage example:
    python tathya.py C:/path/to/dataset --output ./report

Full options:
    python tathya.py --help
"""

from tathya.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
