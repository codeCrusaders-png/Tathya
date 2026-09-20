#!/usr/bin/env python3
"""Dataset Detective — investigate an image dataset and generate a report.

Usage example:
    python dataset_detective.py C:/path/to/dataset --output ./report

Full options:
    python dataset_detective.py --help
"""

from dataset_detective.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
