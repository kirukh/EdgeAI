#!/usr/bin/env python3
"""Entry point: ``python main.py web --camera pi`` (see qc/cli.py for all commands)."""

import sys

from qc.cli import main

if __name__ == "__main__":
    sys.exit(main())
