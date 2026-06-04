#!/usr/bin/env python
"""Compatibility wrapper for the preliminary-stage evaluator.

Usage:
    python self_score.py path/to/team_dir
"""

from __future__ import annotations

from evaluate_submission import main


if __name__ == "__main__":
    raise SystemExit(main())
