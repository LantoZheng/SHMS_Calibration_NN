"""One-command entry point for global FP5D discovery followed by local fields."""

from __future__ import annotations

import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    # Stage 1 writes the only input consumed by stage 2.  Keeping the stages in
    # separate scripts also makes it possible to rerun only the conservative
    # local refinement after changing its thresholds.
    runpy.run_path(str(HERE / "run_raw_fullroot_continuous_flow.py"), run_name="__main__")
    runpy.run_path(str(HERE / "run_linear_local_field_pipeline.py"), run_name="__main__")


if __name__ == "__main__":
    main()
