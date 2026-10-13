"""The scan pipeline end to end: CAD -> scanify -> scanprep -> scanransac labels -> edgebuild EB_SCAN -> STEP.

Parts where CAD-truth labels already build a valid solid (2026-10-13, tools/scanpipe_eval.py): detect()'s labels
must too. The whole-set number is scanpipe_eval on all 30 parts; this guards the pipeline, not a stage.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools")]
import scanpipe_eval as E  # noqa: E402


@pytest.mark.parametrize("part", ["122740_fc5c94eb_34", "27663_023746a1_1"])
def test_ransac_labels_build_a_valid_solid(part):
    row = E.run_part(part, variants=("ransac",))
    r = row["ransac"]
    assert r["valid"] and r["solids"] == 1, r
    assert r["recovered"] >= 1, r
