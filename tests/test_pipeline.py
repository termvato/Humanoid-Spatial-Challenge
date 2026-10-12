from pathlib import Path

import pytest

from roomdiff.pipeline import run_lidar

PAIR01 = Path("data/raw/pair01")


@pytest.mark.skipif(not PAIR01.exists(), reason="own capture not available")
def test_lidar_pipeline_finds_the_suitcase_move():
    """S7 regression on real data: registration holds and the suitcase shows as a removed and an added blob."""
    pair = run_lidar(PAIR01, stride=10)
    assert pair.icp[0] > 0.8
    big = [b for b in pair.cm.blobs if len(b.idx) > 60]  # suitcase ~17-23 L = 136-184 voxels of 5 cm
    assert {b.kind for b in big} == {"removed", "added"}
