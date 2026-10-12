import numpy as np

from roomdiff.changemap import Blob, ChangeMap
from roomdiff.report import changes, to_json
from roomdiff.visibility import Grid


def _cm():
    g = Grid(origin=np.zeros(3), size=0.1, shape=(40, 10, 40))

    def blob(i, kind, x, z):
        idx = np.ravel_multi_index(np.stack([np.full(4, x), [0, 1, 2, 3], np.full(4, z)]), g.shape)
        return Blob(i, kind, idx, np.zeros(3), np.zeros(3), 4.0)

    blobs = [blob(0, "removed", 5, 5), blob(1, "added", 35, 5), blob(2, "added", 20, 30)]
    return ChangeMap(g, np.zeros(16000, np.uint8), np.zeros(16000), np.zeros(16000), blobs, 0.1)


def test_changes_merge_moved_pair_with_distance():
    cm = _cm()
    views = {0: {3: None}, 1: {7: None, 8: None}, 2: {9: None}}
    out = changes(cm, views, {0: 1}, np.eye(4), scale=2.0)
    kinds = [c.kind for c in out]
    assert sorted(kinds) == ["added", "moved"]
    moved = out[kinds.index("moved")]
    assert moved.before.centre == (1.0, -1.0) and moved.after.centre == (7.0, -1.0)  # x * 0.1 * 2, y_map = -z
    assert moved.distance == 6.0
    assert moved.before.bottom == -0.1 and moved.before.top == 0.7  # 4 voxels high, scaled
    assert moved.views == {0: [3], 1: [7, 8]}
    assert to_json(out)[0]["id"] == 0
