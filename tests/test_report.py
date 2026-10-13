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


def _cm_fragments():
    g = Grid(origin=np.zeros(3), size=0.1, shape=(40, 10, 40))

    def blob(i, kind, x, z, n=4):
        idx = np.ravel_multi_index(np.stack([np.full(n, x), np.arange(n), np.full(n, z)]), g.shape)
        return Blob(i, kind, idx, np.zeros(3), np.zeros(3), float(n))

    # removed object (0) with a small fragment beside it (1); the fragment's "partner" (2) is far away
    blobs = [blob(0, "removed", 5, 5, 6), blob(1, "removed", 6, 5, 2), blob(2, "added", 30, 30, 3),
             blob(3, "removed", 20, 5), blob(4, "added", 20, 20), blob(5, "added", 21, 20, 2)]
    return ChangeMap(g, np.zeros(16000, np.uint8), np.zeros(16000), np.zeros(16000), blobs, 0.1)


def test_changes_merge_fragments_and_keep_kind():
    cm = _cm_fragments()
    views = {k: {k: None} for k in range(6)}
    out = changes(cm, views, {1: 2, 3: 4}, np.eye(4), merge=0.1)
    by_blobs = {tuple(sorted(c.blobs)): c for c in out}
    assert by_blobs[(0, 1)].kind == "removed"  # fragment link does not make the object "moved"
    assert by_blobs[(2,)].kind == "added"
    moved = by_blobs[(3, 4, 5)]  # moved, with the fragment next to its new place merged in
    assert moved.kind == "moved" and moved.was == [3]
    assert moved.after.size == (0.2, 0.1)  # voxels x = 20, 21


def test_changes_merge_off():
    cm = _cm_fragments()
    out = changes(cm, {k: {k: None} for k in range(6)}, {1: 2}, np.eye(4), merge=0)
    assert len(out) == 5 and sum(c.kind == "moved" for c in out) == 1
