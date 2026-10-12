import numpy as np

from roomdiff.grid2d import FREE_CELL, OCC_CELL, UNKNOWN_CELL, Grid2D
from roomdiff.plan import astar, drivable, to_cell, to_xy


def _room():
    a = np.full((30, 30), FREE_CELL, np.int8)
    a[[0, -1], :] = OCC_CELL
    a[:, [0, -1]] = OCC_CELL
    a[15, :24] = OCC_CELL  # wall with a gap on the right
    return Grid2D(a, 0.1, (0.0, 0.0))


def test_astar_goes_round_the_wall_and_keeps_clear():
    g = _room()
    ok = drivable(g, 0.2)
    path = astar(ok, to_cell(g, (0.5, 0.5)), to_cell(g, (0.5, 2.5)))
    assert path is not None
    assert all(ok[p] for p in path)
    assert max(c for _, c in path) >= 24  # through the gap
    assert np.abs(np.array(to_xy(g, path[-1])) - [0.5, 2.5]).max() <= 0.1


def test_astar_does_not_enter_unknown():
    g = _room()
    g.data[15, 24:29] = UNKNOWN_CELL  # the gap was never seen
    assert astar(drivable(g, 0.2), to_cell(g, (0.5, 0.5)), to_cell(g, (0.5, 2.5))) is None
