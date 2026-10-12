"""The simulated test room: Isaac's Simple_Room plus props, in a "before" and an "after" state.

Shared by record_visits.py and drive_demo.py. Positions are world metres (z up, floor at
FLOOR_Z). Each prop has a state per visit: an (x, y) floor position, or None if absent.
Import only after SimulationApp exists.
"""

from __future__ import annotations

import numpy as np
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd, UsdGeom

FLOOR_Z = -0.77
ROOM = "/Isaac/Environments/Simple_Room/simple_room.usd"

# name: (asset, yaw deg, {1: before (x, y) or None, 2: after}, ground-truth change)
PROPS = {
    "cabinet": ("/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd", 0,
                {1: (-3.2, 2.9), 2: (2.9, -2.3)}, "moved"),
    "dolly": ("/Isaac/Props/Dolly/dolly.usd", 90, {1: (3.2, 2.8), 2: None}, "removed"),
    "bins": ("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", 0, {1: None, 2: (-2.6, -2.0)}, "added"),
    "bins_top": ("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", 0, {1: None, 2: (-2.6, -2.0)}, "added"),
    "corner_bin": ("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", 30, {1: (-3.85, -2.75), 2: (-3.85, -2.75)},
                   "unobserved"),
    "mug_a": ("/Isaac/Props/Mugs/SM_Mug_A2.usd", 0, {1: (-0.4, 0.2), 2: (-0.4, 0.2)}, "unchanged"),
    "mug_b": ("/Isaac/Props/Mugs/SM_Mug_C1.usd", 0, {1: (0.5, -0.3), 2: (0.5, -0.3)}, "unchanged"),
}
ON_TABLE = {"mug_a", "mug_b"}  # sit on the table top (z = 0.01)
STACKED = {"bins_top": "bins"}  # sits on top of another prop
UNOBSERVED_CORNER = (-3.85, -2.75)  # visit 2 never looks here


def _bbox(prim) -> tuple[np.ndarray, np.ndarray]:
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def load(app, settle: int = 30) -> None:
    open_stage(get_assets_root_path() + ROOM)
    stage = get_current_stage()
    root = get_assets_root_path()
    for name, (asset, yaw, _, _) in PROPS.items():
        # a plain parent we move; the asset keeps its own transform ops underneath
        prim = UsdGeom.Xform.Define(stage, f"/World/Props/{name}").GetPrim()
        stage.DefinePrim(f"/World/Props/{name}/asset").GetReferences().AddReference(root + asset)
        UsdGeom.XformCommonAPI(prim).SetRotate(Gf.Vec3f(0, 0, yaw))
    for _ in range(settle):  # let assets stream in before measuring them
        app.update()


def set_visit(app, visit: int) -> dict:
    """Place every prop for `visit` (1 or 2); returns ground truth {name: {pos, bbox_min, bbox_max}} or None."""
    stage = get_current_stage()
    gt: dict = {}
    for name in sorted(PROPS, key=lambda n: n in STACKED):  # bases before what sits on them
        _, _, where, _ = PROPS[name]
        prim = stage.GetPrimAtPath(f"/World/Props/{name}")
        img = UsdGeom.Imageable(prim)
        if where[visit] is None:
            img.MakeInvisible()
            gt[name] = None
            continue
        img.MakeVisible()
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(Gf.Vec3d(0, 0, 0))
        lo, hi = _bbox(prim)
        if name in STACKED:
            base = gt[STACKED[name]]
            z = base["bbox_max"][2]
        else:
            z = 0.01 if name in ON_TABLE else FLOOR_Z
        x, y = where[visit]
        api.SetTranslate(Gf.Vec3d(x - (lo[0] + hi[0]) / 2, y - (lo[1] + hi[1]) / 2, z - lo[2]))
        lo, hi = _bbox(prim)
        gt[name] = {"pos": [x, y], "bbox_min": lo.round(3).tolist(), "bbox_max": hi.round(3).tolist()}
    app.update()
    return gt
