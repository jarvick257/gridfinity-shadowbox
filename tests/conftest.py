"""Shared test helpers."""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from gridfinity_cutter.bins.geometry import to_manifold


@pytest.fixture
def solid_at():
    """``solid_at(mesh, x, y, z)``: True when the point lies inside the closed mesh.

    Slices the solid with manifold3d and counts the rings around the point
    (odd = inside), which avoids trimesh's optional ``rtree`` dependency.
    """

    def check(mesh, x: float, y: float, z: float) -> bool:
        rings = to_manifold(mesh, "test mesh").slice(z).to_polygons()
        p = Point(x, y)
        return sum(Polygon(np.asarray(r)).contains(p) for r in rings) % 2 == 1

    return check
