from __future__ import annotations

import numpy as np


def rootcohesion(classes, nd, b, f, s, h, w):
    out = []
    for val in classes:
        if val == 250:
            out.append(nd)
        elif val in [12, 21, 22, 23, 24, 31]:
            out.append(b)  # bare/developed
        elif val in [41, 42, 43]:
            out.append(f)  # forest
        elif val in [51, 52]:
            out.append(s)  # shrub
        elif val in [71, 72, 73, 74, 81, 82]:
            out.append(h)  # herbaceous
        elif val in [90, 95]:
            out.append(w)  # wetland
        else:
            out.append(nd)
    return out


def vegtype(classes, nd, bare, tree, shrub, grass):
    out = []
    for val in classes:
        if val == 250:
            out.append(nd)
        elif val in [12, 21, 22, 23, 24, 31]:  # Bare
            out.append(bare)
        elif val in [41, 42, 43, 90]:  # Tree (includes woody wetlands)
            out.append(tree)
        elif val in [51, 52]:  # Shrub/Scrub
            out.append(shrub)
        elif val in [71, 72, 73, 74, 81, 82, 95]:  # Grass + herbaceous wetlands
            out.append(grass)
        else:
            out.append(nd)
    return out


def adjust_internal_friction_angle(landcover, internal_friction_angle, delta: float = 3.0):
    landcover = np.asarray(landcover)
    internal_friction_angle = np.asarray(internal_friction_angle)
    mask = np.isin(landcover, [21, 22, 23, 24])
    internal_friction_angle[mask] = internal_friction_angle[mask] + delta
    return internal_friction_angle


def compute_cohesion(landcover):
    c_min = rootcohesion(landcover, -9999.0, 30, 4000, 2000, 1000, 3000)
    c_mode = rootcohesion(landcover, -9999.0, 100, 10000, 4000, 2000, 6000)
    c_max = rootcohesion(landcover, -9999.0, 150, 20000, 10000, 5000, 14000)
    return c_min, c_mode, c_max
