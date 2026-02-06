from __future__ import annotations

import os
from typing import Callable

from landlab.io import esri_ascii

try:
    from landlab.io import write_esri_ascii as _write_esri_ascii
except Exception:  # pragma: no cover - fallback for older landlab
    _write_esri_ascii = None


def read_nodata_value(ascii_path: str, default: float = -9999.0) -> float:
    nodata_val = None
    with open(ascii_path, "r") as f:
        for line in f:
            if line.strip().upper().startswith("NODATA_VALUE"):
                nodata_val = float(line.split()[1])
                break
    return default if nodata_val is None else nodata_val


def load_grid(ascii_path: str, field_name: str):
    with open(ascii_path, "r") as f:
        return esri_ascii.load(f, name=field_name)


def add_ascii_field(
    master_grid,
    ascii_path: str,
    field_name: str,
    scale: float = 1.0,
    offset: float = 0.0,
    close_nodata: bool = True,
    extra_close_values: list[float] | None = None,
    rename_file: bool = True,
    transform: Callable | None = None,
):
    nodata_val = read_nodata_value(ascii_path)

    with open(ascii_path, "r") as f:
        tmp = esri_ascii.load(f, name=field_name)

    raw_vals = tmp.at_node[field_name].copy()
    if transform is not None:
        raw_vals = transform(raw_vals)

    vals = raw_vals * scale + offset
    master_grid.add_field(field_name, vals, at="node", clobber=True)

    if close_nodata:
        master_grid.set_nodata_nodes_to_closed(raw_vals, nodata_val)

    if extra_close_values:
        for v in extra_close_values:
            master_grid.set_nodata_nodes_to_closed(raw_vals, v)

    if rename_file:
        new_path = os.path.join(os.path.dirname(ascii_path), f"{field_name}.asc")
        if ascii_path != new_path:
            os.rename(ascii_path, new_path)

    return master_grid


def write_ascii_field(path: str, grid, field_name: str, clobber: bool = True) -> None:
    try:
        esri_ascii.dump(path, grid, field_name, clobber=clobber)
        return
    except Exception:
        try:
            with open(path, "w") as f:
                esri_ascii.dump(f, grid, field_name, clobber=clobber)
            return
        except Exception:
            if _write_esri_ascii is None:
                raise

    _write_esri_ascii(path, grid, field_name, clobber=clobber)
