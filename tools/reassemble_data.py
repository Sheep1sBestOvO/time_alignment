"""Reassemble split HDF5 data files after cloning the repository.

GitHub blocks normal Git files larger than 100 MB. The HDF5 files in this repo
are therefore stored as smaller ``.part`` files under ``data_parts/``. Run this
script once on the HPC after cloning to recreate ``data/*.hdf5``.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PARTS = ROOT / "data_parts"
DATA = ROOT / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    stems = sorted({path.name.split(".part")[0] for path in DATA_PARTS.glob("*.part*")})
    if not stems:
        raise SystemExit(f"No .part files found in {DATA_PARTS}")

    for stem in stems:
        parts = sorted(DATA_PARTS.glob(f"{stem}.part*"))
        output = DATA / stem
        with output.open("wb") as out_file:
            for part in parts:
                out_file.write(part.read_bytes())
        print(f"Reassembled {output} from {len(parts)} parts")


if __name__ == "__main__":
    main()
