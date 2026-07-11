"""CLI entrypoint for LAMMPS trajectory projected primitive extraction."""

from __future__ import annotations

from conductivity.physical_library.extract_projected_primitives import main


if __name__ == "__main__":
    raise SystemExit(main())
