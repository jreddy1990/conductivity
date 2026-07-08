"""Strict physical-library loader for projected conductivity."""

from conductivity.physical_library.schema import (
    PhysicalLibraryRecords,
    load_physical_library,
    validate_physical_library_records,
)

__all__ = [
    "PhysicalLibraryRecords",
    "load_physical_library",
    "validate_physical_library_records",
]
