from __future__ import annotations


def validate_column_name(value: str, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def validate_distinct_columns(*columns: str) -> None:
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"Duplicate column names are not allowed: {duplicates}")
